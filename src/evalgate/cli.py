"""Command line for evalgate: run a dataset, list runs, compare two of them.

Owning artifact: `product context/00_PROJECT_BRIEF.md`, section 4.

Deliberate boundary: this module parses arguments, prints and chooses an exit code. Every
decision it reports was made in `compare`, and every number in `stats`. Keeping it that thin
means the verdict a person reads here is the same object a test can assert on.

The exit code is the product. `0` when nothing regressed, `1` when something did, and `2` when
the comparison could not answer -- which is not the same as passing, and must not share an
exit code with it.
"""

from __future__ import annotations

import importlib
from enum import IntEnum
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer

from evalgate.compare import SliceComparison, compare_slices
from evalgate.dataset import DatasetError, load_dataset
from evalgate.runner import Target, run_dataset
from evalgate.runs import RunStore, StoreError
from evalgate.scorers import ExactMatch, Scorer, ScorerError
from evalgate.stats import Verdict

__all__ = ["Exit", "app", "main"]


class Exit(IntEnum):
    """Process exit codes, which are what CI actually reads."""

    OK = 0
    """Nothing regressed beyond its tolerance."""

    REGRESSED = 1
    """At least one slice got worse by more than its tolerance."""

    INCONCLUSIVE = 2
    """Nothing was shown to regress, and nothing was shown not to.

    Separate from OK on purpose. A slice too thin to detect the regression it was asked about
    has not passed; it has failed to answer, and a build that cannot tell the difference will
    treat every underpowered slice as a green tick.
    """

    MISUSE = 3
    """The command could not run: a bad path, an unloadable target, a stale hash."""


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A CI check that fails a pull request when an LLM feature gets worse on a slice.",
)


def _load_target(reference: str) -> Target:
    """Import a target from a `module:function` reference.

    This imports and runs code the caller names, which is the same trust as a test runner
    collecting a test file: the reference comes from the person running the command or from
    their checked-in config, never from a dataset.

    Raises:
        typer.Exit: If the reference is malformed, or the module or attribute is missing.
    """
    if ":" not in reference:
        _fail(f"target must look like 'module:function', got {reference!r}")
    module_name, _, attribute = reference.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        _fail(f"cannot import {module_name!r}: {error}")
    if not hasattr(module, attribute):
        _fail(f"{module_name!r} has no attribute {attribute!r}")
    found: object = getattr(module, attribute)
    if not callable(found):
        _fail(f"{reference!r} is not callable")
    return cast("Target", found)


def _fail(message: str) -> NoReturn:
    """Print a message and exit with the misuse code.

    Typed as never returning, so the type checker knows the lines after a call to it are
    unreachable rather than reporting every value they touch as possibly unbound.

    Raises:
        typer.Exit: Always.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(Exit.MISUSE)


@app.command()
def run(
    dataset_path: Annotated[Path, typer.Option("--dataset", help="Path to the manifest.")],
    target: Annotated[str, typer.Option("--target", help="Target as 'module:function'.")],
    label: Annotated[str, typer.Option("--label", help="What to call this version.")],
    store_path: Annotated[
        Path, typer.Option("--store", help="SQLite file for run records.")
    ] = Path("runs.sqlite"),
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress per-item progress.")] = False,
) -> None:
    """Run a dataset through a target and record every item."""
    try:
        dataset = load_dataset(dataset_path)
    except DatasetError as error:
        _fail(str(error))

    function = _load_target(target)
    scorers: list[Scorer] = [ExactMatch()]

    typer.echo(f"{dataset.manifest.name} v{dataset.manifest.version}: {len(dataset)} items")
    try:
        with RunStore.open(store_path) as store:
            record = run_dataset(
                dataset,
                function,
                scorers,
                store,
                label=label,
                on_item=None if quiet else _progress(len(dataset)),
            )
            outcomes = store.outcomes(record.run_id)
    except (StoreError, ScorerError) as error:
        _fail(str(error))

    failures = sum(1 for outcome in outcomes if outcome.error)
    passed = sum(1 for outcome in outcomes if outcome.scores["exact_match"].value == 1.0)
    typer.echo(f"\nrun {record.run_id}")
    typer.echo(f"  exact_match  {passed}/{len(outcomes)}")
    if failures:
        typer.secho(f"  target errors  {failures}", fg=typer.colors.YELLOW)


def _progress(total: int):  # noqa: ANN202 - a closure over the item callback's exact signature
    """Build a per-item progress callback that overwrites one line."""

    def report(index: int, item: object) -> None:
        del item
        typer.echo(f"\r  {index + 1}/{total}", nl=False)

    return report


@app.command(name="runs")
def list_runs(
    store_path: Annotated[Path, typer.Option("--store")] = Path("runs.sqlite"),
    limit: Annotated[int, typer.Option("--limit")] = 20,
) -> None:
    """List recorded runs, most recent first."""
    try:
        with RunStore.open(store_path) as store:
            records = store.list_runs(limit=limit)
    except StoreError as error:
        _fail(str(error))

    if not records:
        typer.echo("no runs recorded")
        return
    for record in records:
        state = "finished" if record.finished_at else "UNFINISHED"
        typer.echo(
            f"{record.run_id[:12]}  {record.dataset_name} v{record.dataset_version}"
            f"  {record.target:<16} {state}"
        )


@app.command()
def compare(
    baseline: Annotated[str, typer.Option("--baseline", help="Baseline run id.")],
    candidate: Annotated[str, typer.Option("--candidate", help="Candidate run id.")],
    dataset_path: Annotated[Path, typer.Option("--dataset", help="Path to the manifest.")],
    store_path: Annotated[Path, typer.Option("--store")] = Path("runs.sqlite"),
    scorer: Annotated[str, typer.Option("--scorer")] = "exact_match",
    tolerance: Annotated[
        float, typer.Option("--tolerance", help="How much worse a slice may get.")
    ] = 0.05,
) -> None:
    """Compare two runs slice by slice and set an exit code from the verdicts."""
    try:
        dataset = load_dataset(dataset_path)
        with RunStore.open(store_path) as store:
            comparisons = compare_slices(
                store,
                dataset,
                store.resolve(baseline),
                store.resolve(candidate),
                scorer,
                default_tolerance=tolerance,
            )
    except (DatasetError, StoreError, ValueError) as error:
        _fail(str(error))

    _print_comparisons(comparisons, tolerance)
    raise typer.Exit(_exit_code(comparisons))


def _print_comparisons(comparisons: list[SliceComparison], tolerance: float) -> None:
    """Render one line per slice, widest problem first in the eye's path."""
    typer.echo(f"\nscorer={comparisons[0].scorer if comparisons else '-'}  tolerance={tolerance}")
    typer.echo(f"{'slice':<24} {'n':>4}  {'difference':>26}  verdict")

    for comparison in comparisons:
        interval = comparison.result.interval
        name = f"{comparison.dimension}={comparison.value}"
        band = f"{interval.point:+.3f} ({interval.low:+.3f} to {interval.high:+.3f})"
        colour = {
            Verdict.NON_INFERIOR: typer.colors.GREEN,
            Verdict.REGRESSED: typer.colors.RED,
            Verdict.INCONCLUSIVE: typer.colors.YELLOW,
        }[comparison.verdict]
        typer.secho(
            f"{name:<24} {comparison.result.n:>4}  {band:>26}  {comparison.verdict.value}",
            fg=colour,
        )
        if not comparison.is_complete:
            typer.secho(
                f"{'':<24}       {len(comparison.unpaired)} item(s) scored in only one run",
                fg=typer.colors.YELLOW,
            )

    inconclusive = [c for c in comparisons if c.verdict is Verdict.INCONCLUSIVE]
    if inconclusive:
        typer.echo(
            f"\n{len(inconclusive)} slice(s) could not answer at this tolerance. "
            "That is a gap in coverage, not a pass: add items or widen the tolerance on purpose."
        )


def _exit_code(comparisons: list[SliceComparison]) -> Exit:
    """Pick the exit code: regression beats inconclusive, which beats success."""
    if not comparisons:
        return Exit.INCONCLUSIVE
    if any(c.verdict is Verdict.REGRESSED for c in comparisons):
        return Exit.REGRESSED
    if any(c.verdict is Verdict.INCONCLUSIVE for c in comparisons):
        return Exit.INCONCLUSIVE
    return Exit.OK


def main() -> None:
    """Entry point for the `evalgate` command."""
    app()


if __name__ == "__main__":
    main()
