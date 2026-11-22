"""Command line for evalgate: produce a results file, and compare two of them.

Deliberate boundary: this module parses arguments, prints and chooses an exit code. Every
decision it reports was made in `gate`, every number in `stats`, and all wording in `report`.
Keeping it that thin means what a person reads here is the same object a test asserts on.

The exit code is the product, because it is what CI reads:

- `0`: no gate slice regressed. Slices that could not answer are reported as not guarded, and
  still exit 0 unless the configuration says `on_inconclusive = "fail"`.
- `1`: a gate slice regressed.
- `2`: no gate slice regressed, but at least one could not answer and the policy is `fail`.
- `3`: the command could not run at all -- a bad path, an unloadable target, a broken config.
  Kept apart from `1` so a typo can never be read as "the candidate regressed".
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer

from evalgate import gate
from evalgate.config import Config, ConfigError, load_config
from evalgate.dataset import Item
from evalgate.model_run import ModelRunError, plan_model_run
from evalgate.ollama import Ollama, OllamaError
from evalgate.report import render_markdown, render_text
from evalgate.results import (
    DatasetRef,
    ItemResult,
    ResultsError,
    ResultsHeader,
    read_results,
    write_results,
)
from evalgate.runner import TargetError, load_target, run_dataset
from evalgate.scorers import ScorerError

__all__ = ["Exit", "app", "exit_code", "main"]


class Exit(IntEnum):
    """Process exit codes."""

    OK = 0
    """No gate slice regressed, and the policy allows any slice that was not guarded."""

    REGRESSED = 1
    """A gate slice got worse by more than its tolerance."""

    NOT_GUARDED = 2
    """A gate slice could not answer, and the policy is to fail on that."""

    MISUSE = 3
    """The command could not run."""


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="A CI check that fails a pull request when an LLM feature gets worse on a slice.",
)

ConfigOption = Annotated[
    Path, typer.Option("--config", help="Path to evalgate.toml.", show_default=True)
]


def exit_code(outcome: gate.Outcome, on_inconclusive: Literal["warn", "fail"]) -> Exit:
    """Turn the check's outcome and the configured policy into an exit code.

    Args:
        outcome: The comparison's outcome.
        on_inconclusive: Whether a gate slice that could not answer fails the build.

    Returns:
        The exit code.
    """
    if outcome is gate.Outcome.REGRESSED:
        return Exit.REGRESSED
    if outcome is gate.Outcome.UNGUARDED and on_inconclusive == "fail":
        return Exit.NOT_GUARDED
    return Exit.OK


def _fail(message: str) -> NoReturn:
    """Print an error and exit with the misuse code.

    Typed as never returning, so the type checker treats the lines after a call as unreachable.
    """
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(Exit.MISUSE)


def _config(path: Path) -> Config:
    """Load the configuration, or exit with a readable error."""
    try:
        return load_config(path)
    except ConfigError as error:
        _fail(str(error))


@app.command()
def run(
    label: Annotated[str, typer.Option("--label", help="What to call this run.")],
    out: Annotated[Path, typer.Option("--out", help="Results file to write.")],
    config_path: ConfigOption = Path("evalgate.toml"),
    target: Annotated[
        str | None,
        typer.Option("--target", help="module:function to call, overriding [target]."),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="No per-item progress.")] = False,
    ollama_model: Annotated[
        str | None, typer.Option(help="Ollama model to record or replay.")
    ] = None,
    ollama_port: Annotated[int, typer.Option(min=1, max=65535)] = 11434,
    cassette_dir: Annotated[
        Path | None, typer.Option(help="Directory of model recordings.")
    ] = None,
    allow_local_model: Annotated[
        bool, typer.Option(help="Allow cache misses to call cloud-disabled local Ollama.")
    ] = False,
    max_calls: Annotated[int | None, typer.Option(min=0, help="Cap on live generations.")] = None,
    max_cost_usd: Annotated[
        float | None, typer.Option(help="Allowed inference spend in USD.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Print the model call estimate without writing.")
    ] = False,
    redact: Annotated[
        str | None, typer.Option(help="Trusted module:function for model outputs.")
    ] = None,
) -> None:
    """Run every item of the dataset through a target and write a results file."""
    config = _config(config_path)
    model_options_used = (
        ollama_port != 11434
        or cassette_dir is not None
        or allow_local_model
        or max_calls is not None
        or max_cost_usd is not None
        or dry_run
        or redact is not None
    )
    if ollama_model is not None:
        if target is not None or cassette_dir is None:
            _fail("--ollama-model requires --cassette-dir and cannot be combined with --target")
        if max_cost_usd is not None and not 0 <= max_cost_usd < float("inf"):
            _fail("--max-cost-usd must be finite and non-negative")
        if allow_local_model and max_calls is None and not dry_run:
            _fail("--allow-local-model requires an explicit --max-calls")
        _run_model(
            config,
            label,
            out,
            ollama_model,
            ollama_port,
            cassette_dir,
            allow_local_model=allow_local_model,
            max_calls=max_calls or 0,
            dry_run=dry_run,
            redact=redact,
        )
        return
    if model_options_used:
        _fail("model run options require --ollama-model")

    spec = config.target
    reference = target or (spec.reference if spec is not None else None)
    if reference is None:
        _fail("no target: set [target] callable in the config, or pass --target")
    # With no [target] table, the config's own directory is where a target is most likely to be.
    pythonpath = spec.pythonpath if spec is not None else (config.source.resolve().parent,)

    try:
        function = load_target(reference, pythonpath)
    except TargetError as error:
        _fail(str(error))

    dataset = config.dataset
    progress = None if quiet else _progress(len(dataset))
    items = run_dataset(dataset, function, on_item=progress)
    if progress is not None:
        typer.echo("")
    _write_run(config, label, out, items)


def _run_model(
    config: Config,
    label: str,
    out: Path,
    model: str,
    port: int,
    directory: Path,
    *,
    allow_local_model: bool,
    max_calls: int,
    dry_run: bool,
    redact: str | None,
) -> None:
    """Finish model preflight, then either report the estimate or produce results."""
    try:
        provider = Ollama(model, port)
        plan = plan_model_run(config.dataset, provider, directory, redactor=redact)
        typer.echo(
            f"{plan.hits} replay hit(s), {plan.unrecorded_items} item(s) missing, "
            f"up to {plan.misses} live generation(s) if enabled, estimated USD 0.00"
        )
        if dry_run:
            return
        redactor = None
        if allow_local_model and 0 < plan.misses <= max_calls:
            provider.validate_local()
            if redact is not None:
                redactor = load_target(redact, (config.source.resolve().parent,))
        items = plan.execute(allow_local=allow_local_model, max_calls=max_calls, redact=redactor)
    except (OllamaError, ModelRunError) as error:
        _fail(str(error))
    except TargetError:
        _fail("could not load the model output redactor")
    not_run = 0 if allow_local_model else plan.unrecorded_items
    _write_run(config, label, out, items, not_run=not_run)
    if not_run:
        typer.echo(f"  {not_run} item(s) not run (no recording)")


def _write_run(
    config: Config, label: str, out: Path, items: tuple[ItemResult, ...], *, not_run: int = 0
) -> None:
    """Write a run in the same results format for every target."""
    manifest = config.dataset.manifest
    header = ResultsHeader(
        label=label,
        dataset=DatasetRef(
            name=manifest.name, version=manifest.version, sha256=manifest.content_sha256
        ),
        created_at=datetime.now(UTC),
        not_run=not_run or None,
    )
    try:
        write_results(out, header, items)
    except ResultsError as error:
        _fail(str(error))

    errors = sum(1 for item in items if item.error is not None)
    typer.echo(f"{label}: {len(items)} items from {manifest.name} v{manifest.version} -> {out}")
    if errors:
        typer.secho(f"  {errors} item(s) raised; they count as failures", fg=typer.colors.YELLOW)


def _progress(total: int) -> Callable[[int, Item], None]:
    """A per-item callback that overwrites one terminal line."""

    def report(index: int, item: Item) -> None:
        del item
        typer.echo(f"\r  {index + 1}/{total}", nl=False)

    return report


@app.command()
def compare(
    base: Annotated[Path, typer.Argument(help="Results file of the baseline.")],
    head: Annotated[Path, typer.Argument(help="Results file of the version under test.")],
    config_path: ConfigOption = Path("evalgate.toml"),
    markdown: Annotated[
        Path | None, typer.Option("--markdown", help="Also write the report here, as markdown.")
    ] = None,
) -> None:
    """Compare two results files slice by slice, print the report, and exit with a verdict."""
    config = _config(config_path)
    try:
        comparison = gate.compare(config, read_results(base), read_results(head))
    except (ResultsError, gate.GateError, ScorerError) as error:
        _fail(str(error))

    confidence, policy = config.confidence, config.on_inconclusive
    typer.echo(render_text(comparison, confidence=confidence, on_inconclusive=policy))
    if markdown is not None:
        try:
            markdown.parent.mkdir(parents=True, exist_ok=True)
            text = render_markdown(comparison, confidence=confidence, on_inconclusive=policy)
            markdown.write_text(text, encoding="utf-8")
        except OSError as error:
            _fail(f"{markdown}: cannot write report: {error}")
    raise typer.Exit(exit_code(comparison.outcome, config.on_inconclusive))


def main() -> None:
    """Entry point for the `evalgate` command."""
    app()


if __name__ == "__main__":
    main()
