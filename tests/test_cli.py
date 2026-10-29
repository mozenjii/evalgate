"""The command line, end to end on the example dataset.

This is the acceptance test for the whole chain: load a versioned dataset, run two versions of
a target, score every item, pair the runs by item within each slice, and turn the intervals
into an exit code. The example seeds a known regression in one slice, so the test can check
that the gate finds it there and does not claim harm elsewhere.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalgate.cli import Exit, app

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "examples" / "triage" / "manifest.json"

runner = CliRunner()


def run_version(store: Path, target: str, label: str) -> str:
    """Run one version of the example target and return its run id."""
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(MANIFEST),
            "--target",
            target,
            "--label",
            label,
            "--store",
            str(store),
            "--quiet",
        ],
    )
    assert result.exit_code == Exit.OK, result.output
    for line in result.output.splitlines():
        if line.startswith("run "):
            return line.split()[1]
    raise AssertionError(f"no run id in output:\n{result.output}")


@pytest.fixture
def both_runs(tmp_path: Path) -> tuple[Path, str, str]:
    """Run the baseline and the seeded-regression candidate over the example dataset."""
    store = tmp_path / "runs.sqlite"
    baseline = run_version(store, "target:classify_v1", "v1")
    candidate = run_version(store, "target:classify_v2", "v2")
    return store, baseline, candidate


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_run_scores_every_item_in_the_example_dataset(tmp_path: Path) -> None:
    store = tmp_path / "runs.sqlite"
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(MANIFEST),
            "--target",
            "target:classify_v1",
            "--label",
            "v1",
            "--store",
            str(store),
            "--quiet",
        ],
    )

    assert result.exit_code == Exit.OK
    assert "120 items" in result.output
    assert "exact_match  120/120" in result.output


def test_the_seeded_regression_shows_in_the_run_totals(tmp_path: Path) -> None:
    store = tmp_path / "runs.sqlite"
    run_version(store, "target:classify_v1", "v1")
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(MANIFEST),
            "--target",
            "target:classify_v2",
            "--label",
            "v2",
            "--store",
            str(store),
            "--quiet",
        ],
    )

    # Two warranty cues were removed, costing it the forty warranty items minus the ones the
    # remaining cues still catch.
    assert "exact_match  96/120" in result.output


def test_runs_are_listed_after_being_recorded(both_runs: tuple[Path, str, str]) -> None:
    store, baseline, candidate = both_runs
    result = runner.invoke(app, ["runs", "--store", str(store)])

    assert result.exit_code == Exit.OK
    assert baseline[:12] in result.output
    assert candidate[:12] in result.output
    assert "UNFINISHED" not in result.output


# --------------------------------------------------------------------------- #
# Comparing
# --------------------------------------------------------------------------- #


def test_the_gate_finds_the_regression_in_the_slice_that_has_it(
    both_runs: tuple[Path, str, str],
) -> None:
    store, baseline, candidate = both_runs
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            baseline,
            "--candidate",
            candidate,
            "--dataset",
            str(MANIFEST),
            "--store",
            str(store),
            "--tolerance",
            "0.05",
        ],
    )

    assert result.exit_code == Exit.REGRESSED
    lines = {line.split()[0]: line for line in result.output.splitlines() if "=" in line}
    assert "regressed" in lines["intent=warranty"]
    # The other two intents were untouched, so the gate must not claim harm there.
    assert "regressed" not in lines["intent=billing"]
    assert "regressed" not in lines["intent=refunds"]


def test_an_unchanged_candidate_passes_when_the_slice_can_support_the_tolerance(
    both_runs: tuple[Path, str, str],
) -> None:
    store, baseline, _ = both_runs
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            baseline,
            "--candidate",
            baseline,
            "--dataset",
            str(MANIFEST),
            "--store",
            str(store),
            "--tolerance",
            "0.10",
        ],
    )

    assert result.exit_code == Exit.OK
    assert "non_inferior" in result.output


def test_a_tolerance_the_slices_cannot_support_is_inconclusive_not_a_pass(
    both_runs: tuple[Path, str, str],
) -> None:
    # Identical runs, but forty items cannot rule out a two-point drop. Sharing an exit code
    # with success here would turn every underpowered slice into a green tick.
    store, baseline, _ = both_runs
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            baseline,
            "--candidate",
            baseline,
            "--dataset",
            str(MANIFEST),
            "--store",
            str(store),
            "--tolerance",
            "0.02",
        ],
    )

    assert result.exit_code == Exit.INCONCLUSIVE
    assert "inconclusive" in result.output
    assert "gap in coverage, not a pass" in result.output


def test_every_exit_code_is_reachable(both_runs: tuple[Path, str, str]) -> None:
    store, baseline, candidate = both_runs

    def compare(baseline_id: str, candidate_id: str, tolerance: str) -> int:
        result = runner.invoke(
            app,
            [
                "compare",
                "--baseline",
                baseline_id,
                "--candidate",
                candidate_id,
                "--dataset",
                str(MANIFEST),
                "--store",
                str(store),
                "--tolerance",
                tolerance,
            ],
        )
        return result.exit_code

    assert {
        compare(baseline, baseline, "0.10"),
        compare(baseline, candidate, "0.05"),
        compare(baseline, baseline, "0.02"),
    } == {Exit.OK, Exit.REGRESSED, Exit.INCONCLUSIVE}


def test_a_run_id_prefix_is_enough(both_runs: tuple[Path, str, str]) -> None:
    # The listing prints a short prefix, so the next command has to accept one.
    store, baseline, candidate = both_runs
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            baseline[:12],
            "--candidate",
            candidate[:12],
            "--dataset",
            str(MANIFEST),
            "--store",
            str(store),
        ],
    )

    assert result.exit_code == Exit.REGRESSED


# --------------------------------------------------------------------------- #
# Misuse
# --------------------------------------------------------------------------- #


def test_an_unloadable_target_is_misuse_not_a_failed_gate(tmp_path: Path) -> None:
    # A typo in a target name must not be reportable as "the candidate regressed".
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(MANIFEST),
            "--target",
            "target:does_not_exist",
            "--label",
            "v1",
            "--store",
            str(tmp_path / "runs.sqlite"),
        ],
    )

    assert result.exit_code == Exit.MISUSE
    assert "has no attribute" in result.output


def test_a_target_without_a_colon_is_rejected(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(MANIFEST),
            "--target",
            "target.classify_v1",
            "--label",
            "v1",
            "--store",
            str(tmp_path / "runs.sqlite"),
        ],
    )

    assert result.exit_code == Exit.MISUSE
    assert "module:function" in result.output


def test_a_missing_dataset_is_misuse(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--dataset",
            str(tmp_path / "nope.json"),
            "--target",
            "target:classify_v1",
            "--label",
            "v1",
            "--store",
            str(tmp_path / "runs.sqlite"),
        ],
    )

    assert result.exit_code == Exit.MISUSE
    assert "cannot read manifest" in result.output


def test_an_unknown_run_id_is_misuse(both_runs: tuple[Path, str, str]) -> None:
    store, baseline, _ = both_runs
    result = runner.invoke(
        app,
        [
            "compare",
            "--baseline",
            baseline,
            "--candidate",
            "nosuchrun",
            "--dataset",
            str(MANIFEST),
            "--store",
            str(store),
        ],
    )

    assert result.exit_code == Exit.MISUSE
    assert "no such run" in result.output


def test_the_example_dataset_still_matches_its_manifest() -> None:
    # The dataset is generated and committed. If someone edits items.jsonl by hand instead of
    # the generator, the hash check catches it here rather than in a confusing comparison.
    from evalgate.dataset import load_dataset

    dataset = load_dataset(MANIFEST)
    assert len(dataset) == 120
    assert dataset.slice_sizes()[("intent", "warranty")] == 40
