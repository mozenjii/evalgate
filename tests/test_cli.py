"""The command line end to end, on the example dataset.

This is the acceptance test for the walking skeleton: produce a results file for two versions of
a target, compare them per slice, and turn the answer into an exit code under a stated policy.
The example seeds a known regression in one slice, so the test can check that the gate finds it
there, and only there.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from evalgate.cli import Exit, app, exit_code
from evalgate.gate import Outcome
from evalgate.report import MARKER
from evalgate.results import read_results

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "examples" / "triage"
CONFIG = EXAMPLE / "evalgate.toml"

runner = CliRunner()


def invoke(*args: str) -> Result:
    return runner.invoke(app, list(args))


def produce(root: Path, label: str, target: str | None = None, config: Path = CONFIG) -> Path:
    """Write a results file for one version of the example target."""
    out = root / f"{label}.jsonl"
    args = ["run", "--config", str(config), "--label", label, "--out", str(out), "--quiet"]
    if target is not None:
        args += ["--target", target]
    result = invoke(*args)
    assert result.exit_code == Exit.OK, result.output
    return out


def compare(base: Path, head: Path, config: Path = CONFIG, *extra: str) -> Result:
    return invoke("compare", str(base), str(head), "--config", str(config), *extra)


def config_with(root: Path, policy: str) -> Path:
    """A copy of the example's configuration with a different policy, pointing at its data."""
    body = f"""
[dataset]
manifest = "{(EXAMPLE / "manifest.json").as_posix()}"

[target]
callable = "target:classify_v1"
pythonpath = ["{EXAMPLE.as_posix()}"]

[[scorer]]
type = "exact_match"

[[gate]]
scorer = "exact_match"
slices = ["intent=*"]
tolerance = 0.05

[policy]
on_inconclusive = "{policy}"
"""
    path = root / "evalgate.toml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def versions(tmp_path: Path) -> tuple[Path, Path]:
    """Results for the baseline and for the candidate with the seeded warranty regression."""
    return produce(tmp_path, "v1"), produce(tmp_path, "v2", "target:classify_v2")


def row(output: str, slice_name: str) -> str:
    return next(line for line in output.splitlines() if line.strip().startswith(slice_name))


# --------------------------------------------------------------------------- run


def test_run_writes_a_results_file_for_every_item(tmp_path: Path) -> None:
    out = produce(tmp_path, "v1")

    results = read_results(out)
    assert results.header is not None
    assert results.header.label == "v1"
    assert results.header.dataset is not None
    assert results.header.dataset.name == "triage"
    assert len(results.items) == 120
    assert all(item.input_sha256 is not None for item in results.items)


def test_target_on_the_command_line_overrides_the_config(tmp_path: Path) -> None:
    baseline = read_results(produce(tmp_path, "v1")).by_id()
    candidate = read_results(produce(tmp_path, "v2", "target:classify_v2")).by_id()

    differing = [key for key in baseline if baseline[key].output != candidate[key].output]
    assert differing
    assert all(key.startswith("warranty-") for key in differing)


# --------------------------------------------------------------------------- compare


def test_the_gate_finds_the_seeded_regression_in_its_slice(versions: tuple[Path, Path]) -> None:
    result = compare(*versions)

    assert result.exit_code == Exit.REGRESSED
    assert result.output.startswith("evalgate: regression found")
    assert row(result.output, "intent=warranty").endswith("regressed")
    # Billing and refunds did not change; forty items cannot certify five points, and the report
    # says so instead of calling them passes.
    assert row(result.output, "intent=billing").endswith("not guarded")
    assert row(result.output, "intent=refunds").endswith("not guarded")


def test_the_markdown_report_is_written_for_the_pull_request(
    versions: tuple[Path, Path], tmp_path: Path
) -> None:
    report = tmp_path / "out" / "report.md"

    compare(*versions, CONFIG, "--markdown", str(report))

    text = report.read_text(encoding="utf-8")
    assert text.startswith(MARKER)
    assert "| intent=warranty |" in text


def test_an_unchanged_run_warns_but_does_not_fail_under_the_default_policy(tmp_path: Path) -> None:
    baseline = produce(tmp_path, "v1")

    result = compare(baseline, baseline)

    assert result.exit_code == Exit.OK
    assert "not guarded" in result.output.splitlines()[0]


def test_the_fail_policy_turns_an_unguarded_check_into_a_failure(tmp_path: Path) -> None:
    config = config_with(tmp_path, "fail")
    baseline = produce(tmp_path, "v1", config=config)

    result = compare(baseline, baseline, config)

    assert result.exit_code == Exit.NOT_GUARDED


def test_exit_codes_follow_outcome_and_policy() -> None:
    assert exit_code(Outcome.OK, "fail") is Exit.OK
    assert exit_code(Outcome.UNGUARDED, "warn") is Exit.OK
    assert exit_code(Outcome.UNGUARDED, "fail") is Exit.NOT_GUARDED
    assert exit_code(Outcome.REGRESSED, "warn") is Exit.REGRESSED


# --------------------------------------------------------------------------- misuse


def test_a_missing_config_is_misuse_not_a_failed_gate(tmp_path: Path) -> None:
    # A typo must never be readable as "the candidate regressed".
    result = invoke("run", "--config", str(tmp_path / "nope.toml"), "--label", "x", "--out", "x")

    assert result.exit_code == Exit.MISUSE
    assert "cannot read config" in result.output


def test_a_missing_results_file_is_misuse(tmp_path: Path) -> None:
    result = compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl")

    assert result.exit_code == Exit.MISUSE
    assert "cannot read results" in result.output


def test_an_unloadable_target_is_misuse(tmp_path: Path) -> None:
    result = invoke(
        "run", "--config", str(CONFIG), "--label", "x", "--out", str(tmp_path / "x.jsonl"),
        "--target", "target:does_not_exist",
    )  # fmt: skip

    assert result.exit_code == Exit.MISUSE
    assert "has no attribute" in result.output


def test_a_config_without_a_target_needs_one_on_the_command_line(tmp_path: Path) -> None:
    config = config_with(tmp_path, "warn")
    body = config.read_text(encoding="utf-8")
    body = body.replace('[target]\ncallable = "target:classify_v1"\n', "")
    body = body.replace(f'pythonpath = ["{EXAMPLE.as_posix()}"]\n', "")
    config.write_text(body, encoding="utf-8")

    result = invoke("run", "--config", str(config), "--label", "x", "--out", str(tmp_path / "x"))

    assert result.exit_code == Exit.MISUSE
    assert "no target" in result.output


def test_the_example_dataset_still_matches_its_manifest() -> None:
    # The dataset is generated and committed. Editing items.jsonl by hand instead of the
    # generator is caught here rather than in a confusing comparison.
    from evalgate.dataset import load_dataset

    dataset = load_dataset(EXAMPLE / "manifest.json")
    assert len(dataset) == 120
    assert dataset.slice_sizes()[("intent", "warranty")] == 40
