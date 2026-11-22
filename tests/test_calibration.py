"""Public calibration contract on synthetic, offline inputs."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import typer
from numpy.random import default_rng
from typer.testing import CliRunner

from evalgate.calibration import (
    CalibrationError,
    CalibrationRow,
    calibrate,
    calibrate_command,
    read_calibration,
)


def write_rows(path: Path, *rows: object) -> Path:
    """Write JSONL fixtures without depending on the production loader."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_reports_agreement_separately_by_rubric() -> None:
    rows = (
        CalibrationRow(id="1", rubric="correct", human=True, judge=True, repeat=True),
        CalibrationRow(
            id="2", rubric="correct", human=False, judge=True, repeat=False, length_variant=True
        ),
        CalibrationRow(id="3", rubric="correct", human=False, judge=False),
        CalibrationRow(id="4", rubric="correct", human=False, judge=False),
        CalibrationRow(id="1", rubric="tone", human=True, judge=True),
    )

    correct, tone = calibrate(rows)
    assert correct.rubric == "correct"
    assert asdict(correct.confusion) == {
        "true_positive": 1,
        "true_negative": 2,
        "false_positive": 1,
        "false_negative": 0,
    }
    assert correct.kappa == pytest.approx(0.5)
    assert correct.bias == pytest.approx(0.25)
    repeat, position, length, family = correct.probes
    assert (repeat.count, repeat.flips, repeat.acceptance_delta) == (2, 1, -0.5)
    assert (length.count, length.flips, length.acceptance_delta) == (1, 0, 0.0)
    assert position.count == family.count == 0
    assert position.acceptance_delta is None
    assert tone.kappa is None
    assert tone.human_rate == 1.0
    assert tone.corrected_interval == (1.0, 1.0)


def test_partial_labels_need_an_explicit_sampling_design() -> None:
    rows = (
        CalibrationRow(id="1", rubric="correct", human=False, judge=True),
        CalibrationRow(id="2", rubric="correct", human=False, judge=False),
        CalibrationRow(id="3", rubric="correct", judge=True),
        CalibrationRow(id="4", rubric="correct", judge=True),
    )

    report = calibrate(rows)[0]
    assert report.judge_rate == 0.75
    assert report.human_rate == 0.0
    assert report.corrected_rate is None
    assert report.corrected_interval is None

    sampled = calibrate(rows, random_sample=True)[0]
    assert sampled.corrected_rate == 0.25
    assert sampled.corrected_interval == (0.0, 0.5)
    assert sampled.human_interval == (0.0, 0.5)


def test_no_human_labels_is_unavailable_even_when_sampling_is_declared() -> None:
    report = calibrate((CalibrationRow(id="1", rubric="correct", judge=True),), random_sample=True)[
        0
    ]
    assert report.kappa is None
    assert report.bias is None
    assert report.human_rate is None
    assert report.corrected_rate is None


def test_human_agreement_and_ensembles_use_matched_evidence() -> None:
    rows = (
        CalibrationRow(
            id="a",
            rubric="r",
            judge=True,
            human=True,
            second_human=True,
            ensemble_2=True,
            ensemble_3=True,
            judge_cost_usd=0.10,
            ensemble_2_cost_usd=0.20,
            ensemble_3_cost_usd=0.30,
        ),
        CalibrationRow(
            id="b",
            rubric="r",
            judge=True,
            human=False,
            second_human=False,
            ensemble_2=False,
            ensemble_3=False,
            judge_cost_usd=0.10,
            ensemble_2_cost_usd=0.20,
            ensemble_3_cost_usd=0.30,
        ),
        CalibrationRow(id="c", rubric="r", judge=True, human=True, ensemble_2=False),
    )
    report = calibrate(rows)[0]
    assert report.second_rater_count == 2
    assert report.second_rater_kappa == 1.0
    assert report.ensemble_count == 2
    one, two, three = report.ensembles
    assert one.kappa == 0.0
    assert two.kappa == three.kappa == 1.0
    assert two.total_cost_usd == pytest.approx(0.40)
    assert three.gain_per_added_usd == pytest.approx(2.5)


@pytest.mark.parametrize(
    "extra",
    [{"judge": 1}, {"human": "false"}, {"human_pass": True}, {"rubric": "bad\x9brubric"}],
)
def test_loader_refuses_loose_types_and_unknown_fields(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    path = write_rows(tmp_path / "labels.jsonl", {"id": "a", "rubric": "r", "judge": True} | extra)
    with pytest.raises(CalibrationError, match=r"labels.jsonl:1"):
        read_calibration(path)


def test_loader_rejects_duplicate_item_within_the_same_rubric(tmp_path: Path) -> None:
    line = {"id": "a", "rubric": "r", "judge": True}
    path = write_rows(tmp_path / "labels.jsonl", line, line)
    with pytest.raises(CalibrationError, match=r"labels.jsonl:2.*duplicate"):
        read_calibration(path)


@pytest.mark.parametrize("fingerprint", ["AB" * 32, "ab" * 31, "xz" * 32, " " + "ab" * 32, 123])
def test_loader_rejects_invalid_judgement_fingerprint(tmp_path: Path, fingerprint: object) -> None:
    path = write_rows(
        tmp_path / "labels.jsonl",
        {"id": "a", "rubric": "r", "judge": True, "judgement_sha256": fingerprint},
    )
    with pytest.raises(CalibrationError, match=r"labels.jsonl:1"):
        read_calibration(path)


@pytest.mark.parametrize("other", [None, "bc" * 32])
def test_refuses_mixed_judgement_fingerprints_per_rubric(tmp_path: Path, other: str | None) -> None:
    path = write_rows(
        tmp_path / "labels.jsonl",
        {"id": "a", "rubric": "r", "judge": True, "judgement_sha256": "ab" * 32},
        {"id": "b", "rubric": "r", "judge": True, "judgement_sha256": other},
    )
    with pytest.raises(CalibrationError, match=r"labels.jsonl:2.*fingerprint"):
        read_calibration(path)

    rows = (
        CalibrationRow(id="a", rubric="r", judge=True, judgement_sha256="ab" * 32),
        CalibrationRow(id="b", rubric="r", judge=True, judgement_sha256=other),
    )
    with pytest.raises(CalibrationError, match="fingerprint"):
        calibrate(rows)


def test_fingerprint_can_differ_across_rubrics(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "labels.jsonl",
        {"id": "a", "rubric": "r", "judge": True, "judgement_sha256": "ab" * 32},
        {"id": "b", "rubric": "r", "judge": True, "judgement_sha256": "ab" * 32},
        {"id": "a", "rubric": "other", "judge": False, "judgement_sha256": "bc" * 32},
    )
    rows = read_calibration(path)
    assert rows[0].judgement_sha256 == "ab" * 32
    assert [report.rubric for report in calibrate(rows)] == ["other", "r"]


def test_cli_reports_counts_and_missing_probe_evidence(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "labels.jsonl", {"id": "a", "rubric": "r", "judge": False, "human": False}
    )
    app = typer.Typer()
    app.command()(calibrate_command)
    result = CliRunner().invoke(app, [str(path)])
    assert result.exit_code == 0, result.output
    assert "1/1 human-labeled" in result.output
    assert "kappa: unavailable" in result.output
    assert "true negative=1" in result.output
    assert "Position swap: not measured" in result.output


def test_seeded_synthetic_population_showcases_bias_correction() -> None:
    # The fixed judge overaccepts one in 20 cases. Its residual has much lower variance
    # than the human label. This scenario demonstrates a possible gain, never a guarantee.
    size = 4000
    sample_size = 2000
    generator = default_rng(271828)
    errors = []
    width_gains = []
    for _ in range(32):
        reviewed = {int(index) for index in generator.choice(size, size=sample_size, replace=False)}
        rows = tuple(
            CalibrationRow(
                id=str(index),
                rubric="synthetic",
                human=index < size // 2 if index in reviewed else None,
                judge=index < size * 55 // 100,
            )
            for index in range(size)
        )
        report = calibrate(rows, random_sample=True)[0]
        assert report.judge_rate == pytest.approx(0.55)
        assert report.corrected_rate is not None
        assert report.corrected_interval is not None
        assert report.human_interval is not None
        low, high = report.corrected_interval
        baseline_low, baseline_high = report.human_interval
        assert low <= 0.5 <= high
        errors.append(abs(report.corrected_rate - 0.5))
        width_gains.append(baseline_high - baseline_low - (high - low))
    assert sum(errors) / len(errors) < 0.01
    assert sum(width_gains) / len(width_gains) > 0
