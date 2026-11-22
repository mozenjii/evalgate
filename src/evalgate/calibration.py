"""Measure a binary judge against independently supplied human judgments.

The calibration guide specifies the sampling design, optional paired probes and cost evidence.
This module reads only decisions and optional costs, never raw outputs, and performs no inference
calls or writes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

__all__ = [
    "CalibrationError",
    "CalibrationRow",
    "calibrate",
    "calibrate_command",
    "read_calibration",
]

_Interval = tuple[float, float]
_Label = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[^\x00-\x1f\x7f-\x9f]+$")]
_Cost = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_Fingerprint = Annotated[
    str,
    StringConstraints(
        strip_whitespace=False, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$"
    ),
]


class CalibrationError(ValueError):
    """Invalid calibration data or inference settings."""


class CalibrationRow(BaseModel):
    """A frozen judge decision for one item and rubric, with optional audit evidence."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", str_strip_whitespace=True)

    id: _Label
    rubric: _Label
    judge: bool
    human: bool | None = None
    judgement_sha256: _Fingerprint | None = None
    second_human: bool | None = None
    repeat: bool | None = None
    position_swap: bool | None = None
    length_variant: bool | None = None
    family_variant: bool | None = None
    ensemble_2: bool | None = None
    ensemble_3: bool | None = None
    judge_cost_usd: _Cost | None = None
    ensemble_2_cost_usd: _Cost | None = None
    ensemble_3_cost_usd: _Cost | None = None


@dataclass(frozen=True)
class _Confusion:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def kappa(self) -> float | None:
        """Return agreement above chance when the marginal totals define it."""
        tp, tn, fp, fn = (
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
        )
        count = tp + tn + fp + fn
        agreement = count * (tp + tn)
        expected = (tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)
        denominator = count**2 - expected
        return (agreement - expected) / denominator if denominator else None


@dataclass(frozen=True)
class _ProbeResult:
    name: str
    count: int
    flips: int
    acceptance_delta: float | None


@dataclass(frozen=True)
class _EnsembleResult:
    judges: int
    kappa: float | None
    total_cost_usd: float | None
    gain_per_added_usd: float | None = None


@dataclass(frozen=True)
class _RubricReport:
    rubric: str
    count: int
    labeled: int
    confusion: _Confusion
    bias: float | None
    judge_rate: float
    human_rate: float | None
    corrected_rate: float | None
    corrected_interval: _Interval | None
    human_interval: _Interval | None
    probes: tuple[_ProbeResult, ...]
    second_rater_count: int
    second_rater_kappa: float | None
    ensemble_count: int
    ensembles: tuple[_EnsembleResult, ...]

    @property
    def kappa(self) -> float | None:
        """Return the judge's Cohen kappa for the reviewed cases."""
        return self.confusion.kappa


_PROBES: tuple[tuple[str, Callable[[CalibrationRow], bool | None]], ...] = (
    ("Repeat", lambda row: row.repeat),
    ("Position swap", lambda row: row.position_swap),
    ("Length variant", lambda row: row.length_variant),
    ("Family variant", lambda row: row.family_variant),
)


def _remember_fingerprint(
    fingerprints: dict[str, str | None], row: CalibrationRow, *, location: str = ""
) -> None:
    recorded = fingerprints.setdefault(row.rubric, row.judgement_sha256)
    if recorded != row.judgement_sha256:
        raise CalibrationError(f"{location}mixed judgement fingerprints within a rubric")


def read_calibration(path: Path) -> tuple[CalibrationRow, ...]:
    """Read strict UTF-8 JSONL with one decision record per item and rubric."""
    rows: list[CalibrationRow] = []
    seen: set[tuple[str, str]] = set()
    fingerprints: dict[str, str | None] = {}
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = CalibrationRow.model_validate_json(line)
                except ValidationError as error:
                    issues = error.errors(include_input=False, include_context=False)
                    # Do not echo external labels, costs, invalid JSON, or unexpected keys.
                    kind = issues[0]["type"]
                    raise CalibrationError(
                        f"{path}:{number}: invalid calibration row ({kind})"
                    ) from None
                key = row.rubric, row.id
                if key in seen:
                    raise CalibrationError(f"{path}:{number}: duplicate item and rubric")
                seen.add(key)
                _remember_fingerprint(fingerprints, row, location=f"{path}:{number}: ")
                rows.append(row)
    except (OSError, UnicodeError) as error:
        raise CalibrationError(f"{path}: cannot read UTF-8 calibration JSONL") from error
    if not rows:
        raise CalibrationError(f"{path}: no calibration rows")
    return tuple(rows)


def calibrate(
    rows: Sequence[CalibrationRow], *, random_sample: bool = False, confidence: float = 0.95
) -> tuple[_RubricReport, ...]:
    """Summarize agreement per rubric and infer population accuracy when labels support it."""
    if not rows:
        raise CalibrationError("no calibration rows")
    if not 0 < confidence < 1:
        raise CalibrationError("confidence must be strictly between 0 and 1")
    groups: dict[str, list[CalibrationRow]] = {}
    fingerprints: dict[str, str | None] = {}
    for row in rows:
        _remember_fingerprint(fingerprints, row)
        groups.setdefault(row.rubric, []).append(row)
    return tuple(
        _report(rubric, group, random_sample=random_sample, confidence=confidence)
        for rubric, group in sorted(groups.items())
    )


def _confusion(pairs: Sequence[tuple[bool, bool]]) -> _Confusion:
    return _Confusion(
        true_positive=sum(reference and tested for reference, tested in pairs),
        true_negative=sum(not reference and not tested for reference, tested in pairs),
        false_positive=sum(not reference and tested for reference, tested in pairs),
        false_negative=sum(reference and not tested for reference, tested in pairs),
    )


def _probe(
    rows: Sequence[CalibrationRow], name: str, value: Callable[[CalibrationRow], bool | None]
) -> _ProbeResult:
    changes = tuple(
        int(probe) - int(row.judge) for row in rows if (probe := value(row)) is not None
    )
    return _ProbeResult(
        name, len(changes), sum(map(abs, changes)), mean(changes) if changes else None
    )


@dataclass(frozen=True)
class _Estimates:
    corrected_rate: float | None = None
    corrected_interval: _Interval | None = None
    human_interval: _Interval | None = None


def _estimates(
    pairs: Sequence[tuple[bool, bool]],
    *,
    size: int,
    judge_rate: float,
    random_sample: bool,
    confidence: float,
) -> _Estimates:
    count = len(pairs)
    if count == 0:
        return _Estimates()
    observed = tuple(int(human) for human, _ in pairs)
    successes = sum(observed)
    if count == size:
        truth = successes / size
        return _Estimates(truth, (truth, truth), (truth, truth))
    if not random_sample:
        return _Estimates()
    differences = tuple(int(human) - int(judge) for human, judge in pairs)
    corrected = judge_rate + mean(differences)
    # These deterministic bounds hold for every completion of the unreviewed cases.
    feasible = successes / size, (successes + size - count) / size
    return _Estimates(
        corrected,
        _bounded(corrected, _radius(differences, size, 2.0, confidence), feasible),
        _bounded(mean(observed), _radius(observed, size, 1.0, confidence), feasible),
    )


def _radius(values: Sequence[int], size: int, width: float, confidence: float) -> float:
    # Bardenet & Maillard Theorem 4.3 has one-sided error at most 5 * delta.
    # Apply it to X and -X with delta = alpha / 10. Its variance uses divisor n.
    count = len(values)
    fraction = count / size
    correction = (1 - (count - 1) / size) if fraction <= 0.5 else (1 - fraction) * (1 + 1 / count)
    threshold = log(10 / (1 - confidence))
    kappa = 7 / 3 + sqrt(3) / 2
    return (
        pstdev(values) * sqrt(2 * correction * threshold / count)
        + kappa * width * threshold / count
    )


def _bounded(estimate: float, radius: float, feasible: _Interval) -> _Interval:
    low = max(feasible[0], estimate - radius)
    high = min(feasible[1], estimate + radius)
    # An empty intersection has already missed the target. Use the deterministic
    # envelope rather than emitting an impossible interval.
    return (low, high) if low <= high else feasible


def _sum_costs(costs: Sequence[float | None]) -> float | None:
    if any(cost is None for cost in costs):
        return None
    return sum(cost for cost in costs if cost is not None)


@dataclass(frozen=True)
class _EnsembleRow:
    human: bool
    judge: bool
    two: bool
    three: bool
    costs: tuple[float | None, float | None, float | None]


def _ensemble(rows: Sequence[CalibrationRow]) -> tuple[int, tuple[_EnsembleResult, ...]]:
    matched = tuple(
        _EnsembleRow(
            row.human,
            row.judge,
            row.ensemble_2,
            row.ensemble_3,
            (row.judge_cost_usd, row.ensemble_2_cost_usd, row.ensemble_3_cost_usd),
        )
        for row in rows
        if row.human is not None and row.ensemble_2 is not None and row.ensemble_3 is not None
    )
    if not matched:
        return 0, ()
    kappas = tuple(
        _confusion(tuple(zip((row.human for row in matched), verdicts, strict=True))).kappa
        for verdicts in (
            (row.judge for row in matched),
            (row.two for row in matched),
            (row.three for row in matched),
        )
    )
    costs = tuple(
        _sum_costs(column) for column in zip(*(row.costs for row in matched), strict=True)
    )
    reports = tuple(
        _EnsembleResult(judges, kappa, cost, _cost_gain(kappas[0], costs[0], kappa, cost))
        for judges, (kappa, cost) in enumerate(zip(kappas, costs, strict=True), start=1)
    )
    return len(matched), reports


def _cost_gain(
    baseline: float | None, baseline_cost: float | None, kappa: float | None, cost: float | None
) -> float | None:
    if (
        baseline is None
        or baseline_cost is None
        or kappa is None
        or cost is None
        or cost <= baseline_cost
    ):
        return None
    return (kappa - baseline) / (cost - baseline_cost)


def _report(
    rubric: str, rows: Sequence[CalibrationRow], *, random_sample: bool, confidence: float
) -> _RubricReport:
    if len({row.id for row in rows}) != len(rows):
        raise CalibrationError("duplicate item and rubric")
    pairs = tuple((row.human, row.judge) for row in rows if row.human is not None)
    second = tuple(
        (row.human, row.second_human)
        for row in rows
        if row.human is not None and row.second_human is not None
    )
    judge_rate = mean(row.judge for row in rows)
    inference = _estimates(
        pairs,
        size=len(rows),
        judge_rate=judge_rate,
        random_sample=random_sample,
        confidence=confidence,
    )
    ensemble_count, ensembles = _ensemble(rows)
    return _RubricReport(
        rubric=rubric,
        count=len(rows),
        labeled=len(pairs),
        confusion=_confusion(pairs),
        bias=mean(int(judge) - int(human) for human, judge in pairs) if pairs else None,
        judge_rate=judge_rate,
        human_rate=mean(human for human, _ in pairs) if pairs else None,
        corrected_rate=inference.corrected_rate,
        corrected_interval=inference.corrected_interval,
        human_interval=inference.human_interval,
        probes=tuple(_probe(rows, name, value) for name, value in _PROBES),
        second_rater_count=len(second),
        second_rater_kappa=_confusion(second).kappa,
        ensemble_count=ensemble_count,
        ensembles=ensembles,
    )


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _interval(value: _Interval | None) -> str:
    return "unavailable" if value is None else f"[{value[0]:.4f}, {value[1]:.4f}]"


def _render(report: _RubricReport) -> tuple[str, ...]:
    counts = report.confusion
    lines = [
        f"{report.rubric}: {report.labeled}/{report.count} human-labeled",
        f"  kappa: {_number(report.kappa)}",
        f"  confusion: true positive={counts.true_positive}, true negative={counts.true_negative}, "
        f"false positive={counts.false_positive}, false negative={counts.false_negative}",
        f"  label bias (judge minus human): {_number(report.bias)}",
        f"  judge rate: {_number(report.judge_rate)}",
        f"  human sample rate: {_number(report.human_rate)}",
        f"  human-only interval: {_interval(report.human_interval)}",
        f"  PPI corrected rate: {_number(report.corrected_rate)} "
        f"{_interval(report.corrected_interval)}",
        f"  second rater kappa: {_number(report.second_rater_kappa)} "
        f"({report.second_rater_count} matched)",
    ]
    for probe in report.probes:
        detail = (
            f"{probe.flips}/{probe.count} flips, acceptance delta {_number(probe.acceptance_delta)}"
            if probe.count
            else "not measured"
        )
        lines.append(f"  {probe.name}: {detail}")
    if report.ensembles:
        lines.append(f"  Judge ensemble ({report.ensemble_count} matched)")
        for ensemble in report.ensembles:
            lines.append(
                f"    {ensemble.judges} judge(s): kappa {_number(ensemble.kappa)}, "
                f"total USD {_number(ensemble.total_cost_usd)}, "
                f"kappa gain per added USD {_number(ensemble.gain_per_added_usd)}"
            )
    else:
        lines.append("  Judge ensemble: not measured")
    return tuple(lines)


def calibrate_command(
    labels: Annotated[
        Path, typer.Argument(help="JSONL with frozen judge decisions and human labels.")
    ],
    random_sample: Annotated[
        bool,
        typer.Option("--random-sample", help="Assert uniform random review within each rubric."),
    ] = False,
    confidence: Annotated[float, typer.Option("--confidence")] = 0.95,
) -> None:
    """Report human agreement, paired bias probes and corrected finite-set acceptance."""
    try:
        reports = calibrate(
            read_calibration(labels), random_sample=random_sample, confidence=confidence
        )
    except CalibrationError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(3) from error
    typer.echo(
        f"Per-rubric intervals use {confidence:g} confidence with a declared random sample. "
        "A fully reviewed set is exact."
    )
    if not random_sample:
        typer.echo("Partial-label intervals withheld unless --random-sample is declared.")
    for report in reports:
        typer.echo("\n".join(_render(report)))
