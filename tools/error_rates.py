"""Measure how often evalgate's statistics are wrong at the margin, and publish the numbers.

Writes `docs/error-rates.md`. With `--check` it regenerates the file in memory and fails if the
committed copy differs, which is how CI keeps the published numbers equal to what the current
code produces. Everything is seeded; the binary figures are exact.

What is measured, and why at the margin: a slice whose true difference is exactly `-tolerance`
is the hardest case for both errors. Calling it non-inferior is a false pass; calling it
regressed is a false fail. Each should happen at most 2.5% of the time at 95% confidence.

- Binary scorers, one slice: exact enumeration over every outcome table, for Tango's interval
  (what evalgate uses), the conditional interval it replaced, and Newcombe's hybrid interval.
- Continuous scorers, one slice: Monte Carlo through `paired_continuous`, on differences where
  only some items move, which is what real evaluation data looks like.
- Continuous scorers, the threshold: how often a plain BCa bootstrap (scipy's) is wrong, by how
  many items moved. This is the evidence for `MIN_MOVED_FOR_BOOTSTRAP`.
- Many slices: how often a whole check falsely passes or falsely fails, with and without the
  correction across slices.

Deliberate boundary: the two methods evalgate no longer uses, and the plain bootstrap behind the
threshold, are computed here with scipy as references. Everything attributed to evalgate itself
goes through `evalgate.stats`, so the numbers describe the shipped code and nothing else.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.stats

from evalgate.stats import (
    MIN_MOVED_FOR_BOOTSTRAP,
    Correction,
    PairedResult,
    Verdict,
    family_verdicts,
    judge,
    paired_binary,
    paired_continuous,
    tango_statistic,
)

__all__ = [
    "BinaryCase",
    "Grid",
    "binary_errors",
    "conditional_decision",
    "main",
    "newcombe_interval",
    "render",
    "tango_decision",
]

# scipy ships no type information; one `Any` alias keeps that to a single line.
_scipy_stats: Any = scipy.stats

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "error-rates.md"

CONFIDENCE = 0.95
TAIL = (1.0 - CONFIDENCE) / 2.0
Z = NormalDist().inv_cdf(1.0 - TAIL)

#: Below this a cell of the outcome table is skipped. Summed over every skipped cell, the
#: missing probability is far below the 0.1% the tables are printed to.
_NEGLIGIBLE = 1e-15

Decision = Callable[[int, int, int, float], tuple[bool, bool]]
"""`(favour_candidate, favour_baseline, n, tolerance) -> (passes, fails)` for one slice."""


@dataclass(frozen=True)
class BinaryCase:
    """One cell of the binary grid: a slice size, a tolerance, and how often items disagree.

    The true difference is fixed at `-tolerance`, so `discordance` must be at least the
    tolerance. At `discordance == tolerance` every disagreement favours the baseline: a change
    that breaks items and fixes none.
    """

    n: int
    tolerance: float
    discordance: float

    @property
    def favour_candidate(self) -> float:
        """Probability an item is fixed by the candidate."""
        return (self.discordance - self.tolerance) / 2.0

    @property
    def favour_baseline(self) -> float:
        """Probability an item is broken by the candidate."""
        return (self.discordance + self.tolerance) / 2.0


@dataclass(frozen=True)
class Grid:
    """How much to measure. The defaults produce the published file."""

    binary_sizes: tuple[int, ...] = (10, 20, 40, 100, 200, 500)
    tolerances: tuple[float, ...] = (0.02, 0.05, 0.10)
    discordances: tuple[float, ...] = (0.2, 0.4)
    newcombe_reps: int = 20_000
    continuous_sizes: tuple[int, ...] = (10, 20, 40, 100)
    moved_shares: tuple[float, ...] = (0.2, 1.0)
    spreads: tuple[float, ...] = (0.1, 0.3)
    continuous_reps: int = 1_000
    resamples: int = 2_000
    threshold_reps: int = 1_000
    family_sizes: tuple[int, ...] = (1, 4, 10)
    family_reps: int = 2_000


def binary_cases(grid: Grid) -> Iterator[BinaryCase]:
    """Every binary cell on the grid.

    Each size and tolerance gets the grid's discordance rates plus two more: the tolerance
    itself and twice it. Those one-sided cases are where conditioning on the observed
    discordance goes wrong.
    """
    for n in grid.binary_sizes:
        for tolerance in grid.tolerances:
            rates = sorted({tolerance, 2.0 * tolerance, *grid.discordances})
            for discordance in rates:
                yield BinaryCase(n, tolerance, discordance)


# --------------------------------------------------------------------------- binary


def tango_decision(b: int, c: int, n: int, tolerance: float) -> tuple[bool, bool]:
    """Binary decision as evalgate makes it, from its own score statistic at the margin.

    Reading the statistic at `-tolerance` is the same decision as reading the interval: the
    interval is the inversion of this test, and the oracle checks the two agree.
    """
    z = tango_statistic(b, c, n, -tolerance)
    return z > Z, z < -Z


@cache
def _share_bounds(successes: int, trials: int) -> tuple[float, float]:
    """Clopper-Pearson bounds on the share of discordant items favouring the candidate."""
    low = 0.0 if successes == 0 else _scipy_stats.beta.ppf(TAIL, successes, trials - successes + 1)
    high = (
        1.0
        if successes == trials
        else _scipy_stats.beta.ppf(1.0 - TAIL, successes + 1, trials - successes)
    )
    return float(low), float(high)


def conditional_decision(b: int, c: int, n: int, tolerance: float) -> tuple[bool, bool]:
    """Binary decision by the interval evalgate used before.

    Clopper-Pearson on the discordant share, scaled by the observed discordance as if that rate
    were known.
    """
    discordant = b + c
    if discordant == 0:
        bound = 1.0 - TAIL ** (1.0 / n)
        return bound < tolerance, False
    share_low, share_high = _share_bounds(b, discordant)
    scale = discordant / n
    return (2.0 * share_low - 1.0) * scale > -tolerance, (
        2.0 * share_high - 1.0
    ) * scale < -tolerance


def binary_errors(case: BinaryCase, decide: Decision) -> tuple[float, float]:
    """Exact false-pass and false-fail rates of one decision rule on one cell.

    Sums the trinomial probability of every (fixed, broken, unchanged) table, so there is no
    simulation error.
    """
    n = case.n
    log_fixed = _log(case.favour_candidate)
    log_broken = _log(case.favour_baseline)
    log_same = _log(1.0 - case.discordance)
    false_pass = 0.0
    false_fail = 0.0
    for fixed in range(n + 1):
        if fixed and log_fixed == -math.inf:
            break
        for broken in range(n - fixed + 1):
            same = n - fixed - broken
            log_p = (
                math.lgamma(n + 1)
                - math.lgamma(fixed + 1)
                - math.lgamma(broken + 1)
                - math.lgamma(same + 1)
                + (fixed * log_fixed if fixed else 0.0)
                + (broken * log_broken if broken else 0.0)
                + (same * log_same if same else 0.0)
            )
            probability = math.exp(log_p)
            if probability < _NEGLIGIBLE:
                continue
            passes, fails = decide(fixed, broken, n, case.tolerance)
            false_pass += probability * passes
            false_fail += probability * fails
    return false_pass, false_fail


def _log(probability: float) -> float:
    return math.log(probability) if probability > 0.0 else -math.inf


def newcombe_interval(both: int, fixed: int, broken: int, neither: int) -> tuple[float, float]:
    """Newcombe's hybrid score interval for a paired difference (his method 10).

    `both` and `neither` are items both versions got right and wrong; `fixed` and `broken` are
    the discordant ones. Unlike the other two it needs the full table, which is why it is
    measured by simulation.
    """
    n = both + fixed + broken + neither
    candidate = (both + fixed) / n
    baseline = (both + broken) / n
    cand_low, cand_high = _wilson(both + fixed, n)
    base_low, base_high = _wilson(both + broken, n)
    margins = (both + fixed) * (broken + neither) * (both + broken) * (fixed + neither)
    phi = 0.0 if margins == 0 else (both * neither - fixed * broken) / math.sqrt(margins)
    below = math.sqrt(
        max(
            (candidate - cand_low) ** 2
            - 2.0 * phi * (candidate - cand_low) * (base_high - baseline)
            + (base_high - baseline) ** 2,
            0.0,
        )
    )
    above = math.sqrt(
        max(
            (cand_high - candidate) ** 2
            - 2.0 * phi * (cand_high - candidate) * (baseline - base_low)
            + (baseline - base_low) ** 2,
            0.0,
        )
    )
    difference = candidate - baseline
    return max(difference - below, -1.0), min(difference + above, 1.0)


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    share = successes / trials
    z2 = Z * Z
    centre = (share + z2 / (2 * trials)) / (1 + z2 / trials)
    half = (
        Z * math.sqrt(share * (1 - share) / trials + z2 / (4 * trials * trials)) / (1 + z2 / trials)
    )
    return centre - half, centre + half


def newcombe_errors(case: BinaryCase, reps: int, seed: int) -> tuple[float, float]:
    """Simulated false-pass and false-fail rates of Newcombe's interval on one cell.

    The concordant items are split with an 80% baseline accuracy; the discordant probabilities
    are those of the cell.
    """
    both = 0.8 - case.favour_baseline
    probabilities = [
        both,
        case.favour_candidate,
        case.favour_baseline,
        1.0 - case.discordance - both,
    ]
    generator = np.random.default_rng(seed)
    tables: npt.NDArray[np.int64] = generator.multinomial(case.n, probabilities, size=reps)
    seen: dict[tuple[int, int, int, int], tuple[bool, bool]] = {}
    false_pass = 0
    false_fail = 0
    for row in tables:
        key = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        if key not in seen:
            low, high = newcombe_interval(*key)
            seen[key] = (low > -case.tolerance, high < -case.tolerance)
        passes, fails = seen[key]
        false_pass += passes
        false_fail += fails
    return false_pass / reps, false_fail / reps


# --------------------------------------------------------------------------- continuous


def _differences(
    generator: np.random.Generator, n: int, moved: float, mean: float, spread: float
) -> npt.NDArray[np.float64]:
    """Paired differences where a share `moved` of items change, by `mean / moved` on average.

    Moved items' differences are uniform with standard deviation `spread`, so the true mean
    difference is exactly `mean` and no clipping distorts it.
    """
    half_width = spread * math.sqrt(3.0)
    centre = mean / moved
    moves: npt.NDArray[np.bool_] = generator.random(n) < moved
    shifts: npt.NDArray[np.float64] = generator.uniform(centre - half_width, centre + half_width, n)
    # Multiplying by the mask leaves unmoved items at exactly zero.
    return shifts * moves


def _scores(
    generator: np.random.Generator, differences: npt.NDArray[np.float64]
) -> tuple[list[float], list[float]]:
    """Baseline and candidate scores in [0, 1] that differ by exactly `differences`."""
    floor = np.maximum(0.0, -differences)
    ceiling = np.minimum(1.0, 1.0 - differences)
    baseline = floor + generator.random(differences.size) * (ceiling - floor)
    # Exact by construction; the clip only absorbs rounding at the ends of the range.
    candidate: npt.NDArray[np.float64] = np.minimum(np.maximum(baseline + differences, 0.0), 1.0)
    return [float(value) for value in baseline], [float(value) for value in candidate]


@dataclass(frozen=True)
class ContinuousRow:
    """One continuous cell: errors at the margin and power when nothing changed on average."""

    n: int
    moved: float
    spread: float
    false_pass: float
    false_fail: float
    power: float


def continuous_rows(grid: Grid, tolerance: float, seed: int) -> list[ContinuousRow]:
    """Monte Carlo through `paired_continuous`, the shipped code."""
    generator = np.random.default_rng(seed)
    rows: list[ContinuousRow] = []
    for n in grid.continuous_sizes:
        for moved in grid.moved_shares:
            for spread in grid.spreads:
                passes = fails = powered = 0
                for _ in range(grid.continuous_reps):
                    worse = paired_continuous(
                        *_scores(generator, _differences(generator, n, moved, -tolerance, spread)),
                        tolerance=tolerance,
                        resamples=grid.resamples,
                    )
                    verdict = judge(worse.interval, tolerance)
                    passes += verdict is Verdict.NON_INFERIOR
                    fails += verdict is Verdict.REGRESSED
                    same = paired_continuous(
                        *_scores(generator, _differences(generator, n, moved, 0.0, spread)),
                        tolerance=tolerance,
                        resamples=grid.resamples,
                    )
                    powered += judge(same.interval, tolerance) is Verdict.NON_INFERIOR
                reps = grid.continuous_reps
                rows.append(
                    ContinuousRow(n, moved, spread, passes / reps, fails / reps, powered / reps)
                )
    return rows


def threshold_rows(grid: Grid, tolerance: float, seed: int) -> list[tuple[str, float, int]]:
    """How often a plain BCa bootstrap passes a slice at the margin, by moved-item count.

    Uses scipy's BCa, as a reference independent of evalgate's. With no moved item at all
    there is nothing to resample, and the interval evalgate used to report was `[0, 0]`, which
    passes; that row is counted as such.
    """
    generator = np.random.default_rng(seed)
    passes: dict[int, int] = {}
    totals: dict[int, int] = {}
    for n in (20, 40, 100):
        for moved in (0.1, 0.2, 0.35):
            for spread in (0.1, 0.3):
                for _ in range(grid.threshold_reps):
                    differences = _differences(generator, n, moved, -tolerance, spread)
                    changed: npt.NDArray[np.bool_] = differences != 0.0
                    count = int(changed.sum())
                    passed = True if count == 0 else _bca_low(differences) > -tolerance
                    bucket = min(count, 12)
                    passes[bucket] = passes.get(bucket, 0) + passed
                    totals[bucket] = totals.get(bucket, 0) + 1
    labels = {12: "12 or more"}
    return [
        (labels.get(bucket, str(bucket)), passes[bucket] / totals[bucket], totals[bucket])
        for bucket in sorted(totals)
    ]


def _bca_low(differences: npt.NDArray[np.float64]) -> float:
    result = _scipy_stats.bootstrap(
        (differences,),
        np.mean,
        confidence_level=CONFIDENCE,
        n_resamples=1_000,
        method="BCa",
        random_state=np.random.default_rng(1),
    )
    low = float(result.confidence_interval.low)
    # scipy returns nan when every resample is identical; that is no evidence of safety.
    return -math.inf if math.isnan(low) else low


# --------------------------------------------------------------------------- many slices


@dataclass(frozen=True)
class FamilyRow:
    """Check-level error rates for one number of gate slices."""

    slices: int
    false_pass: float
    false_fail_uncorrected: float
    false_fail_holm: float


def _binary_slice(
    generator: np.random.Generator, n: int, difference: float, discordance: float, tolerance: float
) -> PairedResult:
    fixed = (discordance + difference) / 2.0
    broken = (discordance - difference) / 2.0
    draw = generator.random(n)
    agree_right = generator.random(n) < 0.5
    baseline: list[bool] = []
    candidate: list[bool] = []
    for value, right in zip(draw, agree_right, strict=True):
        if value < broken:
            baseline.append(True)
            candidate.append(False)
        elif value < broken + fixed:
            baseline.append(False)
            candidate.append(True)
        else:
            baseline.append(bool(right))
            candidate.append(bool(right))
    return paired_binary(baseline, candidate, tolerance=tolerance)


def family_rows(grid: Grid, tolerance: float, seed: int) -> list[FamilyRow]:
    """Simulated check-level errors through `paired_binary` and `family_verdicts`.

    False pass: one slice is truly worse by exactly the tolerance, the rest unchanged; the
    check passes only if every slice is non-inferior. False fail: every slice is exactly at the
    tolerance, so none truly regressed beyond it; the check fails if any is called regressed.
    """
    generator = np.random.default_rng(seed)
    n = 100
    discordance = 0.2
    rows: list[FamilyRow] = []
    for count in grid.family_sizes:
        passes = fails_alone = fails_holm = 0
        for _ in range(grid.family_reps):
            worse = [_binary_slice(generator, n, -tolerance, discordance, tolerance)]
            # The other slices are clearly safe -- unchanged, with few disagreements -- so the
            # check can pass, and only the slice at the margin can make that pass false.
            safe = [_binary_slice(generator, n, 0.0, 0.02, tolerance) for _ in range(count - 1)]
            passes += all(
                verdict is Verdict.NON_INFERIOR
                for verdict in family_verdicts([*worse, *safe], Correction.HOLM)
            )
            margin = [
                _binary_slice(generator, n, -tolerance, discordance, tolerance)
                for _ in range(count)
            ]
            fails_alone += any(
                judge(result.interval, tolerance) is Verdict.REGRESSED for result in margin
            )
            fails_holm += Verdict.REGRESSED in family_verdicts(margin, Correction.HOLM)
        reps = grid.family_reps
        rows.append(FamilyRow(count, passes / reps, fails_alone / reps, fails_holm / reps))
    return rows


# --------------------------------------------------------------------------- report


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(grid: Grid) -> str:
    """Measure everything on `grid` and return the markdown document."""
    lines = [
        "# Measured error rates",
        "",
        "Written by `tools/error_rates.py`; do not edit by hand. CI regenerates this file and",
        "fails if it differs, so these are the numbers the current code produces.",
        "",
        "Every case puts the true difference **exactly at the tolerance**, the hardest point for",
        "both errors. A *false pass* calls such a slice non-inferior; a *false fail* calls it",
        "regressed. At 95% confidence each should happen at most **2.5%** of the time. The",
        "method is in [statistics.md](statistics.md).",
        "",
        "## Binary scorers, one slice",
        "",
        "Exact: every possible outcome table is weighed by its probability, so there is no",
        "simulation error, except in the Newcombe column, which needs the full table and is",
        f"simulated ({grid.newcombe_reps:,} tables per row, 80% baseline accuracy).",
        "*Discordance* is the share of items the two versions disagree on; when it equals the",
        "tolerance, the change breaks items and fixes none.",
        "",
        "| n | tolerance | discordance | before: false pass | Newcombe: false pass "
        "| **Tango: false pass** | **Tango: false fail** |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ]
    worst = {"before": 0.0, "newcombe": 0.0, "pass": 0.0, "fail": 0.0}
    for index, case in enumerate(binary_cases(grid)):
        before, _ = binary_errors(case, conditional_decision)
        newcombe, _ = newcombe_errors(case, grid.newcombe_reps, seed=index)
        passes, fails = binary_errors(case, tango_decision)
        worst = {
            "before": max(worst["before"], before),
            "newcombe": max(worst["newcombe"], newcombe),
            "pass": max(worst["pass"], passes),
            "fail": max(worst["fail"], fails),
        }
        lines.append(
            f"| {case.n} | {case.tolerance:.2f} | {case.discordance:.2f} | {_pct(before)} "
            f"| {_pct(newcombe)} | {_pct(passes)} | {_pct(fails)} |"
        )
    lines += [
        "",
        f"Worst false pass: **{_pct(worst['before'])}** for the conditional interval evalgate "
        f"used before, {_pct(worst['newcombe'])} for Newcombe's, **{_pct(worst['pass'])}** for "
        f"Tango's. Tango's worst false fail is {_pct(worst['fail'])}.",
        "",
    ]

    tolerance = 0.05
    lines += [
        "## Continuous scorers, one slice",
        "",
        f"Simulated through `paired_continuous` at a tolerance of {tolerance:.2f}, "
        f"{grid.continuous_reps:,} slices per row and {grid.resamples:,} resamples per interval "
        "(the shipped default is 10,000, which only steadies the bounds). A share *moved* of",
        "items change; the rest score identically, which is what most evaluation data looks",
        "like. *Spread* is the standard deviation of a moved item's change. *Power* is how often",
        "a slice with no change on average is certified safe.",
        "",
        "| n | moved | spread | false pass | false fail | power |",
        "|--:|--:|--:|--:|--:|--:|",
    ]
    for row in continuous_rows(grid, tolerance, seed=101):
        lines.append(
            f"| {row.n} | {row.moved:.1f} | {row.spread:.1f} | {_pct(row.false_pass)} "
            f"| {_pct(row.false_fail)} | {_pct(row.power)} |"
        )
    lines += [
        "",
        f"### Why a slice needs {MIN_MOVED_FOR_BOOTSTRAP} moved items before it is bootstrapped",
        "",
        "A plain BCa bootstrap (scipy's), on the same kind of data at the margin, grouped by how",
        "many items actually moved. Below the threshold evalgate uses the count bound instead,",
        "which holds at any size.",
        "",
        "| items moved | bootstrap false pass | slices |",
        "|--:|--:|--:|",
    ]
    for label, rate, total in threshold_rows(grid, tolerance, seed=202):
        lines.append(f"| {label} | {_pct(rate)} | {total:,} |")
    lines += [
        "",
        "## Many slices",
        "",
        f"Simulated through `paired_binary` and `family_verdicts`: {grid.family_reps:,} checks per",
        "row, 100 items per slice, tolerance 0.05. *False pass*: one slice is truly worse by the",
        "tolerance (20% discordance) and the rest are unchanged (2% discordance, so they can",
        "pass), yet the check passes. *False fail*: every slice is exactly at the tolerance (20%",
        "discordance), yet the check fails.",
        "",
        "| gate slices | false pass | false fail, no correction | **false fail, Holm** |",
        "|--:|--:|--:|--:|",
    ]
    for row in family_rows(grid, tolerance, seed=303):
        lines.append(
            f"| {row.slices} | {_pct(row.false_pass)} | {_pct(row.false_fail_uncorrected)} "
            f"| {_pct(row.false_fail_holm)} |"
        )
    lines += [
        "",
        "The false-pass rate needs no correction: a check passes only when every slice does.",
        "Without a correction the false-fail rate grows with the number of slices; Holm holds it",
        "at or below one slice's.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, grid: Grid | None = None) -> int:
    """Write the document, or with `--check` confirm the committed copy is current.

    `grid` exists for tests, which cannot afford the published one.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output: Path = args.output

    document = render(grid or Grid())
    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != document:
            print(f"{output} is out of date; run: uv run python tools/error_rates.py")
            return 1
        print(f"{output} is current")
        return 0
    output.write_text(document, encoding="utf-8", newline="\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
