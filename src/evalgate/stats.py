"""Paired statistics for the non-inferiority gate.

Owning artifact: `docs/statistics.md`. That document states the mathematics; this module
implements it. Where the two disagree, the document is right.

The gate asks "is the candidate at most `delta` worse than the baseline", not "is it different".
A significance test on a forty-item slice passes almost everything, so its greenness means
nothing. Everything here therefore produces an interval on `candidate - baseline` and two
one-sided p-values at the tolerance, and the verdict reads the interval.

Every method here was chosen by measuring how often it is wrong at the margin, not by argument.
`tools/error_rates.py` re-measures them and writes `docs/error-rates.md`.

Deliberate boundary: this module imports neither scipy nor statsmodels, at any point. Those two
are test-only dependencies and are the independent oracle in `tests/oracle/`. An implementation
that called scipy would make "agrees with scipy" a statement about nothing. stdlib `math` and
`statistics` carry the exact binomial work and the score test; numpy is here for the bootstrap's
array arithmetic only, which is not a statistical claim.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

__all__ = [
    "MIN_MOVED_FOR_BOOTSTRAP",
    "Correction",
    "Interval",
    "PairedResult",
    "Verdict",
    "benjamini_hochberg",
    "clopper_pearson",
    "family_verdicts",
    "holm",
    "jackknife_acceleration",
    "judge",
    "paired_binary",
    "paired_continuous",
    "tango_interval",
    "tango_statistic",
]

_NORMAL = NormalDist()

#: Bisection stops here. Two to the minus sixty is already finer than a float can express, so
#: further iterations buy nothing; the cap exists so a pathological input cannot spin forever.
_BISECTION_STEPS = 200

MIN_MOVED_FOR_BOOTSTRAP = 8
"""Fewest moved items for which a continuous slice is bootstrapped.

Measured, not derived. Below this the bootstrap rarely resamples the items that moved, and at a
true difference of exactly the tolerance it called the slice non-inferior 3-20% of the time
against a nominal 2.5%. From eight up it held the nominal rate. `tools/error_rates.py`
re-measures it.
"""


class Verdict(StrEnum):
    """The three answers the gate can give about one slice.

    `INCONCLUSIVE` is a real answer and never collapses into `NON_INFERIOR`. A slice too thin to
    detect the regression it was asked about is a gap in coverage, and reporting that as a pass
    is the precise failure this project exists to prevent.
    """

    NON_INFERIOR = "non_inferior"
    """The data rule out a regression worse than the tolerance."""

    REGRESSED = "regressed"
    """The data rule out the candidate being within the tolerance."""

    INCONCLUSIVE = "inconclusive"
    """The interval straddles the tolerance. Nothing is ruled out, in either direction."""


class Correction(StrEnum):
    """How a family of slices is corrected for testing many at once.

    Only the false-fail side is corrected. A check passes only when every gate slice passes, so
    its false-pass rate is already at most one slice's; see `docs/statistics.md` section 5.
    """

    HOLM = "holm"
    """Family-wise error control, for gate slices: one false failure is the thing to avoid."""

    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    """False discovery control, for watch slices: a false flag costs a reader one line."""


@dataclass(frozen=True)
class Interval:
    """An estimate with its uncertainty, on the scale of `candidate - baseline`.

    Attributes:
        point: The observed difference. Negative means the candidate is worse.
        low: Lower bound. This is the number the gate's verdict reads.
        high: Upper bound.
        confidence: The nominal confidence level, for example 0.95.
    """

    point: float
    low: float
    high: float
    confidence: float

    @property
    def width(self) -> float:
        """How wide the interval is, which is how little the slice knows."""
        return self.high - self.low


@dataclass(frozen=True)
class PairedResult:
    """One slice's paired comparison at its tolerance, with the method that produced it.

    Attributes:
        n: Number of paired items.
        interval: The difference and its bounds.
        method: Which estimator ran: `tango_score`, `bootstrap_bca`, `count_bound` or
            `constant_shift`.
        tolerance: The tolerance the p-values were computed at.
        p_non_inferior: One-sided p-value against "worse by the tolerance or more".
        p_regressed: One-sided p-value against "within the tolerance". This is what the
            correction across slices adjusts.
    """

    n: int
    interval: Interval
    method: str
    tolerance: float
    p_non_inferior: float
    p_regressed: float


# ---------------------------------------------------------------------------
# Exact binomial machinery
# ---------------------------------------------------------------------------


def _log_binomial_pmf(successes: int, trials: int, probability: float) -> float:
    """Log of the binomial probability mass, computed with log-gamma.

    Working in logs keeps the factorials from overflowing at trial counts where a direct
    `math.comb` would produce an integer with thousands of digits.
    """
    if probability <= 0.0:
        return 0.0 if successes == 0 else -math.inf
    if probability >= 1.0:
        return 0.0 if successes == trials else -math.inf
    return (
        math.lgamma(trials + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(trials - successes + 1)
        + successes * math.log(probability)
        + (trials - successes) * math.log1p(-probability)
    )


def _binomial_at_most(successes: int, trials: int, probability: float) -> float:
    """`P(X <= successes)` for `X ~ Binomial(trials, probability)`.

    Summed directly from the mass function rather than as `1 - P(X > k)`: the subtraction
    loses every significant digit in the far tail, which is exactly where Clopper-Pearson
    bounds are determined.
    """
    if successes < 0:
        return 0.0
    if successes >= trials:
        return 1.0
    return math.fsum(
        math.exp(_log_binomial_pmf(i, trials, probability)) for i in range(successes + 1)
    )


def _binomial_at_least(successes: int, trials: int, probability: float) -> float:
    """`P(X >= successes)` for `X ~ Binomial(trials, probability)`, summed directly."""
    if successes <= 0:
        return 1.0
    if successes > trials:
        return 0.0
    return math.fsum(
        math.exp(_log_binomial_pmf(i, trials, probability)) for i in range(successes, trials + 1)
    )


def _bisect(
    low: float, high: float, target: float, *, increasing: bool, tail: Callable[[float], float]
) -> float:
    """Solve a monotone equation `tail(x) = target` on `[low, high]` by bisection.

    Args:
        low: Lower end of the bracket.
        high: Upper end of the bracket.
        target: The value to hit.
        increasing: True when `tail` rises with its argument.
        tail: The monotone function.

    Returns:
        The argument at which `tail` crosses `target`.
    """
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2.0
        if middle <= low or middle >= high:
            break
        if (tail(middle) < target) == increasing:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion.

    Exact rather than Wald because the gate lives on thin slices, and the Wald interval's
    coverage falls well below its nominal level exactly there. Here it bounds the share of
    continuous items that moved, when too few moved for a bootstrap to see them.

    The bounds are defined by the binomial tails and found by bisection on them. That is the
    same quantity a Beta quantile would give, reached without a special-function library.

    Args:
        successes: Observed successes, `0 <= successes <= trials`.
        trials: Number of trials, which must be positive.
        confidence: Nominal confidence level, strictly between 0 and 1.

    Returns:
        `(low, high)`. `low` is 0.0 when `successes` is 0; `high` is 1.0 when all trials succeed.

    Raises:
        ValueError: If the counts or the confidence level are out of range.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must be in [0, {trials}], got {successes}")
    _check_confidence(confidence)

    target = (1.0 - confidence) / 2.0
    observed = successes / trials

    low = 0.0
    if successes > 0:
        low = _bisect(
            0.0,
            observed,
            target,
            increasing=True,
            tail=lambda p: _binomial_at_least(successes, trials, p),
        )

    high = 1.0
    if successes < trials:
        high = _bisect(
            observed,
            1.0,
            target,
            increasing=False,
            tail=lambda p: _binomial_at_most(successes, trials, p),
        )

    return low, high


# ---------------------------------------------------------------------------
# Tango's score test, for binary scorers
# ---------------------------------------------------------------------------


def tango_statistic(
    favour_candidate: int, favour_baseline: int, n: int, difference: float
) -> float:
    """Tango's score statistic for a hypothesised paired difference.

    The variance is taken under the hypothesis being tested, using the restricted maximum
    likelihood estimate of how often an item favours the baseline. That is what keeps the test
    honest when every disagreement goes one way: an interval that scales by the observed
    disagreement rate instead treats that rate as known, and at the margin it passed a slice
    that was truly worse up to 48% of the time.

    Args:
        favour_candidate: Items the baseline got wrong and the candidate got right.
        favour_baseline: Items the baseline got right and the candidate got wrong.
        n: All paired items, agreeing ones included.
        difference: The hypothesised `candidate - baseline`, in `[-1, 1]`.

    Returns:
        The statistic. Large positive values say the true difference is above `difference`.
        Infinite where the variance under the hypothesis is zero and the data disagree with it.

    Raises:
        ValueError: If the counts are inconsistent or `difference` is outside `[-1, 1]`.
    """
    _check_counts(favour_candidate, favour_baseline, n)
    if not -1.0 <= difference <= 1.0:
        raise ValueError(f"difference must be in [-1, 1], got {difference}")
    return _tango(favour_candidate, favour_baseline, n, difference)


def _tango(b: int, c: int, n: int, d: float) -> float:
    """`tango_statistic` without the argument checks, for the root-finders' inner loops."""
    w = -b - c + (2 * n - b + c) * d
    # The restricted estimate of P(favours baseline), the positive root of a quadratic. The
    # discriminant is never negative in exact arithmetic; the clamp absorbs rounding near d = 1.
    restricted = (math.sqrt(max(w * w + 8.0 * n * c * d * (1.0 - d), 0.0)) - w) / (4.0 * n)
    variance = n * (2.0 * restricted + d * (1.0 - d))
    numerator = b - c - n * d
    if variance <= 0.0:
        if numerator == 0.0:
            return 0.0
        return math.copysign(math.inf, numerator)
    return numerator / math.sqrt(variance)


def tango_interval(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """The set of differences Tango's test does not reject, as `(low, high)`.

    The statistic falls as the hypothesised difference rises, so each bound is found by
    bisecting on its own side of the point estimate, where the statistic is zero.

    Args:
        favour_candidate: Items the baseline got wrong and the candidate got right.
        favour_baseline: Items the baseline got right and the candidate got wrong.
        n: All paired items.
        confidence: Nominal two-sided confidence level.

    Returns:
        `(low, high)`, within `[-1, 1]`. Never a single point: with no disagreements at all it
        is `+-z^2 / (n + z^2)`, because zero disagreements is not proof of equality.

    Raises:
        ValueError: If the counts or the confidence level are out of range.
    """
    _check_counts(favour_candidate, favour_baseline, n)
    _check_confidence(confidence)
    z = _NORMAL.inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    point = (favour_candidate - favour_baseline) / n

    def statistic(d: float) -> float:
        return _tango(favour_candidate, favour_baseline, n, d)

    low = -1.0
    if statistic(-1.0) > z:
        low = _bisect(-1.0, point, z, increasing=False, tail=statistic)
    high = 1.0
    if statistic(1.0) < -z:
        high = _bisect(point, 1.0, -z, increasing=False, tail=statistic)
    return low, high


def paired_binary(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    tolerance: float,
    confidence: float = 0.95,
) -> PairedResult:
    """Compare two versions on a binary scorer, item by item.

    Pairing is the point. Comparing two independent proportions discards the item-level
    correlation that makes a small slice readable at all, and inflates the variance for nothing.

    Args:
        baseline: Per-item outcomes for the baseline version.
        candidate: Per-item outcomes for the candidate, in the same item order.
        tolerance: How much worse the candidate may be, as a share of items.
        confidence: Nominal confidence level.

    Returns:
        The paired result, with `method` set to `tango_score`. Its interval and its p-values are
        the same test read two ways, so they always agree.

    Raises:
        ValueError: If the inputs differ in length or are empty, or the tolerance or confidence
            level is out of range.
    """
    n = _check_pair(baseline, candidate)
    _check_tolerance(tolerance)
    _check_confidence(confidence)

    favour_candidate = sum(1 for b, c in zip(baseline, candidate, strict=True) if not b and c)
    favour_baseline = sum(1 for b, c in zip(baseline, candidate, strict=True) if b and not c)
    low, high = tango_interval(favour_candidate, favour_baseline, n, confidence)
    at_margin = _tango(favour_candidate, favour_baseline, n, -tolerance)

    return PairedResult(
        n=n,
        interval=Interval(
            point=(favour_candidate - favour_baseline) / n,
            low=low,
            high=high,
            confidence=confidence,
        ),
        method="tango_score",
        tolerance=tolerance,
        p_non_inferior=1.0 - _phi(at_margin),
        p_regressed=_phi(at_margin),
    )


# ---------------------------------------------------------------------------
# Continuous scorers
# ---------------------------------------------------------------------------


def jackknife_acceleration(values: Sequence[float]) -> float:
    """Acceleration constant for a BCa interval on the mean, from leave-one-out estimates.

    This corrects for the estimator's variance changing with the true value. It is a
    deterministic function of the data: no resampling enters it, so it must agree with an
    independent computation exactly rather than approximately.

    Args:
        values: The paired differences.

    Returns:
        The acceleration. Zero when every value is identical, where the formula is otherwise
        0/0 -- and zero is the honest answer, since a sample with no spread gives no evidence
        of skew.

    Raises:
        ValueError: If fewer than two values are given, where leaving one out is meaningless.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"jackknife needs at least 2 values, got {n}")

    total = math.fsum(values)
    mean = total / n
    # Leaving item i out gives (total - v_i)/(n - 1), whose own mean is exactly the full mean,
    # so deviations can be taken against `mean` directly.
    deviations = [mean - (total - value) / (n - 1) for value in values]

    sum_squares = math.fsum(d * d for d in deviations)
    if sum_squares <= 0.0:
        return 0.0
    sum_cubes = math.fsum(d * d * d for d in deviations)
    return sum_cubes / (6.0 * sum_squares**1.5)


def paired_continuous(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    tolerance: float,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int | None = None,
) -> PairedResult:
    """Compare two versions on a continuous scorer in `[0, 1]`.

    Three regimes, chosen by what the data can support:

    - Fewer than `MIN_MOVED_FOR_BOOTSTRAP` items moved: the **count bound**. A bootstrap that
      rarely resamples the few items that moved reports far too narrow an interval -- with none
      moved, it reported `[0, 0]` and passed every such slice. The count bound uses only the
      fact that a difference cannot pass the score range, so it holds at every size.
    - Every item moved by the same amount: the **constant shift** bound, for the same reason.
    - Otherwise a **BCa bootstrap** on the paired differences. Bias-corrected and accelerated
      rather than plain percentile, because scorer distributions are bounded and skewed.

    With no seed, one is derived from the data rather than left to chance. A gate whose
    verdict can change between two runs of the same commit is the failure a CI check exists
    to prevent.

    Args:
        baseline: Per-item scores for the baseline version, each in `[0, 1]`.
        candidate: Per-item scores for the candidate, in the same item order.
        tolerance: How much worse the candidate may be, on the score scale.
        confidence: Nominal confidence level.
        resamples: Bootstrap resample count. More costs time and buys resolution in the tails.
        seed: Seed for the resampling. The same seed gives the same interval.

    Returns:
        The paired result; `method` names the regime that ran.

    Raises:
        ValueError: If the inputs differ in length, are empty, hold a single item, hold a score
            outside `[0, 1]`, or a parameter is out of range.
    """
    n = _check_pair(baseline, candidate)
    if n < 2:
        raise ValueError("an interval on one item is meaningless; need at least 2")
    _check_tolerance(tolerance)
    _check_confidence(confidence)
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    for side, scores in (("baseline", baseline), ("candidate", candidate)):
        for index, score in enumerate(scores):
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{side} score at index {index} is outside [0, 1]: {score}")

    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    observed = float(np.mean(differences))
    rose: npt.NDArray[np.bool_] = differences > 0.0
    fell: npt.NDArray[np.bool_] = differences < 0.0
    up = int(rose.sum())
    down = int(fell.sum())

    if up + down < MIN_MOVED_FOR_BOOTSTRAP:
        return _count_bound(n, up, down, observed, tolerance, confidence)

    # Compared against a tolerance, not exactly. A constant shift computed in floating point
    # leaves differences that disagree in the last bit, and bootstrapping that noise produces
    # an interval narrower than the arithmetic that built it.
    if float(np.ptp(differences)) <= 1e-12 * max(abs(observed), 1.0):
        return _constant_shift(n, observed, tolerance, confidence)

    return _bootstrap(differences, observed, tolerance, confidence, resamples, seed)


def _count_bound(
    n: int, up: int, down: int, observed: float, tolerance: float, confidence: float
) -> PairedResult:
    """The interval for a slice where too few items moved for a bootstrap to see them.

    A difference is never below -1, so the mean difference is never below minus the share of
    items that moved down, and that share has an exact Clopper-Pearson bound. The same holds
    upwards. Nothing here is approximate, which is the point on a slice this thin.
    """
    _, down_share = clopper_pearson(down, n, confidence)
    _, up_share = clopper_pearson(up, n, confidence)
    return PairedResult(
        n=n,
        interval=Interval(point=observed, low=-down_share, high=up_share, confidence=confidence),
        method="count_bound",
        tolerance=tolerance,
        # Inverting the Clopper-Pearson upper bound at the tolerance gives a binomial tail.
        p_non_inferior=_binomial_at_most(down, n, tolerance),
        # The upper bound is never below zero, so this slice can never be shown to regress.
        p_regressed=1.0,
    )


def _constant_shift(n: int, shift: float, tolerance: float, confidence: float) -> PairedResult:
    """The interval when every item moved by the same amount.

    Items the data never saw could differ from the shift by up to the score range. Bounding the
    share of such items with the zero-success Clopper-Pearson bound keeps the interval honest
    instead of collapsing it to a point.
    """
    unseen = 1.0 - ((1.0 - confidence) / 2.0) ** (1.0 / n)
    p_non_inferior = (
        1.0 if shift <= -tolerance else (1.0 - (shift + tolerance) / (1.0 + shift)) ** n
    )
    p_regressed = 1.0 if shift >= -tolerance else (1.0 - (-tolerance - shift) / (1.0 - shift)) ** n
    return PairedResult(
        n=n,
        interval=Interval(
            point=shift,
            low=shift - unseen * (1.0 + shift),
            high=shift + unseen * (1.0 - shift),
            confidence=confidence,
        ),
        method="constant_shift",
        tolerance=tolerance,
        p_non_inferior=p_non_inferior,
        p_regressed=p_regressed,
    )


def _bootstrap(
    differences: npt.NDArray[np.float64],
    observed: float,
    tolerance: float,
    confidence: float,
    resamples: int,
    seed: int | None,
) -> PairedResult:
    """BCa interval and margin p-values from one set of resampled means.

    The per-item differences are resampled, never the two arrays separately: resampling them
    independently would destroy the pairing that makes the comparison worth doing.
    """
    n = int(differences.size)
    generator = np.random.default_rng(seed if seed is not None else _seed_from(differences))
    draws = generator.integers(0, n, size=(resamples, n))
    resampled: npt.NDArray[np.float64] = differences[draws]
    means: npt.NDArray[np.float64] = resampled.mean(axis=1)

    bias = _NORMAL.inv_cdf(_mid_rank(means, observed))
    acceleration = jackknife_acceleration([float(value) for value in differences])

    alpha = 1.0 - confidence
    low_pct = _adjusted_percentile(bias, acceleration, _NORMAL.inv_cdf(alpha / 2.0))
    high_pct = _adjusted_percentile(bias, acceleration, _NORMAL.inv_cdf(1.0 - alpha / 2.0))

    ordered: npt.NDArray[np.float64] = means.copy()
    ordered.sort()

    # The BCa mapping run backwards: the one-sided level at which the lower bound would sit
    # exactly on the margin.
    w = _NORMAL.inv_cdf(_mid_rank(means, -tolerance)) - bias
    if 1.0 + acceleration * w > 0.0:
        p_non_inferior = _NORMAL.cdf(w / (1.0 + acceleration * w) - bias)
    else:
        # The margin lies beyond every percentile the mapping can reach.
        p_non_inferior = 0.0 if w < 0.0 else 1.0

    return PairedResult(
        n=n,
        interval=Interval(
            point=observed,
            low=_percentile(ordered, low_pct),
            high=_percentile(ordered, high_pct),
            confidence=confidence,
        ),
        method="bootstrap_bca",
        tolerance=tolerance,
        p_non_inferior=p_non_inferior,
        p_regressed=1.0 - p_non_inferior,
    )


def _mid_rank(means: npt.NDArray[np.float64], value: float) -> float:
    """Share of resample means below `value`, ties counted as half, clamped off 0 and 1.

    Ties count as half rather than as "strictly less than". On continuous data the two agree,
    because exact ties essentially never occur. They diverge systematically once the differences
    take few distinct values -- a scorer rounded to two decimals, or set F1 over small sets --
    and the strict form then biases every such slice, which no tolerance separates from a bug.

    The clamp to half a resample is the finest resolution the resampling can express; past it the
    normal quantile is infinite, and that would report the limit of the evidence as certainty.
    """
    count = int(means.size)
    fewer: npt.NDArray[np.bool_] = means < value
    tied: npt.NDArray[np.bool_] = means == value
    share = (int(fewer.sum()) + int(tied.sum()) / 2.0) / count
    resolution = 1.0 / (2.0 * count)
    return min(max(share, resolution), 1.0 - resolution)


def _seed_from(differences: npt.NDArray[np.float64]) -> int:
    """Derive a reproducible seed from the data itself.

    Used when the caller supplies none. Hashing the raw bytes of the differences means two
    runs over the same data resample identically, so a verdict cannot change between two
    builds of the same commit, while genuinely different data still gets its own stream.
    """
    digest = hashlib.blake2b(differences.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _percentile(ordered: npt.NDArray[np.float64], fraction: float) -> float:
    """Linearly interpolated percentile of an already sorted array.

    Written out rather than delegated, because the interpolation rule changes the reported
    bound and so belongs in the module whose numbers are being defended. This is the standard
    linear rule: position the fraction across `count - 1` intervals and interpolate between
    the two neighbouring order statistics.

    Args:
        ordered: Resample statistics, sorted ascending.
        fraction: Where to cut, in [0, 1]. Values outside are clamped.

    Returns:
        The interpolated value at that fraction.
    """
    count = int(ordered.size)
    if count == 1:
        return float(ordered[0])
    position = min(max(fraction, 0.0), 1.0) * (count - 1)
    lower_index = math.floor(position)
    upper_index = min(lower_index + 1, count - 1)
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return lower + (position - lower_index) * (upper - lower)


def _adjusted_percentile(bias: float, acceleration: float, z: float) -> float:
    """Map a normal quantile to a BCa-adjusted percentile of the resample distribution."""
    shifted = bias + z
    denominator = 1.0 - acceleration * shifted
    if denominator == 0.0:
        # The adjustment is undefined at this point; falling back to the uncorrected percentile
        # is the conservative choice, and it keeps the function total.
        return _NORMAL.cdf(shifted)
    return _NORMAL.cdf(bias + shifted / denominator)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_pair(baseline: Sequence[object], candidate: Sequence[object]) -> int:
    """Validate that two per-item sequences can be paired, returning their length.

    Raises:
        ValueError: If the lengths differ or the sequences are empty.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired inputs must be the same length, got {len(baseline)} and {len(candidate)}"
        )
    if len(baseline) == 0:
        raise ValueError("cannot compare empty sequences; there is no estimate to report")
    return len(baseline)


def _check_counts(favour_candidate: int, favour_baseline: int, n: int) -> None:
    """Validate discordant counts against the number of paired items.

    Raises:
        ValueError: If `n` is not positive, a count is negative, or the counts exceed `n`.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if favour_candidate < 0 or favour_baseline < 0:
        raise ValueError(
            f"counts must be non-negative, got {favour_candidate} and {favour_baseline}"
        )
    if favour_candidate + favour_baseline > n:
        raise ValueError(f"discordant counts {favour_candidate} + {favour_baseline} exceed n = {n}")


def _check_tolerance(tolerance: float) -> None:
    """Refuse a tolerance outside (0, 1].

    A tolerance of zero turns the question back into a significance test, which is the thing
    this gate exists to avoid. One above 1 asks nothing, because no difference can exceed the
    score range. Written so that NaN fails too.

    Raises:
        ValueError: If `tolerance` is not in (0, 1].
    """
    if not 0.0 < tolerance <= 1.0:
        raise ValueError(
            f"tolerance must be in (0, 1], got {tolerance}; "
            "a tolerance of zero asks for significance, not non-inferiority"
        )


def _check_confidence(confidence: float) -> None:
    """Refuse a confidence level outside (0, 1).

    Raises:
        ValueError: If the level is out of range.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _phi(value: float) -> float:
    """Standard normal CDF, exact at the infinities the score statistic can return."""
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    return _NORMAL.cdf(value)


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjustment, controlling the family-wise error rate.

    Args:
        p_values: Raw p-values, in whatever order the caller holds them.

    Returns:
        Adjusted p-values **in the input order**, each at least its raw value and at most 1.0.

    Raises:
        ValueError: If any p-value lies outside [0, 1].
    """
    order = _validated_order(p_values)
    count = len(p_values)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjustment, controlling the false discovery rate.

    Args:
        p_values: Raw p-values, in whatever order the caller holds them.

    Returns:
        Adjusted p-values **in the input order**, each at most 1.0 and monotone in rank.

    Raises:
        ValueError: If any p-value lies outside [0, 1].
    """
    order = _validated_order(p_values)
    count = len(p_values)
    adjusted = [0.0] * count
    running = 1.0
    # Walking down from the largest p-value keeps the result monotone, which is what stops a
    # strongly significant slice from being adjusted above a weaker one.
    for rank in range(count - 1, -1, -1):
        index = order[rank]
        running = min(running, min(1.0, count / (rank + 1) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _validated_order(p_values: Sequence[float]) -> list[int]:
    """Return indices that sort `p_values` ascending, after range-checking them.

    Raises:
        ValueError: If any p-value lies outside [0, 1].
    """
    for index, value in enumerate(p_values):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"p-value at index {index} is outside [0, 1]: {value}")
    return sorted(range(len(p_values)), key=lambda index: p_values[index])


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def judge(interval: Interval, tolerance: float) -> Verdict:
    """Turn an interval and a tolerance into the gate's answer for one slice on its own.

    Args:
        interval: An interval on `candidate - baseline`.
        tolerance: How much worse the candidate is allowed to be, as a positive number on the
            scorer's own scale.

    Returns:
        One of the three verdicts. The function is total: every interval and every positive
        tolerance yields an answer.

    Raises:
        ValueError: If `tolerance` is not in (0, 1].
    """
    _check_tolerance(tolerance)
    if interval.low > -tolerance:
        return Verdict.NON_INFERIOR
    if interval.high < -tolerance:
        return Verdict.REGRESSED
    return Verdict.INCONCLUSIVE


def family_verdicts(results: Sequence[PairedResult], correction: Correction) -> list[Verdict]:
    """The verdicts for slices tested together, corrected for testing many at once.

    Only `regressed` is corrected. A check reports ok only when every slice is non-inferior,
    which already holds its false-pass rate to one slice's, so correcting that side would only
    cost power. A check fails when any slice regresses, so that error grows with the number of
    slices, and it is the one adjusted here.

    A slice is `regressed` only when its own interval says so **and** its adjusted p-value
    agrees; requiring both means a p-value can never produce a verdict the interval contradicts.

    Args:
        results: One family's results: all gate slices, or all watch slices.
        correction: Holm for gate slices, Benjamini-Hochberg for watch slices.

    Returns:
        One verdict per result, in the input order.

    Raises:
        ValueError: If the results were built at different confidence levels.
    """
    if not results:
        return []
    levels = {result.interval.confidence for result in results}
    if len(levels) > 1:
        raise ValueError(f"a family must share one confidence level, got {sorted(levels)}")
    threshold = (1.0 - levels.pop()) / 2.0

    raw = [result.p_regressed for result in results]
    adjusted = holm(raw) if correction is Correction.HOLM else benjamini_hochberg(raw)

    verdicts: list[Verdict] = []
    for result, p_value in zip(results, adjusted, strict=True):
        own = judge(result.interval, result.tolerance)
        if own is Verdict.REGRESSED and p_value >= threshold:
            own = Verdict.INCONCLUSIVE
        verdicts.append(own)
    return verdicts
