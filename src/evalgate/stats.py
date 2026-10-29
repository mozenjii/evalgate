"""Paired statistics for the non-inferiority gate.

Owning artifact: `app/technical context/02_STATISTICS_CONTRACT.md`. That document states the
mathematics; this module implements it. Where the two disagree, the contract is right.

The gate asks "is the candidate at most `delta` worse than the baseline", not "is it different".
A significance test on a forty-item slice passes almost everything, so its greenness means
nothing. Everything here therefore produces an interval on `candidate - baseline`, and the
verdict reads its lower bound.

Deliberate boundary: this module imports neither scipy nor statsmodels, at any point. Those two
are test-only dependencies and are the independent oracle in `tests/oracle/`. An implementation
that called scipy would make "agrees with scipy" a statement about nothing. stdlib `math` and
`statistics` carry the exact binomial work; numpy is here for the bootstrap's array arithmetic
only, which is not a statistical claim.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

__all__ = [
    "Interval",
    "PairedResult",
    "Verdict",
    "benjamini_hochberg",
    "clopper_pearson",
    "holm",
    "jackknife_acceleration",
    "judge",
    "mcnemar_exact_p",
    "paired_binary",
    "paired_continuous",
]

_NORMAL = NormalDist()

#: Bisection stops here. Two to the minus sixty is already finer than a float can express, so
#: further iterations buy nothing; the cap exists so a pathological input cannot spin forever.
_BISECTION_STEPS = 200


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
    """One slice's paired comparison, with the method that produced it.

    Attributes:
        n: Number of paired items.
        interval: The difference and its bounds.
        method: Which estimator ran: `exact_conditional` or `bootstrap_bca`.
        p_value: Exact McNemar for binary outcomes, `None` for continuous. Reported for
            information only -- the verdict never reads it, because a p-value answers a
            question the gate is not asking.
    """

    n: int
    interval: Interval
    method: str
    p_value: float | None


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
    low: float, high: float, target: float, *, increasing: bool, trials: int, successes: int
) -> float:
    """Solve a monotone binomial tail equation for the probability.

    Args:
        low: Lower end of the bracket.
        high: Upper end of the bracket.
        target: The tail probability to hit.
        increasing: True when the tail rises with the probability (`P(X >= k)`).
        trials: Binomial trial count.
        successes: Binomial success count.

    Returns:
        The probability at which the tail equals `target`.
    """
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2.0
        if middle <= low or middle >= high:
            break
        tail = (
            _binomial_at_least(successes, trials, middle)
            if increasing
            else _binomial_at_most(successes, trials, middle)
        )
        if (tail < target) == increasing:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion.

    Exact rather than Wald because the gate lives on thin slices, and the Wald interval's
    coverage falls well below its nominal level exactly there. An interval that is too narrow
    on a small slice produces confident wrong verdicts.

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
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    tail = (1.0 - confidence) / 2.0
    observed = successes / trials

    low = 0.0
    if successes > 0:
        low = _bisect(0.0, observed, tail, increasing=True, trials=trials, successes=successes)

    high = 1.0
    if successes < trials:
        high = _bisect(observed, 1.0, tail, increasing=False, trials=trials, successes=successes)

    return low, high


def mcnemar_exact_p(favour_candidate: int, favour_baseline: int) -> float:
    """Exact two-sided McNemar p-value.

    Conditional on the discordant pairs, the count favouring the candidate is binomial with
    probability one half under the null. The two-sided p-value is twice the smaller tail,
    which for a symmetric null is identical to summing every outcome at most as probable as
    the observed one.

    Args:
        favour_candidate: Items the baseline got wrong and the candidate got right.
        favour_baseline: Items the baseline got right and the candidate got wrong.

    Returns:
        The p-value, clamped to 1.0. With no discordant pairs there is nothing to test and the
        answer is 1.0.

    Raises:
        ValueError: If either count is negative.
    """
    if favour_candidate < 0 or favour_baseline < 0:
        raise ValueError(
            f"counts must be non-negative, got {favour_candidate} and {favour_baseline}"
        )
    discordant = favour_candidate + favour_baseline
    if discordant == 0:
        return 1.0
    smaller = min(favour_candidate, favour_baseline)
    return min(1.0, 2.0 * _binomial_at_most(smaller, discordant, 0.5))


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def paired_binary(
    baseline: Sequence[bool], candidate: Sequence[bool], confidence: float = 0.95
) -> PairedResult:
    """Compare two versions on a binary scorer, item by item.

    Pairing is the point. Comparing two independent proportions discards the item-level
    correlation that makes a small slice readable at all, and inflates the variance for nothing.

    Only discordant items carry information about the difference, so the interval is built by
    conditioning on them: the count favouring the candidate is binomial over the discordant
    pairs, and an exact interval for that proportion maps back onto the difference scale.

    The conditioning treats the observed discordance rate as fixed rather than estimated, so the
    interval is exact given that rate and mildly optimistic as an unconditional claim. That is a
    deliberate trade for correct small-sample behaviour, and the report says so rather than
    hiding it.

    Args:
        baseline: Per-item outcomes for the baseline version.
        candidate: Per-item outcomes for the candidate, in the same item order.
        confidence: Nominal confidence level.

    Returns:
        The paired result, with `method` set to `exact_conditional`.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    n = _check_pair(baseline, candidate)

    favour_candidate = sum(1 for b, c in zip(baseline, candidate, strict=True) if not b and c)
    favour_baseline = sum(1 for b, c in zip(baseline, candidate, strict=True) if b and not c)
    discordant = favour_candidate + favour_baseline
    point = (favour_candidate - favour_baseline) / n

    if discordant == 0:
        # Every item scored the same under both versions, so the difference is exactly zero --
        # but zero disagreements out of twenty is not proof of equality. Bound the discordance
        # rate itself instead. This is the two-sided bound, 1 - (alpha/2)**(1/n), which is
        # roughly 3.7/n at 95% rather than the one-sided rule of three's 3/n; two-sided keeps
        # it consistent with every other interval here.
        _, bound = clopper_pearson(0, n, confidence)
        interval = Interval(point=0.0, low=-bound, high=bound, confidence=confidence)
    else:
        share_low, share_high = clopper_pearson(favour_candidate, discordant, confidence)
        scale = discordant / n
        interval = Interval(
            point=point,
            low=(2.0 * share_low - 1.0) * scale,
            high=(2.0 * share_high - 1.0) * scale,
            confidence=confidence,
        )

    return PairedResult(
        n=n,
        interval=interval,
        method="exact_conditional",
        p_value=mcnemar_exact_p(favour_candidate, favour_baseline),
    )


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
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int | None = None,
) -> PairedResult:
    """Compare two versions on a continuous scorer with a BCa bootstrap.

    With no seed, one is derived from the data rather than left to chance. A gate whose
    verdict can change between two runs of the same commit is the failure a CI check exists
    to prevent, and an unseeded default would make that the normal case. Deriving it from the
    differences keeps reruns identical while still giving different data a different stream.

    The per-item differences are resampled, never the two arrays separately: resampling them
    independently would destroy the pairing that makes the comparison worth doing.

    Bias-corrected and accelerated rather than plain percentile, because scorer distributions
    are bounded and skewed, and the plain interval sits visibly off centre on them.

    Args:
        baseline: Per-item scores for the baseline version.
        candidate: Per-item scores for the candidate, in the same item order.
        confidence: Nominal confidence level.
        resamples: Bootstrap resample count. More costs time and buys resolution in the tails.
        seed: Seed for the resampling. The same seed gives the same interval.

    Returns:
        The paired result, with `method` set to `bootstrap_bca` and no p-value.

    Raises:
        ValueError: If the inputs differ in length, are empty, hold a single item, or
            `resamples` is not positive.
    """
    n = _check_pair(baseline, candidate)
    if n < 2:
        raise ValueError("a bootstrap interval on one item is meaningless; need at least 2")
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    differences = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    observed = float(np.mean(differences))

    # Compared against a tolerance, not exactly. A constant shift computed in floating point
    # leaves differences that disagree in the last bit, and bootstrapping that noise produces
    # an interval narrower than the arithmetic that built it -- narrow enough to fail to
    # bracket its own point estimate. Agreement to a trillionth of the scale is a constant
    # shift as far as any scorer is concerned.
    scale = max(abs(observed), 1.0)
    if float(np.ptp(differences)) <= 1e-12 * scale:
        # Every item moved by the same amount. There is nothing for a resample to vary, and
        # collapsing to the point estimate is the honest report rather than a manufactured width.
        return PairedResult(
            n=n,
            interval=Interval(observed, observed, observed, confidence),
            method="bootstrap_bca",
            p_value=None,
        )

    generator = np.random.default_rng(seed if seed is not None else _seed_from(differences))
    draws = generator.integers(0, n, size=(resamples, n))
    resampled: npt.NDArray[np.float64] = differences[draws]
    means: npt.NDArray[np.float64] = resampled.mean(axis=1)

    # Mid-rank, counting ties as half. The obvious reading is "strictly less than", and on
    # continuous data the two agree because exact ties essentially never occur. They diverge
    # systematically once the differences take few distinct values -- a scorer rounded to two
    # decimals, or set F1 over small sets -- and the strict form then biases the correction
    # downwards on every such slice, which no tolerance would separate from a bug.
    fewer: npt.NDArray[np.bool_] = means < observed
    tied: npt.NDArray[np.bool_] = means == observed
    below = (int(fewer.sum()) + int(tied.sum()) / 2.0) / resamples
    # At the extremes the normal quantile is infinite. Clamping to half a resample is the finest
    # resolution the resampling can express, so this reports the limit of the evidence rather
    # than an infinity.
    resolution = 1.0 / (2.0 * resamples)
    bias = _NORMAL.inv_cdf(min(max(below, resolution), 1.0 - resolution))
    acceleration = jackknife_acceleration([float(value) for value in differences])

    alpha = 1.0 - confidence
    low_pct = _adjusted_percentile(bias, acceleration, _NORMAL.inv_cdf(alpha / 2.0))
    high_pct = _adjusted_percentile(bias, acceleration, _NORMAL.inv_cdf(1.0 - alpha / 2.0))

    ordered: npt.NDArray[np.float64] = means.copy()
    ordered.sort()
    low = _percentile(ordered, low_pct)
    high = _percentile(ordered, high_pct)

    return PairedResult(
        n=n,
        interval=Interval(point=observed, low=low, high=high, confidence=confidence),
        method="bootstrap_bca",
        p_value=None,
    )


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


def _check_pair(baseline: Sequence[object], candidate: Sequence[object]) -> int:
    """Validate that two per-item sequences can be paired, returning their length.

    Raises:
        ValueError: If the lengths differ or the sequences are empty.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired inputs must be the same length, got {len(baseline)} and {len(candidate)}"
        )
    if not baseline:
        raise ValueError("cannot compare empty sequences; there is no estimate to report")
    return len(baseline)


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjustment, controlling the family-wise error rate.

    Used for gate slices, the ones that can fail a build, where any single false alarm is the
    thing to avoid.

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

    Used for watch slices, which are reported and never fail a build. A looser correction is
    right there: the cost of a false flag is someone reading one more line.

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
    """Turn an interval and a tolerance into the gate's answer for one slice.

    Args:
        interval: An interval on `candidate - baseline`.
        tolerance: How much worse the candidate is allowed to be, as a positive number on the
            scorer's own scale.

    Returns:
        One of the three verdicts. The function is total: every interval and every positive
        tolerance yields an answer.

    Raises:
        ValueError: If `tolerance` is not strictly positive. A tolerance of zero turns the
            question back into a significance test, which is the thing this gate exists to
            avoid, so it is refused rather than quietly accepted.
    """
    if tolerance <= 0.0:
        raise ValueError(
            f"tolerance must be strictly positive, got {tolerance}; "
            "a tolerance of zero asks for significance, not non-inferiority"
        )
    if interval.low > -tolerance:
        return Verdict.NON_INFERIOR
    if interval.high < -tolerance:
        return Verdict.REGRESSED
    return Verdict.INCONCLUSIVE
