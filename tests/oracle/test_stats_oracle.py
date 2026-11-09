"""Independent oracle for `evalgate.stats`, checked against scipy and statsmodels.

Owning artifact: `docs/statistics.md`. Section numbers in the comments below refer to
that document.

Every expectation here was authored from the contract and from scipy/statsmodels only.
Nothing in this file was written while reading `src/evalgate/stats.py`, which is the
whole point: an oracle shaped by the code it judges agrees with that code's bugs. When
this file and the implementation disagree, the disagreement is resolved by finding the
cause in one of them. This file is never edited to agree.

Tango's score statistic (contract 3.3) has no library implementation to compare with, so
the reference here is built from its definition rather than from the contract's closed
form: the restricted maximum-likelihood estimate is found numerically from the trinomial
likelihood, the statistic is formed from it, and the interval is found by scanning that
statistic on a dense grid and refining each crossing with scipy's root-finder. The closed
form is then checked against the numerical estimate, so the contract itself is under
test as well as the module.

Deliberate boundary: scipy and statsmodels are test-only dependencies (contract 9).
They are imported here and must never be imported by the module under test;
`test_stats_module_imports_neither_scipy_nor_statsmodels` enforces that directly.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import scipy.optimize
import scipy.stats
import statsmodels.stats.multitest
import statsmodels.stats.proportion

from evalgate import stats

# scipy and statsmodels ship no type information, so pyright infers `Unknown` for every
# symbol imported from them, which is an error under strict mode. Binding each module
# to an explicitly `Any`-typed name narrows the untypedness to these lines instead of
# scattering suppression comments through the file, and every value that crosses back
# out of a reference helper is converted with an explicit `float`/`list`.
_scipy_stats: Any = scipy.stats
_scipy_optimize: Any = scipy.optimize
_sm_proportion: Any = statsmodels.stats.proportion
_sm_multitest: Any = statsmodels.stats.multitest

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

# Deterministic exact quantities (Clopper-Pearson bounds, Holm and Benjamini-Hochberg
# adjustments, Tango interval endpoints, margin p-values) are computed by both sides in
# double precision from closed forms or root-finds. scipy and statsmodels agree with each
# other to about 2e-14 on the Clopper-Pearson grid below, so a well-implemented third
# computation has ~1e-12 of genuine headroom. 1e-9 leaves room for an implementation that
# inverts a distribution or a score statistic by its own bisection rather than by a library
# special function, and is still six orders of magnitude tighter than any interval the gate
# ever reports.
_EXACT_TOLERANCE = 1e-9

# A tighter band for identities that are pure arithmetic on the same inputs, where any
# disagreement beyond accumulated rounding is a real difference in definition.
_ARITHMETIC_TOLERANCE = 1e-12

# Tango's statistic, compared relative to max(1, |Z|). The contract's closed form for q(D)
# loses digits as D approaches -1 when nearly every discordant item favours the baseline:
# the square root is then of a near-cancelling sum. Measured against this file's numerical
# reference on n up to 1000, a direct transcription of the closed form disagrees by at
# most 3.6e-9 (relative) while D stays at least 1e-4 from either end, and by 2.5e-6 at
# 1e-6 from the end. The comparison grid therefore stops at +-0.9999, the ends themselves
# are checked by the contract's exact rule, and 1e-7 leaves ~30x headroom over the worst
# honest rounding while any real error in the statistic moves it by orders of magnitude more.
_TANGO_RELATIVE_TOLERANCE = 1e-7

# The tolerance passed to the paired functions where the test is about the interval, which
# does not depend on it (contract 3.3, 4.2, 4.4, 4.5).
_DELTA = 0.05


# ---------------------------------------------------------------------------
# Reference helpers. Each one wraps a single scipy or statsmodels call, or writes one
# contract formula out with no shortcuts, and returns plain floats, so a mistake in how
# the reference is invoked is visible in one place rather than smeared across thirty
# assertions.
# ---------------------------------------------------------------------------


def _clopper_pearson_scipy(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Clopper-Pearson interval from `scipy.stats.binomtest(...).proportion_ci`."""
    interval = _scipy_stats.binomtest(successes, trials, 0.5).proportion_ci(
        confidence_level=confidence, method="exact"
    )
    return float(interval.low), float(interval.high)


def _clopper_pearson_statsmodels(
    successes: int, trials: int, confidence: float
) -> tuple[float, float]:
    """Clopper-Pearson interval from `proportion_confint(method="beta")`, as a second opinion."""
    low, high = _sm_proportion.proportion_confint(
        successes, trials, alpha=1.0 - confidence, method="beta"
    )
    return float(low), float(high)


def _clopper_pearson_upper_beta(successes: int, trials: int, confidence: float) -> float:
    """Clopper-Pearson upper bound as a beta quantile, with contract 4.4's `k = n` rule.

    The upper bound solving `P(X <= k | n, hi) = alpha / 2` is the `1 - alpha/2` quantile
    of Beta(k + 1, n - k). At `k = n` that distribution does not exist and the contract
    pins the bound to exactly 1.
    """
    if successes == trials:
        return 1.0
    alpha = 1.0 - confidence
    return float(_scipy_stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))


def _normal_quantile(probability: float) -> float:
    """`Phi^-1` from scipy."""
    return float(_scipy_stats.norm.ppf(probability))


def _normal_cdf(value: float) -> float:
    """`Phi` from scipy, which already gives `Phi(+inf) = 1` and `Phi(-inf) = 0`."""
    return float(_scipy_stats.norm.cdf(value))


def _normal_sf(value: float) -> float:
    """`1 - Phi` from scipy's survival function, which does not cancel in the upper tail."""
    return float(_scipy_stats.norm.sf(value))


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """`P(Binomial(trials, probability) <= successes)` from scipy."""
    return float(_scipy_stats.binom.cdf(successes, trials, probability))


def _multipletests(p_values: Sequence[float], method: str) -> list[float]:
    """Adjusted p-values from statsmodels, in input order."""
    adjusted = _sm_multitest.multipletests(list(p_values), method=method)[1]
    return [float(value) for value in adjusted]


def _bca_scipy(
    differences: FloatArray, confidence: float, resamples: int, seed: int
) -> tuple[float, float]:
    """BCa interval for the mean of a one-sample vector of paired differences."""
    result = _scipy_stats.bootstrap(
        (differences,),
        np.mean,
        confidence_level=confidence,
        n_resamples=resamples,
        method="BCa",
        rng=np.random.default_rng(seed),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _paired_mean_difference_via_ttest(baseline: FloatArray, candidate: FloatArray) -> float:
    """Recover `mean(candidate - baseline)` from scipy's paired t statistic.

    The t statistic is `mean / (sd / sqrt(n))` with `sd` the sample standard deviation,
    so multiplying it back by the standard error is an independent route to the point
    estimate that never forms the difference vector the way the implementation does.
    """
    statistic = float(_scipy_stats.ttest_rel(candidate, baseline).statistic)
    differences = candidate - baseline
    standard_error = float(np.std(differences, ddof=1)) / math.sqrt(differences.size)
    return statistic * standard_error


def _jackknife_acceleration_reference(values: Sequence[float]) -> float:
    """The acceleration written straight from contract 4.2, with no shortcuts.

    `theta_i` is the mean leaving item `i` out, `theta_bar` is the mean of those, and
    `a = sum((theta_bar - theta_i)^3) / (6 * (sum((theta_bar - theta_i)^2))^(3/2))`.
    Contract 4.7 defines `a = 0` when the jackknife spread is zero.
    """
    data = np.asarray(values, dtype=np.float64)
    count = int(data.size)
    total = float(np.sum(data))
    leave_one_out = np.asarray(
        [(total - float(value)) / (count - 1) for value in data], dtype=np.float64
    )
    theta_bar = float(np.mean(leave_one_out))
    deviations = theta_bar - leave_one_out
    cubed = float(np.sum(deviations**3))
    squared = float(np.sum(deviations**2))
    if squared == 0.0:
        return 0.0
    return cubed / (6.0 * squared**1.5)


def _jackknife_acceleration_closed_form(values: Sequence[float]) -> float:
    """The same acceleration derived algebraically, as a check on the loop above.

    For the mean statistic `theta_bar - theta_i` is exactly `(x_i - xbar) / (n - 1)`,
    and the `(n - 1)` factors cancel between the cube and the three-halves power, so
    `a = sum(c^3) / (6 * (sum(c^2))^(3/2))` with `c = x - xbar`. Two derivations that
    agree make it much less likely the oracle's own formula is the one that is wrong.
    """
    data = np.asarray(values, dtype=np.float64)
    centred = data - float(np.mean(data))
    squared = float(np.sum(centred**2))
    if squared == 0.0:
        return 0.0
    return float(np.sum(centred**3)) / (6.0 * squared**1.5)


def _close(actual: float, expected: float, tolerance: float) -> bool:
    """True when two floats agree to an absolute tolerance, NaN never counting as agreement."""
    return math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= tolerance


# ---------------------------------------------------------------------------
# Tango reference (contract 3.3), from the likelihood rather than the closed form.
# ---------------------------------------------------------------------------


def _restricted_mle(
    favour_candidate: int, favour_baseline: int, n: int, differences: FloatArray
) -> FloatArray:
    """The probability an item favours the baseline, maximised with the difference held at D.

    Under a true difference `D` an item favours the candidate with probability `p + D`,
    the baseline with probability `p`, and neither with `1 - 2p - D`, so the restricted
    log-likelihood is `b log(p + D) + c log(p) + m log(1 - 2p - D)` over the feasible
    `p` in `[max(0, -D), (1 - D) / 2]`. It is a sum of logs of affine functions of `p`,
    hence concave, so its derivative (the score) decreases across the feasible interval
    and the maximiser is where the score changes sign -- or the end it runs into when it
    never does. Bisection on the sign of the score finds that point using nothing but the
    likelihood: 100 halvings of an interval no wider than 1 reach below double-precision
    resolution. A term with a zero count is dropped (`0 log 0 = 0`). At `D = +-1` the
    feasible set is a single point and the loop leaves it there, which is exactly where
    contract 3.3's zero-denominator rule takes over.
    """
    neither = n - favour_candidate - favour_baseline
    low: FloatArray = np.maximum(0.0, -differences)
    high: FloatArray = (1.0 - differences) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(100):
            middle = (low + high) / 2.0
            score = np.zeros_like(middle)
            if favour_candidate:
                score = score + favour_candidate / (middle + differences)
            if favour_baseline:
                score = score + favour_baseline / middle
            if neither:
                score = score - 2.0 * neither / (1.0 - 2.0 * middle - differences)
            rising = score > 0.0
            low = np.where(rising, middle, low)
            high = np.where(rising, high, middle)
    return (low + high) / 2.0


def _tango_z_reference(
    favour_candidate: int, favour_baseline: int, n: int, differences: FloatArray
) -> FloatArray:
    """Tango's Z(D) from the numerical restricted MLE, with contract 3.3's zero rule.

    `Z(D) = (b - c - n D) / sqrt(n (2 q(D) + D (1 - D)))`. When the denominator is zero,
    `Z` is `0` if the numerator is `0` and an infinity of the numerator's sign otherwise.
    """
    restricted = _restricted_mle(favour_candidate, favour_baseline, n, differences)
    numerator = (favour_candidate - favour_baseline) - n * differences
    variance = n * (2.0 * restricted + differences * (1.0 - differences))
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = numerator / np.sqrt(np.maximum(variance, 0.0))
        by_rule = np.where(numerator == 0.0, 0.0, np.copysign(np.inf, numerator))
    return np.where(variance <= 0.0, by_rule, statistic)


def _tango_z_at(favour_candidate: int, favour_baseline: int, n: int, difference: float) -> float:
    """The reference statistic at a single hypothesised difference."""
    values = np.asarray([difference], dtype=np.float64)
    return float(_tango_z_reference(favour_candidate, favour_baseline, n, values)[0])


def _tango_closed_form_q(
    favour_candidate: int, favour_baseline: int, n: int, differences: FloatArray
) -> FloatArray:
    """The contract's closed form for `q(D)`, used only to test the contract against the MLE."""
    w = (
        -favour_candidate
        - favour_baseline
        + (2 * n - favour_candidate + favour_baseline) * (differences)
    )
    inside = w * w + 8.0 * n * favour_baseline * differences * (1.0 - differences)
    return (np.sqrt(inside) - w) / (4.0 * n)


def _likelihood_maximiser(
    favour_candidate: int, favour_baseline: int, n: int, difference: float
) -> float:
    """The restricted MLE by direct numerical maximisation, as a check on the bisection.

    `scipy.optimize.minimize_scalar(method="bounded")` on the negative log-likelihood,
    which never looks at the score. A smooth maximum is flat to second order, so this
    locates it only to about the square root of machine precision; it is a second
    derivation of the reference, not a replacement for it.
    """
    neither = n - favour_candidate - favour_baseline
    low = max(0.0, -difference)
    high = (1.0 - difference) / 2.0
    if high <= low:
        return low

    def negative_log_likelihood(p: float) -> float:
        total = 0.0
        if favour_candidate:
            total += favour_candidate * math.log(max(p + difference, 1e-300))
        if favour_baseline:
            total += favour_baseline * math.log(max(p, 1e-300))
        if neither:
            total += neither * math.log(max(1.0 - 2.0 * p - difference, 1e-300))
        return -total

    result = _scipy_optimize.minimize_scalar(
        negative_log_likelihood, bounds=(low, high), method="bounded", options={"xatol": 1e-12}
    )
    return float(result.x)


#: Dense enough that an accepted region narrower than 0.0005 is the only thing a scan
#: can miss; every Tango interval tested here is at least an order of magnitude wider.
_TANGO_GRID: FloatArray = np.linspace(-1.0, 1.0, 4001)


@functools.cache
def _tango_grid_statistic(favour_candidate: int, favour_baseline: int, n: int) -> FloatArray:
    """The reference Z over the whole grid, computed once per count triple."""
    values = _tango_z_reference(favour_candidate, favour_baseline, n, _TANGO_GRID)
    values.setflags(write=False)
    return values


def _refine_acceptance_boundary(
    favour_candidate: int,
    favour_baseline: int,
    n: int,
    critical: float,
    rejected: float,
    accepted: float,
) -> float:
    """Root of `|Z(D)| - z` between a rejected and an accepted grid point, by `brentq`.

    `Z` is infinite only where the denominator vanishes, which on the grid is at
    `D = +-1`. The root-finder needs finite values, so a rejected endpoint there is moved
    1e-12 inside; if that point is already accepted, the boundary is within 1e-12 of the
    end and that is the answer.
    """

    def excess(difference: float) -> float:
        return abs(_tango_z_at(favour_candidate, favour_baseline, n, difference)) - critical

    if not math.isfinite(excess(rejected)):
        rejected += math.copysign(1e-12, accepted - rejected)
        if excess(rejected) <= 0.0:
            return rejected
    return float(_scipy_optimize.brentq(excess, rejected, accepted, xtol=1e-15, maxiter=200))


@functools.cache
def _tango_interval_reference(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float
) -> tuple[float, float, bool]:
    """Contract 3.3's interval by inverting the reference statistic: (low, high, connected).

    The accepted set `{D : -z <= Z(D) <= z}` is found on the grid, its first and last
    accepted points are refined against their rejected neighbours, and an end that is
    itself accepted is reported exactly (`-1` or `1`). `connected` says whether the
    accepted grid points form one run; a second accepted region would make "the smallest
    such D" and "the largest such D" describe something other than an interval.
    """
    critical = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    statistic = _tango_grid_statistic(favour_candidate, favour_baseline, n)
    indices = np.flatnonzero(np.abs(statistic) <= critical)
    assert indices.size > 0, "the point estimate always has Z = 0, so something is accepted"
    first = int(indices[0])
    last = int(indices[-1])
    connected = last - first + 1 == int(indices.size)
    if first == 0:
        low = -1.0
    else:
        low = _refine_acceptance_boundary(
            favour_candidate,
            favour_baseline,
            n,
            critical,
            float(_TANGO_GRID[first - 1]),
            float(_TANGO_GRID[first]),
        )
    if last == int(_TANGO_GRID.size) - 1:
        high = 1.0
    else:
        high = _refine_acceptance_boundary(
            favour_candidate,
            favour_baseline,
            n,
            critical,
            float(_TANGO_GRID[last + 1]),
            float(_TANGO_GRID[last]),
        )
    return low, high, connected


def _judge_reference(low: float, high: float, tolerance: float) -> stats.Verdict:
    """Contract 6.1's table, written out independently of `stats.judge`."""
    if low > -tolerance:
        return stats.Verdict.NON_INFERIOR
    if high < -tolerance:
        return stats.Verdict.REGRESSED
    return stats.Verdict.INCONCLUSIVE


def _family_verdicts_reference(
    results: Sequence[stats.PairedResult], method: str
) -> list[stats.Verdict]:
    """Contract 6.2 with statsmodels doing the adjustment (`holm` or `fdr_bh`)."""
    if not results:
        return []
    alpha = 1.0 - results[0].interval.confidence
    adjusted = _multipletests([result.p_regressed for result in results], method)
    verdicts: list[stats.Verdict] = []
    for result, adjusted_p in zip(results, adjusted, strict=True):
        own = _judge_reference(result.interval.low, result.interval.high, result.tolerance)
        if own is stats.Verdict.NON_INFERIOR:
            verdicts.append(stats.Verdict.NON_INFERIOR)
        elif own is stats.Verdict.REGRESSED and adjusted_p < alpha / 2.0:
            verdicts.append(stats.Verdict.REGRESSED)
        else:
            verdicts.append(stats.Verdict.INCONCLUSIVE)
    return verdicts


# ---------------------------------------------------------------------------
# Case grids and sample builders.
# ---------------------------------------------------------------------------

_CONFIDENCES: tuple[float, ...] = (0.80, 0.90, 0.95, 0.99)


def _clopper_pearson_cases() -> list[tuple[int, int, float]]:
    cases: list[tuple[int, int, float]] = []
    for trials in (1, 2, 3, 5, 10, 23, 40, 97):
        for successes in sorted({0, 1, trials // 3, trials // 2, trials - 1, trials}):
            for confidence in _CONFIDENCES:
                cases.append((successes, trials, confidence))
    return cases


_CLOPPER_PEARSON_CASES: list[tuple[int, int, float]] = _clopper_pearson_cases()

#: (favour_candidate, favour_baseline, n): no discordance, one side empty, every item
#: favouring one side (c = n, b = n), lopsided and balanced splits, from n = 1 to 200.
_TANGO_COUNTS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (0, 0, 20),
    (3, 0, 20),
    (0, 3, 20),
    (20, 0, 20),
    (0, 20, 20),
    (5, 5, 20),
    (2, 9, 40),
    (9, 2, 40),
    (12, 4, 40),
    (10, 30, 40),
    (0, 40, 40),
    (40, 0, 40),
    (13, 15, 97),
    (31, 2, 97),
    (1, 1, 200),
    (0, 7, 200),
    (60, 58, 200),
)

_TANGO_IDS: list[str] = [f"b{b}-c{c}-n{n}" for b, c, n in _TANGO_COUNTS]

#: Interior hypothesised differences, including both sides of zero and near the ends.
#: The ends themselves are the zero-denominator rule's cases and are tested separately.
_TANGO_DIFFERENCES: tuple[float, ...] = (
    -0.9999,
    -0.999,
    -0.9,
    -0.5,
    -0.25,
    -0.1,
    -0.01,
    0.0,
    0.01,
    0.1,
    0.25,
    0.5,
    0.9,
    0.999,
    0.9999,
)

#: Tolerances for the margin p-values. 1.0 puts `-delta` on the end of the range, where
#: Z is given by the zero-denominator rule.
_MARGINS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.5, 0.999, 1.0)

#: p-value vectors: unsorted, tied, ascending, descending, clamping, single, empty.
_P_VALUE_CASES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("unsorted", (0.01, 0.04, 0.03, 0.005, 0.2, 1.0, 0.9)),
    ("single", (0.031,)),
    ("ties", (0.02, 0.02, 0.02)),
    ("ties_mixed", (0.001, 0.02, 0.02, 0.5, 0.5, 0.5)),
    ("ascending", (0.001, 0.01, 0.02, 0.2, 0.44, 0.8)),
    ("descending", (0.8, 0.44, 0.2, 0.02, 0.01, 0.001)),
    ("clamping", (0.4, 0.5, 0.6, 0.7)),
    ("all_large", (0.9, 0.95, 0.99, 1.0)),
    ("two", (0.6, 0.02)),
    ("extremes", (0.0, 1.0, 0.5)),
)


def _binary_sample(seed: int, count: int, drift: float) -> tuple[list[bool], list[bool]]:
    """A paired binary sample where the candidate is shifted by `drift` relative to baseline."""
    rng = np.random.default_rng(seed)
    base_rate = 0.65
    baseline = rng.random(count) < base_rate
    candidate = rng.random(count) < min(max(base_rate + drift, 0.0), 1.0)
    return [bool(value) for value in baseline], [bool(value) for value in candidate]


def _binary_from_counts(
    favour_candidate: int, favour_baseline: int, n: int, seed: int = 0
) -> tuple[list[bool], list[bool]]:
    """Boolean samples with exactly these discordant counts, concordant items mixed, shuffled."""
    pairs: list[tuple[bool, bool]] = [(False, True)] * favour_candidate
    pairs += [(True, False)] * favour_baseline
    for index in range(n - favour_candidate - favour_baseline):
        pairs.append((True, True) if index % 2 == 0 else (False, False))
    order = np.random.default_rng(seed).permutation(n)
    shuffled = [pairs[int(index)] for index in order]
    return [base for base, _ in shuffled], [cand for _, cand in shuffled]


def _counts_of(baseline: Sequence[bool], candidate: Sequence[bool]) -> tuple[int, int]:
    """Contract 3.1's discordant counts, read off the samples."""
    favour_candidate = sum(1 for b, c in zip(baseline, candidate, strict=True) if not b and c)
    favour_baseline = sum(1 for b, c in zip(baseline, candidate, strict=True) if b and not c)
    return favour_candidate, favour_baseline


#: Four paired continuous samples with different shapes, every score in [0, 1], every
#: item moved and the differences spread out, so each one takes contract 4.2's bootstrap
#: branch (checked by `test_bootstrap_samples_satisfy_the_bootstrap_precondition`).
#: `normal` is here for breadth; `gamma` and `lognormal` are here because BCa and the
#: plain percentile interval are visibly different on them, which is what gives the
#: bootstrap comparison its power.
_CONTINUOUS_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("normal", 60, 101),
    ("gamma", 40, 202),
    ("beta", 30, 303),
    ("lognormal", 50, 404),
)


def _continuous_sample(shape: str, count: int, seed: int) -> tuple[list[float], list[float]]:
    """A paired continuous sample in [0, 1] whose differences follow `shape`.

    The baseline range for each shape is chosen so the candidate stays inside [0, 1]
    without help; the clip is a guarantee of contract 4.3's bound, and on these seeds it
    never binds, so the differences keep the shape they were drawn with.
    """
    rng = np.random.default_rng(seed)
    if shape == "normal":
        differences = rng.normal(-0.03, 0.12, count)
        floor, ceiling = 0.3, 0.7
    elif shape == "gamma":
        differences = rng.gamma(1.2, 0.1, count) - 0.12
        floor, ceiling = 0.15, 0.35
    elif shape == "beta":
        differences = rng.beta(1.5, 6.0, count) - 0.2
        floor, ceiling = 0.2, 0.35
    elif shape == "lognormal":
        differences = 0.12 * (np.exp(rng.normal(0.0, 0.8, count)) - 1.3)
        floor, ceiling = 0.16, 0.3
    elif shape == "regressed":
        differences = rng.normal(-0.3, 0.08, count)
        floor, ceiling = 0.55, 0.9
    else:  # pragma: no cover - a typo in a parametrize id must not pass silently
        raise AssertionError(f"unknown shape {shape!r}")
    baseline = rng.uniform(floor, ceiling, count)
    candidate = np.clip(baseline + differences, 0.0, 1.0)
    return [float(value) for value in baseline], [float(value) for value in candidate]


def _differences(baseline: Sequence[float], candidate: Sequence[float]) -> FloatArray:
    """The paired differences the implementation is required to form (contract 4.1)."""
    return np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)


def _takes_bootstrap_branch(baseline: Sequence[float], candidate: Sequence[float]) -> bool:
    """Contract 4.2's precondition: scores in [0, 1], 8+ items moved, not all the same."""
    scores = np.asarray([*baseline, *candidate], dtype=np.float64)
    if not bool(np.all((scores >= 0.0) & (scores <= 1.0))):
        return False
    differences = _differences(baseline, candidate)
    moved = int(np.count_nonzero(differences))
    spread = float(np.max(differences) - np.min(differences))
    identical = spread <= 1e-12 * max(abs(float(np.mean(differences))), 1.0)
    return moved >= 8 and not identical


def _few_moved_sample(count: int, up: int, down: int, seed: int) -> tuple[list[float], list[float]]:
    """Scores in [0, 1] where exactly `up` items rose and `down` fell, the rest unchanged.

    Magnitudes are drawn from (0.05, 1], and when at least three items move in a
    direction one of them moves the full score range, so the bound's reliance on
    "a difference is never below -1" is exercised at its limit.
    """
    rng = np.random.default_rng(seed)
    baseline: list[float] = []
    candidate: list[float] = []
    for index in range(up):
        magnitude = 1.0 if index == 0 and up >= 3 else float(rng.uniform(0.05, 1.0))
        start = float(rng.uniform(0.0, 1.0 - magnitude))
        baseline.append(start)
        candidate.append(min(1.0, start + magnitude))
    for index in range(down):
        magnitude = 1.0 if index == 0 and down >= 3 else float(rng.uniform(0.05, 1.0))
        start = float(rng.uniform(magnitude, 1.0))
        baseline.append(start)
        candidate.append(max(0.0, start - magnitude))
    for _ in range(count - up - down):
        value = float(rng.uniform(0.0, 1.0))
        baseline.append(value)
        candidate.append(value)
    order = rng.permutation(count)
    return [baseline[int(i)] for i in order], [candidate[int(i)] for i in order]


def _constant_shift_sample(count: int, shift: float, seed: int) -> tuple[list[float], list[float]]:
    """Every item moved by exactly `shift`, on a dyadic grid so the arithmetic is exact."""
    rng = np.random.default_rng(seed)
    floor = max(0.0, -shift)
    ceiling = min(1.0, 1.0 - shift)
    steps = rng.integers(0, 65, count)
    baseline = [floor + (ceiling - floor) * int(step) / 64.0 for step in steps]
    candidate = [value + shift for value in baseline]
    return baseline, candidate


def _paired_result(
    low: float,
    high: float,
    tolerance: float,
    p_regressed: float,
    confidence: float = 0.95,
) -> stats.PairedResult:
    """A `PairedResult` built directly from contract 7's fields, for the family tests."""
    return stats.PairedResult(
        n=40,
        interval=stats.Interval(
            point=(low + high) / 2.0, low=low, high=high, confidence=confidence
        ),
        method="tango_score",
        tolerance=tolerance,
        p_non_inferior=1.0 - p_regressed,
        p_regressed=p_regressed,
    )


# ---------------------------------------------------------------------------
# 1. Clopper-Pearson (contract 4.4, which defines it).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("successes", "trials", "confidence"), _CLOPPER_PEARSON_CASES)
def test_clopper_pearson_matches_scipy(successes: int, trials: int, confidence: float) -> None:
    # A failure means the exact binomial interval is wrong, which moves every count bound
    # the gate reports. The grid covers both extremes, where the two tails are defined by
    # a boundary rule rather than by a root-find.
    expected = _clopper_pearson_scipy(successes, trials, confidence)
    actual = stats.clopper_pearson(successes, trials, confidence)

    assert _close(actual[0], expected[0], _EXACT_TOLERANCE), (
        f"lower bound for {successes}/{trials} at {confidence}: "
        f"{actual[0]!r} != scipy {expected[0]!r}"
    )
    assert _close(actual[1], expected[1], _EXACT_TOLERANCE), (
        f"upper bound for {successes}/{trials} at {confidence}: "
        f"{actual[1]!r} != scipy {expected[1]!r}"
    )


@pytest.mark.parametrize(("successes", "trials", "confidence"), _CLOPPER_PEARSON_CASES)
def test_clopper_pearson_matches_statsmodels(
    successes: int, trials: int, confidence: float
) -> None:
    # A second, independently implemented reference. If scipy and statsmodels both
    # disagree with the module, the module is wrong; if only one does, the reference
    # call in this file is the suspect.
    expected = _clopper_pearson_statsmodels(successes, trials, confidence)
    actual = stats.clopper_pearson(successes, trials, confidence)

    assert _close(actual[0], expected[0], _EXACT_TOLERANCE)
    assert _close(actual[1], expected[1], _EXACT_TOLERANCE)


@pytest.mark.parametrize("trials", [1, 5, 20, 97])
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_clopper_pearson_boundary_cases_are_exactly_zero_and_one(
    trials: int, confidence: float
) -> None:
    # Contract 4.4 pins these two by rule, not by a root-find. A solver that returns
    # 1e-17 instead of 0 would pass a loose comparison and be wrong in principle.
    assert stats.clopper_pearson(0, trials, confidence)[0] == 0.0
    assert stats.clopper_pearson(trials, trials, confidence)[1] == 1.0


@pytest.mark.parametrize(("successes", "trials"), [(0, 10), (3, 10), (7, 20), (20, 20), (1, 97)])
def test_clopper_pearson_widens_with_confidence(successes: int, trials: int) -> None:
    # Invariant 4 at the level of the proportion interval. A tail direction swapped in
    # the solver shows up here as an interval that shrinks when confidence rises.
    widths = [
        stats.clopper_pearson(successes, trials, confidence)[1]
        - stats.clopper_pearson(successes, trials, confidence)[0]
        for confidence in _CONFIDENCES
    ]
    for narrower, wider in pairwise(widths):
        assert wider >= narrower - _ARITHMETIC_TOLERANCE


@pytest.mark.parametrize(("successes", "trials", "confidence"), _CLOPPER_PEARSON_CASES)
def test_clopper_pearson_brackets_the_observed_proportion(
    successes: int, trials: int, confidence: float
) -> None:
    # The interval must contain the point it is an interval for, and stay in [0, 1].
    low, high = stats.clopper_pearson(successes, trials, confidence)
    assert 0.0 <= low <= successes / trials <= high <= 1.0


# ---------------------------------------------------------------------------
# 2. The oracle's own Tango reference, checked before anything is held to it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
def test_oracle_restricted_mle_agrees_with_direct_likelihood_maximisation(
    favour_candidate: int, favour_baseline: int, n: int
) -> None:
    # Two derivations of q(D): bisection on the score, and scipy maximising the
    # likelihood without ever looking at its derivative. If they disagree, the reference
    # every Tango test below leans on is the suspect, not the module. 1e-6 because a
    # bounded Brent search locates a flat maximum only to about sqrt(machine epsilon)
    # (measured worst 2.8e-8 on this grid).
    differences = np.asarray(_TANGO_DIFFERENCES, dtype=np.float64)
    bisected = _restricted_mle(favour_candidate, favour_baseline, n, differences)
    for difference, by_bisection in zip(_TANGO_DIFFERENCES, bisected, strict=True):
        maximised = _likelihood_maximiser(favour_candidate, favour_baseline, n, difference)
        assert _close(float(by_bisection), maximised, 1e-6), (
            f"D={difference}: bisection {float(by_bisection)!r} vs maximiser {maximised!r}"
        )


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
def test_contract_closed_form_for_q_is_the_restricted_mle(
    favour_candidate: int, favour_baseline: int, n: int
) -> None:
    # Contract 3.3 says q(D) "is the maximum-likelihood probability that an item favours
    # the baseline, restricted to the true difference being D". This checks the
    # contract's own formula against that claim, independently of any implementation.
    # On this grid the two agree to about 3e-13; 1e-9 covers the closed form's rounding.
    differences = np.asarray([-1.0, *_TANGO_DIFFERENCES, 1.0], dtype=np.float64)
    numerical = _restricted_mle(favour_candidate, favour_baseline, n, differences)
    closed = _tango_closed_form_q(favour_candidate, favour_baseline, n, differences)
    for difference, by_mle, by_formula in zip(differences, numerical, closed, strict=True):
        assert _close(float(by_mle), float(by_formula), _EXACT_TOLERANCE), (
            f"D={float(difference)}: MLE {float(by_mle)!r} vs closed form {float(by_formula)!r}"
        )


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_tango_acceptance_region_is_one_connected_set(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float
) -> None:
    # Contract 3.3 defines low and high as the smallest and largest accepted D. That is an
    # interval only if nothing outside [low, high] is accepted as well. On a 0.0005 grid
    # the accepted set is one run for every case tested; if it were not, "the interval"
    # would silently include rejected differences between two accepted regions.
    _, _, connected = _tango_interval_reference(favour_candidate, favour_baseline, n, confidence)
    assert connected, "the accepted set has more than one region on the grid"


# ---------------------------------------------------------------------------
# 3. Tango's score statistic and interval (contract 3.3, 3.4).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
def test_tango_statistic_matches_the_likelihood_reference(
    favour_candidate: int, favour_baseline: int, n: int
) -> None:
    # The statistic is the root of the interval and of both margin p-values, so an error
    # here moves every binary verdict. The reference never uses the closed form.
    for difference in _TANGO_DIFFERENCES:
        expected = _tango_z_at(favour_candidate, favour_baseline, n, difference)
        actual = stats.tango_statistic(favour_candidate, favour_baseline, n, difference)
        bound = _TANGO_RELATIVE_TOLERANCE * max(1.0, abs(expected))
        assert _close(actual, expected, bound), (
            f"Z({difference}) for b={favour_candidate}, c={favour_baseline}, n={n}: "
            f"{actual!r} != reference {expected!r}"
        )


@pytest.mark.parametrize("n", [1, 2, 7, 40, 200])
def test_tango_statistic_follows_the_zero_denominator_rule_at_the_ends(n: int) -> None:
    # Contract 3.3: "When the denominator is zero, Z(D) is 0 if the numerator is 0, and
    # otherwise +inf or -inf with the numerator's sign." The denominator vanishes at
    # D = 1 and D = -1 always, and at D = 0 when there is no discordance. These are pinned
    # by rule, so they are compared exactly, infinities included.
    for favour_candidate, favour_baseline in {(0, 0), (1, 0), (0, 1), (n // 2, n - n // 2)}:
        if favour_candidate + favour_baseline > n:
            continue
        at_one = stats.tango_statistic(favour_candidate, favour_baseline, n, 1.0)
        at_minus_one = stats.tango_statistic(favour_candidate, favour_baseline, n, -1.0)
        # Numerator at D = 1 is b - c - n, zero only when b = n.
        expected_one = 0.0 if favour_candidate == n else -math.inf
        # Numerator at D = -1 is b - c + n, zero only when c = n.
        expected_minus_one = 0.0 if favour_baseline == n else math.inf
        assert at_one == expected_one, f"Z(1) for b={favour_candidate}, c={favour_baseline}"
        assert at_minus_one == expected_minus_one, (
            f"Z(-1) for b={favour_candidate}, c={favour_baseline}"
        )

    assert stats.tango_statistic(n, 0, n, 1.0) == 0.0
    assert stats.tango_statistic(0, n, n, -1.0) == 0.0
    assert stats.tango_statistic(0, 0, n, 0.0) == 0.0


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_tango_interval_matches_the_inverted_reference(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float
) -> None:
    # Contract 3.3 end to end: the set of D the score test does not reject, found here by
    # scanning the reference statistic and refining each crossing with brentq.
    expected_low, expected_high, _ = _tango_interval_reference(
        favour_candidate, favour_baseline, n, confidence
    )
    low, high = stats.tango_interval(favour_candidate, favour_baseline, n, confidence=confidence)

    assert _close(low, expected_low, _EXACT_TOLERANCE), (
        f"low for b={favour_candidate}, c={favour_baseline}, n={n} at {confidence}: "
        f"{low!r} != reference {expected_low!r}"
    )
    assert _close(high, expected_high, _EXACT_TOLERANCE), (
        f"high for b={favour_candidate}, c={favour_baseline}, n={n} at {confidence}: "
        f"{high!r} != reference {expected_high!r}"
    )


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
@pytest.mark.parametrize("confidence", [0.80, 0.95])
def test_paired_binary_interval_matches_the_tango_reference(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float
) -> None:
    # The same interval reached from boolean samples rather than counts, so a transposed
    # count (b and c swapped) or a concordant item leaking into the statistic is visible.
    baseline, candidate = _binary_from_counts(favour_candidate, favour_baseline, n, seed=n)
    assert _counts_of(baseline, candidate) == (favour_candidate, favour_baseline)
    expected_low, expected_high, _ = _tango_interval_reference(
        favour_candidate, favour_baseline, n, confidence
    )

    result = stats.paired_binary(baseline, candidate, tolerance=_DELTA, confidence=confidence)

    assert _close(result.interval.low, expected_low, _EXACT_TOLERANCE), (
        f"low {result.interval.low!r} != reference {expected_low!r}"
    )
    assert _close(result.interval.high, expected_high, _EXACT_TOLERANCE), (
        f"high {result.interval.high!r} != reference {expected_high!r}"
    )
    assert _close(
        result.interval.point, (favour_candidate - favour_baseline) / n, _ARITHMETIC_TOLERANCE
    )
    assert result.method == "tango_score"
    assert result.n == n
    assert result.interval.confidence == confidence
    assert result.tolerance == _DELTA


@pytest.mark.parametrize("n", [1, 5, 20, 64])
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_tango_interval_with_no_discordant_items_is_the_contract_closed_form(
    n: int, confidence: float
) -> None:
    # Contract 3.4: with discordant = 0 the interval is [-z^2/(n+z^2), +z^2/(n+z^2)], and
    # never [0, 0], which would make every such slice non_inferior by arithmetic. The
    # contract forbids a special case, so the module is expected to reach this by the same
    # root-find as everywhere else; hence the root-find tolerance rather than exact equality.
    critical = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    half_width = critical**2 / (n + critical**2)
    baseline = [True] * (n // 2) + [False] * (n - n // 2)
    candidate = list(baseline)

    low, high = stats.tango_interval(0, 0, n, confidence=confidence)
    result = stats.paired_binary(baseline, candidate, tolerance=_DELTA, confidence=confidence)

    for got_low, got_high in ((low, high), (result.interval.low, result.interval.high)):
        assert _close(got_low, -half_width, _EXACT_TOLERANCE), f"low {got_low!r} != {-half_width!r}"
        assert _close(got_high, half_width, _EXACT_TOLERANCE), (
            f"high {got_high!r} != {half_width!r}"
        )
        assert got_low < 0.0 < got_high
    assert result.interval.point == 0.0


@pytest.mark.parametrize("n", [1, 3, 10, 50, 200])
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_tango_interval_reaches_the_end_when_every_item_favours_one_side(
    n: int, confidence: float
) -> None:
    # Contract 3.4: c = n gives low = -1, "nothing in the data excludes the worst case",
    # and b = n gives high = 1. Z(-1) is 0 by rule there, so -1 is itself accepted and is
    # exactly the smallest accepted D: pinned exactly, as Clopper-Pearson's ends are.
    worst_low, worst_high = stats.tango_interval(0, n, n, confidence=confidence)
    best_low, best_high = stats.tango_interval(n, 0, n, confidence=confidence)

    assert worst_low == -1.0, f"c=n={n}: low {worst_low!r} is not -1"
    assert best_high == 1.0, f"b=n={n}: high {best_high!r} is not 1"
    # The other end comes from the ordinary inversion and must match the reference.
    assert _close(worst_high, _tango_interval_reference(0, n, n, confidence)[1], _EXACT_TOLERANCE)
    assert _close(best_low, _tango_interval_reference(n, 0, n, confidence)[0], _EXACT_TOLERANCE)

    worst = stats.paired_binary([True] * n, [False] * n, tolerance=_DELTA, confidence=confidence)
    best = stats.paired_binary([False] * n, [True] * n, tolerance=_DELTA, confidence=confidence)
    assert worst.interval.low == -1.0
    assert best.interval.high == 1.0


# ---------------------------------------------------------------------------
# 4. Binary margin p-values (contract 3.5) and invariant 8 for binary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
def test_paired_binary_margin_p_values_match_the_reference(
    favour_candidate: int, favour_baseline: int, n: int
) -> None:
    # p_non_inferior = 1 - Phi(Z(-delta)), p_regressed = Phi(Z(-delta)), with the
    # reference Z and scipy's normal. These are what the family correction reads, so a
    # p-value on the wrong tail would turn Holm's control into nonsense.
    baseline, candidate = _binary_from_counts(favour_candidate, favour_baseline, n, seed=7)
    for delta in _MARGINS:
        statistic = _tango_z_at(favour_candidate, favour_baseline, n, -delta)
        expected_ni = _normal_sf(statistic)
        expected_reg = _normal_cdf(statistic)

        result = stats.paired_binary(baseline, candidate, tolerance=delta)

        assert _close(result.p_non_inferior, expected_ni, _EXACT_TOLERANCE), (
            f"delta={delta}: p_non_inferior {result.p_non_inferior!r} != {expected_ni!r}"
        )
        assert _close(result.p_regressed, expected_reg, _EXACT_TOLERANCE), (
            f"delta={delta}: p_regressed {result.p_regressed!r} != {expected_reg!r}"
        )
        assert _close(result.p_non_inferior + result.p_regressed, 1.0, _ARITHMETIC_TOLERANCE)
        assert result.tolerance == delta


def _invariant_8_holds(result: stats.PairedResult) -> tuple[bool, bool]:
    """Invariant 8's two equivalences for one result."""
    alpha = 1.0 - result.interval.confidence
    delta = result.tolerance
    lower_side = (result.interval.low > -delta) == (result.p_non_inferior < alpha / 2.0)
    upper_side = (result.interval.high < -delta) == (result.p_regressed < alpha / 2.0)
    return lower_side, upper_side


def _margins_around(low: float, high: float) -> list[float]:
    """Tolerances placing -delta 1e-6 either side of each interval end, kept in (0, 1]."""
    offset = 1e-6
    candidates = [-low - offset, -low + offset, -high - offset, -high + offset]
    return [delta for delta in candidates if 0.0 < delta <= 1.0]


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"), _TANGO_COUNTS, ids=_TANGO_IDS
)
@pytest.mark.parametrize("confidence", [0.90, 0.95])
def test_invariant_8_binary_p_values_agree_with_the_interval(
    favour_candidate: int, favour_baseline: int, n: int, confidence: float
) -> None:
    # Invariant 8: low > -delta exactly when p_non_inferior < alpha/2, and high < -delta
    # exactly when p_regressed < alpha/2. Contract 3.5 says they "agree with the interval
    # exactly", which is the property that lets section 6.2 read a p-value without ever
    # contradicting the interval. The near-margin tolerances put -delta 1e-6 from an end,
    # which is where an off-by-a-tail or a different critical value would show first.
    baseline, candidate = _binary_from_counts(favour_candidate, favour_baseline, n, seed=3)
    first = stats.paired_binary(baseline, candidate, tolerance=_DELTA, confidence=confidence)
    margins = [*_MARGINS, *_margins_around(first.interval.low, first.interval.high)]

    for delta in margins:
        result = stats.paired_binary(baseline, candidate, tolerance=delta, confidence=confidence)
        lower_side, upper_side = _invariant_8_holds(result)
        assert lower_side, (
            f"delta={delta}: low={result.interval.low!r}, p_ni={result.p_non_inferior!r}"
        )
        assert upper_side, (
            f"delta={delta}: high={result.interval.high!r}, p_reg={result.p_regressed!r}"
        )


# ---------------------------------------------------------------------------
# 5. Holm and Benjamini-Hochberg (contract 5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "p_values"), _P_VALUE_CASES, ids=[c[0] for c in _P_VALUE_CASES])
def test_holm_matches_statsmodels(name: str, p_values: tuple[float, ...]) -> None:
    # Holm controls the family-wise error rate for the gate slices. Getting it wrong
    # changes which slices are allowed to fail a build.
    expected = _multipletests(p_values, "holm")
    actual = stats.holm(p_values)

    assert len(actual) == len(expected), f"{name}: wrong length"
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert _close(got, want, _EXACT_TOLERANCE), (
            f"{name}: position {index} is {got!r}, statsmodels says {want!r}"
        )


@pytest.mark.parametrize(("name", "p_values"), _P_VALUE_CASES, ids=[c[0] for c in _P_VALUE_CASES])
def test_benjamini_hochberg_matches_statsmodels(name: str, p_values: tuple[float, ...]) -> None:
    # Benjamini-Hochberg controls the false discovery rate for the watch slices. The
    # two corrections must not be swapped: BH is uniformly less conservative.
    expected = _multipletests(p_values, "fdr_bh")
    actual = stats.benjamini_hochberg(p_values)

    assert len(actual) == len(expected), f"{name}: wrong length"
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert _close(got, want, _EXACT_TOLERANCE), (
            f"{name}: position {index} is {got!r}, statsmodels says {want!r}"
        )


def test_both_corrections_return_empty_for_empty_input() -> None:
    # Contract 5 says an empty input returns empty. statsmodels raises ZeroDivisionError
    # on an empty vector, so it cannot be the reference here; the contract is.
    assert stats.holm([]) == []
    assert stats.benjamini_hochberg([]) == []


@pytest.mark.parametrize("p_value", [0.0, 0.031, 0.5, 0.999, 1.0])
def test_a_single_p_value_is_returned_unchanged(p_value: float) -> None:
    # With m = 1 both formulas reduce to min(1, p). A multiplier that used m + 1, or
    # that counted from zero, would show up here before anywhere else.
    assert _close(stats.holm([p_value])[0], min(1.0, p_value), _ARITHMETIC_TOLERANCE)
    assert _close(stats.benjamini_hochberg([p_value])[0], min(1.0, p_value), _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize(("name", "p_values"), _P_VALUE_CASES, ids=[c[0] for c in _P_VALUE_CASES])
def test_adjustment_follows_a_permutation_of_the_input(
    name: str, p_values: tuple[float, ...]
) -> None:
    # Returning sorted output instead of input order is the single most likely bug in
    # either correction, and it is invisible on an already-sorted example. Permuting the
    # input must permute the output identically; nothing else can satisfy this.
    if len(p_values) < 2:
        pytest.skip("a permutation test needs at least two values")
    order = list(np.random.default_rng(99).permutation(len(p_values)))
    permuted = [p_values[index] for index in order]

    for adjust in (stats.holm, stats.benjamini_hochberg):
        direct = adjust(p_values)
        reordered = adjust(permuted)
        expected = [direct[index] for index in order]
        for position, (got, want) in enumerate(zip(reordered, expected, strict=True)):
            assert _close(got, want, _ARITHMETIC_TOLERANCE), (
                f"{name}: {adjust.__name__} lost input order at position {position}"
            )


@pytest.mark.parametrize(("name", "p_values"), _P_VALUE_CASES, ids=[c[0] for c in _P_VALUE_CASES])
def test_adjusted_p_values_are_never_smaller_than_the_raw_ones(
    name: str, p_values: tuple[float, ...]
) -> None:
    # Invariant 6. An adjustment that made a p-value smaller would make a correction
    # that is supposed to be conservative into one that invents significance.
    for adjust in (stats.holm, stats.benjamini_hochberg):
        for raw, adjusted in zip(p_values, adjust(p_values), strict=True):
            assert adjusted >= raw - _ARITHMETIC_TOLERANCE, (
                f"{name}: {adjust.__name__} shrank {raw}"
            )
            assert adjusted <= 1.0, f"{name}: {adjust.__name__} returned {adjusted} > 1"


@pytest.mark.parametrize(("name", "p_values"), _P_VALUE_CASES, ids=[c[0] for c in _P_VALUE_CASES])
def test_adjusted_p_values_are_monotone_in_the_sorted_order(
    name: str, p_values: tuple[float, ...]
) -> None:
    # Contract 5: "both are monotone after adjustment". Checked in the sorted order,
    # because that is the order the monotonicity is stated in.
    for adjust in (stats.holm, stats.benjamini_hochberg):
        adjusted = adjust(p_values)
        pairs = sorted(zip(p_values, adjusted, strict=True))
        values = [value for _, value in pairs]
        for earlier, later in pairwise(values):
            assert later >= earlier - _ARITHMETIC_TOLERANCE, f"{name}: {adjust.__name__}"


def test_holm_is_never_less_conservative_than_benjamini_hochberg() -> None:
    # Family-wise control is strictly stronger than false-discovery control, so mixing
    # the two up shows here as well as in the statsmodels comparisons.
    rng = np.random.default_rng(31337)
    for _ in range(50):
        p_values = [float(value) for value in rng.random(8)]
        for holm_value, bh_value in zip(
            stats.holm(p_values), stats.benjamini_hochberg(p_values), strict=True
        ):
            assert holm_value >= bh_value - _ARITHMETIC_TOLERANCE


# ---------------------------------------------------------------------------
# 6. Jackknife acceleration (contract 4.2 and 4.7).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 8, 13, 21])
@pytest.mark.parametrize("count", [3, 7, 40])
def test_jackknife_acceleration_matches_the_contract_formula(seed: int, count: int) -> None:
    # Contract 4.2 is explicit that `a` is deterministic and "must agree with an
    # independent computation exactly, not approximately". A wrong `a` tilts both BCa
    # endpoints in the same direction, which is exactly the error a bootstrap comparison
    # with Monte Carlo noise in it is worst at seeing.
    rng = np.random.default_rng(seed)
    values = [float(value) for value in rng.normal(0.1, 0.4, count)]

    expected = _jackknife_acceleration_reference(values)
    actual = stats.jackknife_acceleration(values)

    assert _close(actual, expected, _ARITHMETIC_TOLERANCE * max(1.0, abs(expected))), (
        f"acceleration for seed {seed}, n={count}: {actual!r} != {expected!r}"
    )


@pytest.mark.parametrize("seed", [4, 6, 9, 11])
@pytest.mark.parametrize("count", [4, 12, 65])
def test_jackknife_acceleration_matches_the_closed_form(seed: int, count: int) -> None:
    # The same quantity by a second derivation, so a mistake in this file's own
    # leave-one-out loop cannot quietly become the standard the module is held to.
    rng = np.random.default_rng(seed)
    values = [float(value) for value in rng.gamma(2.0, 0.3, count)]

    expected = _jackknife_acceleration_closed_form(values)
    assert _close(
        _jackknife_acceleration_reference(values),
        expected,
        _ARITHMETIC_TOLERANCE * max(1.0, abs(expected)),
    )
    assert _close(
        stats.jackknife_acceleration(values),
        expected,
        _ARITHMETIC_TOLERANCE * max(1.0, abs(expected)),
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [-1.5, -1.5, -1.5, -1.5, -1.5],
    ],
)
def test_jackknife_acceleration_is_zero_when_every_value_is_identical(
    values: list[float],
) -> None:
    # Contract 4.7 defines `a = 0` here rather than leaving 0/0. A NaN would propagate
    # into the adjusted percentiles and turn the interval into two NaNs, which the
    # verdict rule would then compare against a tolerance and silently mis-answer.
    assert stats.jackknife_acceleration(values) == 0.0


@pytest.mark.parametrize(
    "values",
    [
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        [0.5, 1.5, 2.5, 3.5],
        [-0.25, 0.25],
    ],
)
def test_jackknife_acceleration_is_zero_for_a_symmetric_sample(values: list[float]) -> None:
    # The numerator is a sum of cubes of centred deviations, so a symmetric sample has
    # `a = 0`. This catches an implementation that squares where it should cube.
    assert _close(stats.jackknife_acceleration(values), 0.0, _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize("seed", [17, 23, 29])
def test_jackknife_acceleration_negates_with_the_data(seed: int) -> None:
    # `a` is an odd function of the centred data, so negating every value negates it.
    # A missing sign here is a tail-direction error in the BCa interval.
    rng = np.random.default_rng(seed)
    values = [float(value) for value in rng.gamma(1.5, 0.2, 25)]
    negated = [-value for value in values]

    forward = stats.jackknife_acceleration(values)
    backward = stats.jackknife_acceleration(negated)
    assert _close(backward, -forward, _ARITHMETIC_TOLERANCE * max(1.0, abs(forward)))


# ---------------------------------------------------------------------------
# 7. Paired binary basics (contract 3.1, 3.2, 3.4).
# ---------------------------------------------------------------------------

_BINARY_SAMPLES: tuple[tuple[int, int, float], ...] = (
    (11, 40, -0.12),
    (12, 40, 0.10),
    (13, 20, -0.25),
    (14, 97, 0.03),
    (15, 7, -0.4),
    (16, 200, 0.0),
)


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
def test_paired_binary_point_equals_the_mean_difference(
    seed: int, count: int, drift: float
) -> None:
    # Invariant 2, stated in the contract as an exact identity: the discordant-count
    # form and the mean-difference form are the same number.
    baseline, candidate = _binary_sample(seed, count, drift)
    expected = float(np.mean(candidate)) - float(np.mean(baseline))

    result = stats.paired_binary(baseline, candidate, tolerance=_DELTA)

    assert _close(result.interval.point, expected, _ARITHMETIC_TOLERANCE), (
        f"point {result.interval.point!r} != mean(candidate) - mean(baseline) {expected!r}"
    )


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
@pytest.mark.parametrize("confidence", [0.80, 0.95])
def test_paired_binary_random_samples_match_the_tango_reference(
    seed: int, count: int, drift: float, confidence: float
) -> None:
    # The Tango comparison again, on samples whose counts were not chosen by hand.
    baseline, candidate = _binary_sample(seed, count, drift)
    favour_candidate, favour_baseline = _counts_of(baseline, candidate)
    expected_low, expected_high, _ = _tango_interval_reference(
        favour_candidate, favour_baseline, count, confidence
    )

    result = stats.paired_binary(baseline, candidate, tolerance=_DELTA, confidence=confidence)

    assert _close(result.interval.low, expected_low, _EXACT_TOLERANCE)
    assert _close(result.interval.high, expected_high, _EXACT_TOLERANCE)


def test_paired_result_has_exactly_the_contract_fields() -> None:
    # Contract 7. McNemar is gone (contract 3.5) and so is its `p_value`; the two margin
    # p-values and the tolerance they were computed at replace it. A stale field would
    # be a number in a report that nothing computes the way the contract says.
    paired = {field.name for field in dataclasses.fields(stats.PairedResult)}
    interval = {field.name for field in dataclasses.fields(stats.Interval)}
    assert paired == {"n", "interval", "method", "tolerance", "p_non_inferior", "p_regressed"}
    assert interval == {"point", "low", "high", "confidence"}


def test_paired_binary_rejects_an_empty_sample() -> None:
    # Contract 3.4: "n = 0. There is no estimate. Raise, rather than return a number
    # that means nothing." Contract 7a pins the exception type to ValueError.
    with pytest.raises(ValueError):
        stats.paired_binary([], [], tolerance=_DELTA)


def test_paired_binary_rejects_mismatched_lengths() -> None:
    # Pairing is by item id (contract 2). Two vectors of different lengths cannot be
    # paired, and silently truncating to the shorter one would drop real items.
    with pytest.raises(ValueError):
        stats.paired_binary([True, False, True], [True, False], tolerance=_DELTA)


# ---------------------------------------------------------------------------
# 8. Few items moved: the count bound (contract 4.4).
# ---------------------------------------------------------------------------

#: (n, up, down) with fewer than 8 moved: nothing moved, one direction only, both, the
#: largest count-bound case (7), and every item moved in a slice smaller than 8.
_FEW_MOVED_CASES: tuple[tuple[int, int, int], ...] = (
    (2, 0, 0),
    (10, 0, 0),
    (40, 0, 0),
    (40, 1, 0),
    (40, 0, 1),
    (40, 3, 2),
    (40, 2, 5),
    (40, 7, 0),
    (40, 0, 7),
    (40, 4, 3),
    (12, 3, 3),
    (100, 1, 6),
    (5, 5, 0),
    (6, 0, 6),
)

_FEW_MOVED_IDS: list[str] = [f"n{n}-up{up}-down{down}" for n, up, down in _FEW_MOVED_CASES]


@pytest.mark.parametrize(("count", "up", "down"), _FEW_MOVED_CASES, ids=_FEW_MOVED_IDS)
@pytest.mark.parametrize("confidence", [0.90, 0.95])
def test_count_bound_interval_is_minus_u_down_to_u_up(
    count: int, up: int, down: int, confidence: float
) -> None:
    # Contract 4.4: with fewer than 8 moved items the interval is not bootstrapped but is
    # [-U(down, n), +U(up, n)], U the two-sided Clopper-Pearson upper bound. The bootstrap
    # it replaces reported non_inferior at true -delta up to every time on such slices.
    baseline, candidate = _few_moved_sample(count, up, down, seed=count + 10 * up + down)
    differences = _differences(baseline, candidate)
    assert int(np.count_nonzero(differences > 0)) == up
    assert int(np.count_nonzero(differences < 0)) == down

    result = stats.paired_continuous(
        baseline, candidate, tolerance=_DELTA, confidence=confidence, seed=1
    )

    expected_low = -_clopper_pearson_upper_beta(down, count, confidence)
    expected_high = _clopper_pearson_upper_beta(up, count, confidence)
    assert result.method == "count_bound"
    assert _close(result.interval.low, expected_low, _EXACT_TOLERANCE), (
        f"low {result.interval.low!r} != -U({down}, {count}) = {expected_low!r}"
    )
    assert _close(result.interval.high, expected_high, _EXACT_TOLERANCE), (
        f"high {result.interval.high!r} != U({up}, {count}) = {expected_high!r}"
    )
    assert _close(result.interval.point, float(np.mean(differences)), _ARITHMETIC_TOLERANCE)
    assert result.interval.low <= result.interval.point <= result.interval.high
    assert result.interval.confidence == confidence
    assert result.n == count


@pytest.mark.parametrize("count", [2, 3, 8, 20, 40, 200])
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_count_bound_with_nothing_moved_is_symmetric_and_never_a_point(
    count: int, confidence: float
) -> None:
    # Contract 4.4 with up = down = 0: [-u, u] with u = U(0, n) = 1 - (alpha/2)^(1/n).
    # BCa reported [0, 0] here and called every such slice non_inferior. Written from the
    # closed form, so it also checks the beta-quantile helper used elsewhere.
    alpha = 1.0 - confidence
    bound = 1.0 - (alpha / 2.0) ** (1.0 / count)
    assert _close(bound, _clopper_pearson_upper_beta(0, count, confidence), _ARITHMETIC_TOLERANCE)
    scores = [float(value) for value in np.linspace(0.0, 1.0, count)]

    result = stats.paired_continuous(
        scores, list(scores), tolerance=_DELTA, confidence=confidence, seed=1
    )

    assert result.method == "count_bound"
    assert result.interval.point == 0.0
    assert _close(result.interval.low, -bound, _EXACT_TOLERANCE)
    assert _close(result.interval.high, bound, _EXACT_TOLERANCE)
    assert result.interval.low < 0.0 < result.interval.high


@pytest.mark.parametrize(("count", "up", "down"), _FEW_MOVED_CASES, ids=_FEW_MOVED_IDS)
def test_count_bound_margin_p_values(count: int, up: int, down: int) -> None:
    # p_non_inferior = P(Binomial(n, delta) <= down), p_regressed = 1 (high >= 0 > -delta,
    # so a slice where almost nothing moved can never be shown to have regressed).
    baseline, candidate = _few_moved_sample(count, up, down, seed=count + 10 * up + down)
    for delta in (0.01, 0.05, 0.1, 0.3, 0.75):
        result = stats.paired_continuous(baseline, candidate, tolerance=delta, seed=1)
        expected = _binomial_cdf(down, count, delta)
        assert _close(result.p_non_inferior, expected, _EXACT_TOLERANCE), (
            f"delta={delta}: p_non_inferior {result.p_non_inferior!r} != {expected!r}"
        )
        assert result.p_regressed == 1.0
        assert result.tolerance == delta


@pytest.mark.parametrize(("count", "up", "down"), _FEW_MOVED_CASES, ids=_FEW_MOVED_IDS)
def test_invariant_8_count_bound_p_values_agree_with_the_interval(
    count: int, up: int, down: int
) -> None:
    # Invariant 8 for the count bound, including tolerances 1e-6 either side of U(down, n),
    # where the binomial tail crosses alpha/2 exactly as the Clopper-Pearson bound does.
    baseline, candidate = _few_moved_sample(count, up, down, seed=count + 10 * up + down)
    margin = _clopper_pearson_upper_beta(down, count, 0.95)
    deltas = [0.01, 0.05, 0.2, 0.5]
    if margin < 1.0 - 1e-6:
        deltas += [margin - 1e-6, margin + 1e-6]
    for delta in deltas:
        result = stats.paired_continuous(baseline, candidate, tolerance=delta, seed=1)
        lower_side, upper_side = _invariant_8_holds(result)
        assert lower_side, (
            f"delta={delta}: low={result.interval.low!r}, p_ni={result.p_non_inferior!r}"
        )
        assert upper_side, (
            f"delta={delta}: high={result.interval.high!r}, p_reg={result.p_regressed!r}"
        )


def test_the_threshold_is_eight_moved_items() -> None:
    # Contract 4.4: "when fewer than 8 items moved" the count bound applies; 4.2: the
    # bootstrap is used "when at least 8 items moved and their differences are not all
    # the same". Seven moved items is the last count-bound case, eight the first bootstrap.
    seven = _few_moved_sample(40, 4, 3, seed=70)
    eight = _few_moved_sample(40, 5, 3, seed=80)
    assert int(np.count_nonzero(_differences(*seven))) == 7
    assert _takes_bootstrap_branch(*eight)

    at_seven = stats.paired_continuous(*seven, tolerance=_DELTA, resamples=2_000, seed=1)
    at_eight = stats.paired_continuous(*eight, tolerance=_DELTA, resamples=2_000, seed=1)

    assert at_seven.method == "count_bound"
    assert at_eight.method == "bootstrap_bca"


# ---------------------------------------------------------------------------
# 9. Every item moved by the same amount (contract 4.5).
# ---------------------------------------------------------------------------

#: (n, shift): both signs, the two extreme shifts, and the smallest slice (8) it can apply to.
_CONSTANT_SHIFTS: tuple[tuple[int, float], ...] = (
    (8, 0.25),
    (8, -0.25),
    (12, -0.5),
    (40, 0.125),
    (40, -0.0625),
    (50, -0.75),
    (20, 1.0),
    (20, -1.0),
)


def _constant_shift_expectation(
    shift: float, count: int, delta: float, confidence: float
) -> tuple[float, float, float, float]:
    """Contract 4.5's (low, high, p_non_inferior, p_regressed), written out literally."""
    alpha = 1.0 - confidence
    bound = 1.0 - (alpha / 2.0) ** (1.0 / count)
    low = shift - bound * (1.0 + shift)
    high = shift + bound * (1.0 - shift)
    above = (1.0 - (shift + delta) / (1.0 + shift)) ** count if shift > -delta else 1.0
    below = (1.0 - (-delta - shift) / (1.0 - shift)) ** count if shift < -delta else 1.0
    # p_non_inferior is 1 when v <= -delta, p_regressed is 1 when v >= -delta.
    return low, high, above, below


@pytest.mark.parametrize(("count", "shift"), _CONSTANT_SHIFTS)
@pytest.mark.parametrize("confidence", [0.90, 0.95])
def test_constant_shift_interval_and_p_values(count: int, shift: float, confidence: float) -> None:
    # Contract 4.5: a resample has nothing to vary, and [v, v] would claim the data rule
    # out every other difference. The interval is v - u(1 + v) .. v + u(1 - v) with
    # u = 1 - (alpha/2)^(1/n). The values are dyadic, so the differences are exactly
    # equal and the branch is reached without leaning on the tolerance rule.
    baseline, candidate = _constant_shift_sample(count, shift, seed=count)
    assert len(set(_differences(baseline, candidate).tolist())) == 1

    for delta in (0.05, 0.25, 0.3, 0.6):
        result = stats.paired_continuous(
            baseline, candidate, tolerance=delta, confidence=confidence, seed=1
        )
        low, high, p_ni, p_reg = _constant_shift_expectation(shift, count, delta, confidence)

        assert result.method == "constant_shift"
        assert _close(result.interval.point, shift, _ARITHMETIC_TOLERANCE)
        assert _close(result.interval.low, low, _EXACT_TOLERANCE), (
            f"low {result.interval.low!r} != {low!r}"
        )
        assert _close(result.interval.high, high, _EXACT_TOLERANCE), (
            f"high {result.interval.high!r} != {high!r}"
        )
        assert _close(result.p_non_inferior, p_ni, _EXACT_TOLERANCE), (
            f"delta={delta}: p_non_inferior {result.p_non_inferior!r} != {p_ni!r}"
        )
        assert _close(result.p_regressed, p_reg, _EXACT_TOLERANCE), (
            f"delta={delta}: p_regressed {result.p_regressed!r} != {p_reg!r}"
        )
        assert result.interval.low < result.interval.high
        assert result.interval.low <= result.interval.point <= result.interval.high


@pytest.mark.parametrize(("shift", "seed"), [(0.1, 61), (-0.3, 62)])
def test_constant_shift_tolerates_floating_point_disagreement(shift: float, seed: int) -> None:
    # Contract 4.5: "The same value" is judged by max - min <= 1e-12 * max(|mean|, 1),
    # "not by exact equality: a constant shift computed in floating point leaves
    # differences that disagree in the last bit". These samples are built so that they
    # do disagree in the last bit, and must still take the constant-shift branch.
    rng = np.random.default_rng(seed)
    floor = max(0.0, -shift) + 0.01
    ceiling = min(1.0, 1.0 - shift) - 0.01
    baseline = [float(value) for value in rng.uniform(floor, ceiling, 30)]
    candidate = [value + shift for value in baseline]
    differences = _differences(baseline, candidate)
    spread = float(np.max(differences) - np.min(differences))
    assert spread > 0.0, "the sample must exercise the tolerance, not exact equality"
    assert spread <= 1e-12 * max(abs(float(np.mean(differences))), 1.0)

    result = stats.paired_continuous(baseline, candidate, tolerance=0.2, seed=1)

    low, high, p_ni, p_reg = _constant_shift_expectation(float(np.mean(differences)), 30, 0.2, 0.95)
    assert result.method == "constant_shift"
    assert _close(result.interval.low, low, _EXACT_TOLERANCE)
    assert _close(result.interval.high, high, _EXACT_TOLERANCE)
    assert _close(result.p_non_inferior, p_ni, _EXACT_TOLERANCE)
    assert _close(result.p_regressed, p_reg, _EXACT_TOLERANCE)


@pytest.mark.parametrize(("count", "shift"), _CONSTANT_SHIFTS)
def test_invariant_8_constant_shift_p_values_agree_with_the_interval(
    count: int, shift: float
) -> None:
    # Invariant 8 for the constant shift, with tolerances 1e-6 either side of each end.
    baseline, candidate = _constant_shift_sample(count, shift, seed=count)
    first = stats.paired_continuous(baseline, candidate, tolerance=_DELTA, seed=1)
    deltas = [0.01, 0.05, 0.25, 0.5, *_margins_around(first.interval.low, first.interval.high)]
    for delta in deltas:
        result = stats.paired_continuous(baseline, candidate, tolerance=delta, seed=1)
        lower_side, upper_side = _invariant_8_holds(result)
        assert lower_side, (
            f"delta={delta}: low={result.interval.low!r}, p_ni={result.p_non_inferior!r}"
        )
        assert upper_side, (
            f"delta={delta}: high={result.interval.high!r}, p_reg={result.p_regressed!r}"
        )


# ---------------------------------------------------------------------------
# 10. BCa bootstrap (contract 4.2, 4.6, 4.7).
# ---------------------------------------------------------------------------

#: Both sides draw their own resamples, so exact agreement is impossible and the only
#: honest comparison is one with a stated Monte Carlo budget.
#:
#: Measured on the four samples below, with scipy against itself across 15 reference
#: seeds at 50_000 resamples: the per-endpoint standard deviation is at most 0.006 of the
#: interval width and the full observed range is under 0.024 of it. The difference
#: between two *independent* runs therefore has a standard deviation near 0.008 of the
#: width, so 0.05 of the width sits at roughly six standard deviations -- a false
#: failure is a one-in-a-million event, not a flake that gets re-run.
#:
#: It is not so loose that the test cannot fail. On `gamma` and `lognormal`, replacing
#: BCa with the plain percentile interval moves an endpoint by 0.056 to 0.15 of the
#: width (measured with scipy's own percentile method); dropping the acceleration,
#: flipping the sign of the bias correction, or swapping the two adjusted percentiles move
#: it further still, and a sign error in the point estimate moves it by more than a whole
#: width. On a symmetric sample BCa and percentile genuinely coincide, which is a property
#: of the estimator rather than a weakness of this number: `normal` is in the grid for
#: breadth, and the two skewed samples are what give this comparison its power.
_BCA_WIDTH_TOLERANCE = 0.05

#: High enough that the Monte Carlo error above is small next to a real mistake.
_BCA_RESAMPLES = 50_000


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_bootstrap_samples_satisfy_the_bootstrap_precondition(
    shape: str, count: int, seed: int
) -> None:
    # Contract 4.2 applies only to bounded scores with at least 8 moved items that are
    # not all identical. A sample that drifted into another branch would make every BCa
    # comparison below a comparison against the wrong method.
    baseline, candidate = _continuous_sample(shape, count, seed)
    assert _takes_bootstrap_branch(baseline, candidate), f"{shape} is not a bootstrap sample"


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_paired_continuous_bca_matches_scipy(shape: str, count: int, seed: int) -> None:
    baseline, candidate = _continuous_sample(shape, count, seed)
    differences = _differences(baseline, candidate)
    expected_low, expected_high = _bca_scipy(differences, 0.95, _BCA_RESAMPLES, seed + 7)
    width = expected_high - expected_low
    tolerance = _BCA_WIDTH_TOLERANCE * width

    result = stats.paired_continuous(
        baseline,
        candidate,
        tolerance=_DELTA,
        confidence=0.95,
        resamples=_BCA_RESAMPLES,
        seed=seed,
    )

    assert result.method == "bootstrap_bca"
    assert _close(result.interval.low, expected_low, tolerance), (
        f"{shape}: low {result.interval.low!r} vs scipy {expected_low!r} "
        f"({abs(result.interval.low - expected_low) / width:.4f} of the width)"
    )
    assert _close(result.interval.high, expected_high, tolerance), (
        f"{shape}: high {result.interval.high!r} vs scipy {expected_high!r} "
        f"({abs(result.interval.high - expected_high) / width:.4f} of the width)"
    )


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
@pytest.mark.parametrize("confidence", [0.80, 0.99])
def test_paired_continuous_bca_matches_scipy_at_other_confidences(
    shape: str, count: int, seed: int, confidence: float
) -> None:
    # The adjusted percentiles depend on the confidence through `z_alpha`, so a sign or
    # tail mix-up can hide at 95% and appear at 80% or 99%.
    baseline, candidate = _continuous_sample(shape, count, seed)
    differences = _differences(baseline, candidate)
    expected_low, expected_high = _bca_scipy(differences, confidence, _BCA_RESAMPLES, seed + 19)
    tolerance = _BCA_WIDTH_TOLERANCE * (expected_high - expected_low)

    result = stats.paired_continuous(
        baseline,
        candidate,
        tolerance=_DELTA,
        confidence=confidence,
        resamples=_BCA_RESAMPLES,
        seed=seed,
    )

    assert _close(result.interval.low, expected_low, tolerance)
    assert _close(result.interval.high, expected_high, tolerance)


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
@pytest.mark.parametrize("level", [0.80, 0.90])
def test_bca_p_non_inferior_is_the_level_where_the_lower_bound_meets_the_margin(
    shape: str, count: int, seed: int, level: float
) -> None:
    # Contract 4.6: "p_non_inferior is the one-sided level at which the BCa lower bound
    # would sit exactly on -delta". So scipy's BCa interval at two-sided confidence
    # 1 - 2 * p_non_inferior must have its lower bound on -delta.
    #
    # delta is taken from scipy's own lower bound at `level`, so p_non_inferior should
    # come out near (1 - level) / 2 and -delta sits well inside the resample distribution.
    # The tolerance is the same 0.05 of the 95% width as the interval comparisons, and for
    # the same reason: three independent resample streams (the delta, the module's p-value
    # and the check) each carry the Monte Carlo error measured above. Emulating contract 4.6
    # with numpy resamples and checking it this way against scipy, the worst disagreement
    # over six seeds per case was 0.013 of the width.
    baseline, candidate = _continuous_sample(shape, count, seed)
    differences = _differences(baseline, candidate)
    reference_low, _ = _bca_scipy(differences, level, _BCA_RESAMPLES, seed + 31)
    assert reference_low < 0.0, "delta must be positive"
    delta = -reference_low
    low_95, high_95 = _bca_scipy(differences, 0.95, _BCA_RESAMPLES, seed + 37)
    tolerance = _BCA_WIDTH_TOLERANCE * (high_95 - low_95)

    result = stats.paired_continuous(
        baseline, candidate, tolerance=delta, resamples=_BCA_RESAMPLES, seed=seed
    )
    p_non_inferior = result.p_non_inferior
    assert 0.0 < p_non_inferior < 0.5, f"{shape}: p_non_inferior {p_non_inferior!r}"

    check_low, _ = _bca_scipy(differences, 1.0 - 2.0 * p_non_inferior, _BCA_RESAMPLES, seed + 43)

    assert _close(check_low, -delta, tolerance), (
        f"{shape}: at confidence {1.0 - 2.0 * p_non_inferior:.4f} scipy's lower bound is "
        f"{check_low!r}, not -delta = {-delta!r}"
    )
    assert _close(result.p_regressed, 1.0 - p_non_inferior, _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize(
    ("shape", "delta", "expected"),
    [
        ("normal", 0.5, stats.Verdict.NON_INFERIOR),
        ("normal", 0.02, stats.Verdict.INCONCLUSIVE),
        ("regressed", 0.05, stats.Verdict.REGRESSED),
        ("regressed", 0.9, stats.Verdict.NON_INFERIOR),
    ],
)
def test_bca_p_values_point_the_same_way_as_the_interval_in_clear_cases(
    shape: str, delta: float, expected: stats.Verdict
) -> None:
    # Contract 4.6 allows the p-values and the interval to "disagree by a hair at the
    # margin", so invariant 8 is not required of the bootstrap there. Far from the
    # margin they must agree, and both p-values must be probabilities summing to one.
    count, seed = (60, 101) if shape == "normal" else (40, 505)
    baseline, candidate = _continuous_sample(shape, count, seed)
    assert _takes_bootstrap_branch(baseline, candidate)

    result = stats.paired_continuous(
        baseline, candidate, tolerance=delta, resamples=20_000, seed=seed
    )
    alpha = 1.0 - result.interval.confidence

    assert result.method == "bootstrap_bca"
    assert _judge_reference(result.interval.low, result.interval.high, delta) is expected
    assert 0.0 <= result.p_non_inferior <= 1.0
    assert 0.0 <= result.p_regressed <= 1.0
    assert _close(result.p_regressed, 1.0 - result.p_non_inferior, _ARITHMETIC_TOLERANCE)
    lower_side, upper_side = _invariant_8_holds(result)
    assert lower_side, f"low={result.interval.low!r}, p_ni={result.p_non_inferior!r}, {alpha}"
    assert upper_side, f"high={result.interval.high!r}, p_reg={result.p_regressed!r}, {alpha}"


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_paired_continuous_point_is_the_mean_paired_difference(
    shape: str, count: int, seed: int
) -> None:
    # Contract 4.1, checked twice: directly, and by recovering the mean from scipy's
    # paired t statistic, which never forms the difference vector the same way.
    baseline, candidate = _continuous_sample(shape, count, seed)
    differences = _differences(baseline, candidate)
    expected = float(np.mean(differences))

    result = stats.paired_continuous(
        baseline, candidate, tolerance=_DELTA, resamples=2_000, seed=seed
    )

    assert _close(result.interval.point, expected, _ARITHMETIC_TOLERANCE)
    via_ttest = _paired_mean_difference_via_ttest(
        np.asarray(baseline, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    )
    assert _close(result.interval.point, via_ttest, 1e-9 * max(1.0, abs(expected)))


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_paired_continuous_reports_its_shape(shape: str, count: int, seed: int) -> None:
    baseline, candidate = _continuous_sample(shape, count, seed)
    result = stats.paired_continuous(
        baseline, candidate, tolerance=0.07, confidence=0.9, resamples=2_000, seed=1
    )

    assert result.n == count
    assert result.method == "bootstrap_bca"
    assert result.interval.confidence == 0.9
    assert result.tolerance == 0.07
    assert 0.0 <= result.p_non_inferior <= 1.0
    assert _close(result.p_regressed, 1.0 - result.p_non_inferior, _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize("seed", [0, 5, 77])
def test_paired_continuous_is_deterministic_for_a_fixed_seed(seed: int) -> None:
    # Invariant 7 and contract 4.7: "the same seed gives the same interval". Without it
    # a gate's verdict can flip between two runs of the same commit, which destroys the
    # only thing a CI check is for.
    baseline, candidate = _continuous_sample("gamma", 35, 909)
    assert _takes_bootstrap_branch(baseline, candidate)

    first = stats.paired_continuous(
        baseline, candidate, tolerance=_DELTA, resamples=3_000, seed=seed
    )
    second = stats.paired_continuous(
        baseline, candidate, tolerance=_DELTA, resamples=3_000, seed=seed
    )

    assert first.interval == second.interval
    assert first.p_non_inferior == second.p_non_inferior
    assert first.p_regressed == second.p_regressed


def test_paired_continuous_without_a_seed_is_still_deterministic() -> None:
    # Contract 4.7: "With no seed given, one is derived from the data". The default path
    # is the common one, and a fresh random stream there would let the verdict change
    # between two builds of the same commit.
    baseline, candidate = _continuous_sample("beta", 30, 303)

    first = stats.paired_continuous(baseline, candidate, tolerance=_DELTA, resamples=3_000)
    second = stats.paired_continuous(baseline, candidate, tolerance=_DELTA, resamples=3_000)

    assert first.interval == second.interval
    assert first.p_non_inferior == second.p_non_inferior


def test_paired_continuous_reseeded_differently_still_lands_in_the_same_place() -> None:
    # Determinism must come from the seed, not from a resampler that ignores it. Two
    # different seeds should give slightly different intervals -- close, but not equal.
    baseline, candidate = _continuous_sample("lognormal", 50, 404)

    first = stats.paired_continuous(baseline, candidate, tolerance=_DELTA, resamples=4_000, seed=1)
    second = stats.paired_continuous(baseline, candidate, tolerance=_DELTA, resamples=4_000, seed=2)
    width = first.interval.high - first.interval.low

    assert (first.interval.low, first.interval.high) != (second.interval.low, second.interval.high)
    assert abs(first.interval.low - second.interval.low) < 0.3 * width
    assert abs(first.interval.high - second.interval.high) < 0.3 * width


def test_paired_continuous_rejects_an_empty_sample() -> None:
    # Contract 4.7: "n = 0. Raise."
    with pytest.raises(ValueError):
        stats.paired_continuous([], [], tolerance=_DELTA)


def test_paired_continuous_rejects_a_single_item() -> None:
    # Contract 4.7: "n = 1. No interval is meaningful. Raise rather than report a
    # degenerate one."
    with pytest.raises(ValueError):
        stats.paired_continuous([0.5], [0.75], tolerance=_DELTA)


def test_paired_continuous_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        stats.paired_continuous([0.1, 0.2, 0.3], [0.1, 0.2], tolerance=_DELTA)


# ---------------------------------------------------------------------------
# 11. The invariants of contract section 8.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_invariant_1_binary_interval_brackets_its_point(
    seed: int, count: int, drift: float, confidence: float
) -> None:
    # Invariant 1: low <= point <= high, always.
    baseline, candidate = _binary_sample(seed, count, drift)
    interval = stats.paired_binary(
        baseline, candidate, tolerance=_DELTA, confidence=confidence
    ).interval
    assert interval.low <= interval.point <= interval.high


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_invariant_1_continuous_interval_brackets_its_point(
    shape: str, count: int, seed: int
) -> None:
    baseline, candidate = _continuous_sample(shape, count, seed)
    interval = stats.paired_continuous(
        baseline, candidate, tolerance=_DELTA, resamples=4_000, seed=seed
    ).interval
    assert interval.low <= interval.point <= interval.high


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_invariant_3_swapping_the_two_versions_mirrors_the_binary_interval(
    seed: int, count: int, drift: float, confidence: float
) -> None:
    # Invariant 3. The binary interval is deterministic, so the mirror is exact up to
    # the root-find. This is the cheapest catch there is for b and c swapped inside the
    # statistic: it leaves the forward answer looking plausible and the mirrored one wrong.
    baseline, candidate = _binary_sample(seed, count, drift)

    forward = stats.paired_binary(
        baseline, candidate, tolerance=_DELTA, confidence=confidence
    ).interval
    reverse = stats.paired_binary(
        candidate, baseline, tolerance=_DELTA, confidence=confidence
    ).interval

    assert _close(reverse.point, -forward.point, _ARITHMETIC_TOLERANCE)
    assert _close(reverse.low, -forward.high, _EXACT_TOLERANCE)
    assert _close(reverse.high, -forward.low, _EXACT_TOLERANCE)


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_invariant_3_swapping_the_two_versions_mirrors_the_continuous_interval(
    shape: str, count: int, seed: int
) -> None:
    # Invariant 3 for the bootstrap. The point estimate mirrors exactly. The endpoints
    # need a tolerance: negating the differences can move a single order statistic
    # across the bias correction's comparison, which shifts an endpoint by
    # O(1 / resamples) of the resample spread. 0.02 of the width is far above that and
    # far below a sign error, which would be about a whole width.
    baseline, candidate = _continuous_sample(shape, count, seed)

    forward = stats.paired_continuous(
        baseline, candidate, tolerance=_DELTA, resamples=20_000, seed=seed
    ).interval
    reverse = stats.paired_continuous(
        candidate, baseline, tolerance=_DELTA, resamples=20_000, seed=seed
    ).interval
    tolerance = 0.02 * (forward.high - forward.low)

    assert _close(reverse.point, -forward.point, _ARITHMETIC_TOLERANCE)
    assert _close(reverse.low, -forward.high, tolerance)
    assert _close(reverse.high, -forward.low, tolerance)


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
def test_invariant_4_binary_interval_widens_with_confidence(
    seed: int, count: int, drift: float
) -> None:
    # Invariant 4. A critical value taken from the wrong tail shows up as a 99% interval
    # narrower than the 80% one.
    baseline, candidate = _binary_sample(seed, count, drift)
    widths: list[float] = []
    for confidence in _CONFIDENCES:
        interval = stats.paired_binary(
            baseline, candidate, tolerance=_DELTA, confidence=confidence
        ).interval
        widths.append(interval.high - interval.low)
    for narrower, wider in pairwise(widths):
        assert wider >= narrower - _ARITHMETIC_TOLERANCE, f"widths out of order: {widths}"


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_invariant_4_continuous_interval_widens_with_confidence(
    shape: str, count: int, seed: int
) -> None:
    # Same seed and resample count throughout, so the only thing that changes is the
    # confidence level. The gaps between 80%, 95% and 99% are many times the Monte Carlo
    # noise, so this needs no tolerance beyond rounding.
    baseline, candidate = _continuous_sample(shape, count, seed)
    widths: list[float] = []
    for confidence in (0.80, 0.90, 0.95, 0.99):
        interval = stats.paired_continuous(
            baseline,
            candidate,
            tolerance=_DELTA,
            confidence=confidence,
            resamples=20_000,
            seed=seed,
        ).interval
        widths.append(interval.high - interval.low)
    for narrower, wider in pairwise(widths):
        assert wider >= narrower, f"{shape}: widths out of order: {widths}"


def test_invariant_5_judge_is_total() -> None:
    # Invariant 5: every interval and every positive tolerance yields one of the three
    # verdicts. A rule written as two `if`s with no final `else` returns None instead,
    # and None in a report reads as "no opinion" rather than "inconclusive".
    rng = np.random.default_rng(2718)
    verdicts = {stats.Verdict.NON_INFERIOR, stats.Verdict.REGRESSED, stats.Verdict.INCONCLUSIVE}
    for _ in range(400):
        low = float(rng.uniform(-1.0, 1.0))
        high = low + float(rng.uniform(0.0, 1.0))
        point = float(rng.uniform(low, high))
        tolerance = float(rng.uniform(1e-6, 1.0))
        interval = stats.Interval(point=point, low=low, high=high, confidence=0.95)

        assert stats.judge(interval, tolerance) in verdicts


def test_invariant_6_covers_both_corrections_on_real_p_values() -> None:
    # Invariant 6 on p-values that came out of the module itself rather than out of a
    # literal list, so the two halves of the pipeline are checked joined up.
    p_values: list[float] = []
    for seed, count, drift in _BINARY_SAMPLES:
        baseline, candidate = _binary_sample(seed, count, drift)
        p_values.append(stats.paired_binary(baseline, candidate, tolerance=_DELTA).p_regressed)
    p_values.extend([0.001, 0.04, 0.6, 1.0])

    for adjust in (stats.holm, stats.benjamini_hochberg):
        for before, after in zip(p_values, adjust(p_values), strict=True):
            assert before <= after + _ARITHMETIC_TOLERANCE
            assert after <= 1.0


def test_invariant_9_no_interval_over_two_or_more_items_is_a_point() -> None:
    # Invariant 9 across every branch, on the inputs most likely to collapse: nothing
    # discordant, nothing moved, and every item moved by the same amount.
    binary = stats.paired_binary([True, False], [True, False], tolerance=_DELTA).interval
    unmoved = stats.paired_continuous([0.5, 0.5], [0.5, 0.5], tolerance=_DELTA).interval
    one_moved = stats.paired_continuous([0.5, 0.5], [0.5, 0.75], tolerance=_DELTA).interval
    shifted = stats.paired_continuous(
        [0.25] * 8, [0.5] * 8, tolerance=_DELTA, resamples=2_000
    ).interval
    for interval in (binary, unmoved, one_moved, shifted):
        assert interval.low < interval.high, f"{interval!r} is a single point"


# ---------------------------------------------------------------------------
# 12. The verdict (contract 6.1).
# ---------------------------------------------------------------------------


def test_verdict_and_correction_values_are_the_contract_strings() -> None:
    # The verdict and the correction name are written into a report and into CI output,
    # so their spelling is part of the contract, not an internal detail.
    assert str(stats.Verdict.NON_INFERIOR) == "non_inferior"
    assert str(stats.Verdict.REGRESSED) == "regressed"
    assert str(stats.Verdict.INCONCLUSIVE) == "inconclusive"
    assert str(stats.Correction.HOLM) == "holm"
    assert str(stats.Correction.BENJAMINI_HOCHBERG) == "benjamini_hochberg"


def test_judge_returns_non_inferior_when_the_interval_rules_out_the_regression() -> None:
    # Contract 6.1, row 1: low > -delta.
    interval = stats.Interval(point=-0.01, low=-0.03, high=0.01, confidence=0.95)
    assert stats.judge(interval, 0.05) is stats.Verdict.NON_INFERIOR


def test_judge_returns_regressed_when_the_interval_rules_out_being_within_tolerance() -> None:
    # Contract 6.1, row 2: high < -delta.
    interval = stats.Interval(point=-0.20, low=-0.30, high=-0.12, confidence=0.95)
    assert stats.judge(interval, 0.05) is stats.Verdict.REGRESSED


def test_judge_returns_inconclusive_when_the_interval_straddles_the_tolerance() -> None:
    # Contract 6.1, row 3, and the rule that matters most: "`inconclusive` is a
    # first-class state and must never collapse into `non_inferior`".
    interval = stats.Interval(point=-0.02, low=-0.18, high=0.09, confidence=0.95)
    assert stats.judge(interval, 0.05) is stats.Verdict.INCONCLUSIVE


def test_judge_treats_an_endpoint_exactly_on_the_tolerance_as_inconclusive() -> None:
    # The contract's conditions are strict inequalities, so equality falls through to
    # "otherwise". Written as `>=` instead, the first case would turn a slice that only
    # just fails to rule out the regression into a pass.
    on_the_low_side = stats.Interval(point=0.0, low=-0.05, high=0.2, confidence=0.95)
    assert stats.judge(on_the_low_side, 0.05) is stats.Verdict.INCONCLUSIVE

    on_the_high_side = stats.Interval(point=-0.1, low=-0.3, high=-0.05, confidence=0.95)
    assert stats.judge(on_the_high_side, 0.05) is stats.Verdict.INCONCLUSIVE


def test_all_three_verdicts_are_reachable_from_one_interval_by_moving_the_tolerance() -> None:
    # Reachability stated as the contract states it: all three states must be reachable.
    interval = stats.Interval(point=-0.10, low=-0.22, high=0.02, confidence=0.95)

    assert stats.judge(interval, 0.5) is stats.Verdict.NON_INFERIOR
    assert stats.judge(interval, 0.15) is stats.Verdict.INCONCLUSIVE
    assert stats.judge(interval, 0.005) is stats.Verdict.INCONCLUSIVE

    regressed = stats.Interval(point=-0.6, low=-0.9, high=-0.4, confidence=0.95)
    assert stats.judge(regressed, 0.1) is stats.Verdict.REGRESSED


@pytest.mark.parametrize("tolerance", [0.0, -0.0, -1e-12, -0.05, -1.0])
def test_judge_rejects_a_non_positive_tolerance(tolerance: float) -> None:
    # Contract 6.1: "`delta` must be strictly positive; a tolerance of zero turns the
    # question back into a significance test. Reject it."
    interval = stats.Interval(point=-0.01, low=-0.03, high=0.01, confidence=0.95)
    with pytest.raises(ValueError):
        stats.judge(interval, tolerance)


# ---------------------------------------------------------------------------
# 13. A family of slices (contract 5 and 6.2).
# ---------------------------------------------------------------------------

_CORRECTIONS: tuple[tuple[stats.Correction, str], ...] = (
    (stats.Correction.HOLM, "holm"),
    (stats.Correction.BENJAMINI_HOCHBERG, "fdr_bh"),
)


def test_family_verdicts_on_an_empty_family_is_empty() -> None:
    # Contract 5: an empty input returns empty, and 6.2 is built on those corrections.
    assert stats.family_verdicts([], stats.Correction.HOLM) == []
    assert stats.family_verdicts([], stats.Correction.BENJAMINI_HOCHBERG) == []


def test_a_slice_regressed_alone_is_inconclusive_once_holm_counts_the_family() -> None:
    # Contract 6.2: a slice whose own interval says regressed but which, "once the number
    # of slices tested is accounted for, is within what chance produces" is inconclusive.
    # Alone, p_regressed = 0.02 < 0.025 keeps it regressed; among five gate slices Holm
    # multiplies it by 5 and it is not.
    regressed = _paired_result(-0.30, -0.10, 0.05, p_regressed=0.02)
    others = [
        _paired_result(-0.02, 0.04, 0.05, p_regressed=0.9),
        _paired_result(-0.04, 0.03, 0.05, p_regressed=0.8),
        _paired_result(-0.12, 0.05, 0.05, p_regressed=0.3),
        _paired_result(-0.01, 0.06, 0.05, p_regressed=0.95),
    ]
    family = [others[0], regressed, *others[1:]]

    alone = stats.family_verdicts([regressed], stats.Correction.HOLM)
    together = stats.family_verdicts(family, stats.Correction.HOLM)

    assert alone == [stats.Verdict.REGRESSED]
    assert together == _family_verdicts_reference(family, "holm")
    assert together[1] is stats.Verdict.INCONCLUSIVE
    assert together[0] is stats.Verdict.NON_INFERIOR
    assert together[3] is stats.Verdict.INCONCLUSIVE


def test_benjamini_hochberg_keeps_regressions_that_holm_does_not() -> None:
    # Watch slices use BH (contract 5), which is less conservative. On raw p_regressed
    # 0.01, 0.012, 0.014 among four slices Holm's adjusted values are all 0.04 and BH's
    # are all ~0.0187, so the same family reads inconclusive under one and regressed under
    # the other. Swapping the two corrections is visible here.
    family = [
        _paired_result(-0.40, -0.20, 0.05, p_regressed=0.01),
        _paired_result(-0.35, -0.15, 0.05, p_regressed=0.012),
        _paired_result(-0.30, -0.10, 0.05, p_regressed=0.014),
        _paired_result(-0.02, 0.04, 0.05, p_regressed=0.9),
    ]

    holm = stats.family_verdicts(family, stats.Correction.HOLM)
    bh = stats.family_verdicts(family, stats.Correction.BENJAMINI_HOCHBERG)

    assert holm == _family_verdicts_reference(family, "holm")
    assert bh == _family_verdicts_reference(family, "fdr_bh")
    assert holm[:3] == [stats.Verdict.INCONCLUSIVE] * 3
    assert bh[:3] == [stats.Verdict.REGRESSED] * 3
    assert holm[3] is bh[3] is stats.Verdict.NON_INFERIOR


def test_non_inferior_is_never_adjusted() -> None:
    # Contract 6.2: non_inferior "exactly when its own verdict is non_inferior. There is
    # no adjustment." Contract 5 explains why: the intersection-union test needs none.
    family = [
        _paired_result(-0.01, 0.05, 0.05, p_regressed=0.999),
        _paired_result(-0.04, 0.02, 0.05, p_regressed=0.6),
        _paired_result(-0.049, 0.0, 0.05, p_regressed=0.03),
    ]
    for correction, _ in _CORRECTIONS:
        assert stats.family_verdicts(family, correction) == [stats.Verdict.NON_INFERIOR] * 3


def test_family_verdicts_follow_a_permutation_of_the_input() -> None:
    # Contract 7: "one verdict per result, in the input order". Sorting by p-value inside
    # the correction and forgetting to unsort is the likely bug.
    family = [
        _paired_result(-0.40, -0.20, 0.05, p_regressed=0.001),
        _paired_result(-0.02, 0.04, 0.05, p_regressed=0.9),
        _paired_result(-0.30, 0.02, 0.05, p_regressed=0.2),
        _paired_result(-0.35, -0.12, 0.05, p_regressed=0.004),
        _paired_result(-0.25, -0.08, 0.05, p_regressed=0.02),
    ]
    order = [int(index) for index in np.random.default_rng(5).permutation(len(family))]
    permuted = [family[index] for index in order]
    for correction, method in _CORRECTIONS:
        direct = stats.family_verdicts(family, correction)
        reordered = stats.family_verdicts(permuted, correction)
        assert direct == _family_verdicts_reference(family, method)
        assert reordered == [direct[index] for index in order]


@pytest.mark.parametrize("correction", [stats.Correction.HOLM, stats.Correction.BENJAMINI_HOCHBERG])
def test_family_verdicts_reject_mixed_confidence_levels(correction: stats.Correction) -> None:
    # Contract 6.2 and 7a: "Every result in one family must share a confidence level;
    # mixing them is an error." alpha/2 would otherwise mean two things in one family.
    family = [
        _paired_result(-0.30, -0.10, 0.05, p_regressed=0.01, confidence=0.95),
        _paired_result(-0.30, -0.10, 0.05, p_regressed=0.01, confidence=0.90),
    ]
    with pytest.raises(ValueError):
        stats.family_verdicts(family, correction)


def test_family_verdicts_match_the_reference_and_invariant_10_on_random_families() -> None:
    # Random families of one to eight slices at several confidence levels, compared with
    # the statsmodels-based reference, and invariant 10: the family step can only demote
    # a verdict to inconclusive, never produce non_inferior or regressed that `judge` did
    # not already give.
    rng = np.random.default_rng(6262)
    for _ in range(300):
        size = int(rng.integers(1, 9))
        confidence = float(rng.choice(np.asarray([0.90, 0.95, 0.99])))
        family: list[stats.PairedResult] = []
        for _ in range(size):
            tolerance = float(rng.uniform(0.01, 0.3))
            low = float(rng.uniform(-0.6, 0.2))
            high = low + float(rng.uniform(0.01, 0.5))
            if rng.random() < 0.4:
                p_regressed = float(10.0 ** rng.uniform(-5.0, -1.3))
            else:
                p_regressed = float(rng.uniform(0.0, 1.0))
            family.append(_paired_result(low, high, tolerance, p_regressed, confidence))

        for correction, method in _CORRECTIONS:
            verdicts = stats.family_verdicts(family, correction)
            assert verdicts == _family_verdicts_reference(family, method)
            for result, verdict in zip(family, verdicts, strict=True):
                own = stats.judge(result.interval, result.tolerance)
                if verdict is not stats.Verdict.INCONCLUSIVE:
                    assert verdict is own, f"{correction}: {verdict} where judge said {own}"


# ---------------------------------------------------------------------------
# 14. Argument errors (contract 7a).
# ---------------------------------------------------------------------------

#: NaN is included because it is not strictly positive; a check written as
#: `tolerance <= 0` lets it through, and a NaN tolerance makes every comparison False.
_BAD_TOLERANCES: tuple[float, ...] = (0.0, -0.0, -1e-12, -0.05, math.nan)


@pytest.mark.parametrize("tolerance", _BAD_TOLERANCES)
def test_paired_functions_reject_a_tolerance_that_is_not_strictly_positive(
    tolerance: float,
) -> None:
    # Contract 7a: "So is a tolerance that is not strictly positive", for both functions.
    with pytest.raises(ValueError):
        stats.paired_binary([True, False, True], [True, True, False], tolerance=tolerance)
    baseline, candidate = _continuous_sample("normal", 60, 101)
    with pytest.raises(ValueError):
        stats.paired_continuous(baseline, candidate, tolerance=tolerance, resamples=1_000)


@pytest.mark.parametrize(
    ("index", "side", "value"),
    [
        (0, "baseline", -0.01),
        (3, "candidate", 1.01),
        (5, "baseline", 1.0 + 1e-9),
        (9, "candidate", -1e-12),
        (2, "baseline", 2.0),
        (4, "candidate", math.nan),
    ],
)
def test_paired_continuous_rejects_a_score_outside_zero_to_one(
    index: int, side: str, value: float
) -> None:
    # Contract 4.3 and 7a: a score outside [0, 1] is an error. The bound is what makes the
    # count bound valid, so a score of 1.01 silently accepted would void its guarantee.
    baseline = [0.1 * step for step in range(10)]
    candidate = [0.05 + 0.09 * step for step in range(10)]
    if side == "baseline":
        baseline[index] = value
    else:
        candidate[index] = value
    with pytest.raises(ValueError):
        stats.paired_continuous(baseline, candidate, tolerance=_DELTA, resamples=1_000)


def test_paired_continuous_accepts_scores_exactly_on_the_bounds() -> None:
    # [0, 1] is closed: an item going from 0 to 1 is the largest legal move, and it is
    # the case the constant-shift formulas are built around (v = 1).
    result = stats.paired_continuous([0.0] * 10, [1.0] * 10, tolerance=_DELTA)
    assert result.method == "constant_shift"
    assert result.interval.point == 1.0


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n", "difference"),
    [
        (0, 0, 0, 0.0),
        (0, 0, -3, 0.0),
        (-1, 0, 5, 0.0),
        (0, -1, 5, 0.0),
        (3, 3, 5, 0.0),
        (1, 1, 5, 1.0000001),
        (1, 1, 5, -1.5),
    ],
)
def test_tango_statistic_rejects_invalid_arguments(
    favour_candidate: int, favour_baseline: int, n: int, difference: float
) -> None:
    # Contract 7a: n > 0, non-negative counts, b + c <= n, and difference in [-1, 1].
    with pytest.raises(ValueError):
        stats.tango_statistic(favour_candidate, favour_baseline, n, difference)


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n"),
    [(0, 0, 0), (0, 0, -1), (-1, 0, 5), (0, -1, 5), (4, 2, 5)],
)
def test_tango_interval_rejects_invalid_arguments(
    favour_candidate: int, favour_baseline: int, n: int
) -> None:
    # Contract 7a: n > 0, non-negative counts, b + c <= n.
    with pytest.raises(ValueError):
        stats.tango_interval(favour_candidate, favour_baseline, n)


@pytest.mark.parametrize(
    ("successes", "trials", "confidence"),
    [(0, 0, 0.95), (-1, 5, 0.95), (6, 5, 0.95), (1, 5, 0.0), (1, 5, 1.0), (1, 5, 1.5)],
)
def test_clopper_pearson_rejects_invalid_arguments(
    successes: int, trials: int, confidence: float
) -> None:
    # Contract 7a: trials > 0, 0 <= successes <= trials and 0 < confidence < 1.
    with pytest.raises(ValueError):
        stats.clopper_pearson(successes, trials, confidence)


# ---------------------------------------------------------------------------
# 15. The dependency rule (contract 9).
# ---------------------------------------------------------------------------


def test_stats_module_imports_neither_scipy_nor_statsmodels() -> None:
    # Contract 9. If the implementation imported scipy, every comparison above would be
    # scipy agreeing with itself, and this whole file would be evidence of nothing. The
    # source is parsed rather than grepped so that a mention in a docstring -- which is
    # fine, and likely -- is not confused with an import, and so that an import nested
    # inside a function is caught as readily as one at the top.
    tree = ast.parse(inspect.getsource(stats))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])

    assert "scipy" not in imported, "evalgate.stats imports scipy; contract 9 forbids it"
    assert "statsmodels" not in imported, (
        "evalgate.stats imports statsmodels; contract 9 forbids it"
    )
    # Contract 9 names what is allowed: stdlib plus numpy. A dynamic import would slip
    # past the AST walk above, so it is ruled out separately.
    assert "importlib" not in imported, "evalgate.stats imports importlib; the rule is checkable"
    assert "__import__" not in inspect.getsource(stats)


def test_scipy_and_statsmodels_are_not_runtime_dependencies_of_the_package() -> None:
    # The dependency rule only means something while the two packages stay out of the
    # distribution's runtime requirements. If they were added there, importing scipy in
    # `stats.py` would stop being a mistake anyone noticed.
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = manifest["project"]
    assert isinstance(project, dict)
    runtime = [str(entry) for entry in project["dependencies"]]
    names = {
        entry.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip() for entry in runtime
    }

    assert "scipy" not in names
    assert "statsmodels" not in names
