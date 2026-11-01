"""Independent oracle for `evalgate.stats`, checked against scipy and statsmodels.

Owning artifact: `docs/statistics.md`. Section numbers in the comments below refer to
that document.

Every expectation here was authored from the contract and from scipy/statsmodels only.
Nothing in this file was written while reading `src/evalgate/stats.py`, which is the
whole point: an oracle shaped by the code it judges agrees with that code's bugs. When
this file and the implementation disagree, the disagreement is resolved by finding the
cause in one of them. This file is never edited to agree.

Deliberate boundary: scipy and statsmodels are test-only dependencies (contract 9).
They are imported here and must never be imported by the module under test;
`test_stats_module_imports_neither_scipy_nor_statsmodels` enforces that directly.

Until `src/evalgate/stats.py` lands this module fails to import. That is the expected
state and is not something to work around by stubbing the module.
"""

from __future__ import annotations

import ast
import inspect
import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import scipy.stats
import statsmodels.stats.contingency_tables
import statsmodels.stats.multitest
import statsmodels.stats.proportion

from evalgate import stats

# scipy and statsmodels ship no type information, so pyright infers `Unknown` for every
# symbol imported from them, which is an error under strict mode. Binding each module
# to an explicitly `Any`-typed name narrows the untypedness to these four lines instead
# of scattering suppression comments through the file, and every value that crosses
# back out of a reference helper is converted with an explicit `float`/`list`.
_scipy_stats: Any = scipy.stats
_sm_proportion: Any = statsmodels.stats.proportion
_sm_tables: Any = statsmodels.stats.contingency_tables
_sm_multitest: Any = statsmodels.stats.multitest

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

# Deterministic exact quantities (Clopper-Pearson bounds, McNemar p-values, Holm and
# Benjamini-Hochberg adjustments, the jackknife acceleration) are computed by both
# sides in double precision from closed forms or root-finds. scipy and statsmodels
# agree with each other to about 2e-14 on the Clopper-Pearson grid below, so a
# well-implemented third computation has ~1e-12 of genuine headroom. 1e-9 leaves room
# for an implementation that inverts the beta distribution by its own bisection rather
# than by a library special function, and is still six orders of magnitude tighter than
# any interval the gate ever reports.
_EXACT_TOLERANCE = 1e-9

# A tighter band for identities that are pure arithmetic on the same inputs, where any
# disagreement beyond accumulated rounding is a real difference in definition.
_ARITHMETIC_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# Reference helpers. Each one wraps a single scipy or statsmodels call and
# returns plain floats, so a mistake in how the reference is invoked is visible
# in one place rather than smeared across thirty assertions.
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


def _mcnemar_scipy(favour_candidate: int, favour_baseline: int) -> float:
    """Exact McNemar as the two-sided binomial test of the discordant split (contract 3.5)."""
    discordant = favour_candidate + favour_baseline
    return float(_scipy_stats.binomtest(favour_candidate, discordant, 0.5).pvalue)


def _mcnemar_statsmodels(favour_candidate: int, favour_baseline: int) -> float:
    """Exact McNemar from a 2x2 table; the diagonal is irrelevant to the exact test."""
    table = [[11, favour_candidate], [favour_baseline, 13]]
    return float(_sm_tables.mcnemar(table, exact=True).pvalue)


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
    Contract 4.3 defines `a = 0` when the jackknife spread is zero.
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
# Case grids.
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

#: Discordant splits, including lopsided ones and both one-sided extremes.
_DISCORDANT_SPLITS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 7),
    (7, 3),
    (5, 5),
    (9, 0),
    (0, 9),
    (12, 4),
    (20, 18),
    (31, 2),
)

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


#: Four paired continuous samples with different shapes. `normal` is here for breadth;
#: `gamma` and `lognormal` are here because BCa and the plain percentile interval are
#: visibly different on them, which is what gives the bootstrap comparison its power.
_CONTINUOUS_SHAPES: tuple[tuple[str, int, int], ...] = (
    ("normal", 60, 101),
    ("gamma", 40, 202),
    ("beta", 30, 303),
    ("lognormal", 50, 404),
)


def _continuous_sample(shape: str, count: int, seed: int) -> tuple[list[float], list[float]]:
    """A paired continuous sample whose differences follow `shape`."""
    rng = np.random.default_rng(seed)
    if shape == "normal":
        differences = rng.normal(-0.03, 0.25, count)
    elif shape == "gamma":
        differences = rng.gamma(1.2, 0.15, count) - 0.18
    elif shape == "beta":
        differences = rng.beta(1.5, 6.0, count) - 0.2
    elif shape == "lognormal":
        differences = np.exp(rng.normal(0.0, 0.8, count)) - 1.3
    else:  # pragma: no cover - a typo in a parametrize id must not pass silently
        raise AssertionError(f"unknown shape {shape!r}")
    baseline = rng.uniform(0.3, 0.8, count)
    candidate = baseline + differences
    return [float(value) for value in baseline], [float(value) for value in candidate]


def _differences(baseline: Sequence[float], candidate: Sequence[float]) -> FloatArray:
    """The paired differences the implementation is required to form (contract 4.1)."""
    return np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)


# ---------------------------------------------------------------------------
# 1. Clopper-Pearson (contract 3.3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("successes", "trials", "confidence"), _CLOPPER_PEARSON_CASES)
def test_clopper_pearson_matches_scipy(successes: int, trials: int, confidence: float) -> None:
    # A failure means the exact binomial interval is wrong, which moves every binary
    # verdict the gate reports. The grid covers both extremes, where the two tails are
    # defined by a boundary rule rather than by a root-find.
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
    # Contract 3.3 pins these two by rule, not by a root-find. A solver that returns
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
# 2. Exact McNemar (contract 3.5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("favour_candidate", "favour_baseline"), _DISCORDANT_SPLITS)
def test_mcnemar_exact_matches_scipy_binomtest(favour_candidate: int, favour_baseline: int) -> None:
    # The p-value is reported for information only, but a wrong one is still a wrong
    # number in a report a human reads, and it is the input to both corrections.
    expected = _mcnemar_scipy(favour_candidate, favour_baseline)
    actual = stats.mcnemar_exact_p(favour_candidate, favour_baseline)

    assert _close(actual, expected, _EXACT_TOLERANCE), (
        f"McNemar p for ({favour_candidate}, {favour_baseline}): {actual!r} != {expected!r}"
    )


@pytest.mark.parametrize(("favour_candidate", "favour_baseline"), _DISCORDANT_SPLITS)
def test_mcnemar_exact_matches_statsmodels_table(
    favour_candidate: int, favour_baseline: int
) -> None:
    # statsmodels reaches the same number from a 2x2 table rather than from the counts,
    # so agreement here also confirms the discordant counts are the right two cells.
    expected = _mcnemar_statsmodels(favour_candidate, favour_baseline)
    actual = stats.mcnemar_exact_p(favour_candidate, favour_baseline)

    assert _close(actual, expected, _EXACT_TOLERANCE)


def test_mcnemar_with_no_discordant_pairs_is_one() -> None:
    # scipy's binomtest refuses zero trials, so statsmodels is the only reference here:
    # it reports 1.0, which is also the only defensible answer. Anything else would be
    # a claim of evidence from a sample that contains none.
    expected = _mcnemar_statsmodels(0, 0)
    assert expected == 1.0
    assert _close(stats.mcnemar_exact_p(0, 0), 1.0, _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize(("favour_candidate", "favour_baseline"), _DISCORDANT_SPLITS)
def test_mcnemar_is_symmetric_in_its_arguments(favour_candidate: int, favour_baseline: int) -> None:
    # The test is two-sided, so relabelling which version is "candidate" cannot change
    # it. A failure means a one-sided tail was used somewhere.
    forward = stats.mcnemar_exact_p(favour_candidate, favour_baseline)
    reverse = stats.mcnemar_exact_p(favour_baseline, favour_candidate)
    assert _close(forward, reverse, _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize(("favour_candidate", "favour_baseline"), _DISCORDANT_SPLITS)
def test_mcnemar_p_value_is_a_probability(favour_candidate: int, favour_baseline: int) -> None:
    value = stats.mcnemar_exact_p(favour_candidate, favour_baseline)
    assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# 3. Holm and Benjamini-Hochberg (contract 5).
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
# 4. Jackknife acceleration (contract 4.2 and 4.3).
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
    # Contract 4.3 defines `a = 0` here rather than leaving 0/0. A NaN would propagate
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
# 5. Paired binary (contract 3).
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

    result = stats.paired_binary(baseline, candidate)

    assert _close(result.interval.point, expected, _ARITHMETIC_TOLERANCE), (
        f"point {result.interval.point!r} != mean(candidate) - mean(baseline) {expected!r}"
    )


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_paired_binary_interval_matches_a_clopper_pearson_interval_from_scipy(
    seed: int, count: int, drift: float, confidence: float
) -> None:
    # Contract 3.3 end to end: build the discordant counts, take scipy's exact binomial
    # interval on the discordant split, and map it back with
    # `(2 * pi - 1) * discordant / n`. A failure means either the counts or the mapping
    # is wrong, and both change every binary verdict.
    baseline, candidate = _binary_sample(seed, count, drift)
    favour_candidate = sum(1 for b, c in zip(baseline, candidate, strict=True) if not b and c)
    favour_baseline = sum(1 for b, c in zip(baseline, candidate, strict=True) if b and not c)
    discordant = favour_candidate + favour_baseline
    assert discordant > 0, "this sample was chosen to have discordant pairs"

    pi_low, pi_high = _clopper_pearson_scipy(favour_candidate, discordant, confidence)
    scale = discordant / count
    expected_low = (2.0 * pi_low - 1.0) * scale
    expected_high = (2.0 * pi_high - 1.0) * scale

    result = stats.paired_binary(baseline, candidate, confidence)

    assert _close(result.interval.low, expected_low, _EXACT_TOLERANCE), (
        f"low {result.interval.low!r} != {expected_low!r} "
        f"(k={favour_candidate}, d={discordant}, n={count})"
    )
    assert _close(result.interval.high, expected_high, _EXACT_TOLERANCE), (
        f"high {result.interval.high!r} != {expected_high!r} "
        f"(k={favour_candidate}, d={discordant}, n={count})"
    )


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
def test_paired_binary_p_value_matches_statsmodels_mcnemar(
    seed: int, count: int, drift: float
) -> None:
    # Built from the real 2x2 table rather than from counts this file computed, so a
    # transposed table in the implementation is visible here.
    baseline, candidate = _binary_sample(seed, count, drift)
    table = [[0, 0], [0, 0]]
    for base, cand in zip(baseline, candidate, strict=True):
        table[int(base)][int(cand)] += 1
    expected = float(_sm_tables.mcnemar(table, exact=True).pvalue)

    result = stats.paired_binary(baseline, candidate)

    assert result.p_value is not None, "binary results carry an exact McNemar p-value"
    assert _close(result.p_value, expected, _EXACT_TOLERANCE)


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
def test_paired_binary_reports_its_shape(seed: int, count: int, drift: float) -> None:
    # The report names the method and the confidence it used. A result that claims
    # "exact_conditional" while holding a bootstrap interval misleads its reader.
    baseline, candidate = _binary_sample(seed, count, drift)
    result = stats.paired_binary(baseline, candidate, 0.9)

    assert result.n == count
    assert result.method == "exact_conditional"
    assert result.interval.confidence == 0.9


@pytest.mark.parametrize("count", [1, 5, 20, 64])
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_paired_binary_with_no_discordant_pairs_is_never_a_point_interval(
    count: int, confidence: float
) -> None:
    # Contract 3.4 is emphatic: "zero observed disagreements on 20 items is not proof of
    # equality", and `[0, 0]` must never be reported. Reporting it would make every such
    # slice `non_inferior` by arithmetic rather than by evidence -- the precise failure
    # the project exists to prevent.
    baseline = [True] * (count // 2) + [False] * (count - count // 2)
    candidate = list(baseline)

    result = stats.paired_binary(baseline, candidate, confidence)

    assert result.interval.point == 0.0
    assert result.interval.low < 0.0 < result.interval.high, (
        f"n={count}: interval {result.interval.low!r}..{result.interval.high!r} is degenerate"
    )
    assert _close(result.interval.low, -result.interval.high, _ARITHMETIC_TOLERANCE)


@pytest.mark.parametrize("count", [1, 5, 20, 64])
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_paired_binary_with_no_discordant_pairs_uses_the_clopper_pearson_upper_bound(
    count: int, confidence: float
) -> None:
    # Contract 3.4: "take the Clopper-Pearson upper bound `u` for 0 successes in n
    # trials and report [-u, +u]". Read together with 3.3, which defines both bounds at
    # `alpha / 2`, that is the two-sided upper bound, which is what this asserts. See
    # the oracle's report: the contract's aside that `u ~= 3/n` describes the *one-sided*
    # bound (3.69/n two-sided), so this expectation is the one place where a reasonable
    # implementation could differ from the oracle without either being careless.
    baseline = [True] * (count // 2) + [False] * (count - count // 2)
    candidate = list(baseline)
    expected = _clopper_pearson_scipy(0, count, confidence)[1]

    result = stats.paired_binary(baseline, candidate, confidence)

    assert _close(result.interval.high, expected, _EXACT_TOLERANCE), (
        f"n={count}, confidence={confidence}: bound {result.interval.high!r} != {expected!r}"
    )


@pytest.mark.parametrize("count", [3, 10, 50])
def test_paired_binary_when_every_discordant_pair_favours_one_side(count: int) -> None:
    # Contract 3.4's third case. With `pi_hi = 1` the upper end is exactly
    # `discordant / n`, so the interval stays finite; the lower end must still be
    # strictly below the point estimate.
    baseline = [False] * count
    candidate = [True] * count

    result = stats.paired_binary(baseline, candidate, 0.95)

    assert _close(result.interval.point, 1.0, _ARITHMETIC_TOLERANCE)
    assert _close(result.interval.high, 1.0, _ARITHMETIC_TOLERANCE)
    assert result.interval.low < 1.0


def test_paired_binary_rejects_an_empty_sample() -> None:
    # Contract 3.4: "n = 0. There is no estimate. Raise, rather than return a number
    # that means nothing." The exception type is not pinned by the contract; ValueError
    # is the Python convention for an argument whose value is unusable.
    with pytest.raises(ValueError):
        stats.paired_binary([], [])


def test_paired_binary_rejects_mismatched_lengths() -> None:
    # Pairing is by item id (contract 2). Two vectors of different lengths cannot be
    # paired, and silently truncating to the shorter one would drop real items.
    with pytest.raises(ValueError):
        stats.paired_binary([True, False, True], [True, False])


# ---------------------------------------------------------------------------
# 6. BCa bootstrap (contract 4.2).
# ---------------------------------------------------------------------------

#: Both sides draw their own resamples, so exact agreement is impossible and the only
#: honest comparison is one with a stated Monte Carlo budget.
#:
#: Measured on the four samples below, with scipy against itself across 15 reference
#: seeds at 50_000 resamples: the per-endpoint standard deviation is about 0.006 of the
#: interval width and the full observed range is under 0.026 of it. The difference
#: between two *independent* runs therefore has a standard deviation near 0.008 of the
#: width, so 0.05 of the width sits at roughly six standard deviations -- a false
#: failure is a one-in-a-million event, not a flake that gets re-run.
#:
#: It is not so loose that the test cannot fail. On `gamma` and `lognormal`, replacing
#: BCa with the plain percentile interval moves an endpoint by 0.05 to 0.15 of the
#: width; dropping the acceleration, flipping the sign of the bias correction, or
#: swapping the two adjusted percentiles move it further still, and a sign error in the
#: point estimate moves it by more than a whole width. On a symmetric sample BCa and
#: percentile genuinely coincide, which is a property of the estimator rather than a
#: weakness of this number: `normal` is in the grid for breadth, and the two skewed
#: samples are what give this comparison its power.
_BCA_WIDTH_TOLERANCE = 0.05

#: High enough that the Monte Carlo error above is small next to a real mistake.
_BCA_RESAMPLES = 50_000


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
        baseline, candidate, confidence=0.95, resamples=_BCA_RESAMPLES, seed=seed
    )

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
        baseline, candidate, confidence=confidence, resamples=_BCA_RESAMPLES, seed=seed
    )

    assert _close(result.interval.low, expected_low, tolerance)
    assert _close(result.interval.high, expected_high, tolerance)


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

    result = stats.paired_continuous(baseline, candidate, resamples=2_000, seed=seed)

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
    result = stats.paired_continuous(baseline, candidate, confidence=0.9, resamples=2_000, seed=1)

    assert result.n == count
    assert result.method == "bootstrap_bca"
    assert result.interval.confidence == 0.9
    # Contract 7: a continuous result has no p-value, because no test was run.
    assert result.p_value is None


@pytest.mark.parametrize("seed", [0, 5, 77])
def test_paired_continuous_is_deterministic_for_a_fixed_seed(seed: int) -> None:
    # Invariant 7 and contract 4.3: "the same seed gives the same interval". Without it
    # a gate's verdict can flip between two runs of the same commit, which destroys the
    # only thing a CI check is for.
    baseline, candidate = _continuous_sample("gamma", 35, 909)

    first = stats.paired_continuous(baseline, candidate, resamples=3_000, seed=seed)
    second = stats.paired_continuous(baseline, candidate, resamples=3_000, seed=seed)

    assert first.interval.low == second.interval.low
    assert first.interval.high == second.interval.high
    assert first.interval.point == second.interval.point


def test_paired_continuous_reseeded_differently_still_lands_in_the_same_place() -> None:
    # Determinism must come from the seed, not from a resampler that ignores it. Two
    # different seeds should give slightly different intervals -- close, but not equal.
    baseline, candidate = _continuous_sample("lognormal", 50, 404)

    first = stats.paired_continuous(baseline, candidate, resamples=4_000, seed=1)
    second = stats.paired_continuous(baseline, candidate, resamples=4_000, seed=2)
    width = first.interval.high - first.interval.low

    assert (first.interval.low, first.interval.high) != (second.interval.low, second.interval.high)
    assert abs(first.interval.low - second.interval.low) < 0.3 * width
    assert abs(first.interval.high - second.interval.high) < 0.3 * width


@pytest.mark.parametrize("shift", [0.0, 0.25, -0.5])
def test_paired_continuous_collapses_when_every_difference_is_identical(shift: float) -> None:
    # Contract 4.3. The values are dyadic so `candidate - baseline` is exact in binary
    # floating point and the degenerate branch is genuinely reached. scipy returns NaN
    # here (DegenerateDataWarning), so the contract is the only reference: the interval
    # collapses to the point estimate, which is the honest answer when every item moved
    # by the same amount.
    baseline = [0.5, 1.25, 2.0, 3.75, 4.5, 0.25, 7.0]
    candidate = [value + shift for value in baseline]

    result = stats.paired_continuous(baseline, candidate, resamples=2_000, seed=3)

    assert _close(result.interval.point, shift, _ARITHMETIC_TOLERANCE)
    assert _close(result.interval.low, shift, _ARITHMETIC_TOLERANCE)
    assert _close(result.interval.high, shift, _ARITHMETIC_TOLERANCE)


def test_paired_continuous_rejects_an_empty_sample() -> None:
    # Contract 4.3: "n = 0. Raise."
    with pytest.raises(ValueError):
        stats.paired_continuous([], [])


def test_paired_continuous_rejects_a_single_item() -> None:
    # Contract 4.3: "n = 1. No interval is meaningful. Raise rather than report a
    # degenerate one." A single item resampled with replacement always gives the same
    # mean, so a bootstrap would report a zero-width interval and the gate would call
    # the slice conclusive on one example.
    with pytest.raises(ValueError):
        stats.paired_continuous([0.5], [0.75])


def test_paired_continuous_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        stats.paired_continuous([0.1, 0.2, 0.3], [0.1, 0.2])


# ---------------------------------------------------------------------------
# 7. The invariants of contract section 8.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_invariant_1_binary_interval_brackets_its_point(
    seed: int, count: int, drift: float, confidence: float
) -> None:
    # Invariant 1: low <= point <= high, always.
    baseline, candidate = _binary_sample(seed, count, drift)
    interval = stats.paired_binary(baseline, candidate, confidence).interval
    assert interval.low <= interval.point <= interval.high


@pytest.mark.parametrize(
    ("shape", "count", "seed"), _CONTINUOUS_SHAPES, ids=[s[0] for s in _CONTINUOUS_SHAPES]
)
def test_invariant_1_continuous_interval_brackets_its_point(
    shape: str, count: int, seed: int
) -> None:
    baseline, candidate = _continuous_sample(shape, count, seed)
    interval = stats.paired_continuous(baseline, candidate, resamples=4_000, seed=seed).interval
    assert interval.low <= interval.point <= interval.high


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
@pytest.mark.parametrize("confidence", _CONFIDENCES)
def test_invariant_3_swapping_the_two_versions_mirrors_the_binary_interval(
    seed: int, count: int, drift: float, confidence: float
) -> None:
    # Invariant 3. The binary interval is deterministic, so the mirror is exact. This is
    # the cheapest catch there is for a sign error or a swapped Clopper-Pearson tail:
    # both leave the forward answer looking plausible and the mirrored one wrong.
    baseline, candidate = _binary_sample(seed, count, drift)

    forward = stats.paired_binary(baseline, candidate, confidence).interval
    reverse = stats.paired_binary(candidate, baseline, confidence).interval

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
    # across the strict-inequality in the bias correction, which shifts an endpoint by
    # O(1 / resamples) of the resample spread. 0.02 of the width is far above that and
    # far below a sign error, which would be about a whole width.
    baseline, candidate = _continuous_sample(shape, count, seed)

    forward = stats.paired_continuous(baseline, candidate, resamples=20_000, seed=seed).interval
    reverse = stats.paired_continuous(candidate, baseline, resamples=20_000, seed=seed).interval
    tolerance = 0.02 * (forward.high - forward.low)

    assert _close(reverse.point, -forward.point, _ARITHMETIC_TOLERANCE)
    assert _close(reverse.low, -forward.high, tolerance)
    assert _close(reverse.high, -forward.low, tolerance)


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
def test_invariant_4_binary_interval_widens_with_confidence(
    seed: int, count: int, drift: float
) -> None:
    # Invariant 4. A tail direction swapped in the quantile solver shows up as a 99%
    # interval narrower than the 80% one.
    baseline, candidate = _binary_sample(seed, count, drift)
    widths = [
        stats.paired_binary(baseline, candidate, confidence).interval.high
        - stats.paired_binary(baseline, candidate, confidence).interval.low
        for confidence in _CONFIDENCES
    ]
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
            baseline, candidate, confidence=confidence, resamples=20_000, seed=seed
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


@pytest.mark.parametrize(("seed", "count", "drift"), _BINARY_SAMPLES)
def test_invariant_6_covers_both_corrections_on_real_p_values(
    seed: int, count: int, drift: float
) -> None:
    # Invariant 6 on p-values that came out of the module itself rather than out of a
    # literal list, so the two halves of the pipeline are checked joined up.
    baseline, candidate = _binary_sample(seed, count, drift)
    raw = [stats.paired_binary(baseline, candidate).p_value]
    raw.extend([0.001, 0.04, 0.6, 1.0])
    p_values = [value for value in raw if value is not None]

    for adjust in (stats.holm, stats.benjamini_hochberg):
        for before, after in zip(p_values, adjust(p_values), strict=True):
            assert before <= after + _ARITHMETIC_TOLERANCE
            assert after <= 1.0


# ---------------------------------------------------------------------------
# 8. The verdict (contract 6).
# ---------------------------------------------------------------------------


def test_verdict_values_are_the_contract_strings() -> None:
    # The verdict is written into a report and into CI output, so its spelling is part
    # of the contract, not an internal detail.
    assert str(stats.Verdict.NON_INFERIOR) == "non_inferior"
    assert str(stats.Verdict.REGRESSED) == "regressed"
    assert str(stats.Verdict.INCONCLUSIVE) == "inconclusive"


def test_judge_returns_non_inferior_when_the_interval_rules_out_the_regression() -> None:
    # Contract 6, row 1: low > -delta.
    interval = stats.Interval(point=-0.01, low=-0.03, high=0.01, confidence=0.95)
    assert stats.judge(interval, 0.05) is stats.Verdict.NON_INFERIOR


def test_judge_returns_regressed_when_the_interval_rules_out_being_within_tolerance() -> None:
    # Contract 6, row 2: high < -delta.
    interval = stats.Interval(point=-0.20, low=-0.30, high=-0.12, confidence=0.95)
    assert stats.judge(interval, 0.05) is stats.Verdict.REGRESSED


def test_judge_returns_inconclusive_when_the_interval_straddles_the_tolerance() -> None:
    # Contract 6, row 3, and the rule that matters most: "`inconclusive` is a
    # first-class state and must never collapse into `non_inferior`". A thin slice that
    # cannot detect the regression it was asked about is a gap in coverage, and
    # reporting it as a pass is the precise failure this project exists to prevent.
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
    # Contract 6: "`delta` must be strictly positive; a tolerance of zero turns the
    # question back into a significance test. Reject it." The exception type is not
    # pinned by the contract; ValueError is the Python convention for an unusable
    # argument value.
    interval = stats.Interval(point=-0.01, low=-0.03, high=0.01, confidence=0.95)
    with pytest.raises(ValueError):
        stats.judge(interval, tolerance)


# ---------------------------------------------------------------------------
# 9. The dependency rule (contract 9).
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
