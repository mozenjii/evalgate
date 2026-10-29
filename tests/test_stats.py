"""Contract behaviour of the paired statistics.

These check the semantics the gate depends on: the sign convention, the three verdict states,
the edge cases the contract calls out, and the invariants. Agreement with scipy and statsmodels
is checked separately and independently in `tests/oracle/`.
"""

import math
from collections.abc import Callable, Sequence

import pytest

from evalgate.stats import (
    Interval,
    Verdict,
    benjamini_hochberg,
    clopper_pearson,
    holm,
    jackknife_acceleration,
    judge,
    mcnemar_exact_p,
    paired_binary,
    paired_continuous,
)

# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #


#: A p-value adjustment: raw values in, adjusted values out, same order.
Adjustment = Callable[[Sequence[float]], list[float]]


def test_interval_brackets_the_observed_proportion() -> None:
    low, high = clopper_pearson(30, 100)
    assert low < 0.30 < high


def test_zero_successes_pins_the_lower_bound_at_zero() -> None:
    low, high = clopper_pearson(0, 20)
    assert low == 0.0
    # Two-sided exact bound for zero events in twenty trials: 1 - 0.025**(1/20) ~= 0.168.
    assert 0.12 < high < 0.20


def test_all_successes_pins_the_upper_bound_at_one() -> None:
    low, high = clopper_pearson(20, 20)
    assert high == 1.0
    assert 0.80 < low < 0.88


def test_more_confidence_gives_a_wider_interval() -> None:
    narrow = clopper_pearson(30, 100, confidence=0.80)
    wide = clopper_pearson(30, 100, confidence=0.99)
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


def test_more_data_gives_a_narrower_interval() -> None:
    small = clopper_pearson(5, 10)
    large = clopper_pearson(500, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


@pytest.mark.parametrize(
    ("successes", "trials", "confidence"),
    [(-1, 10, 0.95), (11, 10, 0.95), (5, 0, 0.95), (5, 10, 0.0), (5, 10, 1.0)],
)
def test_rejects_impossible_arguments(successes: int, trials: int, confidence: float) -> None:
    with pytest.raises(ValueError):
        clopper_pearson(successes, trials, confidence)


# --------------------------------------------------------------------------- #
# McNemar
# --------------------------------------------------------------------------- #


def test_no_discordant_pairs_is_not_evidence_of_a_difference() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0


def test_balanced_discordance_is_not_evidence_either() -> None:
    assert mcnemar_exact_p(10, 10) == 1.0


def test_lopsided_discordance_is_evidence() -> None:
    # Twelve items moved one way and none the other; that is 2 * 0.5**12.
    assert mcnemar_exact_p(12, 0) == pytest.approx(2.0 * 0.5**12)


def test_mcnemar_is_symmetric_in_its_arguments() -> None:
    assert mcnemar_exact_p(9, 3) == mcnemar_exact_p(3, 9)


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact_p(-1, 3)


# --------------------------------------------------------------------------- #
# Paired binary
# --------------------------------------------------------------------------- #


def test_point_estimate_equals_the_difference_in_means() -> None:
    baseline = [True, True, False, False, True, False, True, False]
    candidate = [True, False, True, True, True, False, False, True]

    result = paired_binary(baseline, candidate)

    expected = sum(candidate) / len(candidate) - sum(baseline) / len(baseline)
    assert result.interval.point == pytest.approx(expected)


def test_interval_contains_the_point_estimate() -> None:
    baseline = [True] * 30 + [False] * 70
    candidate = [True] * 20 + [False] * 80

    result = paired_binary(baseline, candidate)

    assert result.interval.low <= result.interval.point <= result.interval.high


def test_swapping_the_versions_mirrors_the_interval() -> None:
    baseline = [True, False, True, True, False, False, True, False, True, False]
    candidate = [False, True, True, False, False, True, True, True, False, False]

    forward = paired_binary(baseline, candidate).interval
    reverse = paired_binary(candidate, baseline).interval

    assert forward.point == pytest.approx(-reverse.point)
    assert forward.low == pytest.approx(-reverse.high)
    assert forward.high == pytest.approx(-reverse.low)


def test_total_agreement_is_reported_as_uncertainty_not_as_proof() -> None:
    # The contract is explicit: zero disagreements on twenty items is not proof of equality,
    # and reporting [0, 0] here would let a thin slice pass as if it had been measured.
    outcomes = [True, False] * 10

    result = paired_binary(outcomes, outcomes)

    assert result.interval.point == 0.0
    assert result.interval.low < 0.0 < result.interval.high
    assert result.p_value == 1.0


def test_total_agreement_narrows_as_the_slice_grows() -> None:
    small = paired_binary([True] * 20, [True] * 20).interval
    large = paired_binary([True] * 400, [True] * 400).interval
    assert large.width < small.width


def test_method_is_recorded_on_the_result() -> None:
    result = paired_binary([True, False], [False, True])
    assert result.method == "exact_conditional"
    assert result.n == 2


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [([], []), ([True], [True, False])],
)
def test_paired_binary_rejects_unusable_input(baseline: list[bool], candidate: list[bool]) -> None:
    with pytest.raises(ValueError):
        paired_binary(baseline, candidate)


# --------------------------------------------------------------------------- #
# Jackknife acceleration
# --------------------------------------------------------------------------- #


def test_acceleration_is_zero_for_a_symmetric_sample() -> None:
    # Symmetric data has no skew for the acceleration to correct.
    assert jackknife_acceleration([-2.0, -1.0, 0.0, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


def test_acceleration_is_zero_when_every_value_is_identical() -> None:
    # The formula is 0/0 here; the contract fixes the answer at zero rather than leaving a nan.
    assert jackknife_acceleration([3.0] * 8) == 0.0


def test_acceleration_has_the_sign_of_the_skew() -> None:
    # Positive for right-skewed data. The leave-one-out deviations are large and positive for
    # the large values, so their cubes dominate the numerator. Getting this sign backwards
    # would tilt every bootstrap interval the wrong way.
    right_skewed = [0.0, 0.0, 0.0, 0.0, 10.0]
    left_skewed = [-value for value in right_skewed]
    assert jackknife_acceleration(right_skewed) > 0.0
    assert jackknife_acceleration(left_skewed) < 0.0


def test_acceleration_needs_more_than_one_value() -> None:
    with pytest.raises(ValueError):
        jackknife_acceleration([1.0])


# --------------------------------------------------------------------------- #
# Paired continuous
# --------------------------------------------------------------------------- #


def test_bootstrap_is_reproducible_under_a_seed() -> None:
    baseline = [float(i % 7) for i in range(60)]
    candidate = [value + 0.5 for value in baseline]

    first = paired_continuous(baseline, candidate, seed=11, resamples=2_000)
    second = paired_continuous(baseline, candidate, seed=11, resamples=2_000)

    assert first.interval == second.interval


def test_bootstrap_interval_brackets_the_mean_difference() -> None:
    baseline = [float(i % 5) for i in range(80)]
    # Differences that genuinely vary, so this exercises the bootstrap rather than the
    # collapse-to-a-point branch.
    candidate = [value - 0.3 + ((i * 7) % 5) / 10.0 for i, value in enumerate(baseline)]

    result = paired_continuous(baseline, candidate, seed=3, resamples=4_000)

    assert result.interval.low < result.interval.point < result.interval.high
    assert result.p_value is None
    assert result.method == "bootstrap_bca"


def test_a_constant_shift_survives_floating_point_noise() -> None:
    # Subtracting a constant in floating point leaves differences that disagree in the last
    # bit. Bootstrapping that noise once produced an interval too narrow to contain its own
    # point estimate, so near-constant differences must be recognised as constant.
    baseline = [float(i % 5) for i in range(80)]
    candidate = [value - 0.3 for value in baseline]

    result = paired_continuous(baseline, candidate, seed=3, resamples=4_000)

    assert result.interval.point == pytest.approx(-0.3)
    assert result.interval.low <= result.interval.point <= result.interval.high
    assert result.interval.width == pytest.approx(0.0, abs=1e-12)


def test_a_constant_shift_collapses_the_interval() -> None:
    # Every item moved by exactly the same amount, so there is nothing for a resample to vary.
    baseline = [1.0, 2.0, 3.0, 4.0]
    candidate = [value + 2.0 for value in baseline]

    result = paired_continuous(baseline, candidate, seed=1, resamples=500)

    assert result.interval.point == pytest.approx(2.0)
    assert result.interval.low == pytest.approx(2.0)
    assert result.interval.high == pytest.approx(2.0)


def test_bootstrap_narrows_as_the_slice_grows() -> None:
    def spread(size: int) -> float:
        baseline = [float((i * 37) % 11) for i in range(size)]
        candidate = [value + ((i * 13) % 5) / 10.0 for i, value in enumerate(baseline)]
        return paired_continuous(baseline, candidate, seed=5, resamples=4_000).interval.width

    assert spread(400) < spread(40)


@pytest.mark.parametrize(
    ("baseline", "candidate", "resamples"),
    [([], [], 100), ([1.0], [1.0], 100), ([1.0, 2.0], [1.0], 100), ([1.0, 2.0], [1.0, 2.0], 0)],
)
def test_paired_continuous_rejects_unusable_input(
    baseline: list[float], candidate: list[float], resamples: int
) -> None:
    with pytest.raises(ValueError):
        paired_continuous(baseline, candidate, resamples=resamples)


# --------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------- #


def test_holm_scales_the_smallest_p_value_by_the_family_size() -> None:
    assert holm([0.01, 0.04, 0.03])[0] == pytest.approx(0.03)


def test_holm_never_reduces_a_p_value() -> None:
    raw = [0.001, 0.02, 0.3, 0.5]
    assert all(adj >= p for adj, p in zip(holm(raw), raw, strict=True))


def test_benjamini_hochberg_is_no_stricter_than_holm() -> None:
    raw = [0.001, 0.008, 0.02, 0.04, 0.3]
    assert all(bh <= hl for bh, hl in zip(benjamini_hochberg(raw), holm(raw), strict=True))


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
def test_adjustment_preserves_input_order(adjust: Adjustment) -> None:
    # Returning sorted output is the likeliest bug here, and it would silently attach each
    # slice's verdict to the wrong slice.
    shuffled = [0.5, 0.001, 0.2, 0.01]
    ascending = sorted(shuffled)
    by_input = adjust(shuffled)
    by_sorted = adjust(ascending)
    for value, adjusted in zip(shuffled, by_input, strict=True):
        assert adjusted == pytest.approx(by_sorted[ascending.index(value)])


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
def test_adjustment_clamps_at_one(adjust: Adjustment) -> None:
    assert all(value <= 1.0 for value in adjust([0.6, 0.7, 0.9]))


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
def test_adjustment_handles_empty_and_single_inputs(adjust: Adjustment) -> None:
    assert adjust([]) == []
    assert adjust([0.02]) == pytest.approx([0.02])


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
def test_adjustment_is_monotone_in_rank(adjust: Adjustment) -> None:
    raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    adjusted = adjust(raw)
    assert adjusted == sorted(adjusted)


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
def test_adjustment_rejects_values_outside_the_unit_interval(adjust: Adjustment) -> None:
    with pytest.raises(ValueError):
        adjust([0.5, 1.5])


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #


def test_a_clearly_safe_interval_passes() -> None:
    assert judge(Interval(0.01, -0.01, 0.03, 0.95), tolerance=0.05) is Verdict.NON_INFERIOR


def test_a_clearly_bad_interval_fails() -> None:
    assert judge(Interval(-0.20, -0.28, -0.12, 0.95), tolerance=0.05) is Verdict.REGRESSED


def test_a_straddling_interval_is_inconclusive_and_not_a_pass() -> None:
    # This is the case the whole project exists for: a slice too thin to answer the question
    # must not be reported as having answered it.
    assert judge(Interval(-0.02, -0.18, 0.14, 0.95), tolerance=0.05) is Verdict.INCONCLUSIVE


def test_the_boundary_belongs_to_inconclusive() -> None:
    # Exactly on the tolerance, nothing is ruled out, so the benefit of the doubt is withheld.
    assert judge(Interval(-0.05, -0.05, 0.01, 0.95), tolerance=0.05) is Verdict.INCONCLUSIVE


@pytest.mark.parametrize("tolerance", [0.0, -0.01])
def test_a_non_positive_tolerance_is_refused(tolerance: float) -> None:
    with pytest.raises(ValueError):
        judge(Interval(0.0, -0.1, 0.1, 0.95), tolerance)


def test_every_verdict_is_reachable() -> None:
    reached = {
        judge(Interval(0.0, -0.01, 0.01, 0.95), 0.05),
        judge(Interval(-0.3, -0.4, -0.2, 0.95), 0.05),
        judge(Interval(0.0, -0.5, 0.5, 0.95), 0.05),
    }
    assert reached == set(Verdict)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def test_wider_confidence_never_narrows_a_binary_interval() -> None:
    baseline = [i % 3 == 0 for i in range(120)]
    candidate = [i % 4 == 0 for i in range(120)]

    narrow = paired_binary(baseline, candidate, confidence=0.80).interval
    wide = paired_binary(baseline, candidate, confidence=0.99).interval

    assert wide.low <= narrow.low
    assert wide.high >= narrow.high


def test_interval_width_is_the_distance_between_its_bounds() -> None:
    interval = Interval(0.0, -0.1, 0.25, 0.95)
    assert interval.width == pytest.approx(0.35)


def test_binary_bounds_stay_on_the_difference_scale() -> None:
    # A difference of proportions cannot leave [-1, 1]; a bound outside it would mean the
    # conditional scaling had been applied wrongly.
    baseline = [True] * 10 + [False] * 10
    candidate = [False] * 20

    interval = paired_binary(baseline, candidate).interval

    assert -1.0 <= interval.low <= interval.high <= 1.0
    assert not math.isnan(interval.low)


def test_an_unseeded_bootstrap_is_still_reproducible() -> None:
    # The gate must not be able to change its mind between two builds of the same commit, and
    # the unseeded call is the common path, so it cannot be left to a fresh random stream.
    baseline = [float((i * 17) % 13) for i in range(50)]
    candidate = [value + ((i * 11) % 7) / 10.0 for i, value in enumerate(baseline)]

    first = paired_continuous(baseline, candidate, resamples=2_000)
    second = paired_continuous(baseline, candidate, resamples=2_000)

    assert first.interval == second.interval


def test_different_data_gets_a_different_resample_stream() -> None:
    # Deriving the seed from the data must not collapse into one fixed seed for everything.
    first = [float((i * 17) % 13) for i in range(50)]
    second = [value + 1.0 if index % 3 == 0 else value for index, value in enumerate(first)]
    shifted = [value + 0.25 for value in first]

    one = paired_continuous(first, shifted, resamples=2_000)
    two = paired_continuous(second, shifted, resamples=2_000)

    assert one.interval != two.interval


def test_ties_in_the_resample_distribution_count_as_half() -> None:
    # Differences taking few distinct values make exact ties common. Counting only the strictly
    # smaller resample means would bias the correction downwards on every such slice.
    baseline = [0.0] * 40
    candidate = [0.0 if index % 2 else 1.0 for index in range(40)]

    result = paired_continuous(baseline, candidate, seed=7, resamples=4_000)

    assert result.interval.low < result.interval.point < result.interval.high
    assert result.interval.low >= 0.0 and result.interval.high <= 1.0
