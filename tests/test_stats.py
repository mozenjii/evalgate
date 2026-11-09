"""Contract behaviour of the paired statistics.

These check the semantics the gate depends on: the sign convention, the three verdict states,
the edge cases the contract calls out, the correction across slices, and the invariants.
Agreement with scipy and statsmodels is checked separately and independently in `tests/oracle/`.
"""

import math
from collections.abc import Callable, Sequence

import pytest

from evalgate.stats import (
    MIN_MOVED_FOR_BOOTSTRAP,
    Correction,
    Interval,
    PairedResult,
    Verdict,
    benjamini_hochberg,
    clopper_pearson,
    family_verdicts,
    holm,
    jackknife_acceleration,
    judge,
    paired_binary,
    paired_continuous,
    tango_interval,
    tango_statistic,
)

#: A p-value adjustment: raw values in, adjusted values out, same order.
Adjustment = Callable[[Sequence[float]], list[float]]

TOLERANCE = 0.05


def _varied(size: int, shift: float = 0.0) -> tuple[list[float], list[float]]:
    """Scores in [0, 1] whose differences genuinely vary, so the bootstrap runs."""
    baseline = [0.2 + ((i * 37) % 11) / 20.0 for i in range(size)]
    candidate = [
        min(1.0, max(0.0, value + shift + (((i * 13) % 5) - 2) / 20.0))
        for i, value in enumerate(baseline)
    ]
    return baseline, candidate


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #


def test_interval_brackets_the_observed_proportion() -> None:
    low, high = clopper_pearson(30, 100)
    assert low < 0.30 < high


def test_zero_successes_pins_the_lower_bound_at_zero() -> None:
    low, high = clopper_pearson(0, 20)
    assert low == 0.0
    # Two-sided exact bound for zero events in twenty trials: 1 - 0.025**(1/20) ~= 0.168.
    assert high == pytest.approx(1.0 - 0.025 ** (1.0 / 20.0))


def test_all_successes_pins_the_upper_bound_at_one() -> None:
    low, high = clopper_pearson(20, 20)
    assert high == 1.0
    assert 0.80 < low < 0.88


def test_more_confidence_gives_a_wider_interval() -> None:
    narrow = clopper_pearson(30, 100, confidence=0.80)
    wide = clopper_pearson(30, 100, confidence=0.99)
    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


@pytest.mark.parametrize(
    ("successes", "trials", "confidence"),
    [(-1, 10, 0.95), (11, 10, 0.95), (5, 0, 0.95), (5, 10, 0.0), (5, 10, 1.0)],
)
def test_rejects_impossible_arguments(successes: int, trials: int, confidence: float) -> None:
    with pytest.raises(ValueError):
        clopper_pearson(successes, trials, confidence)


# --------------------------------------------------------------------------- #
# Tango's score test
# --------------------------------------------------------------------------- #


def test_the_statistic_is_zero_at_the_observed_difference() -> None:
    assert tango_statistic(9, 3, 40, (9 - 3) / 40) == pytest.approx(0.0, abs=1e-12)


def test_the_statistic_falls_as_the_hypothesis_rises() -> None:
    values = [tango_statistic(5, 12, 60, d / 10.0) for d in range(-9, 10)]
    assert values == sorted(values, reverse=True)


def test_no_disagreement_gives_the_closed_form_width() -> None:
    # With nothing to go on, the score interval is +-z^2/(n + z^2), not a point.
    z = 1.959963984540054
    low, high = tango_interval(0, 0, 20)
    assert high == pytest.approx(z * z / (20 + z * z), abs=1e-9)
    assert low == pytest.approx(-high, abs=1e-12)


def test_every_item_favouring_the_baseline_leaves_the_worst_case_open() -> None:
    low, high = tango_interval(0, 20, 20)
    assert low == -1.0
    assert high < -0.5


def test_one_sided_breakage_is_not_pinned_to_the_observed_drop() -> None:
    # The interval this replaced put the lower bound exactly on the point estimate whenever
    # every disagreement went one way, so it could not say "maybe worse than observed". That is
    # how it passed slices that were truly worse than their tolerance.
    low, _ = tango_interval(0, 12, 20)
    assert low < -0.6 - 0.1


@pytest.mark.parametrize(
    ("favour_candidate", "favour_baseline", "n", "difference"),
    [(-1, 0, 10, 0.0), (0, -1, 10, 0.0), (6, 6, 10, 0.0), (0, 0, 0, 0.0), (1, 1, 10, 1.5)],
)
def test_tango_rejects_impossible_arguments(
    favour_candidate: int, favour_baseline: int, n: int, difference: float
) -> None:
    with pytest.raises(ValueError):
        tango_statistic(favour_candidate, favour_baseline, n, difference)


# --------------------------------------------------------------------------- #
# Paired binary
# --------------------------------------------------------------------------- #


def test_point_estimate_equals_the_difference_in_means() -> None:
    baseline = [True, True, False, False, True, False, True, False]
    candidate = [True, False, True, True, True, False, False, True]

    result = paired_binary(baseline, candidate, tolerance=TOLERANCE)

    expected = sum(candidate) / len(candidate) - sum(baseline) / len(baseline)
    assert result.interval.point == pytest.approx(expected)


def test_swapping_the_versions_mirrors_the_interval() -> None:
    baseline = [True, False, True, True, False, False, True, False, True, False]
    candidate = [False, True, True, False, False, True, True, True, False, False]

    forward = paired_binary(baseline, candidate, tolerance=TOLERANCE).interval
    reverse = paired_binary(candidate, baseline, tolerance=TOLERANCE).interval

    assert forward.point == pytest.approx(-reverse.point)
    assert forward.low == pytest.approx(-reverse.high)
    assert forward.high == pytest.approx(-reverse.low)


def test_total_agreement_is_reported_as_uncertainty_not_as_proof() -> None:
    # Zero disagreements on twenty items is not proof of equality, and reporting [0, 0] here
    # would let a thin slice pass as if it had been measured.
    outcomes = [True, False] * 10

    result = paired_binary(outcomes, outcomes, tolerance=TOLERANCE)

    assert result.interval.point == 0.0
    assert result.interval.low < 0.0 < result.interval.high
    assert judge(result.interval, TOLERANCE) is Verdict.INCONCLUSIVE


def test_total_agreement_on_a_large_slice_is_safe() -> None:
    result = paired_binary([True] * 400, [True] * 400, tolerance=TOLERANCE)
    assert judge(result.interval, TOLERANCE) is Verdict.NON_INFERIOR


@pytest.mark.parametrize("drop", [0, 2, 4, 6, 9, 14])
def test_binary_p_values_agree_with_the_interval(drop: int) -> None:
    # The interval and the p-values are one test read two ways, so they cannot disagree.
    baseline = [True] * 60 + [False] * 40
    candidate = [False] * drop + [True] * (60 - drop) + [False] * 40

    result = paired_binary(baseline, candidate, tolerance=TOLERANCE)

    assert (result.interval.low > -TOLERANCE) == (result.p_non_inferior < 0.025)
    assert (result.interval.high < -TOLERANCE) == (result.p_regressed < 0.025)
    assert result.p_non_inferior + result.p_regressed == pytest.approx(1.0)


def test_method_and_tolerance_are_recorded_on_the_result() -> None:
    result = paired_binary([True, False], [False, True], tolerance=0.1)
    assert result.method == "tango_score"
    assert result.tolerance == 0.1
    assert result.n == 2


@pytest.mark.parametrize(
    ("baseline", "candidate", "tolerance"),
    [([], [], 0.05), ([True], [True, False], 0.05), ([True], [True], 0.0)],
)
def test_paired_binary_rejects_unusable_input(
    baseline: list[bool], candidate: list[bool], tolerance: float
) -> None:
    with pytest.raises(ValueError):
        paired_binary(baseline, candidate, tolerance=tolerance)


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
    # Positive for right-skewed data. Getting this sign backwards would tilt every bootstrap
    # interval the wrong way.
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
    baseline, candidate = _varied(60)

    first = paired_continuous(baseline, candidate, tolerance=TOLERANCE, seed=11, resamples=2_000)
    second = paired_continuous(baseline, candidate, tolerance=TOLERANCE, seed=11, resamples=2_000)

    assert first == second


def test_bootstrap_interval_brackets_the_mean_difference() -> None:
    baseline, candidate = _varied(80, shift=-0.02)

    result = paired_continuous(baseline, candidate, tolerance=TOLERANCE, seed=3, resamples=4_000)

    assert result.method == "bootstrap_bca"
    assert result.interval.low < result.interval.point < result.interval.high
    assert result.p_non_inferior + result.p_regressed == pytest.approx(1.0)


def test_nothing_moved_is_not_proof_of_nothing_moving() -> None:
    # Before the count bound this returned [0, 0] and every such slice passed, including ones
    # where a real regression was simply not sampled.
    scores = [0.1 * (i % 10) for i in range(20)]

    result = paired_continuous(scores, scores, tolerance=TOLERANCE)

    assert result.method == "count_bound"
    bound = 1.0 - 0.025 ** (1.0 / 20.0)
    assert result.interval.low == pytest.approx(-bound, abs=1e-9)
    assert result.interval.high == pytest.approx(bound, abs=1e-9)
    assert judge(result.interval, TOLERANCE) is Verdict.INCONCLUSIVE


def test_too_few_moved_items_use_the_count_bound() -> None:
    size = 60
    baseline = [0.5] * size
    moved = MIN_MOVED_FOR_BOOTSTRAP - 1
    candidate = [0.2] * 4 + [0.9] * (moved - 4) + [0.5] * (size - moved)

    result = paired_continuous(baseline, candidate, tolerance=TOLERANCE)

    assert result.method == "count_bound"
    assert result.interval.low == pytest.approx(-clopper_pearson(4, size)[1])
    assert result.interval.high == pytest.approx(clopper_pearson(moved - 4, size)[1])
    assert result.p_regressed == 1.0


def test_enough_moved_items_are_bootstrapped() -> None:
    size = 60
    baseline = [0.5] * size
    moved = MIN_MOVED_FOR_BOOTSTRAP
    candidate = [0.2] * 4 + [0.9] * (moved - 4) + [0.5] * (size - moved)

    result = paired_continuous(baseline, candidate, tolerance=TOLERANCE)

    assert result.method == "bootstrap_bca"


def test_some_items_moving_by_the_same_amount_is_not_a_constant_shift() -> None:
    # Ten of forty moved identically and thirty did not move: the differences are not all the
    # same, so the bootstrap runs. The constant-shift bound assumes every item moved.
    baseline = [0.5] * 40
    candidate = [0.4] * 10 + [0.5] * 30

    result = paired_continuous(baseline, candidate, tolerance=TOLERANCE)

    assert result.method == "bootstrap_bca"
    assert result.interval.point == pytest.approx(-0.025)


def test_a_constant_shift_survives_floating_point_noise() -> None:
    # Subtracting a constant in floating point leaves differences that disagree in the last
    # bit. Bootstrapping that noise once produced an interval too narrow to contain its own
    # point estimate, so near-constant differences must be recognised as constant.
    baseline = [0.31 + (i % 5) / 10.0 for i in range(80)]
    candidate = [value - 0.3 for value in baseline]

    result = paired_continuous(baseline, candidate, tolerance=TOLERANCE)

    assert result.method == "constant_shift"
    assert result.interval.point == pytest.approx(-0.3)
    assert result.interval.low < result.interval.point < result.interval.high


def test_a_large_constant_drop_is_a_regression() -> None:
    baseline = [0.8] * 200
    candidate = [0.6] * 200

    result = paired_continuous(baseline, candidate, tolerance=TOLERANCE)

    assert judge(result.interval, TOLERANCE) is Verdict.REGRESSED
    assert result.p_regressed < 0.025


def test_bootstrap_narrows_as_the_slice_grows() -> None:
    def width(size: int) -> float:
        baseline, candidate = _varied(size)
        result = paired_continuous(
            baseline, candidate, tolerance=TOLERANCE, seed=5, resamples=4_000
        )
        return result.interval.width

    assert width(400) < width(40)


@pytest.mark.parametrize(
    ("baseline", "candidate", "resamples", "tolerance"),
    [
        ([], [], 100, 0.05),
        ([0.1], [0.1], 100, 0.05),
        ([0.1, 0.2], [0.1], 100, 0.05),
        ([0.1, 0.2], [0.1, 0.2], 0, 0.05),
        ([0.1, 0.2], [0.1, 0.2], 100, 0.0),
        ([0.1, 1.2], [0.1, 0.2], 100, 0.05),
        ([0.1, 0.2], [-0.1, 0.2], 100, 0.05),
    ],
)
def test_paired_continuous_rejects_unusable_input(
    baseline: list[float], candidate: list[float], resamples: int, tolerance: float
) -> None:
    with pytest.raises(ValueError):
        paired_continuous(baseline, candidate, tolerance=tolerance, resamples=resamples)


# --------------------------------------------------------------------------- #
# Multiplicity
# --------------------------------------------------------------------------- #


def test_holm_scales_the_smallest_p_value_by_the_family_size() -> None:
    assert holm([0.01, 0.04, 0.03])[0] == pytest.approx(0.03)


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
def test_adjustment_handles_empty_and_single_inputs(adjust: Adjustment) -> None:
    assert adjust([]) == []
    assert adjust([0.02]) == pytest.approx([0.02])


@pytest.mark.parametrize("adjust", [holm, benjamini_hochberg])
def test_adjustment_is_monotone_in_rank_and_clamped(adjust: Adjustment) -> None:
    raw = [0.001, 0.01, 0.02, 0.04, 0.2, 0.9]
    adjusted = adjust(raw)
    assert adjusted == sorted(adjusted)
    assert all(value <= 1.0 for value in adjust([0.6, 0.7, 0.9]))


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


@pytest.mark.parametrize("tolerance", [0.0, -0.01, 1.5, float("nan")])
def test_a_tolerance_outside_the_score_range_is_refused(tolerance: float) -> None:
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
# A family of slices
# --------------------------------------------------------------------------- #


def _result(low: float, high: float, p_regressed: float, confidence: float = 0.95) -> PairedResult:
    return PairedResult(
        n=100,
        interval=Interval((low + high) / 2.0, low, high, confidence),
        method="tango_score",
        tolerance=TOLERANCE,
        p_non_inferior=1.0 - p_regressed,
        p_regressed=p_regressed,
    )


def test_a_lone_regression_stands_on_its_own_evidence() -> None:
    verdicts = family_verdicts([_result(-0.2, -0.06, 0.02)], Correction.HOLM)
    assert verdicts == [Verdict.REGRESSED]


def test_a_marginal_regression_among_many_slices_is_not_called_one() -> None:
    # On its own this slice regressed at p = 0.02. With five slices tested, Holm puts it at 0.10,
    # which is the sort of result one slice in five produces by chance.
    family = [_result(-0.2, -0.06, 0.02)] + [_result(-0.03, 0.04, 0.9)] * 4

    verdicts = family_verdicts(family, Correction.HOLM)

    assert verdicts[0] is Verdict.INCONCLUSIVE
    assert verdicts[1:] == [Verdict.NON_INFERIOR] * 4


def test_a_clear_regression_survives_the_correction() -> None:
    family = [_result(-0.4, -0.2, 0.0001)] + [_result(-0.03, 0.04, 0.9)] * 4
    assert family_verdicts(family, Correction.HOLM)[0] is Verdict.REGRESSED


def test_non_inferiority_is_never_corrected() -> None:
    # The check passes only if every slice passes, which already holds its false-pass rate to
    # one slice's. Correcting this side would only cost power.
    family = [_result(-0.04, 0.02, 0.99)] * 30
    assert family_verdicts(family, Correction.HOLM) == [Verdict.NON_INFERIOR] * 30


def test_benjamini_hochberg_is_gentler_on_watch_slices() -> None:
    family = [_result(-0.2, -0.06, 0.01), _result(-0.2, -0.06, 0.012), _result(-0.2, -0.06, 0.014)]
    assert family_verdicts(family, Correction.HOLM)[2] is Verdict.INCONCLUSIVE
    assert family_verdicts(family, Correction.BENJAMINI_HOCHBERG) == [Verdict.REGRESSED] * 3


def test_a_family_must_share_a_confidence_level() -> None:
    with pytest.raises(ValueError):
        family_verdicts([_result(-0.1, 0.1, 0.5), _result(-0.1, 0.1, 0.5, 0.9)], Correction.HOLM)


def test_an_empty_family_has_no_verdicts() -> None:
    assert family_verdicts([], Correction.HOLM) == []


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def test_wider_confidence_never_narrows_a_binary_interval() -> None:
    baseline = [i % 3 == 0 for i in range(120)]
    candidate = [i % 4 == 0 for i in range(120)]

    narrow = paired_binary(baseline, candidate, tolerance=TOLERANCE, confidence=0.80).interval
    wide = paired_binary(baseline, candidate, tolerance=TOLERANCE, confidence=0.99).interval

    assert wide.low <= narrow.low
    assert wide.high >= narrow.high


def test_interval_width_is_the_distance_between_its_bounds() -> None:
    interval = Interval(0.0, -0.1, 0.25, 0.95)
    assert interval.width == pytest.approx(0.35)


def test_binary_bounds_stay_on_the_difference_scale() -> None:
    baseline = [True] * 10 + [False] * 10
    candidate = [False] * 20

    interval = paired_binary(baseline, candidate, tolerance=TOLERANCE).interval

    assert -1.0 <= interval.low <= interval.high <= 1.0
    assert not math.isnan(interval.low)


def test_an_unseeded_bootstrap_is_still_reproducible() -> None:
    # The gate must not be able to change its mind between two builds of the same commit, and
    # the unseeded call is the common path, so it cannot be left to a fresh random stream.
    baseline, candidate = _varied(50)

    first = paired_continuous(baseline, candidate, tolerance=TOLERANCE, resamples=2_000)
    second = paired_continuous(baseline, candidate, tolerance=TOLERANCE, resamples=2_000)

    assert first == second


def test_different_data_gets_a_different_resample_stream() -> None:
    # Deriving the seed from the data must not collapse into one fixed seed for everything.
    baseline, candidate = _varied(50)
    other = [
        min(1.0, value + 0.05) if index % 3 == 0 else value for index, value in enumerate(baseline)
    ]

    one = paired_continuous(baseline, candidate, tolerance=TOLERANCE, resamples=2_000)
    two = paired_continuous(other, candidate, tolerance=TOLERANCE, resamples=2_000)

    assert one.interval != two.interval


def test_ties_in_the_resample_distribution_count_as_half() -> None:
    # Differences taking few distinct values make exact ties common: here every resample mean is
    # a multiple of 1/40, and about one in eight lands exactly on the observed mean. The data are
    # symmetric and so is the acceleration, so counting ties as half leaves the interval centred.
    # Counting only the strictly smaller means pushes the bias correction down and slides the
    # whole interval a full grid step (0.025) lower -- an asymmetry of 0.05, measured over twenty
    # seeds. Half a grid step separates the two with room on both sides.
    baseline = [0.0] * 40
    candidate = [0.0 if index % 2 else 1.0 for index in range(40)]

    interval = paired_continuous(
        baseline, candidate, tolerance=TOLERANCE, seed=7, resamples=20_000
    ).interval

    above = interval.high - interval.point
    below = interval.point - interval.low
    assert abs(above - below) < 0.0125
