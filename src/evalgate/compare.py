"""Turn two recorded runs into a per-slice verdict.

Owning artifact: `docs/statistics.md` for the mathematics.

This is the join between the stored runs and the statistics: pair two runs by item within a
slice, pick the paired test the scorer's kind calls for, and read the interval against that
slice's tolerance.

Deliberate boundary: this decides a verdict per slice. It does not render anything and it does
not choose an exit code. Those belong to the report and the gate, and keeping them out means a
verdict can be inspected without a terminal attached.

Two things it refuses to hide. A slice whose interval straddles the tolerance is reported as
inconclusive and never as a pass. And a comparison that could not pair every item says so,
because the item count is what a reader uses to judge whether a verdict means anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from evalgate.dataset import Dataset
from evalgate.runs import Pairing, RunStore
from evalgate.scorers import ScoreKind
from evalgate.stats import PairedResult, Verdict, holm, judge, paired_binary, paired_continuous

__all__ = ["SliceComparison", "compare_slice", "compare_slices"]


@dataclass(frozen=True)
class SliceComparison:
    """One slice's answer.

    Attributes:
        dimension: The slice dimension, for example `intent`.
        value: The slice value, for example `billing`.
        scorer: Which scorer this compares.
        tolerance: How much worse the candidate was allowed to be.
        result: The paired comparison, with its interval and method.
        verdict: Non-inferior, regressed, or inconclusive.
        unpaired: Items scored in only one of the two runs. Non-empty means this verdict
            covers fewer items than the slice contains.
        adjusted_p_value: The p-value after correction across the family of gate slices, or
            `None` for a continuous scorer, which produces no p-value.
    """

    dimension: str
    value: str
    scorer: str
    tolerance: float
    result: PairedResult
    verdict: Verdict
    unpaired: tuple[str, ...] = ()
    adjusted_p_value: float | None = None

    @property
    def is_complete(self) -> bool:
        """Whether every item in the slice was scored in both runs."""
        return not self.unpaired


def compare_slice(pairing: Pairing, tolerance: float, *, confidence: float = 0.95) -> PairedResult:
    """Run the paired test the scorer's kind calls for.

    The kind is taken from the recorded scores rather than inferred from the values, because a
    binary slice where every item passed looks continuous and would silently get the wrong
    test.

    Args:
        pairing: Two runs' values for one scorer, lined up by item.
        tolerance: Unused here; the verdict is applied by the caller. Present so the signature
            reads the same as the rest of this module.
        confidence: Nominal confidence level for the interval.

    Returns:
        The paired result.

    Raises:
        ValueError: If the pairing is empty, or too small for the chosen test.
    """
    del tolerance
    if pairing.kind is ScoreKind.BINARY:
        return paired_binary(
            [value >= 0.5 for value in pairing.baseline],
            [value >= 0.5 for value in pairing.candidate],
            confidence=confidence,
        )
    return paired_continuous(list(pairing.baseline), list(pairing.candidate), confidence=confidence)


def compare_slices(
    store: RunStore,
    dataset: Dataset,
    baseline_run: str,
    candidate_run: str,
    scorer: str,
    *,
    tolerances: dict[tuple[str, str], float] | None = None,
    default_tolerance: float = 0.05,
    confidence: float = 0.95,
) -> list[SliceComparison]:
    """Compare two runs across every declared slice of a dataset.

    P-values are corrected with Holm across the slices compared here, because they are one
    family of questions asked of one change; leaving them uncorrected would make a false alarm
    somewhere nearly certain once a dataset has a dozen slices. The correction is reported but
    does not drive the verdict, which reads the interval.

    Args:
        store: Where the runs live.
        dataset: The dataset both runs used, which declares the slices.
        baseline_run: The run to compare against.
        candidate_run: The run under test.
        scorer: Which scorer's values to compare.
        tolerances: Per-slice tolerance, keyed by `(dimension, value)`. A slice's tolerance is
            a judgement about that slice and belongs to a person, not to a default.
        default_tolerance: Used for slices with no entry in `tolerances`.
        confidence: Nominal confidence level.

    Returns:
        One comparison per non-empty slice, in the dataset's declared order. Empty slices are
        skipped here and reported as coverage gaps by the caller, which knows the full list.
    """
    tolerances = tolerances or {}
    comparisons: list[SliceComparison] = []

    for dimension, values in dataset.manifest.slice_dimensions.items():
        for value in values:
            items = [item.id for item in dataset.slice(dimension, value)]
            if not items:
                continue

            pairing = store.pair(baseline_run, candidate_run, scorer, items=items)
            if len(pairing) < 2:
                # One item cannot support an interval. Saying so is the honest report; the
                # alternative is a verdict resting on a single observation.
                continue

            tolerance = tolerances.get((dimension, value), default_tolerance)
            result = compare_slice(pairing, tolerance, confidence=confidence)
            comparisons.append(
                SliceComparison(
                    dimension=dimension,
                    value=value,
                    scorer=scorer,
                    tolerance=tolerance,
                    result=result,
                    verdict=judge(result.interval, tolerance),
                    unpaired=tuple(
                        sorted(set(pairing.only_in_baseline) | set(pairing.only_in_candidate))
                    ),
                )
            )

    return _with_adjusted_p_values(comparisons)


def _with_adjusted_p_values(comparisons: list[SliceComparison]) -> list[SliceComparison]:
    """Attach Holm-adjusted p-values across the slices that produced one.

    Slices with no p-value, which is every continuous scorer, are left alone rather than being
    given a placeholder: a number that looks like a p-value but is not one is worse than none.
    """
    indexed = [
        (index, comparison.result.p_value)
        for index, comparison in enumerate(comparisons)
        if comparison.result.p_value is not None
    ]
    if not indexed:
        return comparisons

    adjusted = holm([p_value for _, p_value in indexed])
    by_index = {index: value for (index, _), value in zip(indexed, adjusted, strict=True)}

    return [
        SliceComparison(
            dimension=comparison.dimension,
            value=comparison.value,
            scorer=comparison.scorer,
            tolerance=comparison.tolerance,
            result=comparison.result,
            verdict=comparison.verdict,
            unpaired=comparison.unpaired,
            adjusted_p_value=by_index.get(index),
        )
        for index, comparison in enumerate(comparisons)
    ]
