"""The gate: score two results files, compare them slice by slice, and decide.

Owning artifact: `docs/statistics.md` for the mathematics.

Every slice the configuration names gets a line in the answer, including a slice with no items
and a slice with one. Dropping those would make the report look complete when it is not, and a
gap in coverage is exactly what a reader needs to see.

Two runs are only compared on items where both answered the same question. Each result can
record a hash of the input it ran on; where either side's hash disagrees with the dataset as it
stands now, that item is left out and listed. In CI the baseline runs from the base commit's
checkout, so a pull request that edits an item's input would otherwise pair an answer to the old
question with an answer to the new one.

Deliberate boundary: this decides. It does not read files, render anything or choose an exit
code. Corrections across many slices are not applied yet, so nothing here claims a family-wide
error rate; see `docs/statistics.md` section 5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from evalgate.config import Config, Rule, SliceRef, Tier
from evalgate.dataset import Dataset
from evalgate.results import DatasetRef, Results, input_digest
from evalgate.scorers import Score, ScoreKind, Scorer
from evalgate.stats import PairedResult, Verdict, judge, paired_binary, paired_continuous

__all__ = [
    "Comparison",
    "Gap",
    "GateError",
    "Outcome",
    "RunSummary",
    "SliceComparison",
    "compare",
    "score_results",
]


class GateError(Exception):
    """Two results files cannot be compared at all."""


class Gap(StrEnum):
    """Why a slice produced no interval."""

    NO_PAIRED = "no paired items"
    ONE_PAIRED = "one paired item"


class Outcome(StrEnum):
    """The answer for the whole check, before any policy turns it into an exit code."""

    OK = "ok"
    """No gate slice regressed, and every gate slice could rule out its tolerance."""

    UNGUARDED = "unguarded"
    """No gate slice regressed, but at least one could not rule out a regression its size.

    Never reported as a pass. Whether it fails the build is the configuration's call.
    """

    REGRESSED = "regressed"
    """At least one gate slice got worse by more than its tolerance."""


@dataclass(frozen=True)
class SliceComparison:
    """One slice's answer on one scorer.

    Attributes:
        tier: Whether this slice can fail the build.
        scorer: The scorer compared.
        kind: Binary or continuous, which picked the paired test.
        slice: The slice.
        size: Items the dataset holds in this slice.
        tolerance: How much worse the candidate was allowed to be.
        result: The paired comparison, or `None` when the slice could not support one.
        verdict: Non-inferior, regressed or inconclusive. A gap is always inconclusive.
        gap: Why there is no interval, when there is none.
        unpaired: Slice items that could not be compared: missing from a run, or answering a
            different input. Non-empty means this verdict rests on fewer items than `size`.
    """

    tier: Tier
    scorer: str
    kind: ScoreKind
    slice: SliceRef
    size: int
    tolerance: float
    result: PairedResult | None
    verdict: Verdict
    gap: Gap | None
    unpaired: tuple[str, ...]

    @property
    def n(self) -> int:
        """Items actually compared."""
        return self.result.n if self.result is not None else self.size - len(self.unpaired)

    @property
    def rules_out(self) -> float | None:
        """The largest drop this slice's data rules out, on the score scale.

        This is what turns "inconclusive" into something a reader can act on: a slice that rules
        out drops over 0.09 cannot certify a tolerance of 0.05, so either it needs more items or
        the tolerance needs widening on purpose. `None` when there is no interval.
        """
        if self.result is None:
            return None
        return max(0.0, -self.result.interval.low)


@dataclass(frozen=True)
class RunSummary:
    """What a report needs to say about one side of the comparison.

    Attributes:
        label: The run's label.
        dataset: The dataset its header names, when it has one.
        errors: Ids of items whose output was an error.
        unknown: Ids in the file that the dataset no longer holds. Ignored.
    """

    label: str
    dataset: DatasetRef | None
    errors: tuple[str, ...]
    unknown: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    """The whole answer.

    Attributes:
        dataset: The dataset slices and expectations came from.
        base: The baseline run.
        head: The run under test.
        slices: Every slice of every rule, gate rules first, in configuration order.
        input_changed: Items left out because a run answered a different input than the
            dataset now holds.
    """

    dataset: Dataset
    base: RunSummary
    head: RunSummary
    slices: tuple[SliceComparison, ...]
    input_changed: tuple[str, ...]

    @property
    def gate(self) -> tuple[SliceComparison, ...]:
        """Slices that can fail the build."""
        return tuple(entry for entry in self.slices if entry.tier is Tier.GATE)

    @property
    def watch(self) -> tuple[SliceComparison, ...]:
        """Slices that are reported only."""
        return tuple(entry for entry in self.slices if entry.tier is Tier.WATCH)

    @property
    def outcome(self) -> Outcome:
        """The answer for the check as a whole, read from the gate slices only."""
        verdicts = {entry.verdict for entry in self.gate}
        if Verdict.REGRESSED in verdicts:
            return Outcome.REGRESSED
        if Verdict.INCONCLUSIVE in verdicts:
            return Outcome.UNGUARDED
        return Outcome.OK


def score_results(
    dataset: Dataset, results: Results, scorers: Sequence[Scorer]
) -> dict[str, dict[str, Score]]:
    """Score every item of a results file against the dataset's expectations.

    An item whose output is an error scores the lowest possible value on every scorer. From the
    user's side a crash is a failure, and leaving it out would let a change that makes the
    feature crash on one slice look like a change that made that slice smaller.

    Args:
        dataset: Supplies each item's expectation.
        results: The outputs to score. Ids the dataset does not hold are skipped.
        scorers: The scorers to apply.

    Returns:
        Scores keyed by scorer name, then by item id.

    Raises:
        ScorerError: If a scorer cannot read an output. That is true of the configuration, not
            of one item, so it is not caught here.
    """
    expected = {item.id: item.expected for item in dataset}
    scored: dict[str, dict[str, Score]] = {scorer.name: {} for scorer in scorers}
    for result in results.items:
        if result.id not in expected:
            continue
        for scorer in scorers:
            if result.error is not None:
                scored[scorer.name][result.id] = Score(
                    value=0.0, kind=scorer.kind, detail={"error": result.error}
                )
            else:
                scored[scorer.name][result.id] = scorer.score(result.output, expected[result.id])
    return scored


def compare(config: Config, base: Results, head: Results) -> Comparison:
    """Compare two runs on every slice the configuration names.

    Args:
        config: Dataset, scorers, rules and confidence.
        base: The baseline run.
        head: The run under test.

    Returns:
        The comparison, with one entry per slice per rule.

    Raises:
        GateError: If either run names a different dataset than the configuration uses.
        ScorerError: If a scorer cannot read an output.
    """
    dataset = config.dataset
    for results in (base, head):
        named = results.header.dataset if results.header is not None else None
        if named is not None and named.name != dataset.manifest.name:
            raise GateError(
                f"{results.label}: produced from dataset {named.name!r}, but the "
                f"configuration uses {dataset.manifest.name!r}"
            )

    current = {item.id: input_digest(item.input) for item in dataset}
    changed = sorted(
        {
            result.id
            for results in (base, head)
            for result in results.items
            if result.id in current
            and result.input_sha256 is not None
            and result.input_sha256 != current[result.id]
        }
    )

    scorers = list(config.scorers.values())
    base_scores = score_results(dataset, base, scorers)
    head_scores = score_results(dataset, head, scorers)
    excluded = set(changed)

    slices = tuple(
        _compare_slice(
            rule,
            slice_ref,
            dataset,
            config.scorers[rule.scorer].kind,
            base_scores[rule.scorer],
            head_scores[rule.scorer],
            excluded,
            config.confidence,
        )
        for rule in config.rules
        for slice_ref in rule.slices
    )

    return Comparison(
        dataset=dataset,
        base=_summarise(base, current),
        head=_summarise(head, current),
        slices=slices,
        input_changed=tuple(changed),
    )


def _compare_slice(
    rule: Rule,
    slice_ref: SliceRef,
    dataset: Dataset,
    kind: ScoreKind,
    base: Mapping[str, Score],
    head: Mapping[str, Score],
    excluded: set[str],
    confidence: float,
) -> SliceComparison:
    """Pair one slice's items across the two runs and run the test the scorer's kind calls for."""
    members = [item.id for item in dataset.slice(slice_ref.dimension, slice_ref.value)]
    paired = [item for item in members if item in base and item in head and item not in excluded]
    unpaired = tuple(item for item in members if item not in paired)
    tolerance = rule.tolerance_for(slice_ref)

    gap: Gap | None = None
    result: PairedResult | None = None
    if not paired:
        gap = Gap.NO_PAIRED
    elif len(paired) == 1:
        # One observation cannot support an interval. Reporting the gap is honest; a verdict
        # resting on a single item is not.
        gap = Gap.ONE_PAIRED
    elif kind is ScoreKind.BINARY:
        result = paired_binary(
            [base[item].value == 1.0 for item in paired],
            [head[item].value == 1.0 for item in paired],
            confidence=confidence,
        )
    else:
        result = paired_continuous(
            [base[item].value for item in paired],
            [head[item].value for item in paired],
            confidence=confidence,
        )

    return SliceComparison(
        tier=rule.tier,
        scorer=rule.scorer,
        kind=kind,
        slice=slice_ref,
        size=len(members),
        tolerance=tolerance,
        result=result,
        verdict=Verdict.INCONCLUSIVE if result is None else judge(result.interval, tolerance),
        gap=gap,
        unpaired=unpaired,
    )


def _summarise(results: Results, current: Mapping[str, str]) -> RunSummary:
    """Collect what a report says about one run."""
    header = results.header
    return RunSummary(
        label=results.label,
        dataset=header.dataset if header is not None else None,
        errors=tuple(
            item.id for item in results.items if item.error is not None and item.id in current
        ),
        unknown=tuple(item.id for item in results.items if item.id not in current),
    )
