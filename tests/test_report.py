"""The report puts the worst first, never calls a gap a pass, and says what was left out."""

from typing import Literal

from evalgate.config import SliceRef, Tier
from evalgate.dataset import Dataset, Item, Manifest, Provenance
from evalgate.gate import Comparison, Gap, RunSummary, SliceComparison
from evalgate.report import MARKER, render_markdown, render_text
from evalgate.results import DatasetRef
from evalgate.scorers import ScoreKind
from evalgate.stats import Interval, PairedResult, Verdict

SHA = "a" * 64


def dataset() -> Dataset:
    manifest = Manifest(
        name="triage",
        version="2",
        items_file="items.jsonl",
        content_sha256=SHA,
        slice_dimensions={"intent": ("billing", "refunds", "warranty")},
    )
    item = Item(
        id="billing-00",
        input="x",
        expected="billing",
        slices={"intent": "billing"},
        provenance=Provenance.SYNTHETIC,
    )
    return Dataset(manifest=manifest, items=(item,))


def entry(
    value: str,
    verdict: Verdict,
    interval: tuple[float, float, float] | None,
    *,
    tier: Tier = Tier.GATE,
    size: int = 40,
    gap: Gap | None = None,
    unpaired: tuple[str, ...] = (),
) -> SliceComparison:
    result = None
    if interval is not None:
        point, low, high = interval
        result = PairedResult(
            n=size - len(unpaired),
            interval=Interval(point=point, low=low, high=high, confidence=0.95),
            method="exact_conditional",
            p_value=None,
        )
    return SliceComparison(
        tier=tier,
        scorer="exact_match",
        kind=ScoreKind.BINARY,
        slice=SliceRef("intent", value),
        size=size,
        tolerance=0.05,
        result=result,
        verdict=verdict,
        gap=gap,
        unpaired=unpaired,
    )


def side(
    label: str,
    *,
    dataset: DatasetRef | None = None,
    errors: tuple[str, ...] = (),
    unknown: tuple[str, ...] = (),
) -> RunSummary:
    return RunSummary(label=label, dataset=dataset, errors=errors, unknown=unknown)


def comparison(
    *entries: SliceComparison,
    base: RunSummary | None = None,
    head: RunSummary | None = None,
    input_changed: tuple[str, ...] = (),
) -> Comparison:
    return Comparison(
        dataset=dataset(),
        base=base or side("main"),
        head=head or side("pr-42"),
        slices=entries,
        input_changed=input_changed,
    )


REGRESSED = entry("warranty", Verdict.REGRESSED, (-0.6, -0.6, -0.429))
UNGUARDED = entry("billing", Verdict.INCONCLUSIVE, (0.0, -0.088, 0.088))
FINE = entry("refunds", Verdict.NON_INFERIOR, (0.0, -0.02, 0.03))


def markdown(value: Comparison, policy: Literal["warn", "fail"] = "warn") -> str:
    return render_markdown(value, confidence=0.95, on_inconclusive=policy)


def test_the_comment_starts_with_the_marker_the_action_looks_for() -> None:
    assert markdown(comparison(FINE)).startswith(MARKER)


def test_the_headline_says_regression_when_one_slice_regressed() -> None:
    text = markdown(comparison(FINE, UNGUARDED, REGRESSED))

    assert "### evalgate: regression found" in text
    assert "**1 regressed · 1 not guarded · 1 ok**" in text


def test_an_unguarded_check_is_never_headlined_as_a_pass() -> None:
    text = markdown(comparison(FINE, UNGUARDED))

    assert "no regression found, but 1 gate slice is not guarded" in text


def test_a_fully_guarded_check_says_so_plainly() -> None:
    assert "### evalgate: no regression found\n" in markdown(comparison(FINE))


def test_the_worst_slices_come_first() -> None:
    text = markdown(comparison(FINE, UNGUARDED, REGRESSED))

    assert (
        text.index("intent=warranty") < text.index("intent=billing") < text.index("intent=refunds")
    )


def test_differences_are_in_signed_points_with_their_interval() -> None:
    text = markdown(comparison(REGRESSED))

    assert "-60.0 (-60.0 to -42.9)" in text
    assert "**regressed**" in text


def test_a_slice_says_what_drop_it_can_rule_out() -> None:
    row = next(
        line for line in markdown(comparison(UNGUARDED)).splitlines() if "intent=billing" in line
    )

    assert "| 8.8 |" in row
    assert "not guarded" in row


def test_a_regressed_slice_does_not_claim_to_rule_anything_out() -> None:
    row = next(
        line for line in markdown(comparison(REGRESSED)).splitlines() if "intent=warranty" in line
    )

    assert "| — |" in row


def test_a_gap_is_named_in_the_verdict() -> None:
    empty = entry("warranty", Verdict.INCONCLUSIVE, None, size=0, gap=Gap.NO_PAIRED)

    assert "not guarded (no paired items)" in markdown(comparison(empty))


def test_a_partly_paired_slice_shows_how_many_were_compared() -> None:
    partial = entry("billing", Verdict.INCONCLUSIVE, (0.0, -0.09, 0.09), unpaired=("billing-03",))

    assert "| 39 of 40 |" in markdown(comparison(partial))


def test_watch_slices_get_their_own_table() -> None:
    watched = entry("billing", Verdict.REGRESSED, (-0.3, -0.4, -0.2), tier=Tier.WATCH)

    text = markdown(comparison(FINE, watched))

    assert "Watch slices (reported, never fail the build)" in text
    assert text.index("Gate slices") < text.index("Watch slices")


def test_the_policy_for_unguarded_slices_is_stated() -> None:
    assert "do not fail the build (policy: warn)" in markdown(comparison(UNGUARDED))
    assert "fail the build (policy: fail)" in markdown(comparison(UNGUARDED), "fail")


def test_the_lack_of_a_cross_slice_correction_is_admitted() -> None:
    assert "no correction across slices is applied yet" in markdown(comparison(FINE))


def test_errors_ignored_ids_and_changed_inputs_are_listed() -> None:
    text = markdown(
        comparison(
            FINE,
            base=side("main", unknown=("retired-1",)),
            head=side("pr-42", errors=("refunds-01", "refunds-02")),
            input_changed=("billing-05",),
        )
    )

    assert "`pr-42`: 2 items errored and count as failures: `refunds-01`, `refunds-02`." in text
    assert "`main`: 1 id not in the current dataset was ignored: `retired-1`." in text
    assert "1 item was left out because the input changed between runs: `billing-05`." in text


def test_long_id_lists_are_cut_short() -> None:
    many = tuple(f"refunds-{index:02d}" for index in range(12))

    text = markdown(comparison(FINE, head=side("pr-42", errors=many)))

    assert "and 7 more" in text


def test_a_run_on_an_older_dataset_is_flagged() -> None:
    older = side("main", dataset=DatasetRef(name="triage", version="1", sha256="b" * 64))

    assert "`main` ran on dataset v1, whose items differ from the current v2" in markdown(
        comparison(FINE, base=older)
    )


def test_the_terminal_version_has_no_markdown() -> None:
    text = render_text(
        comparison(FINE, UNGUARDED, REGRESSED), confidence=0.95, on_inconclusive="warn"
    )

    assert text.startswith("evalgate: regression found\n")
    assert "|" not in text
    assert "**" not in text
    assert "](" not in text
    assert "intent=warranty" in text
