"""The gate answers for every slice it was asked about, and never hides a gap."""

import json
from pathlib import Path
from typing import Any

import pytest

from evalgate.config import Config, Tier, load_config
from evalgate.dataset import Item, content_hash
from evalgate.gate import Comparison, Gap, GateError, Outcome, compare, score_results
from evalgate.results import DatasetRef, ItemResult, Results, ResultsHeader, input_digest
from evalgate.stats import Verdict

INTENTS = {"billing": 40, "refunds": 40, "warranty": 0}

CONFIG = """
[dataset]
manifest = "manifest.json"

[[scorer]]
type = "exact_match"

[[scorer]]
type = "edit_similarity"

[[gate]]
scorer = "exact_match"
slices = {gate}
tolerance = {tolerance}

[[watch]]
scorer = "exact_match"
slices = {watch}
tolerance = 0.05
"""


def items() -> list[dict[str, Any]]:
    return [
        {
            "id": f"{intent}-{index:02d}",
            "input": f"{intent} message {index}",
            "expected": intent,
            "slices": {"intent": intent, "length": "short" if index % 2 else "long"},
            "provenance": "synthetic",
        }
        for intent, count in INTENTS.items()
        for index in range(count)
    ]


def make_config(
    root: Path,
    tolerance: float = 0.05,
    extra: str = "",
    *,
    gate: tuple[str, ...] = ("intent=*",),
    watch: tuple[str, ...] = ("length=*",),
) -> Config:
    root.mkdir(parents=True, exist_ok=True)
    data = items()
    (root / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in data), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "name": "toy",
                "version": "1",
                "items_file": "items.jsonl",
                "content_sha256": content_hash([Item.model_validate(item) for item in data]),
                "slice_dimensions": {"intent": list(INTENTS), "length": ["short", "long"]},
            }
        ),
        encoding="utf-8",
    )
    body = CONFIG.format(
        tolerance=tolerance, gate=json.dumps(list(gate)), watch=json.dumps(list(watch))
    )
    path = root / "evalgate.toml"
    path.write_text(body + extra, encoding="utf-8")
    return load_config(path)


def run(
    label: str,
    answer: dict[str, str] | None = None,
    *,
    skip: frozenset[str] = frozenset(),
    errors: frozenset[str] = frozenset(),
    stale: frozenset[str] = frozenset(),
    extra: tuple[ItemResult, ...] = (),
    dataset_name: str = "toy",
) -> Results:
    """A run that answers every item correctly unless told otherwise."""
    answer = answer or {}
    produced: list[ItemResult] = []
    for item in items():
        if item["id"] in skip:
            continue
        if item["id"] in errors:
            produced.append(ItemResult(id=item["id"], error="TimeoutError"))
            continue
        asked = "an older input" if item["id"] in stale else item["input"]
        produced.append(
            ItemResult(
                id=item["id"],
                output=answer.get(item["id"], item["expected"]),
                input_sha256=input_digest(asked),
            )
        )
    header = ResultsHeader(
        label=label, dataset=DatasetRef(name=dataset_name, version="1", sha256="a" * 64)
    )
    return Results(header=header, items=(*produced, *extra))


def refunds_broken() -> dict[str, str]:
    return {f"refunds-{index:02d}": "billing" for index in range(40)}


def gate_slices(comparison: Comparison) -> dict[str, Any]:
    return {str(entry.slice): entry for entry in comparison.gate}


# --------------------------------------------------------------------------- verdicts


def test_a_regression_is_found_in_the_slice_that_has_it(tmp_path: Path) -> None:
    comparison = compare(
        make_config(tmp_path, tolerance=0.1), run("base"), run("head", refunds_broken())
    )

    gate = gate_slices(comparison)
    assert gate["intent=refunds"].verdict is Verdict.REGRESSED
    assert gate["intent=refunds"].result.interval.point == pytest.approx(-1.0)
    assert gate["intent=billing"].verdict is Verdict.NON_INFERIOR
    assert comparison.outcome is Outcome.REGRESSED


def test_a_slice_too_thin_for_its_tolerance_is_not_a_pass(tmp_path: Path) -> None:
    # Forty identical answers cannot rule out a two-point drop; saying so is the whole point.
    comparison = compare(make_config(tmp_path, tolerance=0.02), run("base"), run("head"))

    billing = gate_slices(comparison)["intent=billing"]
    assert billing.verdict is Verdict.INCONCLUSIVE
    assert billing.gap is None
    assert comparison.outcome is Outcome.UNGUARDED


def test_every_outcome_is_reachable(tmp_path: Path) -> None:
    def outcome(name: str, tolerance: float, head: Results) -> Outcome:
        config = make_config(tmp_path / name, tolerance, gate=("intent=billing", "intent=refunds"))
        return compare(config, run("base"), head).outcome

    assert outcome("ok", 0.2, run("head")) is Outcome.OK
    assert outcome("thin", 0.02, run("head")) is Outcome.UNGUARDED
    assert outcome("broken", 0.2, run("head", refunds_broken())) is Outcome.REGRESSED


def test_an_empty_gate_slice_keeps_the_check_from_being_ok(tmp_path: Path) -> None:
    # Warranty is declared but holds no items. A check that covers it cannot claim to be fully
    # guarded, however well the other slices did.
    comparison = compare(make_config(tmp_path, tolerance=0.2), run("base"), run("head"))

    assert gate_slices(comparison)["intent=billing"].verdict is Verdict.NON_INFERIOR
    assert comparison.outcome is Outcome.UNGUARDED


def test_a_regressed_watch_slice_does_not_decide_the_outcome(tmp_path: Path) -> None:
    config = make_config(
        tmp_path, tolerance=0.2, gate=("intent=billing",), watch=("intent=refunds",)
    )

    comparison = compare(config, run("base"), run("head", refunds_broken()))

    (refunds,) = comparison.watch
    assert refunds.tier is Tier.WATCH
    assert refunds.verdict is Verdict.REGRESSED
    assert comparison.outcome is Outcome.OK


def test_a_tolerance_override_applies_to_its_slice(tmp_path: Path) -> None:
    config = make_config(
        tmp_path, tolerance=0.2, extra='\n[gate.tolerances]\n"intent=billing" = 0.01\n'
    )

    gate = gate_slices(compare(config, run("base"), run("head")))
    assert gate["intent=billing"].tolerance == 0.01
    assert gate["intent=billing"].verdict is Verdict.INCONCLUSIVE
    assert gate["intent=refunds"].tolerance == 0.2


# --------------------------------------------------------------------------- gaps


def test_a_slice_with_no_items_is_reported_as_a_gap_not_dropped(tmp_path: Path) -> None:
    warranty = gate_slices(compare(make_config(tmp_path), run("base"), run("head")))[
        "intent=warranty"
    ]

    assert warranty.size == 0
    assert warranty.gap is Gap.NO_PAIRED
    assert warranty.verdict is Verdict.INCONCLUSIVE
    assert warranty.result is None
    assert warranty.rules_out is None


def test_a_slice_with_one_paired_item_is_a_gap(tmp_path: Path) -> None:
    missing = frozenset(f"refunds-{index:02d}" for index in range(1, 40))

    refunds = gate_slices(compare(make_config(tmp_path), run("base"), run("head", skip=missing)))[
        "intent=refunds"
    ]

    assert refunds.gap is Gap.ONE_PAIRED
    assert refunds.n == 1
    assert refunds.size == 40
    assert len(refunds.unpaired) == 39


def test_items_missing_from_one_run_are_named(tmp_path: Path) -> None:
    comparison = compare(
        make_config(tmp_path), run("base", skip=frozenset({"refunds-03"})), run("head")
    )

    refunds = gate_slices(comparison)["intent=refunds"]
    assert refunds.unpaired == ("refunds-03",)
    assert refunds.n == 39


# --------------------------------------------------------------------------- what is compared


def test_an_item_whose_input_changed_is_left_out_and_listed(tmp_path: Path) -> None:
    # The base side answered an older version of the question; pairing it would compare two
    # different questions.
    comparison = compare(
        make_config(tmp_path), run("base", stale=frozenset({"billing-05"})), run("head")
    )

    assert comparison.input_changed == ("billing-05",)
    assert "billing-05" in gate_slices(comparison)["intent=billing"].unpaired


def test_an_error_counts_as_a_failure_not_a_missing_item(tmp_path: Path) -> None:
    # Leaving crashes out would make a change that crashes on a slice look like a smaller slice.
    crashed = frozenset(f"refunds-{index:02d}" for index in range(40))

    comparison = compare(
        make_config(tmp_path, tolerance=0.1), run("base"), run("head", errors=crashed)
    )

    refunds = gate_slices(comparison)["intent=refunds"]
    assert refunds.verdict is Verdict.REGRESSED
    assert refunds.n == 40
    assert len(comparison.head.errors) == 40


def test_ids_the_dataset_no_longer_holds_are_ignored_and_listed(tmp_path: Path) -> None:
    gone = ItemResult(id="retired-item", output="billing")

    comparison = compare(make_config(tmp_path), run("base", extra=(gone,)), run("head"))

    assert comparison.base.unknown == ("retired-item",)


def test_runs_of_a_different_dataset_are_refused(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="produced from dataset 'other'"):
        compare(make_config(tmp_path), run("base", dataset_name="other"), run("head"))


def test_a_continuous_scorer_gets_the_bootstrap(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        extra=(
            '\n[[watch]]\nscorer = "edit_similarity"\nslices = ["intent=billing"]\n'
            "tolerance = 0.05\n"
        ),
    )

    comparison = compare(config, run("base"), run("head", refunds_broken()))

    (similarity,) = [entry for entry in comparison.watch if entry.scorer == "edit_similarity"]
    assert similarity.result is not None
    assert similarity.result.method == "bootstrap_bca"


def test_rules_out_reads_the_lower_bound(tmp_path: Path) -> None:
    billing = gate_slices(compare(make_config(tmp_path), run("base"), run("head")))[
        "intent=billing"
    ]

    assert billing.result is not None
    assert billing.rules_out == pytest.approx(-billing.result.interval.low)
    assert billing.rules_out > 0


def test_scoring_skips_ids_outside_the_dataset(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    results = run("base", extra=(ItemResult(id="retired-item", output="x"),))

    scored = score_results(config.dataset, results, list(config.scorers.values()))

    assert "retired-item" not in scored["exact_match"]
    assert len(scored["exact_match"]) == 80
