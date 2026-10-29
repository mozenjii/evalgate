"""Running a dataset records every item, and treats the two kinds of failure differently."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from evalgate.compare import compare_slices
from evalgate.dataset import Item, content_hash, load_dataset
from evalgate.runner import run_dataset
from evalgate.runs import RunStore
from evalgate.scorers import ExactMatch, Scorer, ScorerError
from evalgate.stats import Verdict


def build_dataset(root: Path, count: int = 12) -> Path:
    """Write a small dataset: half billing, half refunds, balanced across lengths."""
    items: list[dict[str, Any]] = []
    for index in range(count):
        intent = "billing" if index % 2 == 0 else "refunds"
        items.append(
            {
                "id": f"item-{index:02d}",
                "input": intent,
                "expected": intent,
                "slices": {"intent": intent, "length": "short" if index < count // 2 else "long"},
                "provenance": "synthetic",
            }
        )
    (root / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )
    manifest = {
        "name": "toy",
        "version": "1",
        "items_file": "items.jsonl",
        "content_sha256": content_hash([Item.model_validate(item) for item in items]),
        "slice_dimensions": {"intent": ["billing", "refunds"], "length": ["short", "long"]},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def echo(value: JsonValue) -> JsonValue:
    """A perfect target: it returns the expected label."""
    return value


def always_billing(value: JsonValue) -> JsonValue:
    """A target that got worse on refunds and stayed right on billing."""
    del value
    return "billing"


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_every_item_is_recorded(tmp_path: Path) -> None:
    dataset = load_dataset(build_dataset(tmp_path))
    scorers: list[Scorer] = [ExactMatch()]

    with RunStore.open(":memory:") as store:
        record = run_dataset(dataset, echo, scorers, store, label="v1")

        assert len(store.outcomes(record.run_id)) == len(dataset)
        assert record.finished_at is not None
        assert record.dataset_sha256 == dataset.manifest.content_sha256


def test_latency_is_measured_for_every_item(tmp_path: Path) -> None:
    dataset = load_dataset(build_dataset(tmp_path))

    with RunStore.open(":memory:") as store:
        record = run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        latencies = [outcome.latency_ms for outcome in store.outcomes(record.run_id)]

    assert all(value is not None and value >= 0.0 for value in latencies)


def test_a_target_that_raises_is_recorded_and_the_run_continues(tmp_path: Path) -> None:
    # A model failing on one item out of hundreds is ordinary; discarding the rest is worse.
    dataset = load_dataset(build_dataset(tmp_path))
    seen: list[str] = []

    def flaky(value: JsonValue) -> JsonValue:
        seen.append(str(value))
        if len(seen) == 3:
            raise TimeoutError("upstream took too long")
        return value

    with RunStore.open(":memory:") as store:
        record = run_dataset(dataset, flaky, [ExactMatch()], store, label="v1")
        outcomes = store.outcomes(record.run_id)

    assert len(outcomes) == len(dataset)
    failed = [outcome for outcome in outcomes if outcome.error]
    assert len(failed) == 1
    assert "TimeoutError" in str(failed[0].error)
    assert failed[0].scores == {}


def test_a_scorer_that_cannot_run_stops_the_run(tmp_path: Path) -> None:
    # A scorer that cannot read one item cannot read any of them, so continuing would produce
    # a run whose scores mean nothing. The run is left unfinished, which is how it reads later.
    items = [
        {
            "id": "a",
            "input": "x",
            "expected": None,
            "slices": {"intent": "billing", "length": "short"},
            "provenance": "synthetic",
        }
    ]
    (tmp_path / "items.jsonl").write_text(json.dumps(items[0]) + "\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "name": "toy",
                "version": "1",
                "items_file": "items.jsonl",
                "content_sha256": content_hash([Item.model_validate(items[0])]),
                "slice_dimensions": {"intent": ["billing"], "length": ["short"]},
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(tmp_path / "manifest.json")

    with RunStore.open(":memory:") as store:
        with pytest.raises(ScorerError):
            run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        assert store.list_runs()[0].finished_at is None


def test_progress_is_reported_through_the_callback(tmp_path: Path) -> None:
    # Kept as a callback so this module never decides how output should look.
    dataset = load_dataset(build_dataset(tmp_path))
    seen: list[int] = []

    with RunStore.open(":memory:") as store:
        run_dataset(
            dataset,
            echo,
            [ExactMatch()],
            store,
            label="v1",
            on_item=lambda index, item: seen.append(index),
        )

    assert seen == list(range(len(dataset)))


# --------------------------------------------------------------------------- #
# Comparing
# --------------------------------------------------------------------------- #


def test_a_regression_is_found_in_the_slice_that_has_it(tmp_path: Path) -> None:
    dataset = load_dataset(build_dataset(tmp_path, count=40))

    with RunStore.open(":memory:") as store:
        baseline = run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        candidate = run_dataset(dataset, always_billing, [ExactMatch()], store, label="v2")

        comparisons = compare_slices(
            store, dataset, baseline.run_id, candidate.run_id, "exact_match", default_tolerance=0.05
        )

    by_slice = {(c.dimension, c.value): c for c in comparisons}
    assert by_slice[("intent", "refunds")].verdict is Verdict.REGRESSED
    assert by_slice[("intent", "refunds")].result.interval.point == pytest.approx(-1.0)
    # Billing was untouched, but twenty identical items cannot rule out a five-point drop.
    assert by_slice[("intent", "billing")].verdict is not Verdict.REGRESSED


def test_unchanged_runs_are_not_reported_as_regressed(tmp_path: Path) -> None:
    dataset = load_dataset(build_dataset(tmp_path, count=40))

    with RunStore.open(":memory:") as store:
        baseline = run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        candidate = run_dataset(dataset, echo, [ExactMatch()], store, label="v1-again")

        comparisons = compare_slices(
            store, dataset, baseline.run_id, candidate.run_id, "exact_match", default_tolerance=0.2
        )

    assert all(c.verdict is Verdict.NON_INFERIOR for c in comparisons)
    assert all(c.is_complete for c in comparisons)


def test_a_thin_slice_is_inconclusive_rather_than_a_pass(tmp_path: Path) -> None:
    # The failure this project exists to prevent: too few items to answer, reported as green.
    dataset = load_dataset(build_dataset(tmp_path, count=8))

    with RunStore.open(":memory:") as store:
        baseline = run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        candidate = run_dataset(dataset, echo, [ExactMatch()], store, label="v1-again")

        comparisons = compare_slices(
            store, dataset, baseline.run_id, candidate.run_id, "exact_match", default_tolerance=0.01
        )

    assert all(c.verdict is Verdict.INCONCLUSIVE for c in comparisons)


def test_per_slice_tolerances_are_respected(tmp_path: Path) -> None:
    dataset = load_dataset(build_dataset(tmp_path, count=40))

    with RunStore.open(":memory:") as store:
        baseline = run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        candidate = run_dataset(dataset, always_billing, [ExactMatch()], store, label="v2")

        comparisons = compare_slices(
            store,
            dataset,
            baseline.run_id,
            candidate.run_id,
            "exact_match",
            tolerances={("intent", "refunds"): 2.0},
            default_tolerance=0.05,
        )

    by_slice = {(c.dimension, c.value): c for c in comparisons}
    # A tolerance wide enough to swallow a total failure makes it non-inferior, which is the
    # point of the tolerance being a human's choice and recorded next to the verdict.
    assert by_slice[("intent", "refunds")].verdict is Verdict.NON_INFERIOR
    assert by_slice[("intent", "refunds")].tolerance == 2.0


def test_p_values_are_holm_adjusted_across_the_slices_compared(tmp_path: Path) -> None:
    dataset = load_dataset(build_dataset(tmp_path, count=40))

    with RunStore.open(":memory:") as store:
        baseline = run_dataset(dataset, echo, [ExactMatch()], store, label="v1")
        candidate = run_dataset(dataset, always_billing, [ExactMatch()], store, label="v2")

        comparisons = compare_slices(
            store, dataset, baseline.run_id, candidate.run_id, "exact_match"
        )

    for comparison in comparisons:
        assert comparison.result.p_value is not None
        assert comparison.adjusted_p_value is not None
        assert comparison.adjusted_p_value >= comparison.result.p_value
