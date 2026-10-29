"""Run records survive a round trip, and pairing never quietly drops an item."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalgate.runs import ItemOutcome, RunStore, StoreError
from evalgate.scorers import Score, ScoreKind


def store() -> RunStore:
    return RunStore.open(":memory:")


def start(run_store: RunStore, target: str, run_id: str | None = None) -> str:
    record = run_store.start_run(
        dataset_name="toy",
        dataset_version="1",
        dataset_sha256="a" * 64,
        target=target,
        run_id=run_id,
        started_at=datetime(2026, 10, 29, 12, 0, tzinfo=UTC),
    )
    return record.run_id


def binary(value: float) -> dict[str, Score]:
    return {"exact_match": Score(value=value, kind=ScoreKind.BINARY)}


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_a_run_and_its_outcomes_survive_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    with RunStore.open(path) as writing:
        run_id = start(writing, "prompt-v1")
        writing.record(
            run_id,
            ItemOutcome(
                item_id="a",
                output={"intent": "billing"},
                latency_ms=12.5,
                prompt_tokens=30,
                completion_tokens=4,
                cost_usd=0.0001,
                scores={
                    "exact_match": Score(1.0, ScoreKind.BINARY, {"output": "billing"}),
                    "set_f1": Score(0.75, ScoreKind.CONTINUOUS),
                },
            ),
        )
        writing.finish_run(run_id, datetime(2026, 10, 29, 12, 5, tzinfo=UTC))

    with RunStore.open(path) as reading:
        record = reading.get_run(run_id)
        assert record.target == "prompt-v1"
        assert record.finished_at == datetime(2026, 10, 29, 12, 5, tzinfo=UTC)

        outcome = reading.outcomes(run_id)[0]
        assert outcome.output == {"intent": "billing"}
        assert outcome.latency_ms == 12.5
        assert outcome.prompt_tokens == 30
        assert outcome.cost_usd == pytest.approx(0.0001)
        assert outcome.scores["exact_match"].value == 1.0
        assert outcome.scores["exact_match"].detail == {"output": "billing"}
        assert outcome.scores["set_f1"].kind is ScoreKind.CONTINUOUS


def test_an_unfinished_run_is_distinguishable_from_a_finished_one() -> None:
    # A run that died part way must not be compared against as if it were complete.
    with store() as run_store:
        run_id = start(run_store, "v1")
        assert run_store.get_run(run_id).finished_at is None
        run_store.finish_run(run_id)
        assert run_store.get_run(run_id).finished_at is not None


def test_an_error_outcome_records_why() -> None:
    with store() as run_store:
        run_id = start(run_store, "v1")
        run_store.record(run_id, ItemOutcome(item_id="a", error="timeout after 30s"))
        assert run_store.outcomes(run_id)[0].error == "timeout after 30s"


def test_runs_are_listed_most_recent_first() -> None:
    with store() as run_store:
        run_store.start_run(
            dataset_name="toy",
            dataset_version="1",
            dataset_sha256="a" * 64,
            target="old",
            started_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        run_store.start_run(
            dataset_name="toy",
            dataset_version="1",
            dataset_sha256="a" * 64,
            target="new",
            started_at=datetime(2026, 10, 2, tzinfo=UTC),
        )
        assert [record.target for record in run_store.list_runs()] == ["new", "old"]


def test_runs_can_be_listed_for_one_dataset() -> None:
    with store() as run_store:
        start(run_store, "v1")
        run_store.start_run(
            dataset_name="other",
            dataset_version="1",
            dataset_sha256="b" * 64,
            target="v1",
        )
        assert len(run_store.list_runs(dataset_name="toy")) == 1


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_a_duplicate_run_id_is_refused() -> None:
    with store() as run_store:
        start(run_store, "v1", run_id="fixed")
        with pytest.raises(StoreError, match="already exists"):
            start(run_store, "v2", run_id="fixed")


def test_recording_the_same_item_twice_is_refused() -> None:
    # A second recording would be a second answer for one item, and nothing says which counts.
    with store() as run_store:
        run_id = start(run_store, "v1")
        run_store.record(run_id, ItemOutcome(item_id="a", output="x"))
        with pytest.raises(StoreError, match="cannot record item"):
            run_store.record(run_id, ItemOutcome(item_id="a", output="y"))


def test_recording_against_an_unknown_run_is_refused() -> None:
    # Foreign keys are off by default in SQLite, so this is enabled on purpose.
    with store() as run_store, pytest.raises(StoreError, match="cannot record item"):
        run_store.record("nope", ItemOutcome(item_id="a", output="x"))


def test_finishing_an_unknown_run_is_refused() -> None:
    with store() as run_store, pytest.raises(StoreError, match="no such run"):
        run_store.finish_run("nope")


def test_fetching_an_unknown_run_is_refused() -> None:
    with store() as run_store, pytest.raises(StoreError, match="no such run"):
        run_store.get_run("nope")


def test_a_failed_record_leaves_nothing_behind() -> None:
    # The outcome and its scores go in under one transaction. A half-written item would later
    # read as a scorer that silently produced nothing.
    with store() as run_store:
        run_id = start(run_store, "v1")
        run_store.record(run_id, ItemOutcome(item_id="a", output="x", scores=binary(1.0)))
        with pytest.raises(StoreError):
            run_store.record(run_id, ItemOutcome(item_id="a", output="y", scores=binary(0.0)))
        assert len(run_store.outcomes(run_id)) == 1
        assert run_store.outcomes(run_id)[0].output == "x"


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #


def two_runs(
    run_store: RunStore,
    baseline: dict[str, float],
    candidate: dict[str, float],
    kind: ScoreKind = ScoreKind.BINARY,
) -> tuple[str, str]:
    left = start(run_store, "baseline", run_id="left")
    right = start(run_store, "candidate", run_id="right")
    for run_id, values in ((left, baseline), (right, candidate)):
        for item_id, value in values.items():
            run_store.record(
                run_id,
                ItemOutcome(
                    item_id=item_id,
                    output=value,
                    scores={"exact_match": Score(value=value, kind=kind)},
                ),
            )
    return left, right


def test_pairing_lines_up_items_present_in_both_runs() -> None:
    with store() as run_store:
        left, right = two_runs(run_store, {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})

        pairing = run_store.pair(left, right, "exact_match")

        assert pairing.item_ids == ("a", "b")
        assert pairing.baseline == (1.0, 0.0)
        assert pairing.candidate == (0.0, 1.0)
        assert pairing.kind is ScoreKind.BINARY
        assert pairing.is_complete
        assert len(pairing) == 2


def test_pairing_names_items_that_only_one_run_scored() -> None:
    # Dropping them silently would compare a narrower slice than the one being reported.
    with store() as run_store:
        left, right = two_runs(run_store, {"a": 1.0, "b": 1.0}, {"a": 1.0, "c": 0.0})

        pairing = run_store.pair(left, right, "exact_match")

        assert pairing.item_ids == ("a",)
        assert pairing.only_in_baseline == ("b",)
        assert pairing.only_in_candidate == ("c",)
        assert not pairing.is_complete


def test_pairing_can_be_restricted_to_one_slice() -> None:
    with store() as run_store:
        left, right = two_runs(
            run_store, {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 0.0, "b": 1.0, "c": 0.0}
        )

        pairing = run_store.pair(left, right, "exact_match", items=["a", "c"])

        assert pairing.item_ids == ("a", "c")
        assert pairing.is_complete


def test_pairing_an_unknown_scorer_is_refused() -> None:
    with store() as run_store:
        left, right = two_runs(run_store, {"a": 1.0}, {"a": 1.0})
        with pytest.raises(StoreError, match="no scores for scorer"):
            run_store.pair(left, right, "nope")


def test_pairing_refuses_a_scorer_that_changed_kind_between_runs() -> None:
    # Comparing a binary slice against a continuous one would silently pick the wrong test.
    with store() as run_store:
        left = start(run_store, "baseline", run_id="left")
        right = start(run_store, "candidate", run_id="right")
        run_store.record(left, ItemOutcome(item_id="a", output=1.0, scores=binary(1.0)))
        run_store.record(
            right,
            ItemOutcome(
                item_id="a",
                output=1.0,
                scores={"exact_match": Score(0.5, ScoreKind.CONTINUOUS)},
            ),
        )
        with pytest.raises(StoreError, match="not comparable"):
            run_store.pair(left, right, "exact_match")


def test_pairing_an_unknown_run_is_refused() -> None:
    with store() as run_store:
        left, _ = two_runs(run_store, {"a": 1.0}, {"a": 1.0})
        with pytest.raises(StoreError, match="no such run"):
            run_store.pair(left, "nope", "exact_match")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_a_store_written_by_a_newer_schema_is_refused(tmp_path: Path) -> None:
    # Reading it anyway would mean guessing at columns, and a guessed-at run record is worse
    # than none.
    path = tmp_path / "runs.sqlite"
    with RunStore.open(path) as run_store:
        start(run_store, "v1")
    # Written from outside, the way a newer build of this tool would have written it.
    foreign = sqlite3.connect(path, isolation_level=None)
    foreign.execute("UPDATE schema_version SET version = 99")
    foreign.close()

    with pytest.raises(StoreError, match="schema version 99"):
        RunStore.open(path)


def test_reopening_an_existing_store_keeps_its_runs(tmp_path: Path) -> None:
    path = tmp_path / "runs.sqlite"
    with RunStore.open(path) as run_store:
        start(run_store, "v1")
    with RunStore.open(path) as run_store:
        assert len(run_store.list_runs()) == 1
