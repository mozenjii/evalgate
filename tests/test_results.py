"""Results files round-trip, and refuse anything that would silently change what a run means."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalgate.results import (
    DatasetRef,
    ItemResult,
    ResultsError,
    ResultsHeader,
    input_digest,
    read_results,
    write_results,
)

HEADER = ResultsHeader(
    label="v1",
    dataset=DatasetRef(name="triage", version="1", sha256="a" * 64),
    created_at=datetime(2026, 11, 6, 12, 0, tzinfo=UTC),
)


def write_lines(path: Path, *lines: object) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def test_a_written_file_reads_back_the_same(tmp_path: Path) -> None:
    items = [
        ItemResult(id="a", output="billing", latency_ms=3.5, input_sha256="b" * 64),
        ItemResult(id="b", output=None, error="TimeoutError: slow"),
        ItemResult(id="c", output={"intent": "refunds"}, meta={"trace": "t-1"}),
    ]
    path = tmp_path / "out" / "v1.jsonl"

    write_results(path, HEADER, items)
    results = read_results(path)

    assert results.header == HEADER
    assert results.items == tuple(items)
    assert results.label == "v1"


def test_a_file_without_a_header_is_complete(tmp_path: Path) -> None:
    # The whole point of the format: a team's own pipeline can write it in any language.
    path = write_lines(
        tmp_path / "head.jsonl", {"id": "a", "output": "x"}, {"id": "b", "output": "y"}
    )

    results = read_results(path)

    assert results.header is None
    assert [item.id for item in results.items] == ["a", "b"]
    assert results.label == "head"


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"id": "a", "output": 1}\n\n{"id": "b", "output": 2}\n', encoding="utf-8")

    assert len(read_results(path).items) == 2


def test_a_mistyped_field_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    # Ignored, "ouput" would read as an item that produced nothing: a failure nobody caused.
    path = write_lines(tmp_path / "r.jsonl", {"id": "a", "ouput": "billing"})

    with pytest.raises(ResultsError, match=r"r\.jsonl:1: invalid item"):
        read_results(path)


def test_a_duplicate_id_is_refused_with_both_lines(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "r.jsonl", {"id": "a", "output": 1}, {"id": "a", "output": 2})

    with pytest.raises(ResultsError, match="duplicate id 'a', first seen on line 1"):
        read_results(path)


def test_a_header_anywhere_but_first_is_refused(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path / "r.jsonl", {"id": "a", "output": 1}, {"format": "evalgate.results", "label": "x"}
    )

    with pytest.raises(ResultsError, match="must be the first line"):
        read_results(path)


def test_a_newer_format_version_is_refused(tmp_path: Path) -> None:
    path = write_lines(
        tmp_path / "r.jsonl",
        {"format": "evalgate.results", "version": 99, "label": "x"},
        {"id": "a", "output": 1},
    )

    with pytest.raises(ResultsError, match="format 99"):
        read_results(path)


def test_an_unknown_format_is_refused(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "r.jsonl", {"format": "something-else"}, {"id": "a"})

    with pytest.raises(ResultsError, match="unknown format"):
        read_results(path)


def test_a_file_with_no_items_is_refused(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "r.jsonl", {"format": "evalgate.results", "label": "x"})

    with pytest.raises(ResultsError, match="no items"):
        read_results(path)


def test_invalid_json_names_its_line(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('{"id": "a", "output": 1}\n{ nope\n', encoding="utf-8")

    with pytest.raises(ResultsError, match=r"r\.jsonl:2: not valid JSON"):
        read_results(path)


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ResultsError, match="cannot read results"):
        read_results(tmp_path / "nope.jsonl")


@pytest.mark.parametrize("field", ["latency_ms", "prompt_tokens", "completion_tokens", "cost_usd"])
def test_negative_measurements_are_refused(tmp_path: Path, field: str) -> None:
    path = write_lines(tmp_path / "r.jsonl", {"id": "a", "output": 1, field: -1})

    with pytest.raises(ResultsError, match="invalid item"):
        read_results(path)


def test_writing_replaces_the_file_whole_and_leaves_nothing_behind(tmp_path: Path) -> None:
    # A run interrupted half way must not leave a truncated file that reads as a smaller run.
    path = tmp_path / "v1.jsonl"
    write_results(path, HEADER, [ItemResult(id="a", output=1), ItemResult(id="b", output=2)])
    write_results(path, HEADER, [ItemResult(id="c", output=3)])

    assert [item.id for item in read_results(path).items] == ["c"]
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["v1.jsonl"]


def test_input_digest_ignores_key_order_but_not_content() -> None:
    assert input_digest({"a": 1, "b": 2}) == input_digest({"b": 2, "a": 1})
    assert input_digest({"a": 1}) != input_digest({"a": 2})
    assert input_digest("text") != input_digest("text ")


def test_by_id_indexes_every_item(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "r.jsonl", {"id": "a", "output": 1}, {"id": "b", "output": 2})

    assert read_results(path).by_id()["b"].output == 2
