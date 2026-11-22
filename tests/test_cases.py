"""Offline log selection and reviewed dataset versioning."""

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from evalgate.cases import CaseError, mine, mine_command, promote, promote_command
from evalgate.dataset import Item, Manifest, Provenance, content_hash, load_dataset


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    item = Item(
        id="original",
        input="An original request",
        expected="approved",
        slices={"category": "support"},
        provenance=Provenance.HAND_WRITTEN,
    )
    manifest = Manifest(
        name="demo",
        version="1",
        items_file="items.jsonl",
        content_sha256=content_hash([item]),
        slice_dimensions={"category": ("support", "billing")},
    )
    (tmp_path / "items.jsonl").write_text(item.model_dump_json() + "\n", encoding="utf-8")
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json() + "\n", encoding="utf-8")
    return path


def _row(**overrides: object) -> dict[str, object]:
    return {
        "classification": "redacted",
        "input": "A relevant request",
        "slices": {"category": "support"},
        "origin_run": "run-42",
        "suspected_stage": "ranking",
        "failure": True,
        **overrides,
    }


def _write(path: Path, *rows: dict[str, object]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines()]


def test_mine_prioritizes_disagreement_and_removes_exact_duplicates(tmp_path: Path) -> None:
    source, output = tmp_path / "logs.jsonl", tmp_path / "review.jsonl"
    _write(
        source,
        _row(input={"a": "hello", "b": 3}, judge_votes=[True, True]),
        _row(input="another case", judge_votes=[False, False]),
        _row(input={"b": 3, "a": "hello"}, judge_votes=[True, False], origin_run="run-99"),
    )
    assert mine(source, output) == 2
    rows = _read(output)
    assert [row["origin_run"] for row in rows] == ["run-99", "run-42"]
    assert all(row.get("reviewed", False) is False for row in rows)


def test_mine_near_duplicate_threshold_preserves_distinct_slices(tmp_path: Path) -> None:
    source, output = tmp_path / "logs.jsonl", tmp_path / "review.jsonl"
    _write(
        source,
        _row(input="please reset my forgotten password now", origin_run="run-1"),
        _row(input="please reset my forgotten password today", origin_run="run-2"),
        _row(
            input="please reset my forgotten password today",
            origin_run="run-3",
            slices={"category": "billing"},
        ),
    )
    assert mine(source, output, near_duplicate_threshold=0.6) == 2
    assert [row["origin_run"] for row in _read(output)] == ["run-1", "run-3"]


def test_mine_preserves_explicit_null_input(tmp_path: Path) -> None:
    source, output = tmp_path / "logs.jsonl", tmp_path / "review.jsonl"
    _write(source, _row(input=None))
    assert mine(source, output) == 1
    assert _read(output)[0]["input"] is None


@pytest.mark.parametrize(
    "line",
    [
        '{"classification":"unredacted","classification":"redacted"}',
        '{"classification":"unredacted","input":"PRIVATE_MARKER"}',
        '{"classification":"redacted","input":"PRIVATE_MARKER","raw_output":"sensitive"}',
        '{"classification":"redacted","input":NaN}',
    ],
)
def test_mine_fails_closed_without_printing_input(tmp_path: Path, line: str) -> None:
    source, output = tmp_path / "logs.jsonl", tmp_path / "review.jsonl"
    source.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(CaseError) as failure:
        mine(source, output)
    assert "PRIVATE_MARKER" not in str(failure.value)
    assert not output.exists()


def test_mine_never_replaces_existing_file(tmp_path: Path) -> None:
    source, output = tmp_path / "logs.jsonl", tmp_path / "review.jsonl"
    _write(source, _row())
    output.write_text("keep me", encoding="utf-8")
    with pytest.raises(CaseError, match="already exists"):
        mine(source, output)
    assert output.read_text(encoding="utf-8") == "keep me"


def test_promote_writes_new_loadable_version_with_stable_provenance(
    tmp_path: Path, manifest_path: Path
) -> None:
    reviews = tmp_path / "review.jsonl"
    _write(
        reviews,
        _row(reviewed=True, expected="approved"),
        _row(
            reviewed=True,
            expected=False,
            input="Synthetic billing example",
            classification="synthetic",
            slices={"category": "billing"},
        ),
    )
    before = load_dataset(manifest_path)
    first_output = tmp_path / "v2"
    new = load_dataset(promote(reviews, manifest_path, "2", first_output))
    assert new.manifest.version == "2"
    assert new.items[:1] == before.items
    first, second = new.items[1:]
    assert first.provenance is Provenance.PROMOTED_FAILURE
    assert first.source and "run-42" in first.source and "ranking" in first.source
    assert second.provenance is Provenance.SYNTHETIC and second.expected is False
    rerun = load_dataset(promote(reviews, manifest_path, "2", tmp_path / "v2-reproduced"))
    assert rerun == new
    assert load_dataset(manifest_path) == before
    with pytest.raises(CaseError, match="already exists"):
        promote(reviews, manifest_path, "2", first_output)
    assert load_dataset(first_output / "manifest.json") == new


@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"reviewed": False}, "reviewed"),
        ({"failure": False}, "failure"),
        ({"expected": None}, "expectation"),
        ({"suspected_stage": None}, "suspected_stage"),
        ({"slices": {"category": "unknown"}}, "slice"),
        ({"input": "An original request"}, "duplicate"),
    ],
)
def test_promote_rejects_invalid_rows_without_creating_partial_version(
    tmp_path: Path, manifest_path: Path, overrides: dict[str, object], expected_message: str
) -> None:
    reviews, output = tmp_path / "review.jsonl", tmp_path / "v2"
    row = _row(reviewed=True, expected="fixed")
    row.update(overrides)
    _write(reviews, _row(input="another valid candidate", expected="fixed", reviewed=True), row)
    with pytest.raises(CaseError, match=expected_message):
        promote(reviews, manifest_path, "2", output)
    assert not output.exists()


def test_promote_rolls_back_its_items_when_manifest_publication_fails(
    tmp_path: Path, manifest_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews, output = tmp_path / "review.jsonl", tmp_path / "v2"
    _write(reviews, _row(reviewed=True, expected="fixed"))
    hardlink = Path.hardlink_to

    def fail_on_manifest(path: Path, source: Path) -> None:
        if path.name == "manifest.json":
            raise OSError("staging failure")
        hardlink(path, source)

    monkeypatch.setattr(Path, "hardlink_to", fail_on_manifest)
    with pytest.raises(CaseError, match="cannot publish"):
        promote(reviews, manifest_path, "2", output)
    assert not output.exists()
    assert load_dataset(manifest_path).manifest.version == "1"


def test_cli_connects_mine_and_promote(tmp_path: Path, manifest_path: Path) -> None:
    cli = typer.Typer()
    cli.command("mine")(mine_command)
    cli.command("promote")(promote_command)
    runner = CliRunner()
    source, review, destination = (tmp_path / name for name in ("logs.jsonl", "review.jsonl", "v2"))
    _write(source, _row(classification="synthetic"))
    result = runner.invoke(cli, ["mine", str(source), "--out", str(review)])
    assert result.exit_code == 0, result.output
    row = _read(review)[0]
    row.update(reviewed=True, expected="fixed")
    _write(review, row)
    result = runner.invoke(
        cli,
        [
            "promote",
            str(review),
            "--manifest",
            str(manifest_path),
            "--version",
            "2",
            "--out-dir",
            str(destination),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(load_dataset(destination / "manifest.json")) == 2
