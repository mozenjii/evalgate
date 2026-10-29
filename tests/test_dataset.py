"""Loading a dataset refuses anything it cannot vouch for."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalgate.dataset import (
    Dataset,
    DatasetError,
    Item,
    Provenance,
    content_hash,
    load_dataset,
)

DIMENSIONS: dict[str, list[str]] = {
    "intent": ["billing", "refunds", "warranty"],
    "length": ["short", "long"],
}


def make_item(
    item_id: str,
    *,
    intent: str = "billing",
    length: str = "short",
    provenance: str = "hand_written",
    **overrides: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "input": f"text for {item_id}",
        "expected": intent,
        "slices": {"intent": intent, "length": length},
        "provenance": provenance,
    }
    item.update(overrides)
    return item


def write_dataset(
    root: Path,
    items: list[dict[str, Any]],
    *,
    dimensions: Mapping[str, list[str]] | None = None,
    content_sha256: str | None = None,
) -> Path:
    """Write a manifest and its JSONL, returning the manifest path."""
    items_path = root / "items.jsonl"
    items_path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")

    if content_sha256 is None:
        content_sha256 = content_hash([Item.model_validate(item) for item in items])

    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "toy",
                "version": "1",
                "items_file": "items.jsonl",
                "content_sha256": content_sha256,
                # `or` would turn an intentionally empty mapping back into the default,
                # which is exactly what the empty-dimensions test is trying to produce.
                "slice_dimensions": dict(DIMENSIONS if dimensions is None else dimensions),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #


def test_loads_items_in_file_order(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [make_item("c"), make_item("a"), make_item("b")])

    dataset = load_dataset(manifest)

    assert [item.id for item in dataset] == ["c", "a", "b"]
    assert len(dataset) == 3


def test_blank_lines_are_formatting_not_data(tmp_path: Path) -> None:
    items_path = tmp_path / "items.jsonl"
    write_dataset(tmp_path, [make_item("a"), make_item("b")])
    body = items_path.read_text(encoding="utf-8")
    items_path.write_text(body.replace("\n", "\n\n", 1), encoding="utf-8")

    assert len(load_dataset(tmp_path / "manifest.json")) == 2


# --------------------------------------------------------------------------- #
# Provenance and slices are required
# --------------------------------------------------------------------------- #


def test_an_item_without_provenance_is_refused(tmp_path: Path) -> None:
    item = make_item("a")
    del item["provenance"]
    manifest = write_dataset(tmp_path, [item], content_sha256="0" * 64)

    with pytest.raises(DatasetError, match="provenance"):
        load_dataset(manifest)


def test_an_unknown_provenance_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(
        tmp_path, [make_item("a", provenance="vibes")], content_sha256="0" * 64
    )

    with pytest.raises(DatasetError, match="provenance"):
        load_dataset(manifest)


def test_an_item_with_no_slices_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [make_item("a", slices={})], content_sha256="0" * 64)

    with pytest.raises(DatasetError, match="slice"):
        load_dataset(manifest)


def test_an_item_missing_a_declared_dimension_is_refused(tmp_path: Path) -> None:
    # Present in the dataset but absent from the length comparison: the slice would look
    # thinner than the dataset with nothing to explain the difference.
    manifest = write_dataset(
        tmp_path, [make_item("a", slices={"intent": "billing"})], content_sha256="0" * 64
    )

    with pytest.raises(DatasetError, match="missing slice dimension"):
        load_dataset(manifest)


def test_an_undeclared_slice_value_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(
        tmp_path,
        [make_item("a", slices={"intent": "chargebacks", "length": "short"})],
        content_sha256="0" * 64,
    )

    with pytest.raises(DatasetError, match="does not allow"):
        load_dataset(manifest)


def test_an_undeclared_dimension_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(
        tmp_path,
        [make_item("a", slices={"intent": "billing", "length": "short", "tone": "angry"})],
        content_sha256="0" * 64,
    )

    with pytest.raises(DatasetError, match="does not know"):
        load_dataset(manifest)


def test_an_unknown_field_is_refused_rather_than_dropped(tmp_path: Path) -> None:
    # A typo in a field name would otherwise be silently ignored, and the item would be
    # scored against an expectation nobody wrote.
    manifest = write_dataset(tmp_path, [make_item("a", expcted="billing")], content_sha256="0" * 64)

    with pytest.raises(DatasetError, match="invalid item"):
        load_dataset(manifest)


# --------------------------------------------------------------------------- #
# Identity and integrity
# --------------------------------------------------------------------------- #


def test_a_duplicate_id_is_refused(tmp_path: Path) -> None:
    # Runs are paired by id, so a repeat would pair the wrong items against each other.
    manifest = write_dataset(tmp_path, [make_item("a"), make_item("a")], content_sha256="0" * 64)

    with pytest.raises(DatasetError, match="duplicate item id"):
        load_dataset(manifest)


def test_a_stale_content_hash_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [make_item("a")])
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(make_item("a", expected="refunds")) + "\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="content hash does not match"):
        load_dataset(manifest)


def test_the_hash_check_can_be_skipped_for_tools_that_rewrite_it(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [make_item("a")], content_sha256="0" * 64)

    assert len(load_dataset(manifest, verify_hash=False)) == 1


def test_the_hash_ignores_formatting_but_not_content(tmp_path: Path) -> None:
    # Reordering keys and adding whitespace is a formatting change; a checkout on another
    # platform must not look like tampering.
    first = Item.model_validate(make_item("a"))
    same = Item.model_validate(
        {
            "provenance": "hand_written",
            "slices": {"length": "short", "intent": "billing"},
            "expected": "billing",
            "input": "text for a",
            "id": "a",
        }
    )
    different = Item.model_validate(make_item("a", expected="refunds"))

    assert content_hash([first]) == content_hash([same])
    assert content_hash([first]) != content_hash([different])


def test_the_hash_covers_item_order(tmp_path: Path) -> None:
    one = Item.model_validate(make_item("a"))
    two = Item.model_validate(make_item("b"))
    assert content_hash([one, two]) != content_hash([two, one])


def test_an_empty_dataset_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [], content_sha256="0" * 64)

    with pytest.raises(DatasetError, match="nothing to compare"):
        load_dataset(manifest)


def test_a_malformed_line_names_its_line_number(tmp_path: Path) -> None:
    write_dataset(tmp_path, [make_item("a"), make_item("b")], content_sha256="0" * 64)
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(make_item("a")) + "\n{ not json\n", encoding="utf-8")

    with pytest.raises(DatasetError, match=r"items\.jsonl:2"):
        load_dataset(tmp_path / "manifest.json")


def test_a_missing_items_file_is_reported_against_the_manifest(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [make_item("a")])
    (tmp_path / "items.jsonl").unlink()

    with pytest.raises(DatasetError, match="cannot read items"):
        load_dataset(manifest)


def test_a_manifest_without_dimensions_is_refused(tmp_path: Path) -> None:
    manifest = write_dataset(tmp_path, [make_item("a")], dimensions={}, content_sha256="0" * 64)

    with pytest.raises(DatasetError, match="no slice dimensions"):
        load_dataset(manifest)


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #


def toy(tmp_path: Path) -> Dataset:
    return load_dataset(
        write_dataset(
            tmp_path,
            [
                make_item("a", intent="billing", length="short"),
                make_item("b", intent="billing", length="long"),
                make_item("c", intent="refunds", length="short"),
            ],
        )
    )


def test_slice_returns_matching_items_in_order(tmp_path: Path) -> None:
    assert [item.id for item in toy(tmp_path).slice("intent", "billing")] == ["a", "b"]


def test_a_declared_but_empty_slice_is_a_real_answer(tmp_path: Path) -> None:
    # An empty declared slice is a coverage gap the report must be able to show.
    assert toy(tmp_path).slice("intent", "warranty") == ()


def test_asking_for_an_undeclared_slice_is_an_error_not_an_empty_result(tmp_path: Path) -> None:
    dataset = toy(tmp_path)

    with pytest.raises(DatasetError, match="not a declared value"):
        dataset.slice("intent", "chargebacks")
    with pytest.raises(DatasetError, match="no slice dimension"):
        dataset.slice("tone", "angry")


def test_slice_sizes_counts_every_declared_slice_including_empty_ones(tmp_path: Path) -> None:
    sizes = toy(tmp_path).slice_sizes()

    assert sizes[("intent", "billing")] == 2
    assert sizes[("intent", "warranty")] == 0
    assert sizes[("length", "short")] == 2
    assert len(sizes) == 5


# --------------------------------------------------------------------------- #
# Item semantics
# --------------------------------------------------------------------------- #


def test_items_are_immutable(tmp_path: Path) -> None:
    # A loaded item is evidence about a run. Letting a caller edit one in place would mean a
    # report could disagree with the dataset it names.
    item = toy(tmp_path).items[0]

    with pytest.raises(ValidationError):
        setattr(item, "id", "changed")  # noqa: B010 - the assignment is what must fail


def test_expected_may_be_absent_for_judge_only_items() -> None:
    item = Item.model_validate(make_item("a", expected=None))
    assert item.expected is None


def test_expected_may_be_structured() -> None:
    item = Item.model_validate(make_item("a", expected={"intent": "billing", "urgent": True}))
    assert item.expected == {"intent": "billing", "urgent": True}


def test_every_provenance_value_is_usable() -> None:
    for provenance in Provenance:
        item = Item.model_validate(make_item("a", provenance=provenance.value))
        assert item.provenance is provenance
