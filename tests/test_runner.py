"""The runner records an answer for every item, and a crash on one item does not end the run."""

import uuid
from pathlib import Path

import pytest
from pydantic import JsonValue

from evalgate.dataset import Dataset, Item, Manifest, Provenance
from evalgate.results import input_digest
from evalgate.runner import TargetError, load_target, run_dataset


def dataset(count: int = 5) -> Dataset:
    items = tuple(
        Item(
            id=f"item-{index}",
            input=f"message {index}",
            expected="billing",
            slices={"intent": "billing"},
            provenance=Provenance.SYNTHETIC,
        )
        for index in range(count)
    )
    manifest = Manifest(
        name="toy",
        version="1",
        items_file="items.jsonl",
        content_sha256="a" * 64,
        slice_dimensions={"intent": ("billing",)},
    )
    return Dataset(manifest=manifest, items=items)


def write_module(root: Path, body: str) -> str:
    """Write a uniquely named module, so tests never collide in the import cache."""
    name = f"evalgate_test_target_{uuid.uuid4().hex}"
    (root / f"{name}.py").write_text(body, encoding="utf-8")
    return name


# --------------------------------------------------------------------------- loading targets


def test_a_target_is_loaded_from_the_given_directory(tmp_path: Path) -> None:
    name = write_module(tmp_path, "def classify(text):\n    return 'billing'\n")

    target = load_target(f"{name}:classify", [tmp_path])

    assert target("anything") == "billing"


def test_a_nested_attribute_can_be_named(tmp_path: Path) -> None:
    name = write_module(
        tmp_path,
        "class Model:\n    @staticmethod\n    def predict(text):\n        return text.upper()\n",
    )

    assert load_target(f"{name}:Model.predict", [tmp_path])("hi") == "HI"


@pytest.mark.parametrize("reference", ["no_colon", ":classify", "module:"])
def test_a_malformed_reference_is_refused(reference: str) -> None:
    with pytest.raises(TargetError, match="must look like"):
        load_target(reference)


def test_a_module_that_cannot_be_imported_is_refused() -> None:
    with pytest.raises(TargetError, match="cannot import"):
        load_target(f"evalgate_missing_{uuid.uuid4().hex}:classify")


def test_a_missing_attribute_is_refused(tmp_path: Path) -> None:
    name = write_module(tmp_path, "def classify(text):\n    return text\n")

    with pytest.raises(TargetError, match="has no attribute 'predict'"):
        load_target(f"{name}:predict", [tmp_path])


def test_something_that_is_not_callable_is_refused(tmp_path: Path) -> None:
    name = write_module(tmp_path, "classify = 'not a function'\n")

    with pytest.raises(TargetError, match="is not callable"):
        load_target(f"{name}:classify", [tmp_path])


# --------------------------------------------------------------------------- running


def test_every_item_gets_a_result_in_dataset_order() -> None:
    results = run_dataset(dataset(), lambda text: f"seen {text}")

    assert [result.id for result in results] == [f"item-{index}" for index in range(5)]
    assert results[2].output == "seen message 2"


def test_each_result_records_the_input_it_answered() -> None:
    # This is what lets a later comparison notice that an item's input changed between runs.
    results = run_dataset(dataset(), lambda text: text)

    assert results[0].input_sha256 == input_digest("message 0")


def test_latency_is_measured() -> None:
    results = run_dataset(dataset(), lambda text: text)

    assert all(result.latency_ms is not None and result.latency_ms >= 0 for result in results)


def test_a_target_that_raises_is_recorded_and_the_run_continues() -> None:
    # A model failing on one item is ordinary; losing every other item would be worse.
    def flaky(text: JsonValue) -> JsonValue:
        if text == "message 2":
            raise TimeoutError("upstream took too long")
        return text

    results = run_dataset(dataset(), flaky)

    assert len(results) == 5
    failed = [result for result in results if result.error is not None]
    assert [result.id for result in failed] == ["item-2"]
    assert failed[0].error == "TimeoutError: upstream took too long"
    assert failed[0].output is None
    assert failed[0].input_sha256 == input_digest("message 2")


def test_progress_is_reported_through_the_callback() -> None:
    seen: list[int] = []

    run_dataset(dataset(), lambda text: text, on_item=lambda index, item: seen.append(index))

    assert seen == [0, 1, 2, 3, 4]
