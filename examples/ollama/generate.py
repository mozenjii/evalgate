"""Adapt nine public UCI Iris rows into handwritten classification cases.

Run from the repository root with `uv run python examples/ollama/generate.py`.
The measurements and labels below are fixed to the cited rows of UCI's iris.data, so
regenerating the example needs no network. The generated files are versioned together.
"""

from __future__ import annotations

from pathlib import Path

from evalgate.dataset import Item, Manifest, Provenance, content_hash

HERE = Path(__file__).resolve().parent
SOURCE = "Fisher (1936), UCI Iris, https://doi.org/10.24432/C56C76, iris.data row"
SPECIES = ("setosa", "versicolor", "virginica")
ROWS = (
    (1, "5.1", "3.5", "1.4", "0.2", "setosa"),
    (11, "5.4", "3.7", "1.5", "0.2", "setosa"),
    (21, "5.4", "3.4", "1.7", "0.2", "setosa"),
    (51, "7.0", "3.2", "4.7", "1.4", "versicolor"),
    (61, "5.0", "2.0", "3.5", "1.0", "versicolor"),
    (71, "5.9", "3.2", "4.8", "1.8", "versicolor"),
    (101, "6.3", "3.3", "6.0", "2.5", "virginica"),
    (111, "6.5", "3.2", "5.1", "2.0", "virginica"),
    (121, "6.9", "3.2", "5.7", "2.3", "virginica"),
)


def main() -> None:
    """Write the versioned items and their manifest."""
    items = tuple(
        Item(
            id=f"iris-{row:03d}",
            input=(
                "Classify an iris by its measurements in centimeters. "
                f"Sepal length: {sl}, sepal width: {sw}, petal length: {pl}, petal width: {pw}. "
                "Return only a JSON object with the key species. "
                "Use one of these lowercase values: setosa, versicolor, virginica."
            ),
            expected={"species": species},
            slices={"species": species},
            provenance=Provenance.HAND_WRITTEN,
            source=f"{SOURCE} {row}",
        )
        for row, sl, sw, pl, pw, species in ROWS
    )
    manifest = Manifest(
        name="uci-iris-ollama",
        version="1",
        items_file="items.jsonl",
        content_sha256=content_hash(items),
        slice_dimensions={"species": SPECIES},
        description="Nine public Iris cases adapted for a local model and replay demonstration.",
    )
    (HERE / "items.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in items), encoding="utf-8"
    )
    (HERE / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
