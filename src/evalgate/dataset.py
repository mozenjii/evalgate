"""Versioned evaluation datasets: items, slices, provenance and a content hash.

A dataset is a JSONL file of items plus a manifest beside it. JSONL because a dataset is
reviewed in pull requests like any other source, and one item per line gives a readable diff
when someone adds forty cases; a single JSON array re-indents the whole file instead.

Every item must say where it came from and which slice it belongs to. Both are refused when
missing rather than defaulted, because a per-slice gate built on items of unknown origin
cannot say anything about any slice, and an item quietly filed under "unknown" is how a
dataset stops meaning what its manifest claims.

Deliberate boundary: this module loads, validates and slices. It does not run anything, score
anything or know what a model is.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

__all__ = [
    "Dataset",
    "DatasetError",
    "Item",
    "Manifest",
    "Provenance",
    "content_hash",
    "load_dataset",
]


class Provenance(StrEnum):
    """Where an item came from.

    Recorded per item because the origin changes what a result means. A slice built entirely
    from promoted production failures is not a sample of anything, and a reader comparing it
    with a hand-written slice needs to see that without going back to the commit history.
    """

    HAND_WRITTEN = "hand_written"
    """Written by a person as a deliberate test case."""

    MINED_FROM_LOGS = "mined_from_logs"
    """Taken from real traffic, redacted, and reviewed before being added."""

    SYNTHETIC = "synthetic"
    """Generated. Cheap and plentiful, and never evidence on its own."""

    PROMOTED_FAILURE = "promoted_failure"
    """A failure seen in production, reduced to a minimal case. Never removed silently."""


class DatasetError(Exception):
    """A dataset could not be loaded, with the file and line that caused it.

    One exception type rather than a hierarchy: every caller does the same thing with it,
    which is to stop and show the message to a person. The message carries the detail.
    """


class Item(BaseModel):
    """One evaluation case.

    Attributes:
        id: Stable identifier, unique within the dataset. Results are paired across runs by
            this, so renaming one breaks the comparison it appears in.
        input: What the feature under test receives.
        expected: What it should produce, where that can be written down. `None` is allowed
            for items only a judge can assess, and is not the same as an empty expectation.
        slices: One value for every slice dimension the manifest declares.
        provenance: Where this item came from.
        source: Optional free-text pointer, such as a ticket or a run id.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    input: JsonValue
    expected: JsonValue | None = None
    slices: Mapping[str, str]
    provenance: Provenance
    source: str | None = None

    @model_validator(mode="after")
    def _require_a_slice(self) -> Self:
        """Refuse an item that belongs to no slice.

        A gate reports per slice. An item outside every slice is never compared and never
        reported, so it is carried as reassurance that nothing checks.
        """
        if not self.slices:
            raise ValueError("item declares no slices; every item must belong to at least one")
        return self


class Manifest(BaseModel):
    """What a dataset claims about itself.

    Attributes:
        name: Dataset name, used in reports.
        version: The author's version string. Changing items without changing this is what
            makes two runs silently incomparable.
        items_file: Path to the JSONL file, relative to the manifest.
        content_sha256: Hash of the items, as computed by `content_hash`.
        slice_dimensions: Each slice dimension and the values it is allowed to take.
        description: Optional note for a reader.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    items_file: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    slice_dimensions: Mapping[str, tuple[str, ...]]
    description: str | None = None

    @model_validator(mode="after")
    def _require_dimensions_with_values(self) -> Self:
        """Refuse a manifest whose slice dimensions are empty or valueless."""
        if not self.slice_dimensions:
            raise ValueError("manifest declares no slice dimensions; a gate reports per slice")
        for dimension, values in self.slice_dimensions.items():
            if not values:
                raise ValueError(f"slice dimension {dimension!r} declares no values")
            if len(set(values)) != len(values):
                raise ValueError(f"slice dimension {dimension!r} repeats a value")
        return self


@dataclass(frozen=True)
class Dataset:
    """A manifest and the items it describes, already validated against each other.

    Attributes:
        manifest: The dataset's own claims.
        items: Every item, in file order. Order is preserved because paired comparisons read
            far better when two runs list their items the same way.
    """

    manifest: Manifest
    items: tuple[Item, ...]

    def __len__(self) -> int:
        """Number of items."""
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        """Iterate items in file order."""
        return iter(self.items)

    def slice(self, dimension: str, value: str) -> tuple[Item, ...]:
        """Return the items whose `dimension` equals `value`.

        Args:
            dimension: A slice dimension declared in the manifest.
            value: One of that dimension's declared values.

        Returns:
            The matching items, in file order. Possibly empty, which is a real answer: a
            declared slice with no items is a coverage gap worth reporting.

        Raises:
            DatasetError: If the dimension or value is not declared. Asking for a slice that
                does not exist is a typo, and returning an empty tuple would hide it.
        """
        allowed = self.manifest.slice_dimensions.get(dimension)
        if allowed is None:
            known = ", ".join(sorted(self.manifest.slice_dimensions))
            raise DatasetError(f"no slice dimension {dimension!r}; declared dimensions: {known}")
        if value not in allowed:
            raise DatasetError(
                f"{value!r} is not a declared value of {dimension!r}; allowed: {', '.join(allowed)}"
            )
        return tuple(item for item in self.items if item.slices[dimension] == value)

    def slice_sizes(self) -> dict[tuple[str, str], int]:
        """Count the items in every declared slice, including the empty ones.

        Returns:
            A mapping from `(dimension, value)` to item count. Empty slices are present with
            a count of zero, because the report needs to show them as gaps rather than omit
            them.
        """
        return {
            (dimension, value): sum(1 for item in self.items if item.slices[dimension] == value)
            for dimension, values in self.manifest.slice_dimensions.items()
            for value in values
        }


def content_hash(items: Sequence[Item]) -> str:
    """Hash the items' content, independently of how the file was formatted.

    Hashing the raw bytes would make the digest depend on key order, indentation and line
    endings, so a checkout on another platform would look like tampering. This hashes each
    item's canonical JSON in file order instead, which changes when the data changes and not
    when the formatting does.

    Args:
        items: The items, in file order. Order is part of the hash.

    Returns:
        Hex-encoded SHA-256.
    """
    digest = hashlib.sha256()
    for item in items:
        canonical = json.dumps(
            item.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_dataset(manifest_path: Path, *, verify_hash: bool = True) -> Dataset:
    """Load and validate a dataset from its manifest.

    Every failure here is refused rather than repaired. A dataset that loads with warnings is
    a dataset whose warnings nobody reads.

    Args:
        manifest_path: Path to the manifest JSON file.
        verify_hash: Whether to require the manifest's hash to match the items. Only a tool
            that is rewriting the manifest should pass `False`, and it must then write a
            fresh hash.

    Returns:
        The validated dataset.

    Raises:
        DatasetError: If the manifest or any item is missing, malformed, duplicated,
            mis-sliced, or if the content hash does not match.
    """
    manifest = _load_manifest(manifest_path)
    items_path = (manifest_path.parent / manifest.items_file).resolve()
    items = _load_items(items_path, manifest)

    if verify_hash:
        actual = content_hash(items)
        if actual != manifest.content_sha256:
            raise DatasetError(
                f"{manifest_path}: content hash does not match {items_path.name}.\n"
                f"  manifest says {manifest.content_sha256}\n"
                f"  items hash to {actual}\n"
                "The items changed without the manifest being updated, so two runs of this "
                "dataset are not comparable. Update the manifest on purpose."
            )

    return Dataset(manifest=manifest, items=items)


def _load_manifest(path: Path) -> Manifest:
    """Read and validate the manifest file.

    Raises:
        DatasetError: If the file is missing, is not JSON, or fails validation.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatasetError(f"{path}: cannot read manifest: {error}") from error

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise DatasetError(
            f"{path}:{error.lineno}: manifest is not valid JSON: {error.msg}"
        ) from error

    try:
        return Manifest.model_validate(parsed)
    except ValidationError as error:
        raise DatasetError(f"{path}: manifest is invalid:\n{_explain(error)}") from error


def _load_items(path: Path, manifest: Manifest) -> tuple[Item, ...]:
    """Read the JSONL items and check each one against the manifest.

    Raises:
        DatasetError: If the file is missing, a line is malformed, an id repeats, or an item's
            slices do not match the manifest's declared dimensions.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DatasetError(f"{path}: cannot read items: {error}") from error

    items: list[Item] = []
    seen: dict[str, int] = {}

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue  # A blank line between groups of items is formatting, not data.

        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"{path}:{number}: not valid JSON: {error.msg}") from error

        try:
            item = Item.model_validate(parsed)
        except ValidationError as error:
            raise DatasetError(f"{path}:{number}: invalid item:\n{_explain(error)}") from error

        if item.id in seen:
            raise DatasetError(
                f"{path}:{number}: duplicate item id {item.id!r}, first seen on line "
                f"{seen[item.id]}. Runs are paired by id, so a repeat silently pairs the "
                "wrong items."
            )
        seen[item.id] = number

        _check_slices(item, manifest, path, number)
        items.append(item)

    if not items:
        raise DatasetError(f"{path}: dataset is empty; there is nothing to compare")

    return tuple(items)


def _check_slices(item: Item, manifest: Manifest, path: Path, number: int) -> None:
    """Require every declared dimension, with a declared value, on one item.

    Every dimension is required rather than optional. A gate reports one verdict per slice, so
    an item missing a dimension would be absent from that dimension's comparison while still
    counting towards the dataset's size -- the slice would look thinner than the dataset and
    nobody would know why. Where a dimension genuinely does not apply, the manifest should
    declare an explicit value for that, so the choice is visible.

    Raises:
        DatasetError: If a dimension is missing, undeclared, or carries an undeclared value.
    """
    declared = set(manifest.slice_dimensions)
    present = set(item.slices)

    missing = declared - present
    if missing:
        raise DatasetError(
            f"{path}:{number}: item {item.id!r} is missing slice dimension(s): "
            f"{', '.join(sorted(missing))}"
        )

    undeclared = present - declared
    if undeclared:
        raise DatasetError(
            f"{path}:{number}: item {item.id!r} declares slice dimension(s) the manifest does "
            f"not know: {', '.join(sorted(undeclared))}"
        )

    for dimension, value in item.slices.items():
        allowed = manifest.slice_dimensions[dimension]
        if value not in allowed:
            raise DatasetError(
                f"{path}:{number}: item {item.id!r} has {dimension}={value!r}, which the "
                f"manifest does not allow. Allowed: {', '.join(allowed)}"
            )


def _explain(error: ValidationError) -> str:
    """Render a pydantic error as indented `field: message` lines.

    Pydantic's own rendering carries urls and type codes that bury the one line a person needs.
    """
    lines: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "(root)"
        lines.append(f"  {location}: {detail['msg']}")
    return "\n".join(lines)
