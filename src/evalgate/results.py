"""Results files: what one version of a feature produced for each item of a dataset.

A results file is JSONL. An optional first line describes the run; every other line is one
item. It is the hand-off between producing outputs and judging them, and it is deliberately
simple enough for a team's own pipeline, in any language, to write: one
`{"id": ..., "output": ...}` object per line is already a complete, valid file.

That simplicity is the point. Most teams already have a way to call their model. What they
lack is a trustworthy comparison, and they should be able to get it without handing their
pipeline to this package.

Deliberate boundary: this module reads, writes and validates files. It does not score, pair or
compare anything.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

__all__ = [
    "FORMAT",
    "VERSION",
    "DatasetRef",
    "ItemResult",
    "Results",
    "ResultsError",
    "ResultsHeader",
    "input_digest",
    "read_results",
    "write_results",
]

FORMAT = "evalgate.results"
"""The value of `format` in a header line."""

VERSION = 1
"""The results format version this build reads and writes."""


class ResultsError(Exception):
    """A results file could not be read or written, with the file and line that caused it."""


class DatasetRef(BaseModel):
    """Which dataset a run used, as its manifest described it at the time.

    Attributes:
        name: Dataset name.
        version: Dataset version.
        sha256: Content hash of the items. A later comparison uses it to notice that the
            dataset has changed since this run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)


class ResultsHeader(BaseModel):
    """The optional first line of a results file.

    Attributes:
        format: Always `evalgate.results`. It is how a header line is told apart from an item.
        version: The format version.
        label: What this run was: a prompt version, a model, a commit.
        dataset: The dataset the run used, when known.
        created_at: When the run finished.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["evalgate.results"] = FORMAT
    version: int = VERSION
    label: str = Field(min_length=1)
    dataset: DatasetRef | None = None
    created_at: datetime | None = None


class ItemResult(BaseModel):
    """What one item produced in one run.

    Unknown fields are refused rather than ignored: a mistyped `ouput` would otherwise read as
    an item that produced nothing, and be scored as a failure nobody caused. Anything else a
    pipeline wants to carry goes in `meta`.

    Attributes:
        id: The dataset item's id. Runs are paired by it.
        output: What the feature returned. `None` when it failed.
        error: Why it failed, or `None`.
        input_sha256: Hash of the input this output answered, from `input_digest`. Lets a
            comparison notice that an item's input changed between two runs, in which case the
            two outputs answer different questions and must not be paired.
        latency_ms: Wall-clock milliseconds for the call.
        prompt_tokens: Tokens sent, where known.
        completion_tokens: Tokens received, where known.
        cost_usd: Cost of the call, where known.
        meta: Free-form extras, carried through untouched.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    output: JsonValue = None
    error: str | None = None
    input_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    latency_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    meta: dict[str, JsonValue] | None = None


@dataclass(frozen=True)
class Results:
    """A results file, read and validated.

    Attributes:
        header: The header line, or `None` for a file a pipeline wrote without one.
        items: Every item, in file order.
        source: Where it was read from, for messages.
    """

    header: ResultsHeader | None
    items: tuple[ItemResult, ...]
    source: Path | None = None

    @property
    def label(self) -> str:
        """The run's label, falling back to the file name when there is no header."""
        if self.header is not None:
            return self.header.label
        return self.source.stem if self.source is not None else "results"

    def by_id(self) -> dict[str, ItemResult]:
        """Items keyed by id."""
        return {item.id: item for item in self.items}


def input_digest(value: JsonValue) -> str:
    """Hash an item's input the same way regardless of how it was formatted.

    Canonical JSON -- sorted keys, no whitespace -- so the digest depends on the input and not
    on the key order a particular producer happened to write.

    Args:
        value: The input, as the dataset holds it.

    Returns:
        Hex-encoded SHA-256.
    """
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_results(path: Path) -> Results:
    """Read and validate a results file.

    Args:
        path: The JSONL file.

    Returns:
        The header, if the first line is one, and every item.

    Raises:
        ResultsError: If the file is missing, a line is not JSON or not a valid item, a header
            appears anywhere but the first line, the format version is newer than this build,
            an id repeats, or there are no items at all.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResultsError(f"{path}: cannot read results: {error}") from error

    header: ResultsHeader | None = None
    items: list[ItemResult] = []
    seen: dict[str, int] = {}
    first = True

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed: JsonValue = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResultsError(f"{path}:{number}: not valid JSON: {error.msg}") from error

        if isinstance(parsed, dict) and "format" in parsed:
            if not first:
                raise ResultsError(f"{path}:{number}: a header line must be the first line")
            header = _parse_header(parsed, path, number)
        else:
            item = _parse_item(parsed, path, number)
            if item.id in seen:
                raise ResultsError(
                    f"{path}:{number}: duplicate id {item.id!r}, first seen on line "
                    f"{seen[item.id]}. Two outputs for one item leave nothing to say which counts."
                )
            seen[item.id] = number
            items.append(item)
        first = False

    if not items:
        raise ResultsError(f"{path}: no items; there is nothing to compare")
    return Results(header=header, items=tuple(items), source=path)


def write_results(path: Path, header: ResultsHeader, items: Sequence[ItemResult]) -> None:
    """Write a results file, replacing any file already there.

    The file is written beside its destination and moved into place, so a run interrupted half
    way never leaves a truncated file that reads as a run with fewer items.

    Args:
        path: Destination.
        header: The run's header line.
        items: Every item, in order.

    Raises:
        ResultsError: If the file cannot be written.
    """
    lines = [header.model_dump_json(exclude_none=True)]
    lines.extend(item.model_dump_json(exclude_none=True) for item in items)
    staging = path.with_name(path.name + ".partial")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging.write_text("\n".join(lines) + "\n", encoding="utf-8")
        staging.replace(path)
    except OSError as error:
        staging.unlink(missing_ok=True)
        raise ResultsError(f"{path}: cannot write results: {error}") from error


def _parse_header(parsed: dict[str, JsonValue], path: Path, number: int) -> ResultsHeader:
    """Validate a header line, refusing formats this build does not understand."""
    if parsed.get("format") != FORMAT:
        raise ResultsError(f"{path}:{number}: unknown format {parsed.get('format')!r}")
    version = parsed.get("version")
    if isinstance(version, int) and version > VERSION:
        raise ResultsError(
            f"{path}:{number}: written in results format {version}, but this build reads "
            f"{VERSION}. Upgrade evalgate rather than reading it as if it were older."
        )
    try:
        return ResultsHeader.model_validate(parsed)
    except ValidationError as error:
        raise ResultsError(f"{path}:{number}: invalid header:\n{_explain(error)}") from error


def _parse_item(parsed: JsonValue, path: Path, number: int) -> ItemResult:
    """Validate one item line."""
    try:
        return ItemResult.model_validate(parsed)
    except ValidationError as error:
        raise ResultsError(f"{path}:{number}: invalid item:\n{_explain(error)}") from error


def _explain(error: ValidationError) -> str:
    """Render a pydantic error as indented `field: message` lines."""
    return "\n".join(
        f"  {'.'.join(str(part) for part in detail['loc']) or '(root)'}: {detail['msg']}"
        for detail in error.errors()
    )
