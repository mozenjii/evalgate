"""Prepare redacted case candidates and append reviewed failures to a new dataset version."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, TypeVar

import typer
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from evalgate.dataset import Dataset, DatasetError, Item, Provenance, content_hash, load_dataset
from evalgate.results import input_digest

__all__ = ["CaseError", "mine", "mine_command", "promote", "promote_command"]

_TraceLabel = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
_SKETCH_SIZE = 64
_SHINGLE_SIZE = 3
_WORDS = re.compile(r"\w+", re.UNICODE)


class CaseError(Exception):
    """The case records or requested dataset publication could not be used safely."""


class _Trace(BaseModel):
    """Allowlisted projection of an already scrubbed trace or a synthetic record."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    classification: Literal["redacted", "synthetic"]
    input: JsonValue
    slices: dict[str, str] = Field(min_length=1)
    origin_run: _TraceLabel
    suspected_stage: _TraceLabel | None = None
    failure: bool = False
    judge_votes: list[bool] | None = None


class _Review(_Trace):
    """One trace with an explicitly recorded human review and expected result."""

    reviewed: bool = False
    expected: JsonValue | None = None


_Record = TypeVar("_Record", bound=_Trace)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Refuse duplicate keys, including a trace with two different classifications."""
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate object key")
        result[key] = value
    return result


def _read_records(path: Path, model: type[_Record]) -> Iterator[tuple[int, _Record]]:
    """Read JSONL without putting the rejected input or its validation values in diagnostics."""
    try:
        with path.open(encoding="utf-8") as file:
            for number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    # Use one parser with duplicate-key checks, then reject nonfinite values
                    # in nested JSON as well as at the top level.
                    parsed: JsonValue = json.loads(line, object_pairs_hook=_unique_object)
                    json.dumps(parsed, allow_nan=False, ensure_ascii=False).encode("utf-8")
                    yield number, model.model_validate(parsed)
                except (ValueError, RecursionError):
                    raise CaseError(
                        f"{path}:{number}: invalid {model.__name__.lstrip('_')} row"
                    ) from None
    except (OSError, UnicodeError):
        raise CaseError(f"{path}: cannot read UTF-8 JSONL") from None


def _case_digest(value: JsonValue, slices: Mapping[str, str]) -> str:
    """Identify the same input independently of JSON object ordering."""
    slice_values: dict[str, JsonValue] = dict(slices)
    return input_digest({"input": value, "slices": slice_values})


def _has_disagreement(trace: _Trace) -> bool:
    """Give divided votes priority without giving an unreviewed trace a label."""
    return trace.judge_votes is not None and len(set(trace.judge_votes)) > 1


def _sketch(value: JsonValue) -> frozenset[bytes]:
    """Take a deterministic bottom-k MinHash sketch of a string's word shingles."""
    if not isinstance(value, str):
        return frozenset()
    words = _WORDS.findall(value.casefold())
    if len(words) < _SHINGLE_SIZE:
        # A near-duplicate rule on one or two words would erase distinct short intents.
        return frozenset()
    shingles = {
        " ".join(words[index : index + _SHINGLE_SIZE])
        for index in range(len(words) - _SHINGLE_SIZE + 1)
    }
    hashes = (hashlib.sha256(shingle.encode("utf-8")).digest() for shingle in shingles)
    return frozenset(heapq.nsmallest(_SKETCH_SIZE, hashes))


def _similarity(left: frozenset[bytes], right: frozenset[bytes]) -> float:
    """Estimate Jaccard similarity from the smallest hashes in the shingle union."""
    sampled = set(heapq.nsmallest(_SKETCH_SIZE, left | right))
    return len(sampled & left & right) / len(sampled)


def _select_candidates(traces: Iterable[_Trace], threshold: float | None) -> Iterator[_Review]:
    """Prefer disagreements and keep one representative of each near-duplicate group."""
    ordered = sorted(traces, key=_has_disagreement, reverse=True)
    exact: set[str] = set()
    sketches: dict[tuple[str, tuple[tuple[str, str], ...]], list[frozenset[bytes]]] = {}
    for trace in ordered:
        digest = _case_digest(trace.input, trace.slices)
        if digest in exact:
            continue
        exact.add(digest)
        if threshold is not None and (sketch := _sketch(trace.input)):
            group = (trace.classification, tuple(sorted(trace.slices.items())))
            selected = sketches.setdefault(group, [])
            if any(_similarity(sketch, prior) >= threshold for prior in selected):
                continue
            selected.append(sketch)
        yield _Review.model_validate(trace.model_dump())


def _jsonl(items: Iterable[BaseModel]) -> str:
    """Preserve explicit JSON null inputs when writing line-oriented records."""
    return "".join(item.model_dump_json() + "\n" for item in items)


def _publish_text(path: Path, value: str) -> None:
    """Publish a complete file atomically, refusing to overwrite an existing path."""
    staging: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False, dir=path.parent
        ) as file:
            staging = Path(file.name)
            file.write(value)
        path.hardlink_to(staging)
    except FileExistsError:
        raise CaseError(f"{path}: already exists") from None
    except OSError:
        raise CaseError(f"{path}: cannot publish a new file") from None
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def mine(logs: Path, output: Path, *, near_duplicate_threshold: float | None = None) -> int:
    """Mine an inspectable review queue from explicitly classified offline JSONL."""
    if near_duplicate_threshold is not None and not 0 < near_duplicate_threshold <= 1:
        raise CaseError("near-duplicate threshold must be greater than 0 and at most 1")
    selected = tuple(
        _select_candidates(
            (trace for _, trace in _read_records(logs, _Trace)), near_duplicate_threshold
        )
    )
    _publish_text(output, _jsonl(selected))
    return len(selected)


def _reviewed_case(candidate: _Review, dataset: Dataset, location: str) -> Item:
    """Convert a reviewed and reproducible failure into a fully sliced golden case."""
    if not candidate.reviewed or not candidate.failure:
        raise CaseError(f"{location}: promotion requires a reviewed failure")
    if candidate.expected is None:
        raise CaseError(f"{location}: promotion requires an explicit non-null expectation")
    if candidate.suspected_stage is None:
        raise CaseError(f"{location}: promotion requires a suspected_stage")
    allowed = dataset.manifest.slice_dimensions
    if set(candidate.slices) != set(allowed) or any(
        value not in allowed[dimension] for dimension, value in candidate.slices.items()
    ):
        raise CaseError(f"{location}: slices must match the source manifest")
    provenance = (
        Provenance.SYNTHETIC
        if candidate.classification == "synthetic"
        else Provenance.PROMOTED_FAILURE
    )
    return Item(
        id=f"case-{_case_digest(candidate.input, candidate.slices)}",
        input=candidate.input,
        expected=candidate.expected,
        slices=candidate.slices,
        provenance=provenance,
        ratchet=True,
        source=json.dumps(
            {"origin_run": candidate.origin_run, "suspected_stage": candidate.suspected_stage},
            ensure_ascii=False,
        ),
    )


def _append_cases(reviews: Path, dataset: Dataset) -> tuple[Item, ...]:
    """Reject all conflicts before writing either file of the new version."""
    seen = {_case_digest(item.input, item.slices) for item in dataset.items}
    ids = {item.id for item in dataset.items}
    added: list[Item] = []
    for number, candidate in _read_records(reviews, _Review):
        location = f"{reviews}:{number}"
        item = _reviewed_case(candidate, dataset, location)
        digest = _case_digest(item.input, item.slices)
        if digest in seen or item.id in ids:
            raise CaseError(f"{location}: duplicate case input and slices or conflicting id")
        seen.add(digest)
        ids.add(item.id)
        added.append(item)
    if not added:
        raise CaseError(f"{reviews}: no reviewed failures")
    return (*dataset.items, *added)


def promote(reviews: Path, manifest_path: Path, version: str, output_dir: Path) -> Path:
    """Publish the unchanged source cases plus reviewed failures as a new version."""
    try:
        dataset = load_dataset(manifest_path)
    except (DatasetError, UnicodeError):
        raise CaseError(f"{manifest_path}: cannot load a validated source dataset") from None
    if not version.strip() or version != version.strip() or version == dataset.manifest.version:
        raise CaseError("promotion requires a new nonblank dataset version")
    items = _append_cases(reviews, dataset)
    manifest = dataset.manifest.model_copy(
        update={
            "version": version,
            "items_file": "items.jsonl",
            "content_sha256": content_hash(items),
        }
    )
    encoded_items = _jsonl(items)
    encoded_manifest = manifest.model_dump_json(indent=2, exclude_none=True) + "\n"
    try:
        output_dir.mkdir()
    except FileExistsError:
        raise CaseError(f"{output_dir}: already exists") from None
    except OSError:
        raise CaseError(f"{output_dir}: cannot create a new dataset directory") from None

    items_path = output_dir / manifest.items_file
    new_manifest = output_dir / "manifest.json"
    published_items = False
    try:
        _publish_text(items_path, encoded_items)
        published_items = True
        _publish_text(new_manifest, encoded_manifest)
    except CaseError as error:
        try:
            if published_items:
                items_path.unlink()
            output_dir.rmdir()
        except OSError:
            raise CaseError(
                f"{output_dir}: incomplete dataset directory could not be cleaned"
            ) from error
        raise
    return new_manifest


def mine_command(
    logs: Annotated[Path, typer.Argument(help="Redacted or synthetic JSONL traces.")],
    out: Annotated[Path, typer.Option(help="New review JSONL file.")],
    near_duplicate_threshold: Annotated[
        float | None, typer.Option(help="Optional MinHash similarity threshold for text traces.")
    ] = None,
) -> None:
    """Deduplicate offline traces and prepare candidates for human review."""
    try:
        count = mine(logs, out, near_duplicate_threshold=near_duplicate_threshold)
    except CaseError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(3) from error
    typer.echo(f"{count} candidate(s) written to {out}. Review each before promotion.")


def promote_command(
    reviews: Annotated[Path, typer.Argument(help="Reviewed failure cases in JSONL.")],
    manifest: Annotated[Path, typer.Option(help="Source dataset manifest.")],
    version: Annotated[str, typer.Option(help="New dataset version.")],
    out_dir: Annotated[Path, typer.Option(help="New dataset directory.")],
) -> None:
    """Append reviewed failures and write a new immutable dataset version."""
    try:
        new_manifest = promote(reviews, manifest, version, out_dir)
    except CaseError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(3) from error
    typer.echo(f"Dataset version {version} written to {new_manifest}")
