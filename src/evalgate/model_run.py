"""Plan and execute model runs with bounded local inference and redacted replay.

The model safety contract lives in the model-run guide. This layer owns persistence and
preflight. Transport, scoring and CLI presentation stay outside it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from evalgate.dataset import Dataset, Item
from evalgate.results import ItemResult, input_digest
from evalgate.runner import Target

__all__ = ["ModelRun", "ModelRunError", "Provider", "plan_model_run"]


class ModelRunError(Exception):
    """Planning or safe persistence failed, so the run cannot continue."""


class Provider(Protocol):
    """Model transport with a stable name and a complete, cacheable request body."""

    @property
    def name(self) -> str:
        """Identify the provider in cassette keys."""
        ...

    def request(self, value: JsonValue) -> dict[str, JsonValue]:
        """Prepare the request without making a network call."""
        ...

    def generate(self, request: dict[str, JsonValue]) -> JsonValue:
        """Make one configured generation attempt."""
        ...


class _Recording(BaseModel):
    """Only the redacted output is stored. A present JSON null is a valid recording."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    output: JsonValue


def _resolve(entry: _PlannedItem, provider: Provider, redact: Target | None) -> _Recording | None:
    """Make one attempt and persist only a validated, redacted success."""
    try:
        value = provider.generate(entry.request)
    except Exception:
        # Provider exceptions can contain prompts, tokens or response text.
        return None
    try:
        safe_value = redact(value) if redact is not None else value
        recording = _Recording(output=safe_value)
        encoded = json.dumps(recording.model_dump(), allow_nan=False, ensure_ascii=False).encode(
            "utf-8"
        )
    except Exception:
        raise ModelRunError("could not prepare a redacted model output") from None
    return _write_recording(entry.path, encoded) or recording


@dataclass(frozen=True)
class _PlannedItem:
    """One immutable decision about whether to replay or make a generation attempt."""

    item: Item
    request: dict[str, JsonValue]
    path: Path
    recording: _Recording | None


@dataclass(frozen=True)
class ModelRun:
    """A snapshot of replay hits and a conservative bound on live attempts."""

    _items: tuple[_PlannedItem, ...]
    _provider: Provider

    @property
    def misses(self) -> int:
        """Return the maximum number of generation attempts this run can make."""
        return len({entry.path for entry in self._items if entry.recording is None})

    @property
    def unrecorded_items(self) -> int:
        """Count dataset items for which replay has no recording."""
        return sum(entry.recording is None for entry in self._items)

    @property
    def hits(self) -> int:
        """Return the number of dataset items with usable recordings."""
        return len(self._items) - self.unrecorded_items

    def execute(
        self, *, allow_local: bool, max_calls: int, redact: Target | None = None
    ) -> tuple[ItemResult, ...]:
        """Enforce the call cap before execution, or omit missing items in replay mode."""
        if max_calls < 0 or (allow_local and self.misses > max_calls):
            raise ModelRunError(f"run needs up to {self.misses} model calls, cap is {max_calls}")
        attempts: dict[Path, _Recording | None] = {}
        results: list[ItemResult] = []
        for entry in self._items:
            recording = entry.recording
            if recording is None:
                if not allow_local:
                    continue
                if entry.path not in attempts:
                    attempts[entry.path] = _resolve(entry, self._provider, redact)
                recording = attempts[entry.path]
            results.append(_result(entry.item, recording))
        return tuple(results)


def _result(item: Item, recording: _Recording | None) -> ItemResult:
    """Distinguish a valid JSON null output from a failed generation."""
    digest = input_digest(item.input)
    if recording is None:
        return ItemResult(id=item.id, error="model generation failed", input_sha256=digest)
    return ItemResult(id=item.id, input_sha256=digest, output=recording.output)


_MAX_RECORDING = 8 * 1024 * 1024


def _read_recording(path: Path) -> _Recording | None:
    """Treat only a missing file as a miss. Corruption must not trigger a live call."""
    try:
        with path.open("rb") as file:
            data = file.read(_MAX_RECORDING + 1)
        if len(data) > _MAX_RECORDING:
            raise ModelRunError("model recording exceeds the size limit")
        recording = _Recording.model_validate_json(data)
        json.dumps(recording.output, allow_nan=False)
    except FileNotFoundError:
        return None
    except (OSError, ValidationError, ValueError):
        raise ModelRunError(f"cannot read model recording {path.name}") from None
    return recording


def _write_recording(path: Path, data: bytes) -> _Recording | None:
    """Publish atomically and return the winning recording if another writer got there first."""
    if len(data) > _MAX_RECORDING:
        raise ModelRunError("model recording exceeds the size limit")
    staging: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            staging = Path(file.name)
            file.write(data)
        path.hardlink_to(staging)
    except FileExistsError:
        existing = _read_recording(path)
        if existing is None:
            raise ModelRunError("model recording was removed during capture") from None
        return existing
    except OSError:
        raise ModelRunError(f"cannot write model recording {path.name}") from None
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    return None


def plan_model_run(
    dataset: Dataset,
    provider: Provider,
    directory: Path,
    *,
    redactor: str | None = None,
) -> ModelRun:
    """Read existing cache entries without mutating files or contacting the provider."""
    plan: list[_PlannedItem] = []
    recordings: dict[Path, _Recording | None] = {}
    for item in dataset:
        request = provider.request(item.input)
        key = input_digest([1, provider.name, redactor, request])
        path = directory / f"{key}.json"
        if path not in recordings:
            recordings[path] = _read_recording(path)
        plan.append(_PlannedItem(item, request, path, recordings[path]))
    return ModelRun(tuple(plan), provider)
