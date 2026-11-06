"""Produce results: run each item of a dataset through a target and record what came back.

The runner is the convenient way to make a results file, not the only one. It calls a Python
function once per item, times it, and records the output with a hash of the input it answered.
It does not score anything: scoring happens at comparison time, against the dataset as it
stands then, so a results file stays valid when a scorer is added or fixed.

Deliberate boundary: a target that raises on one item is recorded against that item and the run
continues. A model timing out on item 40 of 400 is ordinary, and losing the other 399 would be
the larger failure. The comparison counts such an item as a failure, not as missing.
"""

from __future__ import annotations

import importlib
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from evalgate.dataset import Dataset, Item
from evalgate.results import ItemResult, input_digest

__all__ = ["Target", "TargetError", "load_target", "run_dataset"]

Target = Callable[[JsonValue], JsonValue]
"""What is under test: anything that turns an item's input into an output.

A plain callable rather than a class, so a prompt, an HTTP client or a shell command can each
be adapted to it in a few lines without inheriting anything from this package.
"""


class TargetError(Exception):
    """A target reference could not be turned into something callable."""


def load_target(reference: str, pythonpath: Sequence[Path] = ()) -> Target:
    """Import a target from a `module:function` reference.

    The directories in `pythonpath` are put at the front of the import path and left there for
    the life of the process, so a target that imports its own helpers lazily, inside the
    function, still finds them. That trust is the same a test runner extends to the tests it
    collects: the reference comes from the person running the command or their checked-in
    configuration, never from a dataset or a results file.

    Args:
        reference: `module:function`, for example `myapp.triage:classify`.
        pythonpath: Directories to search first.

    Returns:
        The callable.

    Raises:
        TargetError: If the reference is malformed, the module cannot be imported, or the
            attribute is missing or not callable.
    """
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise TargetError(f"target must look like 'module:function', got {reference!r}")

    for directory in reversed(pythonpath):
        entry = str(directory)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise TargetError(f"cannot import {module_name!r}: {error}") from error

    found: object = module
    for part in attribute.split("."):
        if not hasattr(found, part):
            raise TargetError(f"{module_name!r} has no attribute {attribute!r}")
        found = getattr(found, part)
    if not callable(found):
        raise TargetError(f"{reference!r} is not callable")
    return cast("Target", found)


def run_dataset(
    dataset: Dataset, target: Target, *, on_item: Callable[[int, Item], None] | None = None
) -> tuple[ItemResult, ...]:
    """Call the target once per item, in dataset order.

    Args:
        dataset: The items to run.
        target: What is being tested.
        on_item: Called with `(index, item)` before each call, for progress. A callback, so this
            module never decides how output should look.

    Returns:
        One result per item, each carrying the hash of the input it answered.
    """
    produced: list[ItemResult] = []
    for index, item in enumerate(dataset):
        if on_item is not None:
            on_item(index, item)
        produced.append(_run_one(item, target))
    return tuple(produced)


def _run_one(item: Item, target: Target) -> ItemResult:
    """Call the target for one item, recording an error instead of raising it."""
    digest = input_digest(item.input)
    started = time.perf_counter()
    try:
        output = target(item.input)
    except Exception as error:
        return ItemResult(
            id=item.id,
            error=f"{type(error).__name__}: {error}",
            input_sha256=digest,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    return ItemResult(
        id=item.id,
        output=output,
        input_sha256=digest,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )
