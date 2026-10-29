"""Run a dataset against a target and record what happened.

Owning artifact: `product context/00_PROJECT_BRIEF.md`, section 4.

The runner is deliberately thin. It walks items in order, calls the target, times the call,
scores the output and writes a row. Everything interesting -- what counts as worse, which
slice regressed -- happens elsewhere, on the recorded rows, where it can be re-examined
without paying for the run again.

Deliberate boundary: two kinds of failure are treated differently, because they are different.
A target that raises on one item is an operational failure: it is recorded against that item
and the run continues, since a model timing out on item 40 of 400 is normal and losing the
other 399 would be worse. A scorer that raises is a configuration failure: the scorer and the
dataset disagree about what an item looks like, that is true of every item, and continuing
would produce a run whose scores mean nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from pydantic import JsonValue

from evalgate.dataset import Dataset, Item
from evalgate.runs import ItemOutcome, RunRecord, RunStore
from evalgate.scorers import Scorer, score_all

__all__ = ["Target", "run_dataset"]

Target = Callable[[JsonValue], JsonValue]
"""What is under test: anything that turns an item's input into an output.

A plain callable rather than a class, so a prompt, an HTTP client or a shell command can all
be adapted to it in a few lines without inheriting anything from this package.
"""


def run_dataset(
    dataset: Dataset,
    target: Target,
    scorers: Sequence[Scorer],
    store: RunStore,
    *,
    label: str,
    run_id: str | None = None,
    notes: str | None = None,
    on_item: Callable[[int, Item], None] | None = None,
) -> RunRecord:
    """Run every item through the target, score it, and record the result.

    Args:
        dataset: The dataset to run. Its name, version and content hash are recorded with the
            run, so a later edit to the items is detectable from the run alone.
        target: What is being tested.
        scorers: Deterministic scorers to apply to every item.
        store: Where to write the run.
        label: What to call this version in reports -- a prompt version, a model, a commit.
        run_id: Identifier, generated when omitted.
        notes: Optional free text stored with the run.
        on_item: Called with `(index, item)` before each item, for progress reporting. Kept as
            a callback so this module never decides how output should look.

    Returns:
        The run's record, already marked finished.

    Raises:
        ScorerError: If a scorer cannot be applied. The run is left unfinished in the store,
            which is how a reader tells it apart from one that completed.
        StoreError: If the run cannot be written.
    """
    record = store.start_run(
        dataset_name=dataset.manifest.name,
        dataset_version=dataset.manifest.version,
        dataset_sha256=dataset.manifest.content_sha256,
        target=label,
        run_id=run_id,
        notes=notes,
    )

    for index, item in enumerate(dataset):
        if on_item is not None:
            on_item(index, item)
        store.record(record.run_id, _run_one(item, target, scorers))

    store.finish_run(record.run_id)
    return store.get_run(record.run_id)


def _run_one(item: Item, target: Target, scorers: Sequence[Scorer]) -> ItemOutcome:
    """Call the target for one item and score what came back.

    A target that raises is recorded as an error on that item rather than ending the run: a
    model failing on one item out of four hundred is ordinary, and discarding the other three
    hundred and ninety-nine would be the larger loss. A scorer that raises is left to
    propagate, because it means the scorer cannot read this dataset at all.
    """
    started = time.perf_counter()
    try:
        output = target(item.input)
    except Exception as error:
        return ItemOutcome(
            item_id=item.id,
            output=None,
            error=f"{type(error).__name__}: {error}",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    latency_ms = (time.perf_counter() - started) * 1000.0
    return ItemOutcome(
        item_id=item.id,
        output=output,
        latency_ms=latency_ms,
        scores=score_all(scorers, output, item.expected),
    )
