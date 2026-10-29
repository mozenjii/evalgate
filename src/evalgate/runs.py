"""Durable run records: what a version produced, what it cost, and how it scored.

Owning artifact: `product context/00_PROJECT_BRIEF.md`, section 4.

SQLite rather than more JSONL. Datasets are read by people in diffs, but runs are read by
queries -- every score for one scorer across two runs, paired by item -- and that is what a
relational store is for. It is also a single file with no server, which keeps a check
runnable on a laptop and in CI without either growing a dependency.

Deliberate boundary: this module stores and retrieves. It does not call targets, score
anything or decide a verdict. The pairing it returns is deliberately explicit about items
that appear in only one of two runs, because a comparison that quietly drops them reports a
narrower slice than the one it names.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from pydantic import JsonValue

from evalgate.scorers import Score, ScoreKind

__all__ = [
    "ItemOutcome",
    "Pairing",
    "RunRecord",
    "RunStore",
    "StoreError",
]

SCHEMA_VERSION = 1
"""Bumped whenever the tables change. A store written by a newer version is refused."""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    dataset_name     TEXT NOT NULL,
    dataset_version  TEXT NOT NULL,
    dataset_sha256   TEXT NOT NULL,
    target           TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    run_id             TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    item_id            TEXT NOT NULL,
    output_json        TEXT,
    error              TEXT,
    latency_ms         REAL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    cost_usd           REAL,
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id       TEXT NOT NULL,
    item_id      TEXT NOT NULL,
    scorer       TEXT NOT NULL,
    value        REAL NOT NULL,
    kind         TEXT NOT NULL,
    detail_json  TEXT NOT NULL,
    PRIMARY KEY (run_id, item_id, scorer),
    FOREIGN KEY (run_id, item_id) REFERENCES outcomes(run_id, item_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS scores_by_scorer ON scores (run_id, scorer);
"""


def _no_scores() -> Mapping[str, Score]:
    """Empty scores, as a typed factory.

    A bare `dict` types the field as a mapping of unknowns, which hides the shape of every
    score an outcome carries.
    """
    return {}


class StoreError(Exception):
    """A run store could not be opened, written to, or read as expected."""


@dataclass(frozen=True)
class RunRecord:
    """One execution of one dataset against one version under test.

    Attributes:
        run_id: Unique identifier.
        dataset_name: Name from the dataset's manifest.
        dataset_version: Version from the manifest. Two runs of different versions are not
            comparable, and the gate must be able to see that without re-reading the data.
        dataset_sha256: The dataset's content hash at the time of the run, so a later edit
            to the items is detectable from the run record alone.
        target: Label for what was run -- a prompt version, a model name, a commit.
        started_at: When the run began, in UTC.
        finished_at: When it completed, or `None` while it is still running or if it died.
        notes: Optional free text.
    """

    run_id: str
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    target: str
    started_at: datetime
    finished_at: datetime | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ItemOutcome:
    """What one item produced in one run.

    Attributes:
        item_id: The dataset item's id.
        output: What the target returned, or `None` if it failed.
        error: Why it failed, or `None`. An outcome may have neither an output nor an error
            only if the target legitimately returns nothing.
        latency_ms: Wall-clock milliseconds for the call.
        prompt_tokens: Tokens sent, where the target reports them.
        completion_tokens: Tokens received, where the target reports them.
        cost_usd: Cost of the call, where it is known.
        scores: Each scorer's result for this item.
    """

    item_id: str
    output: JsonValue = None
    error: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    scores: Mapping[str, Score] = field(default_factory=_no_scores)


@dataclass(frozen=True)
class Pairing:
    """Two runs' scores for one scorer, lined up by item.

    Attributes:
        scorer: Which scorer these values come from.
        kind: Binary or continuous, which decides the paired test.
        item_ids: Items present in both runs, in baseline order.
        baseline: Baseline values, aligned with `item_ids`.
        candidate: Candidate values, aligned with `item_ids`.
        only_in_baseline: Items the candidate run has no score for.
        only_in_candidate: Items the baseline run has no score for.
    """

    scorer: str
    kind: ScoreKind
    item_ids: tuple[str, ...]
    baseline: tuple[float, ...]
    candidate: tuple[float, ...]
    only_in_baseline: tuple[str, ...] = ()
    only_in_candidate: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """Whether every item scored in either run was scored in both.

        An incomplete pairing still compares what it can, but the report must say so: a
        comparison over fewer items than the slice it names is a narrower claim than it looks.
        """
        return not self.only_in_baseline and not self.only_in_candidate

    def __len__(self) -> int:
        """Number of paired items."""
        return len(self.item_ids)


class RunStore:
    """A SQLite file holding runs, outcomes and scores.

    Use as a context manager so the connection closes even when a run fails part way:

        with RunStore.open(Path("runs.sqlite")) as store:
            ...
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Wrap an open connection. Prefer `RunStore.open`."""
        self._connection = connection

    @classmethod
    def open(cls, path: Path | str) -> Self:
        """Open or create a store.

        Args:
            path: File to open. `:memory:` works for tests.

        Returns:
            The store, with its schema present.

        Raises:
            StoreError: If the file cannot be opened, or was written by a newer schema than
                this version understands.
        """
        try:
            connection = sqlite3.connect(path, isolation_level=None)
        except sqlite3.Error as error:
            raise StoreError(f"cannot open run store at {path}: {error}") from error

        connection.row_factory = sqlite3.Row
        # Off by default in SQLite, so a dangling score row would otherwise be allowed.
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA)

        store = cls(connection)
        store._check_schema_version()
        return store

    def _check_schema_version(self) -> None:
        """Record this version, or refuse a file written by a newer one.

        Raises:
            StoreError: If the stored version is newer than this code understands. Reading it
                anyway would mean guessing at columns, and a run record that is guessed at is
                worse than no run record.
        """
        row = self._connection.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
            return
        found = int(row["version"])
        if found > SCHEMA_VERSION:
            raise StoreError(
                f"run store was written with schema version {found}, but this build "
                f"understands {SCHEMA_VERSION}. Upgrade rather than reading it as if it were older."
            )

    def close(self) -> None:
        """Close the underlying connection."""
        self._connection.close()

    def __enter__(self) -> Self:
        """Enter the context, returning the store."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection on the way out, successful or not."""
        del exc_type, exc, traceback
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction, rolling back if it raises."""
        self._connection.execute("BEGIN")
        try:
            yield self._connection
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    # -- writing ----------------------------------------------------------- #

    def start_run(
        self,
        *,
        dataset_name: str,
        dataset_version: str,
        dataset_sha256: str,
        target: str,
        run_id: str | None = None,
        started_at: datetime | None = None,
        notes: str | None = None,
    ) -> RunRecord:
        """Open a new run and return its record.

        Args:
            dataset_name: From the dataset manifest.
            dataset_version: From the dataset manifest.
            dataset_sha256: The dataset's content hash at this moment.
            target: What is being run.
            run_id: Identifier, generated when omitted.
            started_at: Start time, defaulting to now in UTC.
            notes: Optional free text.

        Returns:
            The new run's record.

        Raises:
            StoreError: If the run id is already present.
        """
        record = RunRecord(
            run_id=run_id or uuid.uuid4().hex,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_sha256=dataset_sha256,
            target=target,
            started_at=started_at or datetime.now(UTC),
            notes=notes,
        )
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO runs (run_id, dataset_name, dataset_version, dataset_sha256,"
                    " target, started_at, finished_at, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.run_id,
                        record.dataset_name,
                        record.dataset_version,
                        record.dataset_sha256,
                        record.target,
                        record.started_at.isoformat(),
                        None,
                        record.notes,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise StoreError(f"run {record.run_id!r} already exists") from error
        return record

    def record(self, run_id: str, outcome: ItemOutcome) -> None:
        """Store one item's outcome and its scores.

        The outcome and every score go in under one transaction, so a crash cannot leave an
        item that was run but has no scores, which would later read as a scorer failure.

        Args:
            run_id: The run this belongs to.
            outcome: What the item produced.

        Raises:
            StoreError: If the run does not exist, or the item is already recorded for it.
        """
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO outcomes (run_id, item_id, output_json, error, latency_ms,"
                    " prompt_tokens, completion_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        outcome.item_id,
                        json.dumps(outcome.output, ensure_ascii=False),
                        outcome.error,
                        outcome.latency_ms,
                        outcome.prompt_tokens,
                        outcome.completion_tokens,
                        outcome.cost_usd,
                    ),
                )
                connection.executemany(
                    "INSERT INTO scores (run_id, item_id, scorer, value, kind, detail_json)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            run_id,
                            outcome.item_id,
                            name,
                            score.value,
                            score.kind.value,
                            json.dumps(dict(score.detail), ensure_ascii=False),
                        )
                        for name, score in outcome.scores.items()
                    ],
                )
        except sqlite3.IntegrityError as error:
            raise StoreError(
                f"cannot record item {outcome.item_id!r} for run {run_id!r}: {error}"
            ) from error

    def finish_run(self, run_id: str, finished_at: datetime | None = None) -> None:
        """Mark a run complete.

        A run with no finish time is one that died part way, and the gate must be able to
        tell that from a run that completed, rather than comparing against a partial result.

        Raises:
            StoreError: If the run does not exist.
        """
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?",
                ((finished_at or datetime.now(UTC)).isoformat(), run_id),
            )
            if cursor.rowcount == 0:
                raise StoreError(f"no such run: {run_id!r}")

    # -- reading ----------------------------------------------------------- #

    def get_run(self, run_id: str) -> RunRecord:
        """Fetch one run's record.

        Raises:
            StoreError: If there is no such run.
        """
        row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise StoreError(f"no such run: {run_id!r}")
        return _record_from_row(row)

    def list_runs(
        self, *, dataset_name: str | None = None, limit: int = 50
    ) -> tuple[RunRecord, ...]:
        """List runs, most recently started first.

        Args:
            dataset_name: Restrict to one dataset, or `None` for all.
            limit: Maximum number to return.

        Returns:
            The matching run records.
        """
        if dataset_name is None:
            rows = self._connection.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, run_id DESC LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM runs WHERE dataset_name = ? ORDER BY started_at DESC, run_id DESC"
                " LIMIT ?",
                (dataset_name, limit),
            ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    def outcomes(self, run_id: str) -> tuple[ItemOutcome, ...]:
        """Fetch every outcome for a run, with its scores, in item order."""
        rows = self._connection.execute(
            "SELECT * FROM outcomes WHERE run_id = ? ORDER BY item_id", (run_id,)
        ).fetchall()
        score_rows = self._connection.execute(
            "SELECT * FROM scores WHERE run_id = ?", (run_id,)
        ).fetchall()

        by_item: dict[str, dict[str, Score]] = {}
        for row in score_rows:
            by_item.setdefault(str(row["item_id"]), {})[str(row["scorer"])] = Score(
                value=float(row["value"]),
                kind=ScoreKind(str(row["kind"])),
                detail=json.loads(str(row["detail_json"])),
            )

        return tuple(
            ItemOutcome(
                item_id=str(row["item_id"]),
                output=json.loads(str(row["output_json"])),
                error=row["error"],
                latency_ms=row["latency_ms"],
                prompt_tokens=row["prompt_tokens"],
                completion_tokens=row["completion_tokens"],
                cost_usd=row["cost_usd"],
                scores=by_item.get(str(row["item_id"]), {}),
            )
            for row in rows
        )

    def pair(
        self,
        baseline_run: str,
        candidate_run: str,
        scorer: str,
        *,
        items: Sequence[str] | None = None,
    ) -> Pairing:
        """Line up two runs' scores for one scorer, by item.

        Args:
            baseline_run: The run to compare against.
            candidate_run: The run under test.
            scorer: Which scorer's values to pair.
            items: Restrict to these item ids, for example one slice's. `None` pairs
                everything both runs scored.

        Returns:
            The pairing, naming any items present in only one run rather than dropping them.

        Raises:
            StoreError: If either run is unknown, if the scorer produced nothing in either
                run, or if the two runs disagree about whether the scorer is binary.
        """
        self.get_run(baseline_run)
        self.get_run(candidate_run)

        left = self._scores(baseline_run, scorer)
        right = self._scores(candidate_run, scorer)
        if not left or not right:
            empty = baseline_run if not left else candidate_run
            raise StoreError(f"run {empty!r} has no scores for scorer {scorer!r}")

        left_kind, right_kind = self._kind(baseline_run, scorer), self._kind(candidate_run, scorer)
        if left_kind is not right_kind:
            raise StoreError(
                f"scorer {scorer!r} is {left_kind.value} in run {baseline_run!r} but "
                f"{right_kind.value} in run {candidate_run!r}; these are not comparable"
            )

        wanted = set(items) if items is not None else set(left) | set(right)
        shared = [item for item in left if item in right and item in wanted]

        return Pairing(
            scorer=scorer,
            kind=left_kind,
            item_ids=tuple(shared),
            baseline=tuple(left[item] for item in shared),
            candidate=tuple(right[item] for item in shared),
            only_in_baseline=tuple(sorted((set(left) & wanted) - set(right))),
            only_in_candidate=tuple(sorted((set(right) & wanted) - set(left))),
        )

    def _scores(self, run_id: str, scorer: str) -> dict[str, float]:
        """Every value this scorer produced in a run, keyed by item, in item order."""
        rows = self._connection.execute(
            "SELECT item_id, value FROM scores WHERE run_id = ? AND scorer = ? ORDER BY item_id",
            (run_id, scorer),
        ).fetchall()
        return {str(row["item_id"]): float(row["value"]) for row in rows}

    def _kind(self, run_id: str, scorer: str) -> ScoreKind:
        """Whether a scorer was binary or continuous in a given run.

        Raises:
            StoreError: If the run holds both kinds for one scorer, which means two different
                scorers were recorded under one name.
        """
        rows = self._connection.execute(
            "SELECT DISTINCT kind FROM scores WHERE run_id = ? AND scorer = ?", (run_id, scorer)
        ).fetchall()
        kinds = {ScoreKind(str(row["kind"])) for row in rows}
        if len(kinds) != 1:
            raise StoreError(
                f"scorer {scorer!r} in run {run_id!r} recorded {len(kinds)} different kinds; "
                "two different scorers appear to share one name"
            )
        return kinds.pop()


def _record_from_row(row: sqlite3.Row) -> RunRecord:
    """Rebuild a run record from a database row."""
    finished = row["finished_at"]
    return RunRecord(
        run_id=str(row["run_id"]),
        dataset_name=str(row["dataset_name"]),
        dataset_version=str(row["dataset_version"]),
        dataset_sha256=str(row["dataset_sha256"]),
        target=str(row["target"]),
        started_at=datetime.fromisoformat(str(row["started_at"])),
        finished_at=datetime.fromisoformat(str(finished)) if finished else None,
        notes=row["notes"],
    )
