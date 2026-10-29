"""Deterministic scorers: the checks that run before any model is asked for an opinion.

Owning artifact: `product context/00_PROJECT_BRIEF.md`, section 3.

A scorer turns one item's output into one number. Each declares whether it is binary or
continuous, because that choice decides which paired test the gate runs: binary outcomes get
the exact conditional interval, continuous ones get the bootstrap. Getting it wrong would not
raise anything -- it would quietly produce the wrong interval -- so the scorer declares it
rather than the caller guessing.

Deterministic checks run first and an LLM judge second. A judge that disagrees with a regular
expression about whether two strings are equal is an expensive way to be wrong, and every item
a deterministic scorer can settle is an item the judge never needs to see.

Deliberate boundary: scorers compare an output with an expectation. They do not call models,
read datasets or know which slice an item is in. Anything a scorer cannot settle raises rather
than returning zero, because a scorer that cannot run and a scorer that ran and found a wrong
answer mean entirely different things.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, cast, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import JsonValue

__all__ = [
    "CustomScorer",
    "EditSimilarity",
    "ExactMatch",
    "JsonSchemaMatch",
    "RegexMatch",
    "Score",
    "ScoreKind",
    "Scorer",
    "ScorerError",
    "SetF1",
    "levenshtein",
    "score_all",
]


class ScorerError(Exception):
    """A scorer could not be applied to an item.

    Distinct from a low score on purpose. "This output is wrong" and "this scorer cannot read
    this output" are different facts, and collapsing the second into a zero would let a
    misconfigured scorer report a slice as uniformly failing.
    """


def _issue_path(issue: _SchemaIssue) -> str:
    """Render a schema failure's location as a slash-separated path."""
    return "/".join(str(part) for part in issue.path)


def _no_detail() -> Mapping[str, JsonValue]:
    """Empty detail, as a typed factory.

    A bare `dict` as the default factory types the field as a mapping of unknowns, which
    strict type checking rejects for good reason: it would hide the shape of everything a
    scorer reports.
    """
    return {}


class _SchemaIssue(Protocol):
    """The part of a jsonschema validation error this module relies on."""

    @property
    def message(self) -> str:
        """Human-readable description of what failed."""
        ...

    @property
    def path(self) -> Sequence[object]:
        """Location of the failure within the instance."""
        ...


class _SchemaValidator(Protocol):
    """The part of a jsonschema validator this module relies on.

    Declared here, and the real validator narrowed to it once, because jsonschema ships
    incomplete types. Naming the two members actually used is better than a blanket
    suppression: it says exactly what this module depends on, so an upgrade that changes
    either one fails here rather than somewhere further downstream.
    """

    def iter_errors(self, instance: object) -> Iterator[_SchemaIssue]:
        """Yield every way `instance` fails the schema."""
        ...


class ScoreKind(StrEnum):
    """Whether a scorer's values are two-valued or continuous.

    This selects the paired test downstream, so it is part of the scorer's contract rather
    than something the comparison infers from the values it happens to see. A binary scorer
    that returned 1.0 for every item in a slice would otherwise look continuous.
    """

    BINARY = "binary"
    """Values are exactly 0.0 or 1.0. Compared with the exact conditional interval."""

    CONTINUOUS = "continuous"
    """Values lie anywhere in [0, 1]. Compared with the BCa bootstrap."""


@dataclass(frozen=True)
class Score:
    """One item's result under one scorer.

    Attributes:
        value: The score, in [0, 1], where 1 is best.
        kind: Whether this scorer is binary or continuous.
        detail: Scorer-specific evidence, kept so a report can show why an item scored as it
            did without re-running anything.
    """

    value: float
    kind: ScoreKind
    detail: Mapping[str, JsonValue] = field(default_factory=_no_detail)

    def __post_init__(self) -> None:
        """Refuse a value outside [0, 1], or a non-integral binary value.

        Raises:
            ValueError: If the value is out of range, or a binary scorer produced something
                between 0 and 1, which would make the exact conditional interval meaningless.
        """
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.value}")
        if self.kind is ScoreKind.BINARY and self.value not in (0.0, 1.0):
            raise ValueError(f"a binary score must be 0.0 or 1.0, got {self.value}")


@runtime_checkable
class Scorer(Protocol):
    """What every scorer provides.

    A protocol rather than a base class, so a user's own callable object is a scorer without
    importing anything from here.
    """

    @property
    def name(self) -> str:
        """Short identifier, used as a column heading in reports and run records."""
        ...

    @property
    def kind(self) -> ScoreKind:
        """Whether this scorer is binary or continuous."""
        ...

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Score one item's output against its expectation."""
        ...


def _require_expected(expected: JsonValue | None, scorer: str) -> JsonValue:
    """Return the expectation, or refuse to guess at one.

    Raises:
        ScorerError: If the item has no expectation. An item with nothing to compare against
            is one a judge must handle, and scoring it as zero would make a dataset of
            judge-only items look like a total failure.
    """
    if expected is None:
        raise ScorerError(f"{scorer}: item has no expected value to compare against")
    return expected


def _as_text(value: JsonValue, scorer: str, role: str) -> str:
    """Narrow a JSON value to a string, or refuse.

    Raises:
        ScorerError: If the value is not a string.
    """
    if not isinstance(value, str):
        raise ScorerError(f"{scorer}: {role} must be a string, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class ExactMatch:
    """Binary: the output equals the expectation.

    Attributes:
        name: Identifier for reports.
        case_sensitive: Whether case counts. Off by default, because for labels and short
            answers a capital letter is almost never the difference a reader cares about.
        strip: Whether to ignore leading and trailing whitespace. On by default, for the same
            reason: a trailing newline from a model is not a wrong answer.
    """

    name: str = "exact_match"
    case_sensitive: bool = False
    strip: bool = True

    @property
    def kind(self) -> ScoreKind:
        """Exact match is binary."""
        return ScoreKind.BINARY

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Compare output and expectation after the configured normalisation.

        Non-string values are compared by their canonical JSON, so two equal objects with
        differently ordered keys match.

        Raises:
            ScorerError: If the item has no expectation.
        """
        target = _require_expected(expected, self.name)
        left, right = _canonical(output), _canonical(target)
        if self.strip:
            left, right = left.strip(), right.strip()
        if not self.case_sensitive:
            left, right = left.casefold(), right.casefold()
        matched = left == right
        return Score(
            value=1.0 if matched else 0.0,
            kind=ScoreKind.BINARY,
            detail={"output": left, "expected": right},
        )


def _canonical(value: JsonValue) -> str:
    """Render a JSON value as a stable string for comparison."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class RegexMatch:
    """Binary: the output matches a pattern.

    The pattern is fixed when the scorer is built, not taken from the item, so one
    misconfigured dataset row cannot change what the whole slice is being asked.

    Attributes:
        pattern: The regular expression.
        name: Identifier for reports.
        full: Whether the pattern must consume the whole output. On by default: a pattern that
            merely appears somewhere in a long answer usually is not the check that was meant.
        ignore_case: Whether to match case-insensitively.
    """

    pattern: str
    name: str = "regex_match"
    full: bool = True
    ignore_case: bool = False

    def __post_init__(self) -> None:
        """Compile the pattern now, so a bad one fails at configuration time.

        Raises:
            ScorerError: If the pattern does not compile. Finding that out item by item
                during a run would waste the run.
        """
        try:
            re.compile(self.pattern)
        except re.error as error:
            raise ScorerError(f"{self.name}: pattern does not compile: {error}") from error

    @property
    def kind(self) -> ScoreKind:
        """Regex match is binary."""
        return ScoreKind.BINARY

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Match the output against the pattern. The expectation is not used.

        Raises:
            ScorerError: If the output is not a string.
        """
        del expected  # A regex scorer carries its own expectation in the pattern.
        text = _as_text(output, self.name, "output")
        flags = re.IGNORECASE if self.ignore_case else re.NOFLAG
        compiled = re.compile(self.pattern, flags)
        found = compiled.fullmatch(text) if self.full else compiled.search(text)
        return Score(
            value=1.0 if found else 0.0,
            kind=ScoreKind.BINARY,
            detail={"pattern": self.pattern, "full": self.full},
        )


@dataclass(frozen=True)
class JsonSchemaMatch:
    """Binary: the output is JSON valid against a schema.

    Structure is the cheapest thing to check about a model that is supposed to emit JSON, and
    the failure a prompt change breaks first.

    Attributes:
        schema: A JSON Schema (2020-12).
        name: Identifier for reports.
        parse_strings: Whether a string output should be parsed as JSON first. On by default,
            because a model emitting JSON emits it as text.
    """

    schema: Mapping[str, JsonValue]
    name: str = "json_schema"
    parse_strings: bool = True

    def __post_init__(self) -> None:
        """Validate the schema itself at configuration time.

        Raises:
            ScorerError: If the schema is not a legal JSON Schema. A broken schema would
                otherwise fail every item and look like a total regression.
        """
        try:
            Draft202012Validator.check_schema(dict(self.schema))
        except SchemaError as error:
            raise ScorerError(f"{self.name}: schema is not valid: {error.message}") from error

    @property
    def kind(self) -> ScoreKind:
        """Schema validity is binary."""
        return ScoreKind.BINARY

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Check the output against the schema.

        Unparseable text is a score of zero rather than an error: a model emitting prose where
        JSON was asked for is exactly the regression this scorer exists to catch.
        """
        del expected  # The schema is the expectation.
        candidate: JsonValue = output
        if self.parse_strings and isinstance(output, str):
            try:
                candidate = json.loads(output)
            except json.JSONDecodeError as error:
                return Score(
                    value=0.0,
                    kind=ScoreKind.BINARY,
                    detail={"error": f"output is not JSON: {error.msg}"},
                )

        validator = cast("_SchemaValidator", Draft202012Validator(dict(self.schema)))
        # Sorted by the rendered path, not by the raw path: that is a sequence of mixed
        # strings and integers, and comparing two of them raises rather than ordering.
        errors = sorted(validator.iter_errors(candidate), key=_issue_path)
        if not errors:
            return Score(value=1.0, kind=ScoreKind.BINARY, detail={})
        first = errors[0]
        return Score(
            value=0.0,
            kind=ScoreKind.BINARY,
            detail={
                "error": first.message,
                "path": _issue_path(first),
                "error_count": len(errors),
            },
        )


@dataclass(frozen=True)
class SetF1:
    """Continuous: F1 between the output's items and the expectation's, as sets.

    For extraction tasks, where the answer is a collection and partial credit is the honest
    report. Order and repeats are ignored, which is what makes it a set measure; if order
    matters, this is the wrong scorer.

    Attributes:
        name: Identifier for reports.
        case_sensitive: Whether string members compare case-sensitively.
    """

    name: str = "set_f1"
    case_sensitive: bool = False

    @property
    def kind(self) -> ScoreKind:
        """F1 is continuous."""
        return ScoreKind.CONTINUOUS

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Compute F1 between the two collections.

        Two empty collections score 1.0. Finding nothing where nothing was expected is a
        correct answer, and scoring it zero would punish the model for being right.

        Raises:
            ScorerError: If either side is not a list, or has no expectation.
        """
        target = _require_expected(expected, self.name)
        produced = self._members(output, "output")
        wanted = self._members(target, "expected")

        if not produced and not wanted:
            return Score(value=1.0, kind=ScoreKind.CONTINUOUS, detail={"note": "both empty"})
        overlap = len(produced & wanted)
        if overlap == 0:
            return Score(
                value=0.0,
                kind=ScoreKind.CONTINUOUS,
                detail={"true_positives": 0, "predicted": len(produced), "actual": len(wanted)},
            )

        precision = overlap / len(produced)
        recall = overlap / len(wanted)
        return Score(
            value=2 * precision * recall / (precision + recall),
            kind=ScoreKind.CONTINUOUS,
            detail={
                "true_positives": overlap,
                "predicted": len(produced),
                "actual": len(wanted),
                "precision": precision,
                "recall": recall,
            },
        )

    def _members(self, value: JsonValue, role: str) -> frozenset[str]:
        """Normalise one side into a set of comparable strings.

        Raises:
            ScorerError: If the value is not a list.
        """
        if not isinstance(value, list):
            raise ScorerError(f"{self.name}: {role} must be a list, got {type(value).__name__}")
        members = (_canonical(member) for member in value)
        if self.case_sensitive:
            return frozenset(members)
        return frozenset(member.casefold() for member in members)


def levenshtein(left: str, right: str) -> int:
    """Edit distance between two strings: insertions, deletions and substitutions.

    Written out rather than taken from a dependency. It is twenty lines, it is exactly
    specified, and the scorer's number is easier to defend when the thing producing it is in
    the repository.

    Two rows rather than a full matrix, since only the distance is wanted and not the path.

    Args:
        left: First string.
        right: Second string.

    Returns:
        The minimum number of single-character edits turning one into the other.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (left_char != right_char),  # substitution
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class EditSimilarity:
    """Continuous: 1 minus the normalised edit distance between output and expectation.

    For short free text where "nearly right" is worth distinguishing from "nothing like it".
    On long passages it mostly measures length, so prefer a judge there.

    Attributes:
        name: Identifier for reports.
        case_sensitive: Whether case counts.
        strip: Whether to ignore surrounding whitespace.
    """

    name: str = "edit_similarity"
    case_sensitive: bool = False
    strip: bool = True

    @property
    def kind(self) -> ScoreKind:
        """Edit similarity is continuous."""
        return ScoreKind.CONTINUOUS

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Score similarity in [0, 1], where 1 means identical after normalisation.

        Raises:
            ScorerError: If either side is not a string, or there is no expectation.
        """
        target = _require_expected(expected, self.name)
        left = _as_text(output, self.name, "output")
        right = _as_text(target, self.name, "expected")
        if self.strip:
            left, right = left.strip(), right.strip()
        if not self.case_sensitive:
            left, right = left.casefold(), right.casefold()

        longest = max(len(left), len(right))
        if longest == 0:
            # Two empty strings are identical; dividing by zero to discover that would be worse.
            return Score(value=1.0, kind=ScoreKind.CONTINUOUS, detail={"distance": 0})
        distance = levenshtein(left, right)
        return Score(
            value=1.0 - distance / longest,
            kind=ScoreKind.CONTINUOUS,
            detail={"distance": distance, "length": longest},
        )


@dataclass(frozen=True)
class CustomScorer:
    """Wraps a plain function so a project can score whatever it needs to.

    The escape hatch, kept deliberately small. The wrapper validates what comes back, because
    a custom scorer returning 1.4 or a negative number would otherwise flow into the interval
    arithmetic and produce a confident nonsense verdict.

    Attributes:
        name: Identifier for reports.
        kind: Whether the function returns binary or continuous values.
        function: Takes `(output, expected)` and returns a float in [0, 1].
    """

    name: str
    kind: ScoreKind
    function: Callable[[JsonValue, JsonValue | None], float]

    def score(self, output: JsonValue, expected: JsonValue | None) -> Score:
        """Call the function and check what it returned.

        Raises:
            ScorerError: If the function raises, or returns something outside [0, 1], or
                returns a fractional value while declaring itself binary.
        """
        try:
            value = self.function(output, expected)
        except ScorerError:
            raise
        except Exception as error:
            raise ScorerError(
                f"{self.name}: scorer raised {type(error).__name__}: {error}"
            ) from error

        try:
            return Score(value=float(value), kind=self.kind, detail={})
        except (TypeError, ValueError) as error:
            raise ScorerError(f"{self.name}: returned an unusable score: {error}") from error


def score_all(
    scorers: Sequence[Scorer], output: JsonValue, expected: JsonValue | None
) -> dict[str, Score]:
    """Run every scorer over one item.

    Args:
        scorers: The scorers to apply. Names must be unique.
        output: What the feature produced.
        expected: What the item expected, if anything.

    Returns:
        A mapping from scorer name to its score.

    Raises:
        ScorerError: If two scorers share a name, which would silently drop one of them, or
            if any scorer cannot be applied.
    """
    results: dict[str, Score] = {}
    for scorer in scorers:
        if scorer.name in results:
            raise ScorerError(f"two scorers are both named {scorer.name!r}")
        results[scorer.name] = scorer.score(output, expected)
    return results
