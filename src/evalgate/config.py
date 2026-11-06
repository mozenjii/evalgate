"""`evalgate.toml`: everything about a check that a person decides, kept out of code.

Which dataset, which scorers, which slices can fail the build and which are only watched, and
how much worse each slice is allowed to get. The last of those has no default. A tolerance is a
judgement about one slice of one product, and a default would be a number nobody chose.

Everything is checked against the dataset when the file is loaded, so a typo in a slice name
fails at the start of a run rather than producing a report with a slice quietly missing.

Deliberate boundary: this module reads and validates configuration and builds the scorers it
names. It does not run, score or compare anything.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from evalgate.dataset import Dataset, DatasetError, load_dataset
from evalgate.scorers import (
    EditSimilarity,
    ExactMatch,
    JsonSchemaMatch,
    RegexMatch,
    Scorer,
    ScorerError,
    SetF1,
)

__all__ = [
    "Config",
    "ConfigError",
    "Rule",
    "SliceRef",
    "TargetSpec",
    "Tier",
    "load_config",
]


class ConfigError(Exception):
    """The configuration file is missing, malformed, or disagrees with its dataset."""


class Tier(StrEnum):
    """Whether a slice can fail the build."""

    GATE = "gate"
    """Can fail the build."""

    WATCH = "watch"
    """Reported, never fails the build."""


@dataclass(frozen=True, order=True)
class SliceRef:
    """One slice: a dimension and one of its values.

    Attributes:
        dimension: The slice dimension, for example `intent`.
        value: One of its declared values, for example `billing`.
    """

    dimension: str
    value: str

    def __str__(self) -> str:
        """Render as `dimension=value`, the form the config file uses."""
        return f"{self.dimension}={self.value}"


@dataclass(frozen=True)
class Rule:
    """A set of slices compared on one scorer, with their tolerances.

    Attributes:
        tier: Whether these slices can fail the build.
        scorer: The scorer's name.
        slices: The slices this rule covers, in the dataset's declared order.
        tolerance: How much worse a slice may get, on the scorer's own 0-to-1 scale.
        overrides: Per-slice tolerances that replace `tolerance`.
    """

    tier: Tier
    scorer: str
    slices: tuple[SliceRef, ...]
    tolerance: float
    overrides: Mapping[SliceRef, float]

    def tolerance_for(self, slice_ref: SliceRef) -> float:
        """The tolerance that applies to one slice."""
        return self.overrides.get(slice_ref, self.tolerance)


@dataclass(frozen=True)
class TargetSpec:
    """What `evalgate run` calls.

    Attributes:
        reference: A `module:function` reference.
        pythonpath: Directories to search for the module, already resolved.
    """

    reference: str
    pythonpath: tuple[Path, ...]


@dataclass(frozen=True)
class Config:
    """A loaded, validated configuration.

    Attributes:
        source: The file it came from.
        dataset: The dataset, already loaded and hash-checked.
        scorers: Scorers by name, in the order the file declares them.
        rules: Gate rules first, then watch rules, each in file order.
        confidence: Nominal confidence level for every interval.
        on_inconclusive: Whether a gate slice that cannot answer fails the build.
        target: What `evalgate run` calls, if the file says.
    """

    source: Path
    dataset: Dataset
    scorers: Mapping[str, Scorer]
    rules: tuple[Rule, ...]
    confidence: float
    on_inconclusive: Literal["warn", "fail"]
    target: TargetSpec | None


# --------------------------------------------------------------------------- the file's shape


class _Strict(BaseModel):
    """Base for every table: frozen, and unknown keys refused, so a typo is an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _ExactMatch(_Strict):
    type: Literal["exact_match"]
    name: str | None = None
    case_sensitive: bool = False
    strip: bool = True


class _Regex(_Strict):
    type: Literal["regex"]
    name: str | None = None
    pattern: str
    full: bool = True
    ignore_case: bool = False


class _JsonSchema(_Strict):
    type: Literal["json_schema"]
    name: str | None = None
    schema_file: str


class _SetF1(_Strict):
    type: Literal["set_f1"]
    name: str | None = None
    case_sensitive: bool = False


class _EditSimilarity(_Strict):
    type: Literal["edit_similarity"]
    name: str | None = None
    case_sensitive: bool = False
    strip: bool = True


_ScorerTable = Annotated[
    _ExactMatch | _Regex | _JsonSchema | _SetF1 | _EditSimilarity, Field(discriminator="type")
]


class _Rule(_Strict):
    scorer: str
    slices: list[str] = Field(min_length=1)
    tolerance: float = Field(gt=0, le=1)
    tolerances: dict[str, Annotated[float, Field(gt=0, le=1)]] = Field(default_factory=dict)


class _Dataset(_Strict):
    manifest: str


class _Target(_Strict):
    callable: str = Field(pattern=r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
    pythonpath: list[str] = Field(default_factory=lambda: list[str]())


class _Policy(_Strict):
    confidence: float = Field(default=0.95, gt=0, lt=1)
    on_inconclusive: Literal["warn", "fail"] = "warn"


class _File(_Strict):
    dataset: _Dataset
    scorer: list[_ScorerTable] = Field(min_length=1)
    gate: list[_Rule] = Field(min_length=1)
    watch: list[_Rule] = Field(default_factory=lambda: list[_Rule]())
    policy: _Policy = Field(default_factory=_Policy)
    target: _Target | None = None


# --------------------------------------------------------------------------- loading


def load_config(path: Path) -> Config:
    """Load `evalgate.toml` and check it against its dataset.

    Paths inside the file are relative to the file itself, so a check behaves the same from
    any working directory -- which matters in CI, where the base and head commits are checked
    out in different places.

    Args:
        path: The configuration file.

    Returns:
        The validated configuration, with its dataset loaded and its scorers built.

    Raises:
        ConfigError: If the file cannot be read or parsed, a value is out of range, a scorer
            or slice it names does not exist, a tolerance override names a slice outside its
            rule, or one slice is claimed twice for the same scorer.
    """
    raw = _read(path)
    root = path.resolve().parent

    try:
        dataset = load_dataset(root / raw.dataset.manifest)
    except DatasetError as error:
        raise ConfigError(f"{path}: [dataset] {error}") from error

    scorers = _build_scorers(raw.scorer, root, path)
    rules = tuple(
        _resolve_rule(rule, tier, dataset, scorers, path, position)
        for tier, tables in ((Tier.GATE, raw.gate), (Tier.WATCH, raw.watch))
        for position, rule in enumerate(tables, start=1)
    )
    _refuse_overlaps(rules, path)

    target = None
    if raw.target is not None:
        target = TargetSpec(
            reference=raw.target.callable,
            pythonpath=tuple((root / entry).resolve() for entry in raw.target.pythonpath),
        )

    return Config(
        source=path,
        dataset=dataset,
        scorers=scorers,
        rules=rules,
        confidence=raw.policy.confidence,
        on_inconclusive=raw.policy.on_inconclusive,
        target=target,
    )


def _read(path: Path) -> _File:
    """Parse the TOML and validate its shape."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"{path}: cannot read config: {error}") from error
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from error
    try:
        return _File.model_validate(data)
    except ValidationError as error:
        details = "\n".join(
            f"  {'.'.join(str(part) for part in detail['loc']) or '(root)'}: {detail['msg']}"
            for detail in error.errors()
        )
        raise ConfigError(f"{path}: invalid config:\n{details}") from error


def _build_scorers(tables: list[_ScorerTable], root: Path, path: Path) -> dict[str, Scorer]:
    """Build each scorer the file names, refusing two with the same name."""
    scorers: dict[str, Scorer] = {}
    for table in tables:
        try:
            scorer = _build_scorer(table, root)
        except ScorerError as error:
            raise ConfigError(f"{path}: [[scorer]] {error}") from error
        if scorer.name in scorers:
            raise ConfigError(
                f"{path}: two scorers are named {scorer.name!r}; give one of them a `name`"
            )
        scorers[scorer.name] = scorer
    return scorers


def _build_scorer(table: _ScorerTable, root: Path) -> Scorer:
    """Turn one `[[scorer]]` table into a scorer."""
    name = table.name or table.type
    match table:
        case _ExactMatch():
            return ExactMatch(name=name, case_sensitive=table.case_sensitive, strip=table.strip)
        case _Regex():
            return RegexMatch(
                pattern=table.pattern, name=name, full=table.full, ignore_case=table.ignore_case
            )
        case _JsonSchema():
            schema_path = root / table.schema_file
            try:
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ScorerError(f"{name}: cannot load schema {schema_path}: {error}") from error
            if not isinstance(schema, dict):
                raise ScorerError(f"{name}: schema {schema_path} is not a JSON object")
            return JsonSchemaMatch(schema=cast("dict[str, JsonValue]", schema), name=name)
        case _SetF1():
            return SetF1(name=name, case_sensitive=table.case_sensitive)
        case _EditSimilarity():
            return EditSimilarity(name=name, case_sensitive=table.case_sensitive, strip=table.strip)


def _resolve_rule(
    rule: _Rule,
    tier: Tier,
    dataset: Dataset,
    scorers: Mapping[str, Scorer],
    path: Path,
    position: int,
) -> Rule:
    """Check one rule against the scorers and the dataset, and expand its slice selectors."""
    where = f"{path}: [[{tier.value}]] #{position}"
    if rule.scorer not in scorers:
        known = ", ".join(scorers)
        raise ConfigError(f"{where}: no scorer named {rule.scorer!r}; declared: {known}")

    selected: list[SliceRef] = []
    for selector in rule.slices:
        for slice_ref in _expand(selector, dataset, where):
            if slice_ref not in selected:
                selected.append(slice_ref)

    overrides: dict[SliceRef, float] = {}
    for key, tolerance in rule.tolerances.items():
        matches = _expand(key, dataset, where)
        if len(matches) != 1:
            raise ConfigError(f"{where}: a tolerance override must name one slice, got {key!r}")
        if matches[0] not in selected:
            raise ConfigError(
                f"{where}: tolerance override {key!r} is not one of this rule's slices"
            )
        overrides[matches[0]] = tolerance

    return Rule(
        tier=tier,
        scorer=rule.scorer,
        slices=tuple(selected),
        tolerance=rule.tolerance,
        overrides=overrides,
    )


def _expand(selector: str, dataset: Dataset, where: str) -> list[SliceRef]:
    """Expand `dimension=value` or `dimension=*` into slices the dataset declares.

    Raises:
        ConfigError: If the selector is malformed, or names a dimension or value the dataset
            does not declare. Expanding a typo to nothing would drop a slice from the report
            with no sign that it was ever meant to be there.
    """
    dimension, separator, value = selector.partition("=")
    if not separator or not dimension or not value:
        raise ConfigError(f"{where}: slice {selector!r} must look like 'dimension=value'")
    declared = dataset.manifest.slice_dimensions.get(dimension)
    if declared is None:
        known = ", ".join(dataset.manifest.slice_dimensions)
        raise ConfigError(f"{where}: no slice dimension {dimension!r}; declared: {known}")
    if value == "*":
        return [SliceRef(dimension, each) for each in declared]
    if value not in declared:
        raise ConfigError(
            f"{where}: {value!r} is not a declared value of {dimension!r}; "
            f"declared: {', '.join(declared)}"
        )
    return [SliceRef(dimension, value)]


def _refuse_overlaps(rules: tuple[Rule, ...], path: Path) -> None:
    """Refuse one slice claimed twice for the same scorer.

    Two rules for the same slice and scorer would give it two tolerances, or make it both a
    gate and a watch slice, and the report would have to pick one silently.
    """
    claimed: dict[tuple[str, SliceRef], Tier] = {}
    for rule in rules:
        for slice_ref in rule.slices:
            key = (rule.scorer, slice_ref)
            if key in claimed:
                raise ConfigError(
                    f"{path}: slice {slice_ref} is covered twice for scorer {rule.scorer!r} "
                    f"(as {claimed[key].value} and as {rule.tier.value})"
                )
            claimed[key] = rule.tier
