"""evalgate.toml loads cleanly, and every mistake in it fails at load time, loudly."""

import json
from pathlib import Path
from typing import Any

import pytest

from evalgate.config import ConfigError, SliceRef, Tier, load_config
from evalgate.dataset import Item, content_hash
from evalgate.scorers import ScoreKind

BASE = """
[dataset]
manifest = "data/manifest.json"

[[scorer]]
type = "exact_match"

[[gate]]
scorer = "exact_match"
slices = ["intent=*"]
tolerance = 0.05
"""


def write_dataset(root: Path) -> None:
    items: list[dict[str, Any]] = [
        {
            "id": f"{intent}-{length}",
            "input": f"{intent} {length}",
            "expected": intent,
            "slices": {"intent": intent, "length": length},
            "provenance": "synthetic",
        }
        for intent in ("billing", "refunds")
        for length in ("short", "long")
    ]
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "items.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items), encoding="utf-8"
    )
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "name": "toy",
                "version": "1",
                "items_file": "items.jsonl",
                "content_sha256": content_hash([Item.model_validate(item) for item in items]),
                "slice_dimensions": {"intent": ["billing", "refunds"], "length": ["short", "long"]},
            }
        ),
        encoding="utf-8",
    )


def write_config(root: Path, body: str) -> Path:
    write_dataset(root)
    path = root / "evalgate.toml"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- the happy path


def test_a_minimal_config_loads(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path, BASE))

    assert config.dataset.manifest.name == "toy"
    assert list(config.scorers) == ["exact_match"]
    assert config.scorers["exact_match"].kind is ScoreKind.BINARY
    (rule,) = config.rules
    assert rule.tier is Tier.GATE
    assert rule.slices == (SliceRef("intent", "billing"), SliceRef("intent", "refunds"))
    assert config.confidence == 0.95
    assert config.on_inconclusive == "warn"
    assert config.target is None


def test_paths_are_relative_to_the_config_file_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In CI the base and head commits are checked out in different places.
    path = write_config(tmp_path / "project", BASE)
    monkeypatch.chdir(tmp_path)

    assert load_config(path).dataset.manifest.name == "toy"


def test_overrides_watch_rules_policy_and_target(tmp_path: Path) -> None:
    body = (
        BASE
        + """
[gate.tolerances]
"intent=billing" = 0.02

[[watch]]
scorer = "exact_match"
slices = ["length=*"]
tolerance = 0.1

[policy]
confidence = 0.9
on_inconclusive = "fail"

[target]
callable = "myapp.triage:classify"
pythonpath = ["src"]
"""
    )
    config = load_config(write_config(tmp_path, body))

    gate, watch = config.rules
    assert gate.tolerance_for(SliceRef("intent", "billing")) == 0.02
    assert gate.tolerance_for(SliceRef("intent", "refunds")) == 0.05
    assert watch.tier is Tier.WATCH
    assert config.confidence == 0.9
    assert config.on_inconclusive == "fail"
    assert config.target is not None
    assert config.target.reference == "myapp.triage:classify"
    assert config.target.pythonpath == ((tmp_path / "src").resolve(),)


def test_every_scorer_type_can_be_declared(tmp_path: Path) -> None:
    (tmp_path / "schema.json").write_text('{"type": "object"}', encoding="utf-8")
    body = """
[dataset]
manifest = "data/manifest.json"

[[scorer]]
type = "exact_match"
[[scorer]]
type = "regex"
pattern = "billing|refunds"
[[scorer]]
type = "json_schema"
schema_file = "schema.json"
[[scorer]]
type = "set_f1"
[[scorer]]
type = "edit_similarity"
name = "closeness"

[[gate]]
scorer = "closeness"
slices = ["intent=billing"]
tolerance = 0.05
"""
    config = load_config(write_config(tmp_path, body))

    assert list(config.scorers) == ["exact_match", "regex", "json_schema", "set_f1", "closeness"]
    assert config.scorers["closeness"].kind is ScoreKind.CONTINUOUS


# --------------------------------------------------------------------------- refusals


def test_a_missing_tolerance_is_refused_because_it_has_no_default(tmp_path: Path) -> None:
    # A default would be a number nobody chose.
    body = BASE.replace("tolerance = 0.05\n", "")

    with pytest.raises(ConfigError, match="tolerance"):
        load_config(write_config(tmp_path, body))


@pytest.mark.parametrize("tolerance", ["0", "-0.1", "1.5"])
def test_a_tolerance_outside_the_score_scale_is_refused(tmp_path: Path, tolerance: str) -> None:
    body = BASE.replace("tolerance = 0.05", f"tolerance = {tolerance}")

    with pytest.raises(ConfigError, match="tolerance"):
        load_config(write_config(tmp_path, body))


def test_a_config_without_a_gate_is_refused(tmp_path: Path) -> None:
    # A check that cannot fail is not a check.
    body = BASE.split("[[gate]]")[0]

    with pytest.raises(ConfigError, match="gate"):
        load_config(write_config(tmp_path, body))


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    body = BASE.replace("tolerance = 0.05", "tolerance = 0.05\ntolerence = 0.05")

    with pytest.raises(ConfigError, match="tolerence"):
        load_config(write_config(tmp_path, body))


def test_an_unknown_scorer_type_is_refused(tmp_path: Path) -> None:
    body = BASE.replace('type = "exact_match"', 'type = "vibes"')

    with pytest.raises(ConfigError, match="invalid config"):
        load_config(write_config(tmp_path, body))


def test_a_rule_naming_an_undeclared_scorer_is_refused(tmp_path: Path) -> None:
    body = BASE.replace('scorer = "exact_match"', 'scorer = "exact"')

    with pytest.raises(ConfigError, match="no scorer named 'exact'"):
        load_config(write_config(tmp_path, body))


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ("intent", "must look like"),
        ("tone=*", "no slice dimension 'tone'"),
        ("intent=chargebacks", "not a declared value"),
    ],
)
def test_a_slice_the_dataset_does_not_declare_is_refused(
    tmp_path: Path, selector: str, message: str
) -> None:
    # Expanding a typo to nothing would drop the slice from the report without a trace.
    body = BASE.replace('"intent=*"', f'"{selector}"')

    with pytest.raises(ConfigError, match=message):
        load_config(write_config(tmp_path, body))


def test_an_override_for_a_slice_outside_its_rule_is_refused(tmp_path: Path) -> None:
    body = BASE + '\n[gate.tolerances]\n"length=short" = 0.02\n'

    with pytest.raises(ConfigError, match="not one of this rule's slices"):
        load_config(write_config(tmp_path, body))


def test_an_override_must_name_one_slice(tmp_path: Path) -> None:
    body = BASE + '\n[gate.tolerances]\n"intent=*" = 0.02\n'

    with pytest.raises(ConfigError, match="must name one slice"):
        load_config(write_config(tmp_path, body))


def test_a_slice_claimed_twice_for_one_scorer_is_refused(tmp_path: Path) -> None:
    # It would have two tolerances, or be gate and watch at once, and the report would have to
    # pick one silently.
    body = (
        BASE + '\n[[watch]]\nscorer = "exact_match"\nslices = ["intent=billing"]\ntolerance = 0.1\n'
    )

    with pytest.raises(ConfigError, match="covered twice"):
        load_config(write_config(tmp_path, body))


def test_two_scorers_with_one_name_are_refused(tmp_path: Path) -> None:
    body = BASE.replace(
        '[[scorer]]\ntype = "exact_match"',
        '[[scorer]]\ntype = "exact_match"\n[[scorer]]\ntype = "exact_match"',
    )

    with pytest.raises(ConfigError, match="two scorers are named"):
        load_config(write_config(tmp_path, body))


def test_a_broken_regex_is_reported_against_the_config(tmp_path: Path) -> None:
    body = BASE.replace(
        'type = "exact_match"', 'type = "regex"\npattern = "(unclosed"\nname = "exact_match"'
    )

    with pytest.raises(ConfigError, match="does not compile"):
        load_config(write_config(tmp_path, body))


def test_a_missing_schema_file_is_reported(tmp_path: Path) -> None:
    body = BASE.replace(
        'type = "exact_match"',
        'type = "json_schema"\nschema_file = "nope.json"\nname = "exact_match"',
    )

    with pytest.raises(ConfigError, match="cannot load schema"):
        load_config(write_config(tmp_path, body))


def test_a_malformed_target_reference_is_refused(tmp_path: Path) -> None:
    body = BASE + '\n[target]\ncallable = "myapp.classify"\n'

    with pytest.raises(ConfigError, match="callable"):
        load_config(write_config(tmp_path, body))


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write_config(tmp_path, "[dataset\n"))


def test_a_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read config"):
        load_config(tmp_path / "evalgate.toml")


def test_a_broken_dataset_is_reported_against_the_config(tmp_path: Path) -> None:
    path = write_config(tmp_path, BASE)
    (tmp_path / "data" / "items.jsonl").write_text('{"id": "x"}\n', encoding="utf-8")

    with pytest.raises(ConfigError, match=r"\[dataset\]"):
        load_config(path)
