"""Scorers settle what they can and refuse what they cannot."""

import pytest
from pydantic import JsonValue

from evalgate.scorers import (
    CustomScorer,
    EditSimilarity,
    ExactMatch,
    JsonSchemaMatch,
    RegexMatch,
    Score,
    ScoreKind,
    Scorer,
    ScorerError,
    SetF1,
    levenshtein,
    score_all,
)

# --------------------------------------------------------------------------- #
# Score itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_a_score_outside_the_unit_interval_is_refused(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Score(value=value, kind=ScoreKind.CONTINUOUS)


def test_a_fractional_binary_score_is_refused() -> None:
    # A binary scorer feeds the exact conditional interval, which counts agreements. A 0.5
    # would make that count meaningless rather than merely imprecise.
    with pytest.raises(ValueError, match="binary score"):
        Score(value=0.5, kind=ScoreKind.BINARY)


# --------------------------------------------------------------------------- #
# Exact match
# --------------------------------------------------------------------------- #


def test_exact_match_ignores_case_and_surrounding_whitespace_by_default() -> None:
    assert ExactMatch().score("  Billing\n", "billing").value == 1.0


def test_exact_match_can_be_made_strict() -> None:
    strict = ExactMatch(case_sensitive=True, strip=False)
    assert strict.score("  Billing\n", "billing").value == 0.0


def test_exact_match_compares_structures_by_canonical_json() -> None:
    # Two equal objects written with their keys in a different order are the same answer.
    left: JsonValue = {"intent": "billing", "urgent": True}
    right: JsonValue = {"urgent": True, "intent": "billing"}
    assert ExactMatch().score(left, right).value == 1.0


def test_exact_match_is_binary() -> None:
    assert ExactMatch().kind is ScoreKind.BINARY


def test_exact_match_refuses_an_item_with_no_expectation() -> None:
    # Scoring it zero would make a dataset of judge-only items look like a total failure.
    with pytest.raises(ScorerError, match="no expected value"):
        ExactMatch().score("anything", None)


# --------------------------------------------------------------------------- #
# Regex
# --------------------------------------------------------------------------- #


def test_regex_requires_the_whole_output_by_default() -> None:
    scorer = RegexMatch(pattern=r"\d{4}")
    assert scorer.score("1234", None).value == 1.0
    assert scorer.score("order 1234 placed", None).value == 0.0


def test_regex_can_search_instead_of_matching_the_whole_output() -> None:
    scorer = RegexMatch(pattern=r"\d{4}", full=False)
    assert scorer.score("order 1234 placed", None).value == 1.0


def test_regex_can_ignore_case() -> None:
    assert RegexMatch(pattern="yes", ignore_case=True).score("YES", None).value == 1.0


def test_a_broken_pattern_fails_when_the_scorer_is_built_not_per_item() -> None:
    # Discovering it item by item during a run would waste the run.
    with pytest.raises(ScorerError, match="does not compile"):
        RegexMatch(pattern="(unclosed")


def test_regex_refuses_a_non_string_output() -> None:
    with pytest.raises(ScorerError, match="must be a string"):
        RegexMatch(pattern=".*").score({"a": 1}, None)


# --------------------------------------------------------------------------- #
# JSON schema
# --------------------------------------------------------------------------- #

SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"intent": {"type": "string"}, "urgent": {"type": "boolean"}},
    "required": ["intent"],
    "additionalProperties": False,
}


def test_schema_accepts_a_conforming_object() -> None:
    assert JsonSchemaMatch(schema=SCHEMA).score({"intent": "billing"}, None).value == 1.0


def test_schema_parses_a_json_string_output() -> None:
    # A model asked for JSON emits it as text.
    assert JsonSchemaMatch(schema=SCHEMA).score('{"intent": "billing"}', None).value == 1.0


def test_prose_where_json_was_asked_for_scores_zero_rather_than_raising() -> None:
    # This is the regression the scorer exists to catch, not a misconfiguration.
    result = JsonSchemaMatch(schema=SCHEMA).score("I think it is about billing.", None)
    assert result.value == 0.0
    assert "not JSON" in str(result.detail["error"])


def test_schema_reports_the_first_failure_with_its_path() -> None:
    result = JsonSchemaMatch(schema=SCHEMA).score({"intent": 7}, None)
    assert result.value == 0.0
    assert result.detail["path"] == "intent"


def test_a_broken_schema_fails_when_the_scorer_is_built() -> None:
    # A broken schema would otherwise fail every item and read as a total regression.
    with pytest.raises(ScorerError, match="schema is not valid"):
        JsonSchemaMatch(schema={"type": "not-a-type"})


# --------------------------------------------------------------------------- #
# Set F1
# --------------------------------------------------------------------------- #


def test_f1_is_one_for_identical_sets_regardless_of_order() -> None:
    assert SetF1().score(["b", "a"], ["a", "b"]).value == 1.0


def test_f1_gives_partial_credit() -> None:
    # Two of three predicted are right, and two of two expected were found.
    result = SetF1().score(["a", "b", "c"], ["a", "b"])
    assert result.value == pytest.approx(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0))
    assert result.detail["true_positives"] == 2


def test_f1_is_zero_with_no_overlap() -> None:
    assert SetF1().score(["x"], ["a"]).value == 0.0


def test_finding_nothing_where_nothing_was_expected_is_correct() -> None:
    # Scoring this zero would punish the model for being right.
    assert SetF1().score([], []).value == 1.0


def test_finding_something_where_nothing_was_expected_is_wrong() -> None:
    assert SetF1().score(["a"], []).value == 0.0


def test_f1_ignores_repeats_because_it_is_a_set_measure() -> None:
    assert SetF1().score(["a", "a", "b"], ["a", "b"]).value == 1.0


def test_f1_refuses_a_non_list() -> None:
    with pytest.raises(ScorerError, match="must be a list"):
        SetF1().score("a, b", ["a", "b"])


def test_f1_is_continuous() -> None:
    assert SetF1().kind is ScoreKind.CONTINUOUS


# --------------------------------------------------------------------------- #
# Edit distance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("left", "right", "distance"),
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("a", "b", 1),
    ],
)
def test_levenshtein_matches_known_distances(left: str, right: str, distance: int) -> None:
    assert levenshtein(left, right) == distance


def test_levenshtein_is_symmetric() -> None:
    assert levenshtein("saturday", "sunday") == levenshtein("sunday", "saturday")


def test_edit_similarity_is_one_for_identical_strings() -> None:
    assert EditSimilarity().score("billing", "Billing ").value == 1.0


def test_edit_similarity_scales_with_the_longer_string() -> None:
    # One substitution out of seven characters.
    assert EditSimilarity().score("billing", "billong").value == pytest.approx(1 - 1 / 7)


def test_edit_similarity_of_two_empty_strings_is_one() -> None:
    # Dividing by zero to discover that two empty strings are identical would be worse.
    assert EditSimilarity().score("", "").value == 1.0


def test_edit_similarity_refuses_a_non_string() -> None:
    with pytest.raises(ScorerError, match="must be a string"):
        EditSimilarity().score(["a"], "a")


# --------------------------------------------------------------------------- #
# Custom scorers
# --------------------------------------------------------------------------- #


def test_a_custom_scorer_runs() -> None:
    scorer = CustomScorer(
        name="length_ratio",
        kind=ScoreKind.CONTINUOUS,
        function=lambda output, expected: min(1.0, len(str(output)) / 10),
    )
    assert scorer.score("abcde", None).value == pytest.approx(0.5)


def test_a_custom_scorer_returning_an_impossible_value_is_refused() -> None:
    # Left alone, 1.4 would flow into the interval arithmetic and produce a confident
    # nonsense verdict.
    scorer = CustomScorer(
        name="broken", kind=ScoreKind.CONTINUOUS, function=lambda output, expected: 1.4
    )
    with pytest.raises(ScorerError, match="unusable score"):
        scorer.score("x", None)


def test_a_custom_scorer_that_raises_is_reported_as_a_scorer_error() -> None:
    def explode(output: JsonValue, expected: JsonValue | None) -> float:
        raise KeyError("missing")

    scorer = CustomScorer(name="explodes", kind=ScoreKind.BINARY, function=explode)
    with pytest.raises(ScorerError, match="raised KeyError"):
        scorer.score("x", None)


def test_a_custom_binary_scorer_cannot_return_a_fraction() -> None:
    scorer = CustomScorer(name="half", kind=ScoreKind.BINARY, function=lambda output, expected: 0.5)
    with pytest.raises(ScorerError, match="unusable score"):
        scorer.score("x", None)


# --------------------------------------------------------------------------- #
# Running several
# --------------------------------------------------------------------------- #


def test_score_all_returns_one_entry_per_scorer() -> None:
    scorers: list[Scorer] = [ExactMatch(), EditSimilarity()]
    results = score_all(scorers, "billing", "billing")
    assert set(results) == {"exact_match", "edit_similarity"}
    assert results["exact_match"].value == 1.0


def test_score_all_refuses_two_scorers_with_the_same_name() -> None:
    # One would silently overwrite the other, and the report would be short a column.
    scorers: list[Scorer] = [ExactMatch(), ExactMatch(case_sensitive=True)]
    with pytest.raises(ScorerError, match="both named"):
        score_all(scorers, "a", "a")


def test_every_scorer_satisfies_the_protocol() -> None:
    built: list[Scorer] = [
        ExactMatch(),
        RegexMatch(pattern=".*"),
        JsonSchemaMatch(schema=SCHEMA),
        SetF1(),
        EditSimilarity(),
        CustomScorer(name="c", kind=ScoreKind.BINARY, function=lambda o, e: 1.0),
    ]
    for scorer in built:
        assert isinstance(scorer, Scorer)
        assert scorer.name
        assert scorer.kind in ScoreKind
