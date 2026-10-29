"""Two versions of a keyword triage classifier, so the gate has something to compare.

No model is involved. The point of the example is the measurement machinery, and a rule-based
classifier makes the run instant, free and identical on every machine -- which is what you
want from the thing you are using to test a measuring instrument.

`classify_v1` is the baseline. `classify_v2` is the same classifier with two warranty cues
removed, which costs it accuracy on the warranty slice and leaves billing and refunds
untouched. That is a seeded regression of known shape: the gate should find it in warranty and
report no evidence of harm elsewhere.
"""

from __future__ import annotations

from pydantic import JsonValue

#: Cues for each intent. Order matters: the first intent with a hit wins.
CUES_V1: dict[str, tuple[str, ...]] = {
    "refunds": ("refund", "money back", "return my", "reimburse"),
    "warranty": ("warranty", "guarantee", "covered", "repair"),
    "billing": ("charge", "invoice", "subscription", "billing", "payment"),
}

#: The same, with two warranty cues dropped. Everything else is identical.
CUES_V2: dict[str, tuple[str, ...]] = {
    "refunds": CUES_V1["refunds"],
    "warranty": ("warranty", "repair"),
    "billing": CUES_V1["billing"],
}


def _classify(text: JsonValue, cues: dict[str, tuple[str, ...]]) -> JsonValue:
    """Return the first intent with a cue present, or `unknown`.

    `unknown` rather than a guess: a classifier that falls back to its most common label looks
    better on accuracy than it deserves, and hides exactly the failure a gate should catch.
    """
    if not isinstance(text, str):
        return "unknown"
    lowered = text.casefold()
    for intent, words in cues.items():
        if any(word in lowered for word in words):
            return intent
    return "unknown"


def classify_v1(text: JsonValue) -> JsonValue:
    """Baseline classifier."""
    return _classify(text, CUES_V1)


def classify_v2(text: JsonValue) -> JsonValue:
    """Candidate classifier, with two warranty cues removed."""
    return _classify(text, CUES_V2)
