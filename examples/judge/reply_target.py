"""Synthetic reply templates with a seeded regression on returns."""

from pydantic import JsonValue

REPLIES = {
    "delivery": "Track your delivery using the tracking link in your shipping email.",
    "returns": "Start a return from your order page within 30 days.",
}
"""Fictional support policies used only by the example."""


def baseline(value: JsonValue) -> JsonValue:
    """Give the reference reply for the supplied synthetic intent."""
    if not isinstance(value, dict) or not isinstance(topic := value.get("topic"), str):
        raise ValueError("the example expects an object with a topic")
    return REPLIES[topic]


def candidate(value: JsonValue) -> JsonValue:
    """Seed a broken returns instruction while keeping the delivery replies unchanged."""
    answer = baseline(value)
    return "Returns are never accepted." if answer == REPLIES["returns"] else answer
