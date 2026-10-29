"""Build the toy triage dataset.

Run it from the repository root:

    uv run python examples/triage/generate.py

It writes `items.jsonl` and `manifest.json` next to itself. Both are committed, and both are
generated: edit this file, not them. The output is deterministic, so regenerating without
changing this script produces a byte-identical dataset and an unchanged content hash.

The data is synthetic and says so in every item's provenance. It exists to exercise the
pipeline end to end, not to stand as evidence about any model -- a scorer that looks good on
messages written to be classifiable is not a finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

#: Each intent, with the phrasings a message of that intent might use.
PHRASINGS: dict[str, list[str]] = {
    "billing": [
        "I was charged twice this month",
        "my invoice shows the wrong amount",
        "why has my subscription payment gone up",
        "there is a duplicate charge on my card",
        "the billing date moved without warning",
    ],
    "refunds": [
        "I would like a refund for this order",
        "please return my money for the cancelled item",
        "how long does a refund take to arrive",
        "I want to be reimbursed for the damaged goods",
        "can I get my money back on this purchase",
    ],
    "warranty": [
        "is this still under warranty",
        "the guarantee should cover this fault",
        "my device broke and it is covered I think",
        "can you repair it under the warranty terms",
        "does the guarantee include accidental damage",
    ],
}

#: Openers, so repeated phrasings are not duplicate items. They carry no intent cue of their
#: own, which keeps them out of what is being measured.
OPENERS: tuple[str, ...] = (
    "Hello, ",
    "Hi there. ",
    "Good morning, ",
    "Quick question: ",
    "Sorry to bother you, but ",
    "Following up again: ",
    "To whom it may concern, ",
    "Hi, ",
)

#: Padding that makes a message long without changing what it is about.
PADDING = (
    " I have been a customer for several years and have not had a problem like this before,"
    " so I would appreciate it if someone could look into the details and let me know what"
    " the next step is, because I am not sure who else to ask about it."
)


def build() -> list[dict[str, Any]]:
    """Generate the items, balanced across intent and length."""
    items: list[dict[str, Any]] = []
    for intent, phrasings in PHRASINGS.items():
        for repeat in range(len(OPENERS)):
            for index, phrasing in enumerate(phrasings):
                long = repeat >= len(OPENERS) // 2
                number = repeat * len(phrasings) + index
                items.append(
                    {
                        "id": f"{intent}-{number:03d}",
                        "input": OPENERS[repeat] + phrasing + (PADDING if long else ""),
                        "expected": intent,
                        "slices": {"intent": intent, "length": "long" if long else "short"},
                        "provenance": "synthetic",
                        "source": "examples/triage/generate.py",
                    }
                )
    return items


def main() -> None:
    """Write the dataset and its manifest."""
    # Imported here so the script still runs from a checkout where the package is not yet
    # installed; everything above this line is plain data.
    from evalgate.dataset import Item, content_hash

    items = build()
    (HERE / "items.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )

    manifest = {
        "name": "triage",
        "version": "1",
        "items_file": "items.jsonl",
        "content_sha256": content_hash([Item.model_validate(item) for item in items]),
        "slice_dimensions": {
            "intent": ["billing", "refunds", "warranty"],
            "length": ["short", "long"],
        },
        "description": "Synthetic support-ticket triage, for exercising the pipeline.",
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(items)} items and a manifest to {HERE}")


if __name__ == "__main__":
    main()
