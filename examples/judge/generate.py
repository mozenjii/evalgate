"""Regenerate the synthetic support-reply evaluation set."""

from pathlib import Path

from reply_target import REPLIES

from evalgate.dataset import Item, Manifest, Provenance, content_hash

_QUESTIONS = {
    "delivery": (
        "Where is my order?",
        "Can I track my parcel?",
        "How do I see when my delivery arrives?",
        "Where will I find the tracking link?",
        "How do I find my delivery status?",
        "My package shipped. How can I follow it?",
        "Is there a place to check the shipment?",
        "How can I see where the courier is?",
    ),
    "returns": (
        "Can I return an item?",
        "How do I start a return?",
        "What is the deadline for a return?",
        "Where in my account can I return this?",
        "Can I send it back after two weeks?",
        "I bought this yesterday. How do I return it?",
        "Do I need to start a return on the order page?",
        "How much time do I have to send it back?",
    ),
}


def main() -> None:
    """Write the stable items and their validated content hash beside the config."""
    items = tuple(
        Item(
            id=f"{topic}-{index}",
            input={"message": question, "topic": topic},
            expected=REPLIES[topic],
            slices={"topic": topic},
            provenance=Provenance.SYNTHETIC,
        )
        for topic, questions in _QUESTIONS.items()
        for index, question in enumerate(questions, start=1)
    )
    manifest = Manifest(
        name="synthetic-support-replies",
        version="1",
        items_file="items.jsonl",
        content_sha256=content_hash(items),
        slice_dimensions={"topic": tuple(_QUESTIONS)},
        description="Fictional delivery and returns questions for a local judge replay.",
    )
    root = Path(__file__).resolve().parent
    (root / "items.jsonl").write_text(
        "".join(f"{item.model_dump_json(exclude_none=True)}\n" for item in items),
        encoding="utf-8",
        newline="\n",
    )
    (root / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
