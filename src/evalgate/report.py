"""Render a comparison for a person: markdown for a pull request, plain text for a terminal.

The report is where honesty is won or lost, because it is the only part most people read. It
puts the worst slices first, never calls a slice that could not answer a pass, says in so many
words what each slice can and cannot rule out, and lists everything that was left out of the
comparison and why.

Deliberate boundary: this formats a finished comparison. It decides nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from evalgate.gate import Comparison, Outcome, RunSummary, SliceComparison
from evalgate.stats import Verdict

__all__ = ["MARKER", "render_markdown", "render_text"]

MARKER = "<!-- evalgate-report -->"
"""Hidden marker that lets the GitHub Action find and update its own comment."""

METHOD_URL = "https://github.com/mozenjii/evalgate/blob/main/docs/statistics.md"

_SEVERITY = {Verdict.REGRESSED: 0, Verdict.INCONCLUSIVE: 1, Verdict.NON_INFERIOR: 2}

_ASCII = str.maketrans({"·": "/", "—": "-"})

Policy = Literal["warn", "fail"]


def render_markdown(comparison: Comparison, *, confidence: float, on_inconclusive: Policy) -> str:
    """Render a comparison as a pull-request comment.

    Args:
        comparison: The finished comparison.
        confidence: The confidence level the intervals were built at.
        on_inconclusive: The configured policy, stated in the notes so a reader knows whether
            a slice that is not guarded failed the build.

    Returns:
        Markdown, beginning with `MARKER`.
    """
    lines = [MARKER, f"### evalgate: {_headline(comparison)}", "", f"**{_counts(comparison)}**"]
    lines.append(_compared(comparison))
    lines.append("")
    lines += _markdown_table("Gate slices", comparison.gate, confidence)
    if comparison.watch:
        title = "Watch slices (reported, never fail the build)"
        lines += _markdown_table(title, comparison.watch, confidence)
    lines += ["**Notes**", ""]
    lines += [
        f"- {note}" for note in _notes(comparison, confidence, on_inconclusive, markdown=True)
    ]
    return "\n".join(lines) + "\n"


def render_text(comparison: Comparison, *, confidence: float, on_inconclusive: Policy) -> str:
    """Render a comparison for a terminal.

    Args:
        comparison: The finished comparison.
        confidence: The confidence level the intervals were built at.
        on_inconclusive: The configured policy.

    Returns:
        Plain text with aligned columns.
    """
    lines = [f"evalgate: {_headline(comparison)}", _counts(comparison), _compared(comparison), ""]
    lines += _text_table("Gate slices", comparison.gate, confidence)
    if comparison.watch:
        lines += _text_table("Watch slices (never fail the build)", comparison.watch, confidence)
    lines.append("Notes:")
    lines += [
        f"  - {note}" for note in _notes(comparison, confidence, on_inconclusive, markdown=False)
    ]
    # Plain ASCII for terminals: a Windows console in its default code page prints the middle
    # dot and the em dash as replacement characters. The markdown version keeps them.
    return ("\n".join(lines) + "\n").translate(_ASCII)


# --------------------------------------------------------------------------- shared wording


def _headline(comparison: Comparison) -> str:
    """One line for the whole check, in words that cannot be mistaken for a pass."""
    match comparison.outcome:
        case Outcome.REGRESSED:
            return "regression found"
        case Outcome.UNGUARDED:
            count = sum(1 for entry in comparison.gate if entry.verdict is Verdict.INCONCLUSIVE)
            verb = "is" if count == 1 else "are"
            return f"no regression found, but {_count(count, 'gate slice')} {verb} not guarded"
        case Outcome.OK:
            return "no regression found"


def _counts(comparison: Comparison) -> str:
    """Gate verdict counts."""
    gate = comparison.gate
    regressed = sum(1 for entry in gate if entry.verdict is Verdict.REGRESSED)
    unguarded = sum(1 for entry in gate if entry.verdict is Verdict.INCONCLUSIVE)
    fine = sum(1 for entry in gate if entry.verdict is Verdict.NON_INFERIOR)
    return f"{regressed} regressed · {unguarded} not guarded · {fine} ok"


def _compared(comparison: Comparison) -> str:
    """What was compared against what, on which dataset."""
    manifest = comparison.dataset.manifest
    return (
        f"Comparing `{comparison.head.label}` against `{comparison.base.label}` on "
        f"`{manifest.name}` v{manifest.version} ({len(comparison.dataset)} items)."
    )


def _row(entry: SliceComparison) -> tuple[str, ...]:
    """One slice as table cells, differences in points."""
    if entry.result is None:
        difference = "—"
    else:
        interval = entry.result.interval
        difference = f"{_pts(interval.point)} ({_pts(interval.low)} to {_pts(interval.high)})"
    # A regressed slice's "rules out" figure is true but useless: the point is that it is worse.
    rules_out = entry.rules_out
    if rules_out is None or entry.verdict is Verdict.REGRESSED:
        shown_rules_out = "—"
    else:
        shown_rules_out = f"{rules_out * 100:.1f}"
    n = f"{entry.n}" if entry.n == entry.size else f"{entry.n} of {entry.size}"
    return (
        str(entry.slice),
        entry.scorer,
        n,
        difference,
        shown_rules_out,
        f"{entry.tolerance * 100:.1f}",
        _verdict(entry),
    )


def _verdict(entry: SliceComparison) -> str:
    """The verdict in words. Inconclusive is never called a pass."""
    if entry.verdict is Verdict.REGRESSED:
        return "regressed"
    if entry.verdict is Verdict.NON_INFERIOR:
        return "ok"
    return f"not guarded ({entry.gap.value})" if entry.gap is not None else "not guarded"


def _ordered(entries: Sequence[SliceComparison]) -> list[SliceComparison]:
    """Worst first, keeping configuration order within each verdict."""
    return sorted(entries, key=lambda entry: _SEVERITY[entry.verdict])


def _notes(
    comparison: Comparison, confidence: float, on_inconclusive: Policy, *, markdown: bool
) -> list[str]:
    """Everything a reader needs to interpret the tables, and everything left out."""
    consequence = "fail the build" if on_inconclusive == "fail" else "do not fail the build"
    notes = [
        "Differences are candidate minus baseline in points (score x 100); negative is worse. "
        f"Intervals are {confidence:.0%}.",
        "Not guarded means the slice's data cannot rule out a drop as large as its tolerance. "
        "It is not a pass: add items, or widen the tolerance on purpose. Such slices "
        f"{consequence} (policy: {on_inconclusive}).",
        "Each slice is tested on its own; no correction across slices is applied yet.",
    ]
    for side in (comparison.base, comparison.head):
        notes += _side_notes(side, comparison)
    if comparison.input_changed:
        count = len(comparison.input_changed)
        verb = "was" if count == 1 else "were"
        notes.append(
            f"{_count(count, 'item')} {verb} left out because the input changed between runs: "
            f"{_some(comparison.input_changed)}."
        )
    notes.append(
        f"Method: [statistics notes]({METHOD_URL})." if markdown else f"Method: {METHOD_URL}"
    )
    return notes


def _side_notes(side: RunSummary, comparison: Comparison) -> list[str]:
    """Notes about one run: errors, ignored ids, and a dataset that has since changed."""
    notes: list[str] = []
    if side.errors:
        notes.append(
            f"`{side.label}`: {_count(len(side.errors), 'item')} errored and count as failures: "
            f"{_some(side.errors)}."
        )
    if side.unknown:
        verb = "was" if len(side.unknown) == 1 else "were"
        notes.append(
            f"`{side.label}`: {_count(len(side.unknown), 'id')} not in the current dataset "
            f"{verb} ignored: {_some(side.unknown)}."
        )
    current = comparison.dataset.manifest
    if side.dataset is not None and side.dataset.sha256 != current.content_sha256:
        notes.append(
            f"`{side.label}` ran on dataset v{side.dataset.version}, whose items differ from the "
            f"current v{current.version}; it is scored against the current expectations."
        )
    return notes


# --------------------------------------------------------------------------- tables


def _headers(confidence: float) -> tuple[str, ...]:
    return (
        "slice",
        "scorer",
        "n",
        f"difference ({confidence:.0%} CI)",
        "rules out drops over",
        "tolerance",
        "verdict",
    )


def _markdown_table(title: str, entries: Sequence[SliceComparison], confidence: float) -> list[str]:
    lines = [
        f"**{title}**",
        "",
        "| " + " | ".join(_headers(confidence)) + " |",
        "|:--|:--|--:|:--|--:|--:|:--|",
    ]
    for entry in _ordered(entries):
        cells = [cell.replace("|", "\\|") for cell in _row(entry)]
        if entry.verdict is Verdict.REGRESSED:
            cells[-1] = f"**{cells[-1]}**"
        lines.append("| " + " | ".join(cells) + " |")
    return [*lines, ""]


def _text_table(title: str, entries: Sequence[SliceComparison], confidence: float) -> list[str]:
    rows = [_headers(confidence), *(_row(entry) for entry in _ordered(entries))]
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    right_aligned = {2, 4, 5}
    rendered = [
        "  ".join(
            cell.rjust(widths[column]) if column in right_aligned else cell.ljust(widths[column])
            for column, cell in enumerate(row)
        ).rstrip()
        for row in rows
    ]
    return [f"{title}:", *(f"  {line}" for line in rendered), ""]


# --------------------------------------------------------------------------- small helpers


def _pts(value: float) -> str:
    """A difference in points, signed."""
    return f"{value * 100:+.1f}"


def _count(count: int, noun: str) -> str:
    """`_count(1, "item")` is "1 item"; `_count(3, "item")` is "3 items"."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _some(ids: Sequence[str], limit: int = 5) -> str:
    """List a few ids, saying how many more there are."""
    shown = ", ".join(f"`{item}`" for item in ids[:limit])
    return shown if len(ids) <= limit else f"{shown} and {len(ids) - limit} more"
