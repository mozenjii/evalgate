"""Check that every relative link in the repository's markdown actually resolves.

Owning artifact: the B0 verification pipeline (`tools/check.py` runs this as one gate).

The documentation here is a graph: the entry files point at the execution contract, the
contract points at ADRs, ADRs point back. A renamed file silently breaks that graph, and
nobody notices until someone follows a dead link while trying to understand a decision.

Deliberate boundary: this checks *existence of link targets only*. It does not validate
anchors within a file, and it never follows http(s) links -- a network call in a local
check makes the check flaky, and a flaky check gets switched off.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

__all__ = ["BrokenLink", "find_broken_links", "iter_markdown_files", "parse_links"]

# Matches an inline markdown link: [label](target). The label may contain balanced
# brackets such as [[wiki]]; the target stops at the first unescaped closing paren.
_LINK = re.compile(r"\[(?P<label>[^\]]*)\]\((?P<target>[^()\s]*(?:\([^()]*\)[^()\s]*)*)\)")

# Fenced code blocks hold examples, not navigation. A link in a shell snippet is text.
_FENCE = re.compile(r"^\s*(```|~~~)")

#: Directories that never contain documentation we own.
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "graphify-out",
        "dist",
        "build",
    }
)

#: Link schemes that are someone else's problem to resolve.
_EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "tel:", "ftp://", "data:")


@dataclass(frozen=True)
class BrokenLink:
    """One markdown link whose target does not exist on disk.

    Attributes:
        source: The markdown file containing the link, relative to the repository root.
        line: 1-based line number of the link, so an editor can jump straight to it.
        target: The link target exactly as written, before decoding.
        resolved: The filesystem path the target resolved to, relative to the root.
    """

    source: Path
    line: int
    target: str
    resolved: Path

    def __str__(self) -> str:
        """Render as `path:line: target -> resolved`, the shape editors can jump to."""
        return f"{self.source.as_posix()}:{self.line}: {self.target} -> {self.resolved.as_posix()}"


def iter_markdown_files(root: Path) -> list[Path]:
    """Return every markdown file under `root`, skipping build and vendor directories.

    Args:
        root: Directory to walk. Usually the repository root.

    Returns:
        Absolute paths, sorted, so the report order is stable between runs.
    """
    found: list[Path] = []
    for path in root.rglob("*.md"):
        if any(part in SKIPPED_DIRECTORIES for part in path.relative_to(root).parts):
            continue
        found.append(path)
    return sorted(found)


def parse_links(text: str) -> list[tuple[int, str]]:
    """Extract inline markdown links from `text`, ignoring fenced code blocks.

    Args:
        text: The full contents of a markdown file.

    Returns:
        `(line_number, target)` pairs with 1-based line numbers, in document order.
    """
    links: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        links.extend((lineno, match.group("target")) for match in _LINK.finditer(line))
    return links


def _is_checkable(target: str) -> bool:
    """Report whether a link target names a path this checker can resolve."""
    if not target or target.startswith("#"):
        return False  # A bare anchor points inside the same file; anchors are out of scope.
    return not target.lower().startswith(_EXTERNAL_PREFIXES)


def _resolve(source: Path, target: str) -> Path:
    """Resolve a link target against the file that contains it.

    Percent-encoding is decoded first: the documentation links to real paths that contain
    spaces, written as `app/technical%20context/...`, and those must resolve.
    """
    path_part = target.split("#", 1)[0]
    decoded = urllib.parse.unquote(path_part)
    return (source.parent / decoded).resolve()


def find_broken_links(root: Path) -> list[BrokenLink]:
    """Find every relative markdown link under `root` whose target is missing.

    Args:
        root: Repository root. Links are resolved relative to each containing file.

    Returns:
        Broken links in file then line order. An empty list means the gate passes.
    """
    root = root.resolve()
    broken: list[BrokenLink] = []
    for markdown in iter_markdown_files(root):
        text = markdown.read_text(encoding="utf-8", errors="replace")
        for lineno, target in parse_links(text):
            if not _is_checkable(target):
                continue
            resolved = _resolve(markdown, target)
            if resolved.exists():
                continue
            broken.append(
                BrokenLink(
                    source=markdown.relative_to(root),
                    line=lineno,
                    target=target,
                    resolved=_relative_to_root(resolved, root),
                )
            )
    return broken


def _relative_to_root(path: Path, root: Path) -> Path:
    """Shorten `path` for display, falling back to the absolute path when it escapes `root`."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path
