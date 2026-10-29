"""Command-line front end for the doc-link gate.

Owning artifact: the B0 verification pipeline. The checking logic lives in `doc_links`
so it can be unit-tested without spawning a process; this file is only the shell around it.
"""

from __future__ import annotations

from pathlib import Path

from doc_links import find_broken_links

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    """Report broken relative links under the repository root.

    Returns:
        0 when every relative link resolves, 1 otherwise.
    """
    broken = find_broken_links(ROOT)
    if not broken:
        return 0
    print(f"{len(broken)} broken link(s):")
    for link in broken:
        print(f"  {link}")
    return 1


if __name__ == "__main__":
    # Python puts this file's directory on sys.path, so `doc_links` imports cleanly.
    raise SystemExit(main())
