"""Prove every gate is connected by making each one fail on purpose.

A gate that has never failed is not known to work. A misconfigured path, a tool that
exits zero when it finds nothing, an include list that quietly matches no files: each
produces a green check that means nothing, and the longer it stays green the more it is
trusted. This injects one fault per gate, runs only that gate, and requires a non-zero
exit.

Doing it as a script rather than once by hand means it can be re-run after any change to
the toolchain, which is when a gate most often comes loose.

Deliberate boundary: the faults are the smallest thing that should trip each gate. This
checks that a gate fires, not that it catches every class of fault -- measuring detection
rates properly is what the product itself is for.

Usage:
    uv run python tools/verify_gates.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Probe:
    """One deliberate fault, aimed at exactly one gate.

    Attributes:
        gate: The gate name, as `tools/check.py` selects it.
        path: File to create, relative to the repository root.
        content: The faulty contents.
        fault: What is wrong with it, for the report.
    """

    gate: str
    path: Path
    content: str
    fault: str


PROBES: tuple[Probe, ...] = (
    Probe(
        gate="format",
        path=Path("src/evalgate/_probe.py"),
        content='"""Probe."""\n\nx   =    1\n',
        fault="spacing the formatter would rewrite",
    ),
    Probe(
        gate="lint",
        path=Path("src/evalgate/_probe.py"),
        content='"""Probe."""\n\nimport os\n',
        fault="an imported name nothing uses",
    ),
    Probe(
        gate="types",
        path=Path("src/evalgate/_probe.py"),
        content='"""Probe."""\n\nCOUNT: int = "not a number"\n',
        fault="a string annotated as an int",
    ),
    Probe(
        gate="docs",
        path=Path("_probe.md"),
        content="# Probe\n\nSee [a file that is not there](no_such_file.md).\n",
        fault="a relative link with no target",
    ),
    Probe(
        gate="tests",
        path=Path("tests/test_probe.py"),
        content='"""Probe."""\n\n\ndef test_probe_fails() -> None:\n    assert False\n',
        fault="an assertion that cannot hold",
    ),
)


def say(message: str) -> None:
    """Print a line and flush it, so this script's output interleaves with the gates'."""
    print(message, flush=True)


def run_check(gate: str) -> int:
    """Run one gate through `tools/check.py` with its output suppressed.

    The gate is expected to fail here, so its diagnostics are noise. What matters is the
    exit code.

    Args:
        gate: Gate name to run.

    Returns:
        The exit code of `tools/check.py`.
    """
    completed = subprocess.run(
        (sys.executable, "tools/check.py", gate),
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode


def probe(case: Probe) -> bool:
    """Inject one fault, run its gate, and report whether the gate caught it.

    The file is always removed again, including when the gate behaves unexpectedly, so a
    failed run never leaves a broken repository behind.

    Args:
        case: The fault to inject.

    Returns:
        True when the gate failed as it should.

    Raises:
        SystemExit: If the probe path already exists, which would mean overwriting real work.
    """
    target = ROOT / case.path
    if target.exists():
        raise SystemExit(f"refusing to overwrite an existing file: {case.path.as_posix()}")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(case.content, encoding="utf-8")
        caught = run_check(case.gate) != 0
    finally:
        target.unlink(missing_ok=True)

    mark = "\033[32mcaught\033[0m" if caught else "\033[31mMISSED\033[0m"
    say(f"   {mark}  {case.gate:<7} {case.fault}")
    return caught


def main() -> int:
    """Run every probe, then confirm the repository is clean and green again.

    Returns:
        0 when every gate caught its fault and the full check passes afterwards.
    """
    say("\033[1m-- injecting one fault per gate\033[0m")
    results = [probe(case) for case in PROBES]

    say("\n\033[1m-- confirming the repository is green again\033[0m")
    restored = _run_full_check()
    # Built outside the f-string: a backslash inside an f-string expression is a
    # syntax error before Python 3.12, and this package claims to support 3.11.
    verdict = "\033[32mgreen\033[0m" if restored else "\033[31mNOT GREEN\033[0m"
    say(f"   {verdict}  full check")

    missed = [case.gate for case, caught in zip(PROBES, results, strict=True) if not caught]
    if missed:
        say(f"\n{len(missed)} gate(s) did not fire: {', '.join(missed)}")
        return 1
    if not restored:
        say("\nevery gate fired, but the repository did not return to green")
        return 1
    say(f"\nall {len(PROBES)} gate(s) are connected")
    return 0


def _run_full_check() -> bool:
    """Run every gate and report whether they all pass."""
    completed = subprocess.run(
        (sys.executable, "tools/check.py"),
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
