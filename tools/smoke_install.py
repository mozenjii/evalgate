"""Exercise the installed EvalGate command against the seeded triage regression.

Run this from an isolated environment that contains the wheel or source archive. The example
is supplied by the checkout as input data. The CLI and its dependencies come from the install.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from tempfile import TemporaryDirectory

import evalgate


def smoke(example: Path, project: str) -> None:
    """Use installed metadata and require the documented regression exit and report."""
    installed = distribution(project)
    if installed.version != evalgate.__version__:
        raise RuntimeError("installed package version disagrees with the runtime")
    command = next(
        entry
        for entry in installed.entry_points
        if entry.group == "console_scripts" and entry.name == "evalgate"
    )
    program = shutil.which("evalgate", path=str(Path(sys.executable).parent))
    if command.value != "evalgate.cli:main" or program is None:
        raise RuntimeError("unexpected installed console script")
    config = str(example.resolve() / "evalgate.toml")
    with TemporaryDirectory(prefix="evalgate-install-") as work:
        for label, target in (("base", "classify_v1"), ("head", "classify_v2")):
            subprocess.run(
                [
                    program,
                    "run",
                    "--config",
                    config,
                    "--label",
                    label,
                    "--out",
                    f"{label}.jsonl",
                    "--target",
                    f"target:{target}",
                    "--quiet",
                ],
                cwd=work,
                check=True,
            )
        result = subprocess.run(
            [program, "compare", "base.jsonl", "head.jsonl", "--config", config],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
        )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if (
        result.returncode != 1
        or "1 regressed /" not in result.stdout
        or not any(
            "intent=warranty" in row and row.endswith("regressed")
            for row in result.stdout.splitlines()
        )
    ):
        raise RuntimeError("installed command did not find the seeded warranty regression")


def main() -> None:
    """Accept the example directory and distribution name."""
    if len(sys.argv) != 3:
        raise SystemExit("usage: smoke_install.py EXAMPLE_DIR DISTRIBUTION")
    smoke(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
