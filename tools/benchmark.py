"""Reproduce EvalGate's seeded binary regression-detection benchmark.

The fixed plan varies paired sample size and a one-way scorer mutation. It applies the
same binary inference and Holm correction as the check, with no model or network calls.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evalgate.power import BinaryScenario, estimate

REPORT = Path(__file__).resolve().parents[1] / "docs" / "benchmark.md"
"""Default location for the checked-in generated benchmark."""


@dataclass(frozen=True)
class BenchmarkPlan:
    """Frozen settings for the generated detection and false-alarm curves."""

    sizes: tuple[int, ...] = (25, 50, 100, 200, 400)
    drops: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
    baseline_rate: float = 0.8
    discordance: float = 0.1
    tolerance: float = 0.05
    gate_slices: int = 3
    draws: int = 400
    seed: int = 19


_PLAN = BenchmarkPlan()


def render(plan: BenchmarkPlan = _PLAN) -> str:
    """Render the benchmark, freezing both the protocol and measured frequencies."""
    lines = [
        "# Seeded regression detection",
        "",
        "Reproduce this report with `uv run python tools/benchmark.py --write`.",
        "Check the committed report with `uv run python tools/benchmark.py --check`.",
        "",
        "## Protocol",
        "",
        f"Each row uses {plan.draws} simulated checks with seed {plan.seed}. Each check tests "
        f"{plan.gate_slices} independent gate slices with the same paired sample size. "
        f"The baseline succeeds with probability {plan.baseline_rate:.0%}. Unchanged versions "
        f"disagree on {plan.discordance:.0%} of pairs, with equal probabilities of "
        "a fix and a break.",
        "",
        "The mutation converts some pairs that both succeeded into baseline-only successes in "
        "one slice. A ten-point planted drop subtracts ten points from that slice's population "
        "success rate and increases its disagreement by the same amount. The other slices "
        "use the unchanged population.",
        "",
        f"The gate uses its Tango interval at 95% confidence and a tolerance of "
        f"{plan.tolerance * 100:g} points. The shipped Holm rule controls regression calls "
        "across the family. The zero-drop rows measure the observed family false-alarm rate. "
        "A change exactly at the tolerance is on the decision boundary. The target column "
        "counts how often the planted slice itself was called regressed. Rows reuse the same "
        "seed for reproducibility.",
        "",
        "| Pairs per slice | Planted drop (points) | Target slice detected | Any slice flagged |",
        "| ---: | ---: | ---: | ---: |",
    ]
    curves: dict[int, tuple[float, ...]] = {}
    for size in plan.sizes:
        scenario = BinaryScenario(size, plan.baseline_rate, plan.discordance, plan.tolerance)
        results = tuple(
            estimate(
                scenario, drop=drop, draws=plan.draws, seed=plan.seed, gate_slices=plan.gate_slices
            )
            for drop in plan.drops
        )
        curves[size] = tuple(result.target_regression for result in results)
        for result in results:
            flag = " (False alarm)" if result.drop == 0.0 else ""
            lines.append(
                f"| {size} | {result.drop * 100:g} | {result.target_regression:.1%} | "
                f"{result.family_regression:.1%}{flag} |"
            )
    if len(curves) >= 3 and len(plan.drops) >= 2:
        lines.extend(_plot(curves, plan.drops))
    lines.extend(
        [
            "",
            "## Plan a slice",
            "",
            "```sh",
            "evalgate power --size 100 --baseline-rate 0.80 --discordance 0.10 "
            "--tolerance 0.05 --gate-slices 3",
            "```",
            "",
            "Use the slice's paired size and baseline score. Estimate unchanged discordance from "
            "paired reruns of the same version. The command reports the first drop on its grid "
            "whose observed detection reaches the requested target. Rates near that threshold "
            "can vary with the seed or draw count. A thin slice may have no detectable drop "
            "under the specified model.",
            "",
            "These are synthetic scorer-level perturbations. Prompt edits, model switches and "
            "dependent real slices can have different disagreement and detection rates. A "
            "zero observed false-alarm count is subject to finite simulation uncertainty. "
            "The benchmark makes zero model calls.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot(curves: dict[int, tuple[float, ...]], drops: tuple[float, ...]) -> list[str]:
    """Plot the target detection curve for three slice sizes from the table."""
    sizes = sorted(curves)
    selected = (sizes[0], sizes[len(sizes) // 2], sizes[-1])
    categories = ", ".join(f'"{drop * 100:g}"' for drop in drops)
    lines = [
        "",
        "## Detection curves",
        "",
        f"The blue, orange and teal lines represent {selected[0]}, {selected[1]} and "
        f"{selected[2]} pairs per slice, respectively.",
        "",
        "```mermaid",
        "---",
        "config:",
        "  themeVariables:",
        "    xyChart:",
        "      plotColorPalette: '#0072B2,#D55E00,#009E73'",
        "---",
        "xychart-beta",
        f'  x-axis "Planted drop (points)" [{categories}]',
        '  y-axis "Target slice detected (%)" 0 --> 100',
    ]
    for size in selected:
        percentages = ", ".join(f"{value * 100:.1f}" for value in curves[size])
        lines.append(f"  line [{percentages}]")
    lines.append("```")
    return lines


def main(argv: Sequence[str] | None = None, *, plan: BenchmarkPlan = _PLAN) -> int:
    """Print, generate, or verify the checked-in report."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write the Markdown report.")
    mode.add_argument("--check", action="store_true", help="Fail if the Markdown report is stale.")
    parser.add_argument("--output", type=Path, default=REPORT, help="Markdown report path.")
    args = parser.parse_args(argv)
    report = render(plan)
    if args.write:
        args.output.write_text(report, encoding="utf-8", newline="\n")
    elif args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != report:
            print("Benchmark report is stale. Run uv run python tools/benchmark.py --write.")
            return 1
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
