"""Plan a binary-scored gate using paired outcomes from an explicit synthetic population.

The simulation samples joint outcomes. The same paired interval and Holm decision used by
the gate determine each simulated verdict. Estimates assume the specified baseline and
disagreements, and describe repeated synthetic samples from that population.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from typing import Annotated

import numpy as np
import typer

from evalgate.stats import Correction, PairedResult, Verdict, family_verdicts, paired_binary

__all__ = ["BinaryScenario", "PowerEstimate", "estimate", "minimum_detectable", "power_command"]


@dataclass(frozen=True)
class BinaryScenario:
    """A family of equally sized binary slices with a shared baseline and tolerance.

    Unchanged versions have the same marginal success rate, with half the discordant pairs
    in either direction. A planted drop turns some both-success pairs into baseline-only
    successes in the target slice.
    """

    size: int
    baseline_rate: float
    discordance: float
    tolerance: float
    confidence: float = 0.95

    def __post_init__(self) -> None:
        """Reject scenarios that cannot represent a paired probability distribution."""
        if self.size < 2:
            raise ValueError("slice size must be at least two pairs")
        if not 0.0 <= self.baseline_rate <= 1.0:
            raise ValueError("baseline rate must be in [0, 1]")
        if not 0.0 <= self.discordance <= 1.0:
            raise ValueError("discordance must be in [0, 1]")
        baseline = Decimal(str(self.baseline_rate))
        if Decimal(str(self.discordance)) > 2 * min(baseline, 1 - baseline):
            raise ValueError("discordance exceeds the feasible range for the baseline rate")
        if not 0.0 < self.tolerance <= 1.0:
            raise ValueError("tolerance must be in (0, 1]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")

    @property
    def maximum_drop(self) -> float:
        """The most mass this mutation can move out of the both-success outcome."""
        return float(Decimal(str(self.baseline_rate)) - Decimal(str(self.discordance)) / 2)


@dataclass(frozen=True)
class PowerEstimate:
    """Observed frequency of regression verdicts over simulated gate checks."""

    drop: float
    target_regression: float
    family_regression: float


def _probabilities(scenario: BinaryScenario, drop: float) -> tuple[float, float, float]:
    """Return candidate-only, baseline-only and concordant probabilities."""
    fixed = scenario.discordance / 2.0
    broken = fixed + drop
    return fixed, broken, max(0.0, 1.0 - fixed - broken)


def estimate(
    scenario: BinaryScenario,
    *,
    drop: float,
    draws: int,
    seed: int,
    gate_slices: int = 1,
) -> PowerEstimate:
    """Measure target and family detection, applying Holm across independent gate slices."""
    if not 0.0 <= drop <= scenario.maximum_drop:
        raise ValueError("planted drop must be between zero and the both-success probability")
    if draws < 1 or gate_slices < 1 or seed < 0:
        raise ValueError("draws and gate slices must be positive, and seed must be non-negative")

    generator = np.random.default_rng(seed)
    target = generator.multinomial(scenario.size, _probabilities(scenario, drop), draws)
    unchanged = generator.multinomial(
        scenario.size, _probabilities(scenario, 0.0), (draws, gate_slices - 1)
    )

    @cache
    def evidence(fixed: int, broken: int) -> PairedResult:
        # Only the discordant counts and paired size affect this test. Represent the
        # concordant items by pairs of false values, which contribute no difference.
        trailing = (False,) * (scenario.size - fixed - broken)
        baseline = (False,) * fixed + (True,) * broken + trailing
        candidate = (True,) * fixed + (False,) * broken + trailing
        return paired_binary(
            baseline, candidate, tolerance=scenario.tolerance, confidence=scenario.confidence
        )

    target_regressed = family_regressed = 0
    for case, controls in zip(target, unchanged, strict=True):
        results = [evidence(int(case[0]), int(case[1]))]
        results.extend(evidence(int(control[0]), int(control[1])) for control in controls)
        verdicts = family_verdicts(results, Correction.HOLM)
        target_regressed += verdicts[0] is Verdict.REGRESSED
        family_regressed += Verdict.REGRESSED in verdicts
    return PowerEstimate(drop, target_regressed / draws, family_regressed / draws)


def minimum_detectable(
    scenario: BinaryScenario,
    *,
    target: float,
    step: float,
    draws: int,
    seed: int,
    gate_slices: int = 1,
) -> PowerEstimate | None:
    """Find the first tested drop strictly beyond the tolerance to reach target power."""
    if not 0.0 < target <= 1.0:
        raise ValueError("target power must be in (0, 1]")
    if not 0.001 <= step <= 1.0:
        raise ValueError("grid step must be in [0.001, 1]")
    if draws < 1 or gate_slices < 1 or seed < 0:
        raise ValueError("draws and gate slices must be positive, and seed must be non-negative")

    # Form the user-authored decimal grid before converting its points to probabilities.
    spacing = Decimal(str(step))
    steps = int(Decimal(str(scenario.maximum_drop)) // spacing)
    start = int(Decimal(str(scenario.tolerance)) // spacing) + 1
    for index in range(start, steps + 1):
        drop = float(index * spacing)
        result = estimate(scenario, drop=drop, draws=draws, seed=seed, gate_slices=gate_slices)
        if result.target_regression >= target:
            return result
    return None


def power_command(
    size: Annotated[int, typer.Option(help="Number of paired items per slice.", min=2)],
    baseline_rate: Annotated[float, typer.Option(help="Baseline success probability.")],
    discordance: Annotated[float, typer.Option(help="Unchanged-pair disagreement probability.")],
    tolerance: Annotated[float, typer.Option(help="Allowed difference as a fraction.")],
    confidence: Annotated[float, typer.Option(help="Confidence used by the gate.")] = 0.95,
    target: Annotated[float, typer.Option(help="Target detection probability.")] = 0.8,
    step: Annotated[float, typer.Option(help="Drop grid spacing as a fraction.")] = 0.01,
    draws: Annotated[int, typer.Option(help="Simulated checks per grid point.", min=1)] = 400,
    seed: Annotated[int, typer.Option(help="Random seed.", min=0)] = 19,
    gate_slices: Annotated[int, typer.Option(help="Independent gate slices.", min=1)] = 1,
) -> None:
    """Estimate a binary slice's minimum detectable drop under a synthetic model."""
    try:
        scenario = BinaryScenario(size, baseline_rate, discordance, tolerance, confidence)
        result = minimum_detectable(
            scenario, target=target, step=step, draws=draws, seed=seed, gate_slices=gate_slices
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"Synthetic paired binary slice (n={size}, baseline={baseline_rate:.1%}, "
        f"unchanged discordance={discordance:.1%}, tolerance={tolerance:.1%}, "
        f"confidence={confidence:.1%})."
    )
    typer.echo(
        f"Holm across {gate_slices} slice(s), {draws} checks per grid point, seed {seed}; "
        f"grid step {step * 100:g} points, target power {target:.1%}."
    )
    if result is None:
        typer.echo("Minimum detectable drop could not be estimated at target power on this grid.")
        return
    typer.echo(
        f"Minimum detectable drop on the grid: {result.drop * 100:g} points "
        f"({result.target_regression:.1%} detected in the target slice)."
    )
