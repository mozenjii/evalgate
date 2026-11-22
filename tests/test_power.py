"""Synthetic paired-slice planning exercises the production decision rules."""

import pytest
import typer
from typer.testing import CliRunner

from evalgate.power import BinaryScenario, estimate, minimum_detectable, power_command


def test_detects_a_large_drop_and_separates_unchanged_false_alarms() -> None:
    scenario = BinaryScenario(size=256, baseline_rate=0.8, discordance=0.1, tolerance=0.03)
    unchanged = estimate(scenario, drop=0.0, draws=120, seed=27, gate_slices=3)
    large = estimate(scenario, drop=0.2, draws=120, seed=27, gate_slices=3)
    assert unchanged.family_regression < 0.08
    assert large.target_regression >= 0.95
    assert large.family_regression >= large.target_regression
    assert large == estimate(scenario, drop=0.2, draws=120, seed=27, gate_slices=3)


def test_deterministic_endpoints_agree_with_the_gate() -> None:
    scenario = BinaryScenario(size=100, baseline_rate=1.0, discordance=0.0, tolerance=0.05)
    unchanged = estimate(scenario, drop=0.0, draws=5, seed=0, gate_slices=3)
    complete_break = estimate(scenario, drop=1.0, draws=5, seed=0, gate_slices=3)
    assert unchanged.family_regression == 0.0
    assert complete_break.target_regression == 1.0


def test_minimum_uses_explicit_grid_and_reports_an_unreachable_target() -> None:
    strong = BinaryScenario(size=400, baseline_rate=0.8, discordance=0.0, tolerance=0.05)
    result = minimum_detectable(strong, target=0.8, step=0.05, draws=100, seed=12)
    assert result is not None
    assert result.drop >= 0.1
    assert result.target_regression >= 0.8

    thin = BinaryScenario(size=2, baseline_rate=0.3, discordance=0.4, tolerance=0.05)
    assert minimum_detectable(thin, target=0.99, step=0.05, draws=80, seed=12) is None


def test_infeasible_discordance_and_drop_fail() -> None:
    with pytest.raises(ValueError, match="discordance"):
        BinaryScenario(size=64, baseline_rate=0.1, discordance=0.4, tolerance=0.05)
    scenario = BinaryScenario(size=64, baseline_rate=0.25, discordance=0.3, tolerance=0.05)
    with pytest.raises(ValueError, match="drop"):
        estimate(scenario, drop=0.2, draws=10, seed=0)


def test_authored_decimal_probabilities_include_feasible_endpoints() -> None:
    at_discordance_limit = BinaryScenario(
        size=100, baseline_rate=0.9, discordance=0.2, tolerance=0.05
    )
    assert at_discordance_limit.maximum_drop == 0.8
    at_drop_limit = BinaryScenario(size=100, baseline_rate=0.3, discordance=0.4, tolerance=0.05)
    result = estimate(at_drop_limit, drop=0.1, draws=10, seed=0)
    assert result.drop == at_drop_limit.maximum_drop


def test_power_cli_prints_assumptions_and_handles_invalid_input() -> None:
    app = typer.Typer()
    app.command()(power_command)
    runner = CliRunner()
    args = [
        "--size",
        "150",
        "--baseline-rate",
        "0.8",
        "--discordance",
        "0.1",
        "--tolerance",
        "0.05",
        "--draws",
        "30",
        "--step",
        "0.1",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "Synthetic paired binary slice" in result.output
    assert "grid" in result.output
    assert "Minimum detectable drop" in result.output
    invalid = runner.invoke(app, [*args, "--discordance", "1"])
    assert invalid.exit_code == 2
    assert "discordance" in invalid.output
