"""Generated benchmark reports identify the simulated population and reproduce."""

from pathlib import Path

from benchmark import BenchmarkPlan, main, render


def test_seeded_benchmark_contains_detection_and_false_alarm_results() -> None:
    plan = BenchmarkPlan(sizes=(20, 80, 100), drops=(0.0, 0.15), draws=20, gate_slices=2)
    output = render(plan)
    assert output == render(plan)
    assert "Target slice detected" in output
    assert "False alarm" in output
    assert "20 |" in output
    assert "80 |" in output
    assert "seed 19" in output
    assert "xychart-beta" in output


def test_benchmark_write_and_check_refuse_stale_report(tmp_path: Path) -> None:
    report = tmp_path / "benchmark.md"
    plan = BenchmarkPlan(sizes=(20,), drops=(0.0,), draws=4)
    assert main(["--write", "--output", str(report)], plan=plan) == 0
    assert main(["--check", "--output", str(report)], plan=plan) == 0
    report.write_text("stale", encoding="utf-8")
    assert main(["--check", "--output", str(report)], plan=plan) == 1
