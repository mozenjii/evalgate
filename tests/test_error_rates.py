"""The script that publishes evalgate's error rates measures what it says it measures."""

from pathlib import Path

import pytest

from error_rates import (
    BinaryCase,
    Grid,
    binary_errors,
    conditional_decision,
    main,
    newcombe_interval,
    render,
    tango_decision,
)

TINY = Grid(
    binary_sizes=(10,),
    tolerances=(0.1,),
    discordances=(0.4,),
    newcombe_reps=50,
    continuous_sizes=(10,),
    moved_shares=(1.0,),
    spreads=(0.1,),
    continuous_reps=3,
    resamples=100,
    threshold_reps=1,
    family_sizes=(2,),
    family_reps=3,
)


def test_the_enumeration_covers_every_outcome() -> None:
    # A rule that passes everything must pass with probability one; anything less means the
    # enumeration dropped outcomes that matter.
    case = BinaryCase(n=40, tolerance=0.05, discordance=0.2)
    passes, fails = binary_errors(case, lambda b, c, n, tolerance: (True, False))
    assert passes == pytest.approx(1.0, abs=1e-9)
    assert fails == 0.0


def test_the_old_interval_fails_when_a_change_only_breaks_things() -> None:
    # The measurement that retired it: every disagreement favours the baseline, and the
    # conditional interval passes a slice that is truly worse by the tolerance most of the time.
    case = BinaryCase(n=100, tolerance=0.05, discordance=0.05)
    before, _ = binary_errors(case, conditional_decision)
    now, _ = binary_errors(case, tango_decision)
    assert before > 0.3
    assert now < 0.03


def test_tango_holds_its_level_on_an_ordinary_cell() -> None:
    passes, fails = binary_errors(BinaryCase(200, 0.05, 0.4), tango_decision)
    assert passes == pytest.approx(0.025, abs=0.005)
    assert fails == pytest.approx(0.025, abs=0.005)


def test_newcombe_brackets_the_observed_difference() -> None:
    low, high = newcombe_interval(both=50, fixed=3, broken=9, neither=38)
    assert low < (3 - 9) / 100 < high
    assert -1.0 <= low < high <= 1.0


def test_the_document_has_every_section() -> None:
    document = render(TINY)
    for heading in ("## Binary scorers", "## Continuous scorers", "## Many slices"):
        assert heading in document
    assert "Worst false pass" in document


def test_check_mode_fails_on_a_stale_file_and_passes_on_a_current_one(tmp_path: Path) -> None:
    output = tmp_path / "rates.md"
    assert main(["--check", "--output", str(output)], TINY) == 1
    assert main(["--output", str(output)], TINY) == 0
    assert main(["--check", "--output", str(output)], TINY) == 0
    output.write_text("edited by hand\n", encoding="utf-8")
    assert main(["--check", "--output", str(output)], TINY) == 1
