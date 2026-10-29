"""The gate verifier refuses to destroy work that is already on disk."""

from pathlib import Path

import pytest

import verify_gates


def test_refuses_to_overwrite_an_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every probe writes a file and deletes it again. If a real file ever sat at a probe
    # path, a silent overwrite would destroy it, and the delete afterwards would hide that.
    monkeypatch.setattr(verify_gates, "ROOT", tmp_path)
    existing = tmp_path / "already_here.md"
    existing.write_text("work that must survive", encoding="utf-8")

    case = verify_gates.Probe(
        gate="docs", path=Path("already_here.md"), content="clobbered", fault="probe"
    )

    with pytest.raises(SystemExit):
        verify_gates.probe(case)

    assert existing.read_text(encoding="utf-8") == "work that must survive"


def test_every_probe_targets_a_declared_gate() -> None:
    # A probe aimed at a gate name that no longer exists would run nothing, pass, and
    # report a gate as connected when it was never tested.
    import check

    assert {probe.gate for probe in verify_gates.PROBES} <= {gate.name for gate in check.GATES}


def test_there_is_a_probe_for_every_gate() -> None:
    import check

    assert {probe.gate for probe in verify_gates.PROBES} == {gate.name for gate in check.GATES}
