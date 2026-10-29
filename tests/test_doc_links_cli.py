"""The doc-link gate reports through an exit code CI can act on."""

from pathlib import Path

import pytest

import doc_links_cli


def test_exits_zero_when_every_link_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "index.md").write_text("[contract](contract.md)", encoding="utf-8")
    (tmp_path / "contract.md").write_text("# Contract", encoding="utf-8")
    monkeypatch.setattr(doc_links_cli, "ROOT", tmp_path)

    assert doc_links_cli.main() == 0
    assert capsys.readouterr().out == ""


def test_exits_one_and_names_each_broken_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "index.md").write_text("[gone](gone.md)", encoding="utf-8")
    monkeypatch.setattr(doc_links_cli, "ROOT", tmp_path)

    assert doc_links_cli.main() == 1

    out = capsys.readouterr().out
    assert "1 broken link(s):" in out
    assert "index.md:1: gone.md" in out
