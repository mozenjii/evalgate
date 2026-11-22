"""Exercise model-run safety at the normal CLI entry point."""

import socket
from pathlib import Path

import pytest
from pydantic import JsonValue
from pytest import MonkeyPatch

from evalgate.ollama import Ollama
from evalgate.results import read_results
from test_cli import CONFIG, invoke


def test_model_budget_record_and_replay(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls: list[dict[str, JsonValue]] = []

    def generate(self: Ollama, request: dict[str, JsonValue]) -> JsonValue:
        calls.append(request)
        return {"text": "private"}

    def validate_local(self: Ollama) -> None:
        del self

    monkeypatch.setattr(Ollama, "generate", generate)
    monkeypatch.setattr(Ollama, "validate_local", validate_local)
    hook = tmp_path / "example_scrub.py"
    hook.write_text('def redact(value):\n    return {"text": "[redacted]"}\n', encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "run.jsonl"
    cassette = tmp_path / "cassette"
    args = [
        "run",
        "--config",
        str(CONFIG),
        "--out",
        str(out),
        "--label",
        "model",
        "--quiet",
        "--ollama-model",
        "local-test",
        "--cassette-dir",
        str(cassette),
        "--redact",
        "example_scrub:redact",
    ]
    cold = invoke(*args)
    assert cold.exit_code == 0, cold.output
    assert "120 item(s) not run" in cold.output
    recording = read_results(out)
    assert recording.items == ()
    assert recording.header is not None and recording.header.not_run == 120
    report = invoke("compare", str(out), str(out), "--config", str(CONFIG))
    assert report.exit_code == 0, report.output
    assert "120 items not run" in report.output
    out.unlink()
    dry = invoke(*args, "--dry-run", "--max-calls", "120")
    assert dry.exit_code == 0, dry.output
    assert "120" in dry.output
    assert not cassette.exists()
    assert not out.exists()
    blocked = invoke(*args, "--allow-local-model", "--max-calls", "0")
    assert blocked.exit_code == 3, blocked.output
    assert not calls
    assert not out.exists()

    live = invoke(*args, "--allow-local-model", "--max-calls", "120")
    assert live.exit_code == 0, live.output
    assert len(calls) == 120
    assert len(read_results(out).items) == 120
    assert "[redacted]" in out.read_text(encoding="utf-8")
    assert "private" not in out.read_text(encoding="utf-8")
    assert all("private" not in path.read_text(encoding="utf-8") for path in cassette.iterdir())
    recorded = read_results(out)
    replay = invoke(*args)
    assert replay.exit_code == 0, replay.output
    assert len(calls) == 120
    assert tuple(item.output for item in read_results(out).items) == tuple(
        item.output for item in recorded.items
    )


def test_failed_redaction_leaves_no_raw_output(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    def generate(self: Ollama, request: dict[str, JsonValue]) -> JsonValue:
        del self, request
        return {"text": "unscrubbed"}

    def validate_local(self: Ollama) -> None:
        del self

    monkeypatch.setattr(Ollama, "generate", generate)
    monkeypatch.setattr(Ollama, "validate_local", validate_local)
    hook = tmp_path / "failing_scrub.py"
    hook.write_text("def redact(value):\n    raise ValueError(value)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "run.jsonl"
    cassette = tmp_path / "cassette"
    result = invoke(
        "run",
        "--config",
        str(CONFIG),
        "--out",
        str(out),
        "--label",
        "model",
        "--ollama-model",
        "local-test",
        "--cassette-dir",
        str(cassette),
        "--allow-local-model",
        "--max-calls",
        "120",
        "--redact",
        "failing_scrub:redact",
    )
    assert result.exit_code == 3, result.output
    assert "could not prepare a redacted model output" in result.output
    assert "unscrubbed" not in result.output
    assert not out.exists()
    assert not cassette.exists()


def test_socket_connections_are_blocked_by_default() -> None:
    with socket.socket() as client, pytest.raises(AssertionError, match="network access"):
        client.connect(("127.0.0.1", 11434))


def test_public_example_replays_all_nine_recordings(tmp_path: Path) -> None:
    example = CONFIG.parents[1] / "ollama"
    out = tmp_path / "iris.jsonl"
    result = invoke(
        "run",
        "--config",
        str(example / "evalgate.toml"),
        "--out",
        str(out),
        "--label",
        "iris",
        "--ollama-model",
        "qwen2.5:0.5b",
        "--cassette-dir",
        str(example / "recordings"),
    )
    assert result.exit_code == 0, result.output
    replay = read_results(out)
    assert replay.header is not None and replay.header.not_run is None
    assert len(replay.items) == 9
    assert all(item.error is None for item in replay.items)
