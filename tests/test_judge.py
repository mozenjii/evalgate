"""Judge acceptance at the CLI, with deterministic responses and network blocked."""

import json
from pathlib import Path

from pydantic import JsonValue
from pytest import MonkeyPatch

from evalgate.ollama import Ollama
from evalgate.results import read_results
from test_cli import CONFIG, invoke, produce


def test_judge_enforces_budget_then_replays_comparable_verdicts(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = produce(tmp_path, "source")
    sample = read_results(source).items[:2]
    source.write_text(
        "".join(f"{item.model_dump_json(exclude_none=True)}\n" for item in sample),
        encoding="utf-8",
    )
    rubric = tmp_path / "matching.txt"
    rubric.write_text("Pass when the answer matches the reference.", encoding="utf-8")
    out = tmp_path / "graded.jsonl"
    cassette = tmp_path / "recordings"
    args = [
        "judge",
        str(source),
        "--config",
        str(CONFIG),
        "--rubric",
        str(rubric),
        "--ollama-model",
        "fixture",
        "--cassette-dir",
        str(cassette),
        "--out",
        str(out),
    ]
    calls: list[dict[str, JsonValue]] = []

    def generate(self: Ollama, request: dict[str, JsonValue]) -> JsonValue:
        calls.append(request)
        return {"pass": len(calls) == 1}

    def validate(self: Ollama) -> None:
        del self

    monkeypatch.setattr(Ollama, "generate", generate)
    monkeypatch.setattr(Ollama, "validate_local", validate)
    dry = invoke(*args, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert "up to 2" in dry.output
    denied = invoke(*args, "--allow-local-model", "--max-calls", "1")
    assert denied.exit_code == 3, denied.output
    assert calls == []
    assert not out.exists()

    live = invoke(*args, "--allow-local-model", "--max-calls", "2")
    assert live.exit_code == 0, live.output
    graded = read_results(out)
    assert len(calls) == 2
    assert graded.header is not None and graded.header.judgement_sha256 is not None
    assert [(item.output, item.input_sha256) for item in graded.items] == [
        (True, sample[0].input_sha256),
        (False, sample[1].input_sha256),
    ]
    assert all(
        type(json.loads(path.read_text(encoding="utf-8"))["output"]) is bool
        for path in cassette.glob("*.json")
    )

    replay = invoke(*args)
    assert replay.exit_code == 0, replay.output
    assert read_results(out).items == graded.items
    assert len(calls) == 2
    labels = invoke("judge-labels", str(out))
    assert labels.exit_code == 0, labels.output
    assert [json.loads(line) for line in labels.output.splitlines()] == [
        {
            "id": sample[0].id,
            "rubric": "matching",
            "judge": True,
            "judgement_sha256": graded.header.judgement_sha256,
        },
        {
            "id": sample[1].id,
            "rubric": "matching",
            "judge": False,
            "judgement_sha256": graded.header.judgement_sha256,
        },
    ]
    mixed = invoke("compare", str(source), str(out), "--config", str(CONFIG))
    assert mixed.exit_code == 3
    assert "different grading" in mixed.output


def test_invalid_judge_output_is_not_a_target_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    source = produce(tmp_path, "source")
    first = read_results(source).items[0]
    source.write_text(first.model_dump_json(exclude_none=True) + "\n", encoding="utf-8")
    rubric = tmp_path / "matching.txt"
    rubric.write_text("Pass when the answer matches the reference.", encoding="utf-8")
    out = tmp_path / "graded.jsonl"

    def generate(self: Ollama, request: dict[str, JsonValue]) -> JsonValue:
        del self, request
        return {"pass": "yes", "reason": "raw-response"}

    def validate(self: Ollama) -> None:
        del self

    monkeypatch.setattr(Ollama, "generate", generate)
    monkeypatch.setattr(Ollama, "validate_local", validate)
    result = invoke(
        "judge",
        str(source),
        "--config",
        str(CONFIG),
        "--rubric",
        str(rubric),
        "--ollama-model",
        "fixture",
        "--cassette-dir",
        str(tmp_path / "recordings"),
        "--out",
        str(out),
        "--allow-local-model",
        "--max-calls",
        "1",
    )
    assert result.exit_code == 3
    assert "invalid judge response" in result.output
    assert "raw-response" not in result.output
    assert not out.exists()
    assert not (tmp_path / "recordings").exists()


def test_demo_detects_the_planted_slice_with_known_grades(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    example = CONFIG.parents[1] / "judge"
    config = example / "evalgate.toml"
    args = [
        "--config",
        str(config),
        "--rubric",
        str(example / "policy-accuracy.txt"),
        "--ollama-model",
        "fixture",
        "--cassette-dir",
        str(tmp_path / "recordings"),
    ]

    def grade(self: Ollama, request: dict[str, JsonValue]) -> JsonValue:
        del self
        prompt = request["prompt"]
        assert isinstance(prompt, str)
        payload = json.loads(prompt.split("Input data:\n", maxsplit=1)[1])
        return {"pass": payload["candidate"] == payload["reference"]}

    def validate(self: Ollama) -> None:
        del self

    monkeypatch.setattr(Ollama, "generate", grade)
    monkeypatch.setattr(Ollama, "validate_local", validate)
    baseline = produce(tmp_path, "base", config=config)
    candidate = produce(tmp_path, "head", target="reply_target:candidate", config=config)
    grades = (tmp_path / "base-grades.jsonl", tmp_path / "head-grades.jsonl")
    for source, out in zip((baseline, candidate), grades, strict=True):
        result = invoke(
            "judge",
            str(source),
            *args,
            "--out",
            str(out),
            "--allow-local-model",
            "--max-calls",
            "16",
        )
        assert result.exit_code == 0, result.output
    report = invoke("compare", *(str(path) for path in grades), "--config", str(config))
    assert report.exit_code == 1, report.output
    assert any(
        "topic=returns" in line and line.endswith("regressed")
        for line in report.output.splitlines()
    )
