"""Exercise the loopback provider without a live daemon."""

import json
from unittest.mock import MagicMock, patch

import pytest

from evalgate.ollama import Ollama, OllamaError


def test_request_preserves_text_and_serializes_structured_input_canonically() -> None:
    provider = Ollama("qwen2.5:0.5b")
    assert provider.name == "ollama"
    assert provider.request("Answer in JSON.") == {
        "model": "qwen2.5:0.5b",
        "prompt": "Answer in JSON.",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }
    assert provider.request({"z": [1, True], "a": "text"})["prompt"] == (
        '{"a":"text","z":[1,true]}'
    )


@pytest.mark.parametrize("model", ["", "https://host/model", "model:cloud", "model:70b-cloud"])
def test_invalid_or_cloud_model_is_rejected(model: str) -> None:
    with pytest.raises(OllamaError, match="local model"):
        Ollama(model)


@pytest.mark.parametrize("port", [False, 0, 65536])
def test_invalid_port_is_rejected(port: int) -> None:
    with pytest.raises(OllamaError, match="loopback port"):
        Ollama("local-model", port)


def _response(value: object, *, status: int = 200) -> MagicMock:
    response = MagicMock(status=status)
    response.read.return_value = json.dumps(value).encode()
    return response


def test_generate_checks_daemon_and_model_before_sending_prompt() -> None:
    provider = Ollama("qwen2.5:0.5b")
    request = provider.request("Answer in JSON.")
    with patch("evalgate.ollama.HTTPConnection") as connection:
        http = connection.return_value
        http.getresponse.side_effect = [
            _response({"cloud": {"disabled": True}}),
            _response({"capabilities": ["completion"]}),
            _response({"done": True, "done_reason": "stop", "response": '{"value":42}'}),
        ]
        assert provider.generate(request) == {"value": 42}
    assert {call.args[:2] for call in connection.call_args_list} == {("127.0.0.1", 11434)}
    calls = http.request.call_args_list
    assert [call.args[:2] for call in calls] == [
        ("GET", "/api/status"),
        ("POST", "/api/show"),
        ("POST", "/api/generate"),
    ]
    assert json.loads(calls[1].kwargs["body"]) == {"model": provider.model}
    assert json.loads(calls[2].kwargs["body"]) == request


@pytest.mark.parametrize("status", [{"cloud": {"disabled": False}}, {}, {"cloud": {"disabled": 1}}])
def test_cloud_policy_fails_closed_without_sending_prompt(status: object) -> None:
    provider = Ollama("qwen2.5:0.5b")
    with patch("evalgate.ollama.HTTPConnection") as connection:
        http = connection.return_value
        http.getresponse.return_value = _response(status)
        with pytest.raises(OllamaError, match="cloud"):
            provider.generate(provider.request("prompt"))
    assert [call.args[:2] for call in http.request.call_args_list] == [("GET", "/api/status")]


@pytest.mark.parametrize(
    "metadata", [{"remote_host": "https://host"}, {"remote_model": "upstream"}]
)
def test_remote_alias_is_refused_before_sending_prompt(metadata: object) -> None:
    provider = Ollama("qwen2.5:0.5b")
    with patch("evalgate.ollama.HTTPConnection") as connection:
        http = connection.return_value
        http.getresponse.side_effect = [
            _response({"cloud": {"disabled": True}}),
            _response(metadata),
        ]
        with pytest.raises(OllamaError, match="remote"):
            provider.generate(provider.request("prompt"))
    assert [call.args[:2] for call in http.request.call_args_list] == [
        ("GET", "/api/status"),
        ("POST", "/api/show"),
    ]


def test_missing_completion_capability_is_refused() -> None:
    with patch("evalgate.ollama.HTTPConnection") as connection:
        http = connection.return_value
        http.getresponse.side_effect = [
            _response({"cloud": {"disabled": True}}),
            _response({"capabilities": ["embedding"]}),
        ]
        with pytest.raises(OllamaError, match="completion support"):
            Ollama("local-model", 11435).validate_local()
    assert {call.args[:2] for call in connection.call_args_list} == {("127.0.0.1", 11435)}
    assert http.request.call_count == 2


@pytest.mark.parametrize(
    "response",
    [
        _response({"error": "private-data"}, status=307),
        _response({"done": True, "response": "private-data"}),
        _response({"done": True, "response": '{"value": Infinity}'}),
        _response({"done": True, "response": '{"value": 1e999}'}),
        _response({"done": True, "response": '"\\ud800"'}),
        _response({"done": False, "response": '{"value": "private-data"}'}),
        _response({"done": True, "done_reason": "length", "response": "{}"}),
    ],
)
def test_invalid_generation_is_rejected_without_echoing_payload(response: MagicMock) -> None:
    provider = Ollama("qwen2.5:0.5b")
    with patch("evalgate.ollama.HTTPConnection") as connection:
        http = connection.return_value
        http.getresponse.side_effect = [
            _response({"cloud": {"disabled": True}}),
            _response({"capabilities": ["completion"]}),
            response,
        ]
        with pytest.raises(OllamaError) as error:
            provider.generate(provider.request("private-data"))
    assert "private-data" not in str(error.value)
    assert http.request.call_count == 3


def test_transport_error_does_not_leak_lower_level_message() -> None:
    with patch("evalgate.ollama.HTTPConnection") as connection:
        connection.return_value.request.side_effect = TimeoutError("private-data")
        with pytest.raises(OllamaError, match="cannot verify Ollama cloud status") as error:
            Ollama("local-model").validate_local()
    assert "private-data" not in str(error.value)
    assert error.value.__suppress_context__
