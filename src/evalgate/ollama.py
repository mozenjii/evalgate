"""Run the Ollama provider against a trusted, cloud-disabled daemon on loopback.

The provider builds the exact request used for cassette identity. The run layer owns
budgeting, replay, and redaction, and calls this module only for authorized live misses.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from typing import cast

from pydantic import JsonValue

__all__ = ["Ollama", "OllamaError"]

_LOCAL_NAME = re.compile(
    r"(?:[a-z0-9][a-z0-9_-]*/)?[a-z0-9][a-z0-9_.-]*(?::[a-z0-9][a-z0-9_.-]*)?",
    re.IGNORECASE | re.ASCII,
)
_RESPONSE_LIMIT = 4 * 1024 * 1024


class OllamaError(Exception):
    """An Ollama request was refused or failed without exposing its content."""


@dataclass(frozen=True)
class Ollama:
    """Generate JSON with an installed local Ollama model.

    Even loopback Ollama can broker remote inference. Live calls require a trusted
    daemon that confirms cloud is disabled and a model with no upstream configured.
    """

    model: str
    port: int = 11434

    @property
    def name(self) -> str:
        """Identify this transport independently of its loopback port."""
        return "ollama"

    def __post_init__(self) -> None:
        """Reject remote model references and invalid loopback ports."""
        if (
            len(self.model) > 255
            or not _LOCAL_NAME.fullmatch(self.model)
            or self.model.lower().endswith((":cloud", "-cloud"))
        ):
            raise OllamaError("Ollama requires an installed local model name")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise OllamaError("Ollama requires a loopback port between 1 and 65535")

    def request(self, value: JsonValue) -> dict[str, JsonValue]:
        """Build a generation request without connecting to Ollama."""
        return {
            "model": self.model,
            "prompt": value if isinstance(value, str) else _encode(value),
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }

    def validate_local(self) -> None:
        """Verify daemon cloud policy and model metadata before sending any prompt."""
        try:
            status = _exchange(self.port, "GET", "/api/status")
        except OllamaError:
            raise OllamaError("cannot verify Ollama cloud status on loopback") from None
        cloud = status.get("cloud")
        if not isinstance(cloud, dict) or cloud.get("disabled") is not True:
            raise OllamaError("disable Ollama cloud features with OLLAMA_NO_CLOUD=1 before running")

        model = _exchange(self.port, "POST", "/api/show", {"model": self.model})
        if model.get("remote_model") not in (None, "") or model.get("remote_host") not in (
            None,
            "",
        ):
            raise OllamaError("Ollama model is configured for remote inference")
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, list) or "completion" not in capabilities:
            raise OllamaError("Ollama model does not confirm completion support")

    def generate(self, request: dict[str, JsonValue]) -> JsonValue:
        """Send one request to the verified daemon and decode its JSON output."""
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or request != self.request(prompt):
            raise OllamaError("invalid Ollama generation request")
        self.validate_local()
        result = _exchange(self.port, "POST", "/api/generate", request)
        if result.get("remote_model") or result.get("remote_host"):
            raise OllamaError("Ollama returned a remote generation")
        if result.get("done") is not True or result.get("done_reason") == "length":
            raise OllamaError("Ollama generation did not complete")
        output = result.get("response")
        if not isinstance(output, str):
            raise OllamaError("Ollama response is missing JSON output")
        return _decode(output)


def _finite_float(token: str) -> float:
    """Reject both nonstandard float literals and numeric overflow."""
    result = float(token)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _encode(value: JsonValue) -> str:
    """Serialize a request consistently for caching and transport."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        raise OllamaError("Ollama input must be valid JSON") from None


def _decode(payload: str | bytes) -> JsonValue:
    """Decode only JSON values that can safely round-trip in a results file."""
    try:
        value = cast(
            "JsonValue",
            json.loads(payload, parse_float=_finite_float, parse_constant=_finite_float),
        )
        # The results writer requires UTF-8 even when the source used escapes.
        json.dumps(value, ensure_ascii=False).encode("utf-8")
    except (ValueError, RecursionError):
        raise OllamaError("Ollama returned invalid JSON") from None
    return value


def _exchange(
    port: int, method: str, path: str, body: dict[str, JsonValue] | None = None
) -> dict[str, JsonValue]:
    """Talk directly to loopback and discard HTTP error bodies."""
    connection = HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        encoded = None if body is None else _encode(body).encode("ascii")
        connection.request(method, path, body=encoded, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise OllamaError(f"Ollama returned HTTP {response.status} for {path}")
        payload = response.read(_RESPONSE_LIMIT + 1)
    except (OSError, HTTPException):
        raise OllamaError(f"Ollama request failed on loopback for {path}") from None
    finally:
        connection.close()
    if len(payload) > _RESPONSE_LIMIT:
        raise OllamaError("Ollama response exceeds the size limit")
    decoded = _decode(payload)
    if not isinstance(decoded, dict):
        raise OllamaError("Ollama returned an invalid response")
    return decoded
