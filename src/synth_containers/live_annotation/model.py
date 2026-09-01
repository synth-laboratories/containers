"""Parent-side model calls on behalf of a sandboxed protocol.

The protocol process has no network and no credentials. It asks for a
judgment by emitting ``model_request``; the runner executes it here with the
container's provider configuration, under per-rollout ceilings, and hands the
parsed result back. Keys are read from the environment at call time and are
never persisted or echoed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResult:
    text: str
    parsed: dict[str, Any] | None
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_request_id: str | None = None


class ModelCaller(Protocol):
    """``(instructions, context, schema, max_output_tokens) -> ModelResult``."""

    model: str

    def __call__(
        self,
        *,
        instructions: str,
        context: str,
        schema: dict[str, Any] | None,
        max_output_tokens: int,
    ) -> ModelResult: ...


class ModelUnavailable(RuntimeError):
    """No usable provider configuration or credential; recorded as ``annotation.model.failed``."""


@dataclass
class ModelSettings:
    """Provider settings declared in a protocol's ``configuration.model`` block."""

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    max_calls: int = 20
    max_output_tokens: int = 800
    timeout_seconds: float = 120.0

    @classmethod
    def from_configuration(cls, configuration: dict[str, Any]) -> "ModelSettings | None":
        block = configuration.get("model")
        if not isinstance(block, dict):
            return None
        model = str(block.get("model") or block.get("name") or "").strip()
        if not model:
            return None
        return cls(
            model=model,
            base_url=str(block.get("base_url") or cls.base_url).rstrip("/"),
            api_key_env=str(block.get("api_key_env") or cls.api_key_env),
            max_calls=max(0, int(block.get("max_calls") or cls.max_calls)),
            max_output_tokens=max(16, int(block.get("max_output_tokens") or cls.max_output_tokens)),
            timeout_seconds=float(block.get("timeout_seconds") or cls.timeout_seconds),
        )

    def public(self) -> dict[str, Any]:
        """What may be persisted: never the key or the env var name."""

        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_calls": self.max_calls,
            "max_output_tokens": self.max_output_tokens,
        }


class OpenAICompatibleCaller:
    """Chat-completions caller with JSON-schema structured output when a schema is given."""

    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings
        self.model = settings.model

    def __call__(
        self,
        *,
        instructions: str,
        context: str,
        schema: dict[str, Any] | None,
        max_output_tokens: int,
    ) -> ModelResult:
        api_key = os.environ.get(self.settings.api_key_env, "").strip()
        if not api_key:
            raise ModelUnavailable("api_key_unavailable")
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": context},
            ],
            "temperature": 0,
            "max_tokens": int(max_output_tokens),
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "annotation", "schema": schema, "strict": False},
            }
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://usesynth.ai",
                "X-Title": "Synth Containers live annotation",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                headers = getattr(response, "headers", None)
                request_id = str(headers.get("x-request-id") or "") if headers is not None else ""
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"model_http_{exc.code}:{detail}") from exc
        body = json.loads(raw)
        choices = body.get("choices") or []
        message = (choices[0].get("message") if choices else None) or {}
        text = message.get("content")
        if isinstance(text, list):
            text = "".join(str(part.get("text") or "") for part in text if isinstance(part, dict))
        text = str(text or "")
        parsed: dict[str, Any] | None = None
        if schema is not None:
            try:
                candidate = json.loads(text)
                parsed = candidate if isinstance(candidate, dict) else None
            except json.JSONDecodeError:
                parsed = None
        usage = body.get("usage") or {}

        def _int(key: str) -> int | None:
            value = usage.get(key)
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        return ModelResult(
            text=text,
            parsed=parsed,
            model=str(body.get("model") or self.settings.model),
            input_tokens=_int("prompt_tokens"),
            output_tokens=_int("completion_tokens"),
            total_tokens=_int("total_tokens"),
            provider_request_id=request_id or None,
        )
