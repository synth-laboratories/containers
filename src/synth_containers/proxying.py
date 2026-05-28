from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .serde import JsonDataclassMixin


class InferenceApiFamily(StrEnum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"

    @property
    def endpoint_suffix(self) -> str:
        if self is InferenceApiFamily.RESPONSES:
            return "responses"
        return "chat/completions"

    @classmethod
    def parse(cls, value: Any, *, default: "InferenceApiFamily" = CHAT_COMPLETIONS) -> "InferenceApiFamily":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default
        aliases = {
            "chat": cls.CHAT_COMPLETIONS,
            "chat_completions": cls.CHAT_COMPLETIONS,
            "chat/completions": cls.CHAT_COMPLETIONS,
            "responses": cls.RESPONSES,
            "response": cls.RESPONSES,
        }
        try:
            return aliases[text]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"unsupported inference api family: {value!r}") from exc


class ToolCallStyle(StrEnum):
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    CODEX_SESSION_NATIVE = "codex_session_native"
    NONE = "none"

    @classmethod
    def parse(cls, value: Any, *, default: "ToolCallStyle" = NONE) -> "ToolCallStyle":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default
        aliases = {
            "openai_chat": cls.OPENAI_CHAT,
            "chat": cls.OPENAI_CHAT,
            "openai_responses": cls.OPENAI_RESPONSES,
            "responses": cls.OPENAI_RESPONSES,
            "codex_session_native": cls.CODEX_SESSION_NATIVE,
            "codex": cls.CODEX_SESSION_NATIVE,
            "none": cls.NONE,
        }
        try:
            return aliases[text]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"unsupported tool call style: {value!r}") from exc


class ProxyMode(StrEnum):
    ALLOW_DIRECT = "allow_direct"
    PROXY_ONLY = "proxy_only"
    ASSERT_PROXY = "assert_proxy"

    @classmethod
    def parse(cls, value: Any, *, default: "ProxyMode" = ALLOW_DIRECT) -> "ProxyMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default
        aliases = {
            "allow_direct": cls.ALLOW_DIRECT,
            "allow": cls.ALLOW_DIRECT,
            "proxy_only": cls.PROXY_ONLY,
            "assert_proxy": cls.ASSERT_PROXY,
        }
        try:
            return aliases[text]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"unsupported proxy mode: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class TraceIdentity(JsonDataclassMixin):
    trial_id: str
    correlation_id: str
    run_id: str | None = None
    candidate_id: str | None = None
    rollout_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.trial_id or "").strip():
            raise ValueError("trial_id must not be empty")
        if not str(self.correlation_id or "").strip():
            raise ValueError("correlation_id must not be empty")


@dataclass(frozen=True, slots=True)
class InferenceTarget(JsonDataclassMixin):
    provider: str = ""
    model: str = ""
    api_family: InferenceApiFamily | str | None = None
    inference_url: str = ""
    base_url: str = ""
    proxy_mode: ProxyMode | str = ProxyMode.ALLOW_DIRECT
    credential_mode: str = "byok"
    adapter_ref: str | None = None
    finetune_ref: str | None = None
    compute_pool: str | None = None
    tool_call_style: ToolCallStyle | str | None = None
    response_format_mode: str | None = None

    def normalized_api_family(self) -> InferenceApiFamily:
        return InferenceApiFamily.parse(self.api_family)

    def normalized_proxy_mode(self) -> ProxyMode:
        return ProxyMode.parse(self.proxy_mode)

    def normalized_tool_call_style(self) -> ToolCallStyle:
        return ToolCallStyle.parse(self.tool_call_style)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_family": self.normalized_api_family().value,
            "inference_url": self.inference_url,
            "base_url": self.base_url,
            "proxy_mode": self.normalized_proxy_mode().value,
            "credential_mode": self.credential_mode,
            "adapter_ref": self.adapter_ref,
            "finetune_ref": self.finetune_ref,
            "compute_pool": self.compute_pool,
            "tool_call_style": self.normalized_tool_call_style().value,
            "response_format_mode": self.response_format_mode,
        }


@dataclass(frozen=True, slots=True)
class ProxyResolution(JsonDataclassMixin):
    resolved_inference_url: str
    resolved_base_url: str
    api_family: InferenceApiFamily
    resolution_source: str
    proxy_mode: ProxyMode
    proxy_assertions_applied: bool
    proxy_assertions_passed: bool
    trace: TraceIdentity | None
    tool_call_style: ToolCallStyle
    codex_openai_base_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_synthesized(self) -> bool:
        return self.resolution_source == "synthesized"


@dataclass(frozen=True, slots=True)
class AgentRuntimeTarget(JsonDataclassMixin):
    runtime_family: str
    model: str
    inference_target: InferenceTarget | None = None
    reasoning_effort: str = "medium"
    approval_policy: str = "never"
    sandbox_profile: str | None = None
    auth_source: str | None = None
    provider_base_url_override: str | None = None
    tool_runtime_mode: str | None = None
    adapter_ref: str | None = None
    finetune_ref: str | None = None
    lora_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
