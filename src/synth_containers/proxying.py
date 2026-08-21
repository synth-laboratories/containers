from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

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
    def parse(cls, value: Any, *, default: "InferenceApiFamily | None" = None) -> "InferenceApiFamily":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.CHAT_COMPLETIONS
        aliases = {
            "chat": cls.CHAT_COMPLETIONS,
            "chat_completions": cls.CHAT_COMPLETIONS,
            "chat/completions": cls.CHAT_COMPLETIONS,
            "responses": cls.RESPONSES,
            "response": cls.RESPONSES,
        }
        if text not in aliases:
            raise ValueError(f"unsupported inference api family: {value!r}")
        return aliases[text]


class ToolCallStyle(StrEnum):
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    CODEX_SESSION_NATIVE = "codex_session_native"
    NONE = "none"

    @classmethod
    def parse(cls, value: Any, *, default: "ToolCallStyle | None" = None) -> "ToolCallStyle":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.NONE
        aliases = {
            "openai_chat": cls.OPENAI_CHAT,
            "chat": cls.OPENAI_CHAT,
            "openai_responses": cls.OPENAI_RESPONSES,
            "responses": cls.OPENAI_RESPONSES,
            "codex_session_native": cls.CODEX_SESSION_NATIVE,
            "codex": cls.CODEX_SESSION_NATIVE,
            "none": cls.NONE,
        }
        if text not in aliases:
            raise ValueError(f"unsupported tool call style: {value!r}")
        return aliases[text]


class ProxyMode(StrEnum):
    ALLOW_DIRECT = "allow_direct"
    PROXY_ONLY = "proxy_only"
    ASSERT_PROXY = "assert_proxy"

    @classmethod
    def parse(cls, value: Any, *, default: "ProxyMode | None" = None) -> "ProxyMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.ALLOW_DIRECT
        aliases = {
            "allow_direct": cls.ALLOW_DIRECT,
            "allow": cls.ALLOW_DIRECT,
            "proxy_only": cls.PROXY_ONLY,
            "assert_proxy": cls.ASSERT_PROXY,
        }
        if text not in aliases:
            raise ValueError(f"unsupported proxy mode: {value!r}")
        return aliases[text]


class CredentialMode(StrEnum):
    BYOK = "byok"
    WORKSHOP_PROXY = "workshop_proxy"
    PROXY = "proxy"

    @classmethod
    def parse(cls, value: Any, *, default: "CredentialMode | None" = None) -> "CredentialMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.BYOK
        aliases = {
            "byok": cls.BYOK,
            "bring_your_own_key": cls.BYOK,
            "workshop_proxy": cls.WORKSHOP_PROXY,
            "proxy": cls.WORKSHOP_PROXY,
        }
        if text not in aliases:
            raise ValueError(f"unsupported credential mode: {value!r}")
        return aliases[text]

    def is_proxied(self) -> bool:
        return self in {self.WORKSHOP_PROXY, self.PROXY}


WORKSHOP_API_KEY_SENTINEL = "workshop-proxy"


def sdk_base_url(url: str) -> str:
    text = str(url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/responses"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def resolve_proxied_inference_url(policy: Any) -> str:
    mode = CredentialMode.parse(
        getattr(policy, "credential_mode", None)
        if not isinstance(policy, Mapping)
        else policy.get("credential_mode")
    )
    if not mode.is_proxied():
        raise ValueError("resolve_proxied_inference_url requires workshop_proxy")
    raw = ""
    if isinstance(policy, Mapping):
        raw = str(policy.get("inference_url") or "").strip()
    else:
        raw = str(getattr(policy, "inference_url", "") or "").strip()
    if not raw:
        raise ValueError("inference_url is required when credential_mode=workshop_proxy")
    lower = raw.lower()
    if "api.openai.com" in lower:
        raise ValueError("inference_url must be the Workshop proxy, not api.openai.com")
    if "127.0.0.1" in lower or "localhost" in lower:
        raise ValueError("container inference_url must not point at loopback")
    return sdk_base_url(raw)


def workload_is_proxied(policy: Any | None = None) -> bool:
    mode = ""
    if isinstance(policy, Mapping):
        mode = str(policy.get("credential_mode") or "").strip().lower()
    elif policy is not None:
        mode = str(getattr(policy, "credential_mode", "") or "").strip().lower()
    env_mode = os.environ.get("WORKSHOP_CREDENTIAL_MODE", "").strip().lower()
    return mode in {"workshop_proxy", "proxy"} or env_mode in {"workshop_proxy", "proxy"} or bool(
        os.environ.get("WORKSHOP_CAPABILITY", "").strip()
    )


def workload_proxy_base(policy: Any | None = None) -> str | None:
    """SDK base URL when this workload is Workshop-proxied. None for standalone BYOK."""
    if not workload_is_proxied(policy):
        return None
    candidates: list[Any] = []
    if isinstance(policy, Mapping):
        candidates.extend([policy.get("inference_url"), policy.get("base_url")])
    elif policy is not None:
        candidates.extend(
            [getattr(policy, "inference_url", None), getattr(policy, "base_url", None)]
        )
    candidates.extend(
        [
            os.environ.get("WORKSHOP_OPENAI_BASE_URL"),
            os.environ.get("WORKSHOP_INFERENCE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("EVAL_LLM_ROUTE"),
        ]
    )
    raw = next((str(value).strip() for value in candidates if value and str(value).strip()), "")
    if not raw:
        raise ValueError("inference_url is required when credential_mode=workshop_proxy")
    lower = raw.lower()
    if "api.openai.com" in lower:
        raise ValueError("inference_url must be the Workshop proxy, not api.openai.com")
    if "127.0.0.1" in lower or "localhost" in lower:
        raise ValueError("container inference_url must not point at loopback")
    return sdk_base_url(raw)


class PolicyDisableReasoning(StrEnum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        default: "PolicyDisableReasoning | None" = None,
    ) -> "PolicyDisableReasoning":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.AUTO
        aliases = {
            "auto": cls.AUTO,
            "on": cls.ON,
            "true": cls.ON,
            "1": cls.ON,
            "off": cls.OFF,
            "false": cls.OFF,
            "0": cls.OFF,
        }
        if text not in aliases:
            raise ValueError(f"unsupported policy disable_reasoning value: {value!r}")
        return aliases[text]


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
    credential_mode: CredentialMode | str = CredentialMode.BYOK
    max_tokens: int | None = None
    disable_reasoning: PolicyDisableReasoning | str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    adapter_ref: str | None = None
    finetune_ref: str | None = None
    compute_pool: str | None = None
    tool_call_style: ToolCallStyle | str | None = None
    response_format_mode: str | None = None

    def normalized_api_family(self) -> InferenceApiFamily:
        return InferenceApiFamily.parse(self.api_family)

    def normalized_proxy_mode(self) -> ProxyMode:
        return ProxyMode.parse(self.proxy_mode)

    def normalized_credential_mode(self) -> CredentialMode:
        return CredentialMode.parse(self.credential_mode)

    def normalized_disable_reasoning(self) -> PolicyDisableReasoning:
        return PolicyDisableReasoning.parse(self.disable_reasoning)

    def normalized_tool_call_style(self) -> ToolCallStyle:
        return ToolCallStyle.parse(self.tool_call_style)

    @classmethod
    def from_policy_spec(cls, policy: Any) -> "InferenceTarget":
        if isinstance(policy, cls):
            return policy
        if isinstance(policy, Mapping):
            payload = dict(policy)
        else:
            model_dump = getattr(policy, "model_dump", None)
            if callable(model_dump):
                try:
                    payload = dict(model_dump(mode="python", exclude_none=True))
                except TypeError:
                    payload = dict(model_dump())
            else:
                names = (
                    "provider",
                    "model",
                    "api_family",
                    "inference_url",
                    "base_url",
                    "proxy_mode",
                    "credential_mode",
                    "max_tokens",
                    "disable_reasoning",
                    "config",
                    "tool_call_style",
                )
                payload = {key: getattr(policy, key) for key in names if hasattr(policy, key)}

        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        if not provider:
            raise ValueError("policy.provider must not be empty")
        if not model:
            raise ValueError("policy.model must not be empty")

        max_tokens = payload.get("max_tokens")
        config = payload.get("config")
        return cls(
            provider=provider,
            model=model,
            api_family=InferenceApiFamily.parse(payload.get("api_family")),
            inference_url=str(payload.get("inference_url") or ""),
            base_url=str(payload.get("base_url") or ""),
            proxy_mode=ProxyMode.parse(payload.get("proxy_mode")),
            credential_mode=CredentialMode.parse(payload.get("credential_mode")),
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            disable_reasoning=PolicyDisableReasoning.parse(payload.get("disable_reasoning")),
            config=dict(config) if isinstance(config, Mapping) else {},
            tool_call_style=ToolCallStyle.parse(payload.get("tool_call_style")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_family": self.normalized_api_family().value,
            "inference_url": self.inference_url,
            "base_url": self.base_url,
            "proxy_mode": self.normalized_proxy_mode().value,
            "credential_mode": self.normalized_credential_mode().value,
            "max_tokens": self.max_tokens,
            "disable_reasoning": self.normalized_disable_reasoning().value,
            "config": dict(self.config),
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
