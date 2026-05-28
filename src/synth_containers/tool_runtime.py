from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .serde import JsonDataclassMixin


class ToolRuntimeKind(StrEnum):
    PROVIDER_NATIVE_TOOLS = "provider_native_tools"
    RESPONSES_TOOLS = "responses_tools"
    CODEX_SESSION_NATIVE = "codex_session_native"
    NONE = "none"

    @classmethod
    def parse(cls, value: Any, *, default: "ToolRuntimeKind | None" = None) -> "ToolRuntimeKind":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.NONE
        aliases = {
            "provider_native_tools": cls.PROVIDER_NATIVE_TOOLS,
            "provider_tools": cls.PROVIDER_NATIVE_TOOLS,
            "openai_chat": cls.PROVIDER_NATIVE_TOOLS,
            "responses_tools": cls.RESPONSES_TOOLS,
            "openai_responses": cls.RESPONSES_TOOLS,
            "responses": cls.RESPONSES_TOOLS,
            "codex_session_native": cls.CODEX_SESSION_NATIVE,
            "codex": cls.CODEX_SESSION_NATIVE,
            "none": cls.NONE,
        }
        if text not in aliases:
            raise ValueError(f"unsupported tool runtime kind: {value!r}")
        return aliases[text]


class ToolCallSchemaKind(StrEnum):
    OPENAI_CHAT_FUNCTIONS = "openai_chat_functions"
    OPENAI_RESPONSES_TOOLS = "openai_responses_tools"
    CODEX_SESSION_EVENTS = "codex_session_events"
    NONE = "none"

    @classmethod
    def parse(cls, value: Any, *, default: "ToolCallSchemaKind | None" = None) -> "ToolCallSchemaKind":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.NONE
        aliases = {
            "openai_chat_functions": cls.OPENAI_CHAT_FUNCTIONS,
            "chat_functions": cls.OPENAI_CHAT_FUNCTIONS,
            "provider_native_tools": cls.OPENAI_CHAT_FUNCTIONS,
            "openai_responses_tools": cls.OPENAI_RESPONSES_TOOLS,
            "responses_tools": cls.OPENAI_RESPONSES_TOOLS,
            "responses": cls.OPENAI_RESPONSES_TOOLS,
            "codex_session_events": cls.CODEX_SESSION_EVENTS,
            "codex": cls.CODEX_SESSION_EVENTS,
            "none": cls.NONE,
        }
        if text not in aliases:
            raise ValueError(f"unsupported tool call schema kind: {value!r}")
        return aliases[text]


class ToolOutputMode(StrEnum):
    TOOL_REQUIRED = "tool_required"
    JSON_ONLY = "json_only"
    TEXT_ONLY = "text_only"
    NONE = "none"

    @classmethod
    def parse(cls, value: Any, *, default: "ToolOutputMode | None" = None) -> "ToolOutputMode":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return default or cls.NONE
        aliases = {
            "tool_required": cls.TOOL_REQUIRED,
            "json_only": cls.JSON_ONLY,
            "text_only": cls.TEXT_ONLY,
            "none": cls.NONE,
        }
        if text not in aliases:
            raise ValueError(f"unsupported tool output mode: {value!r}")
        return aliases[text]


@dataclass(frozen=True, slots=True)
class ToolRuntimeCapabilities(JsonDataclassMixin):
    runtime_kind: ToolRuntimeKind | str = ToolRuntimeKind.NONE
    schema_kind: ToolCallSchemaKind | str = ToolCallSchemaKind.NONE
    output_mode: ToolOutputMode | str = ToolOutputMode.NONE
    supports_parallel_tool_calls: bool = False
    supports_streaming_events: bool = False
    supports_mcp: bool = False
    supports_stateful_session: bool = False
    requires_stateful_session: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_runtime_kind(self) -> ToolRuntimeKind:
        return ToolRuntimeKind.parse(self.runtime_kind)

    def normalized_schema_kind(self) -> ToolCallSchemaKind:
        return ToolCallSchemaKind.parse(self.schema_kind)

    def normalized_output_mode(self) -> ToolOutputMode:
        return ToolOutputMode.parse(self.output_mode)

    def validate(self, *, api_family: Any | None = None) -> None:
        runtime_kind = self.normalized_runtime_kind()
        schema_kind = self.normalized_schema_kind()
        output_mode = self.normalized_output_mode()
        normalized_api_family = str(api_family or "").strip().lower().replace("-", "_")
        if runtime_kind is ToolRuntimeKind.PROVIDER_NATIVE_TOOLS and schema_kind is not ToolCallSchemaKind.OPENAI_CHAT_FUNCTIONS:
            raise ValueError("provider-native tool runtime requires openai_chat_functions schema")
        if runtime_kind is ToolRuntimeKind.RESPONSES_TOOLS and schema_kind is not ToolCallSchemaKind.OPENAI_RESPONSES_TOOLS:
            raise ValueError("responses tool runtime requires openai_responses_tools schema")
        if runtime_kind is ToolRuntimeKind.CODEX_SESSION_NATIVE and schema_kind is not ToolCallSchemaKind.CODEX_SESSION_EVENTS:
            raise ValueError("codex session runtime requires codex_session_events schema")
        if runtime_kind is ToolRuntimeKind.NONE and (
            schema_kind is not ToolCallSchemaKind.NONE or output_mode is not ToolOutputMode.NONE
        ):
            raise ValueError("tool runtime kind 'none' requires schema_kind='none' and output_mode='none'")
        if output_mode is ToolOutputMode.TOOL_REQUIRED and runtime_kind is ToolRuntimeKind.NONE:
            raise ValueError("tool_required output mode requires a non-empty tool runtime")
        if self.requires_stateful_session and not self.supports_stateful_session:
            raise ValueError("requires_stateful_session=True requires supports_stateful_session=True")
        if self.requires_stateful_session and runtime_kind is not ToolRuntimeKind.CODEX_SESSION_NATIVE:
            raise ValueError("required stateful sessions are only valid for codex session runtimes")
        if normalized_api_family:
            if runtime_kind is ToolRuntimeKind.PROVIDER_NATIVE_TOOLS and normalized_api_family == "responses":
                raise ValueError("provider-native tools are incompatible with responses api family")
            if runtime_kind is ToolRuntimeKind.RESPONSES_TOOLS and normalized_api_family == "chat_completions":
                raise ValueError("responses tools are incompatible with chat completions api family")
        if output_mode is ToolOutputMode.TOOL_REQUIRED and schema_kind is ToolCallSchemaKind.NONE:
            raise ValueError("tool_required output mode requires a non-empty tool schema")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "runtime_kind": self.normalized_runtime_kind().value,
            "schema_kind": self.normalized_schema_kind().value,
            "output_mode": self.normalized_output_mode().value,
            "supports_parallel_tool_calls": bool(self.supports_parallel_tool_calls),
            "supports_streaming_events": bool(self.supports_streaming_events),
            "supports_mcp": bool(self.supports_mcp),
            "supports_stateful_session": bool(self.supports_stateful_session),
            "requires_stateful_session": bool(self.requires_stateful_session),
            "metadata": dict(self.metadata),
        }
