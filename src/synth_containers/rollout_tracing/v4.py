from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, cast

from synth_containers.serde import JsonDataclassMixin, jsonable


TRACE_SCHEMA_VERSION = 4
TRACE_SCHEMA_VERSION_NAME = "synth_rollout_trace_v4"

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True, frozen=True)
class TextPart(JsonDataclassMixin):
    """Visible text content."""

    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True, frozen=True)
class ReasoningPart(JsonDataclassMixin):
    """Model reasoning/thinking content when the provider exposes it."""

    content: str | None
    kind: str = "reasoning"
    type: Literal["reasoning"] = "reasoning"


@dataclass(slots=True, frozen=True)
class ToolCallPart(JsonDataclassMixin):
    """Assistant tool/function call with JSON arguments preserved as text."""

    id: str
    name: str
    arguments_json: str
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True, frozen=True)
class ToolResultPart(JsonDataclassMixin):
    """Tool/function result paired with a tool call id."""

    tool_call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True, frozen=True)
class UnsupportedPart(JsonDataclassMixin):
    """Placeholder for provider content that a runtime did not normalize."""

    kind: str
    detail: str = ""
    type: Literal["unsupported"] = "unsupported"


ContentPart = TextPart | ReasoningPart | ToolCallPart | ToolResultPart | UnsupportedPart


@dataclass(slots=True, frozen=True)
class CanonicalMessage(JsonDataclassMixin):
    """Provider-neutral message with typed content parts."""

    role: MessageRole
    parts: tuple[ContentPart, ...]
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def text(cls, role: MessageRole, text: str, *, name: str | None = None) -> "CanonicalMessage":
        return cls(role=role, parts=(TextPart(text=text),), name=name)

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: str,
        *,
        is_error: bool = False,
    ) -> "CanonicalMessage":
        return cls(
            role="tool",
            parts=(ToolResultPart(tool_call_id=tool_call_id, content=content, is_error=is_error),),
            tool_call_id=tool_call_id,
        )

    def get_text_content(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    def get_tool_calls(self) -> list[ToolCallPart]:
        return [part for part in self.parts if isinstance(part, ToolCallPart)]

    def to_chat_message(self) -> dict[str, Any]:
        """Render as a broadly OpenAI-compatible chat message for legacy consumers."""

        message: dict[str, Any] = {"role": self.role}
        text = self.get_text_content()
        if text or self.role != "assistant":
            message["content"] = text
        reasoning_parts = [part for part in self.parts if isinstance(part, ReasoningPart)]
        if reasoning_parts:
            message["reasoning_content"] = "\n".join(
                str(part.content or "") for part in reasoning_parts if part.content is not None
            )
        tool_calls = [
            {
                "id": part.id,
                "type": "function",
                "function": {"name": part.name, "arguments": part.arguments_json},
            }
            for part in self.get_tool_calls()
        ]
        if tool_calls:
            message["tool_calls"] = tool_calls
            message.setdefault("content", "")
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            message["name"] = self.name
        return message


@dataclass(slots=True, frozen=True)
class CanonicalUsage(JsonDataclassMixin):
    """Token accounting used by rollout traces."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def to_legacy_usage(self) -> dict[str, Any]:
        usage: dict[str, Any] = {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "total_tokens": self.total_tokens,
        }
        if self.cached_tokens:
            usage["prompt_tokens_details"] = {"cached_tokens": int(self.cached_tokens)}
        if self.reasoning_tokens:
            usage["completion_tokens_details"] = {"reasoning_tokens": int(self.reasoning_tokens)}
        return usage


@dataclass(slots=True, frozen=True)
class CanonicalRequest(JsonDataclassMixin):
    """Provider-neutral request for one model call."""

    messages: tuple[CanonicalMessage, ...]
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: tuple[str, ...] | None = None
    tools: tuple[dict[str, Any], ...] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    provider_hint: str | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    def to_legacy_request(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [message.to_chat_message() for message in self.messages],
            "model": self.model,
        }
        for key in ("temperature", "max_tokens", "top_p", "response_format", "provider_hint"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = jsonable(value)
        if self.stop is not None:
            payload["stop"] = list(self.stop)
        if self.tools is not None:
            payload["tools"] = [dict(tool) for tool in self.tools]
        if self.tool_choice is not None:
            payload["tool_choice"] = jsonable(self.tool_choice)
        return payload


@dataclass(slots=True, frozen=True)
class CanonicalChoice(JsonDataclassMixin):
    """Single model response choice."""

    index: int
    message: CanonicalMessage
    finish_reason: str | None = None


@dataclass(slots=True, frozen=True)
class CanonicalResponse(JsonDataclassMixin):
    """Provider-neutral response for one model call."""

    choices: tuple[CanonicalChoice, ...]
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    model: str = ""
    response_id: str | None = None
    created_at: float | None = None
    provider_hint: str | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    @property
    def first_choice(self) -> CanonicalChoice | None:
        return self.choices[0] if self.choices else None

    def to_legacy_response(self) -> dict[str, Any]:
        choice = self.first_choice
        message = (
            choice.message.to_chat_message()
            if choice is not None
            else {"role": "assistant", "content": ""}
        )
        payload: dict[str, Any] = {
            "message": message,
            "usage": self.usage.to_legacy_usage(),
            "finish_reason": choice.finish_reason if choice is not None else None,
            "model": self.model,
        }
        if isinstance(message.get("tool_calls"), list):
            payload["tool_calls"] = list(message["tool_calls"])
        if self.response_id is not None:
            payload["response_id"] = self.response_id
        if self.created_at is not None:
            payload["created_at"] = self.created_at
        if self.provider_hint is not None:
            payload["provider_hint"] = self.provider_hint
        return payload


@dataclass(slots=True, frozen=True)
class RolloutTraceSpanV4(JsonDataclassMixin):
    """One LLM call/span inside a rollout trace."""

    span_id: str
    call_index: int
    request: CanonicalRequest
    response: CanonicalResponse
    parent_span_id: str | None = None
    run_id: str | None = None
    api_format: str | None = None
    raw_request: dict[str, Any] | None = None
    raw_response: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        """Render as the legacy v3-style lm_call event expected by optimizers."""

        llm_request = self.request.to_legacy_request()
        llm_response = self.response.to_legacy_response()
        response_message = llm_response.get("message")
        canonical_messages = list(llm_request.get("messages") or [])
        if isinstance(response_message, dict):
            canonical_messages.append(response_message)
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for message in canonical_messages:
            if not isinstance(message, Mapping):
                continue
            if isinstance(message.get("tool_calls"), list):
                tool_calls.extend(dict(item) for item in message["tool_calls"] if isinstance(item, Mapping))
            if message.get("role") == "tool":
                tool_results.append(
                    {
                        "tool_call_id": message.get("tool_call_id"),
                        "content": message.get("content", ""),
                    }
                )
        event = {
            "type": "lm_call",
            "event_type": "lm_call",
            "sequence_index": int(self.call_index),
            "trace_id": self.run_id or self.span_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "llm_request": llm_request,
            "llm_response": llm_response,
            "api_format": self.api_format or self.request.provider_hint or "unknown",
            "raw_request": self.raw_request,
            "raw_response": self.raw_response,
            "canonical": {
                "messages": canonical_messages,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            },
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }
        payload = jsonable(event)
        if isinstance(payload, dict):
            return payload
        return {"value": payload}


@dataclass(slots=True, frozen=True)
class RolloutTraceV4(JsonDataclassMixin):
    """Top-level trace payload returned by container rollout trace routes."""

    rollout_id: str
    spans: tuple[RolloutTraceSpanV4, ...] = ()
    trace_correlation_id: str | None = None
    status: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION_NAME
    trace_schema_version: int = TRACE_SCHEMA_VERSION

    def event_history(self) -> list[dict[str, Any]]:
        return [span.to_event() for span in sorted(self.spans, key=lambda item: item.call_index)]

    def to_dict(self) -> dict[str, Any]:
        payload = JsonDataclassMixin.to_dict(self)
        payload["event_history"] = self.event_history()
        payload["span_count"] = len(self.spans)
        return payload


def chat_message_to_canonical(message: Mapping[str, Any]) -> CanonicalMessage:
    """Convert a chat-style message into the v4 canonical message shape."""

    role = str(message.get("role") or "user")
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"
    parts: list[ContentPart] = []
    content = message.get("content")
    if content not in (None, ""):
        parts.append(TextPart(text=str(content)))
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning not in (None, ""):
        parts.append(ReasoningPart(content=str(reasoning), kind="provider_reasoning"))
    for item in list(message.get("tool_calls") or []):
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else {}
        parts.append(
            ToolCallPart(
                id=str(item.get("id") or ""),
                name=str(function.get("name") or item.get("name") or ""),
                arguments_json=str(function.get("arguments") or item.get("arguments") or "{}"),
            )
        )
    if role == "tool":
        tool_call_id = str(message.get("tool_call_id") or "")
        parts.append(ToolResultPart(tool_call_id=tool_call_id, content=str(content or "")))
    if not parts:
        parts.append(TextPart(text=""))
    return CanonicalMessage(
        role=cast(MessageRole, role),
        parts=tuple(parts),
        tool_call_id=str(message.get("tool_call_id") or "") or None,
        name=str(message.get("name") or "") or None,
    )


def make_lm_span(
    *,
    span_id: str,
    call_index: int,
    request_messages: list[dict[str, Any]],
    response_message: dict[str, Any],
    model: str,
    raw_request: dict[str, Any] | None = None,
    raw_response: dict[str, Any] | None = None,
    usage: CanonicalUsage | None = None,
    finish_reason: str | None = None,
    run_id: str | None = None,
) -> RolloutTraceSpanV4:
    """Build a v4 rollout span from common chat request/response dictionaries."""

    request = CanonicalRequest(
        messages=tuple(chat_message_to_canonical(item) for item in request_messages),
        model=model,
        provider_hint="chat",
    )
    response = CanonicalResponse(
        choices=(
            CanonicalChoice(
                index=0,
                message=chat_message_to_canonical(response_message),
                finish_reason=finish_reason,
            ),
        ),
        usage=usage or CanonicalUsage(),
        model=model,
        provider_hint="chat",
    )
    return RolloutTraceSpanV4(
        span_id=span_id,
        call_index=call_index,
        request=request,
        response=response,
        run_id=run_id,
        api_format="chat",
        raw_request=raw_request,
        raw_response=raw_response,
    )
