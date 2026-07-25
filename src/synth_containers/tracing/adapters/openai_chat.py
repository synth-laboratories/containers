"""Normalizer for OpenAI-compatible Chat Completions traffic.

Turns raw request bodies, unary response bodies, and SSE frame sequences into typed
V5 message parts plus observed usage. Every normalization reports what it could not
express instead of dropping it: unknown roles and content shapes survive as
``unsupported`` parts with conversion diagnostics.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..models.messages import (
    MessagePartV5,
    MessageRole,
    PartType,
    ReasoningAvailability,
)
from ..models.spans import UsageProvenance, UsageV5
from .base import NormalizedMessage, NormalizedProviderResult
from .sse import SSEDecoder


NORMALIZER_NAME = "openai_chat_completions"
NORMALIZER_VERSION = "1"

_KNOWN_ROLES = {role.value for role in MessageRole}


def _part(part_id: str, **kwargs: Any) -> MessagePartV5:
    return MessagePartV5(part_id=part_id, **kwargs)


def normalize_request_messages(body: Mapping[str, Any]) -> list[NormalizedMessage]:
    """Convert a chat request's ``messages`` array into typed message records."""

    messages: list[NormalizedMessage] = []
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        return messages
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            messages.append(
                NormalizedMessage(
                    role=MessageRole.USER.value,
                    parts=[
                        _part(
                            f"p{index}-0",
                            type=PartType.UNSUPPORTED,
                            raw_kind=type(raw).__name__,
                            text=str(raw),
                        )
                    ],
                    diagnostics=[f"message[{index}] was not an object"],
                )
            )
            continue
        messages.append(_normalize_one_message(_mapping(raw), prefix=f"p{index}"))
    return messages


def _mapping(value: Any) -> Mapping[str, Any]:
    """A foreign payload section, or an empty one when absent or the wrong shape."""
    return value if isinstance(value, Mapping) else {}


def _normalize_one_message(raw: Mapping[str, Any], *, prefix: str) -> NormalizedMessage:
    role = str(raw.get("role") or "user")
    diagnostics: list[str] = []
    if role not in _KNOWN_ROLES:
        diagnostics.append(f"unknown role {role!r} preserved verbatim")
    parts: list[MessagePartV5] = []
    counter = 0

    content = raw.get("content")
    if isinstance(content, str) and content:
        parts.append(_part(f"{prefix}-{counter}", type=PartType.TEXT, text=content))
        counter += 1
    elif isinstance(content, list):
        for item in content:
            if not isinstance(item, Mapping):
                parts.append(
                    _part(
                        f"{prefix}-{counter}",
                        type=PartType.UNSUPPORTED,
                        raw_kind=type(item).__name__,
                        text=str(item),
                    )
                )
                counter += 1
                continue
            item_type = str(item.get("type") or "")
            if item_type == "text":
                parts.append(
                    _part(
                        f"{prefix}-{counter}", type=PartType.TEXT, text=str(item.get("text") or "")
                    )
                )
            elif item_type in {"thinking", "reasoning"}:
                parts.append(
                    _part(
                        f"{prefix}-{counter}",
                        type=PartType.REASONING,
                        text=str(item.get("thinking") or item.get("text") or ""),
                        reasoning_availability=ReasoningAvailability.CAPTURED,
                    )
                )
            elif item_type in {"image_url", "input_image"}:
                parts.append(
                    _part(
                        f"{prefix}-{counter}",
                        type=PartType.MEDIA,
                        media_type="image",
                        structured=dict(item),
                    )
                )
            else:
                parts.append(
                    _part(
                        f"{prefix}-{counter}",
                        type=PartType.UNSUPPORTED,
                        raw_kind=item_type or "unknown",
                        structured=dict(item),
                    )
                )
                diagnostics.append(f"content part {item_type!r} preserved as unsupported")
            counter += 1

    reasoning = raw.get("reasoning_content") or raw.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        parts.append(
            _part(
                f"{prefix}-{counter}",
                type=PartType.REASONING,
                text=reasoning,
                reasoning_availability=ReasoningAvailability.CAPTURED,
            )
        )
        counter += 1

    for call in list(raw.get("tool_calls") or []):
        if not isinstance(call, Mapping):
            continue
        function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
        parts.append(
            _part(
                f"{prefix}-{counter}",
                type=PartType.TOOL_CALL,
                tool_call_id=str(call.get("id") or ""),
                tool_name=str(function.get("name") or call.get("name") or ""),
                arguments_json=_arguments_json(
                    function.get("arguments", call.get("arguments", "{}"))
                ),
            )
        )
        counter += 1

    tool_call_id = str(raw.get("tool_call_id") or "") or None
    if role == MessageRole.TOOL.value:
        parts.append(
            _part(
                f"{prefix}-{counter}",
                type=PartType.TOOL_RESULT,
                tool_call_id=tool_call_id or "",
                text=content if isinstance(content, str) else json.dumps(content, sort_keys=True),
                is_error=False,
            )
        )
        counter += 1

    if not parts:
        parts.append(_part(f"{prefix}-0", type=PartType.TEXT, text=""))
    return NormalizedMessage(
        role=role,
        parts=parts,
        diagnostics=diagnostics,
        tool_call_id=tool_call_id,
    )


def normalize_unary_response(body: Mapping[str, Any]) -> tuple[NormalizedMessage | None, list[str]]:
    """Convert a non-streaming chat response into one assistant message."""

    diagnostics: list[str] = []
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None, ["response contained no choices"]
    first = choices[0]
    if not isinstance(first, Mapping):
        return None, ["first choice was not an object"]
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None, ["first choice contained no message"]
    normalized = _normalize_one_message(message, prefix="r0")
    normalized.finish_reason = str(first.get("finish_reason") or "") or None
    normalized.diagnostics.extend(diagnostics)
    return normalized, normalized.diagnostics


def assemble_sse_frames(
    frames: list[str],
) -> tuple[NormalizedMessage | None, dict[str, Any] | None, list[str]]:
    """Reassemble streamed deltas into one assistant message plus observed usage."""

    diagnostics: list[str] = []
    text_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    saw_data = False

    for frame in frames:
        for line in frame.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                diagnostics.append("undecodable SSE data frame preserved in raw segments only")
                continue
            saw_data = True
            if isinstance(parsed.get("usage"), Mapping):
                usage = dict(parsed["usage"])
            choices = parsed.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, Mapping):
                continue
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            if isinstance(delta.get("content"), str):
                text_chunks.append(delta["content"])
            if isinstance(delta.get("reasoning_content"), str):
                reasoning_chunks.append(delta["reasoning_content"])
            for call in list(delta.get("tool_calls") or []):
                if not isinstance(call, Mapping):
                    continue
                index = int(call.get("index") or 0)
                slot = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = str(call["id"])
                function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                if function.get("name"):
                    slot["name"] = str(function["name"])
                if isinstance(function.get("arguments"), str):
                    slot["arguments"] += function["arguments"]

    if not saw_data:
        return None, None, ["no decodable SSE data frames were captured"]

    parts: list[MessagePartV5] = []
    counter = 0
    if reasoning_chunks:
        parts.append(
            _part(
                f"r0-{counter}",
                type=PartType.REASONING,
                text="".join(reasoning_chunks),
                reasoning_availability=ReasoningAvailability.CAPTURED,
            )
        )
        counter += 1
    if text_chunks:
        parts.append(_part(f"r0-{counter}", type=PartType.TEXT, text="".join(text_chunks)))
        counter += 1
    for index in sorted(tool_calls):
        slot = tool_calls[index]
        parts.append(
            _part(
                f"r0-{counter}",
                type=PartType.TOOL_CALL,
                tool_call_id=slot["id"],
                tool_name=slot["name"],
                arguments_json=slot["arguments"] or "{}",
            )
        )
        counter += 1
    if not parts:
        parts.append(_part("r0-0", type=PartType.TEXT, text=""))
    message = NormalizedMessage(
        role=MessageRole.ASSISTANT.value,
        parts=parts,
        diagnostics=diagnostics,
        finish_reason=finish_reason,
    )
    return message, usage, diagnostics


def usage_from_provider(payload: Mapping[str, Any] | None) -> UsageV5:
    """Build a usage record. Absent provider usage is ``unavailable``, never zero."""

    if not isinstance(payload, Mapping):
        return UsageV5(
            provenance=UsageProvenance.UNAVAILABLE,
            requests=1,
            unavailable_fields=("prompt_tokens", "completion_tokens", "total_tokens"),
        )
    prompt_details = payload.get("prompt_tokens_details")
    completion_details = payload.get("completion_tokens_details")
    cached = None
    reasoning = None
    if isinstance(prompt_details, Mapping):
        cached = _int_or_none(prompt_details.get("cached_tokens"))
    if isinstance(completion_details, Mapping):
        reasoning = _int_or_none(completion_details.get("reasoning_tokens"))
    missing = tuple(
        name
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if payload.get(name) is None
    )
    return UsageV5(
        provenance=UsageProvenance.OBSERVED_PROVIDER,
        prompt_tokens=_int_or_none(payload.get("prompt_tokens")),
        completion_tokens=_int_or_none(payload.get("completion_tokens")),
        reasoning_tokens=reasoning,
        cached_tokens=cached,
        total_tokens=_int_or_none(payload.get("total_tokens")),
        requests=1,
        unavailable_fields=missing,
    )


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _arguments_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps({"unsupported": str(value)}, sort_keys=True)


class _ChatStream:
    def __init__(self) -> None:
        self.decoder = SSEDecoder()
        self.frames: list[str] = []

    def feed(self, chunk: bytes) -> None:
        for event in self.decoder.feed(chunk):
            self.frames.append(f"data: {event.data}\n\n")

    def finish(self) -> NormalizedProviderResult:
        for event in self.decoder.finish():
            self.frames.append(f"data: {event.data}\n\n")
        message, usage_payload, diagnostics = assemble_sse_frames(self.frames)
        return NormalizedProviderResult(
            messages=[message] if message else [],
            usage=usage_from_provider(usage_payload),
            diagnostics=diagnostics,
            terminal_observed=any("[DONE]" in item for item in self.frames),
        )


class OpenAIChatAdapter:
    name: str = NORMALIZER_NAME
    version: str = NORMALIZER_VERSION
    routes: tuple[str, ...] = ("/v1/chat/completions",)

    def normalize_request(self, body: Mapping[str, Any]) -> list[NormalizedMessage]:
        return normalize_request_messages(body)

    def normalize_unary(self, body: Mapping[str, Any]) -> NormalizedProviderResult:
        message, diagnostics = normalize_unary_response(body)
        return NormalizedProviderResult(
            messages=[message] if message else [],
            usage=usage_from_provider(body.get("usage") if isinstance(body, Mapping) else None),
            diagnostics=diagnostics,
            provider_ids={
                "response_id": str(body.get("id"))
                for _ in (0,)
                if body.get("id") is not None
            },
            terminal_observed=True,
        )

    def new_stream(self) -> _ChatStream:
        return _ChatStream()

    def usage(self, payload: Mapping[str, Any] | None) -> UsageV5:
        return usage_from_provider(payload)


__all__ = [
    "NORMALIZER_NAME",
    "NORMALIZER_VERSION",
    "NormalizedMessage",
    "OpenAIChatAdapter",
    "assemble_sse_frames",
    "normalize_request_messages",
    "normalize_unary_response",
    "usage_from_provider",
]
