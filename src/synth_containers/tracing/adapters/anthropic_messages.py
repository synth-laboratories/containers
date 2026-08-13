"""Normalizer for Anthropic Messages requests, responses, and SSE streams."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..models.messages import MessagePartV5, PartType, ReasoningAvailability
from ..models.spans import UsageProvenance, UsageV5
from .base import NormalizedMessage, NormalizedProviderResult
from .sse import SSEDecoder, SSEEvent


NORMALIZER_NAME = "anthropic_messages"
NORMALIZER_VERSION = "1"
MESSAGES_PATH = "/v1/messages"


def normalize_anthropic_request(body: Mapping[str, Any]) -> list[NormalizedMessage]:
    messages: list[NormalizedMessage] = []
    system = body.get("system")
    if system is not None:
        messages.append(
            NormalizedMessage(
                role="system",
                parts=_anthropic_parts(system, prefix="ar-system"),
            )
        )
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        return messages
    for index, item in enumerate(raw_messages):
        if not isinstance(item, Mapping):
            messages.append(
                NormalizedMessage(
                    role="user",
                    parts=[
                        MessagePartV5(
                            part_id=f"ar-{index}-0",
                            type=PartType.UNSUPPORTED,
                            raw_kind=type(item).__name__,
                            text=str(item),
                        )
                    ],
                )
            )
            continue
        entry = _mapping(item)
        messages.append(
            NormalizedMessage(
                role=str(entry.get("role") or "user"),
                parts=_anthropic_parts(entry.get("content"), prefix=f"ar-{index}"),
            )
        )
    return messages


def _mapping(value: Any) -> Mapping[str, Any]:
    """A foreign payload section, or an empty one when absent or the wrong shape."""
    return value if isinstance(value, Mapping) else {}


def normalize_anthropic_response(body: Mapping[str, Any]) -> NormalizedProviderResult:
    content = body.get("content")
    message = NormalizedMessage(
        role=str(body.get("role") or "assistant"),
        parts=_anthropic_parts(content, prefix="as"),
        finish_reason=str(body.get("stop_reason") or "") or None,
    )
    usage_payload = body.get("usage")
    return NormalizedProviderResult(
        messages=[message],
        usage=usage_from_anthropic(
            usage_payload if isinstance(usage_payload, Mapping) else None
        ),
        provider_ids={
            "message_id": str(body["id"]) for _ in (0,) if body.get("id") is not None
        },
        terminal_observed=True,
    )


def usage_from_anthropic(payload: Mapping[str, Any] | None) -> UsageV5:
    if payload is None:
        return UsageV5(
            provenance=UsageProvenance.UNAVAILABLE,
            requests=1,
            unavailable_fields=("prompt_tokens", "completion_tokens", "total_tokens"),
        )
    prompt = _int(payload.get("input_tokens"))
    completion = _int(payload.get("output_tokens"))
    cached = _int(payload.get("cache_read_input_tokens"))
    cache_write = _int(payload.get("cache_creation_input_tokens"))
    return UsageV5(
        provenance=UsageProvenance.OBSERVED_PROVIDER,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        cache_write_tokens=cache_write,
        total_tokens=(
            int(prompt or 0) + int(completion or 0)
            if prompt is not None or completion is not None
            else None
        ),
        requests=1,
        unavailable_fields=tuple(
            name
            for name, value in (
                ("prompt_tokens", prompt),
                ("completion_tokens", completion),
            )
            if value is None
        ),
    )


class AnthropicStreamAssembler:
    """Enforce documented message/content-block event order while normalizing."""

    def __init__(self) -> None:
        self.decoder = SSEDecoder()
        self.events: list[dict[str, Any]] = []
        self.diagnostics: list[str] = []
        self.state = "before_message"
        self.blocks: dict[int, dict[str, Any]] = {}
        self.usage_payload: dict[str, Any] = {}
        self.provider_ids: dict[str, str] = {}

    def feed(self, chunk: bytes) -> None:
        for event in self.decoder.feed(chunk):
            self._event(event)

    def _event(self, event: SSEEvent) -> None:
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError:
            self.diagnostics.append("undecodable Anthropic SSE event")
            return
        if not isinstance(payload, Mapping):
            return
        item = dict(payload)
        kind = str(item.get("type") or event.event or "")
        item["type"] = kind
        self.events.append(item)
        if kind == "message_start":
            if self.state != "before_message":
                self.diagnostics.append("message_start occurred out of order")
            self.state = "in_message"
            message = item.get("message")
            if isinstance(message, Mapping):
                if message.get("id") is not None:
                    self.provider_ids["message_id"] = str(message["id"])
                if isinstance(message.get("usage"), Mapping):
                    self.usage_payload.update(message["usage"])
        elif kind == "content_block_start":
            if self.state != "in_message":
                self.diagnostics.append("content_block_start occurred outside a message")
            index = int(item.get("index") or 0)
            block = item.get("content_block")
            self.blocks[index] = dict(block) if isinstance(block, Mapping) else {"type": "unknown"}
        elif kind == "content_block_delta":
            index = int(item.get("index") or 0)
            block = self.blocks.setdefault(index, {"type": "unknown"})
            delta = item.get("delta")
            if not isinstance(delta, Mapping):
                return
            delta_type = str(delta.get("type") or "")
            if delta_type == "text_delta":
                block["text"] = str(block.get("text") or "") + str(delta.get("text") or "")
            elif delta_type == "thinking_delta":
                block["type"] = "thinking"
                block["thinking"] = str(block.get("thinking") or "") + str(
                    delta.get("thinking") or ""
                )
            elif delta_type == "signature_delta":
                # Anthropic emits the signature immediately before block stop.
                block["signature"] = str(block.get("signature") or "") + str(
                    delta.get("signature") or ""
                )
            elif delta_type == "input_json_delta":
                block["type"] = "tool_use"
                block["_partial_json"] = str(block.get("_partial_json") or "") + str(
                    delta.get("partial_json") or ""
                )
            else:
                block.setdefault("_unknown_deltas", []).append(dict(delta))
        elif kind == "content_block_stop":
            index = int(item.get("index") or 0)
            block = self.blocks.get(index)
            if block and "_partial_json" in block:
                raw = str(block.pop("_partial_json"))
                try:
                    block["input"] = json.loads(raw)
                except json.JSONDecodeError:
                    block["input_json"] = raw
                    self.diagnostics.append(
                        f"tool input_json_delta for block {index} did not form valid JSON"
                    )
        elif kind == "message_delta":
            if isinstance(item.get("usage"), Mapping):
                self.usage_payload.update(item["usage"])
            delta = item.get("delta")
            if isinstance(delta, Mapping) and delta.get("stop_reason") is not None:
                self.provider_ids["stop_reason"] = str(delta["stop_reason"])
        elif kind == "message_stop":
            if self.state != "in_message":
                self.diagnostics.append("message_stop occurred out of order")
            self.state = "stopped"
        elif kind == "error":
            self.state = "stopped"
        elif kind not in {"ping"}:
            self.diagnostics.append(f"unsupported Anthropic stream event {kind!r} preserved raw")

    def finish(self) -> NormalizedProviderResult:
        for event in self.decoder.finish():
            self._event(event)
        content = [self.blocks[index] for index in sorted(self.blocks)]
        message = NormalizedMessage(
            role="assistant",
            parts=_anthropic_parts(content, prefix="ass"),
            finish_reason=self.provider_ids.get("stop_reason"),
        )
        if self.state != "stopped":
            self.diagnostics.append("Anthropic stream ended without message_stop")
        return NormalizedProviderResult(
            messages=[message] if message.parts else [],
            usage=usage_from_anthropic(self.usage_payload or None),
            diagnostics=self.diagnostics,
            provider_ids=self.provider_ids,
            terminal_observed=self.state == "stopped",
            raw_events=self.events,
        )


class AnthropicMessagesAdapter:
    name: str = NORMALIZER_NAME
    version: str = NORMALIZER_VERSION
    routes: tuple[str, ...] = (MESSAGES_PATH,)

    def normalize_request(self, body: Mapping[str, Any]) -> list[NormalizedMessage]:
        return normalize_anthropic_request(body)

    def normalize_unary(self, body: Mapping[str, Any]) -> NormalizedProviderResult:
        return normalize_anthropic_response(body)

    def new_stream(self) -> AnthropicStreamAssembler:
        return AnthropicStreamAssembler()

    def usage(self, payload: Mapping[str, Any] | None) -> UsageV5:
        return usage_from_anthropic(payload)


def _anthropic_parts(value: Any, *, prefix: str) -> list[MessagePartV5]:
    if isinstance(value, str):
        return [MessagePartV5(part_id=f"{prefix}-0", type=PartType.TEXT, text=value)]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        return [
            MessagePartV5(
                part_id=f"{prefix}-0",
                type=PartType.UNSUPPORTED,
                raw_kind=type(value).__name__,
                text=str(value),
            )
        ]
    parts: list[MessagePartV5] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.UNSUPPORTED,
                    raw_kind=type(item).__name__,
                    text=str(item),
                )
            )
            continue
        kind = str(item.get("type") or "")
        if kind == "text":
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.TEXT,
                    text=str(item.get("text") or ""),
                )
            )
        elif kind in {"thinking", "redacted_thinking"}:
            redacted = kind == "redacted_thinking"
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.REASONING,
                    text=None if redacted else str(item.get("thinking") or ""),
                    reasoning_availability=(
                        ReasoningAvailability.REDACTED
                        if redacted
                        else ReasoningAvailability.CAPTURED
                    ),
                    structured={
                        key: value
                        for key, value in item.items()
                        if key not in {"thinking", "data"}
                    },
                )
            )
        elif kind == "tool_use":
            arguments = item.get("input")
            if arguments is None:
                arguments = item.get("input_json") or {}
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.TOOL_CALL,
                    tool_call_id=str(item.get("id") or ""),
                    tool_name=str(item.get("name") or ""),
                    arguments_json=_json_text(arguments),
                )
            )
        elif kind == "tool_result":
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.TOOL_RESULT,
                    tool_call_id=str(item.get("tool_use_id") or ""),
                    text=_content_text(item.get("content")),
                    is_error=bool(item.get("is_error")),
                )
            )
        elif kind in {"image", "document"}:
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.MEDIA,
                    media_type=kind,
                    structured=dict(item),
                )
            )
        else:
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.UNSUPPORTED,
                    raw_kind=kind or "unknown",
                    structured=dict(item),
                )
            )
    return parts


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "AnthropicMessagesAdapter",
    "AnthropicStreamAssembler",
    "MESSAGES_PATH",
    "NORMALIZER_NAME",
    "NORMALIZER_VERSION",
    "normalize_anthropic_request",
    "normalize_anthropic_response",
    "usage_from_anthropic",
]
