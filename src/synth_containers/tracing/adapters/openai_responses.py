"""Normalizer for OpenAI Responses HTTP, SSE, compaction, and WebSocket events."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..models.messages import MessagePartV5, PartType, ReasoningAvailability
from ..models.spans import UsageProvenance, UsageV5
from ..models.tokens import (
    TokenCaptureProvenance,
    TokenCaptureV5,
    TokenSequenceRefV1,
)
from .base import NormalizedMessage, NormalizedProviderResult
from .sse import SSEDecoder, SSEEvent


NORMALIZER_NAME = "openai_responses"
NORMALIZER_VERSION = "1"
RESPONSES_PATH = "/v1/responses"
RESPONSES_COMPACT_PATH = "/v1/responses/compact"
RESPONSES_WEBSOCKET_URL = "wss://api.openai.com/v1/responses"


def normalize_responses_request(body: Mapping[str, Any]) -> list[NormalizedMessage]:
    source = body
    if body.get("type") == "response.create" and isinstance(
        body.get("response"),
        Mapping,
    ):
        source = body["response"]
    value = source.get("input")
    if isinstance(value, str):
        return [
            NormalizedMessage(
                role="user",
                parts=[MessagePartV5(part_id="rq-0", type=PartType.TEXT, text=value)],
            )
        ]
    if not isinstance(value, list):
        return []
    messages: list[NormalizedMessage] = []
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, Mapping):
            messages.append(_unsupported_message(raw_item, f"rq-{index}"))
            continue
        item = _mapping(raw_item)
        item_type = str(item.get("type") or "message")
        if item_type == "message":
            messages.append(
                NormalizedMessage(
                    role=str(item.get("role") or "user"),
                    parts=_content_parts(item.get("content"), prefix=f"rq-{index}"),
                )
            )
        elif item_type == "function_call_output":
            messages.append(
                NormalizedMessage(
                    role="tool",
                    parts=[
                        MessagePartV5(
                            part_id=f"rq-{index}-0",
                            type=PartType.TOOL_RESULT,
                            tool_call_id=str(item.get("call_id") or ""),
                            text=_text(item.get("output")),
                            is_error=False,
                        )
                    ],
                )
            )
        elif item_type in {"computer_call_output", "local_shell_call_output"}:
            messages.append(
                NormalizedMessage(
                    role="tool",
                    parts=[
                        MessagePartV5(
                            part_id=f"rq-{index}-0",
                            type=PartType.TOOL_RESULT,
                            tool_call_id=str(item.get("call_id") or ""),
                            structured=dict(item),
                        )
                    ],
                )
            )
        else:
            messages.append(_unsupported_message(dict(item), f"rq-{index}", raw_kind=item_type))
    return messages


def normalize_responses_response(body: Mapping[str, Any]) -> NormalizedProviderResult:
    output = body.get("output")
    messages: list[NormalizedMessage] = []
    diagnostics: list[str] = []
    if isinstance(output, list):
        for index, raw_item in enumerate(output):
            if not isinstance(raw_item, Mapping):
                messages.append(_unsupported_message(raw_item, f"rs-{index}"))
                continue
            item = _mapping(raw_item)
            item_type = str(item.get("type") or "")
            if item_type == "message":
                messages.append(
                    NormalizedMessage(
                        role=str(item.get("role") or "assistant"),
                        parts=_content_parts(item.get("content"), prefix=f"rs-{index}"),
                        finish_reason=str(body.get("status") or "") or None,
                    )
                )
            elif item_type in {"reasoning", "reasoning_summary"}:
                text = _text(item.get("summary") or item.get("content"))
                messages.append(
                    NormalizedMessage(
                        role="assistant",
                        parts=[
                            MessagePartV5(
                                part_id=f"rs-{index}-0",
                                type=PartType.REASONING,
                                text=text,
                                reasoning_availability=ReasoningAvailability.CAPTURED,
                                structured={
                                    key: value
                                    for key, value in item.items()
                                    if key not in {"summary", "content"}
                                },
                            )
                        ],
                    )
                )
            elif item_type == "function_call":
                messages.append(
                    NormalizedMessage(
                        role="assistant",
                        parts=[
                            MessagePartV5(
                                part_id=f"rs-{index}-0",
                                type=PartType.TOOL_CALL,
                                tool_call_id=str(item.get("call_id") or item.get("id") or ""),
                                tool_name=str(item.get("name") or ""),
                                arguments_json=_json_text(item.get("arguments") or "{}"),
                            )
                        ],
                    )
                )
            elif item_type in {"computer_call", "local_shell_call", "mcp_call"}:
                messages.append(
                    NormalizedMessage(
                        role="assistant",
                        parts=[
                            MessagePartV5(
                                part_id=f"rs-{index}-0",
                                type=PartType.TOOL_CALL,
                                tool_call_id=str(item.get("call_id") or item.get("id") or ""),
                                tool_name=str(item.get("name") or item_type),
                                arguments_json=_json_text(
                                    item.get("arguments") or item.get("action") or {}
                                ),
                                structured=dict(item),
                            )
                        ],
                    )
                )
            else:
                diagnostics.append(f"Responses output item {item_type!r} preserved as unsupported")
                messages.append(_unsupported_message(dict(item), f"rs-{index}", raw_kind=item_type))
    elif "compaction" in body or "compact_window" in body:
        # /responses/compact returns a compaction window, not a response id.
        messages.append(
            NormalizedMessage(
                role="system",
                parts=[
                    MessagePartV5(
                        part_id="rs-compact-0",
                        type=PartType.STRUCTURED,
                        structured=dict(body),
                        raw_kind="responses.compaction_window",
                    )
                ],
            )
        )
    else:
        diagnostics.append("Responses payload contained no output array")
    usage_payload = body.get("usage")
    try:
        token_capture = _tokens_from_payload(body)
    except (TypeError, ValueError) as exc:
        token_capture = TokenCaptureV5(
            provenance=TokenCaptureProvenance.UNAVAILABLE,
            level="none",
            unavailable_fields=("prompt_token_ids", "completion_token_ids", "logprobs"),
        )
        diagnostics.append(f"Responses token capture could not be parsed: {exc}")
    return NormalizedProviderResult(
        messages=messages,
        usage=usage_from_responses(
            usage_payload if isinstance(usage_payload, Mapping) else None
        ),
        token_capture=token_capture,
        diagnostics=diagnostics,
        provider_ids={
            key: str(body[key])
            for key in ("id", "previous_response_id", "conversation")
            if body.get(key) is not None
        },
        terminal_observed=str(body.get("status") or "") in {"completed", "failed", "cancelled"}
        or "compaction" in body
        or "compact_window" in body,
    )


def usage_from_responses(payload: Mapping[str, Any] | None) -> UsageV5:
    if payload is None:
        return UsageV5(
            provenance=UsageProvenance.UNAVAILABLE,
            requests=1,
            unavailable_fields=("prompt_tokens", "completion_tokens", "total_tokens"),
        )
    input_details = payload.get("input_tokens_details")
    output_details = payload.get("output_tokens_details")
    cached = cache_write = reasoning = None
    if isinstance(input_details, Mapping):
        cached = _int(input_details.get("cached_tokens"))
        cache_write = _int(
            input_details.get("cache_write_tokens")
            or input_details.get("cache_creation_tokens")
        )
    if isinstance(output_details, Mapping):
        reasoning = _int(output_details.get("reasoning_tokens"))
    prompt = _int(payload.get("input_tokens") or payload.get("prompt_tokens"))
    completion = _int(payload.get("output_tokens") or payload.get("completion_tokens"))
    total = _int(payload.get("total_tokens"))
    return UsageV5(
        provenance=UsageProvenance.OBSERVED_PROVIDER,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        total_tokens=total,
        requests=1,
        unavailable_fields=tuple(
            name
            for name, value in (
                ("prompt_tokens", prompt),
                ("completion_tokens", completion),
                ("total_tokens", total),
            )
            if value is None
        ),
    )


class OpenAIResponsesStreamAssembler:
    """SSE/WS event assembler; event order is retained exactly as observed."""

    def __init__(self) -> None:
        self.decoder = SSEDecoder()
        self.events: list[dict[str, Any]] = []
        self.diagnostics: list[str] = []

    def feed(self, chunk: bytes) -> None:
        for event in self.decoder.feed(chunk):
            self._event(event)

    def feed_websocket(self, message: bytes | str | Mapping[str, Any]) -> None:
        """Feed one Responses WebSocket server event.

        The WS protocol uses the same event ordering as SSE. Streaming is implicit,
        ``background`` is unsupported, and callers must enforce one response in flight.
        """

        if isinstance(message, Mapping):
            self.events.append(dict(_mapping(message)))
            return
        text = message.decode("utf-8", errors="replace") if isinstance(message, bytes) else message
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.diagnostics.append("undecodable Responses WebSocket event")
            return
        if isinstance(payload, Mapping):
            self.events.append(dict(payload))

    def _event(self, event: SSEEvent) -> None:
        if not event.data or event.data == "[DONE]":
            return
        try:
            payload = json.loads(event.data)
        except json.JSONDecodeError:
            self.diagnostics.append("undecodable Responses SSE event")
            return
        if isinstance(payload, Mapping):
            item = dict(payload)
            if event.event and "type" not in item:
                item["type"] = event.event
            self.events.append(item)

    def finish(self) -> NormalizedProviderResult:
        for event in self.decoder.finish():
            self._event(event)
        terminal = next(
            (
                item
                for item in reversed(self.events)
                if str(item.get("type") or "")
                in {"response.completed", "response.failed", "response.cancelled"}
            ),
            None,
        )
        if terminal and isinstance(terminal.get("response"), Mapping):
            result = normalize_responses_response(terminal["response"])
            result.raw_events = list(self.events)
            result.diagnostics.extend(self.diagnostics)
            result.terminal_observed = True
            return result
        return _responses_from_events(self.events, diagnostics=self.diagnostics)


def _responses_from_events(
    events: list[dict[str, Any]],
    *,
    diagnostics: list[str],
) -> NormalizedProviderResult:
    text: list[str] = []
    reasoning: list[str] = []
    calls: dict[str, dict[str, str]] = {}
    usage: Mapping[str, Any] | None = None
    terminal = False
    provider_ids: dict[str, str] = {}
    for event in events:
        kind = str(event.get("type") or "")
        if kind in {"response.output_text.delta", "response.refusal.delta"}:
            text.append(str(event.get("delta") or ""))
        elif kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            reasoning.append(str(event.get("delta") or ""))
        elif kind == "response.function_call_arguments.delta":
            call_id = str(event.get("call_id") or event.get("item_id") or "")
            call = calls.setdefault(call_id, {"name": "", "arguments": ""})
            call["arguments"] += str(event.get("delta") or "")
        elif kind == "response.output_item.added" and isinstance(event.get("item"), Mapping):
            item = event["item"]
            if item.get("type") == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "")
                call = calls.setdefault(call_id, {"name": "", "arguments": ""})
                call["name"] = str(item.get("name") or "")
        elif kind in {"response.completed", "response.failed", "response.cancelled"}:
            terminal = True
            response = event.get("response")
            if isinstance(response, Mapping):
                if isinstance(response.get("usage"), Mapping):
                    usage = response["usage"]
                if response.get("id") is not None:
                    provider_ids["response_id"] = str(response["id"])
        elif kind not in {
            "response.created",
            "response.in_progress",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.done",
            "response.function_call_arguments.done",
            "response.queued",
        }:
            diagnostics.append(f"unsupported Responses stream event {kind!r} preserved raw")
    parts: list[MessagePartV5] = []
    if reasoning:
        parts.append(
            MessagePartV5(
                part_id="rs-stream-0",
                type=PartType.REASONING,
                text="".join(reasoning),
                reasoning_availability=ReasoningAvailability.CAPTURED,
            )
        )
    if text:
        parts.append(
            MessagePartV5(
                part_id=f"rs-stream-{len(parts)}",
                type=PartType.TEXT,
                text="".join(text),
            )
        )
    for call_id, call in calls.items():
        parts.append(
            MessagePartV5(
                part_id=f"rs-stream-{len(parts)}",
                type=PartType.TOOL_CALL,
                tool_call_id=call_id,
                tool_name=call["name"],
                arguments_json=call["arguments"] or "{}",
            )
        )
    return NormalizedProviderResult(
        messages=[NormalizedMessage(role="assistant", parts=parts)] if parts else [],
        usage=usage_from_responses(usage),
        diagnostics=list(diagnostics),
        provider_ids=provider_ids,
        terminal_observed=terminal,
        raw_events=list(events),
    )


class OpenAIResponsesAdapter:
    name: str = NORMALIZER_NAME
    version: str = NORMALIZER_VERSION
    routes: tuple[str, ...] = (RESPONSES_PATH, RESPONSES_COMPACT_PATH)

    def normalize_request(self, body: Mapping[str, Any]) -> list[NormalizedMessage]:
        return normalize_responses_request(body)

    def normalize_unary(self, body: Mapping[str, Any]) -> NormalizedProviderResult:
        return normalize_responses_response(body)

    def new_stream(self) -> OpenAIResponsesStreamAssembler:
        return OpenAIResponsesStreamAssembler()

    def usage(self, payload: Mapping[str, Any] | None) -> UsageV5:
        return usage_from_responses(payload)


def _content_parts(value: Any, *, prefix: str) -> list[MessagePartV5]:
    if isinstance(value, str):
        return [MessagePartV5(part_id=f"{prefix}-0", type=PartType.TEXT, text=value)]
    if not isinstance(value, list):
        return [
            MessagePartV5(
                part_id=f"{prefix}-0",
                type=PartType.UNSUPPORTED,
                raw_kind=type(value).__name__,
                structured={"value": value},
            )
        ]
    parts: list[MessagePartV5] = []
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, Mapping):
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.UNSUPPORTED,
                    raw_kind=type(raw_item).__name__,
                    text=str(raw_item),
                )
            )
            continue
        item = _mapping(raw_item)
        kind = str(item.get("type") or "")
        if kind in {"input_text", "output_text", "text", "refusal"}:
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.TEXT,
                    text=str(item.get("text") or item.get("refusal") or ""),
                    raw_kind=kind,
                )
            )
        elif kind in {"reasoning", "reasoning_text"}:
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.REASONING,
                    text=_text(item),
                    reasoning_availability=ReasoningAvailability.CAPTURED,
                )
            )
        elif kind in {"input_image", "input_file"}:
            parts.append(
                MessagePartV5(
                    part_id=f"{prefix}-{index}",
                    type=PartType.MEDIA,
                    structured=dict(item),
                    media_type="image" if kind == "input_image" else "file",
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


def _mapping(value: Any) -> Mapping[str, Any]:
    """A foreign payload section, or an empty one when absent or the wrong shape."""
    return value if isinstance(value, Mapping) else {}


def _unsupported_message(value: Any, prefix: str, *, raw_kind: str | None = None) -> NormalizedMessage:
    return NormalizedMessage(
        role="user",
        parts=[
            MessagePartV5(
                part_id=f"{prefix}-0",
                type=PartType.UNSUPPORTED,
                raw_kind=raw_kind or type(value).__name__,
                structured=value if isinstance(value, dict) else {"value": value},
            )
        ],
    )


def _tokens_from_payload(payload: Mapping[str, Any]) -> TokenCaptureV5 | None:
    prompt = payload.get("prompt_token_ids")
    completion = payload.get("completion_token_ids")
    logprobs = payload.get("logprobs")
    if not isinstance(prompt, list) and not isinstance(completion, list) and not isinstance(logprobs, list):
        return None
    prompt_ids = tuple(int(item) for item in prompt) if isinstance(prompt, list) else ()
    completion_ids = (
        tuple(int(item) for item in completion) if isinstance(completion, list) else ()
    )
    values = tuple(float(item) for item in logprobs) if isinstance(logprobs, list) else ()
    return TokenCaptureV5(
        provenance=TokenCaptureProvenance.OBSERVED_PROVIDER,
        level="full_training" if values else "completion",
        prompt=TokenSequenceRefV1(token_ids=prompt_ids, count=len(prompt_ids))
        if prompt_ids
        else None,
        completion=TokenSequenceRefV1(token_ids=completion_ids, count=len(completion_ids))
        if completion_ids
        else None,
        completion_logprobs=values,
    )


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "NORMALIZER_NAME",
    "NORMALIZER_VERSION",
    "OpenAIResponsesAdapter",
    "OpenAIResponsesStreamAssembler",
    "RESPONSES_COMPACT_PATH",
    "RESPONSES_PATH",
    "RESPONSES_WEBSOCKET_URL",
    "normalize_responses_request",
    "normalize_responses_response",
    "usage_from_responses",
]
