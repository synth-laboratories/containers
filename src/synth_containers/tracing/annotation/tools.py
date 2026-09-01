"""Read-only, bounded trace-inspection tools an annotator may call.

Every response carries stable entity IDs and ready-made selectors so a finding can
cite exactly what was inspected. Tools enforce a call budget, a per-response byte
cap, and a cumulative byte cap; exceeding any of them raises ``ToolLimitExceeded``
and the job fails closed. No tool can write, execute, or reach outside the sealed
document and the declared projections.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from synth_containers.serde import JsonDataclassMixin, jsonable

from ..canonical import canonical_bytes, canonical_text, content_digest, utc_now
from ..models.document import TraceDocumentV5
from ..models.messages import MessageNodeV5, PartType
from ..models.projection import ProjectionManifestV1
from ..models.selectors import (
    SelectorKind,
    TextRangeV1,
    TraceSelectorV1,
    resolve_selector,
)
from ..models.spans import SpanKind, SpanV5
from .jobs import AnnotationJobLimitsV1


TOOL_CONTRACT_VERSION = "synth.trace-inspection-tools.v1"


class ToolLimitExceeded(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ToolArgumentError(ValueError):
    """The annotator called a tool with arguments the contract does not allow."""


@dataclass(frozen=True, slots=True)
class ToolCallRecordV1(JsonDataclassMixin):
    index: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    started_at: str
    ended_at: str
    response_bytes: int = 0
    truncated: bool = False
    error: str | None = None
    response_digest: str | None = None


def _selector_dict(selector: TraceSelectorV1) -> dict[str, Any]:
    return {
        key: value
        for key, value in selector.to_dict().items()
        if value is not None and key != "schema_version"
    }


def _text_window(text: str, offset: int, max_chars: int) -> dict[str, Any]:
    if offset < 0:
        offset = 0
    window = text[offset : offset + max_chars]
    return {
        "text": window,
        "offset": offset,
        "length": len(text),
        "truncated": offset + len(window) < len(text),
    }


_SPEC: list[dict[str, Any]] = [
    {
        "name": "trace_get_manifest",
        "description": (
            "Identity, lifecycle, entity counts, actors, sessions, and extension keys of the "
            "sealed source trace. Read-only. Start here."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "trace_list_entities",
        "description": (
            "Page through entities of one kind (message, span, event, actor, session, "
            "artifact) in trace order with their selectors. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["message", "span", "event", "actor", "session", "artifact"],
                },
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                "session_id": {"type": ["string", "null"]},
                "span_kind": {"type": ["string", "null"]},
                "role": {"type": ["string", "null"]},
                "event_type": {"type": ["string", "null"]},
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_turn",
        "description": (
            "Every span and message that shares one turn_id, plus messages those spans "
            "consumed or produced. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"turn_id": {"type": "string"}},
            "required": ["turn_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_span",
        "description": "One span with its detail, usage, and linked message summaries. Read-only.",
        "inputSchema": {
            "type": "object",
            "properties": {"span_id": {"type": "string"}},
            "required": ["span_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_message",
        "description": (
            "One message: role, parts, metadata, and a bounded text window. Use offset "
            "to page through long text. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 6000},
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_tool_call",
        "description": (
            "A tool call part and its matching tool result part, located by tool_call_id. "
            "Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"tool_call_id": {"type": "string"}},
            "required": ["tool_call_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_environment_transition",
        "description": (
            "One environment_step span with its detail, the previous and next environment "
            "steps in the same session, and the engine events attached to it. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"span_id": {"type": "string"}},
            "required": ["span_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_event",
        "description": (
            "One event with its full payload (reward values, grader scores, achievements, "
            "terminal reasons live here). Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_resolve_selector",
        "description": (
            "Check that a selector resolves against the sealed trace and see the exact "
            "text it cites. Use before citing evidence. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"selector": {"type": "object"}},
            "required": ["selector"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_compare_pre_post_state",
        "description": (
            "Line-level diff between two messages' text (for example the observation "
            "before and after an action). Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "before_message_id": {"type": "string"},
                "after_message_id": {"type": "string"},
                "max_lines": {"type": "integer", "minimum": 1, "maximum": 400, "default": 120},
            },
            "required": ["before_message_id", "after_message_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_search_text",
        "description": (
            "Case-insensitive substring search over message text; returns message ids, "
            "roles, and character offsets suitable for range selectors. Read-only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 200},
                "role": {"type": ["string", "null"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "trace_get_projection_manifest",
        "description": (
            "The lossy projections this job may cite, with their declared losses. "
            "Projections are never trace authority. Read-only."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

TOOL_SPECS: tuple[dict[str, Any], ...] = tuple(_SPEC)
TOOL_NAMES: tuple[str, ...] = tuple(spec["name"] for spec in _SPEC)


def tool_contract(names: tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = [spec for spec in _SPEC if names is None or spec["name"] in names]
    if names is not None:
        unknown = set(names) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown trace inspection tools: {sorted(unknown)}")
    return {"version": TOOL_CONTRACT_VERSION, "tools": selected}


def tool_contract_digest(names: tuple[str, ...] | None = None) -> str:
    return content_digest(tool_contract(names))


class TraceInspectionTools:
    """Bound one sealed document to the tool contract with hard limits."""

    def __init__(
        self,
        document: TraceDocumentV5,
        *,
        limits: AnnotationJobLimitsV1,
        tool_names: tuple[str, ...] | None = None,
        projections: tuple[ProjectionManifestV1, ...] = (),
    ) -> None:
        if not document.content_digest:
            raise ValueError("trace inspection requires a sealed document")
        self.document = document
        self.limits = limits
        self.projections = projections
        self.tool_names = tuple(tool_names) if tool_names is not None else TOOL_NAMES
        unknown = set(self.tool_names) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown trace inspection tools: {sorted(unknown)}")
        self.calls: list[ToolCallRecordV1] = []
        self.total_bytes = 0
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "trace_get_manifest": self.trace_get_manifest,
            "trace_list_entities": self.trace_list_entities,
            "trace_get_turn": self.trace_get_turn,
            "trace_get_span": self.trace_get_span,
            "trace_get_message": self.trace_get_message,
            "trace_get_tool_call": self.trace_get_tool_call,
            "trace_get_environment_transition": self.trace_get_environment_transition,
            "trace_get_event": self.trace_get_event,
            "trace_resolve_selector": self.trace_resolve_selector,
            "trace_compare_pre_post_state": self.trace_compare_pre_post_state,
            "trace_search_text": self.trace_search_text,
            "trace_get_projection_manifest": self.trace_get_projection_manifest,
        }
        self._spans_by_session: dict[str, list[SpanV5]] = {}
        for span in sorted(document.spans, key=lambda item: (item.started_at, item.span_id)):
            self._spans_by_session.setdefault(span.session_id, []).append(span)

    # -- contract -----------------------------------------------------------------

    def specs(self) -> list[dict[str, Any]]:
        return [spec for spec in _SPEC if spec["name"] in self.tool_names]

    def contract_digest(self) -> str:
        return tool_contract_digest(self.tool_names)

    # -- dispatch -----------------------------------------------------------------

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one tool call under the job limits and record it."""

        arguments = dict(arguments or {})
        started = utc_now()
        index = len(self.calls)
        if index >= self.limits.max_tool_calls:
            self.calls.append(
                ToolCallRecordV1(
                    index=index,
                    tool=name,
                    arguments=arguments,
                    ok=False,
                    started_at=started,
                    ended_at=utc_now(),
                    error="tool_call_limit",
                )
            )
            raise ToolLimitExceeded(
                "tool_call_limit",
                f"annotator exceeded {self.limits.max_tool_calls} tool calls",
            )
        if name not in self.tool_names:
            self.calls.append(
                ToolCallRecordV1(
                    index=index,
                    tool=name,
                    arguments=arguments,
                    ok=False,
                    started_at=started,
                    ended_at=utc_now(),
                    error="tool_not_in_contract",
                )
            )
            raise ToolArgumentError(f"tool {name!r} is not in this job's contract")
        handler = self._handlers[name]
        try:
            response = handler(**arguments)
        except TypeError as error:
            self.calls.append(
                ToolCallRecordV1(
                    index=index,
                    tool=name,
                    arguments=arguments,
                    ok=False,
                    started_at=started,
                    ended_at=utc_now(),
                    error=f"bad_arguments: {error}",
                )
            )
            raise ToolArgumentError(str(error)) from error
        except ToolArgumentError as error:
            self.calls.append(
                ToolCallRecordV1(
                    index=index,
                    tool=name,
                    arguments=arguments,
                    ok=False,
                    started_at=started,
                    ended_at=utc_now(),
                    error=f"bad_arguments: {error}",
                )
            )
            raise
        bounded, truncated = bound_payload(response, self.limits.max_tool_response_bytes)
        size = len(canonical_bytes(bounded))
        self.total_bytes += size
        self.calls.append(
            ToolCallRecordV1(
                index=index,
                tool=name,
                arguments=arguments,
                ok=True,
                started_at=started,
                ended_at=utc_now(),
                response_bytes=size,
                truncated=truncated,
                response_digest=content_digest(bounded),
            )
        )
        if self.total_bytes > self.limits.max_total_tool_bytes:
            raise ToolLimitExceeded(
                "tool_byte_limit",
                f"annotator exceeded {self.limits.max_total_tool_bytes} cumulative tool bytes",
            )
        if truncated:
            bounded = {**bounded, "truncated": True}
        return bounded

    # -- selectors ----------------------------------------------------------------

    def selector(
        self,
        kind: SelectorKind | str,
        entity_id: str | None = None,
        *,
        part_id: str | None = None,
    ) -> dict[str, Any]:
        return _selector_dict(
            TraceSelectorV1(
                trace_id=self.document.trace_id,
                trace_digest=self.document.content_digest,
                kind=str(kind),
                entity_id=entity_id,
                part_id=part_id,
            )
        )

    # -- tools --------------------------------------------------------------------

    def trace_get_manifest(self) -> dict[str, Any]:
        document = self.document
        return {
            "trace_id": document.trace_id,
            "trace_digest": document.content_digest,
            "schema_version": document.schema_version,
            "trace_kind": str(document.trace_kind),
            "identity": jsonable(document.identity),
            "lifecycle": jsonable(document.lifecycle),
            "completeness": {
                "capture_status": str(document.completeness.capture_status),
                "terminal_event_observed": document.completeness.terminal_event_observed,
                "reasons": list(document.completeness.reasons),
            },
            "counts": {
                "actors": len(document.actors),
                "sessions": len(document.sessions),
                "messages": len(document.messages),
                "spans": len(document.spans),
                "events": len(document.events),
                "artifacts": len(document.artifacts),
            },
            "span_kinds": sorted({str(span.span_kind) for span in document.spans}),
            "event_types": sorted({str(event.event_type) for event in document.events}),
            "actors": [
                {
                    "actor_id": actor.actor_id,
                    "kind": str(actor.kind),
                    "display_name": actor.display_name,
                    "model": actor.model,
                    "selector": self.selector(SelectorKind.ACTOR, actor.actor_id),
                }
                for actor in document.actors
            ],
            "sessions": [
                {
                    "session_id": session.session_id,
                    "actor_id": session.actor_id,
                    "status": str(session.status),
                    "started_at": session.started_at,
                    "ended_at": session.ended_at,
                    "selector": self.selector(SelectorKind.SESSION, session.session_id),
                }
                for session in document.sessions
            ],
            "extension_keys": sorted(document.extensions),
            "selector": self.selector(SelectorKind.TRACE),
        }

    def trace_list_entities(
        self,
        kind: str,
        offset: int = 0,
        limit: int = 50,
        session_id: str | None = None,
        span_kind: str | None = None,
        role: str | None = None,
        event_type: str | None = None,
    ) -> dict[str, Any]:
        offset = max(0, int(offset))
        limit = max(1, min(200, int(limit)))
        document = self.document
        rows: list[dict[str, Any]]
        if kind == "message":
            items = [
                message
                for message in document.messages
                if (session_id is None or message.session_id == session_id)
                and (role is None or str(message.role) == role)
            ]
            items.sort(key=lambda item: (item.occurred_at or "", item.message_id))
            rows = [self._message_summary(item) for item in items[offset : offset + limit]]
        elif kind == "span":
            items = [
                span
                for span in document.spans
                if (session_id is None or span.session_id == session_id)
                and (span_kind is None or str(span.span_kind) == span_kind)
            ]
            items.sort(key=lambda item: (item.started_at, item.span_id))
            rows = [self._span_summary(item) for item in items[offset : offset + limit]]
        elif kind == "event":
            items = [
                event
                for event in document.events
                if (session_id is None or event.session_id == session_id)
                and (event_type is None or str(event.event_type) == event_type)
            ]
            items.sort(
                key=lambda item: (
                    item.order.chronological_sequence
                    if item.order.chronological_sequence is not None
                    else -1,
                    item.occurred_at,
                    item.event_id,
                )
            )
            rows = [self._event_summary(item) for item in items[offset : offset + limit]]
        elif kind == "actor":
            items = list(document.actors)
            rows = [
                {
                    "actor_id": actor.actor_id,
                    "kind": str(actor.kind),
                    "display_name": actor.display_name,
                    "selector": self.selector(SelectorKind.ACTOR, actor.actor_id),
                }
                for actor in items[offset : offset + limit]
            ]
        elif kind == "session":
            items = list(document.sessions)
            rows = [
                {
                    "session_id": session.session_id,
                    "actor_id": session.actor_id,
                    "status": str(session.status),
                    "selector": self.selector(SelectorKind.SESSION, session.session_id),
                }
                for session in items[offset : offset + limit]
            ]
        elif kind == "artifact":
            items = list(document.artifacts)
            rows = [
                {
                    "artifact_id": artifact.artifact_id,
                    "digest": artifact.digest,
                    "media_type": getattr(artifact, "media_type", None),
                    "selector": self.selector(SelectorKind.ARTIFACT, artifact.artifact_id),
                }
                for artifact in items[offset : offset + limit]
            ]
        else:
            raise ToolArgumentError(f"unsupported entity kind {kind!r}")
        return {
            "kind": kind,
            "offset": offset,
            "limit": limit,
            "total": len(items),
            "items": rows,
        }

    def trace_get_turn(self, turn_id: str) -> dict[str, Any]:
        spans = [span for span in self.document.spans if span.turn_id == turn_id]
        messages = [msg for msg in self.document.messages if msg.turn_id == turn_id]
        linked: dict[str, MessageNodeV5] = {}
        for span in spans:
            for message_id in (*span.input_message_ids, *span.output_message_ids):
                message = self.document.message(message_id)
                if message is not None:
                    linked[message_id] = message
        if not spans and not messages:
            raise ToolArgumentError(f"turn {turn_id!r} not found")
        return {
            "turn_id": turn_id,
            "spans": [self._span_summary(span) for span in spans],
            "messages": [self._message_summary(msg) for msg in messages],
            "linked_messages": [self._message_summary(msg) for msg in linked.values()],
        }

    def trace_get_span(self, span_id: str) -> dict[str, Any]:
        span = self.document.span(span_id)
        if span is None:
            raise ToolArgumentError(f"span {span_id!r} not found")
        payload = self._span_summary(span)
        payload["detail"] = jsonable(span.detail)
        payload["usage"] = jsonable(span.usage) if span.usage is not None else None
        payload["input_messages"] = [
            self._message_summary(message)
            for message in (self.document.message(mid) for mid in span.input_message_ids)
            if message is not None
        ]
        payload["output_messages"] = [
            self._message_summary(message)
            for message in (self.document.message(mid) for mid in span.output_message_ids)
            if message is not None
        ]
        payload["events"] = [
            self._event_summary(event)
            for event in self.document.events
            if event.span_id == span_id
        ]
        return payload

    def trace_get_message(
        self,
        message_id: str,
        offset: int = 0,
        max_chars: int = 6000,
    ) -> dict[str, Any]:
        message = self.document.message(message_id)
        if message is None:
            raise ToolArgumentError(f"message {message_id!r} not found")
        max_chars = max(1, min(20000, int(max_chars)))
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            text = part.text or part.arguments_json or ""
            item: dict[str, Any] = {
                "part_id": part.part_id,
                "type": str(part.type),
                "selector": self.selector(SelectorKind.PART, message.message_id, part_id=part.part_id),
            }
            if str(part.type) == PartType.REASONING:
                item["reasoning_availability"] = (
                    str(part.reasoning_availability) if part.reasoning_availability else None
                )
                item["text"] = None
                item["note"] = "reasoning content is not exposed to annotators"
            else:
                item.update(_text_window(text, int(offset), max_chars))
            if part.tool_call_id:
                item["tool_call_id"] = part.tool_call_id
            if part.tool_name:
                item["tool_name"] = part.tool_name
            if part.is_error is not None:
                item["is_error"] = part.is_error
            if part.structured is not None:
                item["structured"] = jsonable(part.structured)
            parts.append(item)
        payload = self._message_summary(message)
        payload["parts"] = parts
        payload["metadata"] = jsonable(message.metadata)
        payload["predecessor_message_ids"] = list(message.predecessor_message_ids)
        payload["produced_by_span_id"] = message.produced_by_span_id
        payload.update(
            {
                key: value
                for key, value in _text_window(message.text(), int(offset), max_chars).items()
            }
        )
        return payload

    def trace_get_tool_call(self, tool_call_id: str) -> dict[str, Any]:
        call: dict[str, Any] | None = None
        result: dict[str, Any] | None = None
        for message in self.document.messages:
            for part in message.parts:
                if part.tool_call_id != tool_call_id:
                    continue
                item = {
                    "message_id": message.message_id,
                    "role": str(message.role),
                    "part_id": part.part_id,
                    "type": str(part.type),
                    "tool_name": part.tool_name,
                    "text": part.text,
                    "arguments_json": part.arguments_json,
                    "is_error": part.is_error,
                    "selector": self.selector(SelectorKind.PART, message.message_id, part_id=part.part_id),
                }
                if str(part.type) == PartType.TOOL_CALL and call is None:
                    call = item
                elif str(part.type) == PartType.TOOL_RESULT and result is None:
                    result = item
        if call is None and result is None:
            raise ToolArgumentError(f"tool call {tool_call_id!r} not found")
        return {"tool_call_id": tool_call_id, "call": call, "result": result}

    def trace_get_environment_transition(self, span_id: str) -> dict[str, Any]:
        span = self.document.span(span_id)
        if span is None:
            raise ToolArgumentError(f"span {span_id!r} not found")
        if str(span.span_kind) != SpanKind.ENVIRONMENT_STEP:
            raise ToolArgumentError(f"span {span_id!r} is not an environment_step")
        siblings = [
            item
            for item in self._spans_by_session.get(span.session_id, [])
            if str(item.span_kind) == SpanKind.ENVIRONMENT_STEP
        ]
        position = next((i for i, item in enumerate(siblings) if item.span_id == span_id), -1)
        previous = siblings[position - 1] if position > 0 else None
        following = siblings[position + 1] if 0 <= position < len(siblings) - 1 else None
        return {
            "span": {**self._span_summary(span), "detail": jsonable(span.detail)},
            "previous": (
                {**self._span_summary(previous), "detail": jsonable(previous.detail)}
                if previous is not None
                else None
            ),
            "next": (
                {**self._span_summary(following), "detail": jsonable(following.detail)}
                if following is not None
                else None
            ),
            "events": [
                {**self._event_summary(event), "payload": jsonable(event.payload)}
                for event in self.document.events
                if event.span_id == span_id
            ],
            "position": position,
            "session_step_count": len(siblings),
        }

    def trace_get_event(self, event_id: str) -> dict[str, Any]:
        event = self.document.event(event_id)
        if event is None:
            raise ToolArgumentError(f"event {event_id!r} not found")
        payload = self._event_summary(event)
        payload["payload"] = jsonable(event.payload)
        payload["status"] = str(event.status)
        payload["actor_id"] = event.actor_id
        payload["caused_by_event_ids"] = list(event.caused_by_event_ids)
        return payload

    def trace_resolve_selector(self, selector: dict[str, Any]) -> dict[str, Any]:
        built = build_selector(self.document, selector)
        resolution = resolve_selector(self.document, built)
        text = resolution.resolved_text
        return {
            "selector": _selector_dict(built),
            "resolved": resolution.resolved,
            "reason": resolution.reason,
            "entity_kind": resolution.entity_kind,
            "entity_digest": resolution.entity_digest,
            "resolved_text": _text_window(text, 0, 4000) if text is not None else None,
        }

    def trace_compare_pre_post_state(
        self,
        before_message_id: str,
        after_message_id: str,
        max_lines: int = 120,
    ) -> dict[str, Any]:
        before = self.document.message(before_message_id)
        after = self.document.message(after_message_id)
        if before is None:
            raise ToolArgumentError(f"message {before_message_id!r} not found")
        if after is None:
            raise ToolArgumentError(f"message {after_message_id!r} not found")
        max_lines = max(1, min(400, int(max_lines)))
        before_lines = before.text().splitlines()
        after_lines = after.text().splitlines()
        diff = [
            line
            for line in difflib.unified_diff(before_lines, after_lines, lineterm="", n=0)
            if not line.startswith(("---", "+++", "@@"))
        ]
        return {
            "before": self._message_summary(before),
            "after": self._message_summary(after),
            "changed_lines": diff[:max_lines],
            "changed_line_count": len(diff),
            "truncated_diff": len(diff) > max_lines,
        }

    def trace_search_text(
        self,
        query: str,
        role: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        needle = query.lower()
        limit = max(1, min(100, int(limit)))
        hits: list[dict[str, Any]] = []
        for message in self.document.messages:
            if role is not None and str(message.role) != role:
                continue
            text = message.text()
            lowered = text.lower()
            start = lowered.find(needle)
            while start >= 0 and len(hits) < limit:
                end = start + len(query)
                hits.append(
                    {
                        "message_id": message.message_id,
                        "role": str(message.role),
                        "start": start,
                        "end": end,
                        "context": text[max(0, start - 80) : end + 80],
                        "selector": {
                            **self.selector(SelectorKind.MESSAGE, message.message_id),
                            "range": {"start": start, "end": end, "unit": "character"},
                            "quote": text[start:end],
                        },
                    }
                )
                start = lowered.find(needle, end)
            if len(hits) >= limit:
                break
        return {"query": query, "hits": hits, "truncated": len(hits) >= limit}

    def trace_get_projection_manifest(self) -> dict[str, Any]:
        return {
            "projections": [
                {
                    "projection_id": manifest.projection_id,
                    "format": manifest.format,
                    "digest": manifest.content_digest,
                    "target_digest": manifest.target_digest,
                    "losses": [jsonable(loss) for loss in manifest.losses],
                    "included_layers": list(manifest.included_layers),
                    "omitted_layers": list(manifest.omitted_layers),
                }
                for manifest in self.projections
            ],
            "note": "projections are lossy views; findings must cite trace selectors",
        }

    # -- summaries ----------------------------------------------------------------

    def _message_summary(self, message: MessageNodeV5) -> dict[str, Any]:
        text = message.text()
        return {
            "message_id": message.message_id,
            "role": str(message.role),
            "session_id": message.session_id,
            "turn_id": message.turn_id,
            "occurred_at": message.occurred_at,
            "part_types": [str(part.type) for part in message.parts],
            "char_length": len(text),
            "preview": text[:240],
            "call_index": message.metadata.get("call_index"),
            "selector": self.selector(SelectorKind.MESSAGE, message.message_id),
        }

    def _span_summary(self, span: SpanV5) -> dict[str, Any]:
        detail = span.detail
        summary: dict[str, Any] = {
            "span_id": span.span_id,
            "span_kind": str(span.span_kind),
            "session_id": span.session_id,
            "turn_id": span.turn_id,
            "started_at": span.started_at,
            "ended_at": span.ended_at,
            "status": str(span.status),
            "input_message_ids": list(span.input_message_ids),
            "output_message_ids": list(span.output_message_ids),
            "selector": self.selector(SelectorKind.SPAN, span.span_id),
        }
        for key in ("step_index", "call_index", "action", "transition", "reason", "model"):
            if key in detail:
                summary[key] = jsonable(detail[key])
        return summary

    def _event_summary(self, event: Any) -> dict[str, Any]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        summary = {
            "event_id": event.event_id,
            "event_type": str(event.event_type),
            "session_id": event.session_id,
            "span_id": event.span_id,
            "occurred_at": event.occurred_at,
            "kind": payload.get("kind"),
            "step_index": payload.get("step_index"),
            "selector": self.selector(SelectorKind.EVENT, event.event_id),
        }
        # Outcome-bearing scalars ride on the summary so a listing already shows
        # reward values, grades, and terminal reasons; the full payload is one call away.
        preview = {
            key: jsonable(payload[key])
            for key in ("value", "reward", "score", "status", "achievement", "reason", "label", "authority", "stopped_on", "env_steps", "criteria_met", "points")
            if key in payload and isinstance(payload[key], (str, int, float, bool))
        }
        if isinstance(payload.get("payload"), dict):
            inner = payload["payload"]
            for key in ("achievement", "reason", "target"):
                if key in inner and isinstance(inner[key], (str, int, float, bool)):
                    preview[f"payload.{key}"] = inner[key]
        if preview:
            summary["preview"] = preview
        return summary


def build_selector(document: TraceDocumentV5, raw: dict[str, Any]) -> TraceSelectorV1:
    """Turn a proposal selector dict into a selector bound to this sealed trace.

    A proposal may omit trace id/digest (they are implied by the job); if it names
    them they must match, otherwise the selector is bound to the wrong authority.
    """

    if not isinstance(raw, dict):
        raise ToolArgumentError("selector must be an object")
    trace_id = raw.get("trace_id") or document.trace_id
    trace_digest = raw.get("trace_digest") or document.content_digest
    if trace_id != document.trace_id:
        raise ToolArgumentError("selector names a different trace id")
    if trace_digest != document.content_digest:
        raise ToolArgumentError("selector names a different trace digest")
    range_raw = raw.get("range")
    text_range = None
    if range_raw is not None:
        if not isinstance(range_raw, dict):
            raise ToolArgumentError("selector range must be an object")
        text_range = TextRangeV1(
            start=int(range_raw["start"]),
            end=int(range_raw["end"]),
            unit=str(range_raw.get("unit") or "character"),
        )
    kind = str(raw.get("kind") or "")
    entity_id = raw.get("entity_id")
    part_id = raw.get("part_id")
    if kind == SelectorKind.TRACE:
        # Canonical form: a whole-trace selector never names an entity. Models
        # sometimes echo the trace id as ``entity_id``; accept that and drop it so
        # repeats of one annotator land on one target digest (consensus groups by
        # target digest). Any other entity id is a contradiction, not a target.
        if entity_id not in (None, "", trace_id):
            raise ToolArgumentError("trace selector must not name an entity_id")
        if part_id not in (None, ""):
            raise ToolArgumentError("trace selector must not name a part_id")
        entity_id = None
        part_id = None
    selector = TraceSelectorV1(
        trace_id=trace_id,
        trace_digest=trace_digest,
        kind=kind,
        entity_id=entity_id,
        part_id=part_id,
        json_pointer=raw.get("json_pointer"),
        range=text_range,
        token_sequence=raw.get("token_sequence"),
        source_projection=raw.get("source_projection"),
    )
    quote = raw.get("quote")
    if quote is not None:
        selector = selector.with_quote(str(quote))
    return selector


def bound_payload(payload: Any, max_bytes: int) -> tuple[Any, bool]:
    """Shrink a JSON payload so its canonical encoding fits ``max_bytes``.

    Long strings are cut first, then long lists. Structure is preserved so the
    caller can still see what was there and page for more.
    """

    encoded = canonical_bytes(payload)
    if len(encoded) <= max_bytes:
        return payload, False
    strings: list[int] = []

    def count(value: Any) -> None:
        if isinstance(value, str):
            strings.append(len(value))
        elif isinstance(value, dict):
            for item in value.values():
                count(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                count(item)

    count(payload)
    cap = max(200, max_bytes // max(1, len(strings)))
    marker = "…[truncated]"

    def cut(value: Any) -> Any:
        if isinstance(value, str):
            return value if len(value) <= cap else value[:cap] + marker
        if isinstance(value, dict):
            return {key: cut(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cut(item) for item in value]
        return value

    shrunk = cut(payload)
    if len(canonical_bytes(shrunk)) <= max_bytes:
        return shrunk, True

    def trim_lists(value: Any, keep: int) -> Any:
        if isinstance(value, dict):
            return {key: trim_lists(item, keep) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [trim_lists(item, keep) for item in list(value)[:keep]]
        return value

    keep = 64
    while keep >= 1:
        candidate = trim_lists(shrunk, keep)
        if len(canonical_bytes(candidate)) <= max_bytes:
            return candidate, True
        keep //= 2
    return {"truncated": True, "note": "response exceeded the per-call byte limit"}, True


def selector_from_dict(document: TraceDocumentV5, raw: dict[str, Any]) -> TraceSelectorV1:
    return build_selector(document, raw)


def selector_text(document: TraceDocumentV5, selector: TraceSelectorV1) -> Optional[str]:
    resolution = resolve_selector(document, selector)
    return resolution.resolved_text if resolution.resolved else None


def describe_tool_calls(calls: list[ToolCallRecordV1]) -> str:
    return canonical_text([jsonable(call) for call in calls])


__all__ = [
    "TOOL_CONTRACT_VERSION",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "ToolArgumentError",
    "ToolCallRecordV1",
    "ToolLimitExceeded",
    "TraceInspectionTools",
    "bound_payload",
    "build_selector",
    "selector_from_dict",
    "selector_text",
    "tool_contract",
    "tool_contract_digest",
]
