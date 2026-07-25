"""Import a Containers ``RolloutTraceV4`` payload into a sealed V5 document.

Import is lossy in the other direction: V4 carries no capture binding, no raw provider
evidence, and no usage provenance. The resulting document says so — it declares
``imported`` capture, ``unavailable`` raw coverage, and lists every fact the source
format could not supply.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..canonical import canonical_bytes, bytes_digest, record_id, utc_now
from ..models.actors import ActorKind, ActorV5, CoverageState, SessionCoverageV5, SessionV5
from ..models.completeness import CaptureStatus, TraceCompletenessV5, TraceLifecycleV5, TraceStatus
from ..models.document import TraceCaptureSummaryV5, TraceDocumentV5
from ..models.events import EventOrderV1, EventType, EventV5
from ..models.identity import (
    AliasNamespace,
    AliasV1,
    TraceIdentityV5,
    TraceKind,
    TraceProvenanceV5,
)
from ..models.messages import MessageNodeV5, MessagePartV5, PartType, ReasoningAvailability
from ..models.spans import (
    SpanKind,
    SpanStatus,
    SpanV5,
    TransformationRecordV1,
    UsageProvenance,
    UsageV5,
)


IMPORTER_NAME = "containers_rollout_trace_v4"
IMPORTER_VERSION = "1"


def import_rollout_trace_v4(
    payload: Mapping[str, Any],
    *,
    trace_id: str | None = None,
    producer: str = "synth_containers.tracing.adapters.v4",
    imported_at: str = "1970-01-01T00:00:00Z",
) -> TraceDocumentV5:
    """Build a sealed V5 document from a V4 rollout trace payload."""

    source_digest = bytes_digest(canonical_bytes(payload))
    rollout_id = str(payload.get("rollout_id") or "")
    resolved_trace_id = trace_id or record_id(
        "trace", kind="imported_v4", key={"rollout_id": rollout_id, "digest": source_digest}
    )
    actor_id = record_id("actor", kind="actor", scope=(resolved_trace_id,), key="v4_rollout")
    session_id = record_id("sess", kind="session", scope=(resolved_trace_id, actor_id), key=0)
    losses: list[str] = [
        "source V4 payload carries no capture binding",
        "source V4 payload carries no raw provider evidence",
    ]

    spans: list[SpanV5] = []
    events: list[EventV5] = []
    messages: list[MessageNodeV5] = []
    aliases: list[AliasV1] = []
    total = UsageV5(provenance=UsageProvenance.DERIVED, requests=0)
    raw_spans = payload.get("spans")
    ordered = sorted(
        [item for item in list(raw_spans or []) if isinstance(item, Mapping)],
        key=lambda item: int(item.get("call_index") or 0),
    )
    for sequence, raw in enumerate(ordered, start=1):
        source_span_id = str(raw.get("span_id") or f"span-{sequence}")
        span_id = record_id(
            "span", kind="model_call", scope=(resolved_trace_id,), key=source_span_id
        )
        aliases.append(
            AliasV1(
                namespace=AliasNamespace.CONTAINERS_TRACE_V4,
                value=source_span_id,
                target_id=span_id,
                target_kind="span",
            )
        )
        request = _mapping(raw.get("request"))
        response = _mapping(raw.get("response"))
        input_ids: list[str] = []
        previous: tuple[str, ...] = ()
        for index, message in enumerate(list(request.get("messages") or [])):
            if not isinstance(message, Mapping):
                continue
            node = _message_from_v4(
                message,
                trace_id=resolved_trace_id,
                key={"span": source_span_id, "slot": "request", "index": index},
                actor_id=actor_id,
                session_id=session_id,
                span_id=span_id,
                predecessors=previous,
            )
            messages.append(node)
            input_ids.append(node.message_id)
            previous = (node.message_id,)
        output_ids: list[str] = []
        choices = list(response.get("choices") or [])
        if choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping):
                node = _message_from_v4(
                    message,
                    trace_id=resolved_trace_id,
                    key={"span": source_span_id, "slot": "response"},
                    actor_id=actor_id,
                    session_id=session_id,
                    span_id=span_id,
                    predecessors=previous,
                )
                messages.append(node)
                output_ids.append(node.message_id)
        usage_payload = _mapping(response.get("usage"))
        usage = UsageV5(
            provenance=UsageProvenance.OBSERVED_HARNESS,
            prompt_tokens=_int(usage_payload.get("prompt_tokens")),
            completion_tokens=_int(usage_payload.get("completion_tokens")),
            reasoning_tokens=_int(usage_payload.get("reasoning_tokens")),
            cached_tokens=_int(usage_payload.get("cached_tokens")),
            requests=1,
            source_refs=(source_span_id,),
        )
        total = total.merged(usage)
        spans.append(
            SpanV5(
                span_id=span_id,
                span_kind=SpanKind.MODEL_CALL,
                actor_id=actor_id,
                session_id=session_id,
                started_at=imported_at,
                status=SpanStatus.OK,
                detail={
                    "call_index": int(raw.get("call_index") or sequence - 1),
                    "model": str(request.get("model") or response.get("model") or ""),
                    "streaming": False,
                    "source_span_id": source_span_id,
                },
                input_message_ids=tuple(input_ids),
                output_message_ids=tuple(output_ids),
                usage=usage,
                transformations=(
                    TransformationRecordV1(
                        name=IMPORTER_NAME,
                        version=IMPORTER_VERSION,
                        input_refs=(source_digest,),
                        output_refs=tuple(input_ids + output_ids),
                        losses=("V4 usage has no provider provenance",),
                        deterministic=True,
                    ),
                ),
            ).sealed()
        )
        events.append(
            EventV5(
                event_id=record_id(
                    "evt",
                    kind="model_call_finished",
                    scope=(resolved_trace_id,),
                    key=source_span_id,
                ),
                event_type=EventType.MODEL_CALL_FINISHED,
                actor_id=actor_id,
                session_id=session_id,
                occurred_at=imported_at,
                span_id=span_id,
                order=EventOrderV1(chronological_sequence=sequence, source_order_id=source_span_id),
                payload={"call_index": int(raw.get("call_index") or sequence - 1)},
            ).sealed()
        )

    correlation = payload.get("trace_correlation_id")
    if correlation:
        aliases.append(
            AliasV1(
                namespace=AliasNamespace.CORRELATION,
                value=str(correlation),
                target_id=resolved_trace_id,
                target_kind="trace",
            )
        )

    actor = ActorV5(
        actor_id=actor_id,
        kind=ActorKind.AGENT,
        display_name="imported v4 rollout",
        role="rollout",
    ).sealed()
    session = SessionV5(
        session_id=session_id,
        actor_id=actor_id,
        started_at=imported_at,
        coverage=SessionCoverageV5(
            model_calls=CoverageState.PARTIAL,
            usage=CoverageState.AGGREGATE_ONLY,
            raw_provider=CoverageState.UNAVAILABLE,
            reasons=("imported from V4; raw provider bytes were never captured",),
        ),
    ).sealed()

    return TraceDocumentV5(
        trace_id=resolved_trace_id,
        trace_kind=TraceKind.AGENT_ROLLOUT,
        identity=TraceIdentityV5(
            rollout_id=rollout_id or None,
            correlation_id=str(correlation) if correlation else None,
        ),
        lifecycle=TraceLifecycleV5(status=TraceStatus.COMPLETED, started_at=imported_at),
        capture=TraceCaptureSummaryV5(
            capture_id=record_id(
                "cap", kind="imported", scope=(resolved_trace_id,), key=source_digest
            ),
            binding_id="imported",
            binding_digest=source_digest,
            capture_profile="imported_v4",
            interception="none",
            mode="disabled",
        ),
        provenance=TraceProvenanceV5(
            producer=producer,
            producer_version=IMPORTER_VERSION,
            source_format="synth_rollout_trace_v4",
            captured_at=imported_at,
            transformation_chain=(f"{IMPORTER_NAME}@{IMPORTER_VERSION}",),
            extra={"source_digest": source_digest},
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.PARTIAL,
            # The source rollout is a completed terminal artifact even though its
            # raw provider transport is unavailable.
            terminal_event_observed=True,
            model_calls=CoverageState.PARTIAL,
            raw_provider=CoverageState.UNAVAILABLE,
            usage=CoverageState.AGGREGATE_ONLY,
            reasons=tuple(losses),
        ),
        actors=(actor,),
        sessions=(session,),
        messages=tuple(messages),
        spans=tuple(spans),
        events=tuple(events),
        usage=total,
        aliases=tuple(aliases),
    ).sealed()


def _message_from_v4(
    message: Mapping[str, Any],
    *,
    trace_id: str,
    key: dict[str, Any],
    actor_id: str,
    session_id: str,
    span_id: str,
    predecessors: tuple[str, ...],
) -> MessageNodeV5:
    message_id = record_id("msg", kind="message", scope=(trace_id,), key=key)
    parts: list[MessagePartV5] = []
    for index, part in enumerate(list(message.get("parts") or [])):
        if not isinstance(part, Mapping):
            continue
        part_id = f"{message_id}:{index}"
        part_type = str(part.get("type") or "")
        if part_type == "text":
            parts.append(
                MessagePartV5(part_id=part_id, type=PartType.TEXT, text=str(part.get("text") or ""))
            )
        elif part_type == "reasoning":
            parts.append(
                MessagePartV5(
                    part_id=part_id,
                    type=PartType.REASONING,
                    text=part.get("content"),
                    reasoning_availability=ReasoningAvailability.CAPTURED,
                )
            )
        elif part_type == "tool_call":
            parts.append(
                MessagePartV5(
                    part_id=part_id,
                    type=PartType.TOOL_CALL,
                    tool_call_id=str(part.get("id") or ""),
                    tool_name=str(part.get("name") or ""),
                    arguments_json=str(part.get("arguments_json") or "{}"),
                )
            )
        elif part_type == "tool_result":
            parts.append(
                MessagePartV5(
                    part_id=part_id,
                    type=PartType.TOOL_RESULT,
                    tool_call_id=str(part.get("tool_call_id") or ""),
                    text=str(part.get("content") or ""),
                    is_error=bool(part.get("is_error")),
                )
            )
        else:
            parts.append(
                MessagePartV5(
                    part_id=part_id,
                    type=PartType.UNSUPPORTED,
                    raw_kind=part_type or "unknown",
                    structured=dict(part),
                )
            )
    if not parts:
        parts.append(MessagePartV5(part_id=f"{message_id}:0", type=PartType.TEXT, text=""))
    return MessageNodeV5(
        message_id=message_id,
        role=str(message.get("role") or "user"),
        parts=tuple(parts),
        sender_actor_id=actor_id,
        session_id=session_id,
        predecessor_message_ids=predecessors,
        produced_by_span_id=span_id,
    ).sealed()


def _mapping(value: Any) -> Mapping[str, Any]:
    """A foreign payload section, or an empty one when absent or the wrong shape."""
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["IMPORTER_NAME", "IMPORTER_VERSION", "import_rollout_trace_v4"]
