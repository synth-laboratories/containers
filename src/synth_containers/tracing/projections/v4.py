"""Project a sealed V5 trace into the existing ``RolloutTraceV4`` shape.

V4 stays useful for current local consumers, but it is a projection: it names the V5
digest it came from and declares what it cannot express. Nothing reads V4 to learn a
fact that only exists in V5.
"""

from __future__ import annotations

from typing import Any

from synth_containers.rollout_tracing.v4 import (
    CanonicalChoice,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalUsage,
    ContentPart,
    ReasoningPart,
    RolloutTraceSpanV4,
    RolloutTraceV4,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UnsupportedPart,
)

from ..canonical import record_id, utc_now
from ..models.document import TraceDocumentV5
from ..models.messages import MessageNodeV5, PartType
from ..models.projection import ProjectionLossV1, ProjectionManifestV1
from ..models.spans import SpanKind, UsageProvenance


PROJECTION_FORMAT = "synth-v4"
PROJECTION_PRODUCER = "synth_containers.tracing.projections.v4"
PROJECTION_VERSION = "1"

_V4_ROLES = {"system", "user", "assistant", "tool"}


def project_v4(document: TraceDocumentV5) -> tuple[RolloutTraceV4, ProjectionManifestV1]:
    """Return the V4 projection plus a manifest naming its source digest and loss."""

    if not document.content_digest:
        raise ValueError("only a sealed trace can be projected")

    losses: list[ProjectionLossV1] = []
    spans: list[RolloutTraceSpanV4] = []
    by_id = {item.message_id: item for item in document.messages}

    model_calls = document.spans_of_kind(SpanKind.MODEL_CALL.value)
    for span in sorted(model_calls, key=lambda item: int(item.detail.get("call_index") or 0)):
        request_messages: list[CanonicalMessage] = []
        for message_id in span.input_message_ids:
            node = by_id.get(message_id)
            if node is None:
                continue
            converted, message_losses = _to_v4_message(node)
            request_messages.append(converted)
            losses.extend(message_losses)
        response_message = CanonicalMessage(role="assistant", parts=(TextPart(text=""),))
        for message_id in span.output_message_ids:
            node = by_id.get(message_id)
            if node is None:
                continue
            response_message, message_losses = _to_v4_message(node)
            losses.extend(message_losses)
        usage = CanonicalUsage()
        if span.usage is not None:
            usage = CanonicalUsage(
                prompt_tokens=int(span.usage.prompt_tokens or 0),
                completion_tokens=int(span.usage.completion_tokens or 0),
                reasoning_tokens=int(span.usage.reasoning_tokens or 0),
                cached_tokens=int(span.usage.cached_tokens or 0),
            )
            if str(span.usage.provenance) != UsageProvenance.OBSERVED_PROVIDER:
                losses.append(
                    ProjectionLossV1(
                        field_path="spans[].usage.provenance",
                        reason=("V4 usage carries no provenance; unobserved usage renders as zero"),
                        record_count=1,
                    )
                )
        model = str(span.detail.get("model") or "")
        spans.append(
            RolloutTraceSpanV4(
                span_id=span.span_id,
                call_index=int(span.detail.get("call_index") or 0),
                request=CanonicalRequest(
                    messages=tuple(request_messages),
                    model=model,
                    provider_hint="chat",
                ),
                response=CanonicalResponse(
                    choices=(
                        CanonicalChoice(
                            index=0,
                            message=response_message,
                            finish_reason=span.detail.get("finish_reason"),
                        ),
                    ),
                    usage=usage,
                    model=model,
                    provider_hint="chat",
                ),
                run_id=document.identity.run_id,
                api_format="chat",
                metrics={"http_status": span.detail.get("http_status")},
                metadata={
                    "v5_span_id": span.span_id,
                    "v5_trace_digest": document.content_digest,
                    "streaming": span.detail.get("streaming"),
                },
            )
        )

    non_model_events = tuple(
        item for item in document.events if not str(item.event_type).startswith("model_call.")
    )
    if non_model_events:
        losses.append(
            ProjectionLossV1(
                field_path="events",
                reason="V4 has no typed event model; application events are summarized only",
                record_count=len(non_model_events),
            )
        )
    if document.artifacts:
        losses.append(
            ProjectionLossV1(
                field_path="artifacts",
                reason="V4 has no artifact registry; digests are listed in summary only",
                record_count=len(document.artifacts),
            )
        )
    if len(document.actors) > 1:
        losses.append(
            ProjectionLossV1(
                field_path="actors",
                reason="V4 is single-actor; additional actors are not representable",
                record_count=len(document.actors) - 1,
            )
        )

    trace = RolloutTraceV4(
        rollout_id=document.identity.rollout_id or document.trace_id,
        spans=tuple(spans),
        trace_correlation_id=document.identity.correlation_id,
        status=str(document.lifecycle.status),
        summary={
            "source_schema": document.schema_version,
            "source_trace_id": document.trace_id,
            "source_trace_digest": document.content_digest,
            "capture_status": str(document.completeness.capture_status),
            "model_call_count": len(spans),
            "event_count": len(document.events),
            "artifact_digests": [item.digest for item in document.artifacts],
            "usage_provenance": str(document.usage.provenance),
        },
        events=tuple(
            {
                "event_id": item.event_id,
                "event_type": str(item.event_type),
                "occurred_at": item.occurred_at,
                "actor_id": item.actor_id,
                "sequence": item.order.chronological_sequence,
            }
            for item in non_model_events
        ),
        metadata={
            "projection_format": PROJECTION_FORMAT,
            "projection_producer": PROJECTION_PRODUCER,
            "source_trace_digest": document.content_digest,
        },
    )

    manifest = ProjectionManifestV1(
        projection_id=record_id(
            "proj",
            kind="projection",
            scope=(document.trace_id,),
            key={"format": PROJECTION_FORMAT, "digest": document.content_digest},
        ),
        format=PROJECTION_FORMAT,
        source_trace_id=document.trace_id,
        source_trace_digest=document.content_digest,
        producer=PROJECTION_PRODUCER,
        producer_version=PROJECTION_VERSION,
        created_at=utc_now(),
        requested_view={"actor": "root", "layers": ["model_calls"]},
        included_layers=("model_calls", "messages", "usage"),
        omitted_layers=("branches", "selectors", "completeness", "evidence"),
        losses=tuple(losses),
        target_media_type="application/json",
    )
    return trace, manifest


def _to_v4_message(node: MessageNodeV5) -> tuple[CanonicalMessage, list[ProjectionLossV1]]:
    losses: list[ProjectionLossV1] = []
    role = str(node.role)
    if role not in _V4_ROLES:
        losses.append(
            ProjectionLossV1(
                field_path="messages[].role",
                reason=f"V4 has no {role!r} role; projected as user",
                record_count=1,
            )
        )
        role = "user"
    parts: list[ContentPart] = []
    for part in node.parts:
        part_type = str(part.type)
        if part_type == PartType.TEXT:
            parts.append(TextPart(text=part.text or ""))
        elif part_type == PartType.REASONING:
            parts.append(ReasoningPart(content=part.text, kind="provider_reasoning"))
        elif part_type == PartType.TOOL_CALL:
            parts.append(
                ToolCallPart(
                    id=part.tool_call_id or "",
                    name=part.tool_name or "",
                    arguments_json=part.arguments_json or "{}",
                )
            )
        elif part_type == PartType.TOOL_RESULT:
            parts.append(
                ToolResultPart(
                    tool_call_id=part.tool_call_id or "",
                    content=part.text or "",
                    is_error=bool(part.is_error),
                )
            )
        else:
            parts.append(UnsupportedPart(kind=part_type, detail=part.raw_kind or ""))
            losses.append(
                ProjectionLossV1(
                    field_path="messages[].parts",
                    reason=f"V4 cannot express {part_type!r} content parts",
                    record_count=1,
                )
            )
    if not parts:
        parts.append(TextPart(text=""))
    tool_call_id = next(
        (part.tool_call_id for part in node.parts if str(part.type) == PartType.TOOL_RESULT),
        None,
    )
    return (
        CanonicalMessage(role=role, parts=tuple(parts), tool_call_id=tool_call_id),  # type: ignore[arg-type]
        losses,
    )


def v4_payload(document: TraceDocumentV5) -> dict[str, Any]:
    trace, _ = project_v4(document)
    return trace.to_dict()


__all__ = [
    "PROJECTION_FORMAT",
    "PROJECTION_PRODUCER",
    "PROJECTION_VERSION",
    "project_v4",
    "v4_payload",
]
