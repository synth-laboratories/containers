"""Build the annotator's own execution trace as a sealed Trace V5 document.

The annotator is an agent too. Its tool calls, structured output, and token
usage are captured as a separate, independently sealed trace linked to the
source trace it annotated. Hidden reasoning is never captured: the document
declares ``reasoning_policy = not_captured`` and carries no reasoning parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..canonical import content_digest, record_id, utc_now
from ..models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
    Visibility,
)
from ..models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..models.document import TraceCaptureSummaryV5, TraceDocumentV5, TraceLinkV5
from ..models.events import EventOrderV1, EventStatus, EventType, EventV5
from ..models.identity import TraceIdentityV5, TraceKind, TraceProvenanceV5
from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType
from ..models.spans import SpanKind, SpanStatus, SpanV5, UsageProvenance, UsageV5
from .jobs import AnnotationJobUsageV1, AnnotationJobV1
from .tools import ToolCallRecordV1


EXECUTION_TRACE_PRODUCER = "synth_containers.tracing.annotation.execution_trace"
EXECUTION_TRACE_PRODUCER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class ExecutionCapture:
    """What a runner observed while the annotator ran."""

    started_at: str
    ended_at: str
    instructions_digest: str
    tool_calls: tuple[ToolCallRecordV1, ...] = ()
    final_output: dict[str, Any] | None = None
    final_output_text: str | None = None
    usage: AnnotationJobUsageV1 = field(default_factory=AnnotationJobUsageV1)
    runner_kind: str = "codex_app_server"
    model: str | None = None
    reasoning_effort: str | None = None
    transport_events: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    provider_thread_id: str | None = None
    provider_turn_id: str | None = None


def build_execution_trace(
    job: AnnotationJobV1,
    capture: ExecutionCapture,
    *,
    instructions_text: str | None = None,
) -> TraceDocumentV5:
    trace_id = record_id(
        "trace",
        kind="annotator_execution",
        scope=(job.request.source_trace_id,),
        key={"job_id": job.job_id, "started_at": capture.started_at},
    )
    actor_id = record_id("actor", kind="annotator", scope=(trace_id,), key=job.request.annotator_id)
    session_id = record_id("sess", kind="annotator_session", scope=(trace_id, actor_id), key=job.job_id)
    actor = ActorV5(
        actor_id=actor_id,
        kind=ActorKind.EVALUATOR,
        display_name=f"annotator {job.request.annotator_id}",
        role="annotator",
        subtype=capture.runner_kind,
        harness="synth.annotation",
        runtime=capture.runner_kind,
        model=capture.model or job.request.model,
        metadata={
            "annotator_id": job.request.annotator_id,
            "annotator_digest": job.request.annotator_digest,
            "program_digest": job.program_digest,
            "reasoning_effort": capture.reasoning_effort or job.request.reasoning_effort,
        },
    ).sealed()
    tool_coverage = CoverageState.COMPLETE if capture.tool_calls else CoverageState.NOT_CAPTURED
    usage_coverage = (
        CoverageState.AGGREGATE_ONLY
        if capture.usage.total_tokens is not None
        else CoverageState.UNAVAILABLE
    )
    session = SessionV5(
        session_id=session_id,
        actor_id=actor_id,
        started_at=capture.started_at,
        ended_at=capture.ended_at,
        thread_id=capture.provider_thread_id,
        status=SessionStatus.COMPLETED if capture.error is None else SessionStatus.FAILED,
        harness="synth.annotation",
        coverage=SessionCoverageV5(
            model_calls=CoverageState.AGGREGATE_ONLY,
            agent_events=CoverageState.COMPLETE,
            tool_events=tool_coverage,
            usage=usage_coverage,
            raw_provider=CoverageState.UNAVAILABLE,
            reasons=("app-server exposes turn-level usage, not per-call provider payloads",),
        ),
    ).sealed()

    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    events: list[EventV5] = []
    instructions_id = record_id("msg", kind="annotator_instructions", scope=(session_id,), key=capture.instructions_digest)
    messages.append(
        MessageNodeV5(
            message_id=instructions_id,
            role=MessageRole.USER,
            parts=(
                MessagePartV5(
                    part_id=f"{instructions_id}:0",
                    type=PartType.TEXT,
                    text=instructions_text if instructions_text is not None else "",
                    artifact_id=None,
                ),
            ),
            sender_actor_id=actor_id,
            session_id=session_id,
            occurred_at=capture.started_at,
            metadata={"instructions_digest": capture.instructions_digest},
        ).sealed()
    )
    turn_span_id = record_id("span", kind="annotator_turn", scope=(trace_id, session_id), key=job.job_id)
    sequence = 0
    predecessor = instructions_id
    for call in capture.tool_calls:
        span_id = record_id(
            "span", kind="annotator_tool_call", scope=(trace_id, session_id), key=call.index
        )
        spans.append(
            SpanV5(
                span_id=span_id,
                span_kind=SpanKind.TOOL_EXECUTION,
                actor_id=actor_id,
                session_id=session_id,
                started_at=call.started_at,
                ended_at=call.ended_at,
                parent_span_id=turn_span_id,
                status=SpanStatus.OK if call.ok else SpanStatus.ERROR,
                detail={
                    "tool": call.tool,
                    "arguments": call.arguments,
                    "response_bytes": call.response_bytes,
                    "response_digest": call.response_digest,
                    "truncated": call.truncated,
                    "error": call.error,
                    "call_index": call.index,
                },
            ).sealed()
        )
        sequence += 1
        events.append(
            EventV5(
                event_id=record_id("evt", kind="annotator_tool_call", scope=(trace_id,), key=call.index),
                event_type=EventType.TOOL_CALL_EXECUTED,
                actor_id=actor_id,
                session_id=session_id,
                occurred_at=call.ended_at,
                span_id=span_id,
                order=EventOrderV1(chronological_sequence=sequence, actor_sequence=sequence),
                payload={"tool": call.tool, "ok": call.ok, "response_bytes": call.response_bytes},
                status=EventStatus.OK if call.ok else EventStatus.ERROR,
            ).sealed()
        )
    output_id = record_id("msg", kind="annotator_output", scope=(session_id,), key=job.job_id)
    output_parts: tuple[MessagePartV5, ...]
    if capture.final_output is not None:
        output_parts = (
            MessagePartV5(
                part_id=f"{output_id}:0",
                type=PartType.STRUCTURED,
                structured=capture.final_output,
            ),
        )
    else:
        output_parts = (
            MessagePartV5(
                part_id=f"{output_id}:0",
                type=PartType.TEXT,
                text=capture.final_output_text or "",
                conversion_diagnostics=("no structured output",),
            ),
        )
    messages.append(
        MessageNodeV5(
            message_id=output_id,
            role=MessageRole.ASSISTANT,
            parts=output_parts,
            sender_actor_id=actor_id,
            session_id=session_id,
            predecessor_message_ids=(predecessor,),
            produced_by_span_id=turn_span_id,
            occurred_at=capture.ended_at,
        ).sealed()
    )
    usage = UsageV5(
        provenance=(
            UsageProvenance.OBSERVED_HARNESS
            if capture.usage.total_tokens is not None
            else UsageProvenance.UNAVAILABLE
        ),
        prompt_tokens=capture.usage.input_tokens,
        completion_tokens=capture.usage.output_tokens,
        reasoning_tokens=capture.usage.reasoning_output_tokens,
        cached_tokens=capture.usage.cached_input_tokens,
        total_tokens=capture.usage.total_tokens,
        requests=1 if capture.usage.total_tokens is not None else None,
        wall_time_seconds=capture.usage.wall_time_seconds,
        cost_usd=capture.usage.cost_usd,
        unavailable_fields=("cost_usd",) if capture.usage.cost_usd is None else (),
    )
    spans.insert(
        0,
        SpanV5(
            span_id=turn_span_id,
            span_kind=SpanKind.EVALUATOR_EXECUTION,
            actor_id=actor_id,
            session_id=session_id,
            started_at=capture.started_at,
            ended_at=capture.ended_at,
            status=SpanStatus.OK if capture.error is None else SpanStatus.ERROR,
            input_message_ids=(instructions_id,),
            output_message_ids=(output_id,),
            usage=usage,
            detail={
                "runner_kind": capture.runner_kind,
                "model": capture.model or job.request.model,
                "reasoning_effort": capture.reasoning_effort or job.request.reasoning_effort,
                "provider_thread_id": capture.provider_thread_id,
                "provider_turn_id": capture.provider_turn_id,
                "tool_call_count": len(capture.tool_calls),
                "error": capture.error,
                "transport_event_count": len(capture.transport_events),
            },
        ).sealed(),
    )
    lifecycle = TraceLifecycleV5(
        status=TraceStatus.COMPLETED if capture.error is None else TraceStatus.FAILED,
        started_at=capture.started_at,
        ended_at=capture.ended_at,
        termination=TerminationV5(reason="completed" if capture.error is None else "failed", detail=capture.error or ""),
    )
    document = TraceDocumentV5(
        trace_id=trace_id,
        trace_kind=TraceKind.EVALUATION_ATTEMPT,
        identity=TraceIdentityV5(correlation_id=job.job_id),
        lifecycle=lifecycle,
        capture=TraceCaptureSummaryV5(
            capture_id=record_id("cap", kind="annotator_execution", scope=(trace_id,), key=job.job_id),
            binding_id="annotation_job",
            binding_digest=job.content_digest or content_digest(job),
            capture_profile="annotator_execution",
            interception="app_server_protocol" if capture.runner_kind == "codex_app_server" else "in_process",
            mode="structured",
        ),
        provenance=TraceProvenanceV5(
            producer=EXECUTION_TRACE_PRODUCER,
            producer_version=EXECUTION_TRACE_PRODUCER_VERSION,
            source_format="synth.annotation-execution.v1",
            model=capture.model or job.request.model,
            harness="synth.annotation",
            captured_at=utc_now(),
            extra={
                "reasoning_policy": "not_captured",
                "job_id": job.job_id,
                "annotator_id": job.request.annotator_id,
                "transport_events": list(capture.transport_events)[:200],
            },
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.COMPLETE if capture.error is None else CaptureStatus.PARTIAL,
            terminal_event_observed=True,
            model_calls=CoverageState.AGGREGATE_ONLY,
            raw_provider=CoverageState.UNAVAILABLE,
            agent_events=CoverageState.COMPLETE,
            tool_events=tool_coverage,
            usage=usage_coverage,
            reasons=("annotator hidden reasoning is not captured by policy",),
        ),
        actors=(actor,),
        sessions=(session,),
        messages=tuple(messages),
        spans=tuple(spans),
        events=tuple(events),
        usage=usage,
        links=(
            TraceLinkV5(
                relation="annotates",
                target_id=job.request.source_trace_id,
                target_digest=job.request.source_trace_digest,
                target_kind="trace",
                detail={"job_id": job.job_id},
            ),
        ),
        visibility=Visibility.PRIVATE,
    )
    return document.sealed()


__all__ = ["ExecutionCapture", "build_execution_trace"]
