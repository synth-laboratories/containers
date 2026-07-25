"""Seal raw capture segments into an immutable ``TraceDocumentV5``.

Sealing is the only transition that produces a trace digest. It reads the raw
segments back (verifying every segment digest), normalizes provider traffic through
declared transformations, folds in application events and artifacts, and records what
the capture did and did not observe.

Nothing here invents facts: a call with no observed provider usage yields
``UsageProvenance.UNAVAILABLE``, not zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from ..adapters.openai_chat import (
    NORMALIZER_NAME,
    NORMALIZER_VERSION,
    NormalizedMessage,
    assemble_sse_frames,
    normalize_request_messages,
    normalize_unary_response,
    usage_from_provider,
)
from ..canonical import record_id, utc_now
from ..models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
)
from ..models.artifacts import ArtifactRefV5
from ..models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..models.document import TraceCaptureSummaryV5, TraceDocumentV5
from ..models.events import EventOrderV1, EventStatus, EventType, EventV5
from ..models.identity import AliasNamespace, AliasV1, TraceIdentityV5, TraceProvenanceV5
from ..models.messages import MessageNodeV5
from ..models.spans import (
    SpanKind,
    SpanStatus,
    SpanV5,
    TransformationRecordV1,
    UsageProvenance,
    UsageV5,
)
from .binding import TraceCaptureBindingV1
from .coverage import CaptureCoverageReceiptV1, CaptureScope, Completeness
from .envelope import RawRecordType
from .spool import TraceSegmentManifestV1, read_segments


FINALIZER_NAME = "synth-trace-finalizer"
FINALIZER_VERSION = "1"


@dataclass(slots=True)
class _CallState:
    call_id: str
    call_index: int
    started_at: str
    model: str | None
    streaming: bool
    request_body: dict[str, Any]
    request_digest: str
    frames: list[str]
    response_body: dict[str, Any] | None = None
    response_truncated: bool = False
    http_status: int | None = None
    ended_at: str | None = None
    usage_payload: dict[str, Any] | None = None
    usage_observed: bool = False


@dataclass(frozen=True, slots=True)
class SealedCapture:
    """Everything sealing produced, ready to be written into a bundle."""

    document: TraceDocumentV5
    coverage: CaptureCoverageReceiptV1
    segments: tuple[TraceSegmentManifestV1, ...]
    envelope_to_event: dict[str, str]
    call_index_to_span: dict[int, str]


class TraceFinalizer:
    """Builds one sealed trace document from one capture session's raw segments."""

    def __init__(
        self,
        *,
        binding: TraceCaptureBindingV1,
        spool_root: Path,
        segments: tuple[TraceSegmentManifestV1, ...],
        provenance: TraceProvenanceV5,
        identity: TraceIdentityV5,
        root_actor_name: str = "workload",
        root_actor_kind: ActorKind | str = ActorKind.AGENT,
    ) -> None:
        self.binding = binding
        self.spool_root = Path(spool_root)
        self.segments = segments
        self.provenance = provenance
        self.identity = identity
        self.root_actor_name = root_actor_name
        self.root_actor_kind = root_actor_kind

    def seal(
        self,
        *,
        coverage: CaptureCoverageReceiptV1,
        status: TraceStatus | str = TraceStatus.COMPLETED,
        termination: TerminationV5 | None = None,
        extra_actors: tuple[ActorV5, ...] = (),
        extra_sessions: tuple[SessionV5, ...] = (),
        aliases: tuple[AliasV1, ...] = (),
    ) -> SealedCapture:
        records = list(read_segments(self.spool_root, self.segments))
        calls: dict[str, _CallState] = {}
        order: list[str] = []
        spans: list[SpanV5] = []
        events: list[EventV5] = []
        messages: list[MessageNodeV5] = []
        artifacts: list[ArtifactRefV5] = []
        envelope_to_event: dict[str, str] = {}
        call_index_to_span: dict[int, str] = {}
        transformations: list[TransformationRecordV1] = []
        diagnostics: list[str] = []
        application_events = 0
        started_at = records[0]["occurred_at"] if records else utc_now()
        ended_at = records[-1]["occurred_at"] if records else started_at
        sequence = 0

        for record in records:
            record_type = str(record.get("record_type") or "")
            payload = record.get("payload") or {}
            call_id = record.get("call_id")
            if record_type == RawRecordType.MODEL_CALL_STARTED:
                calls[call_id] = _CallState(
                    call_id=str(call_id),
                    call_index=int(payload.get("call_index") or 0),
                    started_at=str(payload.get("started_at") or record["occurred_at"]),
                    model=payload.get("model"),
                    streaming=bool(payload.get("stream")),
                    request_body=dict(payload.get("request_body") or {}),
                    request_digest=str(payload.get("request_digest") or ""),
                    frames=[],
                )
                order.append(str(call_id))
            elif record_type == RawRecordType.RESPONSE_FRAME and call_id in calls:
                calls[call_id].frames.append(str(payload.get("frame") or ""))
            elif record_type == RawRecordType.RESPONSE_BODY and call_id in calls:
                state = calls[call_id]
                body = payload.get("response_body")
                state.response_body = dict(body) if isinstance(body, Mapping) else None
                state.response_truncated = bool(payload.get("truncated"))
                state.http_status = _int(payload.get("http_status"))
            elif record_type == RawRecordType.MODEL_CALL_FINISHED and call_id in calls:
                state = calls[call_id]
                state.ended_at = str(payload.get("ended_at") or record["occurred_at"])
                state.http_status = _int(payload.get("http_status")) or state.http_status
                usage_payload = payload.get("usage")
                state.usage_payload = (
                    dict(usage_payload) if isinstance(usage_payload, Mapping) else None
                )
                state.usage_observed = bool(payload.get("usage_observed"))
            elif record_type == RawRecordType.ARTIFACT:
                artifacts.append(_artifact_from_payload(payload, occurred_at=record["occurred_at"]))
            elif record_type == RawRecordType.APPLICATION_EVENT:
                sequence += 1
                application_events += 1
                event = self._application_event(
                    record, payload, sequence=sequence, envelope_to_event=envelope_to_event
                )
                envelope_to_event[str(record["envelope_id"])] = event.event_id
                events.append(event)

        total_usage = UsageV5(provenance=UsageProvenance.DERIVED, requests=0)
        for call_id in order:
            state = calls[call_id]
            sequence += 1
            built = self._build_call(state, sequence=sequence)
            spans.append(built.span)
            events.extend(built.events)
            messages.extend(built.messages)
            transformations.append(built.transformation)
            diagnostics.extend(built.diagnostics)
            call_index_to_span[state.call_index] = built.span.span_id
            total_usage = total_usage.merged(built.usage)

        if not order:
            total_usage = UsageV5(
                provenance=UsageProvenance.UNAVAILABLE,
                requests=0,
                unavailable_fields=("prompt_tokens", "completion_tokens", "total_tokens"),
            )

        root_actor = ActorV5(
            actor_id=self.binding.workload.root_actor_id,
            kind=self.root_actor_kind,
            display_name=self.root_actor_name,
            role=str(self.binding.workload.kind),
            harness=self.provenance.harness,
            runtime=self.provenance.runtime_version,
            model=self.provenance.model,
            provider=self.provenance.provider,
            task_id=self.binding.context.task_id,
            workflow_id=self.binding.workload.workflow_id,
        ).sealed()

        model_calls_state = CoverageState.COMPLETE if order else CoverageState.NOT_CAPTURED
        usage_state = (
            CoverageState.COMPLETE
            if order and all(calls[item].usage_observed for item in order)
            else (CoverageState.PARTIAL if order else CoverageState.NOT_CAPTURED)
        )
        application_state = (
            CoverageState.COMPLETE if application_events else CoverageState.NOT_CAPTURED
        )
        root_session = SessionV5(
            session_id=self.binding.workload.actor_session_id,
            actor_id=root_actor.actor_id,
            started_at=started_at,
            ended_at=ended_at,
            capture_id=self.binding.capture_id,
            status=(
                SessionStatus.COMPLETED
                if str(status) == TraceStatus.COMPLETED
                else SessionStatus.INTERRUPTED
            ),
            harness=self.provenance.harness,
            provider=self.provenance.provider,
            coverage=SessionCoverageV5(
                model_calls=model_calls_state,
                agent_events=application_state,
                environment_events=application_state,
                tool_events=application_state,
                usage=usage_state,
                raw_provider=CoverageState.COMPLETE if order else CoverageState.NOT_CAPTURED,
            ),
        ).sealed()

        scope = (
            CaptureScope.MODEL_CALLS_AND_APPLICATION
            if application_events and order
            else (CaptureScope.MODEL_CALLS_ONLY if order else CaptureScope.APPLICATION_ONLY)
        )
        segment_digests = tuple(segment.digest for segment in self.segments)
        sealed_coverage = replace(
            coverage,
            scope=scope,
            application_events=application_events,
            artifacts_recorded=len(artifacts),
            segment_count=len(self.segments),
            segment_bytes=sum(segment.byte_size for segment in self.segments),
            segment_digests=segment_digests,
            model_calls=model_calls_state,
            raw_provider=CoverageState.COMPLETE if order else CoverageState.NOT_CAPTURED,
            agent_events=application_state,
            environment_events=application_state,
            tool_events=application_state,
            usage=usage_state,
            completeness=(
                Completeness.COMPLETE
                if str(status) == TraceStatus.COMPLETED
                else Completeness.PARTIAL
            ),
            completeness_reasons=tuple(sorted(set(diagnostics))),
            finalization_status="sealed",
            ended_at=utc_now(),
        ).sealed()

        completeness = TraceCompletenessV5(
            capture_status=(
                CaptureStatus.COMPLETE
                if str(status) == TraceStatus.COMPLETED
                else CaptureStatus.PARTIAL
            ),
            terminal_event_observed=bool(records),
            model_calls=model_calls_state,
            raw_provider=CoverageState.COMPLETE if order else CoverageState.NOT_CAPTURED,
            agent_events=application_state,
            environment_events=application_state,
            tool_events=application_state,
            usage=usage_state,
            captured_record_count=len(records),
            high_water_ordinal=max((int(item["ordinal"]) for item in records), default=None),
            truncation_reasons=tuple(
                sorted(
                    {"response_body_truncated"}
                    if any(calls[i].response_truncated for i in order)
                    else set()
                )
            ),
            reasons=tuple(sorted(set(diagnostics))),
        )

        document = TraceDocumentV5(
            trace_id=self.binding.trace_id,
            trace_kind=self.binding.trace_kind,
            identity=self.identity,
            lifecycle=TraceLifecycleV5(
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                termination=termination,
            ),
            capture=TraceCaptureSummaryV5(
                capture_id=self.binding.capture_id,
                binding_id=self.binding.binding_id,
                binding_digest=self.binding.content_digest,
                capture_profile=self.binding.policy.profile,
                interception=str(self.binding.capture.interception),
                mode=str(self.binding.capture.mode),
                proxy_config_digest=self.binding.capture.proxy_config_digest,
                coverage_receipt_id=sealed_coverage.receipt_id,
                segment_digests=segment_digests,
                segment_count=len(self.segments),
                raw_record_count=len(records),
            ),
            provenance=replace(
                self.provenance,
                transformation_chain=tuple(
                    sorted({f"{item.name}@{item.version}" for item in transformations})
                ),
            ),
            completeness=completeness,
            actors=(root_actor, *extra_actors),
            sessions=(root_session, *extra_sessions),
            messages=tuple(messages),
            spans=tuple(spans),
            events=tuple(sorted(events, key=lambda item: item.order.chronological_sequence or 0)),
            artifacts=tuple(artifacts),
            usage=total_usage,
            aliases=aliases,
        ).sealed()

        return SealedCapture(
            document=document,
            coverage=sealed_coverage,
            segments=self.segments,
            envelope_to_event=envelope_to_event,
            call_index_to_span=call_index_to_span,
        )

    # -- per-call normalization --------------------------------------------------

    def _build_call(self, state: _CallState, *, sequence: int) -> "_BuiltCall":
        trace_id = self.binding.trace_id
        actor_id = self.binding.workload.root_actor_id
        session_id = self.binding.workload.actor_session_id
        span_id = record_id("span", kind="model_call", scope=(trace_id,), key=state.call_id)
        diagnostics: list[str] = []
        messages: list[MessageNodeV5] = []

        request_messages = normalize_request_messages(state.request_body)
        input_ids: list[str] = []
        previous: tuple[str, ...] = ()
        for index, normalized in enumerate(request_messages):
            node = _message_node(
                trace_id=trace_id,
                key={"call": state.call_id, "slot": "request", "index": index},
                normalized=normalized,
                actor_id=actor_id,
                session_id=session_id,
                span_id=span_id,
                occurred_at=state.started_at,
                predecessors=previous,
            )
            diagnostics.extend(normalized.diagnostics)
            messages.append(node)
            input_ids.append(node.message_id)
            previous = (node.message_id,)

        response: NormalizedMessage | None = None
        usage_payload = state.usage_payload
        if state.streaming:
            response, streamed_usage, stream_diagnostics = assemble_sse_frames(state.frames)
            diagnostics.extend(stream_diagnostics)
            usage_payload = usage_payload or streamed_usage
        elif state.response_truncated:
            diagnostics.append("response body exceeded the inline limit and was not normalized")
        elif state.response_body is not None:
            response, response_diagnostics = normalize_unary_response(state.response_body)
            diagnostics.extend(response_diagnostics)
            if usage_payload is None and isinstance(state.response_body.get("usage"), Mapping):
                usage_payload = dict(state.response_body["usage"])

        output_ids: list[str] = []
        if response is not None:
            node = _message_node(
                trace_id=trace_id,
                key={"call": state.call_id, "slot": "response"},
                normalized=response,
                actor_id=actor_id,
                session_id=session_id,
                span_id=span_id,
                occurred_at=state.ended_at or state.started_at,
                predecessors=previous,
            )
            messages.append(node)
            output_ids.append(node.message_id)

        usage = usage_from_provider(usage_payload)
        status = (
            SpanStatus.OK
            if state.http_status is not None and 200 <= state.http_status < 300
            else SpanStatus.ERROR
        )
        span = SpanV5(
            span_id=span_id,
            span_kind=SpanKind.MODEL_CALL,
            actor_id=actor_id,
            session_id=session_id,
            started_at=state.started_at,
            ended_at=state.ended_at,
            status=status,
            detail={
                "call_id": state.call_id,
                "call_index": state.call_index,
                "model": state.model,
                "streaming": state.streaming,
                "http_status": state.http_status,
                "request_digest": state.request_digest,
                "finish_reason": response.finish_reason if response else None,
            },
            input_message_ids=tuple(input_ids),
            output_message_ids=tuple(output_ids),
            usage=usage,
            transformations=(
                TransformationRecordV1(
                    name=NORMALIZER_NAME,
                    version=NORMALIZER_VERSION,
                    input_refs=(state.call_id,),
                    output_refs=tuple(input_ids + output_ids),
                    losses=tuple(sorted(set(diagnostics))),
                    deterministic=True,
                ),
            ),
        ).sealed()

        started_event = EventV5(
            event_id=record_id(
                "evt", kind="model_call_started", scope=(trace_id,), key=state.call_id
            ),
            event_type=EventType.MODEL_CALL_STARTED,
            actor_id=actor_id,
            session_id=session_id,
            occurred_at=state.started_at,
            span_id=span_id,
            order=EventOrderV1(chronological_sequence=sequence, source_order_id=state.call_id),
            payload={"call_index": state.call_index, "model": state.model},
        ).sealed()
        finished_event = EventV5(
            event_id=record_id(
                "evt", kind="model_call_finished", scope=(trace_id,), key=state.call_id
            ),
            event_type=EventType.MODEL_CALL_FINISHED,
            actor_id=actor_id,
            session_id=session_id,
            occurred_at=state.ended_at or state.started_at,
            span_id=span_id,
            order=EventOrderV1(chronological_sequence=sequence, source_order_id=state.call_id),
            caused_by_event_ids=(started_event.event_id,),
            status=EventStatus.OK if status == SpanStatus.OK else EventStatus.ERROR,
            payload={
                "call_index": state.call_index,
                "http_status": state.http_status,
                "usage_observed": usage_payload is not None,
            },
        ).sealed()

        return _BuiltCall(
            span=span,
            events=[started_event, finished_event],
            messages=messages,
            usage=usage,
            transformation=span.transformations[0],
            diagnostics=diagnostics,
        )

    def _application_event(
        self,
        record: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        sequence: int,
        envelope_to_event: Mapping[str, str],
    ) -> EventV5:
        trace_id = self.binding.trace_id
        envelope_id = str(record["envelope_id"])
        structural = payload.get("structural")
        caused_by = tuple(
            envelope_to_event[item]
            for item in list(payload.get("caused_by") or [])
            if item in envelope_to_event
        )
        return EventV5(
            event_id=record_id("evt", kind="application", scope=(trace_id,), key=envelope_id),
            event_type=str(payload.get("event_type") or EventType.APPLICATION),
            actor_id=str(record.get("actor_id") or self.binding.workload.root_actor_id),
            session_id=str(record.get("session_id") or self.binding.workload.actor_session_id),
            occurred_at=str(record["occurred_at"]),
            order=EventOrderV1(
                chronological_sequence=sequence,
                source_order_id=envelope_id,
                structural=_structural(structural),
            ),
            caused_by_event_ids=caused_by,
            payload=dict(payload.get("body") or {}),
            raw_source_ref=envelope_id,
        ).sealed()


@dataclass(slots=True)
class _BuiltCall:
    span: SpanV5
    events: list[EventV5]
    messages: list[MessageNodeV5]
    usage: UsageV5
    transformation: TransformationRecordV1
    diagnostics: list[str]


def _message_node(
    *,
    trace_id: str,
    key: dict[str, Any],
    normalized: NormalizedMessage,
    actor_id: str,
    session_id: str,
    span_id: str,
    occurred_at: str,
    predecessors: tuple[str, ...],
) -> MessageNodeV5:
    message_id = record_id("msg", kind="message", scope=(trace_id,), key=key)
    parts = tuple(
        replace(part, part_id=f"{message_id}:{part.part_id}") for part in normalized.parts
    )
    return MessageNodeV5(
        message_id=message_id,
        role=normalized.role,
        parts=parts,
        sender_actor_id=actor_id,
        session_id=session_id,
        predecessor_message_ids=predecessors,
        produced_by_span_id=span_id,
        occurred_at=occurred_at,
        metadata=(
            {"conversion_diagnostics": normalized.diagnostics} if normalized.diagnostics else {}
        ),
    ).sealed()


def _artifact_from_payload(payload: Mapping[str, Any], *, occurred_at: str) -> ArtifactRefV5:
    return ArtifactRefV5(
        artifact_id=str(payload.get("artifact_id") or ""),
        digest=str(payload.get("digest") or ""),
        media_type=str(payload.get("media_type") or "application/octet-stream"),
        size_bytes=int(payload.get("size_bytes") or 0),
        role=str(payload.get("role") or "other"),
        uri=str(payload.get("uri") or "") or None,
        producer=FINALIZER_NAME,
        visibility=str(payload.get("visibility") or "private"),
        logical_name=str(payload.get("logical_name") or "") or None,
        observed_at=occurred_at,
    )


def _structural(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    from ..models.events import StructuralAddressV1

    workflow_id = value.get("workflow_id")
    if not workflow_id:
        return None
    return StructuralAddressV1(
        workflow_id=str(workflow_id),
        node_path=tuple(str(item) for item in list(value.get("node_path") or [])),
        iteration=int(value.get("iteration") or 0),
        local_sequence=int(value.get("local_sequence") or 0),
    )


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def alias(
    namespace: AliasNamespace | str, value: str, *, target_id: str, target_kind: str
) -> AliasV1:
    return AliasV1(namespace=namespace, value=value, target_id=target_id, target_kind=target_kind)


__all__ = [
    "FINALIZER_NAME",
    "FINALIZER_VERSION",
    "SealedCapture",
    "TraceFinalizer",
    "alias",
]
