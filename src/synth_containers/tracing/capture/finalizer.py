"""Seal raw capture segments into an immutable ``TraceDocumentV5``.

Sealing is the only transition that produces a trace digest. It reads the raw
segments back (verifying every segment digest), normalizes provider traffic through
declared transformations, folds in application events and artifacts, and records what
the capture did and did not observe.

Nothing here invents facts: a call with no observed provider usage yields
``UsageProvenance.UNAVAILABLE``, not zeros.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ..adapters import provider_adapters
from ..adapters.base import NormalizedMessage, NormalizedProviderResult
from ..canonical import bytes_digest, record_id, utc_now
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
from ..models.identity import (
    AliasNamespace,
    AliasV1,
    TraceContextV1,
    TraceIdentityV5,
    TraceProvenanceV5,
)
from ..models.messages import MessageNodeV5
from ..models.spans import (
    SpanKind,
    SpanStatus,
    SpanV5,
    TransformationRecordV1,
    UsageProvenance,
    UsageV5,
)
from ..models.tokens import TokenCaptureV5
from ..store.filesystem import FilesystemBlobStore
from .binding import TraceCaptureBindingV1
from .coverage import (
    CaptureCoverageReceiptV1,
    CaptureScope,
    Completeness,
    finalization_from_dict,
)
from .envelope import RawRecordType
from .redaction import CORRELATION_HEADER_PREFIXES, CORRELATION_HEADERS
from .spool import TraceSegmentManifestV1, read_segments


FINALIZER_NAME = "synth-trace-finalizer"
FINALIZER_VERSION = "1"

# Record types keyed by call id. Without one they cannot be correlated to a call.
_CALL_RECORD_TYPES = frozenset(
    {
        RawRecordType.MODEL_CALL_STARTED,
        RawRecordType.UPSTREAM_ATTEMPT_STARTED,
        RawRecordType.UPSTREAM_ATTEMPT_FINISHED,
        RawRecordType.RESPONSE_FRAME,
        RawRecordType.RESPONSE_BODY,
        RawRecordType.MODEL_CALL_FINISHED,
    }
)
_CALL_FOLLOWUP_RECORD_TYPES = _CALL_RECORD_TYPES - {
    RawRecordType.MODEL_CALL_STARTED,
}

_LOCAL_ALIAS_TARGET_KINDS = frozenset(
    {
        "trace",
        "actor",
        "session",
        "span",
        "event",
        "message",
        "part",
        "artifact",
        "branch",
        "error",
    }
)
_EXTERNAL_ALIAS_TARGET_KINDS = frozenset({"external_trace"})


class FinalizationError(RuntimeError):
    """A capture segment cannot be sealed into a trace without losing facts."""


@dataclass(slots=True)
class _CallState:
    call_id: str
    call_index: int
    actor_id: str
    session_id: str
    started_at: str
    model: str | None
    streaming: bool
    request_body: dict[str, Any]
    request_digest: str
    frames: list[str]
    request_headers: dict[str, str] = field(default_factory=dict)
    response_body: dict[str, Any] | None = None
    response_truncated: bool = False
    response_body_ordinal: int = 0
    http_status: int | None = None
    started_ordinal: int = 0
    finished_ordinal: int = 0
    ended_at: str | None = None
    usage_payload: dict[str, Any] | None = None
    usage_observed: bool = False
    provider_adapter: str = "openai_chat_completions"
    provider_adapter_version: str = "1"
    route: str = ""
    request_body_ref: dict[str, Any] | None = None
    response_body_ref: dict[str, Any] | None = None
    native_correlation: dict[str, str] | None = None
    upstream_attempts: dict[str, int] = field(default_factory=dict)
    finished_upstream_attempts: set[str] = field(default_factory=set)


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
        self.adapters = provider_adapters()

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
        normalized_status = str(status)
        if normalized_status not in {
            str(TraceStatus.COMPLETED),
            str(TraceStatus.FAILED),
            str(TraceStatus.INTERRUPTED),
        }:
            raise FinalizationError("sealed trace status must be terminal")
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
        diagnostics: list[str] = list(coverage.completeness_reasons)
        application_events = 0
        started_at = records[0]["occurred_at"] if records else utc_now()
        ended_at = records[-1]["occurred_at"] if records else started_at
        application_records: list[Mapping[str, Any]] = []
        actor_declaration_records: list[Mapping[str, Any]] = []
        alias_declaration_records: list[Mapping[str, Any]] = []
        child_registration_records: list[Mapping[str, Any]] = []
        session_finish_records: list[Mapping[str, Any]] = []
        capture_finish_records: list[Mapping[str, Any]] = []
        terminal_finalization = None

        for record in records:
            record_type = str(record.get("record_type") or "")
            payload = record.get("payload") or {}
            raw_call_id = record.get("call_id")
            # A model-call record without a call id cannot be correlated. Keying the
            # table on None would stringify to "None" and merge unrelated calls, so
            # fail with the record type rather than seal a wrong trace.
            if raw_call_id is None and record_type in _CALL_RECORD_TYPES:
                raise FinalizationError(
                    f"capture record {record_type!r} is missing call_id at ordinal "
                    f"{record.get('ordinal')!r}"
                )
            call_id = str(raw_call_id) if raw_call_id is not None else ""
            raw_attempt_id = record.get("upstream_attempt_id")
            if (
                raw_attempt_id is None
                and record_type
                in {
                    RawRecordType.UPSTREAM_ATTEMPT_STARTED,
                    RawRecordType.UPSTREAM_ATTEMPT_FINISHED,
                }
            ):
                raise FinalizationError(
                    f"capture record {record_type!r} is missing "
                    f"upstream_attempt_id at ordinal {record.get('ordinal')!r}"
                )
            attempt_id = (
                str(raw_attempt_id)
                if raw_attempt_id is not None
                else ""
            )
            if record_type == RawRecordType.MODEL_CALL_STARTED:
                if call_id in calls:
                    raise FinalizationError(
                        f"call {call_id!r} has duplicate model_call.started records"
                    )
                calls[call_id] = _CallState(
                    call_id=call_id,
                    call_index=int(payload.get("call_index") or 0),
                    actor_id=str(
                        record.get("actor_id") or self.binding.workload.root_actor_id
                    ),
                    session_id=str(
                        record.get("session_id")
                        or self.binding.workload.actor_session_id
                    ),
                    started_at=str(payload.get("started_at") or record["occurred_at"]),
                    model=payload.get("model"),
                    streaming=bool(payload.get("stream")),
                    request_body=dict(payload.get("request_body") or {}),
                    request_body_ref=(
                        dict(payload["request_body_ref"])
                        if isinstance(payload.get("request_body_ref"), Mapping)
                        else None
                    ),
                    request_digest=str(payload.get("request_digest") or ""),
                    frames=[],
                    request_headers={
                        str(key): str(value)
                        for key, value in (payload.get("request_headers") or {}).items()
                    },
                    started_ordinal=int(record["ordinal"]),
                    provider_adapter=str(
                        payload.get("provider_adapter") or "openai_chat_completions"
                    ),
                    provider_adapter_version=str(
                        payload.get("provider_adapter_version") or "1"
                    ),
                    route=str(payload.get("route") or ""),
                )
                started = calls[call_id]
                if not started.request_body and started.request_body_ref:
                    started.request_body = self._load_body_ref(started.request_body_ref)
                order.append(str(call_id))
            elif record_type in _CALL_FOLLOWUP_RECORD_TYPES:
                state = calls.get(call_id)
                if state is None:
                    raise FinalizationError(
                        f"capture record {record_type!r} for call {call_id!r} "
                        "precedes model_call.started"
                    )
                if state.finished_ordinal:
                    if record_type == RawRecordType.MODEL_CALL_FINISHED:
                        raise FinalizationError(
                            f"call {call_id!r} has duplicate model_call.finished records"
                        )
                    raise FinalizationError(
                        f"capture record {record_type!r} for call {call_id!r} "
                        "follows model_call.finished"
                    )
            if record_type == RawRecordType.UPSTREAM_ATTEMPT_STARTED:
                state = calls[call_id]
                if attempt_id in state.upstream_attempts:
                    raise FinalizationError(
                        f"call {call_id!r} has duplicate upstream attempt "
                        f"{attempt_id!r}"
                    )
                attempt = int(payload.get("attempt") or 0)
                if attempt <= 0:
                    raise FinalizationError(
                        f"call {call_id!r} has invalid upstream attempt number"
                    )
                state.upstream_attempts[attempt_id] = attempt
            elif record_type == RawRecordType.UPSTREAM_ATTEMPT_FINISHED:
                state = calls[call_id]
                if attempt_id not in state.upstream_attempts:
                    raise FinalizationError(
                        f"call {call_id!r} finishes unknown upstream attempt "
                        f"{attempt_id!r}"
                    )
                if attempt_id in state.finished_upstream_attempts:
                    raise FinalizationError(
                        f"call {call_id!r} has duplicate upstream attempt finish "
                        f"{attempt_id!r}"
                    )
                if int(payload.get("attempt") or 0) != state.upstream_attempts[attempt_id]:
                    raise FinalizationError(
                        f"call {call_id!r} upstream attempt number changed"
                    )
                state.finished_upstream_attempts.add(attempt_id)
            elif record_type == RawRecordType.RESPONSE_FRAME:
                frame = str(payload.get("frame") or "")
                frame_ref = payload.get("frame_ref")
                if not frame and isinstance(frame_ref, Mapping):
                    frame = self._load_frame_ref(dict(frame_ref))
                calls[call_id].frames.append(frame)
            elif record_type == RawRecordType.RESPONSE_BODY:
                state = calls[call_id]
                if state.response_body_ordinal:
                    raise FinalizationError(
                        f"call {call_id!r} has duplicate response.body records"
                    )
                state.response_body_ordinal = int(record["ordinal"]) + 1
                body = payload.get("response_body")
                state.response_body = dict(body) if isinstance(body, Mapping) else None
                state.response_body_ref = (
                    dict(payload["response_body_ref"])
                    if isinstance(payload.get("response_body_ref"), Mapping)
                    else None
                )
                if state.response_body is None and state.response_body_ref:
                    state.response_body = self._load_body_ref(state.response_body_ref)
                state.response_truncated = bool(payload.get("truncated"))
                state.http_status = _int(payload.get("http_status"))
            elif record_type == RawRecordType.MODEL_CALL_FINISHED:
                state = calls[call_id]
                unfinished_attempts = (
                    set(state.upstream_attempts)
                    - state.finished_upstream_attempts
                )
                if unfinished_attempts:
                    raise FinalizationError(
                        f"call {call_id!r} finished before upstream attempts: "
                        + ", ".join(sorted(unfinished_attempts))
                    )
                state.ended_at = str(payload.get("ended_at") or record["occurred_at"])
                state.http_status = _int(payload.get("http_status")) or state.http_status
                usage_payload = payload.get("usage")
                state.usage_payload = (
                    dict(usage_payload) if isinstance(usage_payload, Mapping) else None
                )
                state.usage_observed = bool(payload.get("usage_observed"))
                state.finished_ordinal = int(record["ordinal"])
                state.provider_adapter = str(
                    payload.get("provider_adapter") or state.provider_adapter
                )
            elif record_type == RawRecordType.ARTIFACT:
                artifacts.append(_artifact_from_payload(payload, occurred_at=record["occurred_at"]))
            elif record_type == RawRecordType.APPLICATION_EVENT:
                application_records.append(record)
            elif record_type == RawRecordType.ACTOR_DECLARED:
                actor_declaration_records.append(record)
            elif record_type == RawRecordType.ALIAS_DECLARED:
                alias_declaration_records.append(record)
            elif record_type == RawRecordType.CHILD_REGISTERED:
                child_registration_records.append(record)
            elif record_type == RawRecordType.SESSION_FINISHED:
                session_finish_records.append(record)
                application_records.append(record)
            elif record_type == RawRecordType.CAPTURE_FINISHED:
                capture_finish_records.append(record)

        if len(capture_finish_records) > 1:
            raise FinalizationError("capture has duplicate capture.finished records")
        if capture_finish_records and (
            not records
            or int(capture_finish_records[0]["ordinal"])
            != int(records[-1]["ordinal"])
        ):
            raise FinalizationError("capture.finished must be the final raw record")
        terminal_capture_record = bool(capture_finish_records)
        if capture_finish_records:
            terminal_record = capture_finish_records[0]
            terminal_payload = terminal_record.get("payload")
            if (
                isinstance(terminal_payload, Mapping)
                and terminal_payload.get("schema_version")
            ):
                try:
                    finalization = finalization_from_dict(dict(terminal_payload))
                except (TypeError, ValueError) as exc:
                    raise FinalizationError(
                        f"capture.finished snapshot is invalid: {exc}"
                    ) from exc
                terminal_occurred_at = str(
                    terminal_record.get("occurred_at") or ""
                )
                if finalization.captured_at != terminal_occurred_at:
                    raise FinalizationError(
                        "capture.finished captured_at must equal record occurrence"
                    )
                if str(finalization.status) != normalized_status:
                    raise FinalizationError(
                        "capture.finished status disagrees with finalizer input"
                    )
                if finalization.termination != termination:
                    raise FinalizationError(
                        "capture.finished termination disagrees with finalizer input"
                    )
                if (
                    finalization.coverage_seed.content_digest
                    != coverage.content_digest
                ):
                    raise FinalizationError(
                        "capture.finished coverage disagrees with finalizer input"
                    )
                if (
                    finalization.coverage_seed.capture_id
                    != self.binding.capture_id
                    or finalization.coverage_seed.binding_id
                    != self.binding.binding_id
                    or finalization.coverage_seed.binding_digest
                    != self.binding.content_digest
                ):
                    raise FinalizationError(
                        "capture.finished coverage does not match the binding"
                    )
                if (
                    finalization.finalizer_name != FINALIZER_NAME
                    or finalization.finalizer_version != FINALIZER_VERSION
                ):
                    raise FinalizationError(
                        "capture.finished requires a different finalizer version"
                    )
                if finalization.provenance != self.provenance:
                    raise FinalizationError(
                        "capture.finished provenance disagrees with finalizer input"
                    )
                if finalization.identity != self.identity:
                    raise FinalizationError(
                        "capture.finished identity disagrees with finalizer input"
                    )
                if finalization.root_actor_name != self.root_actor_name:
                    raise FinalizationError(
                        "capture.finished root actor name disagrees with finalizer input"
                    )
                if str(finalization.root_actor_kind) != str(self.root_actor_kind):
                    raise FinalizationError(
                        "capture.finished root actor kind disagrees with finalizer input"
                    )
                terminal_finalization = finalization
                ended_at = terminal_occurred_at

        extra_actors = _resolve_actor_declarations(
            tuple(actor_declaration_records),
            extra_actors=extra_actors,
            root_actor_id=self.binding.workload.root_actor_id,
            root_session_id=self.binding.workload.actor_session_id,
        )
        aliases = _resolve_alias_declarations(
            tuple(alias_declaration_records),
            aliases=aliases,
            root_actor_id=self.binding.workload.root_actor_id,
            root_session_id=self.binding.workload.actor_session_id,
        )
        extra_actors, extra_sessions = _resolve_child_registrations(
            tuple(child_registration_records),
            extra_actors=extra_actors,
            extra_sessions=extra_sessions,
            trace_id=self.binding.trace_id,
            root_actor_id=self.binding.workload.root_actor_id,
            root_session_id=self.binding.workload.actor_session_id,
            root_capture_id=self.binding.capture_id,
        )
        if terminal_finalization is not None:
            expected_actor_ids, expected_session_ids = _declared_identity_order(
                tuple(records)
            )
            if tuple(actor.actor_id for actor in extra_actors) != expected_actor_ids:
                raise FinalizationError(
                    "typed capture finalization actors disagree with raw declarations"
                )
            if (
                tuple(session.session_id for session in extra_sessions)
                != expected_session_ids
            ):
                raise FinalizationError(
                    "typed capture finalization sessions disagree with raw declarations"
                )
            if aliases != _declared_aliases(tuple(alias_declaration_records)):
                raise FinalizationError(
                    "typed capture finalization aliases disagree with raw declarations"
                )
            if aliases != terminal_finalization.aliases:
                raise FinalizationError(
                    "typed capture finalization aliases disagree with its snapshot"
                )
        started_at = min(
            (started_at, *(session.started_at for session in extra_sessions)),
            key=lambda value: _parse_lifecycle_timestamp(
                value,
                field="trace/session started_at",
            ),
        )
        extra_sessions, lifecycle_diagnostics = _resolve_extra_session_lifecycle(
            extra_sessions,
            finish_records=tuple(session_finish_records),
            records=tuple(records),
            parent_ended_at=ended_at,
            root_session_id=self.binding.workload.actor_session_id,
        )
        diagnostics.extend(lifecycle_diagnostics)
        diagnostics.extend(
            self._correlate_provider_calls(
                calls,
                extra_sessions=extra_sessions,
                aliases=aliases,
            )
        )

        # Model calls are built first so an application event may cite the call that
        # produced it. Chronological sequence is the raw capture ordinal throughout,
        # which keeps model calls and application events in one replayable order.
        total_usage = UsageV5(provenance=UsageProvenance.DERIVED, requests=0)
        call_index_to_event: dict[int, str] = {}
        for call_id in order:
            state = calls[call_id]
            built = self._build_call(state)
            spans.append(built.span)
            events.extend(built.events)
            messages.extend(built.messages)
            transformations.append(built.transformation)
            diagnostics.extend(built.diagnostics)
            call_index_to_span[state.call_index] = built.span.span_id
            call_index_to_event[state.call_index] = built.events[-1].event_id
            total_usage = total_usage.merged(built.usage)

        for record in application_records:
            application_events += 1
            event = self._application_event(
                record,
                record.get("payload") or {},
                sequence=int(record["ordinal"]),
                envelope_to_event=envelope_to_event,
                call_index_to_event=call_index_to_event,
                call_index_to_span=call_index_to_span,
            )
            envelope_to_event[str(record["envelope_id"])] = event.event_id
            events.append(event)

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

        model_calls_state, usage_state, raw_state = _call_coverage(
            tuple(calls[item] for item in order)
        )
        application_state = (
            CoverageState.COMPLETE if application_events else CoverageState.NOT_CAPTURED
        )
        event_names = {
            str((item.get("payload") or {}).get("event_type") or "")
            for item in application_records
        }
        agent_state = _coverage_for_prefixes(event_names, ("agent.", "codex.", "react.", "jesterky."))
        environment_state = _coverage_for_prefixes(event_names, ("environment.", "gamebench."))
        tool_state = _coverage_for_prefixes(event_names, ("tool.", "codex.command", "codex.tool"))
        root_call_coverage = _call_coverage(
            tuple(
                calls[item]
                for item in order
                if calls[item].session_id == self.binding.workload.actor_session_id
            )
        )
        root_event_names = {
            str((item.get("payload") or {}).get("event_type") or "")
            for item in application_records
            if str(
                item.get("session_id")
                or self.binding.workload.actor_session_id
            )
            == self.binding.workload.actor_session_id
        }
        root_session = SessionV5(
            session_id=self.binding.workload.actor_session_id,
            actor_id=root_actor.actor_id,
            started_at=started_at,
            ended_at=ended_at,
            capture_id=self.binding.capture_id,
            status=_session_status(status),
            harness=self.provenance.harness,
            provider=self.provenance.provider,
            coverage=SessionCoverageV5(
                model_calls=root_call_coverage[0],
                agent_events=_coverage_for_prefixes(
                    root_event_names,
                    ("agent.", "codex.", "react.", "jesterky."),
                ),
                environment_events=_coverage_for_prefixes(
                    root_event_names,
                    ("environment.", "gamebench."),
                ),
                tool_events=_coverage_for_prefixes(
                    root_event_names,
                    ("tool.", "codex.command", "codex.tool"),
                ),
                usage=root_call_coverage[1],
                raw_provider=root_call_coverage[2],
            ),
        ).sealed()
        attributed_extra_sessions = tuple(
            _with_observed_session_coverage(
                session,
                call_states=tuple(
                    calls[item]
                    for item in order
                    if calls[item].session_id == session.session_id
                ),
                event_names={
                    str((item.get("payload") or {}).get("event_type") or "")
                    for item in application_records
                    if str(item.get("session_id") or "") == session.session_id
                },
            )
            for session in extra_sessions
        )

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
            raw_provider=raw_state,
            agent_events=agent_state,
            environment_events=environment_state,
            tool_events=tool_state,
            usage=usage_state,
            completeness=(
                Completeness.COMPLETE
                if str(status) == TraceStatus.COMPLETED
                and terminal_capture_record
                and model_calls_state != CoverageState.PARTIAL
                and raw_state != CoverageState.PARTIAL
                and not diagnostics
                else Completeness.PARTIAL
            ),
            completeness_reasons=tuple(sorted(set(diagnostics))),
            finalization_status="sealed",
            ended_at=ended_at,
        ).sealed()

        completeness = TraceCompletenessV5(
            capture_status=(
                CaptureStatus.COMPLETE
                if str(status) == TraceStatus.COMPLETED
                and terminal_capture_record
                and model_calls_state != CoverageState.PARTIAL
                and raw_state != CoverageState.PARTIAL
                and not diagnostics
                else CaptureStatus.PARTIAL
            ),
            terminal_event_observed=terminal_capture_record,
            model_calls=model_calls_state,
            raw_provider=raw_state,
            agent_events=agent_state,
            environment_events=environment_state,
            tool_events=tool_state,
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

        all_actors = (root_actor, *extra_actors)
        all_sessions = (root_session, *attributed_extra_sessions)
        actors_by_id = {actor.actor_id: actor for actor in all_actors}
        sessions_by_id = {session.session_id: session for session in all_sessions}
        if len(actors_by_id) != len(all_actors):
            raise FinalizationError("trace actor identities are not unique")
        if len(sessions_by_id) != len(all_sessions):
            raise FinalizationError("trace session identities are not unique")
        for actor in extra_actors:
            if (
                actor.parent_actor_id is not None
                and actor.parent_actor_id not in actors_by_id
            ):
                raise FinalizationError(
                    f"child actor {actor.actor_id} references unknown parent"
                )
        for child in attributed_extra_sessions:
            if child.actor_id not in actors_by_id:
                raise FinalizationError(
                    f"child session {child.session_id} references unknown actor"
                )
            if child.parent_session_id:
                parent = sessions_by_id.get(child.parent_session_id)
                if parent is None:
                    raise FinalizationError(
                        f"child session {child.session_id} references unknown parent"
                    )
                if actors_by_id[child.actor_id].parent_actor_id != parent.actor_id:
                    raise FinalizationError(
                        "child actor parent disagrees with session topology"
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
                    dict.fromkeys(
                        (
                            *self.provenance.transformation_chain,
                            *sorted(
                                {
                                    f"{item.name}@{item.version}"
                                    for item in transformations
                                }
                            ),
                        )
                    )
                ),
            ),
            completeness=completeness,
            actors=all_actors,
            sessions=all_sessions,
            messages=tuple(messages),
            spans=tuple(spans),
            events=tuple(sorted(events, key=lambda item: item.order.chronological_sequence or 0)),
            artifacts=tuple(artifacts),
            usage=total_usage,
            aliases=aliases,
        )
        _validate_alias_integrity(document)
        document = document.sealed()

        return SealedCapture(
            document=document,
            coverage=sealed_coverage,
            segments=self.segments,
            envelope_to_event=envelope_to_event,
            call_index_to_span=call_index_to_span,
        )

    def _correlate_provider_calls(
        self,
        calls: Mapping[str, _CallState],
        *,
        extra_sessions: tuple[SessionV5, ...],
        aliases: tuple[AliasV1, ...],
    ) -> tuple[str, ...]:
        """Join late native Codex identities to earlier provider captures.

        Codex does not forward arbitrary Synth headers to its provider request, but
        its Responses request carries the Codex thread id as ``prompt_cache_key``.
        The native JSONL stream independently observes that thread id. An exact,
        unique ``codex.thread`` alias is therefore sufficient to attribute the call
        without guessing. Explicit non-root capture context always wins.
        """

        root_session_id = self.binding.workload.actor_session_id
        sessions = {
            session.session_id: session
            for session in extra_sessions
        }
        thread_targets: dict[str, set[str]] = {}
        for alias in aliases:
            if (
                str(alias.namespace) != str(AliasNamespace.CODEX_THREAD)
                or alias.target_kind != "session"
                or not alias.value
            ):
                continue
            thread_targets.setdefault(alias.value, set()).add(alias.target_id)

        diagnostics: list[str] = []
        for state in calls.values():
            cache_key = state.request_body.get("prompt_cache_key")
            if not isinstance(cache_key, str):
                websocket_response = state.request_body.get("response")
                cache_key = (
                    websocket_response.get("prompt_cache_key")
                    if isinstance(websocket_response, Mapping)
                    else None
                )
            if not isinstance(cache_key, str) or not cache_key:
                continue
            targets = thread_targets.get(cache_key)
            if not targets:
                continue
            if len(targets) != 1:
                diagnostics.append(
                    "provider call "
                    f"{state.call_id} has ambiguous codex.thread alias for "
                    "prompt_cache_key"
                )
                continue
            target_id = next(iter(targets))
            if target_id == root_session_id:
                state.native_correlation = {
                    "basis": "exact_native_alias",
                    "request_field": (
                        "response.prompt_cache_key"
                        if isinstance(state.request_body.get("response"), Mapping)
                        else "prompt_cache_key"
                    ),
                    "alias_namespace": str(AliasNamespace.CODEX_THREAD),
                    "alias_value": cache_key,
                    "alias_target_id": target_id,
                }
                continue
            target = sessions.get(target_id)
            if target is None:
                diagnostics.append(
                    "provider call "
                    f"{state.call_id} codex.thread alias targets unknown session "
                    f"{target_id}"
                )
                continue
            if state.session_id != root_session_id:
                if state.session_id != target_id:
                    diagnostics.append(
                        "provider call "
                        f"{state.call_id} explicit session conflicts with "
                        "codex.thread alias"
                    )
                continue
            state.session_id = target.session_id
            state.actor_id = target.actor_id
            state.native_correlation = {
                "basis": "exact_native_alias",
                "request_field": (
                    "response.prompt_cache_key"
                    if isinstance(state.request_body.get("response"), Mapping)
                    else "prompt_cache_key"
                ),
                "alias_namespace": str(AliasNamespace.CODEX_THREAD),
                "alias_value": cache_key,
                "alias_target_id": target.session_id,
            }
        return tuple(diagnostics)

    def _load_body_ref(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        """Rehydrate a captured provider body, which is always a JSON object."""

        inline = ref.get("inline")
        if isinstance(inline, Mapping):
            return dict(inline)
        payload = self._read_stored_body(ref)
        if payload is None:
            return {}
        loaded = json.loads(payload.decode("utf-8"))
        if not isinstance(loaded, Mapping):
            raise FinalizationError(
                f"captured provider body {ref.get('uri')!r} is not a JSON object"
            )
        return dict(loaded)

    def _load_frame_ref(self, ref: Mapping[str, Any]) -> str:
        """Rehydrate a captured stream frame, which is wire text rather than an object."""

        payload = self._read_stored_body(ref)
        if payload is None:
            return ""
        return payload.decode("utf-8", errors="replace")

    def _read_stored_body(self, ref: Mapping[str, Any]) -> bytes | None:
        """The stored bytes behind a body ref, or None when it names no blob."""

        uri = ref.get("uri")
        digest = str(ref.get("stored_digest") or "")
        if not uri or not digest:
            return None
        store = FilesystemBlobStore(self.spool_root.parents[1] / "blobs")
        expected_uri = store.uri(digest)
        if str(uri) != expected_uri:
            raise FinalizationError(
                f"captured body URI does not match its digest: {uri!r} != "
                f"{expected_uri!r}"
            )
        try:
            return store.get(digest)
        except (FileNotFoundError, ValueError) as exc:
            raise FinalizationError(f"captured body cannot be verified: {exc}") from exc

    # -- per-call normalization --------------------------------------------------

    def _build_call(self, state: _CallState) -> "_BuiltCall":
        trace_id = self.binding.trace_id
        actor_id = state.actor_id
        session_id = state.session_id
        span_id = record_id("span", kind="model_call", scope=(trace_id,), key=state.call_id)
        diagnostics: list[str] = []
        messages: list[MessageNodeV5] = []
        adapter = self.adapters.by_name(state.provider_adapter)
        if adapter is None:
            raise FinalizationError(
                f"capture names unsupported provider adapter {state.provider_adapter!r}"
            )
        if state.provider_adapter_version != adapter.version:
            raise FinalizationError(
                "capture provider adapter version "
                f"{state.provider_adapter_version!r} does not match installed "
                f"{adapter.name}@{adapter.version}"
            )

        request_messages = adapter.normalize_request(state.request_body)
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

        provider_result = NormalizedProviderResult()
        if state.streaming:
            stream = adapter.new_stream()
            for frame in state.frames:
                stream.feed(frame.encode("utf-8"))
            provider_result = stream.finish()
        elif state.response_body is not None:
            provider_result = adapter.normalize_unary(state.response_body)
        elif state.response_truncated:
            diagnostics.append("response body was not retained and could not be normalized")
        diagnostics.extend(provider_result.diagnostics)

        output_ids: list[str] = []
        response_previous = previous
        for response_index, response in enumerate(provider_result.messages):
            node = _message_node(
                trace_id=trace_id,
                key={
                    "call": state.call_id,
                    "slot": "response",
                    "index": response_index,
                },
                normalized=response,
                actor_id=actor_id,
                session_id=session_id,
                span_id=span_id,
                occurred_at=state.ended_at or state.started_at,
                predecessors=response_previous,
            )
            messages.append(node)
            output_ids.append(node.message_id)
            response_previous = (node.message_id,)

        usage = (
            adapter.usage(state.usage_payload)
            if state.usage_payload is not None
            else provider_result.usage or adapter.usage(None)
        )
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
                "finish_reason": (
                    provider_result.messages[-1].finish_reason
                    if provider_result.messages
                    else None
                ),
                "provider_adapter": state.provider_adapter,
                "provider_adapter_version": state.provider_adapter_version,
                "route": state.route,
                "provider_ids": provider_result.provider_ids,
                "provider_terminal_observed": provider_result.terminal_observed,
                "correlation_headers": _correlation_headers(state.request_headers),
                "native_correlation": state.native_correlation,
            },
            input_message_ids=tuple(input_ids),
            output_message_ids=tuple(output_ids),
            usage=usage,
            token_capture=provider_result.token_capture,
            aliases=_call_aliases(
                state.request_headers,
                span_id=span_id,
            ),
            transformations=(
                TransformationRecordV1(
                    name=adapter.name,
                    version=adapter.version,
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
            order=EventOrderV1(
                chronological_sequence=state.started_ordinal, source_order_id=state.call_id
            ),
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
            order=EventOrderV1(
                chronological_sequence=state.finished_ordinal or state.started_ordinal,
                source_order_id=state.call_id,
            ),
            caused_by_event_ids=(started_event.event_id,),
            status=EventStatus.OK if status == SpanStatus.OK else EventStatus.ERROR,
            payload={
                "call_index": state.call_index,
                "http_status": state.http_status,
                "usage_observed": (
                    state.usage_payload is not None
                    or (
                        provider_result.usage is not None
                        and str(provider_result.usage.provenance)
                        != UsageProvenance.UNAVAILABLE
                    )
                ),
                "provider_terminal_observed": provider_result.terminal_observed,
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
        call_index_to_event: Mapping[int, str],
        call_index_to_span: Mapping[int, str],
    ) -> EventV5:
        trace_id = self.binding.trace_id
        envelope_id = str(record["envelope_id"])
        structural = payload.get("structural")
        body = dict(payload.get("body") or {})
        caused_by = [
            envelope_to_event[item]
            for item in list(payload.get("caused_by") or [])
            if item in envelope_to_event
        ]
        # A producer may cite the model call that generated this event by its capture
        # ordinal. The linkage is recorded as declared, not observed, because the proxy
        # cannot see which local action a response caused.
        span_id = None
        declared_index = _int(body.get("model_call_index"))
        if declared_index is not None and declared_index in call_index_to_event:
            caused_by.append(call_index_to_event[declared_index])
            span_id = call_index_to_span.get(declared_index)
            body["model_call_link_basis"] = "declared_by_producer"
        return EventV5(
            event_id=application_event_id(trace_id=trace_id, envelope_id=envelope_id),
            event_type=str(payload.get("event_type") or EventType.APPLICATION),
            actor_id=str(record.get("actor_id") or self.binding.workload.root_actor_id),
            session_id=str(record.get("session_id") or self.binding.workload.actor_session_id),
            occurred_at=str(record["occurred_at"]),
            span_id=span_id,
            order=EventOrderV1(
                chronological_sequence=sequence,
                source_order_id=envelope_id,
                structural=_structural(structural),
            ),
            caused_by_event_ids=tuple(caused_by),
            payload=body,
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


def application_event_id(*, trace_id: str, envelope_id: str) -> str:
    """The event id sealing will assign to an application event.

    Deterministic and computable before sealing, so a producer can attach aliases and
    cross-references to an event it has only just appended.
    """

    return record_id("evt", kind="application", scope=(trace_id,), key=envelope_id)


def _correlation_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Correlation headers the caller declared, preserved as span-level topology."""

    return {
        key: value
        for key, value in headers.items()
        if key in CORRELATION_HEADERS or key.startswith(CORRELATION_HEADER_PREFIXES)
    }


def _call_aliases(headers: Mapping[str, str], *, span_id: str) -> tuple[AliasV1, ...]:
    value = headers.get("x-synth-call-correlation-id")
    if not value:
        return ()
    return (
        AliasV1(
            namespace=AliasNamespace.CORRELATION,
            value=str(value),
            target_id=span_id,
            target_kind="span",
            provenance="declared_by_workload",
        ),
    )


def _coverage_for_prefixes(
    event_names: set[str],
    prefixes: tuple[str, ...],
) -> CoverageState:
    return (
        CoverageState.COMPLETE
        if any(name.startswith(prefixes) for name in event_names)
        else CoverageState.NOT_CAPTURED
    )


def _call_coverage(
    calls: tuple[_CallState, ...],
) -> tuple[CoverageState, CoverageState, CoverageState]:
    if not calls:
        return (
            CoverageState.NOT_CAPTURED,
            CoverageState.NOT_CAPTURED,
            CoverageState.NOT_CAPTURED,
        )
    model_calls = (
        CoverageState.COMPLETE
        if all(call.finished_ordinal > 0 for call in calls)
        else CoverageState.PARTIAL
    )
    usage = (
        CoverageState.COMPLETE
        if all(call.usage_observed for call in calls)
        else CoverageState.PARTIAL
    )
    raw_provider = (
        CoverageState.COMPLETE
        if all(
            call.request_body
            and (call.response_body is not None or call.frames)
            for call in calls
        )
        else CoverageState.PARTIAL
    )
    return model_calls, usage, raw_provider


def _declared_identity_order(
    records: tuple[Mapping[str, Any], ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    actor_ids: list[str] = []
    session_ids: list[str] = []
    for record in records:
        record_type = str(record.get("record_type") or "")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if record_type == str(RawRecordType.ACTOR_DECLARED):
            actor_payload = payload.get("actor")
            if not isinstance(actor_payload, Mapping):
                raise FinalizationError("actor.declared payload is invalid")
            actor = _registered_actor(actor_payload)
            if actor.actor_id not in actor_ids:
                actor_ids.append(actor.actor_id)
        elif record_type == str(RawRecordType.CHILD_REGISTERED):
            actor_payload = payload.get("actor")
            session_payload = payload.get("session")
            if not isinstance(actor_payload, Mapping) or not isinstance(
                session_payload,
                Mapping,
            ):
                raise FinalizationError("child.registered identity is invalid")
            actor = _registered_actor(actor_payload)
            session = _registered_session(session_payload)
            if actor.actor_id not in actor_ids:
                actor_ids.append(actor.actor_id)
            session_ids.append(session.session_id)
    return tuple(actor_ids), tuple(session_ids)


def _declared_aliases(
    records: tuple[Mapping[str, Any], ...],
) -> tuple[AliasV1, ...]:
    aliases: list[AliasV1] = []
    for record in records:
        payload = record.get("payload")
        alias_payload = (
            payload.get("alias") if isinstance(payload, Mapping) else None
        )
        if not isinstance(alias_payload, Mapping):
            raise FinalizationError("alias.declared payload is invalid")
        try:
            aliases.append(AliasV1(**dict(alias_payload)))
        except (TypeError, ValueError) as exc:
            raise FinalizationError(
                f"alias.declared payload is invalid: {exc}"
            ) from exc
    return tuple(aliases)


def _resolve_actor_declarations(
    records: tuple[Mapping[str, Any], ...],
    *,
    extra_actors: tuple[ActorV5, ...],
    root_actor_id: str,
    root_session_id: str,
) -> tuple[ActorV5, ...]:
    actors = {actor.actor_id: actor for actor in extra_actors}
    if len(actors) != len(extra_actors):
        raise FinalizationError("extra actors contain duplicate actor ids")
    declared: set[str] = set()
    for record in records:
        if (
            str(record.get("actor_id") or "") != root_actor_id
            or str(record.get("session_id") or "") != root_session_id
        ):
            raise FinalizationError(
                "actor.declared must belong to the root session"
            )
        payload = record.get("payload")
        actor_payload = (
            payload.get("actor") if isinstance(payload, Mapping) else None
        )
        if not isinstance(actor_payload, Mapping):
            raise FinalizationError("actor.declared payload is invalid")
        actor = _registered_actor(actor_payload)
        if actor.actor_id == root_actor_id:
            raise FinalizationError("actor.declared collides with the root actor")
        if actor.parent_actor_id == actor.actor_id:
            raise FinalizationError("actor.declared contains a parent cycle")
        if actor.actor_id in declared:
            raise FinalizationError(
                f"actor {actor.actor_id} has duplicate declarations"
            )
        prior = actors.get(actor.actor_id)
        if prior is not None and prior.content_digest != actor.content_digest:
            raise FinalizationError(
                f"actor {actor.actor_id} declaration conflicts with supplied facts"
            )
        actors[actor.actor_id] = actor
        declared.add(actor.actor_id)
    return tuple(actors.values())


def _resolve_alias_declarations(
    records: tuple[Mapping[str, Any], ...],
    *,
    aliases: tuple[AliasV1, ...],
    root_actor_id: str,
    root_session_id: str,
) -> tuple[AliasV1, ...]:
    by_key = {
        (str(alias.namespace), alias.value, alias.target_kind): alias
        for alias in aliases
    }
    if len(by_key) != len(aliases):
        raise FinalizationError("aliases contain duplicate identities")
    declared: set[tuple[str, str, str]] = set()
    for record in records:
        if (
            str(record.get("actor_id") or "") != root_actor_id
            or str(record.get("session_id") or "") != root_session_id
        ):
            raise FinalizationError(
                "alias.declared must belong to the root session"
            )
        payload = record.get("payload")
        alias_payload = (
            payload.get("alias") if isinstance(payload, Mapping) else None
        )
        if not isinstance(alias_payload, Mapping):
            raise FinalizationError("alias.declared payload is invalid")
        try:
            alias = AliasV1(**dict(alias_payload))
        except (TypeError, ValueError) as exc:
            raise FinalizationError(
                f"alias.declared payload is invalid: {exc}"
            ) from exc
        key = (str(alias.namespace), alias.value, alias.target_kind)
        if key in declared:
            raise FinalizationError(
                f"alias {alias.namespace}:{alias.value} has duplicate declarations"
            )
        prior = by_key.get(key)
        if prior is not None and prior != alias:
            raise FinalizationError(
                "alias.declared conflicts with supplied alias facts"
            )
        by_key[key] = alias
        declared.add(key)
    return tuple(by_key.values())


def _resolve_child_registrations(
    records: tuple[Mapping[str, Any], ...],
    *,
    extra_actors: tuple[ActorV5, ...],
    extra_sessions: tuple[SessionV5, ...],
    trace_id: str,
    root_actor_id: str,
    root_session_id: str,
    root_capture_id: str,
) -> tuple[tuple[ActorV5, ...], tuple[SessionV5, ...]]:
    actors = {actor.actor_id: actor for actor in extra_actors}
    sessions = {session.session_id: session for session in extra_sessions}
    if len(actors) != len(extra_actors) or len(sessions) != len(extra_sessions):
        raise FinalizationError("extra child identities contain duplicate ids")
    registered_sessions: set[str] = set()
    registered_captures: set[str] = set()
    session_actors = {root_session_id: root_actor_id}
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise FinalizationError("child.registered payload is invalid")
        actor_payload = payload.get("actor")
        session_payload = payload.get("session")
        if not isinstance(actor_payload, Mapping) or not isinstance(
            session_payload,
            Mapping,
        ):
            raise FinalizationError("child.registered identity is invalid")
        actor = _registered_actor(actor_payload)
        session = _registered_session(session_payload)
        if session.session_id in registered_sessions:
            raise FinalizationError(
                f"child session {session.session_id} has duplicate registrations"
            )
        if actor.actor_id == root_actor_id or session.session_id == root_session_id:
            raise FinalizationError("child registration collides with root identity")
        if (
            actor.parent_actor_id == actor.actor_id
            or session.parent_session_id == session.session_id
        ):
            raise FinalizationError("child registration contains a parent cycle")
        if actor.actor_id != str(record.get("actor_id") or ""):
            raise FinalizationError("child.registered envelope actor mismatch")
        if (
            session.session_id != str(record.get("session_id") or "")
            or session.actor_id != actor.actor_id
        ):
            raise FinalizationError("child.registered envelope session mismatch")
        context_payload = payload.get("context")
        if context_payload is not None:
            if not isinstance(context_payload, Mapping):
                raise FinalizationError("child.registered context is invalid")
            try:
                context = TraceContextV1(**dict(context_payload))
            except (TypeError, ValueError) as exc:
                raise FinalizationError(
                    f"child.registered context is invalid: {exc}"
                ) from exc
            if (
                context.trace_id != trace_id
                or context.capture_id == root_capture_id
                or context.capture_id in registered_captures
                or context.parent_actor_session_id is None
                or context.actor_id != actor.actor_id
                or context.actor_session_id != session.session_id
                or session.capture_id != context.capture_id
                or actor.parent_actor_id != context.parent_actor_id
                or session.parent_session_id
                != context.parent_actor_session_id
            ):
                raise FinalizationError(
                    "child.registered context does not match actor/session topology"
                )
            registered_captures.add(context.capture_id)
        parent_session_id = session.parent_session_id
        if (
            parent_session_id is not None
            and parent_session_id not in session_actors
        ):
            raise FinalizationError(
                "child.registered parent session must be registered first"
            )
        if parent_session_id is not None:
            parent_actor_id = session_actors[parent_session_id]
            if actor.parent_actor_id != parent_actor_id:
                raise FinalizationError(
                    "child.registered actor parent disagrees with session parent"
                )
        prior_actor = actors.get(actor.actor_id)
        if (
            prior_actor is not None
            and prior_actor.content_digest != actor.content_digest
        ):
            raise FinalizationError(
                f"child actor {actor.actor_id} has conflicting registrations"
            )
        prior_session = sessions.get(session.session_id)
        if (
            prior_session is not None
            and prior_session.content_digest != session.content_digest
        ):
            raise FinalizationError(
                f"child session {session.session_id} has conflicting registrations"
            )
        actors[actor.actor_id] = actor
        sessions[session.session_id] = session
        registered_sessions.add(session.session_id)
        session_actors[session.session_id] = session.actor_id
    return tuple(actors.values()), tuple(sessions.values())


def _registered_actor(payload: Mapping[str, Any]) -> ActorV5:
    values = dict(payload)
    values["external_trace_refs"] = tuple(
        values.get("external_trace_refs") or ()
    )
    values["aliases"] = tuple(
        AliasV1(**item) if isinstance(item, Mapping) else item
        for item in values.get("aliases") or ()
    )
    try:
        return replace(
            ActorV5(**values),
            content_digest="",
        ).sealed()
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"child.registered actor is invalid: {exc}") from exc


def _registered_session(payload: Mapping[str, Any]) -> SessionV5:
    values = dict(payload)
    coverage = values.get("coverage")
    if isinstance(coverage, Mapping):
        coverage_values = dict(coverage)
        coverage_values["reasons"] = tuple(coverage_values.get("reasons") or ())
        values["coverage"] = SessionCoverageV5(**coverage_values)
    values["aliases"] = tuple(
        AliasV1(**item) if isinstance(item, Mapping) else item
        for item in values.get("aliases") or ()
    )
    try:
        return replace(
            SessionV5(**values),
            content_digest="",
        ).sealed()
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"child.registered session is invalid: {exc}") from exc


def _resolve_extra_session_lifecycle(
    sessions: tuple[SessionV5, ...],
    *,
    finish_records: tuple[Mapping[str, Any], ...],
    records: tuple[Mapping[str, Any], ...],
    parent_ended_at: str,
    root_session_id: str,
) -> tuple[tuple[SessionV5, ...], tuple[str, ...]]:
    """Apply durable child terminal facts and interrupt unclosed children."""

    by_id = {session.session_id: session for session in sessions}
    if len(by_id) != len(sessions):
        raise FinalizationError("extra sessions contain duplicate session ids")
    for session in sessions:
        if (
            session.parent_session_id
            and session.parent_session_id != root_session_id
            and session.parent_session_id not in by_id
        ):
            raise FinalizationError(
                f"child session {session.session_id} references unknown parent "
                f"{session.parent_session_id}"
            )
        seen: set[str] = set()
        current: str | None = session.session_id
        while current is not None and current != root_session_id:
            if current in seen:
                raise FinalizationError("child session topology contains a cycle")
            seen.add(current)
            current_session = by_id.get(current)
            if current_session is None:
                break
            current = current_session.parent_session_id
    terminal_statuses = {
        str(SessionStatus.COMPLETED),
        str(SessionStatus.FAILED),
        str(SessionStatus.INTERRUPTED),
    }
    facts: dict[str, tuple[str, str, int]] = {}
    for record in finish_records:
        session_id = str(record.get("session_id") or "")
        if not session_id or session_id == root_session_id:
            raise FinalizationError(
                "session.finished must identify a non-root child session"
            )
        session = by_id.get(session_id)
        if session is None:
            raise FinalizationError(
                f"session.finished references unknown child session {session_id}"
            )
        if str(record.get("actor_id") or "") != session.actor_id:
            raise FinalizationError(
                f"session.finished actor does not own child session {session_id}"
            )
        payload = record.get("payload")
        if not isinstance(payload, Mapping) or str(payload.get("event_type")) != str(
            EventType.SESSION_FINISHED
        ):
            raise FinalizationError("invalid session.finished raw payload")
        body = payload.get("body")
        if not isinstance(body, Mapping):
            raise FinalizationError("session.finished body is required")
        terminal_status = str(body.get("status") or "")
        if terminal_status not in terminal_statuses:
            raise FinalizationError(
                f"session.finished has non-terminal status {terminal_status!r}"
            )
        terminal_at = str(body.get("ended_at") or "")
        if not terminal_at or terminal_at != str(record.get("occurred_at") or ""):
            raise FinalizationError(
                "session.finished ended_at must equal its raw occurrence timestamp"
            )
        _parse_lifecycle_timestamp(terminal_at, field="session.finished ended_at")
        if session_id in facts:
            raise FinalizationError(
                f"child session {session_id} has duplicate terminal facts"
            )
        facts[session_id] = (
            terminal_status,
            terminal_at,
            int(record["ordinal"]),
        )

    for record in records:
        session_id = str(record.get("session_id") or "")
        fact = facts.get(session_id)
        if fact is not None and int(record["ordinal"]) > fact[2]:
            raise FinalizationError(
                f"child session {session_id} emitted a record after session.finished"
            )

    diagnostics: list[str] = []
    resolved: list[SessionV5] = []
    parent_end = _parse_lifecycle_timestamp(
        parent_ended_at,
        field="parent ended_at",
    )
    for session in sessions:
        started = _parse_lifecycle_timestamp(
            session.started_at,
            field=f"session {session.session_id} started_at",
        )
        current_status = str(session.status)
        fact = facts.get(session.session_id)
        if current_status == str(SessionStatus.RUNNING):
            if session.ended_at is not None:
                raise FinalizationError(
                    f"running child session {session.session_id} has ended_at"
                )
            if fact is not None:
                status, terminal_at, _ = fact
                ended = _parse_lifecycle_timestamp(
                    terminal_at,
                    field=f"session {session.session_id} ended_at",
                )
                if ended < started:
                    raise FinalizationError(
                        f"child session {session.session_id} ended before it started"
                    )
                if ended > parent_end:
                    raise FinalizationError(
                        f"child session {session.session_id} ended after its parent trace"
                    )
                resolved.append(
                    replace(
                        session,
                        status=status,
                        ended_at=terminal_at,
                        content_digest="",
                    ).sealed()
                )
                continue
            if parent_end < started:
                raise FinalizationError(
                    f"parent ended before child session {session.session_id} started"
                )
            diagnostics.append(
                f"child_session_not_finished:{session.session_id}"
            )
            resolved.append(
                replace(
                    session,
                    status=SessionStatus.INTERRUPTED,
                    ended_at=parent_ended_at,
                    coverage=replace(
                        session.coverage,
                        reasons=tuple(
                            dict.fromkeys(
                                (
                                    *session.coverage.reasons,
                                    "child_session_not_finished",
                                )
                            )
                        ),
                    ),
                    content_digest="",
                ).sealed()
            )
            continue
        if current_status not in terminal_statuses:
            raise FinalizationError(
                f"child session {session.session_id} has unknown status "
                f"{current_status!r}"
            )
        terminal_at = session.ended_at
        if fact is not None:
            fact_status, fact_ended_at, _ = fact
            if fact_status != current_status or (
                terminal_at is not None and terminal_at != fact_ended_at
            ):
                raise FinalizationError(
                    f"child session {session.session_id} terminal declaration "
                    "conflicts with its raw fact"
                )
            terminal_at = fact_ended_at
        if terminal_at is None:
            raise FinalizationError(
                f"terminal child session {session.session_id} has no ended_at"
            )
        ended = _parse_lifecycle_timestamp(
            terminal_at,
            field=f"session {session.session_id} ended_at",
        )
        if ended < started:
            raise FinalizationError(
                f"child session {session.session_id} ended before it started"
            )
        if ended > parent_end:
            raise FinalizationError(
                f"child session {session.session_id} ended after its parent trace"
            )
        resolved.append(
            replace(
                session,
                ended_at=terminal_at,
                content_digest="",
            ).sealed()
        )
    resolved_by_id = {session.session_id: session for session in resolved}
    for child in resolved:
        if not child.parent_session_id:
            continue
        parent = resolved_by_id.get(child.parent_session_id)
        if parent is None:
            continue
        parent_started = _parse_lifecycle_timestamp(
            parent.started_at,
            field=f"parent session {parent.session_id} started_at",
        )
        child_started = _parse_lifecycle_timestamp(
            child.started_at,
            field=f"child session {child.session_id} started_at",
        )
        if child_started < parent_started:
            raise FinalizationError(
                f"child session {child.session_id} started before parent "
                f"{parent.session_id}"
            )
        if child.ended_at is not None and parent.ended_at is not None:
            child_ended = _parse_lifecycle_timestamp(
                child.ended_at,
                field=f"child session {child.session_id} ended_at",
            )
            parent_ended = _parse_lifecycle_timestamp(
                parent.ended_at,
                field=f"parent session {parent.session_id} ended_at",
            )
            if child_ended > parent_ended:
                raise FinalizationError(
                    f"child session {child.session_id} ended after parent "
                    f"{parent.session_id}"
                )
    return tuple(resolved), tuple(diagnostics)


def _parse_lifecycle_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise FinalizationError(f"{field} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise FinalizationError(f"{field} must include a timezone")
    return parsed


def _with_observed_session_coverage(
    session: SessionV5,
    *,
    call_states: tuple[_CallState, ...],
    event_names: set[str],
) -> SessionV5:
    """Overlay facts captured for one session without erasing declared coverage."""

    model_calls, usage, raw_provider = _call_coverage(call_states)
    declared = session.coverage

    def observed_or_declared(
        observed: CoverageState,
        prior: CoverageState | str,
    ) -> CoverageState | str:
        return prior if observed == CoverageState.NOT_CAPTURED else observed

    agent_events = _coverage_for_prefixes(
        event_names,
        ("agent.", "codex.", "react.", "jesterky."),
    )
    environment_events = _coverage_for_prefixes(
        event_names,
        ("environment.", "gamebench."),
    )
    tool_events = _coverage_for_prefixes(
        event_names,
        ("tool.", "codex.command", "codex.tool"),
    )
    return replace(
        session,
        coverage=SessionCoverageV5(
            model_calls=observed_or_declared(model_calls, declared.model_calls),
            agent_events=observed_or_declared(
                agent_events,
                declared.agent_events,
            ),
            environment_events=observed_or_declared(
                environment_events,
                declared.environment_events,
            ),
            tool_events=observed_or_declared(tool_events, declared.tool_events),
            usage=observed_or_declared(usage, declared.usage),
            raw_provider=observed_or_declared(
                raw_provider,
                declared.raw_provider,
            ),
            reasons=declared.reasons,
        ),
        content_digest="",
    ).sealed()


def _validate_alias_integrity(document: TraceDocumentV5) -> None:
    """Reject aliases that cannot resolve to a declared canonical entity."""

    targets = {
        "trace": {document.trace_id},
        "actor": {item.actor_id for item in document.actors},
        "session": {item.session_id for item in document.sessions},
        "span": {item.span_id for item in document.spans},
        "event": {item.event_id for item in document.events},
        "message": {item.message_id for item in document.messages},
        "part": {
            part.part_id
            for message in document.messages
            for part in message.parts
        },
        "artifact": {item.artifact_id for item in document.artifacts},
        "branch": {item.branch_id for item in document.branches},
        "error": {item.error_id for item in document.errors},
    }
    aliases: list[AliasV1] = [
        *document.aliases,
        *document.provenance.aliases,
    ]
    for collection in (
        document.actors,
        document.sessions,
        document.spans,
        document.events,
        document.messages,
    ):
        for item in collection:
            aliases.extend(item.aliases)

    supported = _LOCAL_ALIAS_TARGET_KINDS | _EXTERNAL_ALIAS_TARGET_KINDS
    for alias_item in aliases:
        target_kind = str(alias_item.target_kind)
        if target_kind not in supported:
            raise FinalizationError(
                "alias has unsupported target_kind "
                f"{alias_item.target_kind!r}: {alias_item.namespace}:{alias_item.value}"
            )
        if not alias_item.target_id:
            raise FinalizationError(
                "alias has dangling target_id: "
                f"{alias_item.namespace}:{alias_item.value}"
            )
        if (
            target_kind in _LOCAL_ALIAS_TARGET_KINDS
            and alias_item.target_id not in targets[target_kind]
        ):
            raise FinalizationError(
                f"alias target {target_kind}:{alias_item.target_id} "
                "is not present in the trace"
            )


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _session_status(status: TraceStatus | str) -> SessionStatus:
    if str(status) == str(TraceStatus.COMPLETED):
        return SessionStatus.COMPLETED
    if str(status) == str(TraceStatus.FAILED):
        return SessionStatus.FAILED
    return SessionStatus.INTERRUPTED


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
    "application_event_id",
]
