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
from ..models.tokens import TokenCaptureV5
from ..store.filesystem import FilesystemBlobStore
from .binding import TraceCaptureBindingV1
from .coverage import CaptureCoverageReceiptV1, CaptureScope, Completeness
from .envelope import RawRecordType
from .redaction import CORRELATION_HEADER_PREFIXES, CORRELATION_HEADERS
from .spool import TraceSegmentManifestV1, read_segments


FINALIZER_NAME = "synth-trace-finalizer"
FINALIZER_VERSION = "1"

# Record types keyed by call id. Without one they cannot be correlated to a call.
_CALL_RECORD_TYPES = frozenset(
    {
        RawRecordType.MODEL_CALL_STARTED,
        RawRecordType.RESPONSE_FRAME,
        RawRecordType.RESPONSE_BODY,
        RawRecordType.MODEL_CALL_FINISHED,
    }
)


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
        terminal_capture_record = False

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
            call_id = str(raw_call_id)
            if record_type == RawRecordType.MODEL_CALL_STARTED:
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
            elif record_type == RawRecordType.RESPONSE_FRAME and call_id in calls:
                frame = str(payload.get("frame") or "")
                frame_ref = payload.get("frame_ref")
                if not frame and isinstance(frame_ref, Mapping):
                    frame = self._load_frame_ref(dict(frame_ref))
                calls[call_id].frames.append(frame)
            elif record_type == RawRecordType.RESPONSE_BODY and call_id in calls:
                state = calls[call_id]
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
            elif record_type == RawRecordType.MODEL_CALL_FINISHED and call_id in calls:
                state = calls[call_id]
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
            elif record_type == RawRecordType.CAPTURE_FINISHED:
                terminal_capture_record = True

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
            ended_at=utc_now(),
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
            sessions=(root_session, *attributed_extra_sessions),
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
