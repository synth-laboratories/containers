from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from synth_containers.tracing.adapters.atif import import_atif
from synth_containers.tracing.capture.binding import (
    BindingCaptureV1,
    BindingWorkloadV1,
    CaptureMode,
    WorkloadKind,
    mint_binding,
)
from synth_containers.tracing.capture.coverage import new_coverage_receipt
from synth_containers.tracing.capture.envelope import RawRecordType
from synth_containers.tracing.capture.finalizer import (
    FinalizationError,
    TraceFinalizer,
)
from synth_containers.tracing.capture.session import CaptureSession
from synth_containers.tracing.capture.spool import RawSpool
from synth_containers.tracing.models.artifacts import ArtifactRefV5
from synth_containers.tracing.models.events import (
    EventOrderV1,
    EventType,
    EventV5,
    TraceErrorV1,
)
from synth_containers.tracing.models.identity import (
    AliasV1,
    TraceIdentityV5,
    TraceProvenanceV5,
)
from synth_containers.tracing.models.messages import BranchV5
from synth_containers.tracing.models.spans import SpanKind, SpanV5
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.validation.validator import validate_trace


def _base_trace():
    return import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "alias-integrity",
            "agent": {"name": "agent", "version": "1"},
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-07-25T00:00:00Z",
                    "source": "user",
                    "message": "hello",
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "target_kind",
    (
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
    ),
)
def test_validator_reports_dangling_alias_for_each_canonical_kind(
    target_kind: str,
) -> None:
    trace = _base_trace()
    forged = replace(
        trace,
        aliases=(
            AliasV1(
                namespace="test.dangling",
                value=target_kind,
                target_id=f"missing-{target_kind}",
                target_kind=target_kind,
            ),
        ),
        content_digest="",
    ).sealed()

    findings = validate_trace(forged)

    assert any(
        item.code == "dangling_alias_target"
        and item.detail["target_kind"] == target_kind
        for item in findings
    )


def test_validator_reports_unsupported_alias_target_kind() -> None:
    trace = _base_trace()
    forged = replace(
        trace,
        aliases=(
            AliasV1(
                namespace="test.unsupported",
                value="unsupported",
                target_id=trace.trace_id,
                target_kind="database_row",
            ),
        ),
        content_digest="",
    ).sealed()

    findings = validate_trace(forged)

    assert any(
        item.code == "unsupported_alias_target_kind"
        and item.detail["target_kind"] == "database_row"
        for item in findings
    )


def test_validator_accepts_all_supported_alias_targets() -> None:
    trace = _base_trace()
    actor = trace.actors[0]
    session = trace.sessions[0]
    message = trace.messages[0]
    started_at = trace.lifecycle.started_at
    span = SpanV5(
        span_id="span-alias-target",
        span_kind=SpanKind.MODEL_CALL,
        actor_id=actor.actor_id,
        session_id=session.session_id,
        started_at=started_at,
        ended_at=started_at,
        aliases=(
            AliasV1(
                "test.span",
                "span",
                "span-alias-target",
                "span",
            ),
        ),
    ).sealed()
    event = EventV5(
        event_id="event-alias-target",
        event_type=EventType.APPLICATION,
        actor_id=actor.actor_id,
        session_id=session.session_id,
        occurred_at=started_at,
        order=EventOrderV1(chronological_sequence=1),
        aliases=(
            AliasV1(
                "test.event",
                "event",
                "event-alias-target",
                "event",
            ),
        ),
    ).sealed()
    artifact = ArtifactRefV5(
        artifact_id="artifact-alias-target",
        digest="sha256:artifact",
        media_type="application/octet-stream",
        size_bytes=0,
    )
    branch = BranchV5(
        branch_id="branch-alias-target",
        head_message_id=message.message_id,
        actor_id=actor.actor_id,
        session_id=session.session_id,
    )
    error = TraceErrorV1(
        error_id="error-alias-target",
        stage="test",
        component="test",
        code="test",
        message="test",
    )
    actor = replace(
        actor,
        aliases=(AliasV1("test.actor", "actor", actor.actor_id, "actor"),),
        content_digest="",
    ).sealed()
    session = replace(
        session,
        aliases=(
            AliasV1("test.session", "session", session.session_id, "session"),
        ),
        content_digest="",
    ).sealed()
    message = replace(
        message,
        aliases=(
            AliasV1("test.message", "message", message.message_id, "message"),
        ),
        content_digest="",
    ).sealed()
    aliases = (
        AliasV1("test.trace", "trace", trace.trace_id, "trace"),
        AliasV1(
            "test.message-part",
            "message-part",
            message.parts[0].part_id,
            "part",
        ),
        AliasV1(
            "test.artifact",
            "artifact",
            artifact.artifact_id,
            "artifact",
        ),
        AliasV1("test.branch", "branch", branch.branch_id, "branch"),
        AliasV1("test.error", "error", error.error_id, "error"),
    )
    provenance = replace(
        trace.provenance,
        aliases=(
            AliasV1(
                "test.external",
                "external",
                "external-trace-id",
                "external_trace",
            ),
        ),
    )
    valid = replace(
        trace,
        provenance=provenance,
        actors=(actor,),
        sessions=(session,),
        messages=(message,),
        spans=(span,),
        events=(event,),
        artifacts=(artifact,),
        branches=(branch,),
        errors=(error,),
        aliases=aliases,
        content_digest="",
    ).sealed()

    findings = validate_trace(valid)

    assert not {
        item.code
        for item in findings
        if item.code in {
            "dangling_alias_target",
            "unsupported_alias_target_kind",
        }
    }


def _raw_finalizer(tmp_path: Path) -> tuple[CaptureSession, TraceFinalizer, object]:
    trace_id = "trace_alias_integrity"
    capture_id = "capture_alias_integrity"
    binding = mint_binding(
        trace_id=trace_id,
        capture_id=capture_id,
        workload=BindingWorkloadV1(
            kind=WorkloadKind.REACT,
            root_actor_id="actor_root",
            actor_session_id="session_root",
        ),
        capture=BindingCaptureV1(
            mode=CaptureMode.REQUIRED,
            output_artifact_root=str(tmp_path),
        ),
    )
    bundle = LocalTraceBundle(tmp_path / "bundle")
    spool = RawSpool(
        bundle.capture_root(trace_id),
        capture_id=capture_id,
        max_segment_records=128,
    )
    session = CaptureSession(binding=binding, spool=spool, blobs=bundle.blobs)
    coverage = new_coverage_receipt(
        binding_id=binding.binding_id,
        binding_digest=binding.content_digest,
        capture_id=capture_id,
        scope="model_calls_and_application",
        requested_mode=str(CaptureMode.REQUIRED),
        resolved_mode=str(CaptureMode.REQUIRED),
        interception="provider_proxy",
        proxy_config_digest=binding.capture.proxy_config_digest or "",
        started_at="2026-07-25T00:00:00Z",
    )
    finalizer = TraceFinalizer(
        binding=binding,
        spool_root=bundle.trace_root(trace_id),
        segments=(),
        provenance=TraceProvenanceV5(
            producer="alias-integrity-test",
            producer_version="1",
        ),
        identity=TraceIdentityV5(),
    )
    return session, finalizer, coverage


@pytest.mark.parametrize(
    ("target_kind", "target_id", "error"),
    (
        ("actor", "missing-actor", "is not present in the trace"),
        ("database_row", "row-1", "unsupported target_kind"),
    ),
)
def test_finalizer_rejects_invalid_alias_before_sealing(
    tmp_path: Path,
    target_kind: str,
    target_id: str,
    error: str,
) -> None:
    session, finalizer, coverage = _raw_finalizer(tmp_path)
    session.append(
        RawRecordType.ALIAS_DECLARED,
        actor_id="actor_root",
        session_id="session_root",
        payload={
            "alias": AliasV1(
                namespace="test.finalizer",
                value=target_kind,
                target_id=target_id,
                target_kind=target_kind,
            ).to_dict()
        },
    )
    session.close()
    finalizer.segments = session.spool.segments

    with pytest.raises(FinalizationError, match=error):
        finalizer.seal(coverage=coverage)


def _call_started_payload() -> dict[str, object]:
    return {
        "call_index": 1,
        "provider_adapter": "openai_responses",
        "provider_adapter_version": "1",
        "route": "/v1/responses",
        "stream": False,
        "request_digest": "sha256:request",
        "request_body": {"model": "gpt-5.4", "input": "hello"},
    }


@pytest.mark.parametrize(
    ("records", "error"),
    (
        (
            (
                (RawRecordType.MODEL_CALL_STARTED, _call_started_payload()),
                (RawRecordType.MODEL_CALL_STARTED, _call_started_payload()),
            ),
            "duplicate model_call.started",
        ),
        (
            ((RawRecordType.RESPONSE_FRAME, {"frame": "data: {}\\n\\n"}),),
            "precedes model_call.started",
        ),
        (
            ((RawRecordType.MODEL_CALL_FINISHED, {"http_status": 200}),),
            "precedes model_call.started",
        ),
        (
            (
                (RawRecordType.MODEL_CALL_STARTED, _call_started_payload()),
                (RawRecordType.RESPONSE_BODY, {"response_body": {}}),
                (RawRecordType.RESPONSE_BODY, {"response_body": {}}),
            ),
            "duplicate response.body",
        ),
        (
            (
                (RawRecordType.MODEL_CALL_STARTED, _call_started_payload()),
                (RawRecordType.MODEL_CALL_FINISHED, {"http_status": 200}),
                (RawRecordType.MODEL_CALL_FINISHED, {"http_status": 200}),
            ),
            "duplicate model_call.finished",
        ),
        (
            (
                (RawRecordType.MODEL_CALL_STARTED, _call_started_payload()),
                (RawRecordType.MODEL_CALL_FINISHED, {"http_status": 200}),
                (RawRecordType.RESPONSE_FRAME, {"frame": "data: {}\\n\\n"}),
            ),
            "follows model_call.finished",
        ),
        (
            (
                (RawRecordType.MODEL_CALL_STARTED, _call_started_payload()),
                (
                    RawRecordType.UPSTREAM_ATTEMPT_FINISHED,
                    {"attempt": 1},
                ),
            ),
            "missing upstream_attempt_id",
        ),
    ),
)
def test_finalizer_rejects_invalid_raw_call_state_machine(
    tmp_path: Path,
    records: tuple[tuple[RawRecordType, dict[str, object]], ...],
    error: str,
) -> None:
    session, finalizer, coverage = _raw_finalizer(tmp_path)
    for record_type, payload in records:
        session.append(
            record_type,
            call_id="call-integrity",
            payload=dict(payload),
        )
    session.close()
    finalizer.segments = session.spool.segments

    with pytest.raises(FinalizationError, match=error):
        finalizer.seal(coverage=coverage)


@pytest.mark.parametrize(
    ("records", "error"),
    (
        (
            (
                (RawRecordType.UPSTREAM_ATTEMPT_FINISHED, "attempt-1", 1),
            ),
            "finishes unknown upstream attempt",
        ),
        (
            (
                (RawRecordType.UPSTREAM_ATTEMPT_STARTED, "attempt-1", 1),
                (RawRecordType.UPSTREAM_ATTEMPT_STARTED, "attempt-1", 1),
            ),
            "duplicate upstream attempt",
        ),
        (
            (
                (RawRecordType.UPSTREAM_ATTEMPT_STARTED, "attempt-1", 1),
                (RawRecordType.UPSTREAM_ATTEMPT_FINISHED, "attempt-1", 2),
            ),
            "attempt number changed",
        ),
        (
            (
                (RawRecordType.UPSTREAM_ATTEMPT_STARTED, "attempt-1", 1),
                (RawRecordType.UPSTREAM_ATTEMPT_FINISHED, "attempt-1", 1),
                (RawRecordType.UPSTREAM_ATTEMPT_FINISHED, "attempt-1", 1),
            ),
            "duplicate upstream attempt finish",
        ),
    ),
)
def test_finalizer_rejects_invalid_upstream_attempt_state_machine(
    tmp_path: Path,
    records: tuple[tuple[RawRecordType, str, int], ...],
    error: str,
) -> None:
    session, finalizer, coverage = _raw_finalizer(tmp_path)
    session.append(
        RawRecordType.MODEL_CALL_STARTED,
        call_id="call-attempt-integrity",
        payload=_call_started_payload(),
    )
    for record_type, attempt_id, attempt in records:
        session.append(
            record_type,
            call_id="call-attempt-integrity",
            upstream_attempt_id=attempt_id,
            payload={"attempt": attempt},
        )
    session.close()
    finalizer.segments = session.spool.segments

    with pytest.raises(FinalizationError, match=error):
        finalizer.seal(coverage=coverage)
