from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
import stat
import sys
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
import zipfile

import httpx
from pydantic import BaseModel
import pytest
import zstandard

from synth_containers.tracing.adapters.experiments_v4 import (
    import_experiments_trace_v4,
)
from synth_containers.tracing.adapters.native import import_native_to_bundle
from synth_containers.tracing.adapters.v4 import import_rollout_trace_v4
from synth_containers.tracing.canonical import (
    bytes_digest,
    canonical_bytes,
    canonical_text,
    content_digest,
)
from synth_containers.tracing.cli import main as trace_cli_main
from synth_containers.tracing.capture.binding import (
    BindingCaptureV1,
    BindingWorkloadV1,
    CaptureMode,
    CapturePolicyV1,
    WorkloadKind,
    mint_binding,
)
from synth_containers.tracing.capture.collector import LocalCollector
from synth_containers.tracing.capture.collector_server import CollectorServer
from synth_containers.tracing.capture.coverage import finalization_from_dict
from synth_containers.tracing.capture.emitter import TraceEmitter
from synth_containers.tracing.capture.egress import EgressAssertion
from synth_containers.tracing.capture.envelope import RawRecordType, make_envelope
from synth_containers.tracing.capture.finalizer import FinalizationError, TraceFinalizer
from synth_containers.tracing.capture.proxy import (
    CaptureClosedError,
    CaptureProxy,
    ProxyStats,
    _StreamingContentDecoder,
    _decode_request_body,
    _forward_headers,
)
from synth_containers.tracing.capture.redaction import REDACTED
from synth_containers.tracing.capture.routes import (
    ProviderEndpointConfig,
    UpstreamAuthKind,
)
from synth_containers.tracing.capture.session import CaptureSession
from synth_containers.tracing.capture.runner import run_captured_command
from synth_containers.tracing.capture.spool import RawSpool, repair
from synth_containers.tracing.capture.supervisor import (
    CaptureNotReady,
    CaptureSupervisor,
    SupervisorConfig,
)
from synth_containers.tracing.capture.websocket import (
    ResponsesWebSocketRelay,
    ResponsesWebSocketServer,
)
from synth_containers.tracing.models.actors import ActorKind, ActorV5, SessionV5
from synth_containers.tracing.models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceStatus,
)
from synth_containers.tracing.models.identity import (
    AliasNamespace,
    AliasV1,
    TraceContextV1,
    TraceIdentityV5,
    TraceProvenanceV5,
)
from synth_containers.tracing.models.spans import SpanKind
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.validation import Severity, validate_trace


def _session(
    tmp_path: Path,
    *,
    capture_id: str = "capture_test",
    mode: CaptureMode | str = CaptureMode.REQUIRED,
    max_inline_bytes: int = 256 * 1024,
) -> CaptureSession:
    trace_id = f"trace_{capture_id}"
    binding = mint_binding(
        trace_id=trace_id,
        capture_id=capture_id,
        workload=BindingWorkloadV1(
            kind=WorkloadKind.REACT,
            root_actor_id="actor_root",
            actor_session_id="session_root",
        ),
        capture=BindingCaptureV1(
            mode=mode,
            output_artifact_root=str(tmp_path),
        ),
        policy=CapturePolicyV1(
            max_inline_bytes=max_inline_bytes,
            max_segment_records=128,
        ),
    )
    bundle = LocalTraceBundle(tmp_path / "bundle")
    return CaptureSession(
        binding=binding,
        spool=RawSpool(
            bundle.capture_root(trace_id),
            capture_id=capture_id,
            max_segment_records=128,
        ),
        blobs=bundle.blobs,
    )


def _supervisor_config(
    bundle_root: Path,
    **changes: object,
) -> SupervisorConfig:
    values: dict[str, object] = {
        "bundle_root": bundle_root,
        "trace_key": {"task": "capture-security-regression"},
        "upstream_base_url": "https://api.openai.com/v1",
        "provenance": TraceProvenanceV5(
            producer="capture-security-test",
            producer_version="1",
        ),
    }
    values.update(changes)
    return SupervisorConfig(**values)


def test_capture_session_concurrent_append_allocates_contiguous_ordinals(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)

    def append(index: int) -> tuple[int, int]:
        envelope = session.append(
            RawRecordType.APPLICATION_EVENT,
            payload={"index": index},
        )
        return envelope.ordinal, index

    with ThreadPoolExecutor(max_workers=16) as pool:
        appended = list(pool.map(append, range(64)))

    manifest = session.spool.close()
    records = list(session.spool.records())

    assert manifest.high_water_ordinal == 63
    assert sorted(ordinal for ordinal, _ in appended) == list(range(64))
    assert [record["ordinal"] for record in records] == list(range(64))
    assert {record["payload"]["index"] for record in records} == set(range(64))


def test_shared_http_websocket_stats_increment_atomically() -> None:
    stats = ProxyStats()

    def record_transport_completion(_: int) -> None:
        stats.increment(
            calls_accepted=1,
            calls_completed=1,
            calls_normalized=1,
            frames=2,
        )

    with ThreadPoolExecutor(max_workers=32) as pool:
        tuple(pool.map(record_transport_completion, range(1024)))

    assert stats.snapshot() == {
        "calls_accepted": 1024,
        "calls_completed": 1024,
        "calls_errored": 0,
        "calls_normalized": 1024,
        "upstream_retries": 0,
        "truncated_records": 0,
        "redacted_headers": (),
        "unsupported_routes": (),
        "frames": 2048,
    }


def test_proxy_stats_restore_reconstructs_errors_and_retry_high_water() -> None:
    stats = ProxyStats()

    stats.restore(
        (
            {
                "record_type": str(RawRecordType.ERROR),
                "call_id": "call_parse",
                "payload": {"stage": "request_parse"},
            },
            {
                "record_type": str(RawRecordType.MODEL_CALL_STARTED),
                "call_id": "call_upstream",
                "payload": {},
            },
            *(
                {
                    "record_type": str(RawRecordType.UPSTREAM_ATTEMPT_STARTED),
                    "call_id": "call_upstream",
                    "payload": {"attempt": attempt},
                }
                for attempt in (1, 2, 3)
            ),
            {
                "record_type": str(RawRecordType.ERROR),
                "call_id": "call_upstream",
                "payload": {"stage": "upstream_request"},
            },
        )
    )

    assert stats.calls_accepted == 2
    assert stats.calls_errored == 2
    assert stats.upstream_retries == 2


def test_proxy_and_collector_stop_before_start_return_and_are_idempotent(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    proxy = CaptureProxy(
        session,
        upstream_base_url="https://api.openai.com/v1",
    )
    collector = CollectorServer(LocalCollector(session))

    proxy.stop(reason="never_started")
    collector.stop()
    proxy.stop(reason="already_stopped")
    collector.stop()
    session.spool.close()


def test_non_loopback_collector_health_requires_registered_capture_auth(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    root_context = session.binding.context_for_child()
    collector = CollectorServer(
        LocalCollector(session),
        host="0.0.0.0",
        collector_token="collector-secret",
    )
    collector.start()
    health_url = f"http://127.0.0.1:{collector.port}/healthz"
    try:
        with httpx.Client(trust_env=False) as client:
            unauthenticated = client.get(health_url)
            authenticated = client.get(
                health_url,
                headers={
                    "authorization": "Bearer collector-secret",
                    "x-synth-trace-id": session.binding.trace_id,
                    "x-synth-capture-id": session.binding.capture_id,
                    "x-synth-actor-id": root_context.actor_id,
                    "x-synth-session-id": root_context.actor_session_id,
                },
            )
    finally:
        collector.stop()

    assert unauthenticated.status_code == 403
    assert authenticated.status_code == 200


def test_collector_infers_registered_identity_for_legacy_bearer_emitter(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, capture_id="capture_legacy_emitter")
    collector = CollectorServer(
        LocalCollector(session),
        collector_token="legacy-collector-secret",
    ).start()
    headers = {
        "authorization": "Bearer legacy-collector-secret",
        "x-synth-trace-id": session.binding.trace_id,
        "x-synth-capture-id": session.binding.capture_id,
    }
    try:
        with httpx.Client(trust_env=False) as client:
            accepted = client.post(
                f"{collector.base_url}/v1/events",
                headers=headers,
                json={
                    "event_type": "legacy.emitter",
                    "payload": {"ok": True},
                },
            )
            partial_identity = client.post(
                f"{collector.base_url}/v1/events",
                headers={
                    **headers,
                    "x-synth-actor-id": session.binding.workload.root_actor_id,
                },
                json={
                    "event_type": "legacy.emitter",
                    "payload": {"must_fail": True},
                },
            )
    finally:
        collector.stop()
        session.spool.close()

    assert accepted.status_code == 200
    assert partial_identity.status_code == 403


def test_best_effort_unknown_route_passthrough_is_same_upstream_only(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, mode=CaptureMode.BEST_EFFORT)
    proxy = CaptureProxy(
        session,
        upstream_base_url="https://provider.example/v1",
    )
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            201,
            headers={"Content-Type": "application/octet-stream"},
            content=b"upstream-body",
        )

    proxy._client.close()
    proxy._client = httpx.Client(
        transport=httpx.MockTransport(upstream),
        trust_env=False,
    )
    proxy.start()
    try:
        response = httpx.post(
            f"{proxy.base_url}/v1/files?purpose=assistants",
            headers={
                "Authorization": "Bearer child-token",
                "Content-Type": "application/octet-stream",
                "Host": "attacker.invalid",
            },
            content=b"opaque-upload",
            timeout=10.0,
        )
    finally:
        proxy.stop(reason="test_complete")
        session.spool.close()

    assert response.status_code == 201
    assert response.content == b"upstream-body"
    assert len(seen) == 1
    assert str(seen[0].url) == "https://provider.example/v1/files?purpose=assistants"
    assert seen[0].method == "POST"
    assert seen[0].content == b"opaque-upload"
    assert seen[0].headers["content-type"] == "application/octet-stream"
    assert seen[0].headers["authorization"] == "Bearer child-token"
    assert seen[0].headers["host"] == "provider.example"
    assert proxy.stats.unsupported_routes == ["/v1/files"]


def test_configured_provider_route_preserves_query_and_retained_blob_is_not_truncated(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, max_inline_bytes=128)
    proxy = CaptureProxy(
        session,
        upstream_base_url="https://provider.example/v1",
        provider_endpoints=(
            ProviderEndpointConfig(
                route="/v1/responses",
                adapter_name="openai_responses",
                upstream_base_url="https://provider.example/v1",
                auth_kind=UpstreamAuthKind.NONE,
            ),
        ),
    )
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={
                "id": "resp_query",
                "object": "response",
                "status": "completed",
                "output": [{"type": "message", "content": "x" * 512}],
                "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
            },
        )

    proxy._client.close()
    proxy._client = httpx.Client(
        transport=httpx.MockTransport(upstream),
        trust_env=False,
    )
    result = proxy.handle_provider_request(
        path="/v1/responses",
        query="include=usage%2Coutput&mode=exact",
        headers={"Content-Type": "application/json"},
        body=canonical_bytes({"model": "gpt-test", "input": "hello"}),
    )
    session.spool.close()
    response_record = next(
        record
        for record in session.spool.records()
        if record["record_type"] == RawRecordType.RESPONSE_BODY
    )
    finished_record = next(
        record
        for record in session.spool.records()
        if record["record_type"] == RawRecordType.MODEL_CALL_FINISHED
    )

    assert result.status_code == 200
    assert str(seen[0].url) == (
        "https://provider.example/v1/responses"
        "?include=usage%2Coutput&mode=exact"
    )
    assert response_record["payload"]["response_body_ref"]["disposition"] == (
        "redacted_artifact"
    )
    assert response_record["payload"]["truncated"] is False
    assert proxy.stats.truncated_records == 0
    assert finished_record["payload"]["usage"]["total_tokens"] == 8


def test_native_import_persists_only_redacted_source_artifact(tmp_path: Path) -> None:
    secret = "sk-test-native-import-secret-123456789"
    source = tmp_path / "react.json"
    source_bytes = canonical_bytes(
        {
            "api_key": secret,
            "trace_correlation_id": secret,
            "events": [
                {
                    "event_id": "step-1",
                    "event_type": "react.step",
                    "payload": {
                        "authorization": f"Bearer {secret}",
                        "action": "wait",
                    },
                }
            ],
        }
    )
    source.write_bytes(source_bytes)
    bundle = LocalTraceBundle(tmp_path / "native-import")

    imported = import_native_to_bundle(
        source,
        source_format="react",
        bundle=bundle,
    )
    stored = bundle.blobs.get(imported["stored_source_digest"])

    assert imported["source_digest"] == bytes_digest(source_bytes)
    assert imported["stored_source_digest"] != imported["source_digest"]
    assert secret.encode() not in stored
    assert REDACTED.encode() in stored
    assert secret.encode() not in bundle.archive_bytes()
    trace = bundle.read_trace(imported["trace_digest"])
    assert trace["identity"]["correlation_id"] == REDACTED
    assert bundle.verify_self_contained() == (True, ())


def test_legacy_cli_import_persists_only_redacted_source_artifact(
    tmp_path: Path,
) -> None:
    secret = "sk-test-legacy-import-secret-123456789"
    source = tmp_path / "rollout-v4.json"
    source.write_bytes(
        canonical_bytes(
            {
                "rollout_id": "rollout-secret-regression",
                "api_key": secret,
                "spans": [
                    {
                        "span_id": "call-secret-regression",
                        "call_index": 0,
                        "request": {
                            "model": "test-model",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"Bearer {secret}",
                                }
                            ],
                        },
                        "response": {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 0,
                            },
                        },
                    }
                ],
            }
        )
    )
    bundle_root = tmp_path / "legacy-import"

    assert (
        trace_cli_main(
            [
                "import",
                str(source),
                "--format",
                "containers.rollout_trace.v4",
                "--bundle",
                str(bundle_root),
            ]
        )
        == 0
    )
    bundle = LocalTraceBundle(bundle_root)

    assert secret.encode() not in bundle.archive_bytes()
    assert any(
        REDACTED.encode() in bundle.blobs.get(digest)
        for digest in bundle.read_manifest()["blob_digests"]
    )
    assert bundle.verify_self_contained() == (True, ())


def test_supervisor_resume_loads_exact_binding_and_continues_high_water(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "resume-bundle"
    first = CaptureSupervisor(_supervisor_config(bundle_root))
    standalone_actor = ActorV5(
        actor_id="actor_resume_environment",
        kind=ActorKind.ENVIRONMENT,
        display_name="resumed environment",
        parent_actor_id=first.binding.workload.root_actor_id,
    )
    durable_alias = AliasV1(
        namespace=AliasNamespace.CORRELATION,
        value="resume-environment",
        target_id=standalone_actor.actor_id,
        target_kind="actor",
    )
    first.declare_actor(standalone_actor)
    first.declare_alias(durable_alias)
    first_envelope = first.session.append(
        RawRecordType.APPLICATION_EVENT,
        payload={"phase": "first"},
    )
    first.spool.close()
    first.proxy.stop(reason="not_started")
    first.collector_server.stop()

    resumed = CaptureSupervisor(
        _supervisor_config(
            bundle_root,
            resume=True,
            trace_id=first.binding.trace_id,
            capture_id=first.binding.capture_id,
        )
    )
    try:
        assert resumed.binding_path == first.binding_path
        assert resumed.binding.to_dict() == first.binding.to_dict()
        assert resumed.spool.high_water_ordinal == first_envelope.ordinal
        assert resumed._extra_actors == [standalone_actor.sealed()]
        assert resumed._aliases == [durable_alias]

        resumed_envelope = resumed.session.append(
            RawRecordType.APPLICATION_EVENT,
            payload={"phase": "resumed"},
        )
        resumed.spool.close()

        assert resumed_envelope.ordinal == first_envelope.ordinal + 1
        assert [record["ordinal"] for record in resumed.spool.records()] == [
            0,
            1,
            2,
            3,
        ]
    finally:
        resumed.proxy.stop(reason="not_started")
        resumed.collector_server.stop()


def test_resume_rotates_child_capability_and_concurrent_finish_is_once(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "resume-child-capability"
    first = CaptureSupervisor(_supervisor_config(bundle_root))
    root = first.binding.context_for_child()
    child_context = TraceContextV1(
        trace_id=first.binding.trace_id,
        capture_id="capture_resume_child",
        actor_id="actor_resume_child",
        actor_session_id="session_resume_child",
        parent_actor_id=root.actor_id,
        parent_actor_session_id=root.actor_session_id,
        delegation_id="delegation_resume_child",
    )
    child_actor = ActorV5(
        actor_id=child_context.actor_id,
        kind=ActorKind.AGENT,
        display_name="resumed child",
        parent_actor_id=root.actor_id,
    )
    child_session = SessionV5(
        session_id=child_context.actor_session_id,
        actor_id=child_actor.actor_id,
        started_at="2026-07-25T00:00:00Z",
        capture_id=child_context.capture_id,
        parent_session_id=root.actor_session_id,
    )
    first_token = first.register_child_context(
        child_context,
        actor=child_actor,
        session=child_session,
    )
    first.spool.close()
    first.proxy.stop(reason="not_started")
    first.collector_server.stop()

    resumed = CaptureSupervisor(
        _supervisor_config(
            bundle_root,
            resume=True,
            trace_id=first.binding.trace_id,
            capture_id=first.binding.capture_id,
        )
    )
    rotated_token = resumed.collector_server.context_token(
        child_context.capture_id
    )

    assert rotated_token != first_token
    assert not resumed.collector_server.token_authorizes(
        child_context.capture_id,
        first_token,
    )
    assert resumed.collector_server.token_authorizes(
        child_context.capture_id,
        rotated_token,
    )
    assert first_token not in json.dumps(tuple(resumed.spool.records()))

    with ThreadPoolExecutor(max_workers=16) as pool:
        envelope_ids = tuple(
            pool.map(
                lambda _: resumed.finish_child_session(
                    child_context.capture_id,
                    ended_at="2026-07-25T01:00:00Z",
                ),
                range(64),
            )
        )

    assert len(set(envelope_ids)) == 1
    resumed.finalize()
    assert len(
        [
            record
            for record in resumed.spool.records()
            if record["record_type"] == str(RawRecordType.SESSION_FINISHED)
        ]
    ) == 1


def test_nonterminal_resume_rebuilds_durable_provider_counters(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "resume-provider-counters"
    first = CaptureSupervisor(_supervisor_config(bundle_root))
    first.session.append(
        RawRecordType.MODEL_CALL_STARTED,
        call_id="call_before_resume",
        payload={
            "call_index": 1,
            "provider_adapter": "openai_responses",
            "provider_adapter_version": "1",
            "route": "/v1/responses",
            "stream": False,
            "request_digest": "sha256:request",
            "request_body": {
                "model": "gpt-5.4",
                "input": "persist counters",
            },
        },
    )
    first.session.append(
        RawRecordType.RESPONSE_BODY,
        call_id="call_before_resume",
        payload={
            "http_status": 200,
            "response_body": {
                "id": "response-before-resume",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            },
        },
    )
    first.session.append(
        RawRecordType.MODEL_CALL_FINISHED,
        call_id="call_before_resume",
        payload={
            "http_status": 200,
            "provider_adapter": "openai_responses",
            "usage_observed": True,
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            },
        },
    )
    first_seen = first.session.first_observed_at
    last_seen = first.session.last_observed_at
    first.spool.close()
    first.proxy.stop(reason="not_started")
    first.collector_server.stop()

    resumed = CaptureSupervisor(
        _supervisor_config(
            bundle_root,
            resume=True,
            trace_id=first.binding.trace_id,
            capture_id=first.binding.capture_id,
        )
    )
    assert resumed.session.first_observed_at == first_seen
    assert resumed.session.last_observed_at == last_seen
    assert resumed.proxy.stats.snapshot()["calls_accepted"] == 1
    assert resumed.proxy.stats.snapshot()["calls_completed"] == 1
    assert resumed.proxy.stats.snapshot()["calls_normalized"] == 1

    sealed = resumed.finalize()

    assert sealed.coverage.calls_accepted == 1
    assert sealed.coverage.calls_completed == 1
    assert sealed.coverage.calls_normalized == 1


def test_terminal_resume_reseals_identical_trace_and_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_root = tmp_path / "terminal-resume"
    provenance_alias = AliasV1(
        namespace=AliasNamespace.CORRELATION,
        value="terminal-provenance",
        target_id="external-terminal-source",
        target_kind="external_trace",
    )
    initial_provenance = TraceProvenanceV5(
        producer="terminal-resume-producer",
        producer_version="7",
        producer_commit="abc123",
        model="gpt-5.4",
        provider="openai",
        harness="terminal-resume-harness",
        aliases=(provenance_alias,),
        extra={"stable": "provenance"},
    )
    initial_identity = TraceIdentityV5(
        run_id="run-terminal-resume",
        task_id="task-terminal-resume",
        benchmark="trace-v5",
        seed=17,
    )
    first = CaptureSupervisor(
        _supervisor_config(
            bundle_root,
            provenance=initial_provenance,
            identity=initial_identity,
            root_actor_name="terminal orchestrator",
            root_actor_kind=ActorKind.ORCHESTRATOR,
        )
    )
    standalone_before = ActorV5(
        actor_id="actor_terminal_standalone_before",
        kind=ActorKind.ENVIRONMENT,
        display_name="standalone before child",
        parent_actor_id=first.binding.workload.root_actor_id,
    )
    child_actor = ActorV5(
        actor_id="actor_terminal_child",
        kind=ActorKind.AGENT,
        display_name="terminal child",
        parent_actor_id=first.binding.workload.root_actor_id,
    )
    child_session = SessionV5(
        session_id="session_terminal_child",
        actor_id=child_actor.actor_id,
        started_at="2026-07-25T00:00:00Z",
        parent_session_id=first.binding.workload.actor_session_id,
    )
    standalone_after = ActorV5(
        actor_id="actor_terminal_standalone_after",
        kind=ActorKind.VERIFIER,
        display_name="standalone after child",
        parent_actor_id=first.binding.workload.root_actor_id,
    )
    first.declare_actor(standalone_before)
    first.declare_actor(child_actor, child_session)
    first.declare_actor(standalone_after)
    durable_alias = AliasV1(
        namespace=AliasNamespace.CORRELATION,
        value="terminal-resume-correlation",
        target_id=first.binding.trace_id,
        target_kind="trace",
    )
    first.declare_alias(durable_alias)

    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_terminal_resume",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "captured"}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 1,
                    "total_tokens": 5,
                },
            },
        )

    first.proxy._client.close()
    first.proxy._client = httpx.Client(
        transport=httpx.MockTransport(upstream),
        trust_env=False,
    )
    response = first.proxy.handle_provider_request(
        path="/v1/responses",
        headers={"content-type": "application/json"},
        body=canonical_bytes(
            {
                "model": "gpt-5.4",
                "input": "preserve this call",
            }
        ),
    )
    assert response.status_code == 200
    termination = TerminationV5(
        reason="child_exit",
        exit_code=17,
        detail="deterministic terminal resume",
    )
    original = first.finalize(
        status=TraceStatus.FAILED,
        termination=termination,
        child_exit_code=17,
    )
    records = tuple(first.spool.records())
    terminal_records = tuple(
        record
        for record in records
        if record["record_type"] == str(RawRecordType.CAPTURE_FINISHED)
    )

    assert len(terminal_records) == 1
    assert terminal_records[0] == records[-1]
    assert terminal_records[0]["payload"]["schema_version"] == (
        "synth.capture-finalization.v1"
    )
    assert original.coverage.calls_accepted == 1
    assert original.coverage.calls_completed == 1
    assert original.coverage.calls_normalized == 1
    assert original.coverage.ended_at == terminal_records[0]["occurred_at"]
    original_coverage_digest = original.coverage.content_digest
    first.materialize_projection("v4")
    assert first.sealed is not None
    assert first.sealed.coverage.content_digest == original_coverage_digest
    assert first._terminal_finalization is not None
    cyclic_seed = json.loads(
        json.dumps(first._terminal_finalization.to_dict())
    )
    cyclic_seed["coverage_seed"]["segment_count"] = 1
    cyclic_seed["coverage_seed"]["content_digest"] = content_digest(
        cyclic_seed["coverage_seed"]
    )
    cyclic_seed["content_digest"] = content_digest(cyclic_seed)
    with pytest.raises(ValueError, match="derived bundle facts"):
        finalization_from_dict(cyclic_seed)
    malformed_alias = json.loads(
        json.dumps(first._terminal_finalization.to_dict())
    )
    malformed_alias["aliases"] = [17]
    malformed_alias["content_digest"] = content_digest(malformed_alias)
    with pytest.raises(ValueError, match="non-object alias"):
        finalization_from_dict(malformed_alias)
    with pytest.raises(FinalizationError, match="provenance disagrees"):
        TraceFinalizer(
            binding=first.binding,
            spool_root=first.bundle.trace_root(first.binding.trace_id),
            segments=first.spool.segments,
            provenance=TraceProvenanceV5(
                producer="unauthorized-reseal",
                producer_version="1",
                captured_at=first._terminal_finalization.captured_at,
            ),
            identity=initial_identity,
            root_actor_name="terminal orchestrator",
            root_actor_kind=ActorKind.ORCHESTRATOR,
        ).seal(
            coverage=first._terminal_finalization.coverage_seed,
            status=TraceStatus.FAILED,
            termination=termination,
            extra_actors=tuple(first._extra_actors),
            extra_sessions=tuple(first._extra_sessions),
            aliases=(durable_alias,),
        )
    import synth_containers.tracing.capture.finalizer as finalizer_module

    with monkeypatch.context() as patch:
        patch.setattr(
            finalizer_module,
            "FINALIZER_VERSION",
            "incompatible-finalizer",
        )
        with pytest.raises(FinalizationError, match="different finalizer version"):
            TraceFinalizer(
                binding=first.binding,
                spool_root=first.bundle.trace_root(first.binding.trace_id),
                segments=first.spool.segments,
                provenance=first._terminal_finalization.provenance,
                identity=initial_identity,
                root_actor_name="terminal orchestrator",
                root_actor_kind=ActorKind.ORCHESTRATOR,
            ).seal(
                coverage=first._terminal_finalization.coverage_seed,
                status=TraceStatus.FAILED,
                termination=termination,
                extra_actors=tuple(first._extra_actors),
                extra_sessions=tuple(first._extra_sessions),
                aliases=(durable_alias,),
            )

    resealed = []
    for _ in range(2):
        resumed = CaptureSupervisor(
            _supervisor_config(
                bundle_root,
                resume=True,
                trace_id=first.binding.trace_id,
                capture_id=first.binding.capture_id,
                provenance=TraceProvenanceV5(
                    producer="changed-resume-config",
                    producer_version="999",
                ),
                identity=TraceIdentityV5(
                    benchmark="changed-resume-config",
                ),
                root_actor_name="changed resume root",
                root_actor_kind=ActorKind.TOOL,
                upstream_base_url="not-a-live-provider-url",
                proxy_host="terminal-resume-must-not-bind.invalid",
                proxy_port=70000,
                collector_host="terminal-resume-must-not-bind.invalid",
                collector_port=70001,
                websocket_host="terminal-resume-must-not-bind.invalid",
                websocket_port=70002,
                responses_websocket=True,
                provider_endpoints=(
                    ProviderEndpointConfig(
                        route="/v1/responses",
                        adapter_name="not-installed-on-resume",
                        upstream_base_url="not-a-live-provider-url",
                        auth_kind=UpstreamAuthKind.NONE,
                    ),
                ),
            )
        )
        assert resumed.proxy._stopped is True
        assert resumed.collector_server._stopped is True
        with pytest.raises(RuntimeError, match="terminal capture cannot resume"):
            resumed.start_capture()
        with pytest.raises(RuntimeError, match="frozen"):
            resumed.collector.event(
                event_type="application.after-terminal",
                payload={"forbidden": True},
            )
        resealed.append(resumed.finalize())

    for candidate in resealed:
        assert candidate.document.content_digest == original.document.content_digest
        assert candidate.coverage.content_digest == original.coverage.content_digest
        assert candidate.coverage.calls_accepted == 1
        assert candidate.coverage.calls_completed == 1
        assert candidate.coverage.calls_normalized == 1
        assert candidate.document.lifecycle.status == TraceStatus.FAILED
        assert candidate.document.lifecycle.termination == termination
        assert candidate.document.identity == initial_identity
        assert candidate.document.provenance.producer == (
            initial_provenance.producer
        )
        assert candidate.document.provenance.producer_commit == "abc123"
        assert candidate.document.provenance.aliases == (provenance_alias,)
        assert candidate.document.provenance.captured_at == (
            terminal_records[0]["occurred_at"]
        )
        assert candidate.document.aliases == (durable_alias,)
        assert candidate.document.actors[0].display_name == "terminal orchestrator"
        assert candidate.document.actors[0].kind == ActorKind.ORCHESTRATOR
        assert tuple(
            actor.actor_id for actor in candidate.document.actors[1:]
        ) == (
            standalone_before.actor_id,
            child_actor.actor_id,
            standalone_after.actor_id,
        )


def test_publication_retry_reuses_identical_terminal_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "publication-retry")
    )
    supervisor.collector.event(
        event_type="application.before-publication-retry",
        payload={"attempt": 1},
    )
    candidates: list[tuple[str, str]] = []
    original_seal = TraceFinalizer.seal
    original_write_trace = supervisor.bundle.write_trace
    write_attempts = 0

    def recording_seal(
        finalizer: TraceFinalizer,
        **kwargs: object,
    ) -> object:
        candidate = original_seal(finalizer, **kwargs)
        candidates.append(
            (
                candidate.document.content_digest,
                candidate.coverage.content_digest,
            )
        )
        return candidate

    def fail_first_write(*args: object, **kwargs: object) -> object:
        nonlocal write_attempts
        write_attempts += 1
        if write_attempts == 1:
            raise OSError("injected trace publication failure")
        return original_write_trace(*args, **kwargs)

    monkeypatch.setattr(TraceFinalizer, "seal", recording_seal)
    monkeypatch.setattr(supervisor.bundle, "write_trace", fail_first_write)

    with pytest.raises(OSError, match="injected trace publication failure"):
        supervisor.finalize()
    sealed = supervisor.finalize()

    assert len(candidates) == 2
    assert candidates[0] == candidates[1]
    assert candidates[-1] == (
        sealed.document.content_digest,
        sealed.coverage.content_digest,
    )
    assert len(
        [
            record
            for record in supervisor.spool.records()
            if record["record_type"] == str(RawRecordType.CAPTURE_FINISHED)
        ]
    ) == 1


def test_finalizer_rejects_provider_adapter_version_drift(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "adapter-version-drift")
    )
    supervisor.session.append(
        RawRecordType.MODEL_CALL_STARTED,
        call_id="call_adapter_version_drift",
        payload={
            "call_index": 1,
            "provider_adapter": "openai_responses",
            "provider_adapter_version": "999",
            "route": "/v1/responses",
            "stream": False,
            "request_digest": "sha256:request",
            "request_body": {
                "model": "gpt-5.4",
                "input": "reject changed normalizer",
            },
        },
    )
    supervisor.session.append(
        RawRecordType.MODEL_CALL_FINISHED,
        call_id="call_adapter_version_drift",
        payload={
            "http_status": 200,
            "provider_adapter": "openai_responses",
            "usage_observed": False,
        },
    )

    with pytest.raises(
        FinalizationError,
        match="does not match installed openai_responses@1",
    ):
        supervisor.finalize()


def test_finalize_waits_for_accepted_direct_provider_call(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "direct-call-drain")
    )
    upstream_entered = threading.Event()
    release_upstream = threading.Event()

    def upstream(_: httpx.Request) -> httpx.Response:
        upstream_entered.set()
        assert release_upstream.wait(timeout=10.0)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "id": "resp_drain",
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "total_tokens": 1,
                },
            },
        )

    supervisor.proxy._client.close()
    supervisor.proxy._client = httpx.Client(
        transport=httpx.MockTransport(upstream),
        trust_env=False,
    )

    def call_provider() -> object:
        return supervisor.proxy.handle_provider_request(
            path="/v1/responses",
            headers={"content-type": "application/json"},
            body=canonical_bytes(
                {"model": "gpt-5.4", "input": "drain before terminal"}
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        call_future = pool.submit(call_provider)
        assert upstream_entered.wait(timeout=10.0)
        finalize_future = pool.submit(supervisor.finalize)
        assert not finalize_future.done()
        release_upstream.set()
        response = call_future.result(timeout=10.0)
        sealed = finalize_future.result(timeout=10.0)

    assert response.status_code == 200
    assert sealed.coverage.calls_accepted == 1
    assert sealed.coverage.calls_completed == 1
    records = tuple(supervisor.spool.records())
    assert records[-1]["record_type"] == str(RawRecordType.CAPTURE_FINISHED)
    assert next(
        record["ordinal"]
        for record in records
        if record["record_type"] == str(RawRecordType.MODEL_CALL_FINISHED)
    ) < records[-1]["ordinal"]
    with pytest.raises(CaptureClosedError):
        call_provider()


def test_finalize_waits_for_models_passthrough(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "models-call-drain")
    )
    upstream_entered = threading.Event()
    release_upstream = threading.Event()

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        upstream_entered.set()
        assert release_upstream.wait(timeout=10.0)
        return httpx.Response(200, json={"object": "list", "data": []})

    supervisor.proxy._client.close()
    supervisor.proxy._client = httpx.Client(
        transport=httpx.MockTransport(upstream),
        trust_env=False,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        models_future = pool.submit(
            supervisor.proxy.handle_models,
            headers={},
        )
        assert upstream_entered.wait(timeout=10.0)
        finalize_future = pool.submit(supervisor.finalize)
        assert not finalize_future.done()
        release_upstream.set()
        response = models_future.result(timeout=10.0)
        sealed = finalize_future.result(timeout=10.0)

    assert response.status_code == 200
    assert sealed.coverage.calls_accepted == 0
    assert tuple(supervisor.spool.records())[-1]["record_type"] == str(
        RawRecordType.CAPTURE_FINISHED
    )
    with pytest.raises(CaptureClosedError):
        supervisor.proxy.handle_models(headers={})


def test_upstream_error_fact_restores_live_proxy_counters(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, capture_id="capture_upstream_error")
    proxy = CaptureProxy(
        session,
        upstream_base_url="https://api.openai.com/v1",
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider unavailable", request=request)

    proxy._client.close()
    proxy._client = httpx.Client(
        transport=httpx.MockTransport(upstream),
        trust_env=False,
    )

    with pytest.raises(httpx.ConnectError):
        proxy.handle_chat_completions(
            headers={"content-type": "application/json"},
            body=canonical_bytes({"model": "gpt-5.4", "messages": []}),
        )

    session.spool.close()
    restored = ProxyStats()
    restored.restore(tuple(session.spool.records()))
    assert proxy.stats.calls_accepted == restored.calls_accepted == 1
    assert proxy.stats.calls_errored == restored.calls_errored == 1


def test_capture_service_shutdown_error_never_writes_terminal_fact(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "shutdown-fail-closed")
    )

    class BrokenWebSocketServer:
        def stop(self) -> None:
            raise RuntimeError("relay still active")

    supervisor.websocket_server = BrokenWebSocketServer()

    with pytest.raises(CaptureNotReady, match="websocket=RuntimeError"):
        supervisor.finalize()

    assert not any(
        record["record_type"] == str(RawRecordType.CAPTURE_FINISHED)
        for record in supervisor.spool.records()
    )


def test_configured_auth_case_insensitively_replaces_caller_auth() -> None:
    endpoint = ProviderEndpointConfig(
        route="/v1/responses",
        adapter_name="openai_responses",
        upstream_base_url="https://api.openai.com/v1",
        auth_kind=UpstreamAuthKind.BEARER,
        auth_header="Authorization",
        api_key="configured-secret",
    )

    forwarded = _forward_headers(
        {
            "aUtHoRiZaTiOn": "Bearer caller-secret",
            "X-Request-ID": "request-1",
        },
        endpoint,
    )

    auth_headers = {
        name: value
        for name, value in forwarded.items()
        if name.lower() == "authorization"
    }
    assert auth_headers == {"Authorization": "Bearer configured-secret"}
    assert forwarded["X-Request-ID"] == "request-1"


def test_forward_headers_does_not_duplicate_lowercase_content_type() -> None:
    endpoint = ProviderEndpointConfig(
        route="/v1/responses",
        adapter_name="openai_responses",
        upstream_base_url="https://chatgpt.com/backend-api/codex",
        upstream_path="/responses",
    )

    forwarded = _forward_headers(
        {
            "content-type": "application/json",
            "content-encoding": "zstd",
        },
        endpoint,
    )

    assert {
        name: value
        for name, value in forwarded.items()
        if name.lower() == "content-type"
    } == {"content-type": "application/json"}


def test_openai_client_relative_upstream_path_handles_chatgpt_base_url() -> None:
    endpoint = ProviderEndpointConfig(
        route="/v1/responses",
        adapter_name="openai_responses",
        upstream_base_url="https://chatgpt.com/backend-api/codex",
        upstream_path="/responses",
    )

    assert (
        endpoint.upstream_url()
        == "https://chatgpt.com/backend-api/codex/responses"
    )


def test_codex_zstd_request_body_is_decoded_for_capture_without_changing_wire_bytes() -> None:
    body = canonical_bytes(
        {
            "model": "gpt-5.4-mini",
            "input": "bounded compressed request",
            "stream": True,
        }
    )
    wire_body = zstandard.ZstdCompressor().compress(body)

    decoded, encodings = _decode_request_body(
        {"Content-Encoding": "zstd"},
        wire_body,
    )

    assert decoded == body
    assert wire_body != body
    assert encodings == ("zstd",)


def test_gzip_response_stream_is_decoded_for_capture_without_changing_wire_chunks() -> None:
    body = b'data: {"type":"response.completed"}\n\n'
    wire_body = gzip.compress(body)
    decoder = _StreamingContentDecoder({"content-encoding": "gzip"})

    captured = b"".join(
        (
            decoder.feed(wire_body[:7]),
            decoder.feed(wire_body[7:]),
            decoder.finish(),
        )
    )

    assert captured == body
    assert wire_body != body


def test_codex_prompt_cache_key_exactly_attributes_provider_call_to_native_child(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(_supervisor_config(tmp_path / "codex-child"))
    child_actor = ActorV5(
        actor_id="actor_codex_child",
        kind=ActorKind.AGENT,
        display_name="Codex child",
        parent_actor_id=supervisor.binding.workload.root_actor_id,
    )
    child_session = SessionV5(
        session_id="session_codex_child",
        actor_id=child_actor.actor_id,
        started_at="2026-07-25T00:00:00Z",
        parent_session_id=supervisor.binding.workload.actor_session_id,
    )
    supervisor.declare_actor(child_actor, child_session)
    supervisor.declare_alias(
        AliasV1(
            namespace=AliasNamespace.CODEX_THREAD,
            value="thread-native-1",
            target_id=child_session.session_id,
            target_kind="session",
        )
    )
    supervisor.session.append(
        RawRecordType.MODEL_CALL_STARTED,
        call_id="call_codex_child",
        payload={
            "call_index": 1,
            "provider_adapter": "openai_responses",
            "route": "/v1/responses",
            "stream": False,
            "request_digest": "sha256:request",
            "request_body": {
                "model": "gpt-5.4",
                "input": "attribute this call",
                "prompt_cache_key": "thread-native-1",
            },
        },
    )
    supervisor.session.append(
        RawRecordType.RESPONSE_BODY,
        call_id="call_codex_child",
        payload={
            "http_status": 200,
            "response_body": {
                "id": "response-1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 1,
                    "total_tokens": 4,
                },
            },
        },
    )
    supervisor.session.append(
        RawRecordType.MODEL_CALL_FINISHED,
        call_id="call_codex_child",
        payload={
            "http_status": 200,
            "provider_adapter": "openai_responses",
            "usage_observed": True,
            "usage": {
                "input_tokens": 3,
                "output_tokens": 1,
                "total_tokens": 4,
            },
        },
    )
    supervisor.finish_child_session(child_session.session_id)
    sealed = supervisor.finalize()
    call_span = next(
        span
        for span in sealed.document.spans
        if str(span.span_kind) == str(SpanKind.MODEL_CALL)
    )
    sessions = {
        session.session_id: session
        for session in sealed.document.sessions
    }

    assert call_span.actor_id == child_actor.actor_id
    assert call_span.session_id == child_session.session_id
    assert call_span.detail["native_correlation"] == {
        "basis": "exact_native_alias",
        "request_field": "prompt_cache_key",
        "alias_namespace": "codex.thread",
        "alias_value": "thread-native-1",
        "alias_target_id": child_session.session_id,
    }
    assert all(
        event.actor_id == child_actor.actor_id
        and event.session_id == child_session.session_id
        for event in sealed.document.events
        if event.span_id == call_span.span_id
    )
    assert sessions[child_session.session_id].coverage.model_calls == "complete"
    assert sessions[supervisor.binding.workload.actor_session_id].coverage.model_calls == (
        "not_captured"
    )


def test_supervisor_finishes_registered_child_by_capture_or_session_id(
    tmp_path: Path,
) -> None:
    with CaptureSupervisor(
        _supervisor_config(tmp_path / "registered-child-lifecycle")
    ) as supervisor:
        child = TraceContextV1(
            trace_id=supervisor.binding.trace_id,
            capture_id="capture_registered_child",
            actor_id="actor_registered_child",
            actor_session_id="session_registered_child",
            parent_actor_id=supervisor.binding.workload.root_actor_id,
            parent_actor_session_id=supervisor.binding.workload.actor_session_id,
            delegation_id="delegation_registered_child",
        )
        actor = ActorV5(
            actor_id=child.actor_id,
            kind=ActorKind.AGENT,
            display_name="registered child",
            parent_actor_id=child.parent_actor_id,
        )
        session = SessionV5(
            session_id=child.actor_session_id,
            actor_id=child.actor_id,
            started_at="2026-07-25T00:00:00Z",
            parent_session_id=child.parent_actor_session_id,
        )
        child_token = supervisor.register_child_context(
            child,
            actor=actor,
            session=session,
        )

        first = supervisor.finish_child_session(
            child.capture_id,
            ended_at="2026-07-25T00:01:00Z",
        )
        assert supervisor.finish_child_session(child.actor_session_id) == first
        with pytest.raises(ValueError, match="conflicting"):
            supervisor.finish_child_session(
                child.actor_session_id,
                status="failed",
            )
        with pytest.raises(ValueError, match="not registered or declared"):
            supervisor.finish_child_session("session_unknown")
        with pytest.raises(ValueError, match="root session lifecycle"):
            supervisor.finish_child_session(
                supervisor.binding.workload.actor_session_id
            )

        emitter = TraceEmitter(
            supervisor.collector_server.base_url,
            child,
            collector_token=child_token,
        )
        try:
            with pytest.raises(httpx.HTTPStatusError) as late_event:
                emitter.event("agent.after_finish", {"late": True})
            assert late_event.value.response.status_code == 409
            assert supervisor._resolve_provider_context(
                emitter.provider_headers()
            ) is None
        finally:
            emitter.close()

    assert supervisor.sealed is not None
    child_session = supervisor.sealed.document.session(child.actor_session_id)
    assert child_session is not None
    assert str(child_session.status) == "completed"
    assert child_session.ended_at == "2026-07-25T00:01:00Z"
    assert str(supervisor.sealed.document.completeness.capture_status) == "complete"
    assert not {
        "session_non_terminal_in_sealed_trace",
        "terminal_session_missing_ended_at",
    } & {finding.code for finding in validate_trace(supervisor.sealed.document)}


def test_registered_parent_cannot_finish_before_declared_descendant_locally(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "mixed-child-local-finish")
    )
    root = supervisor.binding.context_for_child()
    parent = TraceContextV1(
        trace_id=supervisor.binding.trace_id,
        capture_id="capture_mixed_parent_local",
        actor_id="actor_mixed_parent_local",
        actor_session_id="session_mixed_parent_local",
        parent_actor_id=root.actor_id,
        parent_actor_session_id=root.actor_session_id,
        delegation_id="delegation_mixed_parent_local",
    )
    supervisor.register_child_context(
        parent,
        actor=ActorV5(
            actor_id=parent.actor_id,
            kind=ActorKind.AGENT,
            display_name="mixed parent local",
            parent_actor_id=root.actor_id,
        ),
        session=SessionV5(
            session_id=parent.actor_session_id,
            actor_id=parent.actor_id,
            started_at="2026-07-25T00:00:00Z",
            parent_session_id=root.actor_session_id,
        ),
    )
    descendant = SessionV5(
        session_id="session_mixed_declared_local",
        actor_id="actor_mixed_declared_local",
        started_at="2026-07-25T00:01:00Z",
        parent_session_id=parent.actor_session_id,
    )
    supervisor.declare_actor(
        ActorV5(
            actor_id=descendant.actor_id,
            kind=ActorKind.AGENT,
            display_name="mixed declared local",
            parent_actor_id=parent.actor_id,
        ),
        descendant,
    )

    with pytest.raises(ValueError, match="unterminated descendants"):
        supervisor.finish_child_session(
            parent.capture_id,
            ended_at="2026-07-25T00:03:00Z",
        )
    assert supervisor.collector_server.terminal_context_fact(parent.capture_id) is None

    supervisor.finish_child_session(
        descendant.session_id,
        ended_at="2026-07-25T00:02:00Z",
    )
    supervisor.finish_child_session(
        parent.capture_id,
        ended_at="2026-07-25T00:03:00Z",
    )
    sealed = supervisor.finalize()

    assert str(sealed.document.session(parent.actor_session_id).status) == "completed"
    assert str(sealed.document.session(descendant.session_id).status) == "completed"


def test_registered_parent_remote_finish_uses_complete_declared_topology(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "mixed-child-remote-finish")
    )
    supervisor.collector_server.start()
    root = supervisor.binding.context_for_child()
    parent = TraceContextV1(
        trace_id=supervisor.binding.trace_id,
        capture_id="capture_mixed_parent_remote",
        actor_id="actor_mixed_parent_remote",
        actor_session_id="session_mixed_parent_remote",
        parent_actor_id=root.actor_id,
        parent_actor_session_id=root.actor_session_id,
        delegation_id="delegation_mixed_parent_remote",
    )
    parent_token = supervisor.register_child_context(
        parent,
        actor=ActorV5(
            actor_id=parent.actor_id,
            kind=ActorKind.AGENT,
            display_name="mixed parent remote",
            parent_actor_id=root.actor_id,
        ),
        session=SessionV5(
            session_id=parent.actor_session_id,
            actor_id=parent.actor_id,
            started_at="2026-07-25T00:00:00Z",
            parent_session_id=root.actor_session_id,
        ),
    )
    descendant = SessionV5(
        session_id="session_mixed_declared_remote",
        actor_id="actor_mixed_declared_remote",
        started_at="2026-07-25T00:01:00Z",
        parent_session_id=parent.actor_session_id,
    )
    supervisor.declare_actor(
        ActorV5(
            actor_id=descendant.actor_id,
            kind=ActorKind.AGENT,
            display_name="mixed declared remote",
            parent_actor_id=parent.actor_id,
        ),
        descendant,
    )

    with TraceEmitter(
        supervisor.collector_server.base_url,
        parent,
        collector_token=parent_token,
    ) as emitter:
        with pytest.raises(httpx.HTTPStatusError) as premature:
            emitter.finish(ended_at="2026-07-25T00:03:00Z")
        assert premature.value.response.status_code == 400
        assert supervisor.collector_server.terminal_context_fact(parent.capture_id) is None

        supervisor.finish_child_session(
            descendant.session_id,
            ended_at="2026-07-25T00:02:00Z",
        )
        emitter.finish(ended_at="2026-07-25T00:03:00Z")

    sealed = supervisor.finalize()

    assert str(sealed.document.session(parent.actor_session_id).status) == "completed"
    assert str(sealed.document.session(descendant.session_id).status) == "completed"


def test_remote_finish_during_finalization_returns_conflict_response(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "remote-finish-finalization-race")
    )
    supervisor.collector_server.start()
    root = supervisor.binding.context_for_child()
    child = TraceContextV1(
        trace_id=supervisor.binding.trace_id,
        capture_id="capture_finish_finalization_race",
        actor_id="actor_finish_finalization_race",
        actor_session_id="session_finish_finalization_race",
        parent_actor_id=root.actor_id,
        parent_actor_session_id=root.actor_session_id,
        delegation_id="delegation_finish_finalization_race",
    )
    child_token = supervisor.register_child_context(
        child,
        actor=ActorV5(
            actor_id=child.actor_id,
            kind=ActorKind.AGENT,
            display_name="finalization race child",
            parent_actor_id=root.actor_id,
        ),
        session=SessionV5(
            session_id=child.actor_session_id,
            actor_id=child.actor_id,
            started_at="2026-07-25T00:00:00Z",
            parent_session_id=root.actor_session_id,
        ),
    )

    with TraceEmitter(
        supervisor.collector_server.base_url,
        child,
        collector_token=child_token,
    ) as emitter:
        supervisor._finalization_started = True
        try:
            with pytest.raises(httpx.HTTPStatusError) as finishing:
                emitter.finish(ended_at="2026-07-25T00:01:00Z")
            assert finishing.value.response.status_code == 409
            assert supervisor.collector_server.terminal_context_fact(child.capture_id) is None
        finally:
            supervisor._finalization_started = False

        emitter.finish(ended_at="2026-07-25T00:01:00Z")

    sealed = supervisor.finalize()

    assert str(sealed.document.session(child.actor_session_id).status) == "completed"


def test_unfinished_child_is_interrupted_and_makes_capture_partial(
    tmp_path: Path,
) -> None:
    with CaptureSupervisor(
        _supervisor_config(tmp_path / "unfinished-child-lifecycle")
    ) as supervisor:
        child = TraceContextV1(
            trace_id=supervisor.binding.trace_id,
            capture_id="capture_unfinished_child",
            actor_id="actor_unfinished_child",
            actor_session_id="session_unfinished_child",
            parent_actor_id=supervisor.binding.workload.root_actor_id,
            parent_actor_session_id=supervisor.binding.workload.actor_session_id,
            delegation_id="delegation_unfinished_child",
        )
        supervisor.register_child_context(
            child,
            actor=ActorV5(
                actor_id=child.actor_id,
                kind=ActorKind.AGENT,
                display_name="unfinished child",
                parent_actor_id=child.parent_actor_id,
            ),
            session=SessionV5(
                session_id=child.actor_session_id,
                actor_id=child.actor_id,
                started_at="2026-07-25T00:00:00Z",
                parent_session_id=child.parent_actor_session_id,
            ),
        )

    assert supervisor.sealed is not None
    document = supervisor.sealed.document
    child_session = document.session(child.actor_session_id)
    assert child_session is not None
    assert str(child_session.status) == "interrupted"
    assert child_session.ended_at is not None
    assert "child_session_not_finished" in child_session.coverage.reasons
    assert str(document.completeness.capture_status) == "partial"
    assert f"child_session_not_finished:{child.actor_session_id}" in (
        document.completeness.reasons
    )
    assert str(supervisor.sealed.coverage.completeness) == "partial"

    lifecycle_findings = {
        finding.code for finding in validate_trace(document)
    }
    assert "session_non_terminal_in_sealed_trace" not in lifecycle_findings
    forged_sessions = tuple(
        (
            replace(
                session,
                status="running",
                ended_at=None,
                content_digest="",
            ).sealed()
            if session.session_id == child.actor_session_id
            else session
        )
        for session in document.sessions
    )
    forged = replace(
        document,
        sessions=forged_sessions,
        content_digest="",
    ).sealed()
    forged_findings = validate_trace(forged)
    assert any(
        finding.code == "session_non_terminal_in_sealed_trace"
        and finding.entity_id == child.actor_session_id
        for finding in forged_findings
    )


def test_finalizer_rejects_local_child_record_after_terminal_fact(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(tmp_path / "late-local-child-record")
    )
    child_actor = ActorV5(
        actor_id="actor_late_local_child",
        kind=ActorKind.ENVIRONMENT,
        display_name="local environment child",
    )
    child_session = SessionV5(
        session_id="session_late_local_child",
        actor_id=child_actor.actor_id,
        started_at="2026-07-25T00:00:00Z",
    )
    supervisor.declare_actor(child_actor, child_session)
    supervisor.finish_child_session(
        child_session.session_id,
        ended_at="2026-07-25T00:01:00Z",
    )
    supervisor.collector.event(
        event_type="environment.after_finish",
        payload={"late": True},
        actor_id=child_actor.actor_id,
        session_id=child_session.session_id,
        occurred_at="2026-07-25T00:02:00Z",
    )
    supervisor.session.append(
        RawRecordType.CAPTURE_FINISHED,
        payload={"reason": "test"},
    )
    supervisor.session.close()
    finalizer = TraceFinalizer(
        binding=supervisor.binding,
        spool_root=supervisor.bundle.trace_root(supervisor.binding.trace_id),
        segments=supervisor.spool.segments,
        provenance=supervisor.config.provenance,
        identity=supervisor.config.identity,
    )

    with pytest.raises(FinalizationError, match="after session.finished"):
        finalizer.seal(
            coverage=supervisor.receipt,
            extra_actors=tuple(supervisor._extra_actors),
            extra_sessions=tuple(supervisor._extra_sessions),
        )


def test_generic_runner_honors_external_binding_and_materializes_projections(
    tmp_path: Path,
) -> None:
    binding = mint_binding(
        trace_id="trace_external_runner",
        capture_id="capture_external_runner",
        workload=BindingWorkloadV1(
            kind=WorkloadKind.REACT,
            root_actor_id="actor_external_runner",
            actor_session_id="session_external_runner",
        ),
        capture=BindingCaptureV1(
            mode=CaptureMode.REQUIRED,
            output_artifact_root="/artifacts/synth-traces",
        ),
    )
    binding_source = binding.write(tmp_path / "binding-source")
    bundle_root = tmp_path / "runner-bundle"

    result = run_captured_command(
        _supervisor_config(
            bundle_root,
            binding_path=binding_source,
        ),
        (
            sys.executable,
            "-c",
            (
                "import os,sys;"
                "required=('OPENAI_BASE_URL','SYNTH_TRACE_BINDING_PATH',"
                "'SYNTH_TRACE_COLLECTOR_URL','SYNTH_TRACE_ID');"
                "sys.exit(0 if all(os.environ.get(k) for k in required) else 9)"
            ),
        ),
        projections=("v4", "atif"),
    )
    bundle = LocalTraceBundle(bundle_root)

    assert result.exit_code == 0
    assert result.receipt["trace_id"] == binding.trace_id
    assert result.receipt["capture_id"] == binding.capture_id
    assert result.receipt["command"]["argument_count"] == 2
    assert "required=" not in str(result.receipt)
    assert {item["kind"] for item in result.receipt["projections"]} == {
        "atif",
        "v4",
    }
    assert bundle.verify_self_contained() == (True, ())
    manifest = bundle.read_manifest()
    assert len(manifest["projection_digests"]) == 2


def test_responses_websocket_relay_redacts_and_preserves_per_call_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path, capture_id="capture_websocket")
    request = json.dumps(
        {
            "type": "response.create",
            "response": {
                "model": "gpt-5.4",
                "input": "hello",
                "prompt_cache_key": "thread-ws-1",
                "api_key": "sk-websocket-secret-value",
            },
        }
    )
    terminal = json.dumps(
        {
            "type": "response.completed",
            "response": {
                "id": "response-ws-1",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
                "api_key": "sk-websocket-response-secret",
            },
        }
    )

    class FakeDownstream:
        request = SimpleNamespace(
            headers={
                "authorization": "Bearer downstream-memory-only-token",
                "x-synth-trace-id": "must-not-forward",
            }
        )

        def __init__(self) -> None:
            self.received = [request]
            self.sent: list[str] = []

        async def recv(self) -> str:
            if not self.received:
                raise StopAsyncIteration
            return self.received.pop(0)

        async def send(self, message: str) -> None:
            self.sent.append(message)

    class FakeUpstream:
        def __init__(self) -> None:
            self.received: list[str] = []
            self.events = [terminal]

        async def __aenter__(self) -> "FakeUpstream":
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def send(self, message: str) -> None:
            self.received.append(message)

        def __aiter__(self) -> "FakeUpstream":
            return self

        async def __anext__(self) -> str:
            if not self.events:
                raise StopAsyncIteration
            return self.events.pop(0)

    upstream = FakeUpstream()
    connect_arguments: dict[str, object] = {}

    def fake_connect(url: str, **kwargs: object) -> FakeUpstream:
        connect_arguments.update({"url": url, **kwargs})
        return upstream

    import websockets.asyncio.client

    monkeypatch.setattr(websockets.asyncio.client, "connect", fake_connect)
    downstream = FakeDownstream()
    stats = ProxyStats()
    relay = ResponsesWebSocketRelay(session, stats=stats)

    with pytest.raises(StopAsyncIteration):
        asyncio.run(
            relay.relay(
                downstream,
                actor_id="actor_ws_child",
                session_id="session_ws_child",
            )
        )
    session.spool.close()
    records = list(session.spool.records())
    started = next(
        item
        for item in records
        if item["record_type"] == str(RawRecordType.MODEL_CALL_STARTED)
    )
    frame = next(
        item
        for item in records
        if item["record_type"] == str(RawRecordType.RESPONSE_FRAME)
    )
    finished = next(
        item
        for item in records
        if item["record_type"] == str(RawRecordType.MODEL_CALL_FINISHED)
    )

    assert upstream.received == [request]
    assert downstream.sent == [terminal]
    assert started["actor_id"] == "actor_ws_child"
    assert started["session_id"] == "session_ws_child"
    assert started["payload"]["call_index"] == 1
    assert started["payload"]["route"] == "/v1/responses"
    assert started["payload"]["upstream_host"] == "api.openai.com"
    assert started["payload"]["upstream_path"] == "/v1/responses"
    assert started["payload"]["provider_adapter_version"] == "1"
    assert (
        started["payload"]["request_body"]["response"]["api_key"]
        == REDACTED
    )
    assert started["payload"]["request_body"]["response"]["prompt_cache_key"] == (
        "thread-ws-1"
    )
    assert frame["sequence_in_call"] == 0
    assert "sk-websocket-response-secret" not in frame["payload"]["frame"]
    assert finished["payload"]["frames"] == 1
    assert finished["payload"]["usage_observed"] is True
    assert stats.calls_accepted == 1
    assert stats.calls_completed == 1
    assert stats.calls_errored == 0
    assert stats.calls_normalized == 1
    assert stats.frames == 1
    forwarded = connect_arguments["additional_headers"]
    assert isinstance(forwarded, dict)
    assert forwarded["authorization"] == "Bearer downstream-memory-only-token"
    assert "x-synth-trace-id" not in forwarded


def test_capture_session_allocates_unique_call_ids_across_concurrent_transports(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path, capture_id="capture_concurrent_calls")

    with ThreadPoolExecutor(max_workers=8) as pool:
        allocated = list(
            pool.map(
                lambda kind: session.mint_call(kind=kind),
                ("model_call", "responses_websocket") * 16,
            )
        )

    call_ids = [call_id for call_id, _ in allocated]
    call_indices = [call_index for _, call_index in allocated]
    assert len(call_ids) == len(set(call_ids))
    assert sorted(call_indices) == list(range(1, len(allocated) + 1))


def test_responses_websocket_server_rejects_missing_capture_context() -> None:
    class FakeRelay:
        def __init__(self) -> None:
            self.calls = 0

        async def relay(self, *_: object, **__: object) -> None:
            self.calls += 1

    class FakeConnection:
        request = SimpleNamespace(path="/v1/responses", headers={})

        def __init__(self) -> None:
            self.closed: tuple[int, str] | None = None

        async def close(self, *, code: int, reason: str) -> None:
            self.closed = (code, reason)

    relay = FakeRelay()
    server = ResponsesWebSocketServer(
        relay,
        context_resolver=lambda _headers: None,
    )
    connection = FakeConnection()
    try:
        asyncio.run(server._handle_connection(connection))
    finally:
        server._loop.close()

    assert connection.closed == (1008, "capture context required")
    assert relay.calls == 0


def test_runner_websocket_url_uses_secret_context_capability(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        _supervisor_config(
            tmp_path / "runner-websocket",
            responses_websocket=True,
        )
    )
    try:
        supervisor.start_capture()
        environment = supervisor.environment()
        descriptor = supervisor.environment_descriptor()
        parsed = urlsplit(environment["OPENAI_RESPONSES_WEBSOCKET_URL"])
        tokens = parse_qs(parsed.query).get("synth_trace_token", ())

        assert parsed.path == "/v1/responses"
        assert len(tokens) == 1
        assert tokens[0].startswith("sk_trace_")
        assert tokens[0] not in json.dumps(descriptor)
        assert supervisor._resolve_websocket_context_token(tokens[0]) == (
            supervisor.binding.context_for_child()
        )
        assert supervisor._resolve_websocket_context_token("wrong-token") is None
    finally:
        supervisor.finalize(status=TraceStatus.COMPLETED)


def test_responses_websocket_server_accepts_capability_without_custom_headers() -> None:
    context = TraceContextV1(
        trace_id="trace_runner_ws",
        capture_id="capture_runner_ws",
        actor_id="actor_runner_ws",
        actor_session_id="session_runner_ws",
    )

    class FakeRelay:
        def __init__(self) -> None:
            self.received: tuple[str, str] | None = None

        async def relay(
            self,
            _connection: object,
            *,
            actor_id: str,
            session_id: str,
        ) -> None:
            self.received = (actor_id, session_id)

    class FakeConnection:
        request = SimpleNamespace(
            path="/v1/responses?synth_trace_token=runner-capability",
            headers={},
        )

        def __init__(self) -> None:
            self.closed: tuple[int, str] | None = None

        async def close(self, *, code: int, reason: str) -> None:
            self.closed = (code, reason)

    relay = FakeRelay()
    server = ResponsesWebSocketServer(
        relay,
        context_resolver=lambda _headers: None,
        query_context_resolver=(
            lambda token: context if token == "runner-capability" else None
        ),
    )
    connection = FakeConnection()
    try:
        asyncio.run(server._handle_connection(connection))
    finally:
        server._loop.close()

    assert connection.closed is None
    assert relay.received == ("actor_runner_ws", "session_runner_ws")


def test_responses_websocket_stop_aborts_hung_upstream_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import websockets.asyncio.client

    real_connect = websockets.asyncio.client.connect
    upstream_waiting = threading.Event()
    upstream_closed = threading.Event()

    class HungUpstream:
        async def __aenter__(self) -> "HungUpstream":
            return self

        async def __aexit__(self, *_: object) -> None:
            upstream_closed.set()

        async def send(self, _message: str) -> None:
            return None

        def __aiter__(self) -> "HungUpstream":
            return self

        async def __anext__(self) -> str:
            upstream_waiting.set()
            await asyncio.Future()
            raise AssertionError("hung upstream unexpectedly resumed")

    monkeypatch.setattr(
        websockets.asyncio.client,
        "connect",
        lambda *_args, **_kwargs: HungUpstream(),
    )
    session = _session(tmp_path, capture_id="capture_hung_websocket")
    context = session.binding.context_for_child()
    stats = ProxyStats()
    server = ResponsesWebSocketServer(
        ResponsesWebSocketRelay(session, stats=stats),
        context_resolver=lambda _headers: context,
    ).start()

    async def client() -> None:
        async with real_connect(server.url) as connection:
            await connection.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "model": "gpt-5.4",
                            "input": "wait forever",
                        },
                    }
                )
            )
            await connection.wait_closed()

    with ThreadPoolExecutor(max_workers=1) as pool:
        client_future = pool.submit(asyncio.run, client())
        assert upstream_waiting.wait(timeout=10.0)
        server.stop()
        client_future.result(timeout=10.0)

    session.spool.close()
    records = tuple(session.spool.records())
    finished = next(
        record
        for record in records
        if record["record_type"] == str(RawRecordType.MODEL_CALL_FINISHED)
    )
    assert upstream_closed.wait(timeout=1.0)
    assert not server._thread.is_alive()
    assert finished["payload"]["provider_status"] == "capture_shutdown"
    assert stats.calls_accepted == 1
    assert stats.calls_errored == 1
    assert stats.calls_normalized == 1


def test_raw_repair_does_not_promote_complete_logically_forged_envelope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    segments = root / "segments"
    segments.mkdir(parents=True)
    valid = make_envelope(
        capture_id="capture_forgery",
        ordinal=0,
        record_type=RawRecordType.APPLICATION_EVENT,
        actor_id="actor",
        session_id="session",
        payload={"complete": True},
    )
    forged = replace(
        valid,
        envelope_id="raw_forged",
        content_digest="",
    ).sealed()
    assert forged.content_digest == content_digest(forged)
    (segments / "000001.partial").write_text(
        canonical_text(forged) + "\n",
        encoding="utf-8",
    )

    result = repair(root, capture_id="capture_forgery")

    assert result.repaired is True
    assert result.recovered_records == 0
    assert result.reason == "partial_had_no_complete_records"
    assert not list(segments.glob("*.jsonl"))


def test_safe_zip_extraction_succeeds_and_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    bundle = LocalTraceBundle(tmp_path / "bundle")
    bundle.write_receipt("safe", {"ok": True})
    manifest = bundle.write_manifest()
    archive_path = tmp_path / "bundle.zip"
    bundle.write_archive(archive_path)

    imported = LocalTraceBundle.extract_archive(
        archive_path,
        tmp_path / "imported",
    )
    assert imported.read_manifest()["content_digest"] == manifest.content_digest
    assert imported.verify_self_contained() == (True, ())

    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escaped.txt", b"escaped")
    with pytest.raises(ValueError, match="unsafe archive path"):
        LocalTraceBundle.extract_archive(traversal, tmp_path / "traversal-target")
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "traversal-target").exists()

    symlink = tmp_path / "symlink.zip"
    symlink_info = zipfile.ZipInfo("link")
    symlink_info.create_system = 3
    symlink_info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(symlink_info, b"target")
    with pytest.raises(ValueError, match="symlink is forbidden"):
        LocalTraceBundle.extract_archive(symlink, tmp_path / "symlink-target")
    assert not (tmp_path / "symlink-target").exists()


def test_required_egress_failure_writes_verifiable_partial_bundle_before_raising(
    tmp_path: Path,
) -> None:
    assertion = EgressAssertion(
        allowed_hosts=("api.openai.com",),
        observed_hosts=("direct.invalid",),
        violations=("direct.invalid",),
    )
    supervisor: CaptureSupervisor

    with pytest.raises(CaptureNotReady, match="egress assertion failed"):
        with CaptureSupervisor(
            _supervisor_config(
                tmp_path / "egress-bundle",
                mode=CaptureMode.REQUIRED_EGRESS_ASSERTED,
                egress_assertion=lambda: assertion,
            )
        ) as supervisor:
            supervisor.collector.event(
                event_type="application.before-egress-failure",
                payload={"observed": True},
            )

    assert supervisor.sealed is not None
    assert supervisor.sealed.document.completeness.capture_status == CaptureStatus.PARTIAL
    assert supervisor.sealed.document.lifecycle.status == TraceStatus.COMPLETED
    assert all(
        finding.severity != Severity.ERROR
        for finding in validate_trace(supervisor.sealed.document)
    )
    assert supervisor.bundle.verify_self_contained() == (True, ())
    assert len(supervisor.bundle.read_manifest()["traces"]) == 1
    assert supervisor.bundle.archive_bytes()


def test_canonical_rejects_unordered_set_inside_pydantic_model() -> None:
    class Payload(BaseModel):
        labels: set[str]

    with pytest.raises(TypeError, match=r"unordered container.*\$\.labels"):
        canonical_bytes(Payload(labels={"beta", "alpha"}))


def test_v4_importers_produce_valid_completed_terminal_lifecycles() -> None:
    imported = (
        import_rollout_trace_v4(
            {
                "rollout_id": "rollout-v4",
                "trace_correlation_id": "correlation-v4",
                "spans": [],
            }
        ),
        import_experiments_trace_v4(
            {
                "trace_id": "experiments-v4",
                "attempt_id": "attempt-v4",
                "interaction": {"react_turns": [], "nev": []},
                "environment": {"bundle": "craftax"},
                "operations": {"llm_calls": 0},
                "timestamps": {
                    "started_at": "2026-07-25T00:00:00Z",
                    "completed_at": "2026-07-25T00:00:01Z",
                },
            }
        ),
    )

    for document in imported:
        assert document.lifecycle.status == TraceStatus.COMPLETED
        assert document.completeness.terminal_event_observed is True
        assert validate_trace(document) == []
