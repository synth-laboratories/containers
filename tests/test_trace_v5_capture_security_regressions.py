from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
import zipfile

import httpx
from pydantic import BaseModel
import pytest
import zstandard

from synth_containers.tracing.adapters.experiments_v4 import (
    import_experiments_trace_v4,
)
from synth_containers.tracing.adapters.v4 import import_rollout_trace_v4
from synth_containers.tracing.canonical import (
    canonical_bytes,
    canonical_text,
    content_digest,
)
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
from synth_containers.tracing.capture.egress import EgressAssertion
from synth_containers.tracing.capture.envelope import RawRecordType, make_envelope
from synth_containers.tracing.capture.proxy import (
    CaptureProxy,
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
from synth_containers.tracing.models.completeness import CaptureStatus, TraceStatus
from synth_containers.tracing.models.identity import (
    AliasNamespace,
    AliasV1,
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
        policy=CapturePolicyV1(max_segment_records=128),
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
                },
            )
    finally:
        collector.stop()

    assert unauthenticated.status_code == 403
    assert authenticated.status_code == 200


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


def test_supervisor_resume_loads_exact_binding_and_continues_high_water(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "resume-bundle"
    first = CaptureSupervisor(_supervisor_config(bundle_root))
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

        resumed_envelope = resumed.session.append(
            RawRecordType.APPLICATION_EVENT,
            payload={"phase": "resumed"},
        )
        resumed.spool.close()

        assert resumed_envelope.ordinal == first_envelope.ordinal + 1
        assert [record["ordinal"] for record in resumed.spool.records()] == [0, 1]
    finally:
        resumed.proxy.stop(reason="not_started")
        resumed.collector_server.stop()


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
    supervisor.session.append(
        RawRecordType.CAPTURE_FINISHED,
        payload={"reason": "test"},
    )

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
    relay = ResponsesWebSocketRelay(session)

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
