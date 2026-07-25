from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
import zipfile
from types import SimpleNamespace

import httpx
import pytest

from synth_containers.tracing.adapters.atif import export_atif, import_atif
from synth_containers.tracing.adapters.codex_jsonl import import_codex_jsonl
from synth_containers.tracing.adapters.native import (
    import_native_to_bundle,
    write_imported_document,
)
from synth_containers.tracing.adapters.legacy import import_legacy
from synth_containers.tracing.canonical import bytes_digest, canonical_bytes
from synth_containers.tracing.projections.v4 import project_v4
from synth_containers.tracing.models.identity import TraceContextV1
from synth_containers.tracing.models.identity import TraceProvenanceV5
from synth_containers.tracing.models.evidence import TraceEvidenceBundleV5, TraceRefV5
from synth_containers.tracing.capture.collector_server import CollectorServer
from synth_containers.tracing.capture.emitter import TraceEmitter
from synth_containers.tracing.capture.redaction import REDACTED, redact_payload
from synth_containers.tracing.capture.binding import CaptureMode, Interception
from synth_containers.tracing.capture.supervisor import CaptureSupervisor, SupervisorConfig
from synth_containers.tracing.cli import main as trace_cli_main
from synth_containers.tracing.store.s3 import S3BlobStore
from synth_containers.tracing.adapters.openai_responses import OpenAIResponsesAdapter
from synth_containers.tracing.adapters.anthropic_messages import AnthropicMessagesAdapter
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.store.projection import catalog_projection
from synth_containers.tracing.store.sqlite_catalog import SqliteCatalogStore
from synth_containers.tracing.validation.schema import all_schemas
from synth_containers.tracing.models.standards import (
    CriterionDefinitionV1,
    CriterionResultV1,
    RubricDefinitionV2,
    aggregate_rubric_score,
)


def test_top_level_tracing_import_has_no_adapter_capture_cycle() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from synth_containers.tracing import TraceContextV1;"
                "from synth_containers.tracing.adapters import "
                "import_optimizer_event_history, provider_adapters;"
                "assert TraceContextV1 and import_optimizer_event_history;"
                "assert provider_adapters()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_canonical_rejects_unordered_containers() -> None:
    with pytest.raises(TypeError, match="unordered container"):
        canonical_bytes({"unsafe": {"b", "a"}})


def test_payload_redaction_covers_compound_credential_keys() -> None:
    payload = {
        "SYNTH_TRACE_COLLECTOR_TOKEN": "short-random-value",
        "openaiApiKey": "another-short-value",
        "input_tokens": 12,
        "token_usage": {"prompt": 12},
    }
    redacted, report = redact_payload(payload)
    assert redacted["SYNTH_TRACE_COLLECTOR_TOKEN"] == REDACTED
    assert redacted["openaiApiKey"] == REDACTED
    assert redacted["input_tokens"] == 12
    assert redacted["token_usage"] == {"prompt": 12}
    assert report.redacted_body_keys == (
        "openai_api_key",
        "synth_trace_collector_token",
    )


def test_atif_import_is_deterministic_and_exports_17() -> None:
    payload = {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "trajectory-1",
        "agent": {"name": "agent", "version": "1"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2026-01-01T00:00:00Z",
                "source": "user",
                "message": "hello",
            }
        ],
    }
    first = import_atif(payload)
    second = import_atif(payload)
    assert first.content_digest == second.content_digest
    assert import_legacy(payload, source_format="atif").canonical is not None
    exported = export_atif(first)
    assert exported["schema_version"] == "ATIF-v1.7"
    assert exported["trajectory_id"] == first.trace_id


def test_optimizer_event_history_import_preserves_model_and_application_events(
    tmp_path: Path,
) -> None:
    payload = {
        "rollout_id": "optimizer-rollout-1",
        "metadata": {"trace_correlation_id": "corr-optimizer-1"},
        "event_history": [
            {
                "type": "lm_call",
                "span_id": "native-call-1",
                "sequence_index": 0,
                "occurred_at": "2026-07-25T01:02:03Z",
                "llm_request": {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
                "llm_response": {
                    "message": {"role": "assistant", "content": "world"},
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
                "metadata": {"optimizer_step": 7},
            },
            {
                "event_id": "reward-1",
                "event_type": "environment.reward",
                "occurred_at": "2026-07-25T01:02:04Z",
                "reward": 0.75,
                "api_key": "sk-test-secret-value",
                "metadata": {"source": "task"},
            },
        ],
    }
    first = import_legacy(payload, source_format="optimizer.event_history")
    second = import_legacy(payload, source_format="optimizer.event_history")

    assert first.canonical is not None
    assert second.canonical is not None
    assert first.canonical.content_digest == second.canonical.content_digest
    assert first.coverage == "partial_model_and_application_events"
    assert first.canonical.completeness.capture_status == "partial"
    assert first.canonical.completeness.model_calls == "partial"
    assert first.canonical.completeness.environment_events == "partial"
    assert first.canonical.completeness.usage == "aggregate_only"
    assert first.canonical.usage.provenance == "derived"
    assert first.canonical.usage.total_tokens == 5
    assert first.canonical.completeness.expected_record_count == 2
    assert first.canonical.completeness.captured_record_count == 2
    assert first.canonical.lifecycle.started_at == "2026-07-25T01:02:03Z"
    assert first.canonical.lifecycle.ended_at == "2026-07-25T01:02:04Z"
    assert first.canonical.sessions[0].started_at == "2026-07-25T01:02:03Z"
    assert first.canonical.sessions[0].ended_at == "2026-07-25T01:02:04Z"
    assert len(first.canonical.spans_of_kind("model_call")) == 1
    assert [str(item.event_type) for item in first.canonical.events] == [
        "model_call.finished",
        "environment.reward",
    ]
    assert first.canonical.events[0].payload["native_event"]["metadata"] == {
        "optimizer_step": 7
    }
    assert first.canonical.events[1].payload["native_event"]["reward"] == 0.75
    assert first.canonical.events[1].payload["native_event"]["api_key"] == REDACTED

    source = tmp_path / "event-history.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    bundle = LocalTraceBundle(tmp_path / "event-history-bundle")
    assert trace_cli_main(
        [
            "import",
            str(source),
            "--format",
            "optimizer.event_history",
            "--bundle",
            str(bundle.root),
        ]
    ) == 0
    manifest = bundle.read_manifest()
    imported = bundle.read_trace(manifest["traces"][0]["trace_digest"])
    assert imported["provenance"]["source_format"] == "optimizer.event_history"
    assert len(imported["spans"]) == 1
    assert len(imported["events"]) == 2

    missing_usage = import_legacy(
        {
            "event_history": [
                {
                    "type": "lm_call",
                    "llm_request": {"messages": []},
                    "llm_response": {"message": {"role": "assistant", "content": ""}},
                }
            ]
        },
        source_format="optimizer.event_history",
    ).canonical
    assert missing_usage is not None
    assert missing_usage.completeness.usage == "unavailable"
    assert missing_usage.usage.provenance == "unavailable"
    assert missing_usage.usage.total_tokens is None


def test_projection_manifest_uses_unknowns_standalone_and_binding_in_bundle(
    tmp_path: Path,
) -> None:
    standalone = import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "projection-standalone",
            "agent": {"name": "agent", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": "hello"}],
        }
    )
    _, standalone_manifest = project_v4(standalone)
    assert standalone_manifest.source_binding_id is None
    assert standalone_manifest.source_binding_digest is None
    assert standalone_manifest.capture_policy == {}
    assert standalone_manifest.capture_policy_digest is None
    assert standalone_manifest.redaction_profile == "unknown"
    assert standalone_manifest.redaction_provenance == "unknown"

    config = SupervisorConfig(
        bundle_root=tmp_path / "projected",
        trace_key={"task": "projection-policy"},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
    )
    with CaptureSupervisor(config) as supervisor:
        supervisor.collector.event(event_type="application.smoke", payload={"ok": True})
    supervisor.materialize_projection("v4")
    assert trace_cli_main(["project", str(config.bundle_root), "--format", "atif"]) == 0

    projection_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((config.bundle_root / "projections").glob("*/*.json"))
    ]
    assert {item["manifest"]["producer"] for item in projection_records} == {
        "synth_containers.tracing.projections.v4",
        "synth_containers.tracing.cli",
    }
    for record in projection_records:
        manifest = record["manifest"]
        assert manifest["source_binding_id"] == supervisor.binding.binding_id
        assert manifest["source_binding_digest"] == supervisor.binding.content_digest
        assert manifest["capture_policy"] == supervisor.binding.policy.to_dict()
        assert manifest["capture_policy_digest"] == supervisor.binding.policy.digest()
        assert manifest["redaction_profile"] == (
            supervisor.binding.policy.redaction_profile
        )
        assert manifest["redaction_provenance"] == (
            "source_capture_binding.policy.redaction_profile"
        )


def test_bundle_archive_extracts_and_verifies(tmp_path: Path) -> None:
    bundle = LocalTraceBundle(tmp_path / "bundle")
    bundle.write_receipt("smoke", {"ok": True})
    manifest = bundle.write_manifest()
    assert {item.kind for item in manifest.objects} == {"receipt"}
    body = bundle.archive_bytes()
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        archive.extractall(extracted)
    reopened = LocalTraceBundle(extracted)
    assert reopened.read_manifest()["content_digest"] == manifest.content_digest
    assert reopened.verify_self_contained() == (True, ())


def test_real_raw_segment_bundle_verifies_and_archives(tmp_path: Path) -> None:
    config = SupervisorConfig(
        bundle_root=tmp_path / "captured",
        trace_key={"task": "raw-segment"},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
    )
    with CaptureSupervisor(config) as supervisor:
        supervisor.collector.event(
            event_type="application.smoke",
            payload={"ok": True},
        )
    assert supervisor.bundle.verify_self_contained() == (True, ())
    assert supervisor.bundle.archive_bytes()


def test_cli_validate_inventories_its_receipt(tmp_path: Path) -> None:
    config = SupervisorConfig(
        bundle_root=tmp_path / "validated",
        trace_key={"task": "validation-receipt"},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
    )
    with CaptureSupervisor(config) as supervisor:
        supervisor.collector.event(
            event_type="application.smoke",
            payload={"ok": True},
        )

    assert trace_cli_main(["validate", str(config.bundle_root)]) == 0
    manifest = supervisor.bundle.read_manifest()
    assert any(
        "validation" in str(path)
        for path in manifest.get("receipt_paths") or ()
    )
    assert supervisor.bundle.verify_self_contained() == (True, ())


def test_fts_indexes_imported_trace(tmp_path: Path) -> None:
    trace = import_atif(
        {
            "schema_version": "ATIF-v1.5",
            "trajectory_id": "t",
            "agent": {"name": "agent", "version": "1"},
            "steps": [
                {"step_id": 1, "source": "user", "message": "needle"}
            ],
        }
    )
    catalog = SqliteCatalogStore(tmp_path / "catalog.sqlite3")
    try:
        catalog.index_trace(trace)
        assert list(catalog.search("needle"))[0]["kind"] == "message"
    finally:
        catalog.close()


def test_cli_structured_search_accepts_full_text_and_exact_filters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "structured-search",
            "agent": {"name": "agent", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": "needle"}],
        }
    )
    trace = replace(
        original,
        identity=replace(original.identity, task_id="task-structured-search"),
        content_digest="",
    ).sealed()
    bundle = LocalTraceBundle(tmp_path / "bundle")
    imported = write_imported_document(
        trace,
        source_digest=bytes_digest(b"structured-search-source"),
        source_format="atif",
        bundle=bundle,
    )

    assert trace_cli_main(["rebuild", str(bundle.root)]) == 0
    capsys.readouterr()
    assert (
        trace_cli_main(
            [
                "search",
                str(bundle.root),
                "needle",
                "--task-id",
                "task-structured-search",
                "--trace-digest",
                imported["trace_digest"],
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert imported["trace_digest"] in output
    assert "task-structured-search" in output


def test_catalog_projection_matches_managed_store_rows() -> None:
    trace = import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "projection-smoke",
            "agent": {"name": "agent", "version": "1"},
            "steps": [
                {
                    "step_id": 1,
                    "source": "user",
                    "message": "searchable",
                }
            ],
        }
    )
    projected_trace = catalog_projection(trace)
    assert projected_trace["documents"][0]["trace_digest"] == trace.content_digest
    assert projected_trace["entities"][0]["trace_digest"] == trace.content_digest
    assert {"documents", "entities", "relationships", "aliases", "evidence"} <= set(
        projected_trace
    )

    evidence = TraceEvidenceBundleV5(
        bundle_id="evidence_projection",
        trace_ref=TraceRefV5(trace_id=trace.trace_id, content_digest=trace.content_digest),
        created_at="1970-01-01T00:00:00Z",
    ).sealed()
    projected_evidence = catalog_projection(evidence)
    assert projected_evidence["documents"] == []
    assert projected_evidence["evidence"] == []


def test_schema_registry_includes_eval_receipts_and_bundle() -> None:
    names = set(all_schemas())
    assert {
        "EvaluationResultV1",
        "BenchmarkVerdictV1",
        "ReceiptV1",
        "BundleManifestV1",
        "BundleManifestPointerV1",
        "BundleObjectRefV1",
    } <= names


def test_trace_context_parent_delegation_roundtrip() -> None:
    context = TraceContextV1(
        trace_id="trace_1",
        capture_id="cap_1",
        actor_id="actor_1",
        actor_session_id="session_1",
        parent_actor_id="actor_parent",
        parent_actor_session_id="session_parent",
        parent_span_id="span_parent",
        delegation_id="delegation_1",
    )
    restored = TraceContextV1.from_environment(context.to_environment())
    assert restored is not None
    assert restored.parent_actor_id == context.parent_actor_id
    assert restored.parent_actor_session_id == context.parent_actor_session_id
    assert restored.parent_span_id == context.parent_span_id
    assert restored.delegation_id == context.delegation_id
    assert restored.w3c_traceparent == context.traceparent()


def test_registered_child_can_emit_real_event() -> None:
    root_context = TraceContextV1("trace", "root-cap", "root", "root-session")
    binding = SimpleNamespace(
        trace_id="trace",
        capture_id="root-cap",
        context_for_child=lambda: root_context,
    )
    received: list[dict[str, object]] = []
    finished: list[dict[str, object]] = []

    class Collector:
        def __init__(self) -> None:
            self.binding = binding

        def event(self, **kwargs: object) -> str:
            received.append(dict(kwargs))
            return "envelope-child"

        def finish_session(self, **kwargs: object) -> tuple[str, str]:
            finished.append(dict(kwargs))
            return "envelope-finished", str(kwargs["ended_at"])

    registrations: list[tuple[TraceContextV1, dict[str, object], dict[str, object]]] = []
    server = CollectorServer(
        Collector(),
        collector_token="secret",
        on_register_context=lambda context, actor, session: registrations.append(
            (context, actor, session)
        ),
    ).start()
    child = TraceContextV1(
        "trace",
        "child-cap",
        "child",
        "child-session",
        parent_actor_id="root",
        delegation_id="delegation",
    )
    try:
        with TraceEmitter(server.base_url, root_context, collector_token="secret") as parent:
            assert (
                parent.register_context(
                    child,
                    actor={"actor_id": "child"},
                    session={"session_id": "child-session", "actor_id": "child"},
                )
                == "child-cap"
            )
        with TraceEmitter(server.base_url, child, collector_token="secret") as emitter:
            assert emitter.event("agent.child", {"ok": True}) == "envelope-child"
            assert (
                emitter.finish(ended_at="2026-07-25T01:02:03Z")
                == "envelope-finished"
            )
            assert emitter.finish() == "envelope-finished"
            with pytest.raises(httpx.HTTPStatusError) as conflicting:
                emitter.finish(status="failed")
            assert conflicting.value.response.status_code == 400
            with pytest.raises(httpx.HTTPStatusError) as late_event:
                emitter.event("agent.child", {"too_late": True})
            assert late_event.value.response.status_code == 409
            with pytest.raises(httpx.HTTPStatusError) as late_artifact:
                emitter.artifact(
                    "output",
                    "text/plain",
                    b"too late",
                    "late.txt",
                )
            assert late_artifact.value.response.status_code == 409
        with TraceEmitter(
            server.base_url,
            child,
            collector_token="wrong",
        ) as unauthorized:
            with pytest.raises(httpx.HTTPStatusError) as unauthorized_finish:
                unauthorized.finish()
            assert unauthorized_finish.value.response.status_code == 403
        with TraceEmitter(
            server.base_url,
            root_context,
            collector_token="secret",
        ) as root:
            with pytest.raises(httpx.HTTPStatusError) as root_finish:
                root.finish()
            assert root_finish.value.response.status_code == 400
    finally:
        server.stop()
    assert received[0]["actor_id"] == "child"
    assert received[0]["session_id"] == "child-session"
    assert registrations[0][0] == child
    assert finished == [
        {
            "status": "completed",
            "actor_id": "child",
            "session_id": "child-session",
            "ended_at": "2026-07-25T01:02:03Z",
        }
    ]


def test_rubric_required_missing_fails_and_ranges_normalize() -> None:
    criterion = CriterionDefinitionV1(
        criterion_id="quality",
        name="quality",
        requirement="quality",
        min_score=0,
        max_score=10,
        pass_threshold=5,
    )
    rubric = RubricDefinitionV2(
        rubric_id="rubric",
        name="rubric",
        task_family="task",
        criteria=(criterion,),
    )
    assert aggregate_rubric_score(rubric, ()) == (
        None,
        False,
        ("missing:quality",),
    )
    score, passed, failures = aggregate_rubric_score(
        rubric,
        (CriterionResultV1("quality", 8, "valid", passed=True),),
    )
    assert score == pytest.approx(0.8)
    assert passed is True
    assert failures == ()
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_rubric_score(
            rubric,
            (
                CriterionResultV1("quality", 8, "valid"),
                CriterionResultV1("quality", 8, "valid"),
            ),
        )


def test_supervisor_secret_descriptor_and_container_paths(tmp_path: Path) -> None:
    config = SupervisorConfig(
        bundle_root=tmp_path / "bundle",
        trace_key={"task": "smoke"},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
        responses_websocket=True,
        container_binding_path="/trace/binding.json",
        container_output_dir="/trace/output",
    )
    with CaptureSupervisor(config) as supervisor:
        secret_env = supervisor.environment("host.docker.internal")
        descriptor = supervisor.environment_descriptor("host.docker.internal")
        assert secret_env["SYNTH_TRACE_COLLECTOR_TOKEN"]
        assert secret_env["SYNTH_TRACE_COLLECTOR_TOKEN"] not in json.dumps(descriptor)
        assert descriptor["collector_token"] == "present"
        assert secret_env["SYNTH_TRACE_BINDING_PATH"] == "/trace/binding.json"
        assert (
            urlsplit(secret_env["OPENAI_RESPONSES_WEBSOCKET_URL"]).path
            == "/v1/responses"
        )


def test_supervisor_routes_responses_websocket_to_configured_upstream(
    tmp_path: Path,
) -> None:
    supervisor = CaptureSupervisor(
        SupervisorConfig(
            bundle_root=tmp_path / "bundle",
            trace_key={"task": "custom-responses-websocket"},
            upstream_base_url="https://chatgpt.com/backend-api/codex",
            provenance=TraceProvenanceV5(producer="test", producer_version="1"),
            responses_websocket=True,
        )
    )

    assert supervisor.websocket_server is not None
    assert (
        supervisor.websocket_server.relay.upstream_url
        == "wss://chatgpt.com/backend-api/codex/responses"
    )


def test_native_import_terminal_timestamp_and_coverage_receipt_are_reachable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "react.json"
    source.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": "step-1",
                        "event_type": "react.step",
                        "occurred_at": "2026-07-25T01:02:03Z",
                        "payload": {"action": "wait"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    bundle = LocalTraceBundle(tmp_path / "native-bundle")

    imported = import_native_to_bundle(
        source,
        source_format="react",
        bundle=bundle,
    )
    trace = bundle.read_trace(imported["trace_digest"])
    manifest = bundle.read_manifest()
    coverage_paths = [
        bundle.root / path
        for path in manifest["receipt_paths"]
        if Path(path).name.startswith("capture-coverage-")
    ]

    assert trace["lifecycle"]["started_at"] == "2026-07-25T01:02:03Z"
    assert trace["lifecycle"]["ended_at"] == "2026-07-25T01:02:03Z"
    assert len(coverage_paths) == 1
    coverage = json.loads(coverage_paths[0].read_text(encoding="utf-8"))
    assert coverage["receipt_id"] == trace["capture"]["coverage_receipt_id"]


def test_responses_websocket_handshake_falls_back_to_http(tmp_path: Path) -> None:
    config = SupervisorConfig(
        bundle_root=tmp_path / "bundle",
        trace_key={"task": "responses-http-fallback"},
        upstream_base_url="https://api.openai.com/v1",
        provenance=TraceProvenanceV5(producer="test", producer_version="1"),
    )
    with CaptureSupervisor(config) as supervisor:
        response = httpx.get(
            f"{supervisor.openai_base_url}/responses",
            headers={
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Version": "13",
            },
            timeout=10.0,
        )
        assert response.status_code == 426
        assert response.headers["upgrade"].lower() == "websocket"
        assert response.headers["sec-websocket-version"] == "13"
        assert response.content == b""
        assert supervisor.proxy.stats.unsupported_routes == []


def test_supervisor_constructs_scoped_mitm_and_requires_egress_proof(
    tmp_path: Path,
) -> None:
    common = {
        "bundle_root": tmp_path / "mitm-bundle",
        "trace_key": {"task": "smoke"},
        "upstream_base_url": "https://api.openai.com/v1",
        "provenance": TraceProvenanceV5(producer="test", producer_version="1"),
    }
    supervisor = CaptureSupervisor(
        SupervisorConfig(**common, interception=Interception.TLS_MITM)
    )
    assert supervisor.mitm is not None
    assert supervisor.mitm.allowed_hosts == ("api.openai.com",)
    supervisor.finalize()
    with pytest.raises(ValueError, match="egress_assertion"):
        CaptureSupervisor(
            SupervisorConfig(
                **{
                    **common,
                    "bundle_root": tmp_path / "egress-bundle",
                },
                mode=CaptureMode.REQUIRED_EGRESS_ASSERTED,
            )
        )


def test_s3_store_uses_conditional_create_and_pointer_cas() -> None:
    class PreconditionFailed(Exception):
        response = {"ResponseMetadata": {"HTTPStatusCode": 412}}

    class Body:
        def __init__(self, value: bytes) -> None:
            self.value = value

        def read(self) -> bytes:
            return self.value

    class Client:
        def __init__(self) -> None:
            self.objects: dict[str, bytes] = {}
            self.metadata: dict[str, dict[str, str]] = {}
            self.content_types: dict[str, str] = {}
            self.calls: list[dict[str, object]] = []

        def put_object(self, **kwargs: object) -> dict[str, str]:
            self.calls.append(dict(kwargs))
            key = str(kwargs["Key"])
            if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
                raise PreconditionFailed
            self.objects[key] = bytes(kwargs["Body"])
            self.metadata[key] = {
                str(name): str(value)
                for name, value in dict(kwargs.get("Metadata") or {}).items()
            }
            self.content_types[key] = str(
                kwargs.get("ContentType") or "application/octet-stream"
            )
            return {"ETag": '"transport-etag"'}

        def get_object(self, **kwargs: object) -> dict[str, Body]:
            return {"Body": Body(self.objects[str(kwargs["Key"])])}

        def head_object(self, **kwargs: object) -> dict[str, object]:
            key = str(kwargs["Key"])
            value = self.objects[key]
            return {
                "ContentLength": len(value),
                "ContentType": self.content_types[key],
                "Metadata": self.metadata[key],
                "ETag": '"transport-etag"',
            }

    client = Client()
    store = S3BlobStore(client, bucket="bucket")
    first = store.put_if_absent(
        b"body",
        media_type="text/plain",
        metadata={"owner": "first"},
    )
    digest = first.digest
    assert store.get(digest) == b"body"
    assert client.calls[0]["IfNoneMatch"] == "*"
    assert first.metadata.etag == '"transport-etag"'
    assert first.metadata.metadata["synth-content-digest"] == digest
    existing = store.put_if_absent(
        b"body",
        media_type="application/json",
        metadata={"owner": "second"},
    )
    assert not existing.created
    assert existing.metadata.media_type == "text/plain"
    assert existing.metadata.metadata["owner"] == "first"
    with pytest.raises(ValueError, match="reserved store metadata"):
        store.put_if_absent(
            b"other",
            metadata={"Synth-Content-Digest": "caller-controlled"},
        )
    pointer_etag = store.compare_and_swap_pointer(
        name="latest",
        content=b"{}",
        expected_etag='"old"',
    )
    assert client.calls[-1]["IfMatch"] == '"old"'
    assert pointer_etag == '"transport-etag"'
    with pytest.raises(ValueError, match="invalid pointer name"):
        store.compare_and_swap_pointer(
            name="../outside",
            content=b"{}",
            expected_etag=None,
        )
    for malformed in (
        "sha1:" + "0" * 64,
        "sha256:" + "0" * 63,
        "sha256:" + "A" * 64,
        "sha256:../../outside",
    ):
        with pytest.raises(ValueError, match="malformed digest"):
            store.has(malformed)


def test_provider_streams_survive_split_utf8_and_capture_usage() -> None:
    responses = OpenAIResponsesAdapter().new_stream()
    body = (
        'event: response.completed\ndata: {"type":"response.completed","response":'
        '{"status":"completed","output":[{"type":"message","role":"assistant",'
        '"content":[{"type":"output_text","text":"hé"}]}],"usage":'
        '{"input_tokens":2,"output_tokens":1}}}\n\n'
    ).encode()
    split = body.index("é".encode()) + 1
    responses.feed(body[:split])
    responses.feed(body[split:])
    normalized = responses.finish()
    assert normalized.terminal_observed
    assert normalized.usage is not None
    assert normalized.usage.prompt_tokens == 2

    anthropic = AnthropicMessagesAdapter().new_stream()
    frames = [
        {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t", "name": "x"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "1}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    stream = "".join(
        f"event: {item['type']}\ndata: {json.dumps(item)}\n\n" for item in frames
    ).encode()
    for index in range(0, len(stream), 7):
        anthropic.feed(stream[index : index + 7])
    result = anthropic.finish()
    assert result.terminal_observed
    assert result.usage is not None
    assert result.usage.prompt_tokens == 3


def test_current_codex_jsonl_kinds_preserve_native_aliases_and_usage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "codex.jsonl"
    source.write_text(
        "\n".join(
            (
                '{"type":"thread.started","thread_id":"thread-1"}',
                '{"type":"turn.started","turn_id":"turn-1"}',
                '{"type":"item.started","item":{"id":"item-1","type":"command_execution","command":"pwd"}}',
                '{"type":"item.completed","item":{"id":"item-1","type":"command_execution","command":"pwd","aggregated_output":"/workspace","exit_code":0,"status":"completed"}}',
                '{"type":"turn.completed","turn_id":"turn-1","usage":{"input_tokens":10,"cached_input_tokens":4,"output_tokens":2}}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    imported = import_codex_jsonl(source, target_id="session-1")
    assert [item["event_type"] for item in imported.events] == [
        "codex.thread_started",
        "codex.turn_started",
        "codex.command_started",
        "codex.command_finished",
        "codex.turn_finished",
    ]
    assert {
        (str(alias.namespace), alias.value)
        for alias in imported.aliases
    } == {
        ("codex.thread", "thread-1"),
        ("codex.turn", "turn-1"),
        ("codex.item", "item-1"),
    }
    assert imported.usage_snapshots == [
        {"input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}
    ]
    assert imported.unknown_kinds == {}
