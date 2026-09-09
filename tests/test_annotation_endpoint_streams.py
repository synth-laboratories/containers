"""Container annotator HTTP catalog and job event streams."""

from __future__ import annotations

from pathlib import Path

import fastapi
from fastapi.testclient import TestClient

from synth_containers.tracing.annotation import (
    ANNOTATION_API_SCHEMA,
    ANNOTATION_EVENT_KINDS,
    AnnotationService,
    AnnotationStore,
    AnnotationWorker,
    DefinitionRegistry,
    annotation_api_catalog,
    build_craftax_smoke_trace,
    register_builtin_annotators,
)
from synth_containers.tracing.annotation.api import build_annotation_router
from synth_containers.tracing.annotation.builtin import ENVIRONMENT_STEP_STATUS_ID


def _service(tmp_path: Path):
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    store = AnnotationStore(tmp_path / "store")
    service = AnnotationService(store=store, registry=registry)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return service, trace


def test_annotation_catalog_names_core_routes_and_stream() -> None:
    catalog = annotation_api_catalog()
    assert catalog["schema"] == ANNOTATION_API_SCHEMA
    assert catalog["rewrites_reward_signal"] is False
    assert catalog["hidden_cot"] is False
    assert set(catalog["request_must_bear"]) == {"model", "reasoning_effort", "runner_kind"}
    paths = {item["path"] for item in catalog["endpoints"]}
    assert "/annotation/catalog" in paths
    assert "/traces/{trace_id}/annotation-jobs" in paths
    assert "/annotation-jobs/{job_id}/stream" in paths
    assert "/annotation-jobs/{job_id}/events" in paths
    streams = {item["name"] for item in catalog["endpoints"] if item.get("stream")}
    assert {"annotation_events", "annotation_stream"} <= streams


def test_annotation_job_poll_and_sse_cover_lifecycle(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    app = fastapi.FastAPI()
    app.include_router(build_annotation_router(service))
    client = TestClient(app)

    catalog = client.get("/annotation/catalog").json()
    assert catalog["schema"] == ANNOTATION_API_SCHEMA
    names = {item["name"] for item in catalog["operations"]}
    assert "annotation_events" in names

    request = service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID).to_dict()
    started = client.post(f"/traces/{trace.trace_id}/annotation-jobs", json={"request": request})
    assert started.status_code == 202, started.text
    body = started.json()
    job_id = body["job"]["job_id"]
    assert body["stream"]["schema"] == "synth.annotation.stream.v1"
    assert body["stream"]["transports"]["poll"]["url"] == f"/annotation-jobs/{job_id}/events"
    prepared = client.get(f"/annotation-jobs/{job_id}/events", params={"after": 0}).json()
    kinds = [row["kind"] for row in prepared["events"]]
    assert "stream.subscribed" in kinds
    assert "annotation.prepared" in kinds
    assert prepared["terminal"] is False

    assert AnnotationWorker(service).run_once() == 1
    done = client.get(f"/annotation-jobs/{job_id}/events", params={"after": 0}).json()
    kinds = [row["kind"] for row in done["events"] if not row.get("control")]
    assert kinds[0] == "annotation.prepared"
    assert "annotation.running" in kinds
    assert "annotation.validating" in kinds
    assert "annotation.finding" in kinds
    assert kinds[-1] == "annotation.sealed"
    assert done["terminal"] is True
    payloads = [row["payload"] for row in done["events"] if row["kind"] == "annotation.sealed"]
    assert payloads[0]["runner_kind"] == "deterministic"
    assert payloads[0]["applied_count"] >= 1
    blob = str(done)
    assert "chain of thought" not in blob.lower()
    assert "hidden_cot" not in blob.lower() or done["stream"]["hidden_cot"] is False

    with client.stream("GET", f"/annotation-jobs/{job_id}/stream") as response:
        text = "".join(response.iter_text())
    assert response.status_code == 200
    assert "event: annotation.sealed" in text
    assert text.count("event: stream.subscribed") == 1


def test_annotation_catalog_lists_every_advertised_event_kind() -> None:
    assert "annotation.tool" in ANNOTATION_EVENT_KINDS
    assert "annotation.cancelled" in ANNOTATION_EVENT_KINDS
