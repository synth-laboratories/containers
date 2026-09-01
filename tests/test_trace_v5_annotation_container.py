"""In-container mount: sealed bundles become annotatable without any live provider."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.tracing.annotation import AnnotationJobState, ThroughputLimits
from synth_containers.tracing.annotation.builtin import ENVIRONMENT_STEP_STATUS_ID, TOOL_CALL_INTEGRITY_ID
from synth_containers.tracing.annotation.container import install_from_env, mount_annotation

BODY = {
    "rollout_id": "annot_echo_1",
    "task_instance_id": "seed:7",
    "policy_ref": {"harness": "gym_loop", "config": "echo"},
    "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
}


def _rollout(client: TestClient, rollout_id: str) -> dict:
    body = {**BODY, "rollout_id": rollout_id}
    assert client.post("/rollouts/prepare", json={"rollout_id": rollout_id, "telemetry": body["telemetry"]}).status_code == 200
    assert client.post("/rollouts", json=body).status_code == 200
    status = client.get(f"/rollouts/{rollout_id}").json()
    assert status["terminated"] is True and status["trace"] is not None
    return status


def test_container_mount_annotates_sealed_bundles_and_serves_evidence(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    mounted = mount_annotation(app, storage_root=tmp_path, limits=ThroughputLimits(max_concurrent_total=2, poll_seconds=0.01), post_rollout=(TOOL_CALL_INTEGRITY_ID,), start=False)
    client = TestClient(app)
    status = _rollout(client, "annot_echo_1")
    assert "bundle_url" in status["trace"], status["trace"]
    refs = client.get("/annotation/traces").json()["traces"]
    assert len(refs) == 1 and refs[0]["kind"] == "trace_v5" and refs[0]["digest"].startswith("sha256:")
    trace_id, digest = refs[0]["id"], refs[0]["digest"]
    # the service resolves the sealed document straight from the container's bundle
    document = mounted.service.resolve_trace(trace_id, digest)
    assert document.content_digest == digest and document.trace_id == trace_id
    # explicit job over HTTP: 202, queued, then run by the scheduler
    definitions = client.get(f"/traces/{trace_id}/annotation-definitions").json()["annotators"]
    assert {ENVIRONMENT_STEP_STATUS_ID, TOOL_CALL_INTEGRITY_ID} <= {item["annotator_id"] for item in definitions}
    request = mounted.service.request_for(document, ENVIRONMENT_STEP_STATUS_ID).to_dict()
    accepted = client.post(f"/traces/{trace_id}/annotation-jobs", json={"request": request})
    assert accepted.status_code == 202 and accepted.json()["queue_position"] == 1
    job_id = accepted.json()["job"]["job_id"]
    assert mounted.scheduler.drain(timeout=30)
    job = client.get(f"/annotation-jobs/{job_id}").json()
    assert job["job"]["state"] in {AnnotationJobState.SEALED.value, AnnotationJobState.ABSTAINED.value}, job["job"].get("error")
    bundles = client.get(f"/traces/{trace_id}/evidence-bundles").json()["bundles"]
    assert bundles and bundles[-1]["trace_digest"] == digest
    head = client.get(f"/traces/{trace_id}/evidence-head").json()["head"]
    assert head["bundle_digest"] == bundles[-1]["bundle_digest"]
    assert "verifier_results" in head
    assert isinstance(head["verifier_results"], list)
    one = client.get(f"/traces/{trace_id}/evidence-bundles/{head['bundle_digest']}").json()["bundle"]
    assert one["bundle_digest"] == head["bundle_digest"]
    # post-rollout stage: the watcher submits the configured annotator for every new seal
    run = mounted.watcher.poll_once()
    assert run is not None and run.enqueued + run.cache_hits == 1 and not run.refused
    assert mounted.watcher.poll_once() is None  # nothing new
    _rollout(client, "annot_echo_2")
    second = mounted.watcher.poll_once()
    assert second is not None and len(second.plan.traces) == 1 and second.plan.traces[0][0] != trace_id
    assert mounted.scheduler.drain(timeout=30)
    for job_id in run.job_ids + second.job_ids:
        assert mounted.service.get(job_id).terminal
    status_payload = client.get("/annotation/status").json()
    assert status_payload["post_rollout"] == [TOOL_CALL_INTEGRITY_ID] and status_payload["post_rollout_runs"] == 2
    assert status_payload["broker"] == "DenyAllBroker"
    campaign = client.post("/annotation/campaigns", json={"annotators": [ENVIRONMENT_STEP_STATUS_ID, TOOL_CALL_INTEGRITY_ID], "session_id": "s"}).json()
    assert len(campaign["jobs"]) == 4 and campaign["cache_hits"] >= 3 and not campaign["refused"]


def test_install_from_env_is_gated_and_fail_soft(tmp_path: Path, monkeypatch) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    monkeypatch.setenv("SYNTH_ANNOTATION", "off")
    assert install_from_env(app, storage_root=tmp_path) is None
    monkeypatch.setenv("SYNTH_ANNOTATION", "on")
    monkeypatch.delenv("SYNTH_ANNOTATION_DOMAINS", raising=False)
    monkeypatch.setenv("SYNTH_ANNOTATION_POST_ROLLOUT", f"{TOOL_CALL_INTEGRITY_ID},does.not.exist")
    mounted = install_from_env(app, storage_root=tmp_path)
    assert mounted is not None and [p.annotator_id for p in mounted.watcher.annotators] == [TOOL_CALL_INTEGRITY_ID]
    with TestClient(app) as client:  # startup/shutdown events run the scheduler and watcher
        status = client.get("/annotation/status").json()
        assert status["post_rollout"] == [TOOL_CALL_INTEGRITY_ID]
        assert status["api"] == "synth.container.annotation-api.v1"
        info = client.get("/info").json()
        assert info["annotation_api"]["mounted"] is True
        assert info["annotation_api"]["catalog"] == "/annotation/catalog"
        assert info["annotation_api"]["rewrites_reward_signal"] is False
        catalog = client.get("/annotation/catalog").json()
        assert catalog["schema"] == "synth.container.annotation-api.v1"
        assert "/annotation-jobs/{job_id}/stream" in {item["path"] for item in catalog["endpoints"]}
        health = client.get("/health").json()
        assert health["annotation"] == "mounted"


def test_install_from_env_declared_registrar_fails_closed(tmp_path: Path, monkeypatch) -> None:
    from synth_containers.tracing.annotation.container import RegistrarLoadError

    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    monkeypatch.setenv("SYNTH_ANNOTATION", "on")
    monkeypatch.setenv("SYNTH_ANNOTATION_DOMAINS", "nope.module:register")
    with pytest.raises(RegistrarLoadError, match="nope.module:register"):
        install_from_env(app, storage_root=tmp_path)
