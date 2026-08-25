"""A terminal rollout record must name the trace this process sealed.

Workshop read terminal rollout records during the five-chat Craftax review,
found no trace identity on them, and its own trace index stayed empty while the
container held a complete sealed trace on disk. The seal was reachable, but only
by a consumer that already knew to ask — which is not discoverability.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


BODY = {
    "rollout_id": "seal_reference_1",
    "task_instance_id": "seed:7",
    "policy_ref": {"harness": "react", "config": "luna_med"},
    "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
}


def test_terminal_record_names_the_sealed_trace(tmp_path) -> None:
    app = create_compat_app("craftax_engine", storage_root=tmp_path)
    client = TestClient(app)
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": BODY["rollout_id"], "telemetry": BODY["telemetry"]},
    )
    assert prepared.status_code == 200, prepared.text
    started = client.post("/rollouts", json=BODY)
    assert started.status_code == 200, started.text

    status = client.get(f"/rollouts/{BODY['rollout_id']}").json()
    assert status["terminated"] is True
    reference = status["trace"]
    assert reference is not None, "a terminal rollout must announce its seal"

    seal = client.get(f"/rollouts/{BODY['rollout_id']}/trace").json()
    assert reference["trace_id"] == seal["trace_id"]
    assert reference["content_digest"] == seal["content_digest"]
    assert reference["event_count"] == len(seal["events"])
    assert reference["high_water"] == seal["high_water"]
    assert reference["closed"] is True
    assert reference["url"] == f"/rollouts/{BODY['rollout_id']}/trace"
    assert reference["bundle_url"] == f"/rollouts/{BODY['rollout_id']}/trace/bundle"
    assert reference["kind"] == "trace_v5_bundle"
    assert reference["inspectable"] is True


def test_prepared_but_unstarted_rollout_announces_no_trace(tmp_path) -> None:
    app = create_compat_app("craftax_engine", storage_root=tmp_path)
    client = TestClient(app)
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "seal_reference_2", "telemetry": BODY["telemetry"]},
    )
    assert prepared.status_code == 200, prepared.text
    status = client.get("/rollouts/seal_reference_2").json()
    # A prepared rollout has sealed nothing; claiming otherwise would be worse
    # than saying nothing at all.
    assert status.get("trace") is None
