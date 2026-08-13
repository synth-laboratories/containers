"""TS-D close, seal, and reconciliation against the façade."""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def test_ts_d02_d03_closed_high_water_matches_seal() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()
    semantic = [item for item in events["events"] if not item.get("control")]
    assert any(item["kind"] == "capture.closed" for item in semantic)
    high_water = max(item["sequence"] for item in semantic)
    later = [item for item in semantic if item["sequence"] > high_water]
    assert later == []
    seal = client.get(f"/rollouts/{rid}/trace").json()
    assert seal["closed"] is True
    assert seal["high_water"] == high_water
    assert seal["content_digest"]
    assert seal["trace_id"] == rid


def test_ts_d07_failed_rollout_is_not_a_trusted_success(monkeypatch) -> None:
    monkeypatch.delenv("DIGBENCH_API_TOKEN", raising=False)
    client = TestClient(create_compat_app("digbench_public"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "failed"
    rid = body["rollout_id"]
    seal = client.get(f"/rollouts/{rid}/trace").json()
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"]["status"] == "failed"
    assert seal["trace_id"] == rid
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
    assert scored.json().get("reward") is None
