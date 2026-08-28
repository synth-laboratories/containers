"""C-1: async acceptance receipt. Background progresses without /complete."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.http_requests import RequestParseError, parse_create_rollout

_POLICY = {"harness": "react", "config": "luna_med"}
_TELEMETRY = {"enabled": True, "transport": "sse"}


def _wait_terminal(client: TestClient, rollout_id: str, *, timeout_s: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/rollouts/{rollout_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last.get("terminated") is True:
            return last
        time.sleep(0.05)
    raise AssertionError(f"rollout {rollout_id} did not reach terminal: {last}")


def test_parse_async_defaults_to_background() -> None:
    req = parse_create_rollout(
        {
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
        }
    )
    assert req.execution == "background"
    sync = parse_create_rollout(
        {
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "sync",
        }
    )
    assert sync.execution is None
    with pytest.raises(RequestParseError, match="background or on_complete"):
        parse_create_rollout(
            {
                "telemetry": _TELEMETRY,
                "policy_ref": _POLICY,
                "submission_mode": "async",
                "execution": "deferred",
            }
        )


def test_async_start_returns_202_acceptance_receipt(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p0"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c1_receipt",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
            "idempotency_key": "idem-c1-receipt",
            "task_instance_id": "seed:0",
        },
    )
    assert started.status_code == 202, started.text
    body = started.json()
    assert body["accepted"] is True
    assert body["rollout_id"] == "roll_c1_receipt"
    assert body["execution"] == "background"
    assert body["idempotency_key"] == "idem-c1-receipt"
    assert "id" in body["stream"]
    assert body["lease"]["scale_leases"] == 10
    assert body["contract"]["config_digest"].startswith("sha256:")
    assert body["contract"]["capability_digest"].startswith("sha256:")
    _wait_terminal(client, "roll_c1_receipt")


def test_background_reaches_terminal_without_complete(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p1"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c1_bg",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
            "task_instance_id": "seed:1",
        },
    )
    assert started.status_code == 202, started.text
    assert started.json()["execution"] == "background"
    status = _wait_terminal(client, "roll_c1_bg")
    assert status["status"] == "completed"
    assert client.post("/rollouts/roll_c1_bg/complete").status_code == 200


def test_on_complete_holds_until_complete(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p2"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c1_hold",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
            "execution": "on_complete",
            "task_instance_id": "seed:2",
        },
    )
    assert started.status_code == 202, started.text
    assert started.json()["execution"] == "on_complete"
    held = client.get("/rollouts/roll_c1_hold")
    assert held.status_code == 200
    assert held.json()["terminated"] is False
    assert held.json()["status"] == "running"
    completed = client.post("/rollouts/roll_c1_hold/complete")
    assert completed.status_code == 200, completed.text
    assert completed.json()["terminated"] is True


def test_same_id_retry_returns_same_record_and_spawns_nothing(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p3"))
    payload = {
        "rollout_id": "roll_c1_idem",
        "telemetry": _TELEMETRY,
        "policy_ref": _POLICY,
        "submission_mode": "async",
        "idempotency_key": "idem-c1",
        "task_instance_id": "seed:3",
    }
    first = client.post("/rollouts", json=payload)
    assert first.status_code == 202, first.text
    platform = client.app.state.platform
    spawned = list(platform._background_threads)
    retry = client.post("/rollouts", json=payload)
    assert retry.status_code == 202, retry.text
    body = retry.json()
    assert body["accepted"] is True
    assert body["replayed"] is True
    assert body["rollout_id"] == "roll_c1_idem"
    assert body["idempotency_key"] == first.json()["idempotency_key"]
    assert body["contract"] == first.json()["contract"]
    assert list(platform._background_threads) == spawned
    _wait_terminal(client, "roll_c1_idem")
    again = client.post("/rollouts", json=payload)
    assert again.status_code == 202
    assert again.json()["replayed"] is True
    assert list(platform._background_threads) == spawned
