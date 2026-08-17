"""C-4: stream backpressure — 5s heartbeats, per-rollout cap, reconnect, grace close."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from synth_containers.event_log import (
    MAX_STREAMS_PER_ROLLOUT,
    STREAM_HEARTBEAT_INTERVAL_S,
    STREAM_TERMINAL_GRACE_S,
)
from synth_containers.platform import create_compat_app

_POLICY = {"harness": "react", "config": "luna_med"}
_TELEMETRY = {"enabled": True, "transport": "sse"}


def test_heartbeat_interval_is_at_least_five_seconds() -> None:
    assert STREAM_HEARTBEAT_INTERVAL_S >= 5.0
    assert STREAM_TERMINAL_GRACE_S >= 5.0


def test_stream_descriptor_advertises_reconnect() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c4_reconnect",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
            "execution": "on_complete",
            "task_instance_id": "seed:0",
        },
    )
    assert started.status_code == 202, started.text
    reconnect = started.json()["stream"]["reconnect"]
    assert reconnect["min_backoff_s"] <= reconnect["max_backoff_s"]
    assert 0 <= reconnect["jitter"] <= 1
    client.post("/rollouts/roll_c4_reconnect/complete")


def test_stream_cap_returns_429_with_retry_after() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c4_cap",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
            "execution": "on_complete",
            "task_instance_id": "seed:1",
        },
    )
    assert started.status_code == 202, started.text
    platform = client.app.state.platform
    for _ in range(MAX_STREAMS_PER_ROLLOUT):
        assert platform.acquire_stream("roll_c4_cap") is None
    response = client.get("/rollouts/roll_c4_cap/stream")
    assert response.status_code == 429, response.text
    assert response.headers["retry-after"] == "5"
    body = response.json()
    assert body["error"] == "stream_backpressure"
    assert body["max_streams"] == MAX_STREAMS_PER_ROLLOUT
    for _ in range(MAX_STREAMS_PER_ROLLOUT):
        platform.release_stream("roll_c4_cap")
    client.post("/rollouts/roll_c4_cap/complete")


def test_terminal_unclosed_stream_closes_after_grace(monkeypatch) -> None:
    monkeypatch.setattr(
        "synth_containers.platform.app.STREAM_HEARTBEAT_INTERVAL_S",
        0.05,
    )
    monkeypatch.setattr(
        "synth_containers.platform.app.STREAM_TERMINAL_GRACE_S",
        0.05,
    )
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c4_grace",
            "telemetry": _TELEMETRY,
            "policy_ref": _POLICY,
            "submission_mode": "async",
            "execution": "on_complete",
            "task_instance_id": "seed:2",
        },
    )
    assert started.status_code == 202, started.text
    platform = client.app.state.platform
    pin = platform.pins["roll_c4_grace"]
    pin.terminal = True
    pin.status = "completed"
    started_at = time.monotonic()
    with client.stream("GET", "/rollouts/roll_c4_grace/stream") as response:
        assert response.status_code == 200
        payload = b"".join(response.iter_bytes())
    elapsed = time.monotonic() - started_at
    assert elapsed < 2.0
    assert b"stream.subscribed" in payload or b"trace.opened" in payload
    assert platform._stream_occupancy.get("roll_c4_grace", 0) == 0
