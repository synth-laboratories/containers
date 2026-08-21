"""TS-C transport equivalence. Producer kit; C1 remains the full floor suite."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def _parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for block in body.split("\n\n"):
        data_lines = [
            line[len("data: ") :]
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        if not data_lines:
            continue
        payload = json.loads("\n".join(data_lines))
        if isinstance(payload, dict) and payload.get("kind") != "heartbeat":
            events.append(payload)
    return events


def test_ts_c01_poll_and_sse_same_ids(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p0"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    stream = started.json()["stream"]
    poll = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    with client.stream("GET", stream["transports"]["sse"]["url"]) as response:
        text = "".join(response.iter_text())
    sse = _parse_sse(text)
    poll_ids = [(item.get("sequence"), item.get("digest")) for item in poll if not item.get("control")]
    sse_ids = [(item.get("sequence"), item.get("digest")) for item in sse if not item.get("control")]
    assert poll_ids
    assert poll_ids == sse_ids


def test_ts_c04_subscribed_is_control_and_does_not_advance_sequence(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p1"))
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "roll_ts_c04", "telemetry": {"enabled": True, "transport": "sse"}},
    )
    assert prepared.status_code == 200
    before = client.get("/rollouts/roll_ts_c04/events", params={"after": 0}).json()["events"]
    assert any(item.get("kind") == "stream.subscribed" and item.get("control") for item in before)
    assert all(item.get("sequence") is None for item in before if item.get("control"))


def test_ts_c08_absent_transports_are_null_when_unbound(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p2"))
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "roll_ts_c08", "telemetry": {"enabled": True, "transport": "poll"}},
    )
    stream = prepared.json()["stream"]
    assert stream["transports"]["poll"]["url"]
    assert stream["transports"]["sse"] is None
    assert stream["transports"]["websocket"] is None
