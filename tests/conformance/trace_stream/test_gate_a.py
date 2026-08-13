"""TS-A schema and lifecycle against the Containers façade."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_containers.event_log import RolloutEventLog
from synth_containers.platform import create_compat_app
from synth_containers.tracing.capture.redaction import RedactionError, assert_no_secrets
from synth_containers.tracing.streaming.lifecycle import (
    capture_closed_count,
    first_semantic_kind,
    lifecycle_violations,
    nested_span_violations,
    unknown_namespaced_kinds,
)

_SCHEMA_ROOT = Path(__file__).resolve().parents[3] / "schemas" / "trace-stream"


def _engine_events() -> tuple[dict, list[dict]]:
    client = TestClient(create_compat_app("craftax_engine"))
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "roll_ts_a", "telemetry": {"enabled": True, "transport": "sse"}},
    )
    assert prepared.status_code == 200, prepared.text
    descriptor = prepared.json()["stream"]
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_ts_a",
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get("/rollouts/roll_ts_a/events", params={"after": 0}).json()["events"]
    return descriptor, events


def test_ts_a01_discovery_descriptor() -> None:
    descriptor, _ = _engine_events()
    schema = json.loads((_SCHEMA_ROOT / "stream-descriptor.schema.json").read_text())
    assert descriptor["schema"] == schema["properties"]["schema"]["const"]
    assert descriptor["id"]
    assert descriptor["transports"]["poll"]["url"].endswith("/events")
    assert descriptor["transports"]["sse"]["url"].endswith("/stream")
    assert descriptor["cursor"]["kind"] == "sequence"
    assert descriptor["reward"]["url"]
    assert "poll_url" not in descriptor


def test_ts_a02_trace_opened_then_one_capture_closed() -> None:
    _, events = _engine_events()
    assert first_semantic_kind(events) == "trace.opened"
    assert capture_closed_count(events) == 1
    assert lifecycle_violations(events) == []


def test_ts_a03_nested_spans_parent_before_child() -> None:
    _, events = _engine_events()
    assert nested_span_violations(events) == []


def test_ts_a04_discriminated_payloads() -> None:
    _, events = _engine_events()
    envelope_schema = json.loads((_SCHEMA_ROOT / "envelope.schema.json").read_text())
    required = envelope_schema["required"]
    kinds = {item["kind"]: item for item in events if not item.get("control")}
    for kind in ("reward_signal", "frame", "action", "observation", "status"):
        row = kinds[kind]
        for key in required:
            assert key in row
        assert isinstance(row["payload"], dict)
    assert isinstance(kinds["reward_signal"]["payload"].get("value"), (int, float))
    assert kinds["frame"]["payload"].get("digest")


def test_ts_a05_unknown_namespaced_kinds_survive() -> None:
    log = RolloutEventLog(rollout_id="roll_ns", stream_id="stream_ns")
    log.append("trace.opened", {"rollout_id": "roll_ns"})
    log.append("x.craftax.nev.custom", {"note": "namespaced"})
    log.append("capture.closed", {"high_water": 2})
    events = [item.to_dict() for item in log.after(0)]
    assert unknown_namespaced_kinds(events) == ["x.craftax.nev.custom"]
    assert events[1]["kind"] == "x.craftax.nev.custom"


def test_ts_a06_lifecycle_regressions_fail() -> None:
    log = RolloutEventLog(rollout_id="roll_bad", stream_id="stream_bad")
    log.append("span.policy.closed", {})
    log.append("trace.opened", {})
    log.append("span.policy.opened", {})
    log.append("span.policy.data", {"text": "after close attempt"})
    events = [item.to_dict() for item in log.after(0)]
    issues = lifecycle_violations(events)
    assert "first_semantic_not_trace.opened" in issues
    assert "orphan_close:span.policy.closed" in issues


def test_ts_a07_missing_stays_missing() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "omit_reward": True,
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    missing = [
        item
        for item in events
        if item.get("kind") == "reward_signal" and item["payload"].get("value") is None
    ]
    assert missing
    assert all(item["payload"].get("value") != 0 for item in missing)


def test_ts_a08_secrets_rejected_before_persist() -> None:
    with pytest.raises(RedactionError):
        assert_no_secrets(
            {"Authorization": "Bearer sk_live_this_must_not_persist_12"},
            where="ts-a08",
        )
    _, events = _engine_events()
    blob = json.dumps(events)
    assert "Bearer " not in blob
    assert "sk_live_" not in blob
    assert "Authorization" not in blob
