"""Durability and connect-before-start checks for the §12 façade journal."""

from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_containers.event_log import RolloutEventLog
from synth_containers.platform import create_compat_app
from synth_containers.platform.craftax_world import StepResult
from synth_containers.platform.episode import _emit_obs
from synth_containers.tracing.capture.redaction import RedactionError

_CRAFTAX_PIN = {"harness": "react", "config": "luna_med"}
_HARBOR_PIN = {"harness": "harbor_fused", "config": "luna_med"}


def test_compat_journal_persists_before_poll_and_recovers(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    rollout_id = "persist-before-publish"
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": rollout_id,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    journal = next((tmp_path / "event_logs").glob("*.jsonl"))
    control_rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert control_rows[0]["envelope"]["kind"] == "stream.subscribed"

    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "slot": "stream",
            "task_instance_id": "seed:0",
            "policy_ref": _CRAFTAX_PIN,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert started.status_code == 200, started.text
    published = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()["events"]

    recovered = RolloutEventLog.recover(
        rollout_id=rollout_id,
        stream_id=stream["id"],
        journal_path=journal,
    )
    assert recovered.closed is True
    assert recovered.high_water == max(
        row["sequence"] for row in published if row.get("sequence") is not None
    )
    assert [row.to_dict() for row in recovered.after(0)] == [
        {key: value for key, value in row.items() if key != "rollout_id"}
        for row in published
    ]
    seal = json.loads((tmp_path / "seals" / f"{rollout_id}.trace-v5.json").read_text())
    assert seal["schema_version"] == "synth.trace.v5"
    assert seal["rollout_id"] == rollout_id
    assert seal["closed"] is True
    assert seal["content_digest"].startswith("sha256:")
    timestamps = [row["ts"] for row in published if not row.get("control")]
    assert all("." in timestamp for timestamp in timestamps)


def test_poll_small_pages_reconstruct_exact_evidence_without_advancing_on_control(
    tmp_path: Path,
) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    rollout_id = "small-page-replay"
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": rollout_id,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    ).json()
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "slot": "stream",
            "task_instance_id": "seed:11",
            "policy_ref": _CRAFTAX_PIN,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert started.status_code == 200, started.text
    poll_url = prepared["stream"]["transports"]["poll"]["url"]
    complete = client.get(poll_url, params={"after": 0}).json()
    expected = [row for row in complete["events"] if row.get("sequence") is not None]

    cursor = 0
    reconstructed: list[dict] = []
    saw_control = False
    while True:
        page = client.get(poll_url, params={"after": cursor, "limit": 2}).json()
        evidence = [row for row in page["events"] if row.get("sequence") is not None]
        saw_control |= any(row.get("kind") == "stream.subscribed" for row in page["events"])
        reconstructed.extend(evidence)
        next_cursor = page["cursor"]["next"]
        assert next_cursor >= cursor
        if not page["cursor"]["has_more"]:
            assert next_cursor == page["cursor"]["high_water"]
            assert page["cursor"]["closed"] is True
            break
        assert next_cursor > cursor
        cursor = next_cursor

    assert saw_control is True
    assert [(row["event_id"], row["digest"]) for row in reconstructed] == [
        (row["event_id"], row["digest"]) for row in expected
    ]


def test_failed_persist_never_advances_or_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = RolloutEventLog(
        rollout_id="fail-closed",
        stream_id="stream:fail-closed",
        journal_path=tmp_path / "events.jsonl",
    )

    def fail(_: dict) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(log, "_persist", fail)
    with pytest.raises(OSError, match="disk full"):
        log.append("observation", {"text": "not visible"})
    assert log.high_water == 0
    assert log.after(0) == []


def test_secret_is_refused_before_journal_or_publication(tmp_path: Path) -> None:
    journal = tmp_path / "events.jsonl"
    log = RolloutEventLog(
        rollout_id="redacted",
        stream_id="stream:redacted",
        journal_path=journal,
    )
    with pytest.raises(RedactionError):
        log.append("tools", {"Authorization": "Bearer should-not-persist"})
    assert not journal.exists()
    assert log.high_water == 0
    assert log.after(0) == []


def test_published_payload_cannot_drift_after_persist(tmp_path: Path) -> None:
    journal = tmp_path / "events.jsonl"
    log = RolloutEventLog(
        rollout_id="immutable",
        stream_id="stream:immutable",
        journal_path=journal,
    )
    payload = {"nested": {"value": 1}}
    envelope = log.append("observation", payload)
    payload["nested"]["value"] = 999
    first_view = envelope.to_dict()
    first_view["payload"]["nested"]["value"] = 888
    published = log.after(0)[0].to_dict()
    persisted = json.loads(journal.read_text().strip())["envelope"]
    assert published["payload"] == persisted["payload"] == {"nested": {"value": 1}}


def test_png_frame_is_fsynced_before_available_event_and_served_after_restart(
    tmp_path: Path,
) -> None:
    rollout_id = "durable-frame"
    journal = tmp_path / "event_logs" / "events.jsonl"
    log = RolloutEventLog(
        rollout_id=rollout_id,
        stream_id=f"stream:{rollout_id}",
        journal_path=journal,
    )
    png = b"\x89PNG\r\n\x1a\n" + b"durable-test-frame"
    _emit_obs(
        log,
        StepResult(
            observation={"env_steps": 3},
            reward=0.0,
            done=False,
            valid_actions=["noop"],
            ascii_map="P",
            frame_digest="frame-digest",
            env_steps=3,
            frame_url="http://transient.invalid/current.png",
            frame_bytes=png,
        ),
        seed=7,
    )

    available = [row for row in log.after(0) if row.kind == "artifact.available"]
    assert available[0].payload["url"] == f"/rollouts/{rollout_id}/frames/3.png"
    frame_path = RolloutEventLog.frame_asset_path(tmp_path, rollout_id, 3)
    assert frame_path.read_bytes() == png
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    response = client.get(available[0].payload["url"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == png


def test_missing_or_invalid_png_stays_ascii_and_never_claims_available(tmp_path: Path) -> None:
    log = RolloutEventLog(
        rollout_id="no-fake-frame",
        stream_id="stream:no-fake-frame",
        journal_path=tmp_path / "event_logs" / "events.jsonl",
    )
    _emit_obs(
        log,
        StepResult(
            observation={},
            reward=None,
            done=False,
            valid_actions=[],
            ascii_map="P",
            frame_digest="ascii-only",
            env_steps=0,
            frame_url="http://transient.invalid/current.png",
            frame_bytes=b"not a png",
        ),
        seed=0,
    )
    frames = [row for row in log.after(0) if row.kind == "frame"]
    assert frames[0].payload["format"] == "ascii"
    assert "url" not in frames[0].payload
    assert not any(row.kind == "artifact.available" for row in log.after(0))


def test_closed_log_refuses_control_and_semantic_records(tmp_path: Path) -> None:
    log = RolloutEventLog(
        rollout_id="closed",
        stream_id="stream:closed",
        journal_path=tmp_path / "events.jsonl",
    )
    log.mark_closed()
    with pytest.raises(RuntimeError, match="event_log_closed"):
        log.append_control("stream.subscribed", {"ready": True})
    with pytest.raises(RuntimeError, match="event_log_closed"):
        log.append("status", {"status": "completed"})


def test_concurrent_appends_are_gap_free_and_recoverable(tmp_path: Path) -> None:
    journal = tmp_path / "events.jsonl"
    log = RolloutEventLog(
        rollout_id="concurrent",
        stream_id="stream:concurrent",
        journal_path=journal,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda value: log.append("observation", {"value": value}), range(64)))

    recovered = RolloutEventLog.recover(
        rollout_id=log.rollout_id,
        stream_id=log.stream_id,
        journal_path=journal,
    )
    assert recovered.high_water == 64
    assert [item.sequence for item in recovered.after(0)] == list(range(1, 65))


def test_concurrent_identical_start_executes_once(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    payload = {
        "rollout_id": "same-start",
        "slot": "stream",
        "submission_mode": "async",
        "task_instance_id": "seed:0",
        "policy_ref": _CRAFTAX_PIN,
        "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: client.post("/rollouts", json=payload), range(2)))

    assert [response.status_code for response in responses] == [200, 200]
    assert sum(bool(response.json().get("replayed")) for response in responses) == 1
    journal = next((tmp_path / "event_logs").glob("*.jsonl"))
    recovered = RolloutEventLog.recover(
        rollout_id="same-start",
        stream_id="stream:same-start",
        journal_path=journal,
    )
    assert sum(item.kind == "trace.opened" for item in recovered.after(0)) == 1


def test_concurrent_distinct_starts_enforce_lease_limit_and_isolate_logs(
    tmp_path: Path,
) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))

    def start(index: int):
        return client.post(
            "/rollouts",
            json={
                "rollout_id": f"lease-{index}",
                "slot": "stream",
                "submission_mode": "async",
                "task_instance_id": f"seed:{index}",
                "policy_ref": _CRAFTAX_PIN,
                "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            },
        )

    with ThreadPoolExecutor(max_workers=11) as executor:
        responses = list(executor.map(start, range(11)))

    assert sum(response.status_code == 200 for response in responses) == 10
    assert sum(response.status_code == 429 for response in responses) == 1
    successful_ids = {
        response.json()["rollout_id"] for response in responses if response.status_code == 200
    }
    assert len(successful_ids) == 10
    for rollout_id in successful_ids:
        journal_name = hashlib.sha256(rollout_id.encode()).hexdigest()
        recovered = RolloutEventLog.recover(
            rollout_id=rollout_id,
            stream_id=f"stream:{rollout_id}",
            journal_path=tmp_path / "event_logs" / f"{journal_name}.jsonl",
        )
        assert sum(item.kind == "trace.opened" for item in recovered.after(0)) == 1


def test_prepare_restart_recovers_open_and_refuses_sealed_or_corrupt(tmp_path: Path) -> None:
    payload = {
        "rollout_id": "restart-open",
        "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
    }
    first = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    assert first.post("/rollouts/prepare", json=payload).status_code == 200
    restarted = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    recovered = restarted.post("/rollouts/prepare", json=payload)
    assert recovered.status_code == 200, recovered.text

    started = restarted.post(
        "/rollouts",
        json={
            **payload,
            "slot": "stream",
            "policy_ref": _CRAFTAX_PIN,
        },
    )
    assert started.status_code == 200, started.text
    sealed_restart = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    sealed = sealed_restart.post("/rollouts/prepare", json=payload)
    assert sealed.status_code == 409
    assert "event_log_sealed" in sealed.text

    corrupt_payload = {**payload, "rollout_id": "restart-corrupt"}
    assert first.post("/rollouts/prepare", json=corrupt_payload).status_code == 200
    corrupt_name = hashlib.sha256(b"restart-corrupt").hexdigest()
    corrupt_journal = tmp_path / "event_logs" / f"{corrupt_name}.jsonl"
    corrupt_journal.write_text(corrupt_journal.read_text() + "{not-json}\n")
    corrupt_restart = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    corrupt = corrupt_restart.post("/rollouts/prepare", json=corrupt_payload)
    assert corrupt.status_code == 409
    assert "event_log_unrecoverable:event_log_malformed_json_line" in corrupt.text


def test_prepare_requires_enabled_telemetry(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    refused = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "disabled", "telemetry": {"enabled": False, "transport": "poll"}},
    )
    assert refused.status_code == 400
    assert not (tmp_path / "event_logs").exists()


def test_rollout_id_cannot_escape_event_store_or_break_declared_urls(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    refused = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": "../../escape",
            "telemetry": {"enabled": True, "transport": "sse"},
        },
    )
    assert refused.status_code == 422
    assert not (tmp_path / "event_logs").exists()


def test_prepare_binding_is_stable_and_unadvertised_transports_refuse(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": "poll-only",
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
        },
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    assert stream["transports"]["sse"] is None
    assert client.get("/rollouts/poll-only/stream").status_code == 404

    mismatch = client.post(
        "/rollouts",
        json={
            "rollout_id": "poll-only",
            "slot": "stream",
            "policy_ref": _CRAFTAX_PIN,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )
    assert mismatch.status_code == 409
    assert mismatch.json()["error"] == "stream_binding_mismatch"

    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "poll-only",
            "slot": "stream",
            "policy_ref": _CRAFTAX_PIN,
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
        },
    )
    assert started.status_code == 200, started.text
    assert started.json()["stream"] == stream
    duplicate = client.post(
        "/rollouts",
        json={
            "rollout_id": "poll-only",
            "slot": "stream",
            "policy_ref": _CRAFTAX_PIN,
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
        },
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["replayed"] is True
    assert duplicate.json()["rollout_id"] == "poll-only"


def test_nested_harbor_child_uses_resource_ref_shape(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("deo_nested", storage_root=tmp_path))
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "nested", "telemetry": {"enabled": True, "transport": "sse"}},
    ).json()
    assert any(
        row["kind"] == "stream.subscribed"
        for row in client.get(prepared["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    )
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "nested",
            "slot": "stream",
            "policy_ref": _HARBOR_PIN,
            "telemetry": {"enabled": True, "transport": "sse"},
        },
    )
    assert started.status_code == 200, started.text
    ref = started.json()["child_resource_ref"]
    assert ref == {
        "schema": "synth.resource-ref.v1",
        "kind": "container_rollout",
        "id": "nested:child",
        "attributes": {
            "stream_id": "stream:nested:child",
            "reward_url": "/rollouts/nested:child/reward",
        },
    }
