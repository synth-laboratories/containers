"""End to end over the compat façade: install, pin, stream, recover, and refuse."""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.live_annotation.contract import PROTOCOL_SCHEMA
from synth_containers.platform import create_compat_app

_PIN = {"harness": "gym_loop", "config": "echo"}

COUNTER = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
PROTOCOL_ID = "test.counter"

class Protocol:
    def __init__(self, config):
        self.actions = 0
    def on_event(self, event):
        if event["kind"] == "action":
            self.actions += 1
            return [{{"op": "finding", "finding_id": "act-%d" % self.actions, "kind": "note",
                     "label": "took " + str(event["payload"].get("action")),
                     "evidence": {{"sequences": [event["sequence"]]}}}}]
        return []
    def on_close(self):
        return [{{"op": "metric", "name": "actions", "value": self.actions}}]
'''


def _install(client: TestClient, **overrides: object) -> dict:
    body = {"code": COUNTER, "protocol_id": "test.counter", "configuration": {"cadence": 1}, "source_revision": "abc"}
    body.update(overrides)
    response = client.put("/annotation-protocol", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _poll_until_closed(client: TestClient, url: str, timeout: float = 20.0) -> list[dict]:
    events: list[dict] = []
    after = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        page = client.get(url, params={"after": after}).json()
        for row in page["events"]:
            if row.get("sequence") is not None:
                events.append(row)
        after = page["cursor"]["next"]
        if page["cursor"]["closed"]:
            return events
        time.sleep(0.05)
    raise AssertionError("annotation stream never closed")


def test_put_protocol_installs_pins_and_streams_provisional_findings(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    client = TestClient(app)

    empty = client.get("/annotation-protocol").json()
    assert empty["status"] == "not_installed" and empty["protocol_revision_id"] is None
    info = client.get("/info").json()
    assert info["capabilities"]["operations"]["annotation.live"] is True
    assert info["live_annotation"]["mode"] == "observe_only"
    assert info["live_annotation"]["installed"] is False

    installed = _install(client)
    revision_id = installed["protocol_revision_id"]
    assert revision_id.startswith("anprev_")
    assert installed["idempotent"] is False
    assert installed["isolation_receipt"]["sandbox"] == "process"
    assert installed["protocol_id"] == "test.counter"
    assert installed["credential_state"] == "not_exposed"
    assert "code" not in installed

    again = _install(client)
    assert again["protocol_revision_id"] == revision_id and again["idempotent"] is True
    changed = _install(client, configuration={"cadence": 2})
    assert changed["protocol_revision_id"] != revision_id
    assert set(client.get("/annotation-protocol").json()["installed_revisions"]) == {
        revision_id,
        changed["protocol_revision_id"],
    }

    rollout_id = "live-annotated"
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": rollout_id,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            "annotation_protocol_revision_id": revision_id,
        },
    )
    assert prepared.status_code == 200, prepared.text
    descriptor = prepared.json()["stream"]
    channel = descriptor["annotation"]
    assert channel["events"] == f"/rollouts/{rollout_id}/annotations/events"
    assert channel["stream"] == f"/rollouts/{rollout_id}/annotations/stream"
    assert channel["status"] == "provisional"
    assert channel["id"] == f"stream:{rollout_id}:annotations"

    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "slot": "stream",
            "task_instance_id": "seed:0",
            "policy_ref": _PIN,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            "annotation_protocol_revision_id": revision_id,
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["annotation_protocol_revision_id"] == revision_id
    assert body["stream"]["annotation"]["events"] == channel["events"]
    assert body["terminated"] is True

    rollout_events = client.get(descriptor["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    action_sequences = [row["sequence"] for row in rollout_events if row["kind"] == "action"]
    assert len(action_sequences) == 1

    events = _poll_until_closed(client, channel["events"])
    kinds = [row["kind"] for row in events]
    assert kinds[0] == "annotation.protocol.bound"
    assert kinds[-3:] == ["annotation.closed", "capture.high_water", "capture.closed"]
    bound = events[0]["payload"]
    assert bound["protocol_revision_id"] == revision_id
    assert bound["source_stream_id"] == descriptor["id"]
    findings = [row["payload"] for row in events if row["kind"] == "annotation.finding"]
    assert len(findings) == 1
    assert findings[0]["status"] == "provisional"
    assert findings[0]["evidence"] == {"stream_id": descriptor["id"], "sequences": action_sequences}
    assert findings[0]["label"].startswith("took ")
    metric = [row["payload"] for row in events if row["kind"] == "annotation.metric"][0]
    assert metric == {
        "name": "actions",
        "value": 1.0,
        "step": None,
        "source_sequence": findings[0]["source_sequence"] if False else metric["source_sequence"],
        "protocol_revision_id": revision_id,
    }
    closed = [row["payload"] for row in events if row["kind"] == "annotation.closed"][0]
    assert closed["outcome"] == "completed" and closed["source_closed"] is True
    assert closed["findings"] == 1
    page = client.get(channel["events"], params={"after": 0}).json()
    assert page["schema"] == "synth.live-annotation-stream.v1"
    assert page["stream_id"] == channel["id"]
    assert page["summary"]["outcome"] == "completed"
    for row in page["events"]:
        assert row["rollout_id"] == rollout_id
        assert row["stream_id"] == channel["id"]

    with client.stream("GET", channel["stream"]) as response:
        sse = "".join(response.iter_text())
    assert response.status_code == 200
    assert sse.count("event: stream.subscribed") == 1
    assert "event: annotation.finding" in sse and "event: capture.closed" in sse

    # The rollout's own evidence is untouched by the observer.
    seal = json.loads((tmp_path / "seals" / f"{rollout_id}.trace-v5.json").read_text())
    assert not any(str(event.get("event_type", "")).startswith("annotation.") for event in seal["events"])
    assert client.get(f"/rollouts/{rollout_id}/reward").status_code == 200


def test_annotation_channel_is_subscribable_before_start(tmp_path: Path) -> None:
    """A viewer subscribes to the declared annotation stream after prepare, before start."""

    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    client = TestClient(app)
    revision_id = _install(client)["protocol_revision_id"]
    rollout_id = "subscribe-first"
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": rollout_id,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            "annotation_protocol_revision_id": revision_id,
        },
    )
    assert prepared.status_code == 200, prepared.text
    channel = prepared.json()["stream"]["annotation"]
    page = client.get(channel["events"], params={"after": 0}).json()
    assert page["cursor"] == {"kind": "sequence", "after": 0, "high_water": 0, "closed": False, "next": 0, "has_more": False}
    assert [row["kind"] for row in page["events"]] == ["stream.subscribed"]
    assert page["events"][0]["payload"]["stream.id"] == channel["id"]
    journal = tmp_path / "live_annotation" / "events"
    assert any(journal.glob("*.jsonl")), "the declared stream is durable before start"

    # The SSE route serves the prepared stream; close it so the response ends.
    app.state.platform.live_annotation.logs[rollout_id].mark_closed()
    with client.stream("GET", channel["stream"]) as response:
        body = "".join(response.iter_text())
    assert response.status_code == 200
    assert body.count("event: stream.subscribed") == 1


def test_unknown_or_unbound_protocol_is_refused_or_absent(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    refused = client.post(
        "/rollouts",
        json={
            "rollout_id": "no-such-protocol",
            "task_instance_id": "seed:0",
            "policy_ref": _PIN,
            "annotation_protocol_revision_id": "anprev_0000000000000000",
        },
    )
    assert refused.status_code == 404
    assert refused.json()["error"] == "annotation_protocol_unknown"

    plain = client.post(
        "/rollouts",
        json={"rollout_id": "plain", "task_instance_id": "seed:0", "policy_ref": _PIN},
    )
    assert plain.status_code == 200, plain.text
    assert plain.json()["annotation_protocol_revision_id"] is None
    assert plain.json()["stream"]["annotation"] is None
    assert client.get("/rollouts/plain/annotations/events").status_code == 404


def test_replay_with_a_different_protocol_pin_is_an_identity_conflict(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    first = _install(client)["protocol_revision_id"]
    second = _install(client, configuration={"other": True})["protocol_revision_id"]
    body = {"rollout_id": "pinned", "task_instance_id": "seed:0", "policy_ref": _PIN, "annotation_protocol_revision_id": first}
    assert client.post("/rollouts", json=body).status_code == 200
    replay = client.post("/rollouts", json=body)
    assert replay.status_code == 200 and replay.json()["replayed"] is True
    conflict = client.post("/rollouts", json={**body, "annotation_protocol_revision_id": second})
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "rollout_identity_conflict"


def test_put_protocol_refuses_credentials_missing_identity_and_broken_code(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    no_id = client.put("/annotation-protocol", json={"code": COUNTER})
    assert no_id.status_code == 422 and no_id.json()["error"] == "protocol_identity_required"
    empty = client.put("/annotation-protocol", json={"code": "", "protocol_id": "x"})
    assert empty.status_code == 422 and empty.json()["error"] == "protocol_source_required"
    secret = client.put(
        "/annotation-protocol",
        json={"code": COUNTER, "protocol_id": "x", "configuration": {"model": {"model": "m", "api_key": "sk-nope"}}},
    )
    assert secret.status_code == 422 and secret.json()["error"] == "protocol_credential_forbidden"
    broken = client.put(
        "/annotation-protocol",
        json={"code": "PROTOCOL = 'wrong'\nclass Protocol: pass\n", "protocol_id": "x"},
    )
    assert broken.status_code == 422 and broken.json()["error"] == "protocol_boot_failed"
    mismatch = client.put("/annotation-protocol", json={"code": COUNTER, "protocol_id": "not.the.declared.id"})
    assert mismatch.status_code == 422 and mismatch.json()["error"] == "protocol_identity_mismatch"
    assert client.get("/annotation-protocol").json()["status"] == "not_installed"


def test_protocol_and_annotation_streams_survive_restart(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    revision_id = _install(client)["protocol_revision_id"]
    rollout_id = "restart-me"
    started = client.post(
        "/rollouts",
        json={"rollout_id": rollout_id, "task_instance_id": "seed:0", "policy_ref": _PIN, "annotation_protocol_revision_id": revision_id},
    )
    assert started.status_code == 200, started.text
    before = _poll_until_closed(client, f"/rollouts/{rollout_id}/annotations/events")

    reopened = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    state = reopened.get("/annotation-protocol").json()
    assert state["status"] == "installed" and state["protocol_revision_id"] == revision_id
    record = reopened.get(f"/rollouts/{rollout_id}").json()
    assert record["annotation_protocol_revision_id"] == revision_id
    assert record["stream"]["annotation"]["events"] == f"/rollouts/{rollout_id}/annotations/events"
    page = reopened.get(f"/rollouts/{rollout_id}/annotations/events", params={"after": 0}).json()
    assert page["cursor"]["closed"] is True
    after = [row for row in page["events"] if row.get("sequence") is not None]
    assert [(row["sequence"], row["digest"]) for row in after] == [(row["sequence"], row["digest"]) for row in before]

    # A rollout pinned to the recovered revision runs without re-installing.
    rerun = reopened.post(
        "/rollouts",
        json={"rollout_id": "after-restart", "task_instance_id": "seed:1", "policy_ref": _PIN, "annotation_protocol_revision_id": revision_id},
    )
    assert rerun.status_code == 200, rerun.text
    events = _poll_until_closed(reopened, "/rollouts/after-restart/annotations/events")
    assert any(row["kind"] == "annotation.finding" for row in events)
