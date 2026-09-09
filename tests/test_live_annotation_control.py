"""Consumer -> annotator direction: messages, hot-swap with state, stop, duplex WebSocket."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from synth_containers.event_log import RolloutEventLog
from synth_containers.live_annotation.contract import (
    KIND_CONTROL_RECEIVED,
    KIND_CONTROL_REFUSED,
    KIND_FINDING,
    KIND_METRIC,
    KIND_PROTOCOL_REBOUND,
    PROTOCOL_SCHEMA,
    ProtocolRevision,
    protocol_revision_id,
)
from synth_containers.live_annotation.runner import LiveAnnotationRunner
from synth_containers.platform import create_compat_app

COUNTER_V1 = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
PROTOCOL_ID = "test.counter"
class Protocol:
    def __init__(self, config):
        self.n = 0
        self.tag = str(config.get("tag") or "v1")
    def on_event(self, event):
        if event["kind"] == "action":
            self.n += 1
            return [{{"op": "metric", "name": "count:" + self.tag, "value": self.n}}]
        return []
    def on_message(self, message):
        if message.get("type") == "note":
            return [{{"op": "finding", "finding_id": "note:%d" % self.n, "kind": "note",
                     "label": str(message.get("text") or ""), "evidence": {{"sequences": []}},
                     "detail": {{"basis": "consumer"}}}}]
        return []
    def snapshot(self):
        return {{"n": self.n}}
    def restore(self, state):
        self.n = int(state.get("n") or 0)
'''
COUNTER_V2_NO_STATE = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
PROTOCOL_ID = "test.counter"
class Protocol:
    def __init__(self, config):
        self.n = 0
    def on_event(self, event):
        if event["kind"] == "action":
            self.n += 1
            return [{{"op": "metric", "name": "count:v2", "value": self.n}}]
        return []
'''
BROKEN = "PROTOCOL = 'wrong'\nclass Protocol: pass\n"


def _revision(code: str, protocol_id: str = "test.counter", configuration: dict[str, Any] | None = None) -> ProtocolRevision:
    raw = code.encode("utf-8")
    configuration = configuration or {}
    revision_id, sha, cfg = protocol_revision_id(code=raw, protocol_id=protocol_id, configuration=configuration, source_revision=None)
    return ProtocolRevision(revision_id=revision_id, protocol_id=protocol_id, code=raw, code_sha256=sha,
                            configuration=configuration, configuration_digest=cfg, source_revision=None, installed_at="now")


def _payloads(log: RolloutEventLog, kind: str) -> list[dict[str, Any]]:
    return [item.payload for item in log.after(0) if item.kind == kind]


def _kinds(log: RolloutEventLog) -> list[str]:
    return [item.kind for item in log.after(0) if item.sequence is not None]


def _wait(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def test_message_hot_swap_with_state_and_stop() -> None:
    v1 = _revision(COUNTER_V1, configuration={"tag": "v1"})
    v1b = _revision(COUNTER_V1, configuration={"tag": "v1b"})
    v2 = _revision(COUNTER_V2_NO_STATE)
    revisions = {r.revision_id: r for r in (v1, v1b, v2)}
    source = RolloutEventLog(rollout_id="c1", stream_id="stream:c1")
    output = RolloutEventLog(rollout_id="c1", stream_id="stream:c1:annotations")
    runner = LiveAnnotationRunner(rollout_id="c1", source=source, output=output, revision=v1, resolve_revision=revisions.get).start()

    source.append("action", {"action": "up"})
    source.append("action", {"action": "up"})
    _wait(lambda: runner.summary.consumed_high_water == 2)

    # A consumer message reaches on_message and its emissions are attributed to the current cursor.
    ack = runner.control({"op": "message", "control_id": "human-1", "message": {"type": "note", "text": "looks stuck"}})
    assert ack == {"accepted": True, "control_id": "human-1", "op": "message", "queued": True}
    _wait(lambda: runner.summary.controls_received == 1)
    received = _payloads(output, KIND_CONTROL_RECEIVED)[0]
    assert received["op"] == "message" and received["control_id"] == "human-1" and received["handled"] is True
    assert received["message_kind"] == "note" and "text" not in received
    note = [row for row in _payloads(output, KIND_FINDING) if row["kind"] == "note"][0]
    assert note["label"] == "looks stuck" and note["detail"] == {"basis": "consumer"} and note["source_sequence"] == 2

    # Hot-swap to a same-family revision carrying state: numbering continues.
    ack = runner.control({"op": "protocol.update", "protocol_revision_id": v1b.revision_id})
    assert ack["accepted"] is True and ack["op"] == "protocol.update"
    _wait(lambda: runner.summary.rebinds == 1)
    rebound = _payloads(output, KIND_PROTOCOL_REBOUND)[0]
    assert rebound["previous_protocol_revision_id"] == v1.revision_id
    assert rebound["protocol_revision_id"] == v1b.revision_id
    assert rebound["state_carried"] is True and rebound["state_offered"] is True
    source.append("action", {"action": "up"})
    _wait(lambda: runner.summary.consumed_high_water == 3)
    metrics = _payloads(output, KIND_METRIC)
    assert [(m["name"], m["value"]) for m in metrics] == [("count:v1", 1.0), ("count:v1", 2.0), ("count:v1b", 3.0)]
    assert metrics[-1]["protocol_revision_id"] == v1b.revision_id

    # Swap to a revision without restore: state is offered but not carried.
    runner.control({"op": "protocol.update", "protocol_revision_id": v2.revision_id})
    _wait(lambda: runner.summary.rebinds == 2)
    assert _payloads(output, KIND_PROTOCOL_REBOUND)[1]["state_carried"] is False
    source.append("action", {"action": "up"})
    _wait(lambda: runner.summary.consumed_high_water == 4)
    last = _payloads(output, KIND_METRIC)[-1]
    assert (last["name"], last["value"]) == ("count:v2", 1.0)

    # Unknown revision and malformed controls are refused immediately and durably.
    refused = runner.control({"op": "protocol.update", "protocol_revision_id": "anprev_0000000000000000"})
    assert refused == {"accepted": False, "control_id": refused["control_id"], "reason": "annotation_protocol_unknown"}
    bad = runner.control({"op": "dance"})
    assert bad["accepted"] is False and bad["reason"].startswith("control.op_unknown")
    _wait(lambda: runner.summary.controls_refused == 2)
    reasons = [row["reason"] for row in _payloads(output, KIND_CONTROL_REFUSED)]
    assert reasons == ["annotation_protocol_unknown", "control.op_unknown:'dance'"]

    # Stop: the stream seals with a consumer outcome while the source stays open.
    ack = runner.control({"op": "stop", "reason": "operator done"})
    assert ack["accepted"] is True
    assert runner.join(timeout=10)
    assert source.closed is False and output.closed is True
    closed = [item.payload for item in output.after(0) if item.kind == "annotation.closed"][0]
    assert closed["outcome"] == "stopped_by_consumer" and closed["rebinds"] == 2
    late = runner.control({"op": "stop"})
    assert late["accepted"] is False and late["reason"] == "annotation_stream_sealed"


def test_rebind_to_a_broken_revision_keeps_the_running_protocol() -> None:
    v1 = _revision(COUNTER_V1)
    broken = _revision(BROKEN)
    revisions = {r.revision_id: r for r in (v1, broken)}
    source = RolloutEventLog(rollout_id="c2", stream_id="stream:c2")
    output = RolloutEventLog(rollout_id="c2", stream_id="stream:c2:annotations")
    runner = LiveAnnotationRunner(rollout_id="c2", source=source, output=output, revision=v1, resolve_revision=revisions.get).start()
    source.append("action", {"action": "up"})
    _wait(lambda: runner.summary.consumed_high_water == 1)
    runner.control({"op": "protocol.update", "protocol_revision_id": broken.revision_id})
    _wait(lambda: runner.summary.controls_refused == 1)
    assert _payloads(output, KIND_CONTROL_REFUSED)[0]["reason"].startswith("protocol_boot_failed")
    assert runner.summary.rebinds == 0
    source.append("action", {"action": "up"})
    source.mark_closed()
    assert runner.join(timeout=10)
    assert [m["value"] for m in _payloads(output, KIND_METRIC)] == [1.0, 2.0]
    assert [item.payload for item in output.after(0) if item.kind == "annotation.closed"][0]["outcome"] == "completed"


def _install(client: TestClient, code: str, configuration: dict[str, Any] | None = None) -> str:
    response = client.put("/annotation-protocol", json={"code": code, "protocol_id": "test.counter", "configuration": configuration or {}})
    assert response.status_code == 200, response.text
    return response.json()["protocol_revision_id"]


def _prepared_with_live_runner(client: TestClient, app, rollout_id: str, revision_id: str, transport: str = "websocket"):
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": {"enabled": True, "transport": transport, "retention": "run"}, "annotation_protocol_revision_id": revision_id},
    )
    assert prepared.status_code == 200, prepared.text
    platform = app.state.platform
    # Attach the observer to the prepared (still open) rollout log the way _simulate does.
    runner = platform.live_annotation.attach(rollout_id=rollout_id, source=platform.logs[rollout_id], revision_id=revision_id)
    return prepared.json()["stream"], runner


def test_http_control_route_and_descriptor(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    client = TestClient(app)
    v1 = _install(client, COUNTER_V1, {"tag": "v1"})
    v2 = _install(client, COUNTER_V1, {"tag": "v2"})
    stream, runner = _prepared_with_live_runner(client, app, "ctl-http", v1, transport="sse")
    channel = stream["annotation"]
    assert channel["control"] == "/rollouts/ctl-http/annotations/control"
    assert channel["control_schema"] == "synth.live-annotation-control.v1"
    assert channel["websocket"] is None and channel["stream"].endswith("/annotations/stream")

    source = app.state.platform.logs["ctl-http"]
    source.append("action", {"action": "up"})
    accepted = client.post(channel["control"], json={"op": "message", "message": {"type": "note", "text": "hi"}})
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["accepted"] is True and accepted.json()["rollout_id"] == "ctl-http"
    refused = client.post(channel["control"], json={"op": "message", "message": {"type": "note", "api_key": "sk-x"}})
    assert refused.status_code == 422 and refused.json()["reason"] == "control.message_credential_forbidden"
    swapped = client.post(channel["control"], json={"op": "protocol.update", "protocol_revision_id": v2})
    assert swapped.status_code == 202
    _wait(lambda: runner.summary.rebinds == 1)
    stop = client.post(channel["control"], json={"op": "stop"})
    assert stop.status_code == 202
    assert runner.join(timeout=10)
    page = client.get(channel["events"], params={"after": 0}).json()
    kinds = [row["kind"] for row in page["events"] if row.get("sequence") is not None]
    assert kinds.count(KIND_CONTROL_RECEIVED) == 3 and kinds.count(KIND_CONTROL_REFUSED) == 1
    assert KIND_PROTOCOL_REBOUND in kinds and kinds[-1] == "capture.closed"
    assert client.post(channel["control"], json={"op": "stop"}).status_code == 422
    assert client.post("/rollouts/nope/annotations/control", json={"op": "stop"}).status_code == 404


def test_duplex_websocket_carries_envelopes_and_controls(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    client = TestClient(app)
    v1 = _install(client, COUNTER_V1)
    stream, runner = _prepared_with_live_runner(client, app, "ctl-ws", v1, transport="websocket")
    channel = stream["annotation"]
    assert channel["websocket"] == "/rollouts/ctl-ws/annotations/ws"
    source = app.state.platform.logs["ctl-ws"]
    with client.websocket_connect(channel["websocket"]) as ws:
        first = ws.receive_json()
        assert first["kind"] == "stream.subscribed" and first["stream_id"] == channel["id"]
        bound = ws.receive_json()
        assert bound["kind"] == "annotation.protocol.bound" and bound["rollout_id"] == "ctl-ws"
        source.append("action", {"action": "up"})
        metric = ws.receive_json()
        assert metric["kind"] == KIND_METRIC and metric["payload"]["value"] == 1.0
        ws.send_text(json.dumps({"op": "message", "control_id": "ws-1", "message": {"type": "note", "text": "via ws"}}))
        seen: list[dict[str, Any]] = []
        while len(seen) < 3:
            seen.append(ws.receive_json())
        types = [row.get("type") or row["kind"] for row in seen]
        assert "control.ack" in types and KIND_CONTROL_RECEIVED in types and KIND_FINDING in types
        ack = next(row for row in seen if row.get("type") == "control.ack")
        assert ack["accepted"] is True and ack["control_id"] == "ws-1"
        ws.send_text("not json")
        bad = ws.receive_json()
        while bad.get("type") != "control.ack":
            bad = ws.receive_json()
        assert bad["accepted"] is False and bad["reason"].startswith("control.op_unknown")
        ws.send_text(json.dumps({"op": "stop"}))
        tail: list[str] = []
        try:
            while True:
                row = ws.receive_json()
                tail.append(row.get("type") or row["kind"])
        except Exception:  # noqa: BLE001 - the server closes the socket after capture.closed
            pass
        assert "capture.closed" in tail
    assert runner.join(timeout=10) and runner.summary.outcome == "stopped_by_consumer"
