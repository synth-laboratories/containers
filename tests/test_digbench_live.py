"""Live dig.bench relay. Token required. Mock still invents the locked door."""

from __future__ import annotations

import tempfile
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.runtimes.digbench import CHECKPOINT_KINDS
from synth_containers.platform.targets import TARGETS


SEVEN = {
    "session.opened",
    "observation",
    "legal_actions",
    "stats",
    "action",
    "invalid_action",
    "status",
}


class DigState:
    def __init__(self) -> None:
        self.token = "test-digbench-token"
        self.games: list[str] = ["P-1"]
        self.sessions: dict[str, dict[str, Any]] = {}
        self.gets = 0
        self.illegal_rejected = 0


def _initial_state() -> dict[str, Any]:
    return {
        "observation": "A public dungeon. Legal: inspect, wait.",
        "level": 1,
        "lives_left": 3,
        "steps_remaining": 20,
        "actions": ["inspect", "wait"],
        "status": "in_progress",
        "done": False,
    }


def _handler(state: DigState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _auth(self) -> bool:
            header = self.headers.get("Authorization") or ""
            return header == f"Bearer {state.token}"

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            blob = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self) -> None:  # noqa: N802
            if not self._auth():
                self._json({"detail": "unauthorized"}, 401)
                return
            if self.path == "/games":
                self._json({"games": list(state.games)})
                return
            if self.path.startswith("/sessions/"):
                session_id = self.path.rsplit("/", 1)[-1]
                row = state.sessions.get(session_id)
                if row is None:
                    self._json({"detail": "missing"}, 404)
                    return
                state.gets += 1
                self._json(
                    {
                        "session_id": session_id,
                        "game": row["game"],
                        "step_index": row["step_index"],
                        "done": row["state"]["done"],
                        "state": dict(row["state"]),
                    }
                )
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if not self._auth():
                self._json({"detail": "unauthorized"}, 401)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if self.path == "/sessions":
                game = body.get("game")
                if game not in state.games:
                    self._json({"detail": "unknown game"}, 404)
                    return
                session_id = "sess_live_1"
                row = {
                    "session_id": session_id,
                    "game": game,
                    "step_index": 0,
                    "state": _initial_state(),
                }
                state.sessions[session_id] = row
                self._json(
                    {
                        "session_id": session_id,
                        "game": game,
                        "step_index": 0,
                        "done": False,
                        "state": dict(row["state"]),
                    }
                )
                return
            if self.path.endswith("/step"):
                session_id = self.path.split("/")[2]
                row = state.sessions[session_id]
                action = str(body.get("action") or "")
                legal = list(row["state"]["actions"])
                if action not in legal:
                    state.illegal_rejected += 1
                    self._json(
                        {
                            "session_id": session_id,
                            "invalid_action": True,
                            "step_index": row["step_index"],
                            "done": False,
                            "state": dict(row["state"]),
                        }
                    )
                    return
                next_state = dict(row["state"])
                next_state["status"] = "completed"
                next_state["done"] = True
                next_state["steps_remaining"] = 19
                row["state"] = next_state
                row["step_index"] = int(body.get("step_index") or row["step_index"] + 1)
                self._json(
                    {
                        "session_id": session_id,
                        "invalid_action": False,
                        "step_index": row["step_index"],
                        "done": True,
                        "state": dict(next_state),
                    }
                )
                return
            self.send_response(404)
            self.end_headers()

    return Handler


@pytest.fixture
def digbench_http():
    state = DigState()
    server = HTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    try:
        yield state, url
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _start_public(monkeypatch, digbench_http, **body: Any) -> tuple[DigState, TestClient, dict[str, Any]]:
    state, url = digbench_http
    monkeypatch.setenv("DIGBENCH_API_TOKEN", state.token)
    monkeypatch.setenv("SYNTH_DIGBENCH_URL", url)
    client = TestClient(create_compat_app("digbench_public", storage_root=tempfile.mkdtemp(prefix="test_digbench_live-")))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
            **body,
        },
    )
    assert started.status_code == 200, started.text
    return state, client, started.json()


def test_mock_and_public_split_environments() -> None:
    assert TARGETS["digbench_mock"].environment_ref == "env:digbench_mock"
    assert TARGETS["digbench_public"].environment_ref == "env:digbench_relay"


def test_public_without_token_does_not_invent_a_dungeon(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("DIGBENCH_API_TOKEN", raising=False)
    client = TestClient(create_compat_app("digbench_public", storage_root=tmp_path / "p1"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    blob = json.dumps(events)
    assert "locked door" not in blob.lower()
    assert "start_session" not in [item["kind"] for item in events]
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"]["reason"] == "credential_missing"
    assert "DIGBENCH_API_TOKEN" not in blob
    assert "Bearer " not in blob


def test_live_relay_maps_seven_kinds_and_hides_token(digbench_http, monkeypatch) -> None:
    state, client, started = _start_public(monkeypatch, digbench_http)
    rid = started["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = {item["kind"] for item in events}
    assert SEVEN <= kinds
    assert "frame" not in kinds
    assert "state" not in kinds
    blob = json.dumps(events) + json.dumps(client.get("/info").json())
    assert state.token not in blob
    assert "DIGBENCH_API_TOKEN" not in blob
    assert "Bearer " not in blob
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    assert scored["reward"] == 1.0
    assert scored["start_session_delta"] == 0


def test_live_agentic_mcp_spans_share_the_eval_stream(digbench_http, monkeypatch, tmp_path) -> None:
    state, url = digbench_http
    monkeypatch.setenv("DIGBENCH_API_TOKEN", state.token)
    monkeypatch.setenv("SYNTH_DIGBENCH_URL", url)
    client = TestClient(create_compat_app("digbench_public", storage_root=tmp_path / "p2"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "codex", "config": "agentic_codex"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events if not item.get("control")]
    assert "span.mcp.opened" in kinds
    assert "span.mcp.closed" in kinds
    opened = kinds.index("span.mcp.opened")
    closed = kinds.index("span.mcp.closed")
    action = kinds.index("action")
    assert opened < action < closed
    assert SEVEN <= set(kinds)
    mcp_opened = next(item for item in events if item["kind"] == "span.mcp.opened")
    assert mcp_opened["payload"]["evidence_class"] == "simulated"
    action_event = next(item for item in events if item["kind"] == "action")
    assert action_event["payload"]["action_authority"] == "relay_stub"


def test_live_reconnect_get_does_not_add_checkpoint_kinds(digbench_http, monkeypatch) -> None:
    state, client, started = _start_public(monkeypatch, digbench_http)
    rid = started["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = {item["kind"] for item in events}
    assert state.gets >= 1
    assert kinds.isdisjoint(CHECKPOINT_KINDS)
    assert "true_checkpoint" not in kinds
    assert "restore" not in kinds


def test_live_illegal_action_from_api_is_invalid_action(digbench_http, monkeypatch) -> None:
    state, client, started = _start_public(monkeypatch, digbench_http)
    rid = started["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    assert state.illegal_rejected >= 1
    invalid = next(item for item in events if item["kind"] == "invalid_action")
    assert invalid["payload"]["action"] == "fly"
    assert invalid["payload"]["reason"] == "not_legal"
    actions = [item["payload"]["action"] for item in events if item["kind"] == "action"]
    assert "fly" not in actions
    assert "inspect" in actions


def test_live_token_absent_from_trace_seal(digbench_http, monkeypatch) -> None:
    state, client, started = _start_public(monkeypatch, digbench_http)
    rid = started["rollout_id"]
    seal = client.get(f"/rollouts/{rid}/trace")
    assert seal.status_code == 200, seal.text
    blob = json.dumps(seal.json())
    assert state.token not in blob
    assert "DIGBENCH_API_TOKEN" not in blob
    assert "Authorization" not in blob
    assert "Bearer " not in blob


def test_live_freezes_first_listed_game_when_p1_missing(digbench_http, monkeypatch, tmp_path) -> None:
    state, url = digbench_http
    state.games = ["P-2"]
    monkeypatch.setenv("DIGBENCH_API_TOKEN", state.token)
    monkeypatch.setenv("SYNTH_DIGBENCH_URL", url)
    client = TestClient(create_compat_app("digbench_public", storage_root=tmp_path / "p3"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["world_ref"] == "world:digbench:P-2"
    events = client.get(f"/rollouts/{body['rollout_id']}/events", params={"after": 0}).json()["events"]
    opened = next(item for item in events if item["kind"] == "start_session")
    assert opened["payload"]["game"] == "P-2"


def test_live_does_not_silently_swap_a_pinned_game(digbench_http, monkeypatch, tmp_path) -> None:
    state, url = digbench_http
    state.games = ["P-2"]
    monkeypatch.setenv("DIGBENCH_API_TOKEN", state.token)
    monkeypatch.setenv("SYNTH_DIGBENCH_URL", url)
    client = TestClient(create_compat_app("digbench_public", storage_root=tmp_path / "p4"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "world_ref": "world:digbench:P-3",
            "policy_ref": {"harness": "react_legal_actions", "config": "react_legal_actions"},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get(
        f"/rollouts/{started.json()['rollout_id']}/events", params={"after": 0}
    ).json()["events"]
    blob = json.dumps(events)
    assert "locked door" not in blob.lower()
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"]["reason"] == "digbench_relay_error"
    assert status["payload"]["error_type"] == "digbench_game_missing"
