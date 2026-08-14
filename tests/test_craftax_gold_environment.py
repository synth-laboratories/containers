"""Craftax gold EnvironmentService: sequence relay, no silent 120, fail-closed frames."""

from __future__ import annotations

import json
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.gold_craftax_world import (
    GoldCraftaxWorld,
    GoldEventLogCorrupt,
    GoldFrameMissing,
)
from synth_containers.platform.react import ScriptedReAct
from synth_containers.platform.targets import TARGETS

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


class ScriptedPolicyService(ScriptedReAct):
    def close(self) -> None:
        return


class GoldState:
    def __init__(self) -> None:
        self.rollouts: dict[str, dict[str, Any]] = {}
        self.polls = 0
        self.mutate_prefix = False
        self.shrink_log = False
        self.omit_frames = False
        self.omit_events = False
        self.include_cursor = True
        self.next_rollout = 0


def _gold_handler(state: GoldState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            blob = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _png(self) -> None:
            if state.omit_frames:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(PNG_1X1)))
            self.end_headers()
            self.wfile.write(PNG_1X1)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if self.path == "/rollouts":
                state.next_rollout += 1
                rollout_id = f"gold_roll_{state.next_rollout}"
                max_steps = int(((body.get("task") or {}).get("max_steps")) or 1)
                state.rollouts[rollout_id] = {
                    "steps": 0,
                    "max_steps": max_steps,
                    "events": [
                        {"kind": "task_resolved", "task": "craftax", "seed": body.get("seed")}
                    ],
                }
                self._json(_readout(rollout_id, steps=0, terminated=False))
                return
            if self.path.endswith("/step"):
                rollout_id = self.path.split("/")[2]
                row = state.rollouts[rollout_id]
                row["steps"] += 1
                row["events"].append(
                    {
                        "kind": "action_applied",
                        "action": body.get("action"),
                        "step": row["steps"],
                    }
                )
                terminated = row["steps"] >= row["max_steps"]
                if terminated:
                    row["events"].append({"kind": "terminal", "status": "completed"})
                self._json(_readout(rollout_id, steps=row["steps"], terminated=terminated))
                return
            if self.path.endswith("/checkpoint"):
                rollout_id = self.path.split("/")[2]
                row = state.rollouts[rollout_id]
                blob = base64.b64encode(json.dumps(row).encode("utf-8")).decode("ascii")
                self._json(
                    {
                        "rollout_id": rollout_id,
                        "checkpoint_id": f"gold_checkpoint_{rollout_id}_{row['steps']}",
                        "blob": blob,
                        "bytes": len(blob),
                        "step_index": row["steps"],
                    }
                )
                return
            if self.path.endswith("/restore"):
                rollout_id = self.path.split("/")[2]
                restored = json.loads(base64.b64decode(body["blob"]).decode("utf-8"))
                state.rollouts[rollout_id] = restored
                self._json(
                    {
                        "rollout_id": rollout_id,
                        "restore_report": {"bytes": len(body["blob"])},
                        "state": _readout(
                            rollout_id,
                            steps=restored["steps"],
                            terminated=restored["steps"] >= restored["max_steps"],
                        ),
                    }
                )
                return
            self.send_response(404)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if "/frames/" in self.path:
                self._png()
                return
            if self.path.endswith("/event_log"):
                rollout_id = self.path.split("/")[2]
                row = state.rollouts[rollout_id]
                state.polls += 1
                events = list(row["events"])
                if state.omit_events:
                    self._json({"nev_cursor": 99})
                    return
                if state.shrink_log and state.polls > 1:
                    events = []
                if state.mutate_prefix and state.polls > 1 and events:
                    events[0] = {**events[0], "mutated": True}
                payload: dict[str, Any] = {"events": events}
                if state.include_cursor:
                    payload["nev_cursor"] = 99
                self._json(payload)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def _readout(rollout_id: str, *, steps: int, terminated: bool) -> dict[str, Any]:
    return {
        "rollout_id": rollout_id,
        "terminated": terminated,
        "truncated": False,
        "progress": {"env_steps": steps},
        "readout": {
            "ascii": f"grid-{steps}",
            "grid_hash": f"digest-{steps}",
            "valid_actions": ["noop", "do"],
            "observation_text": "gold",
            "observation": {"x": 0, "y": 0},
            "public": {"done": terminated},
            "private": {"reward_last": 0.5 if steps else 0.0, "total_reward": 0.5 if steps else 0.0},
        },
    }


@pytest.fixture
def gold_http():
    state = GoldState()
    server = HTTPServer(("127.0.0.1", 0), _gold_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    try:
        yield state, url
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_gold_world_requires_explicit_max_steps() -> None:
    with pytest.raises(TypeError):
        GoldCraftaxWorld()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="positive pin"):
        GoldCraftaxWorld(max_steps=0)


def test_gold_relays_nev_and_strips_producer_cursor(gold_http, monkeypatch) -> None:
    state, url = gold_http
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    world = GoldCraftaxWorld(max_steps=1, require_frames=True)
    reset = world.reset(0)
    assert reset.frame_bytes == PNG_1X1
    first = world.drain_native_events()
    assert [event["kind"] for event in first] == ["task_resolved"]
    assert all("nev_cursor" not in event for event in first)
    stepped = world.step("do")
    assert stepped.reward == 0.5
    assert stepped.done is True
    second = world.drain_native_events()
    assert [event["kind"] for event in second] == ["action_applied", "terminal"]


def test_gold_prefix_mutation_fails_closed(gold_http, monkeypatch) -> None:
    state, url = gold_http
    state.mutate_prefix = True
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    world = GoldCraftaxWorld(max_steps=1, require_frames=True)
    world.reset(0)
    world.drain_native_events()
    world.step("do")
    with pytest.raises(GoldEventLogCorrupt, match="prefix mutated"):
        world.drain_native_events()


def test_gold_log_shrink_fails_closed(gold_http, monkeypatch) -> None:
    state, url = gold_http
    state.shrink_log = True
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    world = GoldCraftaxWorld(max_steps=1, require_frames=True)
    world.reset(0)
    world.drain_native_events()
    world.step("do")
    with pytest.raises(GoldEventLogCorrupt, match="shrank"):
        world.drain_native_events()


def test_gold_missing_events_fails_closed(gold_http, monkeypatch) -> None:
    state, url = gold_http
    state.omit_events = True
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    world = GoldCraftaxWorld(max_steps=1, require_frames=True)
    world.reset(0)
    with pytest.raises(GoldEventLogCorrupt, match="omitted events"):
        world.drain_native_events()


def test_gold_missing_frame_fails_closed(gold_http, monkeypatch) -> None:
    state, url = gold_http
    state.omit_frames = True
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    world = GoldCraftaxWorld(max_steps=1, require_frames=True)
    with pytest.raises(GoldFrameMissing):
        world.reset(0)


def test_engine_is_fixture_not_gold() -> None:
    engine = TARGETS["craftax_engine"]
    react = TARGETS["craftax_react"]
    code = TARGETS["craftax_code_policy"]
    assert engine.environment_ref == "env:craftax_fixture"
    assert engine.max_episode_steps == 8
    assert code.environment_ref == "env:craftax_fixture"
    assert code.max_episode_steps == 8
    assert react.environment_ref == "env:craftax_gold"
    assert react.max_episode_steps == 120
    info = TestClient(create_compat_app("craftax_engine")).get("/info").json()
    assert info["environment_ref"] == "env:craftax_fixture"
    assert info["max_episode_steps"] == 8


def test_craftax_react_relays_gold_through_http(gold_http, monkeypatch, tmp_path) -> None:
    state, url = gold_http
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    monkeypatch.setenv("SYNTH_CRAFTAX_MAX_STEPS", "1")
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.craftax.ReactPolicyServiceProcess",
        lambda **kwargs: ScriptedPolicyService(
            config_id=str(kwargs.get("config_id") or "luna_med")
        ),
    )
    app = create_compat_app("craftax_react", storage_root=tmp_path)
    client = TestClient(app)
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "task_resolved" in kinds
    assert "action_applied" in kinds
    assert "terminal" in kinds
    assert "nev_cursor" not in kinds
    assert all("nev_cursor" not in json.dumps(item) for item in events)
    frames = [item for item in events if item["kind"] == "frame"]
    assert frames
    assert any(item["payload"].get("format") == "png" for item in frames)
    assert any(item["kind"] == "artifact.available" for item in events)
    png_frame = next(item for item in frames if item["payload"].get("format") == "png")
    frame_url = png_frame["payload"]["url"]
    fetched = client.get(frame_url)
    assert fetched.status_code == 200, fetched.text
    assert fetched.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert fetched.content == PNG_1X1
    platform = app.state.platform
    frame_rows = [row for row in platform.artifacts.values() if row.get("kind") == "frame"]
    assert frame_rows
    for row in frame_rows:
        assert row["bytes"].startswith(b"\x89PNG\r\n\x1a\n"), row
        assert row["format"] == "png"
        assert row["retention"] == platform.spec.retention
        assert row["digest"]
    digest = png_frame["payload"]["digest"]
    meta = client.get(f"/artifacts/{digest}")
    assert meta.status_code == 200, meta.text
    body = meta.json()
    assert body["format"] == "png"
    assert body["available"] is True
    assert body["kind"] == "frame"
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    assert scored["reward"] == 0.5


def test_craftax_goex_captures_and_forks_true_environment_and_policy_state(
    gold_http, monkeypatch, tmp_path
) -> None:
    _state, url = gold_http
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    monkeypatch.setenv("SYNTH_CRAFTAX_MAX_STEPS", "6")
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.craftax.ReactPolicyServiceProcess",
        lambda **kwargs: ScriptedPolicyService(
            config_id=str(kwargs.get("config_id") or "luna_med")
        ),
    )
    app = create_compat_app("craftax_goex", storage_root=tmp_path)
    client = TestClient(app)
    info = client.get("/info").json()
    assert info["target_id"] == "craftax_goex"
    assert info["affordances"]["environment"]["true_checkpoint"] == "native"
    assert info["affordances"]["environment"]["restore"] == "native"
    assert info["affordances"]["environment"]["fork"] == "native"

    parent = client.post(
        "/rollouts",
        json={
            "rollout_id": "goex_parent",
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            "task_instance_id": "seed:11",
            "policy_ref": {"harness": "react", "config": "luna_med"},
            "recipe": {"require": {"environment.true_checkpoint": "native"}},
            "checkpoint_schedule": {
                "mode": "per_policy_call",
                "checkpoint_id_prefix": "goex_parent_cp",
            },
        },
    )
    assert parent.status_code == 200, parent.text
    checkpoints = parent.json()["scheduled_checkpoints"]
    assert checkpoints
    assert checkpoints[0]["step"] == 0
    assert checkpoints[0]["policy_llm_call_index"] == 0
    checkpoint = next(item for item in checkpoints if item["branchable"] is True)
    assert checkpoint["restore_eligible"] is True
    assert checkpoint["branchable"] is True
    assert "environment_blob" not in checkpoint
    assert "policy_state" not in checkpoint

    child = client.post(
        "/rollouts",
        json={
            "rollout_id": "goex_child",
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            "task_instance_id": "seed:999",
            "policy_ref": {"harness": "react", "config": "sol_med"},
            "recipe": {
                "require": {
                    "environment.true_checkpoint": "native",
                    "environment.restore": "native",
                    "environment.fork": "native",
                }
            },
            "resume_from_checkpoint_id": checkpoint["checkpoint_id"],
            "checkpoint_schedule": {
                "mode": "per_policy_call",
                "checkpoint_id_prefix": "goex_child_cp",
            },
        },
    )
    assert child.status_code == 200, child.text
    body = child.json()
    assert body["rollout_id"] == "goex_child"
    assert body["resume_from_checkpoint_id"] == checkpoint["checkpoint_id"]
    assert body["scheduled_checkpoints"]
    assert body["scheduled_checkpoints"][0]["parent_checkpoint_id"] == checkpoint["checkpoint_id"]
    parent_step = checkpoint["step"]
    child_events = client.get("/rollouts/goex_child/events", params={"after": 0}).json()[
        "events"
    ]
    observations = [event for event in child_events if event["kind"] == "observation"]
    assert observations[0]["payload"]["step"] == parent_step
    assert any(event["kind"] == "rollout.checkpoint" for event in child_events)
    restored = [event for event in child_events if event["kind"] == "rollout.restored"]
    assert len(restored) == 1
    assert restored[0]["payload"]["checkpoint_id"] == checkpoint["checkpoint_id"]
    assert restored[0]["payload"]["step"] == checkpoint["step"]

    terminal = checkpoints[-1]
    if terminal["step"] == 6:
        assert terminal["restore_eligible"] is False
        assert terminal["branchable"] is False
        assert terminal["resume_blockers"] == ["terminal_environment"]

    reopened = create_compat_app("craftax_goex", storage_root=tmp_path)
    recovered = reopened.state.platform.checkpoints[checkpoint["checkpoint_id"]]
    assert recovered["content_digest"]
    assert recovered["environment_blob"]
    assert recovered["policy_state"]["calls"] == checkpoint["policy_llm_call_index"]


def test_engine_fixture_artifacts_are_not_claimed_as_png(tmp_path) -> None:
    app = create_compat_app("craftax_engine", storage_root=tmp_path)
    client = TestClient(app)
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    rid = started.json()["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    frames = [item for item in events if item["kind"] == "frame"]
    assert frames
    assert all(item["payload"].get("format") != "png" for item in frames)
    assert all("url" not in item["payload"] for item in frames)
    assert not any(item["kind"] == "artifact.available" for item in events)
    platform = app.state.platform
    frame_rows = [row for row in platform.artifacts.values() if row.get("kind") == "frame"]
    assert frame_rows
    for row in frame_rows:
        assert row.get("format") != "png"
        assert not bytes(row["bytes"]).startswith(b"\x89PNG\r\n\x1a\n")
        digest = row["digest"]
        meta = client.get(f"/artifacts/{digest}")
        assert meta.status_code == 200, meta.text
        assert meta.json().get("format") != "png"


def test_code_policy_put_and_restart_keep_engine_generation() -> None:
    client = TestClient(create_compat_app("craftax_code_policy"))
    put = client.put("/policy", json={"code": "def act(obs):\n    return 0\n"})
    assert put.status_code == 200, put.text
    generation = put.json()["engine_generation"]
    restart = client.post("/policy/restart")
    assert restart.status_code == 200, restart.text
    body = restart.json()
    assert body["engine_generation"] == generation
    assert body.get("isolation_receipt", {}).get("sandbox") == "process"


def test_craftax_gold_exposes_continuous_levers_and_prompt_program() -> None:
    client = TestClient(create_compat_app("craftax_react"))
    info = client.get("/info").json()
    assert info["environment_ref"] == "env:craftax_gold"
    assert info["affordances"]["environment"]["bind_policy_config"] == "native"
    assert info["affordances"]["environment"]["update_policy_code"] == "native"

    manifest = client.get("/levers").json()
    assert manifest["version"] == "lever_manifest.v1"
    assert {lever["lever_id"] for lever in manifest["levers"]} == {
        "react_system_prompt",
        "policy_config",
        "harness_code",
    }
    harness = next(item for item in manifest["levers"] if item["lever_id"] == "harness_code")
    assert harness["apply_mode"] == "restart"
    assert harness["schema"]["contract"] == "policy_candidate.v1"

    program = client.get("/program").json()
    assert program["version"] == "prompt_program.v1"
    assert program["modules"][0]["module_id"] == "react_system_prompt"
    assert program["rollout_overlay_schema"] == {
        "react_system_prompt": "policy.config.system_prompt"
    }

    config = client.post(
        "/policy-configs/lane3",
        json={"config": {"system_prompt": "Collect wood first."}},
    ).json()
    assert config["digest"].startswith("sha256:")
    assert len(config["digest"]) == len("sha256:") + 64


def test_nonconforming_candidate_is_rejected_before_routing() -> None:
    app = create_compat_app("craftax_react")
    client = TestClient(app)
    before = app.state.platform.current_policy_revision_id
    rejected = client.put(
        "/policy",
        json={"code": "def choose_actions(**kwargs):\n    return {'action': 'noop'}\n"},
    )
    assert rejected.status_code == 422
    receipt = rejected.json()["conformance_receipt"]
    assert receipt["contract"] == "policy_candidate.v1"
    assert receipt["status"] == "rejected"
    assert app.state.platform.current_policy_revision_id == before


def test_conforming_candidate_runs_over_http_and_restarts_on_gold_target() -> None:
    client = TestClient(create_compat_app("craftax_react"))
    accepted = client.put(
        "/policy",
        json={
            "code": (
                "def choose_actions(*, valid_actions, **kwargs):\n"
                "    return {'actions': [valid_actions[0]], 'policy_reason': 'contract_probe'}\n"
            )
        },
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["conformance_receipt"]["status"] == "accepted"
    assert accepted.json()["isolation_receipt"]["transport"] == "http"
    restarted = client.post("/policy/restart")
    assert restarted.status_code == 200
    assert restarted.json()["isolation_receipt"]["contract"] == "policy_candidate.v1"


def test_engine_fixture_frames_are_ascii_not_png() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    artifacts = client.app.state.platform.artifacts
    assert artifacts
    assert all(row.get("format") == "ascii" for row in artifacts.values())
    assert all(row["bytes"] == b"ASCII" for row in artifacts.values())
