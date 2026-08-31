"""The Craftax gold image: its targets, its relay through the platform HTTP app.

The engine-agnostic relay invariants live with the relay itself, in
`containers/tests/test_gold_http.py`.
"""

from __future__ import annotations

import json
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synth_containers.launch import LaunchError, get_image_spec, load_catalog
from synth_containers.platform import create_compat_app

from craftax_gold.targets import CRAFTAX_GOEX, CRAFTAX_NANOHORIZON, CRAFTAX_REACT

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

IMAGE_ROOT = Path(__file__).resolve().parents[1]


def test_image_build_contexts_are_explicit_and_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("CONTAINERS_ROOT", raising=False)
    monkeypatch.delenv("GAMEBENCH_CRAFTAX_ROOT", raising=False)
    with pytest.raises(LaunchError, match=r"build_context_missing.*\$CONTAINERS_ROOT"):
        load_catalog(IMAGE_ROOT)


def test_image_build_contexts_resolve_to_operator_selected_sources(monkeypatch, tmp_path) -> None:
    containers_root = tmp_path / "containers-source"
    gamebench_root = tmp_path / "gamebench-source"
    craftax_root = gamebench_root / "tasks" / "craftax-singleplayer"
    containers_root.mkdir()
    craftax_root.mkdir(parents=True)
    monkeypatch.setenv("CONTAINERS_ROOT", str(containers_root))
    monkeypatch.setenv("GAMEBENCH_CRAFTAX_ROOT", str(gamebench_root))
    specs = load_catalog(IMAGE_ROOT / "catalog.toml")
    assert list(specs) == ["craftax-gamebench-rust"]
    spec = get_image_spec("craftax-gamebench-rust", catalog=IMAGE_ROOT)
    assert spec.root == IMAGE_ROOT
    assert spec.build_contexts == {
        "containers": containers_root.resolve(),
        "gamebench": craftax_root.resolve(),
    }


def test_image_bakes_renderer_native_assets() -> None:
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY --from=gamebench shared/assets/craftax" in dockerfile
    assert "GAMEBENCH_CRAFTAX_ASSETS_DIR=" in dockerfile
    assert 'test -f "${GAMEBENCH_CRAFTAX_ASSETS_DIR}/grass.png"' in dockerfile
    assert 'test -f "${GAMEBENCH_CRAFTAX_ASSETS_DIR}/player-down.png"' in dockerfile


def test_nanohorizon_metadata_advertises_configured_policy_identity() -> None:
    client = TestClient(create_compat_app(CRAFTAX_NANOHORIZON))
    payload = client.get("/metadata").json()
    expected = {
        "harness": "nanohorizon",
        "config": "glm-5.3-flash",
        "code": None,
        "provider": "openrouter",
        "model": "z-ai/glm-5.3-flash",
        "api": "chat_completions",
    }
    assert payload["policy_ref"] == expected
    assert payload["policy_refs"] == [expected]
    assert payload["capabilities"]["policy_refs"] == [expected]
    assert payload["capabilities"]["metadata"]["program_ready"] is True


def test_nanohorizon_program_and_taskset_are_public_but_eval_seeds_are_not() -> None:
    client = TestClient(create_compat_app(CRAFTAX_NANOHORIZON))
    program = client.get("/program")
    assert program.status_code == 200
    body = program.json()
    assert body["version"] == "prompt_program.v1"
    assert body["target_modules"] == [
        {
            "module_id": "system_prompt",
            "candidate_field": "system_prompt",
            "objective": "outcome_reward",
        }
    ]
    assert body["rollout_overlay_schema"] == {"system_prompt": "policy.system_prompt"}

    tasks = client.post("/taskset/tasks", json={"task_ids": ["seed:93001", "seed:94001"]})
    assert tasks.status_code == 200
    assert [task["split"] for task in tasks.json()["tasks"]] == ["train", "heldout"]

    forbidden = client.post("/taskset/tasks", json={"task_ids": ["seed:91001"]})
    assert forbidden.status_code == 422
    assert forbidden.json()["detail"] == "contest_eval_seed_forbidden:91001"


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

        def do_DELETE(self) -> None:  # noqa: N802
            if self.path.startswith("/rollouts/"):
                rollout_id = self.path.split("/")[2]
                if state.rollouts.pop(rollout_id, None) is None:
                    self._json({"error": "unknown_rollout"}, status=404)
                else:
                    self._json({"deleted": True, "rollout_id": rollout_id})
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
            "private": {
                "reward_last": 0.5 if steps else 0.0,
                "total_reward": 0.5 if steps else 0.0,
            },
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


def test_react_is_gold_not_fixture() -> None:
    assert CRAFTAX_REACT.environment_ref == "env:craftax_gold"
    assert CRAFTAX_REACT.max_episode_steps == 120
    info = TestClient(create_compat_app(CRAFTAX_REACT)).get("/info").json()
    assert info["environment_ref"] == "env:craftax_gold"
    assert info["max_episode_steps"] == 120


def test_craftax_react_relays_gold_through_http(gold_http, monkeypatch, tmp_path) -> None:
    state, url = gold_http
    monkeypatch.setenv("SYNTH_CRAFTAX_URL", url)
    monkeypatch.setenv("SYNTH_CRAFTAX_MAX_STEPS", "1")
    app = create_compat_app(CRAFTAX_REACT, storage_root=tmp_path)
    client = TestClient(app)
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "scripted_react", "config": "engine_acceptance"},
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
    app = create_compat_app(CRAFTAX_GOEX, storage_root=tmp_path)
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
            "policy_ref": {"harness": "scripted_react", "config": "engine_acceptance"},
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
            "policy_ref": {"harness": "scripted_react", "config": "engine_acceptance"},
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
    child_events = client.get("/rollouts/goex_child/events", params={"after": 0}).json()["events"]
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

    reopened = create_compat_app(CRAFTAX_GOEX, storage_root=tmp_path)
    recovered = reopened.state.platform.checkpoints[checkpoint["checkpoint_id"]]
    assert recovered["content_digest"]
    assert recovered["environment_blob"]
    assert recovered["policy_state"]["calls"] == checkpoint["policy_llm_call_index"]
