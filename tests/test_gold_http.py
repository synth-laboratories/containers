"""`GoldHttpWorld`: NEV relay, producer-cursor containment, fail-closed frames.

These invariants are shared by every gold HTTP engine (craftax, rogue,
dungeongrid), so they live with the relay rather than with any one image.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from synth_containers.gold_http import (
    GoldEventLogCorrupt,
    GoldFrameMissing,
    GoldHttpWorld,
)


def _task_payload(seed: int, max_steps: int) -> dict[str, Any]:
    return {"seed": seed, "max_steps": max_steps}


def _world(*, max_steps: int, base_url: str) -> GoldHttpWorld:
    return GoldHttpWorld(
        max_steps=max_steps,
        task_payload=_task_payload,
        base_url=base_url,
        engine="craftax",
        require_frames=True,
    )


PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)


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


def test_gold_world_requires_explicit_max_steps(gold_http) -> None:
    _state, url = gold_http
    with pytest.raises(ValueError, match="positive pin"):
        _world(max_steps=0, base_url=url)


def test_gold_relays_nev_and_strips_producer_cursor(gold_http) -> None:
    state, url = gold_http
    world = _world(max_steps=1, base_url=url)
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


def test_gold_prefix_mutation_fails_closed(gold_http) -> None:
    state, url = gold_http
    state.mutate_prefix = True
    world = _world(max_steps=1, base_url=url)
    world.reset(0)
    world.drain_native_events()
    world.step("do")
    with pytest.raises(GoldEventLogCorrupt, match="prefix mutated"):
        world.drain_native_events()


def test_gold_log_shrink_fails_closed(gold_http) -> None:
    state, url = gold_http
    state.shrink_log = True
    world = _world(max_steps=1, base_url=url)
    world.reset(0)
    world.drain_native_events()
    world.step("do")
    with pytest.raises(GoldEventLogCorrupt, match="shrank"):
        world.drain_native_events()


def test_gold_missing_events_fails_closed(gold_http) -> None:
    state, url = gold_http
    state.omit_events = True
    world = _world(max_steps=1, base_url=url)
    world.reset(0)
    with pytest.raises(GoldEventLogCorrupt, match="omitted events"):
        world.drain_native_events()


def test_gold_missing_frame_fails_closed(gold_http) -> None:
    state, url = gold_http
    state.omit_frames = True
    world = _world(max_steps=1, base_url=url)
    with pytest.raises(GoldFrameMissing):
        world.reset(0)


