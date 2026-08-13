#!/usr/bin/env python3
"""Ten Craftax seeds 0–9 through Containers HTTP (C3-01 / A1 headless). Not evals gold CLI."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app

WORLD_REF = "world:craftax_default@symbolic_survival"
POLICY_REF = {"harness": "react", "config": "luna_med"}
TELEMETRY = {"enabled": True, "transport": "sse"}


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("kind")) for row in events]


def run(*, seeds: int = 10) -> dict[str, Any]:
    client = TestClient(create_compat_app("craftax_engine"))
    client.post("/policy-configs", json={"config_id": "luna_med", "config": {"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024, "context_token_budget": 16000, "compact_at": 0.7, "keep_recent_messages": 8, "keep_recent_frames": 2, "observation_mode": "text"}})
    rows: list[dict[str, Any]] = []
    for seed in range(seeds):
        rollout_id = f"craftax_seed_{seed}"
        body: dict[str, Any] = {
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": WORLD_REF,
            "task_instance_id": f"seed:{seed}",
            "policy_ref": POLICY_REF,
        }
        prepared = client.post(
            "/rollouts/prepare",
            json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
        )
        assert prepared.status_code == 200, prepared.text
        stream = prepared.json()["stream"]
        before = client.get(stream["transports"]["poll"]["url"], params={"after": 0})
        assert before.status_code == 200, before.text
        before_events = before.json().get("events") or []
        kinds = _kinds(before_events)
        assert "stream.subscribed" in kinds, f"C1-08: not subscribed before start: {kinds}"
        assert not any(not row.get("control") for row in before_events), (
            f"C1-08: semantic event published before start: {kinds}"
        )
        started = client.post("/rollouts", json=body)
        assert started.status_code == 200, started.text
        started_body = started.json()
        stream = started_body["stream"]
        scored = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"})
        assert scored.status_code == 200, scored.text
        reward_body = scored.json()
        reward = reward_body.get("reward")
        status = reward_body.get("status")
        if status == "absent":
            reward = None
        rows.append(
            {
                "seed": seed,
                "rollout_id": started_body["rollout_id"],
                "task_instance_id": started_body.get("task_instance_id"),
                "world_ref": started_body.get("world_ref"),
                "environment_ref": started_body.get("environment_ref"),
                "policy_ref": {
                    "harness": (started_body.get("policy_ref") or {}).get("harness"),
                    "config": (started_body.get("policy_ref") or {}).get("config"),
                },
                "stream.id": stream.get("id"),
                "subscribed_before_start": True,
                "status": status,
                "reward": reward,
            }
        )
    ids = [row["rollout_id"] for row in rows]
    assert len(set(ids)) == seeds, ids
    assert all(row["environment_ref"] == "env:craftax_fixture" for row in rows)
    return {
        "target": "craftax_engine",
        "harness": POLICY_REF["harness"],
        "note": "headless Containers HTTP; not evals gold CLI; not --paid Luna",
        "leaderboard": rows,
    }


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
