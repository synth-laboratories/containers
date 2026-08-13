#!/usr/bin/env python3
"""Paid real Craftax Rust + Muse Spark 10-lane live-stream receipt.

The visual/consumer handshake is recorded before any completion worker (and
therefore before any model request) starts. Each completion runs in a separate
thread while the main thread polls durable partial events.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app

WORLD_REF = "world:craftax_default@symbolic_survival"
POLICY_REF = {"harness": "react", "config": "muse_spark_medium"}
TELEMETRY = {"enabled": True, "transport": "sse"}


def _semantic(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if not event.get("control")]


def run() -> dict[str, Any]:
    storage_root_raw = os.environ.get("SYNTH_CRAFTAX_RECEIPT_ROOT", "").strip()
    storage_root = Path(storage_root_raw).expanduser().resolve() if storage_root_raw else None
    client = TestClient(create_compat_app("craftax_react", storage_root=storage_root))
    lanes: list[dict[str, Any]] = []
    for seed in range(10):
        rollout_id = f"craftax_muse_seed_{seed}"
        prepared = client.post(
            "/rollouts/prepare",
            json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
        )
        prepared.raise_for_status()
        ready = client.get(prepared.json()["stream"]["transports"]["poll"]["url"], params={"after": 0})
        ready.raise_for_status()
        ready_events = ready.json()["events"]
        assert any(event.get("kind") == "stream.subscribed" for event in ready_events)
        assert not _semantic(ready_events), "semantic evidence appeared before start"
        started = client.post(
            "/rollouts",
            json={
                "rollout_id": rollout_id,
                "submission_mode": "async",
                "telemetry": TELEMETRY,
                "slot": "stream",
                "world_ref": WORLD_REF,
                "task_instance_id": f"seed:{seed}",
                "policy_ref": POLICY_REF,
            },
        )
        started.raise_for_status()
        lanes.append(
            {
                "seed": seed,
                "rollout_id": rollout_id,
                "stream": started.json()["stream"],
                "cursor": 0,
                "events": [],
                "partial_observed": False,
            }
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            lane["rollout_id"]: pool.submit(
                client.post,
                f"/rollouts/{lane['rollout_id']}/complete",
            )
            for lane in lanes
        }
        deadline = time.monotonic() + 900
        while not all(future.done() for future in futures.values()):
            assert time.monotonic() < deadline, "Craftax Muse completion timeout"
            for lane in lanes:
                response = client.get(
                    lane["stream"]["transports"]["poll"]["url"],
                    params={"after": lane["cursor"]},
                )
                response.raise_for_status()
                batch = _semantic(response.json().get("events") or [])
                if batch and not futures[lane["rollout_id"]].done():
                    lane["partial_observed"] = True
                lane["events"].extend(batch)
                if batch:
                    lane["cursor"] = max(int(event["sequence"]) for event in batch)
            time.sleep(0.05)

    leaderboard = []
    for lane in lanes:
        completed = futures[lane["rollout_id"]].result()
        completed.raise_for_status()
        completed_body = completed.json()
        tail = client.get(lane["stream"]["transports"]["poll"]["url"], params={"after": lane["cursor"]})
        tail.raise_for_status()
        lane["events"].extend(_semantic(tail.json().get("events") or []))
        reward = client.post(
            "/reward",
            json={"rollout_id": lane["rollout_id"], "mode": "terminal"},
        )
        reward.raise_for_status()
        reward_body = reward.json()
        seal = client.get(f"/rollouts/{lane['rollout_id']}/trace")
        seal.raise_for_status()
        kinds = [event["kind"] for event in lane["events"]]
        policy_data = [
            event.get("payload") or {}
            for event in lane["events"]
            if event.get("kind") == "span.policy.data"
        ]
        usage = completed_body.get("usage") or {}
        leaderboard.append(
            {
                "seed": lane["seed"],
                "rollout_id": lane["rollout_id"],
                "stream_id": lane["stream"]["id"],
                "subscribed_before_start": True,
                "partial_observed": lane["partial_observed"],
                "event_count": len(lane["events"]),
                "policy_trace_events": sum(kind.startswith("span.policy") for kind in kinds),
                "policy_fallbacks": sum(bool(item.get("fallback")) for item in policy_data),
                "frame_events": kinds.count("frame"),
                "reward": reward_body.get("reward"),
                "reward_status": reward_body.get("status"),
                "trace_digest": seal.json().get("content_digest"),
                "usage": usage,
            }
        )
    assert all(row["partial_observed"] for row in leaderboard)
    assert all(row["policy_trace_events"] > 0 for row in leaderboard)
    assert all(row["frame_events"] > 0 for row in leaderboard)
    assert all(row["policy_fallbacks"] == 0 for row in leaderboard)
    assert all(row["trace_digest"] for row in leaderboard)
    cost_values = [
        row["usage"].get("cost_usd")
        for row in leaderboard
        if isinstance(row["usage"].get("cost_usd"), (int, float))
    ]
    return {
        "schema": "synth.craftax-muse-10.receipt.v1",
        "target": "craftax_react",
        "world": "GameBench Craftax Rust gold",
        "policy_ref": POLICY_REF,
        "storage_root": str(storage_root) if storage_root is not None else None,
        "cost_usd": sum(cost_values) if cost_values else None,
        "leaderboard": leaderboard,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
