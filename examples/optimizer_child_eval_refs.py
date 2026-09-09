#!/usr/bin/env python3
"""Optimizer child-eval resource refs + occupancy 429 on scale_leases."""

from __future__ import annotations

import json
import tempfile

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def child_ref(started: dict) -> dict:
    stream = started.get("stream") or {}
    rollout_id = started["rollout_id"]
    return {
        "schema": "synth.resource-ref.v1",
        "kind": "container_rollout",
        "id": rollout_id,
        "attributes": {
            "stream_id": stream.get("id"),
            "reward_url": (stream.get("reward") or {}).get("url"),
        },
    }


def main() -> int:
    client = TestClient(create_compat_app("harbor_public", storage_root=tempfile.mkdtemp(prefix="optimizer-child-eval-")))
    refs = []
    for index in range(2):
        started = client.post(
            "/rollouts",
            json={
                "telemetry": {"enabled": True, "transport": "sse"},
                "submission_mode": "async",
                "task_instance_id": f"seed:{index}",
                "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
            },
        )
        assert started.status_code == 200, started.text
        refs.append(child_ref(started.json()))
    busy = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "submission_mode": "async",
            "task_instance_id": "seed:2",
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert busy.status_code == 429, busy.text
    body = busy.json()
    assert body.get("affordance") == "scale_leases" or (body.get("detail") or {}).get("affordance") == "scale_leases" or "scale_leases" in json.dumps(body)
    print(json.dumps({"refs": refs, "occupancy": body}, indent=2))
    assert refs[0]["id"] != refs[1]["id"]
    assert refs[0]["attributes"]["stream_id"] != refs[1]["attributes"]["stream_id"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
