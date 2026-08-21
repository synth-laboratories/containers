#!/usr/bin/env python3
"""Parent hillclimb DAG vs child env-sum on deo_nested (C4-06)."""

from __future__ import annotations

import json
import tempfile
from typing import Any

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("kind")) for row in events]


def _has_gate(node_results: list[dict[str, Any]]) -> bool:
    return any(
        item.get("kind") == "gate" or item.get("node_id") in {"gate", "heldout_gate"}
        for item in node_results
    )


def run() -> dict[str, Any]:
    client = TestClient(create_compat_app("deo_nested", storage_root=tempfile.mkdtemp(prefix="deo-nested-reward-")))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "slot": "stream",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    parent_id = body["rollout_id"]
    child_id = body.get("child_rollout_id")
    assert child_id, body

    parent_scored = client.post("/reward", json={"rollout_id": parent_id, "mode": "terminal"})
    assert parent_scored.status_code in {200, 202}, parent_scored.text
    parent_record = parent_scored.json()

    child_scored = client.post("/reward", json={"rollout_id": child_id, "mode": "terminal"})
    assert child_scored.status_code in {200, 202}, child_scored.text
    child_record = child_scored.json()

    parent_nodes = list(parent_record.get("node_results") or [])
    assert _has_gate(parent_nodes), parent_record
    assert parent_record.get("reward") != child_record.get("reward"), {
        "parent": parent_record.get("reward"),
        "child": child_record.get("reward"),
    }

    parent_events = client.get(body["stream"]["transports"]["poll"]["url"], params={"after": 0}).json().get("events") or []
    parent_kinds = _kinds(parent_events)
    assert "frame" not in parent_kinds, parent_kinds

    sibling_dag_landed = any(
        item.get("kind") in {"delta", "hillclimb", "aggregate"}
        or item.get("node_id") in {"delta", "hillclimb", "baseline_delta"}
        for item in parent_nodes
    )
    if not sibling_dag_landed:
        assert _has_gate(parent_nodes)
        assert parent_record.get("reward") != child_record.get("reward")
        assert "frame" not in parent_kinds

    return {
        "parent": parent_record,
        "child": child_record,
        "parent_kinds": parent_kinds,
        "sibling_dag_landed": sibling_dag_landed,
    }


def main() -> int:
    result = run()
    print(
        json.dumps(
            {"parent": result["parent"], "child": result["child"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
