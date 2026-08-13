"""End-to-end Craftax evals through Containers HTTP (ReAct + code-policy)."""

from __future__ import annotations

import json
import sys
from typing import Any

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.policy_process import DEFAULT_HEURISTIC, NOOP_HEURISTIC


def _prepare_and_start(client: TestClient, *, rollout_id: str, body: dict[str, Any]) -> dict[str, Any]:
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": {"enabled": True, "transport": "sse"}},
    )
    if prepared.status_code != 200:
        raise RuntimeError(f"prepare failed: {prepared.status_code} {prepared.text}")
    stream = prepared.json()["stream"]
    subscribed = client.get(stream["transports"]["poll"]["url"], params={"after": 0})
    kinds = [row["kind"] for row in subscribed.json()["events"]]
    if "stream.subscribed" not in kinds:
        raise RuntimeError(f"not subscribed before start: {kinds}")
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            **body,
        },
    )
    if started.status_code != 200:
        raise RuntimeError(f"start failed: {started.status_code} {started.text}")
    return started.json()


def run_react(*, seeds: int = 10) -> dict[str, Any]:
    client = TestClient(create_compat_app("craftax_engine"))
    client.post("/policy-configs", json={"config_id": "luna_med", "config": {"model": "gpt-5.6-luna"}})
    rows = []
    for seed in range(seeds):
        rollout_id = f"react_seed_{seed}"
        started = _prepare_and_start(
            client,
            rollout_id=rollout_id,
            body={
                "world_ref": "world:craftax_default@symbolic_survival",
                "task_instance_id": f"seed:{seed}",
                "policy_ref": {"harness": "react", "config": "luna_med"},
            },
        )
        events = client.get(started["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
        kinds = [row["kind"] for row in events]
        scored = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"}).json()
        rows.append(
            {
                "seed": seed,
                "rollout_id": rollout_id,
                "stream.id": started["stream"]["id"],
                "reward": scored.get("reward"),
                "status": scored.get("status"),
                "policy_spans": kinds.count("span.policy.opened"),
                "actions": kinds.count("action"),
                "first_semantic": next(row["kind"] for row in events if not row.get("control")),
            }
        )
    present = [float(row["reward"]) for row in rows if row["reward"] is not None]
    return {
        "target": "craftax_engine",
        "harness": "react",
        "note": "scripted ReAct; not a Luna eval",
        "leaderboard": rows,
        "mean_reward": (sum(present) / len(present)) if present else None,
        "scored_n": len(present),
        "absent_n": sum(1 for row in rows if row["reward"] is None),
    }


def run_code_policy() -> dict[str, Any]:
    client = TestClient(create_compat_app("craftax_code_policy"))
    do_put = client.put("/policy", json={"code": DEFAULT_HEURISTIC, "harness": "isolated_policy_process"})
    if do_put.status_code != 200:
        raise RuntimeError(f"PUT /policy failed: {do_put.status_code} {do_put.text}")
    engine_gen = do_put.json()["engine_generation"]
    do_run = _prepare_and_start(
        client,
        rollout_id="code_do",
        body={"task_instance_id": "seed:0", "policy_ref": {"harness": "isolated_policy_process"}},
    )
    do_reward = client.post("/reward", json={"rollout_id": "code_do", "mode": "terminal"}).json()

    noop_put = client.put("/policy", json={"code": NOOP_HEURISTIC, "harness": "isolated_policy_process"})
    if noop_put.json().get("engine_generation") != engine_gen:
        raise RuntimeError("PUT /policy mutated engine_generation")
    noop_run = _prepare_and_start(
        client,
        rollout_id="code_noop",
        body={"task_instance_id": "seed:0", "policy_ref": {"harness": "isolated_policy_process"}},
    )
    noop_reward = client.post("/reward", json={"rollout_id": "code_noop", "mode": "terminal"}).json()

    do_actions = [
        row["payload"]["action"]
        for row in client.get(do_run["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
        if row.get("kind") == "action"
    ]
    noop_actions = [
        row["payload"]["action"]
        for row in client.get(noop_run["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
        if row.get("kind") == "action"
    ]
    return {
        "target": "craftax_code_policy",
        "harness": "isolated_policy_process",
        "engine_generation": engine_gen,
        "isolation_receipt": do_put.json().get("isolation_receipt"),
        "do_policy": {"reward": do_reward.get("reward"), "actions": do_actions},
        "noop_policy": {"reward": noop_reward.get("reward"), "actions": noop_actions},
        "distinct_rewards": do_reward.get("reward") != noop_reward.get("reward")
        or do_actions != noop_actions,
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    which = args[0] if args else "both"
    out: dict[str, Any] = {}
    if which in {"react", "both"}:
        out["react"] = run_react()
    if which in {"code-policy", "code_policy", "both"}:
        out["code_policy"] = run_code_policy()
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
