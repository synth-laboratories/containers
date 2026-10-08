#!/usr/bin/env python3
"""Two-seed Luna medium Craftax e2e against a running HTTP façade."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE = os.environ.get("CRAFTAX_CONTAINER_URL", "http://127.0.0.1:8097")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",") if s.strip()]
TELEMETRY = {"enabled": True, "transport": "sse", "detail": "standard", "frame": {"enabled": False}}
LUNA = {
    "provider": "openrouter",
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "api_key_env": "OPENROUTER_API_KEY",
}


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}Z] {msg}", flush=True)


def subscribed(events: list[dict]) -> bool:
    for row in events:
        if row.get("kind") != "stream.subscribed":
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        ready = row.get("ready") if row.get("ready") is not None else payload.get("ready")
        if ready is True:
            return True
    return False


def run_seed(client: httpx.Client, seed: int, out: Path) -> dict:
    rid = f"craftax_image_luna_{seed}"
    log(f"seed {seed}: prepare {rid}")
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rid, "telemetry": TELEMETRY},
        timeout=30.0,
    )
    prepared.raise_for_status()
    stream = prepared.json()["stream"]
    poll = (stream.get("transports") or {}).get("poll") or {}
    poll_url = stream.get("poll_url") or poll.get("url")
    if not poll_url:
        raise RuntimeError("prepare omitted poll url")
    before = client.get(poll_url, params={"after": 0}, timeout=10.0)
    before.raise_for_status()
    if not subscribed(before.json().get("events") or []):
        raise RuntimeError("stream.subscribed not ready")
    log(f"seed {seed}: start luna medium")
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rid,
            "telemetry": TELEMETRY,
            "slot": "stream",
            "world_ref": "world:craftax_default@symbolic_survival",
            "task_instance_id": f"seed:{seed}",
            "policy_ref": {"harness": "react", "config": "luna_med"},
        },
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
    )
    if started.status_code != 200:
        raise RuntimeError(f"start {started.status_code}: {started.text[:800]}")
    body = started.json()
    events = client.get(poll_url, params={"after": 0, "limit": 1000}, timeout=30.0).json()
    (out / f"{rid}.events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    reward = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}, timeout=30.0)
    scored = reward.json() if reward.status_code == 200 else {"status": "error"}
    kinds = [row.get("kind") for row in (events.get("events") or [])]
    result = {
        "seed": seed,
        "rollout_id": body.get("rollout_id"),
        "http_status": body.get("status"),
        "reward": scored.get("reward"),
        "reward_status": scored.get("status"),
        "event_kinds": kinds,
        "policy_usage": body.get("usage"),
    }
    log(
        f"seed {seed}: status={result['http_status']} reward={result['reward']!r} "
        f"kinds={len(kinds)} usage={result['policy_usage']}"
    )
    if os.environ.get("OPENROUTER_API_KEY") and os.environ["OPENROUTER_API_KEY"] in json.dumps(events):
        raise RuntimeError("api_key_leaked_into_event_log")
    return result


def main() -> int:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 78
    out = Path(os.environ.get("RUN_DIR", "temp/_runs/luna_image_e2e"))
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=BASE) as client:
        health = client.get("/health", timeout=5.0)
        health.raise_for_status()
        log(f"health {health.json()}")
        pin = client.post(
            "/policy-configs",
            json={"config_id": "luna_med", "harness": "react", "config": LUNA},
            timeout=15.0,
        )
        log(f"policy-configs {pin.status_code}")
        if pin.status_code not in {200, 201}:
            raise RuntimeError(f"policy pin failed: {pin.text[:400]}")
        rows = [run_seed(client, seed, out) for seed in SEEDS]
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    failed = [row for row in rows if row["http_status"] not in {"completed", "failed"}]
    print(json.dumps({"ok": not failed, "rows": rows}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
