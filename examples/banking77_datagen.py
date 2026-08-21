#!/usr/bin/env python3
"""Banking77 train traces through Containers HTTP (dataset_gold harness).

Writes chat-messages JSONL for hosted Tinker SFT. This is the product data-gen
path. Do not dump HuggingFace directly.

Connect-before-start: prepare → stream.subscribed → POST /rollouts.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.banking77_world import CLASSIFY_SYSTEM, split_size

WORLD_REF = "world:banking77@train"
POLICY_REF = {"harness": "dataset_gold", "config": "dataset_gold"}
TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def _to_messages(text: str, label: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Customer query:\n{text}\n\n"
                    "Return EXACTLY one Banking77 intent label as written, no other text."
                ),
            },
            {"role": "assistant", "content": label},
        ]
    }


def collect(*, seeds: int, world_ref: str = WORLD_REF) -> list[dict[str, Any]]:
    client = TestClient(create_compat_app("banking77_classify", storage_root=tempfile.mkdtemp(prefix="banking77-datagen-")))
    available = split_size("train" if "@train" in world_ref else "heldout")
    if seeds > available:
        raise SystemExit(f"asked for {seeds} seeds; split only has {available} fixture rows")
    rows: list[dict[str, Any]] = []
    for seed in range(seeds):
        rollout_id = f"banking77_train_{seed}"
        prepared = client.post(
            "/rollouts/prepare",
            json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
        )
        assert prepared.status_code == 200, prepared.text
        stream = prepared.json()["stream"]
        before = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
        assert any(row.get("kind") == "stream.subscribed" for row in before)
        started = client.post(
            "/rollouts",
            json={
                "rollout_id": rollout_id,
                "telemetry": TELEMETRY,
                "slot": "stream",
                "world_ref": world_ref,
                "task_instance_id": f"seed:{seed}",
                "evaluation_plan_ref": "banking77_eval.v1",
                "policy_ref": POLICY_REF,
            },
        )
        assert started.status_code == 200, started.text
        events = client.get(started.json()["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
        obs = next(row for row in events if row.get("kind") == "observation")
        action = next(row for row in events if row.get("kind") == "action")
        text = str(obs["payload"]["text"])
        label = str(action["payload"]["label"])
        assert "label" not in obs["payload"]
        rows.append(_to_messages(text, label))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--world-ref", default=WORLD_REF)
    args = parser.parse_args()
    rows = collect(seeds=args.seeds, world_ref=args.world_ref)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "out": str(path), "world_ref": args.world_ref}))


if __name__ == "__main__":
    main()
