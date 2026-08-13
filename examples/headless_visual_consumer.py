#!/usr/bin/env python3
"""C7-W headless visual consumer: subscribe declared stream, persist, replay, project."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app, project_envelopes
from synth_containers.platform.reducer import assert_honest_projection

SLOT = "stream"
POLICY_PINS = {
    "craftax_engine": {"harness": "react", "config": "luna_med"},
    "harbor_public": {"harness": "harbor_fused", "config": "luna_med"},
    "digbench_mock": {"harness": "react_legal_actions", "config": "react_legal_actions"},
}


def consume(target: str, tmp: Path) -> dict:
    client = TestClient(create_compat_app(target))
    prepared = client.post(
        "/rollouts/prepare",
        json={"telemetry": {"enabled": True, "transport": "sse"}},
    ).json()
    stream = prepared["stream"]
    poll_url = stream["transports"]["poll"]["url"]
    before = client.get(poll_url, params={"after": 0}).json()
    before_envelopes = before.get("events") or before.get("items") or []
    assert any(row.get("kind") == "stream.subscribed" for row in before_envelopes)
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": prepared["rollout_id"],
            "telemetry": {"enabled": True, "transport": "sse"},
            "slot": SLOT,
            "policy_ref": POLICY_PINS[target],
        },
    ).json()
    assert started.get("stream", {}).get("id") == stream["id"]
    events = client.get(poll_url, params={"after": 0}).json()
    envelopes = events.get("events") or events.get("items") or []
    if isinstance(events, dict) and "envelopes" in events:
        envelopes = events["envelopes"]
    path = tmp / f"{target}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for envelope in envelopes:
            handle.write(json.dumps(envelope) + "\n")
    replayed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert [row.get("sequence") for row in replayed] == [row.get("sequence") for row in envelopes]
    assert [row.get("digest") for row in replayed] == [row.get("digest") for row in envelopes]
    reward = client.post("/reward", json={"rollout_id": prepared["rollout_id"]}).json()
    projection = project_envelopes(
        replayed,
        reward=reward.get("reward"),
        reward_status=str(reward.get("status") or "absent"),
        usage=started.get("usage"),
    )
    defects = assert_honest_projection(projection)
    assert not defects, defects
    ready = any(row.get("kind") == "stream.subscribed" for row in envelopes)
    return {
        "target": target,
        "stream_id": stream["id"],
        "slot": SLOT,
        "ready": ready,
        "projection": projection,
        "reward": reward.get("reward"),
        "path": str(path),
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        craftax = consume("craftax_engine", tmp)
        harbor = consume("harbor_public", tmp)
        digbench = consume("digbench_mock", tmp)
        assert craftax["ready"] and harbor["ready"] and digbench["ready"]
        assert craftax["slot"] == harbor["slot"] == digbench["slot"] == SLOT
        assert craftax["projection"]["has_live_frames"] is True
        assert craftax["projection"]["has_reward_txt"] is False
        assert harbor["projection"]["has_live_frames"] is False
        assert harbor["projection"]["has_reward_txt"] is True
        assert digbench["projection"]["has_live_frames"] is False
        assert digbench["projection"]["has_reward_txt"] is False
        print(
            json.dumps(
                {
                    "craftax": {k: craftax[k] for k in ("stream_id", "reward", "slot")},
                    "harbor": {k: harbor[k] for k in ("stream_id", "reward", "slot")},
                    "digbench": {k: digbench[k] for k in ("stream_id", "reward", "slot")},
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
