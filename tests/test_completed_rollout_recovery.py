"""Destructive reopen acceptance for completed Containers runs."""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


BODY = {
    "rollout_id": "destructive_reopen_1",
    "task_instance_id": "seed:4",
    "policy_ref": {"harness": "react", "config": "luna_med"},
    "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
}


def test_completed_rollout_reopens_from_durable_storage_without_reexecution(tmp_path) -> None:
    first_app = create_compat_app("craftax_engine", storage_root=tmp_path)
    first = TestClient(first_app)
    prepared = first.post(
        "/rollouts/prepare",
        json={"rollout_id": BODY["rollout_id"], "telemetry": BODY["telemetry"]},
    )
    assert prepared.status_code == 200, prepared.text
    started = first.post("/rollouts", json=BODY)
    assert started.status_code == 200, started.text
    first_platform = first_app.state.platform
    assert first_platform.step_calls > 0
    scored = first.post(
        "/reward", json={"rollout_id": BODY["rollout_id"], "mode": "terminal"}
    )
    assert scored.status_code == 200, scored.text
    original_reward = scored.json()
    original_events = first.get(
        f"/rollouts/{BODY['rollout_id']}/events", params={"after": 0, "limit": 2}
    ).json()
    original_seal = first.get(f"/rollouts/{BODY['rollout_id']}/trace").json()

    # Constructing a new process façade is the destructive boundary: no live
    # in-memory pins, logs, reward cache, or seal cache survive it.
    reopened_app = create_compat_app("craftax_engine", storage_root=tmp_path)
    reopened = TestClient(reopened_app)
    reopened_platform = reopened_app.state.platform
    assert reopened_platform.step_calls == 0
    status = reopened.get(f"/rollouts/{BODY['rollout_id']}")
    assert status.status_code == 200, status.text
    assert status.json()["terminated"] is True
    assert status.json()["status"] == started.json()["status"]

    cursor = 0
    replayed = []
    while True:
        page = reopened.get(
            f"/rollouts/{BODY['rollout_id']}/events",
            params={"after": cursor, "limit": 2},
        ).json()
        semantic = [row for row in page["events"] if not row["control"]]
        replayed.extend(semantic)
        cursor = page["cursor"]["next"]
        if not page["cursor"]["has_more"]:
            break
    assert [row["sequence"] for row in replayed] == list(
        range(1, original_seal["high_water"] + 1)
    )
    assert reopened.get(f"/rollouts/{BODY['rollout_id']}/trace").json() == original_seal
    assert reopened.get("/reward", params={"rollout_id": BODY["rollout_id"]}).json() == original_reward

    replay_start = reopened.post("/rollouts", json=BODY)
    assert replay_start.status_code == 200, replay_start.text
    assert replay_start.json()["replayed"] is True
    assert reopened_platform.step_calls == 0
    assert original_events["events"][0]["kind"] == "stream.subscribed"


def test_reopen_fails_closed_when_seal_is_tampered(tmp_path) -> None:
    first = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    assert first.post("/rollouts", json=BODY).status_code == 200
    seal_path = tmp_path / "seals" / f"{BODY['rollout_id']}.trace-v5.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["high_water"] += 1
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="trace_seal_digest_mismatch"):
        create_compat_app("craftax_engine", storage_root=tmp_path)


def test_reopen_fails_closed_when_reward_receipt_is_tampered(tmp_path) -> None:
    first = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path))
    assert first.post("/rollouts", json=BODY).status_code == 200
    assert first.post(
        "/reward", json={"rollout_id": BODY["rollout_id"], "mode": "terminal"}
    ).status_code == 200
    key = hashlib.sha256(BODY["rollout_id"].encode()).hexdigest()
    reward_path = tmp_path / "reward_receipts" / f"{key}.json"
    wrapper = json.loads(reward_path.read_text(encoding="utf-8"))
    wrapper["receipt"]["reward"] = 999
    reward_path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="reward_receipt_digest"):
        create_compat_app("craftax_engine", storage_root=tmp_path)
