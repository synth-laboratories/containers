from __future__ import annotations

import time

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


def _optimizer_request(*, submission_mode: str, seed: int) -> dict:
    return {
        "run_id": "lane3-smoke",
        "submission_mode": submission_mode,
        "env": {"seed": seed, "split": "train"},
        "policy": {"config": {"model": "fixture", "react_system_prompt": "smoke"}},
        "task_payload": {
            "example": {
                "task_id": "craftax.singleplayer",
                "task_instance_id": f"craftax_singleplayer_search_v1:train:seed:{seed}",
                "seed": seed,
                "is_reference_world": True,
            }
        },
        "metadata": {"optimizer": "ohco", "task_id": "craftax.singleplayer"},
    }


def test_optimizer_sync_rollout_translates_and_returns_typed_reward() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    response = client.post("/rollout", json=_optimizer_request(submission_mode="sync", seed=3))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert isinstance(body["reward_info"]["outcome_reward"], float)
    assert body["summary"]["is_reference_world"] is True
    assert body["metadata"]["optimizer"] == "ohco"
    assert any(event["kind"] == "capture.closed" for event in body["events"])


def test_optimizer_async_rollout_reaches_terminal_record() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    response = client.post("/rollout", json=_optimizer_request(submission_mode="async", seed=4))
    assert response.status_code == 200, response.text
    rollout_id = response.json()["rollout_id"]
    deadline = time.monotonic() + 5
    while True:
        state = client.get(f"/rollouts/{rollout_id}/state")
        assert state.status_code == 200, state.text
        if state.json()["status"] == "completed":
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    record = client.get(f"/rollouts/{rollout_id}").json()
    assert isinstance(record["reward_info"]["outcome_reward"], float)
    assert record["success_status"] == "success"

    first = client.get(f"/rollouts/{rollout_id}/events", params={"after": 0, "limit": 3}).json()
    assert first["cursor"]["has_more"] is True
    second = client.get(
        f"/rollouts/{rollout_id}/events",
        params={"after": first["cursor"]["next"], "limit": 10_000},
    ).json()
    assert second["cursor"]["closed"] is True
    assert all(event.get("kind") != "heartbeat" for event in first["events"] + second["events"])
