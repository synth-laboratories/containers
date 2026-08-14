from __future__ import annotations

import io
from pathlib import Path
import subprocess
import tarfile
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


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def test_local_code_variant_is_conformed_routed_and_deleted(tmp_path: Path) -> None:
    client = TestClient(create_compat_app("craftax_code_policy"))
    archive_response = client.get("/repo/archive")
    assert archive_response.status_code == 200
    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive_response.content), mode="r:gz") as archive:
        archive.extractall(source, filter="data")
    _git(["init", "-q"], source)
    _git(["add", "-A"], source)
    _git(
        [
            "-c",
            "user.email=ohco@synth.local",
            "-c",
            "user.name=ohco",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        source,
    )
    _git(["checkout", "-q", "-b", "ohco/fix-hill-1"], source)
    (source / "policy.py").write_text(
        "def choose_actions(**kwargs):\n"
        "    return {'actions': ['noop'], 'policy_reason': 'candidate'}\n"
    )
    _git(["add", "policy.py"], source)
    _git(
        [
            "-c",
            "user.email=ohco@synth.local",
            "-c",
            "user.name=ohco",
            "commit",
            "-q",
            "-m",
            "candidate",
        ],
        source,
    )
    remote = tmp_path / "git_server.git"
    remote.mkdir()
    _git(["init", "-q", "--bare"], remote)
    _git(["push", "-q", str(remote), "HEAD:refs/heads/ohco/fix-hill-1"], source)

    created = client.post(
        "/variants",
        json={"git_remote": str(remote), "candidate_ref": "ohco/fix-hill-1"},
    )
    assert created.status_code == 200, created.text
    variant = created.json()
    assert variant["status"] == "ready"
    assert variant["conformance_receipt"]["status"] == "accepted"

    request = _optimizer_request(submission_mode="sync", seed=5)
    request["variant_id"] = variant["variant_id"]
    rollout = client.post("/rollout", json=request)
    assert rollout.status_code == 200, rollout.text
    assert rollout.json()["policy_ref"]["harness"] == "isolated_policy_process"
    assert rollout.json()["policy_revision_id"] == variant["policy_revision_id"]

    deleted = client.delete(f"/variants/{variant['variant_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"

    _git(["checkout", "-q", "-b", "ohco/fix-bad-contract"], source)
    (source / "policy.py").write_text("VALUE = 'missing candidate entrypoint'\n")
    _git(["add", "policy.py"], source)
    _git(
        [
            "-c",
            "user.email=ohco@synth.local",
            "-c",
            "user.name=ohco",
            "commit",
            "-q",
            "-m",
            "bad candidate",
        ],
        source,
    )
    _git(["push", "-q", str(remote), "HEAD:refs/heads/ohco/fix-bad-contract"], source)
    rejected = client.post(
        "/variants",
        json={"git_remote": str(remote), "candidate_ref": "ohco/fix-bad-contract"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"] == "candidate_rejected"
    assert rejected.json()["conformance_receipt"]["status"] == "rejected"
