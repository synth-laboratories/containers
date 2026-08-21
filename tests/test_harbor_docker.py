"""Harbor Docker fold: fail closed without a daemon; real trial when present."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.runtimes.harbor import project_harbor_atif
from synth_containers.platform.runtimes.harbor_docker import (
    DockerExecution,
    DockerRunError,
    docker_runtime_available,
)
from synth_containers.platform.targets import PR_TARGETS, TARGETS

_CRAFTAX_KINDS = {
    "frame",
    "observation",
    "action",
    "reward_signal",
    "env.episode.opened",
    "env.episode.closed",
}


def test_harbor_docker_is_not_a_pr_target() -> None:
    assert TARGETS["harbor_docker"].environment_ref == "env:harbor_docker"
    assert "harbor_docker" not in PR_TARGETS


def test_harbor_docker_fails_closed_without_inventing_reward(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: False,
    )
    client = TestClient(create_compat_app("harbor_docker", storage_root=tmp_path / "p0"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "failed"
    rid = body["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "verifier" not in kinds
    assert "trial.planned" not in kinds
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"]["reason"] == "harbor_docker_unavailable"
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
    assert scored.status_code in {200, 202, 409}
    assert scored.json().get("reward") is None


def test_harbor_docker_distinct_executions_read_verifier_reward_txt(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: True,
    )
    calls: list[dict[str, object]] = []

    def fake_execute(
        *,
        role: str,
        image: str,
        command: list[str],
        volumes,
        name: str,
        environment=None,
        allow_nonzero=False,
        timeout_seconds: float = 120.0,
    ):
        del image, timeout_seconds
        assert allow_nonzero is False
        calls.append(
            {"role": role, "name": name, "command": list(command), "environment": environment}
        )
        if role == "agent":
            workspace = Path(volumes["/workspace"])
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "answer.txt").write_text("ok\n", encoding="utf-8")
            return DockerExecution(role=role, exit_code=0, stdout="ok\n", name=name)
        if role == "verifier":
            reward_dir = Path(volumes["/logs"]) / "verifier"
            reward_dir.mkdir(parents=True, exist_ok=True)
            (reward_dir / "reward.txt").write_text("0.25\n", encoding="utf-8")
            return DockerExecution(role=role, exit_code=0, stdout="", name=name)
        raise AssertionError(role)

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.execute_docker_role",
        fake_execute,
    )
    client = TestClient(create_compat_app("harbor_docker", storage_root=tmp_path / "p1"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "completed"
    rid = body["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "trial.planned" in kinds
    assert "span.agent.opened" in kinds
    assert "span.verifier.opened" in kinds
    assert kinds.index("span.agent.closed") < kinds.index("span.verifier.opened")
    assert not _CRAFTAX_KINDS.intersection(kinds)
    assert [row["role"] for row in calls] == ["agent", "verifier"]
    assert calls[0]["environment"] is None
    assert calls[1]["environment"] is None
    assert calls[0]["name"] != calls[1]["name"]
    native = next(
        item["payload"].get("reward.txt") for item in events if item["kind"] == "verifier"
    )
    assert native == 0.25
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
    assert scored.status_code in {200, 202, 409}
    assert scored.json().get("reward") == native == 0.25
    assert project_harbor_atif(events)["reward.txt"] == 0.25
    blob = str(events)
    assert "DOCKER_HOST" not in blob


def test_harbor_docker_run_failure_does_not_invent_reward(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: True,
    )

    def boom(**_kwargs):
        raise DockerRunError("harbor_docker_run_failed")

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.execute_docker_role",
        boom,
    )
    client = TestClient(create_compat_app("harbor_docker", storage_root=tmp_path / "p2"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "failed"
    rid = body["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "verifier" not in kinds
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"]["reason"] == "harbor_docker_run_failed"
    assert status["payload"]["error_type"] == "harbor_docker_run_failed"
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
    assert scored.status_code in {200, 202, 409}
    assert scored.json().get("reward") is None


def test_harbor_docker_missing_reward_txt_stays_null(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: True,
    )

    def fake_execute(
        *,
        role: str,
        image: str,
        command: list[str],
        volumes,
        name: str,
        environment=None,
        allow_nonzero=False,
        timeout_seconds: float = 120.0,
    ):
        del image, command, volumes, environment, allow_nonzero, timeout_seconds
        return DockerExecution(role=role, exit_code=0, stdout="ok\n", name=name)

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.execute_docker_role",
        fake_execute,
    )
    client = TestClient(create_compat_app("harbor_docker", storage_root=tmp_path / "p3"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "failed"
    rid = body["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "verifier" not in kinds
    status = next(item for item in events if item["kind"] == "status")
    assert status["payload"]["reason"] == "harbor_docker_reward_missing"
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
    assert scored.status_code in {200, 202, 409}
    assert scored.json().get("reward") is None


@pytest.mark.skipif(not docker_runtime_available(), reason="docker daemon not present")
def test_harbor_docker_live_agent_and_verifier_are_distinct(tmp_path) -> None:
    client = TestClient(create_compat_app("harbor_docker", storage_root=tmp_path / "p4"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["status"] == "completed", body
    rid = body["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "trial.planned" in kinds
    assert "span.agent.opened" in kinds
    assert "span.verifier.opened" in kinds
    assert kinds.index("span.agent.closed") < kinds.index("span.verifier.opened")
    assert not _CRAFTAX_KINDS.intersection(kinds)
    launched = next(item["payload"] for item in events if item["kind"] == "trial.launched")
    assert launched["sandbox"] == "env:harbor_docker"
    tools = next(item["payload"] for item in events if item["kind"] == "tools")
    verifier = next(item["payload"] for item in events if item["kind"] == "verifier")
    assert str(tools.get("execution") or "").startswith("harbor-agent")
    assert verifier["script"] == "tests/test.sh"
    native = verifier["reward.txt"]
    assert native == 1.0
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
    assert scored.status_code in {200, 202, 409}
    assert scored.json().get("reward") == native
    blob = str(events)
    assert "DOCKER_HOST" not in blob


def test_harbor_fixture_keeps_verifier_on_parent_with_distinct_spans(tmp_path) -> None:
    client = TestClient(create_compat_app("harbor_public", storage_root=tmp_path / "p5"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    ).json()
    rid = started["rollout_id"]
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    kinds = [item["kind"] for item in events]
    assert "trial.planned" in kinds
    assert "span.agent.opened" in kinds
    assert "span.verifier.opened" in kinds
    assert kinds.index("span.agent.closed") < kinds.index("span.verifier.opened")
    native = next(
        item["payload"].get("reward.txt") for item in events if item["kind"] == "verifier"
    )
    scored = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    assert native == scored["reward"] == 1.0
    assert project_harbor_atif(events)["reward.txt"] == 1.0


def test_deo_nested_child_is_code_policy(tmp_path) -> None:
    client = TestClient(create_compat_app("deo_nested", storage_root=tmp_path / "p6"))
    started = client.post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "sse"},
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    child_id = body["child_rollout_id"]
    child = client.get(f"/rollouts/{child_id}").json()
    assert child["environment_ref"] == "env:craftax_fixture"
    assert child["policy_ref"]["harness"] == "isolated_policy_process"
    parent_events = client.get(
        f"/rollouts/{body['rollout_id']}/events", params={"after": 0}
    ).json()["events"]
    child_events = client.get(f"/rollouts/{child_id}/events", params={"after": 0}).json()["events"]
    parent_kinds = {item["kind"] for item in parent_events}
    child_kinds = {item["kind"] for item in child_events}
    assert "frame" in child_kinds
    assert "frame" not in parent_kinds
    parent_reward = client.post(
        "/reward", json={"rollout_id": body["rollout_id"], "mode": "terminal"}
    ).json()
    child_reward = client.post("/reward", json={"rollout_id": child_id, "mode": "terminal"}).json()
    assert parent_reward.get("reward") != child_reward.get("reward")
    assert parent_reward.get("reward") is not None
    assert child_reward.get("reward") is not None
