"""Private Dock content over the public Harbor/Containers eval lifecycle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.platform import create_dock_eval_app
from synth_containers.platform.extensions.dock import DockEvalExtension
from synth_containers.platform.runtimes.harbor_docker import (
    DockerExecution,
    DockerVolume,
    compute_bundle_digest,
)
from synth_containers.platform.targets import TARGETS

FIXTURE = Path(__file__).parent / "fixtures" / "dock_extension" / "extension.json"
TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def _prepare_subscribe(client: TestClient, rollout_id: str) -> dict:
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    subscribed = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()[
        "events"
    ]
    assert [row["kind"] for row in subscribed] == ["stream.subscribed"]
    return stream


def test_dock_extension_uses_pinned_harbor_runtime_and_public_stream(monkeypatch) -> None:
    extension = DockEvalExtension.from_file(FIXTURE)
    assert extension.bundle_digest == compute_bundle_digest(extension.bundle_root)
    assert "dock" not in TARGETS
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: True,
    )

    def fake_execute(
        *,
        role,
        image,
        command,
        volumes,
        name,
        environment=None,
        allow_nonzero=False,
        timeout_seconds=120.0,
    ):
        del timeout_seconds
        assert allow_nonzero is (role == "verifier")
        calls.append(
            {
                "role": role,
                "image": image,
                "command": command,
                "volumes": volumes,
                "name": name,
                "environment": environment,
            }
        )
        bundle_mount = volumes["/harbor/bundle"]
        task_mount = volumes["/workspace/gamebench/tasks/example"]
        assert isinstance(bundle_mount, DockerVolume) and bundle_mount.read_only
        assert isinstance(task_mount, DockerVolume) and task_mount.read_only
        if role == "agent":
            workspace = Path(volumes["/workspace"])
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "answer.txt").write_text("candidate\n", encoding="utf-8")
            return DockerExecution(role=role, exit_code=0, stdout="candidate\n", name=name)
        logs = Path(volumes["/logs"]) / "verifier"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "reward.txt").write_text("0.75\n", encoding="utf-8")
        return DockerExecution(role=role, exit_code=1, stdout="", name=name)

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.execute_docker_role",
        fake_execute,
    )
    client = TestClient(create_dock_eval_app(extension))
    stream = _prepare_subscribe(client, "dock_fixture_1")
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "dock_fixture_1",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    )
    assert started.status_code == 200, started.text
    assert started.json()["status"] == "completed"
    assert [row["role"] for row in calls] == ["agent", "verifier"]
    assert calls[0]["environment"] == {
        "SYNTH_POLICY_HARNESS": "harbor_fused",
        "SYNTH_POLICY_CONFIG": "luna_med",
    }
    assert calls[1]["environment"] is None
    assert calls[0]["name"] != calls[1]["name"]

    events = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    evidence = [row for row in events if not row.get("control")]
    assert all(row["schema"] == "synth.trace-stream-event.v1" for row in evidence)
    kinds = [row["kind"] for row in evidence]
    assert not any(kind.startswith("dock.") for kind in kinds)
    assert kinds.index("span.agent.closed") < kinds.index("span.verifier.opened")
    planned = next(row["payload"] for row in evidence if row["kind"] == "trial.planned")
    assert planned["bundle_digest"] == extension.bundle_digest
    assert planned["task_tree"]["mount"] == "/workspace/gamebench/tasks/example"
    verifier = next(row["payload"] for row in evidence if row["kind"] == "verifier")
    assert verifier["reward.txt"] == 0.75
    assert verifier["exit_code"] == 1
    reward = client.post(
        "/reward", json={"rollout_id": "dock_fixture_1", "mode": "terminal"}
    ).json()
    assert reward["reward"] == 0.75


def test_dock_extension_tampered_bundle_fails_before_docker(monkeypatch, tmp_path: Path) -> None:
    copied = tmp_path / "dock"
    shutil.copytree(FIXTURE.parent, copied)
    (copied / "bundle" / "instruction.md").write_text("tampered\n", encoding="utf-8")
    executed = False

    def unexpected_execute(**_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("docker must not run for a digest mismatch")

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.execute_docker_role",
        unexpected_execute,
    )
    client = TestClient(create_dock_eval_app(copied / "extension.json"))
    _prepare_subscribe(client, "dock_tampered")
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "dock_tampered",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    ).json()
    assert started["status"] == "failed"
    assert executed is False
    events = client.get("/rollouts/dock_tampered/events", params={"after": 0}).json()["events"]
    assert not any(row["kind"] == "verifier" for row in events)
    status = next(row for row in events if row["kind"] == "status")
    assert status["payload"]["reason"] == "harbor_bundle_digest_mismatch"
    reward = client.post("/reward", json={"rollout_id": "dock_tampered", "mode": "terminal"}).json()
    assert reward.get("reward") is None
    assert str(copied) not in json.dumps(events)
