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
        network="none",
    ):
        del timeout_seconds
        # Both roles tolerate a nonzero exit: the verifier is the scoring
        # authority, and a failed authoring turn is a task outcome, not infra.
        assert allow_nonzero is True
        # This bundle declares no agent network, so both roles stay hermetic.
        assert network == "none"
        calls.append(
            {
                "role": role,
                "image": image,
                "command": command,
                "volumes": volumes,
                "name": name,
                "environment": environment,
                "network": network,
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


def test_authoring_agent_gets_declared_egress_while_verifier_stays_hermetic(
    tmp_path, monkeypatch
) -> None:
    """An authoring agent may declare a bridge; the verifier never gets one.

    Credentials are launcher-owned: they mount for the agent only and must not
    reach the bundle digest or the event log.
    """
    copied = tmp_path / "dock"
    shutil.copytree(FIXTURE.parent, copied)
    manifest_path = copied / "bundle" / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agent"]["network"] = "bridge"
    manifest["agent"]["timeout_seconds"] = 900
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    credentials = tmp_path / "codexhome"
    credentials.mkdir()
    (credentials / "auth.json").write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")

    extension_path = copied / "extension.json"
    extension = json.loads(extension_path.read_text(encoding="utf-8"))
    extension["bundle"]["digest"] = compute_bundle_digest(copied / "bundle")
    extension["agent_credentials"] = str(credentials)
    extension_path.write_text(json.dumps(extension, indent=2), encoding="utf-8")

    seen: dict[str, dict] = {}

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.docker_runtime_available",
        lambda: True,
    )

    def fake_execute(*, role, image, command, volumes, name, environment=None,
                     allow_nonzero=False, timeout_seconds=120.0, network="none"):
        del image, command, allow_nonzero
        seen[role] = {
            "network": network,
            "timeout_seconds": timeout_seconds,
            "environment": dict(environment or {}),
            "volumes": dict(volumes),
        }
        if role == "agent":
            Path(volumes["/workspace"]).mkdir(parents=True, exist_ok=True)
            return DockerExecution(role=role, exit_code=0, stdout="wrote\n", name=name)
        logs = Path(volumes["/logs"]) / "verifier"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "reward.txt").write_text("0.0455\n", encoding="utf-8")
        (logs / "result.json").write_text(
            json.dumps(
                {
                    "baseline_score": 0.0303,
                    "best_score": 0.0455,
                    "best_candidate_id": "fixcraft",
                    "delta_vs_baseline": 0.0152,
                    "passed": True,
                    "leaderboard_path": "/workspace/.harbor_hillclimb/leaderboard.json",
                }
            ),
            encoding="utf-8",
        )
        return DockerExecution(role=role, exit_code=0, stdout="", name=name)

    monkeypatch.setattr(
        "synth_containers.platform.runtimes.harbor_docker.execute_docker_role",
        fake_execute,
    )

    client = TestClient(create_dock_eval_app(extension_path))
    _prepare_subscribe(client, "dock_authoring")
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "dock_authoring",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "harbor_fused", "config": "luna_med"},
        },
    ).json()
    assert started["status"] == "completed"

    # Only the agent gets egress and the longer budget.
    assert seen["agent"]["network"] == "bridge"
    assert seen["agent"]["timeout_seconds"] == 900
    assert seen["verifier"]["network"] == "none"

    # Credentials mount on the agent only, never on the verifier.
    assert seen["agent"]["environment"]["CODEX_HOME"] == "/codexhome"
    assert "/codexhome" in seen["agent"]["volumes"]
    assert "/codexhome" not in seen["verifier"]["volumes"]
    assert "CODEX_HOME" not in seen["verifier"]["environment"]

    events = client.get("/rollouts/dock_authoring/events", params={"after": 0}).json()["events"]
    verifier = next(row for row in events if row["kind"] == "verifier")
    assert verifier["payload"]["reward.txt"] == 0.0455
    # Scalar projection of result.json reaches the consumer...
    assert verifier["payload"]["result"]["best_candidate_id"] == "fixcraft"
    assert verifier["payload"]["result"]["delta_vs_baseline"] == 0.0152
    assert verifier["payload"]["result"]["passed"] is True
    # ...without smuggling non-scalar keys through.
    assert "leaderboard_path" not in verifier["payload"]["result"]

    # A failed authoring turn stays visible as an exit code.
    agent_closed = next(row for row in events if row["kind"] == "span.agent.closed")
    assert agent_closed["payload"]["exit_code"] == 0

    # The credential path is never echoed into the durable log.
    assert str(credentials) not in json.dumps(events)
