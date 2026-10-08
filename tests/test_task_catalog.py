"""Authoritative task catalog and retained instance coverage."""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def _body(rollout_id: str) -> dict:
    return {
        "rollout_id": rollout_id,
        "submission_mode": "async",
        "task_instance_id": "seed:780000",
        "policy_ref": {"harness": "gym_loop", "config": "echo"},
        "telemetry": TELEMETRY,
    }


def test_empty_catalog_has_one_definition_and_task_info_agrees() -> None:
    client = TestClient(create_compat_app("openenv_echo"))
    catalog = client.get("/task_catalog").json()
    assert catalog["schema_version"] == "synth.container.task-catalog.v1"
    assert len(catalog["tasks"]) == 1
    assert catalog["instances"] == []
    assert client.get("/task_info").json() == catalog["tasks"][0]


def test_materialize_seed_instances_is_stable_and_separate_from_rollout_identity() -> None:
    client = TestClient(create_compat_app("openenv_echo"))
    task_id = client.get("/task_info").json()["id"]
    request = {"task_id": task_id, "seeds": [780005, 780006]}
    first = client.post("/task_instances/materialize", json=request)
    assert first.status_code == 200, first.text
    second = client.post("/task_instances/materialize", json=request)
    assert second.json() == first.json()
    instances = client.get("/task_catalog").json()["instances"]
    assert [row["id"] for row in instances] == [
        f"{task_id}:seed:780005",
        f"{task_id}:seed:780006",
    ]
    assert [row["rollout_id"] for row in instances] == [None, None]
    assert [row["status"] for row in instances] == ["planned", "planned"]


def test_materialized_seed_instances_survive_container_restart(tmp_path) -> None:
    first = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    task_id = first.get("/task_info").json()["id"]
    assert first.post(
        "/task_instances/materialize",
        json={"task_id": task_id, "seeds": [780005]},
    ).status_code == 200
    reopened = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    instances = reopened.get("/task_catalog").json()["instances"]
    assert [row["task_instance_id"] for row in instances] == [
        f"{task_id}:seed:780005"
    ]


def test_prepare_start_and_terminal_keep_one_instance_identity() -> None:
    client = TestClient(create_compat_app("openenv_echo"))
    body = _body("catalog_identity")
    prepared = client.post(
        "/rollouts/prepare",
        json={
            "rollout_id": body["rollout_id"],
            "task_instance_id": body["task_instance_id"],
            "telemetry": TELEMETRY,
        },
    )
    assert prepared.status_code == 200, prepared.text
    instance = client.get("/task_catalog").json()["instances"][0]
    assert instance["id"] == body["task_instance_id"]
    assert instance["rollout_id"] == body["rollout_id"]
    assert instance["status"] == "prepared"
    assert instance["started"] is False

    started = client.post("/rollouts", json=body)
    assert started.status_code == 200, started.text
    running = client.get("/task_catalog").json()["instances"]
    assert [row["id"] for row in running] == [body["task_instance_id"]]
    assert [row["rollout_id"] for row in running] == [body["rollout_id"]]
    assert running[0]["status"] == "running"
    assert running[0]["reward"] is None

    completed = client.post(f"/rollouts/{body['rollout_id']}/complete")
    assert completed.status_code == 200, completed.text
    terminal = client.get("/task_catalog").json()["instances"]
    assert [row["id"] for row in terminal] == [body["task_instance_id"]]
    assert terminal[0]["status"] == "completed"
    assert terminal[0]["terminal"] is True
    assert terminal[0]["reward"] is None


def test_terminal_catalog_instance_recovers_without_secrets(tmp_path) -> None:
    first_app = create_compat_app("openenv_echo", storage_root=tmp_path)
    first = TestClient(first_app)
    first_app.state.platform.policy_configs["echo"].config.update(
        {
            "model": "safe-model-name",
            "api_key": "MUST_NOT_APPEAR",
            "source": "MUST_NOT_APPEAR",
        }
    )
    body = _body("catalog_recovery")
    body["policy_ref"] = {
        "harness": "gym_loop",
        "config": "echo",
        "code": "MUST_NOT_APPEAR",
    }
    assert first.post("/rollouts", json=body).status_code == 200
    assert first.post(f"/rollouts/{body['rollout_id']}/complete").status_code == 200

    reopened = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path))
    catalog = reopened.get("/task_catalog").json()
    assert [row["id"] for row in catalog["instances"]] == [body["task_instance_id"]]
    assert [row["rollout_id"] for row in catalog["instances"]] == [body["rollout_id"]]
    assert catalog["instances"][0]["reward"] is None
    serialized = str(catalog)
    assert "MUST_NOT_APPEAR" not in serialized
