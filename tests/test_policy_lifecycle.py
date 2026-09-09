"""Immutable NanoHorizon policy installation and rollout admission."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from synth_containers.gold_runtime import GoldRuntime
from synth_containers.platform import create_compat_app
from synth_containers.platform.affordances import AffordanceMap
from synth_containers.platform.targets import OPENENV_ECHO


def _client() -> TestClient:
    affordances = AffordanceMap(
        by_role={
            "environment": {
                "poll": "native",
                "sse": "derived",
                "live_frames": "unsupported",
                "update_policy_code": "native",
                "bind_policy_config": "native",
            },
            "policy": {},
            "evaluator": {},
        }
    )
    spec = replace(
        OPENENV_ECHO,
        target_id="craftax_nanohorizon_test",
        world_ref="world:craftax",
        environment_ref="env:craftax_gold",
        evaluation_plan_ref="eval:craftax",
        default_policy_harness="nanohorizon",
        scale_leases=8,
        policy_seeds=(),
        affordances=affordances,
        task_id="craftax",
        task_family="craftax",
    )
    return TestClient(create_compat_app(spec))


def _install(
    client: TestClient,
    *,
    source_revision: str = "git:abc",
    name: str = "glm-5.3-flash",
    code: str = "def policy(observation):\n    return 'noop'\n",
) -> dict:
    response = client.put(
        "/policy",
        json={
            "code": code,
            "harness": "nanohorizon",
            "namespace": "nanohorizon",
            "name": name,
            "configuration": {"temperature": 0},
            "model": {"provider": "openrouter", "model_id": "z-ai/glm-5.3-flash"},
            "source_revision": source_revision,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _start(client: TestClient, rollout_id: str, revision: str | None, *, config: str = "glm-5.3-flash"):
    body = {
        "rollout_id": rollout_id,
        "submission_mode": "async",
        "task_instance_id": "seed:780005",
        "policy_ref": {"harness": "nanohorizon", "config": config},
        "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
    }
    if revision is not None:
        body["policy_revision_id"] = revision
    return client.post("/rollouts", json=body)


def test_policy_state_is_non_disclosing_and_idempotent() -> None:
    client = _client()
    empty = client.get("/policy").json()
    assert empty["status"] == "not_installed"
    assert empty["policy_revision_id"] is None
    assert empty["credential_state"] == "not_exposed"

    first = _install(client)
    second = _install(client)
    assert first["policy_revision_id"] == second["policy_revision_id"]
    assert second["idempotent"] is True
    assert second["status"] == "installed"
    serialized = str(client.get("/policy").json())
    assert "def policy" not in serialized
    assert "api_key" not in serialized.lower()


def test_material_policy_change_produces_new_revision() -> None:
    client = _client()
    first = _install(client, source_revision="git:abc")
    second = _install(client, source_revision="git:def")
    assert first["policy_revision_id"] != second["policy_revision_id"]


def test_rollout_rejects_missing_unknown_and_mismatched_policy_revision() -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "glm-5.3-flash", "harness": "nanohorizon", "config": {}},
    )
    _install(client, source_revision="git:abc")

    missing = _start(client, "missing_revision", None)
    assert missing.status_code == 422
    assert missing.json()["error"] == "policy_revision_required"

    unknown = _start(client, "unknown_revision", "polrev_does_not_exist")
    assert unknown.status_code == 404
    assert unknown.json()["error"] == "policy_revision_unknown"

    newer = _install(client, source_revision="git:def")["policy_revision_id"]
    client.post(
        "/policy-configs",
        json={"config_id": "other", "harness": "react", "config": {}},
    )
    mismatched = _start(client, "mismatched_revision", newer, config="other")
    assert mismatched.status_code == 409
    assert mismatched.json()["error"] == "policy_configuration_mismatch"
    assert mismatched.json()["requested_policy_ref"] == {
        "harness": "nanohorizon",
        "config": "other",
    }
    assert mismatched.json()["registered_policy_config"] == {
        "harness": "react",
        "config": "other",
    }


def test_two_installed_policy_harness_revisions_can_run_concurrently(
    monkeypatch: MonkeyPatch,
) -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "default", "harness": "nanohorizon", "config": {}},
    )
    client.post(
        "/policy-configs",
        json={"config_id": "with-goal", "harness": "nanohorizon", "config": {}},
    )
    default_revision = _install(
        client,
        source_revision="git:default",
        name="default-tools",
        code="TOOLS = ['craftax_interact']\n",
    )["policy_revision_id"]
    goal_revision = _install(
        client,
        source_revision="git:with-goal",
        name="goal-tools",
        code="TOOLS = ['set_goal', 'craftax_interact']\n",
    )["policy_revision_id"]

    # The second PUT only advances GET /policy's default pointer. Both
    # immutable revisions remain independently selectable and can hold leases
    # at the same time.
    assert client.get("/policy").json()["policy_revision_id"] == goal_revision
    default_run = _start(client, "default_arm", default_revision, config="default")
    goal_run = _start(client, "goal_arm", goal_revision, config="with-goal")

    assert default_run.status_code == 200, default_run.text
    assert goal_run.status_code == 200, goal_run.text
    assert default_run.json()["policy_revision_id"] == default_revision
    assert goal_run.json()["policy_revision_id"] == goal_revision
    assert default_run.json()["policy_ref"]["config"] == "default"
    assert goal_run.json()["policy_ref"]["config"] == "with-goal"
    default_events = client.get(
        "/rollouts/default_arm/events", params={"after": 0}
    ).json()["events"]
    goal_events = client.get(
        "/rollouts/goal_arm/events", params={"after": 0}
    ).json()["events"]
    default_opened = next(row for row in default_events if row["kind"] == "trace.opened")
    goal_opened = next(row for row in goal_events if row["kind"] == "trace.opened")
    assert default_opened["payload"]["policy_revision_id"] == default_revision
    assert goal_opened["payload"]["policy_revision_id"] == goal_revision

    selected_sources: list[bytes] = []

    def build_selected(**kwargs):
        selected_sources.append(kwargs["code"])
        return object()

    monkeypatch.setattr("synth_containers.gold_runtime.build_nanohorizon", build_selected)
    runtime = GoldRuntime(
        environment_ref="env:craftax_gold",
        task_payload=lambda seed, max_steps: {"seed": seed, "max_steps": max_steps},
    )
    platform = client.app.state.platform
    runtime._nanohorizon_planner(platform, platform.pins["default_arm"])
    runtime._nanohorizon_planner(platform, platform.pins["goal_arm"])
    assert selected_sources == [
        b"TOOLS = ['craftax_interact']\n",
        b"TOOLS = ['set_goal', 'craftax_interact']\n",
    ]


def test_rollout_id_cannot_be_replayed_with_a_different_policy_revision() -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "glm-5.3-flash", "harness": "nanohorizon", "config": {}},
    )
    first_revision = _install(client, source_revision="git:first")["policy_revision_id"]
    second_revision = _install(client, source_revision="git:second")["policy_revision_id"]
    first = _start(client, "same_rollout", first_revision)
    conflict = _start(client, "same_rollout", second_revision)

    assert first.status_code == 200, first.text
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "rollout_identity_conflict"


def test_prepare_replay_preserves_seed_task_and_policy_revision_identity() -> None:
    client = _client()
    first_revision = _install(client, source_revision="git:first")["policy_revision_id"]
    second_revision = _install(client, source_revision="git:second")["policy_revision_id"]
    body = {
        "rollout_id": "prepared_identity",
        "task_instance_id": "seed:780000",
        "policy_revision_id": first_revision,
        "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
    }

    prepared = client.post("/rollouts/prepare", json=body)
    replayed = client.post("/rollouts/prepare", json=body)

    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["seed"] == 780000
    assert prepared.json()["task_instance_id"] == "seed:780000"
    assert prepared.json()["policy_revision_id"] == first_revision
    assert replayed.status_code == 200, replayed.text
    assert replayed.json()["replayed"] is True
    assert replayed.json()["policy_revision_id"] == first_revision

    conflict = client.post(
        "/rollouts/prepare",
        json={**body, "policy_revision_id": second_revision},
    )
    assert conflict.status_code == 409
    detail = conflict.json()["detail"]
    assert detail["error"] == "rollout_prepare_identity_conflict"
    assert detail["prepared"]["policy_revision_id"] == first_revision
    assert detail["requested"]["policy_revision_id"] == second_revision


def test_selected_policy_revision_must_match_requested_harness() -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "react-config", "harness": "react", "config": {}},
    )
    revision = _install(client)["policy_revision_id"]
    response = client.post(
        "/rollouts",
        json={
            "rollout_id": "wrong_harness",
            "submission_mode": "async",
            "task_instance_id": "seed:780005",
            "policy_ref": {"harness": "react", "config": "react-config"},
            "policy_revision_id": revision,
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
        },
    )

    assert response.status_code == 409
    assert response.json()["error"] == "policy_harness_mismatch"
    assert response.json()["policy_revision_id"] == revision
    assert response.json()["requested_harness"] == "react"
    assert response.json()["installed_harness"] == "nanohorizon"


def test_rollout_rejects_when_policy_is_not_installed() -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "glm-5.3-flash", "harness": "nanohorizon", "config": {}},
    )
    response = _start(client, "not_installed", "polrev_not_installed")
    assert response.status_code == 409
    assert response.json()["error"] == "policy_not_installed"


def test_rollout_allows_dynamic_config_id_for_the_installed_harness() -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "nh-session-123", "harness": "nanohorizon", "config": {}},
    )
    revision = _install(client)["policy_revision_id"]

    response = _start(
        client,
        "dynamic_config_id",
        revision,
        config="nh-session-123",
    )

    assert response.status_code == 200, response.text
    assert response.json()["policy_ref"] == {
        "harness": "nanohorizon",
        "config": "nh-session-123",
        "code": None,
    }


def test_policy_put_rejects_embedded_credentials() -> None:
    client = _client()
    response = client.put(
        "/policy",
        json={
            "code": "pass",
            "harness": "nanohorizon",
            "namespace": "nanohorizon",
            "name": "unsafe",
            "configuration": {"OPENROUTER_API_KEY": "must-not-enter-policy-state"},
            "model": {},
            "source_revision": "git:abc",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "policy_credential_forbidden"
