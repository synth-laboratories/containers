"""Immutable NanoHorizon policy installation and rollout admission."""

from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

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


def _install(client: TestClient, *, source_revision: str = "git:abc") -> dict:
    response = client.put(
        "/policy",
        json={
            "code": "def policy(observation):\n    return 'noop'\n",
            "harness": "nanohorizon",
            "namespace": "nanohorizon",
            "name": "glm-5.3-flash",
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


def test_rollout_rejects_missing_unknown_stale_and_mismatched_policy_revision() -> None:
    client = _client()
    client.post(
        "/policy-configs",
        json={"config_id": "glm-5.3-flash", "harness": "nanohorizon", "config": {}},
    )
    current = _install(client, source_revision="git:abc")["policy_revision_id"]

    missing = _start(client, "missing_revision", None)
    assert missing.status_code == 422
    assert missing.json()["error"] == "policy_revision_required"

    unknown = _start(client, "unknown_revision", "polrev_does_not_exist")
    assert unknown.status_code == 404
    assert unknown.json()["error"] == "policy_revision_unknown"

    newer = _install(client, source_revision="git:def")["policy_revision_id"]
    stale = _start(client, "stale_revision", current)
    assert stale.status_code == 409
    assert stale.json()["error"] == "policy_revision_mismatch"

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
