"""Conformance for publication, preparation and immutable next-rollout pins."""

from copy import deepcopy
from dataclasses import replace
import time

from fastapi.testclient import TestClient
import pytest

from synth_containers.gold_runtime import GoldRuntime
from synth_containers.http_adapter import create_reference_app
from synth_containers.platform import create_compat_app
from synth_containers.platform.affordances import AffordanceMap
from synth_containers.platform.targets import OPENENV_ECHO
from synth_containers.policy_revision import PolicyManifest
from synth_containers.reference_runtime import ReferenceManagedRuntime


def client(tmp_path, *, code=False):
    roles = deepcopy(OPENENV_ECHO.affordances.by_role)
    roles["environment"].update(bind_policy_config="native", update_policy_code="native")
    runtime = (
        GoldRuntime(environment_ref="env:test", task_payload=lambda seed, steps: {})
        if code
        else None
    )
    return TestClient(
        create_compat_app(
            replace(OPENENV_ECHO, runtime=runtime, affordances=AffordanceMap(by_role=roles)),
            storage_root=tmp_path,
        )
    )


def manifest(value="A"):
    return {
        "schema_version": "synth.policy-revision.v1",
        "harness": "gym_loop",
        "namespace": "conformance",
        "name": "policy",
        "components": {"config": {"value": value}},
    }


def publish(c, body=None):
    r = c.post("/policy-revisions", json=body or manifest())
    assert r.status_code == 200, r.text
    return r.json()["policy_revision_id"]


def prepare(c, revision, operation="prepare", **extra):
    return c.post(
        f"/policy-revisions/{revision}/prepare", json={"operation_id": operation, **extra}
    )


def test_publish_is_immutable_non_selecting_and_normalizes_order(tmp_path):
    c = client(tmp_path)
    revision = publish(c)
    assert c.get("/policy").json()["policy_revision_id"] is None
    assert c.post("/policy-revisions", json=manifest()).json()["idempotent"]
    assert publish(c, manifest("B")) != revision
    assert c.put(f"/policy-revisions/{revision}", json=manifest("B")).status_code == 409
    assert c.get(f"/policy-revisions/{revision}").json()["state"] == "published"


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": True},
        {"components": {"config": {"inference_target": {"auth_bearer": "forbidden"}}}},
        {"schema_version": "unknown"},
        {"components": {"config": {"nested": {"access_token": "never-print"}}}},
        {
            "components": {
                "weights_ref": {"digest": "sha256:" + "a" * 64, "uri": "https://host/a?sig=x"}
            }
        },
    ],
)
def test_invalid_manifests_refused(tmp_path, change):
    c = client(tmp_path)
    assert c.post("/policy-revisions", json={**manifest(), **change}).status_code == 422


def test_no_effect_for_unsupported_profile_and_not_ready(tmp_path):
    c = client(tmp_path)
    revision = publish(c)
    assert prepare(c, revision, profile="live_actor_update.v1").status_code == 409
    r = c.post("/rollouts", json={"policy_revision_id": revision, "rollout_id": "unready"})
    assert r.status_code == 409
    assert "unready" not in c.app.state.platform.logs
    assert c.app.state.platform.current_policy_revision_id is None


def test_ready_receipt_replay_and_operation_conflict(tmp_path):
    c = client(tmp_path)
    revision = publish(c)
    first = prepare(c, revision).json()
    assert first["state"] == "ready"
    assert first["actual_binding"]["policy_revision_id"] == revision
    assert prepare(c, revision).json() == first
    assert prepare(c, publish(c, manifest("B"))).status_code == 409


def test_two_revisions_are_pinned_without_default_mutation(tmp_path):
    c = client(tmp_path)
    a, b = publish(c), publish(c, manifest("B"))
    prepare(c, a, "a")
    prepare(c, b, "b")
    for revision, rollout in [(a, "one"), (b, "two"), (a, "three")]:
        r = c.post(
            "/rollouts",
            json={
                "rollout_id": rollout,
                "policy_revision_id": revision,
                "telemetry": {"enabled": True, "transport": "poll"},
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["policy_revision_id"] == revision
        assert c.app.state.platform.pins[rollout].policy_ref["config"] == revision
    assert c.app.state.platform.policy_configs[a].config == {"value": "A"}
    assert c.app.state.platform.current_policy_revision_id is None


def test_pin_cannot_be_overridden(tmp_path):
    c = client(tmp_path)
    revision = publish(c)
    prepare(c, revision)
    for extra in [
        {"candidate": {"x": "bad"}},
        {"policy_ref": {"harness": "wrong", "config": "echo"}},
    ]:
        assert c.post("/rollouts", json={"policy_revision_id": revision, **extra}).status_code in (
            409,
            422,
        )


@pytest.mark.parametrize(
    "source, failure",
    [
        ("raise RuntimeError('broken')", "policy_startup_failed"),
        ("import time\ntime.sleep(60)", "policy_startup_timeout"),
    ],
)
def test_code_failure_is_not_ready_and_does_not_select(tmp_path, source, failure):
    c = client(tmp_path, code=True)
    body = {
        **manifest(),
        "harness": "isolated_policy_process",
        "components": {
            "code_bundle": {
                "language": "python",
                "entrypoint": "policy.py",
                "files": {"policy.py": source},
                "serve_contract": "process_observation_action.v1",
            }
        },
    }
    revision = publish(c, body)
    start = time.monotonic()
    receipt = prepare(c, revision, timeout_s=1).json()
    assert time.monotonic() - start < 4
    assert receipt["state"] == "failed" and receipt["failure"] == failure
    assert c.app.state.platform.current_policy_revision_id is None
    assert revision not in c.app.state.platform.policy_revisions


def test_bundle_path_escape_refused():
    with pytest.raises(ValueError):
        PolicyManifest.model_validate(
            {
                **manifest(),
                "components": {
                    "code_bundle": {
                        "language": "python",
                        "entrypoint": "../policy.py",
                        "files": {"../policy.py": "x"},
                        "serve_contract": "process_observation_action.v1",
                    }
                },
            }
        )


def test_reference_prompt_adapter_resolves_only_inside_runtime(tmp_path):
    class Runtime(ReferenceManagedRuntime):
        policy_revision_prompt_fields = True

        def program(self):
            return {"modules": [{"mutable": True, "candidate_field": "system"}]}

        async def submit_rollout(self, request):
            self.seen = deepcopy(request)
            return await super().submit_rollout(request)

    runtime = Runtime.counter_default(target=1)
    c = TestClient(create_reference_app(runtime, storage_root=tmp_path))
    revision = publish(
        c,
        {
            **manifest(),
            "harness": "prompt_program",
            "components": {"prompt_fields": {"system": "A"}},
        },
    )
    assert prepare(c, revision).json()["state"] == "ready"
    r = c.post("/rollouts", json={"rollout_id": "prompt", "policy_revision_id": revision})
    assert r.status_code == 200, r.text
    assert r.json()["policy_revision_id"] == revision
    assert runtime.seen["candidate"] == {"system": "A"}
    assert c.get("/rollouts/prompt").json()["policy_revision_id"] == revision
    assert c.get("/rollouts/prompt/state").json()["policy_revision_id"] == revision
    assert (
        c.post(
            "/rollouts", json={"policy_revision_id": revision, "candidate": {"system": "bad"}}
        ).status_code
        == 422
    )


def test_prepared_runtime_data_cannot_mutate_published_content(tmp_path):
    c = client(tmp_path)
    revision = publish(c, manifest({"nested": ["original"]}))
    prepare(c, revision)
    c.app.state.platform.policy_configs[revision].config["value"]["nested"].append("runtime")
    document, _ = c.app.state.platform.revision_registry.resolve(revision)
    assert document.components.config == {"value": {"nested": ["original"]}}


def test_shared_canonical_fixtures():
    import json
    from pathlib import Path
    from synth_containers.package_models import content_bytes, content_digest

    fixtures = json.loads((Path(__file__).parent / "fixtures/policy-revision-v1.json").read_text())
    for vector in fixtures["valid"]:
        document = PolicyManifest.model_validate(vector["manifest"]).model_dump(
            mode="json", exclude_none=True
        )
        assert content_bytes(document).decode() == vector["canonical_json"]
        assert content_digest(document) == vector["digest"]
