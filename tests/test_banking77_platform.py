"""Banking77 through Containers HTTP. Content family, not a Harbor wrap.

See: workshop/docs/aug_12_update.md — env/policy/task_world, stream.subscribed,
missing ≠ 0, gold private, /reward env-authored accuracy.
"""

from __future__ import annotations

import json
import urllib.error

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.banking77_world import (
    CLASSIFY_SYSTEM,
    load_row,
    split_size,
    user_prompt,
)


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}
TARGET = "banking77_classify"


def _client() -> TestClient:
    return TestClient(create_compat_app(TARGET))


def _prepare_start(client: TestClient, *, rollout_id: str, body: dict) -> tuple[dict, dict]:
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    before = client.get(stream["transports"]["poll"]["url"], params={"after": 0}).json()
    kinds = [row["kind"] for row in before["events"]]
    assert "stream.subscribed" in kinds
    assert all(row.get("control") for row in before["events"])
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": rollout_id,
            "telemetry": TELEMETRY,
            "slot": "stream",
            **body,
        },
    )
    assert started.status_code == 200, started.text
    return prepared.json(), started.json()


def test_metadata_is_content_not_a_fold() -> None:
    meta = _client().get("/metadata").json()
    assert meta["runtime_family"] == "banking77"
    assert meta["target_id"] == TARGET
    assert meta["adapter_chain"] == []
    assert meta["world_ref"] == "world:banking77@heldout"
    assert meta["environment_ref"] == "env:banking77_dataset"
    assert meta["evaluation_plan_ref"] == "banking77_eval.v1"
    assert meta["policy_ref"]["harness"] == "classify"
    assert "harness_ref" not in meta
    assert meta["live_frames"] == "unsupported"
    assert meta["reward_authority"] == "environment"


def test_auto_transport_refused() -> None:
    response = _client().post(
        "/rollouts",
        json={
            "telemetry": {"enabled": True, "transport": "auto"},
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"] == "transport_refused"


def test_dataset_gold_keeps_label_out_of_observation() -> None:
    client = _client()
    _, started = _prepare_start(
        client,
        rollout_id="b77_gold_0",
        body={
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:0",
            "evaluation_plan_ref": "banking77_eval.v1",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    events = client.get(started["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    kinds = [row["kind"] for row in events]
    assert kinds[0] == "stream.subscribed" or "stream.subscribed" in kinds
    semantic = [row for row in events if not row.get("control")]
    assert semantic[0]["kind"] == "trace.opened"
    obs = next(row for row in semantic if row["kind"] == "observation")
    action = next(row for row in semantic if row["kind"] == "action")
    gold = load_row("train", 0)
    assert gold is not None
    payload = json.dumps(obs["payload"])
    assert gold.label not in payload
    assert "label" not in obs["payload"]
    assert obs["payload"]["text"] == gold.text
    assert action["payload"]["label"] == gold.label
    assert "capture.closed" in kinds
    assert "env.episode.opened" in kinds
    assert "env.episode.closed" in kinds
    assert "frame" not in kinds
    scored = client.post("/reward", json={"rollout_id": "b77_gold_0", "mode": "terminal"}).json()
    assert scored["status"] == "scored"
    assert scored["reward"] == 1.0
    assert scored["node_results"][0]["authority"] == "environment"
    assert started["usage"]["prompt_tokens"] is None
    assert started["usage"]["completion_tokens"] is None


def test_classify_forced_label_scores_accuracy_not_invented_floats() -> None:
    client = _client()
    gold = load_row("heldout", 0)
    assert gold is not None
    client.post(
        "/policy-configs",
        json={
            "config_id": "forced_correct",
            "harness": "classify",
            "config": {"forced_label": gold.label},
        },
    )
    client.post(
        "/policy-configs",
        json={
            "config_id": "forced_wrong",
            "harness": "classify",
            "config": {"forced_label": "card_arrival"},
        },
    )
    _prepare_start(
        client,
        rollout_id="b77_ok",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "classify", "config": "forced_correct"},
        },
    )
    _prepare_start(
        client,
        rollout_id="b77_miss",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "classify", "config": "forced_wrong"},
        },
    )
    ok = client.post("/reward", json={"rollout_id": "b77_ok", "mode": "terminal"}).json()
    miss = client.post("/reward", json={"rollout_id": "b77_miss", "mode": "terminal"}).json()
    assert ok["reward"] == 1.0
    assert miss["reward"] == 0.0
    assert miss["status"] == "scored"


def test_classify_without_sampler_is_absent_not_zero() -> None:
    client = _client()
    _, started = _prepare_start(
        client,
        rollout_id="b77_absent",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:1",
            "policy_ref": {"harness": "classify", "config": "classify"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": "b77_absent", "mode": "terminal"}).json()
    assert scored["status"] == "absent"
    assert scored["reward"] is None
    events = client.get(started["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()["events"]
    signal = next(row for row in events if row["kind"] == "reward_signal")
    assert signal["payload"]["value"] is None


def test_omit_reward_stays_null() -> None:
    client = _client()
    _prepare_start(
        client,
        rollout_id="b77_omit",
        body={
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:2",
            "omit_reward": True,
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    scored = client.post("/reward", json={"rollout_id": "b77_omit", "mode": "terminal"}).json()
    assert scored["reward"] is None
    assert scored["status"] == "absent"


def test_unknown_seed_does_not_wrap() -> None:
    client = _client()
    overflow = split_size("heldout") + 3
    _, started = _prepare_start(
        client,
        rollout_id="b77_overflow",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": f"seed:{overflow}",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    assert started["status"] == "failed"
    scored = client.post("/reward", json={"rollout_id": "b77_overflow", "mode": "terminal"}).json()
    assert scored["reward"] is None
    assert scored["status"] == "absent"


def test_train_and_heldout_worlds_are_distinct() -> None:
    client = _client()
    _prepare_start(
        client,
        rollout_id="b77_train",
        body={
            "world_ref": "world:banking77@train",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    _prepare_start(
        client,
        rollout_id="b77_held",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    train_events = client.get("/rollouts/b77_train/events", params={"after": 0}).json()["events"]
    held_events = client.get("/rollouts/b77_held/events", params={"after": 0}).json()["events"]
    train_obs = next(row for row in train_events if row["kind"] == "observation")["payload"]["text"]
    held_obs = next(row for row in held_events if row["kind"] == "observation")["payload"]["text"]
    assert train_obs != held_obs


def test_live_frames_refused() -> None:
    response = _client().post(
        "/rollouts",
        json={
            "telemetry": TELEMETRY,
            "recipe": {"require": {"live_frames": True}},
            "policy_ref": {"harness": "dataset_gold", "config": "dataset_gold"},
        },
    )
    assert response.status_code == 403
    assert response.json()["affordance"] == "live_frames"


def test_tinker_endpoint_does_not_branch_on_target_id() -> None:
    from synth_containers.platform.runtimes.banking77 import _tinker_endpoint

    assert (
        _tinker_endpoint(
            {
                "inference_target": {
                    "target_id": "tinker-infer:weights",
                    "provider_endpoint_id": "",
                }
            }
        )
        is None
    )
    assert (
        _tinker_endpoint({"inference_target": {"provider_endpoint_id": "tinker://weights"}})
        == "tinker://weights"
    )


def test_tinker_error_code_is_typed_without_provider_prose() -> None:
    from synth_containers.platform.runtimes.banking77 import _error_code

    ProviderError = type("AuthenticationError", (Exception,), {"__module__": "tinker"})
    error = ProviderError("secret-bearing provider prose request_id=req_live_123")
    error.status_code = 401
    code = _error_code(error)
    assert code == "tinker_authenticationerror_401"
    assert "secret" not in code
    assert "request_id" not in code

    missing = ProviderError(
        "The api_key client option must be set either by passing api_key "
        "or by setting the TINKER_API_KEY environment variable"
    )
    assert _error_code(missing) == "tinker_api_key_missing"
def test_remote_checkpoint_sampler_is_loopback_only_and_secret_free(monkeypatch) -> None:
    client = _client()
    gold = load_row("heldout", 0)
    assert gold is not None
    opaque = "opaque-per-run-bearer.do-not-log"
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit: int) -> bytes:
            return json.dumps(
                {
                    "text": gold.label,
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 2,
                        "total_tokens": 14,
                    },
                }
            ).encode("utf-8")

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    registered = client.post(
        "/policy-configs",
        json={
            "config_id": "remote_checkpoint",
            "harness": "classify",
            "config": {
                "max_tokens": 24,
                "inference_target": {
                    "provider_endpoint_id": "http://127.0.0.1:18880/v1/sft/checkpoints/sample",
                    "provider": "tinker",
                    "auth_bearer": opaque,
                    "run_id": "sft_run_1",
                    "checkpoint_id": "checkpoint_20",
                    "base_model": "fixture/base-model",
                },
            },
        },
    )
    assert registered.status_code == 200, registered.text
    assert opaque not in registered.text
    _, started = _prepare_start(
        client,
        rollout_id="b77_remote_checkpoint",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "classify", "config": "remote_checkpoint"},
        },
    )
    assert started["status"] == "completed"
    assert started["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 2,
        "total_tokens": 14,
    }
    assert captured["authorization"] == f"Bearer {opaque}"
    assert captured["body"] == {
        "run_id": "sft_run_1",
        "checkpoint_id": "checkpoint_20",
        "messages": [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": user_prompt(gold.text)},
        ],
        "max_tokens": 24,
    }
    events = client.get(
        "/rollouts/b77_remote_checkpoint/events", params={"after": 0}
    ).json()["events"]
    serialized = json.dumps(events)
    assert opaque not in serialized
    assert "Authorization" not in serialized
    assert "provider_endpoint_id" not in serialized
    assert "auth_bearer" not in serialized
    reward = client.post(
        "/reward",
        json={"rollout_id": "b77_remote_checkpoint", "mode": "terminal"},
    ).json()
    assert reward["reward"] == 1.0


def test_remote_checkpoint_failure_is_null_with_fixed_error_code(monkeypatch) -> None:
    client = _client()
    opaque = "opaque-refused-secret"

    def unavailable(_request, *, timeout):
        del timeout
        raise urllib.error.HTTPError(
            "http://127.0.0.1:18880/v1/sft/checkpoints/sample",
            409,
            "provider unavailable with sensitive prose",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    client.post(
        "/policy-configs",
        json={
            "config_id": "remote_unavailable",
            "harness": "classify",
            "config": {
                "inference_target": {
                    "provider_endpoint_id": "http://localhost:18880/v1/sft/checkpoints/sample",
                    "provider": "tinker",
                    "auth_bearer": opaque,
                    "run_id": "sft_run_2",
                    "checkpoint_id": "checkpoint_40",
                    "base_model": "fixture/base-model",
                }
            },
        },
    )
    _, started = _prepare_start(
        client,
        rollout_id="b77_remote_unavailable",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:1",
            "policy_ref": {"harness": "classify", "config": "remote_unavailable"},
        },
    )
    assert started["status"] == "failed"
    events = client.get(
        "/rollouts/b77_remote_unavailable/events", params={"after": 0}
    ).json()["events"]
    policy_closed = next(row for row in events if row["kind"] == "span.policy.closed")
    assert policy_closed["payload"]["error_code"] == "remote_checkpoint_unavailable"
    serialized = json.dumps(events)
    assert opaque not in serialized
    assert "sensitive prose" not in serialized
    reward = client.post(
        "/reward",
        json={"rollout_id": "b77_remote_unavailable", "mode": "terminal"},
    ).json()
    assert reward["reward"] is None


def test_remote_checkpoint_refuses_unapproved_endpoint_before_network(monkeypatch) -> None:
    client = _client()
    called = False

    def unexpected(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    client.post(
        "/policy-configs",
        json={
            "config_id": "remote_refused",
            "harness": "classify",
            "config": {
                "inference_target": {
                    "provider_endpoint_id": "https://unapproved.example/v1/sample",
                    "provider": "tinker",
                    "auth_bearer": "opaque",
                    "run_id": "run",
                    "checkpoint_id": "checkpoint",
                    "base_model": "model",
                }
            },
        },
    )
    _, started = _prepare_start(
        client,
        rollout_id="b77_remote_refused",
        body={
            "world_ref": "world:banking77@heldout",
            "task_instance_id": "seed:2",
            "policy_ref": {"harness": "classify", "config": "remote_refused"},
        },
    )
    assert started["status"] == "failed"
    assert called is False
    events = client.get(
        "/rollouts/b77_remote_refused/events", params={"after": 0}
    ).json()["events"]
    closed = next(row for row in events if row["kind"] == "span.policy.closed")
    assert closed["payload"]["error_code"] == "remote_checkpoint_endpoint_refused"
