from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.training_rollout import (
    HostedSamplerClient,
    ROLLOUT_CAPABILITIES_SCHEMA_VERSION,
    ROLLOUT_REQUEST_SCHEMA_VERSION,
    ROLLOUT_SUMMARY_SCHEMA_VERSION,
    SamplerEndpoint,
    SamplerResult,
    TrainingRolloutError,
    canonical_sha256,
)


def test_banking77_advertises_hashed_training_capabilities() -> None:
    payload = (
        TestClient(create_compat_app("banking77_classify")).get("/training/capabilities").json()
    )
    capability_hash = payload.pop("capability_hash")
    assert payload["schema_version"] == ROLLOUT_CAPABILITIES_SCHEMA_VERSION
    assert payload["task_id"] == "banking77"
    assert payload["supports_sampler_https"] is True
    assert payload["supports_idempotency"] is True
    assert payload["max_concurrency"] > 0
    assert payload["container_digest"].startswith("sha256:")
    assert not payload["container_digest"].startswith("sha256:sha256:")
    assert capability_hash == canonical_sha256(payload)


def test_metadata_and_dedicated_capability_route_agree() -> None:
    client = TestClient(create_compat_app("healthbench_chat"))
    assert client.get("/metadata").json()["training"] == client.get("/training/capabilities").json()


def test_sampler_retry_and_single_refresh_keep_idempotency_key() -> None:
    calls: list[tuple[str, str]] = []
    refreshes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(
            (
                request.headers["authorization"],
                request.headers["idempotency-key"],
            )
        )
        if len(calls) == 1:
            return httpx.Response(401, json={"error": "expired"})
        if len(calls) == 2:
            return httpx.Response(503, json={"error": "retry"})
        return httpx.Response(
            200,
            json={
                "text": "card_arrival",
                "prompt_token_ids": [10, 11],
                "token_ids": [1, 2],
                "log_probs": [-0.1, -0.2],
                "usage": {"output_tokens": 2},
            },
        )

    def refresh() -> str:
        refreshes.append(1)
        return "token-v2"

    with HostedSamplerClient(
        SamplerEndpoint("https://sampler.example/v1/sample", "token-v1", "close"),
        refresh=refresh,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.sample({"messages": []}, idempotency_key="rollout-1:action-0")

    assert result.text == "card_arrival"
    assert refreshes == [1]
    assert [key for _, key in calls] == ["rollout-1:action-0"] * 3
    assert [auth for auth, _ in calls] == [
        "Bearer token-v1",
        "Bearer token-v2",
        "Bearer token-v2",
    ]


def test_sampler_rejects_plaintext_remote_endpoint() -> None:
    with pytest.raises(TrainingRolloutError, match="requires_https"):
        HostedSamplerClient(SamplerEndpoint("http://sampler.example/sample", "token"))


def test_sampler_rejects_endpoint_query() -> None:
    with pytest.raises(TrainingRolloutError, match="endpoint_invalid"):
        HostedSamplerClient(SamplerEndpoint("https://sampler.example/sample?secret=bad", "token"))


def test_sampler_endpoint_repr_redacts_token() -> None:
    endpoint = SamplerEndpoint("https://sampler.example/sample", "top-secret")
    assert "top-secret" not in repr(endpoint)
    # Also guard against an accidental JSON-friendly dataclass representation.
    assert "top-secret" not in json.dumps({"endpoint": repr(endpoint)})


def test_training_rollout_is_policy_stamped_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def sample(
        _client: HostedSamplerClient,
        payload: dict[str, object],
        *,
        idempotency_key: str,
    ) -> SamplerResult:
        calls.append(idempotency_key)
        assert payload["schema_version"] == "training.rollout.action.v1"
        assert payload["policy_version"] == "checkpoint:7"
        return SamplerResult(
            text="card_arrival",
            prompt_token_ids=(10, 11),
            token_ids=(1,),
            log_probs=(-0.1,),
            usage={"completion_tokens": 1},
        )

    monkeypatch.setattr(HostedSamplerClient, "sample", sample)
    client = TestClient(create_compat_app("banking77_classify"))
    request = {
        "schema_version": ROLLOUT_REQUEST_SCHEMA_VERSION,
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "rollout_id": "roll_training_1",
        "idempotency_key": "job-1:rollout-1",
        "policy_version": "checkpoint:7",
        "sampler": {
            "url": "https://sampler.example/v1/sample",
            "bearer_token": "job-token",
            "connection_mode": "close",
        },
        "task": {"task_instance_id": "seed:0", "max_tokens": 32, "temperature": 0.7},
    }
    first = client.post("/training/rollouts", json=request)
    second = client.post("/training/rollouts", json=request)

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["schema_version"] == ROLLOUT_SUMMARY_SCHEMA_VERSION
    assert first.json()["policy_version"] == "checkpoint:7"
    assert first.json()["actions"][0]["token_ids"] == [1]
    assert first.json()["actions"][0]["log_probs"] == [-0.1]
    assert len(calls) == 1

    conflict = client.post(
        "/training/rollouts",
        json={**request, "policy_version": "checkpoint:8"},
    )
    assert conflict.status_code == 409
