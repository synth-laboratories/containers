"""P2-5 lock: sampler canary and bearer out of the policy registry."""

from __future__ import annotations

import socket
import time

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.training_rollout import ROLLOUT_REQUEST_SCHEMA_VERSION


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_unreachable_sampler_is_refused_before_a_lease_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINERS_ALLOW_LOOPBACK_SAMPLER", "1")
    storage = tmp_path / "root"
    app = create_compat_app("banking77_classify", storage_root=storage)
    client = TestClient(app)
    port = _closed_port()
    started = time.monotonic()
    response = client.post(
        "/training/rollouts",
        json={
            "schema_version": ROLLOUT_REQUEST_SCHEMA_VERSION,
            "job_id": "job-unreach",
            "attempt_id": "attempt-unreach",
            "rollout_id": "roll_unreach",
            "idempotency_key": "job-unreach:rollout-1",
            "policy_version": "checkpoint:1",
            "sampler": {
                "url": f"http://127.0.0.1:{port}/v1/sample",
                "bearer_token": "job-token",
                "connection_mode": "close",
            },
            "task": {"task_instance_id": "seed:0", "max_tokens": 32},
        },
    )
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, elapsed
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"] == "sampler_unreachable"
    assert body["retryable"] is True
    leases = list((storage / "leases").glob("*.json")) if (storage / "leases").exists() else []
    assert leases == []
    assert app.state.platform.pins == {}


def test_policy_configs_never_contain_auth_bearer(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "synth_containers.platform.app.probe_sampler_health",
        lambda *_args, **_kwargs: None,
    )

    def sample(_client, payload, *, idempotency_key):
        from synth_containers.training_rollout import SamplerResult

        return SamplerResult(
            text="card_arrival",
            prompt_token_ids=(1,),
            token_ids=(1,),
            log_probs=(-0.1,),
            usage={"completion_tokens": 1},
        )

    monkeypatch.setattr(
        "synth_containers.training_rollout.HostedSamplerClient.sample",
        sample,
    )
    app = create_compat_app("banking77_classify", storage_root=tmp_path / "root")
    client = TestClient(app)
    response = client.post(
        "/training/rollouts",
        json={
            "schema_version": ROLLOUT_REQUEST_SCHEMA_VERSION,
            "job_id": "job-bearer",
            "attempt_id": "attempt-bearer",
            "rollout_id": "roll_bearer",
            "idempotency_key": "job-bearer:rollout-1",
            "policy_version": "checkpoint:1",
            "sampler": {
                "url": "https://sampler.example/v1/sample",
                "bearer_token": "secret-token",
                "connection_mode": "close",
            },
            "task": {"task_instance_id": "seed:0", "max_tokens": 32},
        },
    )
    assert response.status_code == 200, response.text
    for cfg in app.state.platform.policy_configs.values():
        dumped = str(cfg.config)
        assert "auth_bearer" not in dumped
        assert "secret-token" not in dumped
        target = (cfg.config or {}).get("inference_target") or {}
        assert "auth_bearer" not in target
    assert app.state.platform.credential_leases == {}
