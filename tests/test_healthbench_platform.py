"""HealthBench is normalized, rubric-authored, and null-cost safe."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.runtimes import healthbench
from synth_containers.platform.targets import HEALTHBENCH_CHAT


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def test_healthbench_declares_thirty_parallel_leases() -> None:
    assert HEALTHBENCH_CHAT.scale_leases == 30


def test_healthbench_info_merges_operations_and_advertises_gepa_v2() -> None:
    info = TestClient(create_compat_app("healthbench_chat")).get("/info").json()
    operations = info["capabilities"]["operations"]
    assert operations["prepare"] is True
    assert operations["start"] is True
    assert operations["get"] is True
    assert operations["poll"] is True
    assert operations["reward"] is True
    gepa = info["capabilities"]["optimizer_contracts"]["gepa"]
    assert gepa["version"] == "synth_optimizers.gepa.v2"
    assert info["optimizer_contracts"]["gepa"]["version"] == "synth_optimizers.gepa.v2"
    configs = {row["config"] for row in info["policy_refs"]}
    assert "openai_gpt41_mini" in configs
    assert "groq_llama31_8b" in configs


def test_healthbench_declares_independent_policy_and_scorer_roles(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHBENCH_GRADER_API_KEY_ENV", "CUSTOM_GRADER_KEY")
    metadata = TestClient(create_compat_app("healthbench_chat")).get("/metadata").json()
    roles = metadata["metadata"]["model_roles"]
    assert roles["policy"]["configuration_authority"] == "policy_ref"
    assert roles["policy"]["usage_lane"] == "policy"
    assert roles["scorer"]["purpose"] == "score_response_against_physician_rubrics"
    assert roles["scorer"]["api_key_env"] == "CUSTOM_GRADER_KEY"
    assert roles["scorer"]["usage_lane"] == "grader"
    assert roles["scorer"]["canonical"] is True


def test_healthbench_info_reports_missing_policy_credential(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHBENCH_POLICY_API_KEY_ENV", "MISSING_POLICY_KEY")
    monkeypatch.setenv("HEALTHBENCH_GRADER_API_KEY_ENV", "PRESENT_GRADER_KEY")
    monkeypatch.setenv("PRESENT_GRADER_KEY", "present")
    info = TestClient(create_compat_app("healthbench_chat")).get("/info").json()
    assert info["metadata"]["model_roles"]["policy"]["credential_present"] is False
    assert info["metadata"]["model_roles"]["scorer"]["credential_present"] is True
    assert info["capabilities"]["metadata"]["policy_ready"] is False
    assert info["capabilities"]["metadata"]["grader_ready"] is True


def test_healthbench_info_reports_missing_grader_credential(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHBENCH_POLICY_API_KEY_ENV", "PRESENT_POLICY_KEY")
    monkeypatch.setenv("HEALTHBENCH_GRADER_API_KEY_ENV", "MISSING_GRADER_KEY")
    monkeypatch.setenv("PRESENT_POLICY_KEY", "present")
    info = TestClient(create_compat_app("healthbench_chat")).get("/info").json()
    assert info["metadata"]["model_roles"]["policy"]["credential_present"] is True
    assert info["metadata"]["model_roles"]["scorer"]["credential_present"] is False
    assert info["capabilities"]["metadata"]["policy_ready"] is True
    assert info["capabilities"]["metadata"]["grader_ready"] is False


def test_healthbench_info_reports_both_credentials_present(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHBENCH_POLICY_API_KEY_ENV", "PRESENT_POLICY_KEY")
    monkeypatch.setenv("HEALTHBENCH_GRADER_API_KEY_ENV", "PRESENT_GRADER_KEY")
    monkeypatch.setenv("PRESENT_POLICY_KEY", "present")
    monkeypatch.setenv("PRESENT_GRADER_KEY", "present")
    info = TestClient(create_compat_app("healthbench_chat")).get("/info").json()
    assert info["metadata"]["model_roles"]["policy"]["credential_present"] is True
    assert info["metadata"]["model_roles"]["scorer"]["credential_present"] is True
    assert info["capabilities"]["metadata"]["policy_ready"] is True
    assert info["capabilities"]["metadata"]["grader_ready"] is True


def test_healthbench_grader_uses_declared_credential(monkeypatch) -> None:
    monkeypatch.setenv("HEALTHBENCH_GRADER_API_KEY_ENV", "CUSTOM_GRADER_KEY")
    captured = []

    def chat(config, messages):
        captured.append(config)
        return {
            "text": '{"explanation":"Appropriate.","criteria_met":true}',
            "usage": healthbench._usage(
                "openai", healthbench.GRADER_MODEL, {"prompt_tokens": 1, "completion_tokens": 1}
            ),
        }

    monkeypatch.setattr(healthbench, "_chat", chat)
    healthbench._grade(_row(), "Rest and seek care for red flags.", log=_RecordingLog())
    assert captured[0]["api_key_env"] == "CUSTOM_GRADER_KEY"


class _RecordingLog:
    def append(self, kind, payload):
        return None


def _row() -> dict:
    return {
        "prompt_id": "health-fixture-1",
        "prompt": [{"role": "user", "content": "I have a headache. What should I do?"}],
        "example_tags": ["advice"],
        "rubrics": [
            {
                "criterion": "Recommends appropriate self-care and escalation.",
                "points": 2,
                "tags": [],
            },
            {
                "criterion": "Claims the user definitely has a brain tumor.",
                "points": -1,
                "tags": [],
            },
        ],
    }


def test_unknown_cost_is_null_but_real_zero_is_preserved() -> None:
    unknown = healthbench._usage("unknown", "unknown", {"prompt_tokens": 5, "completion_tokens": 2})
    known = healthbench._usage(
        "groq", "llama-3.1-8b-instant", {"prompt_tokens": 0, "completion_tokens": 0}
    )
    assert unknown["cost_usd"] is None
    assert unknown["cost_kind"] is None
    assert known["cost_usd"] == 0.0
    assert known["cost_kind"] == "estimated_from_tokens"
    mini = healthbench._usage(
        "openai",
        "gpt-4.1-mini-2025-04-14",
        {"prompt_tokens": 1000, "completion_tokens": 1000},
    )
    assert mini["cost_usd"] == 0.002
    assert mini["cost_kind"] == "estimated_from_tokens"


def test_normalized_lifecycle_streams_rubrics_and_reward(monkeypatch) -> None:
    monkeypatch.setattr(healthbench, "load_row", lambda seed: _row() if seed == 7 else None)
    calls = iter(
        [
            {
                "text": "Rest, hydrate, use usual OTC medicine if safe, and seek urgent care for red flags.",
                "usage": healthbench._usage(
                    "groq", "llama-3.1-8b-instant", {"prompt_tokens": 20, "completion_tokens": 20}
                ),
            },
            {
                "text": '{"explanation":"Appropriate advice.","criteria_met":true}',
                "usage": healthbench._usage(
                    "openai", "gpt-4.1-2025-04-14", {"prompt_tokens": 30, "completion_tokens": 8}
                ),
            },
            {
                "text": '{"explanation":"No diagnosis was asserted.","criteria_met":false}',
                "usage": healthbench._usage(
                    "openai", "gpt-4.1-2025-04-14", {"prompt_tokens": 30, "completion_tokens": 8}
                ),
            },
        ]
    )
    monkeypatch.setattr(healthbench, "_chat", lambda config, messages: next(calls))
    client = TestClient(create_compat_app("healthbench_chat"))
    prepared = client.post(
        "/rollouts/prepare", json={"rollout_id": "hb-7", "telemetry": TELEMETRY}
    ).json()
    ready = client.get(prepared["stream"]["transports"]["poll"]["url"], params={"after": 0}).json()
    assert [event["kind"] for event in ready["events"]] == ["stream.subscribed"]
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "hb-7",
            "slot": "stream",
            "telemetry": TELEMETRY,
            "task_instance_id": "seed:7",
            "world_ref": "world:healthbench@eval",
            "evaluation_plan_ref": "healthbench_eval.v1",
            "policy_ref": {"harness": "chat_completion", "config": "groq_llama31_8b"},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get(
        prepared["stream"]["transports"]["poll"]["url"], params={"after": 0}
    ).json()["events"]
    assert [event["kind"] for event in events].count("rubric.grade") == 2
    assert "capture.closed" in [event["kind"] for event in events]
    reward = client.post("/reward", json={"rollout_id": "hb-7", "mode": "terminal"}).json()
    assert reward["reward"] == 1.0
    usage = started.json()["usage"]
    assert usage["cost_usd"] > 0
    assert usage["cost_kind"] == "estimated_from_tokens"


def test_synchronous_healthbench_rollouts_use_advertised_parallel_leases(monkeypatch) -> None:
    row = _row()
    row["rubrics"] = [row["rubrics"][0]]
    monkeypatch.setattr(healthbench, "load_row", lambda seed: row)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def chat(config, messages):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        if config["model"] == healthbench.GRADER_MODEL:
            text = '{"explanation":"Appropriate.","criteria_met":true}'
        else:
            text = "Rest and seek care for red flags."
        return {
            "text": text,
            "usage": healthbench._usage(
                config["provider"], config["model"], {"prompt_tokens": 1, "completion_tokens": 1}
            ),
        }

    monkeypatch.setattr(healthbench, "_chat", chat)
    client = TestClient(create_compat_app("healthbench_chat"))
    for seed in (0, 1):
        prepared = client.post(
            "/rollouts/prepare",
            json={"rollout_id": f"hb-parallel-{seed}", "telemetry": TELEMETRY},
        )
        assert prepared.status_code == 200, prepared.text

    def start(seed):
        return client.post(
            "/rollouts",
            json={
                "rollout_id": f"hb-parallel-{seed}",
                "slot": "stream",
                "telemetry": TELEMETRY,
                "task_instance_id": f"seed:{seed}",
                "policy_ref": {"harness": "chat_completion", "config": "groq_llama31_8b"},
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(start, (0, 1)))
    assert all(response.status_code == 200 for response in responses)
    assert max_active >= 2, "synchronous rollout execution was serialized by the platform lock"


def test_failed_grader_does_not_fabricate_zero_reward(monkeypatch) -> None:
    monkeypatch.setattr(healthbench, "load_row", lambda seed: _row())
    monkeypatch.setattr(
        healthbench,
        "_chat",
        lambda config, messages: (_ for _ in ()).throw(RuntimeError("provider_timeout")),
    )
    client = TestClient(create_compat_app("healthbench_chat"))
    client.post("/rollouts/prepare", json={"rollout_id": "hb-fail", "telemetry": TELEMETRY})
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "hb-fail",
            "slot": "stream",
            "telemetry": TELEMETRY,
            "task_instance_id": "seed:0",
            "policy_ref": {"harness": "chat_completion", "config": "groq_llama31_8b"},
        },
    )
    assert started.json()["usage"]["cost_usd"] is None
    reward = client.post("/reward", json={"rollout_id": "hb-fail", "mode": "terminal"}).json()
    assert reward["reward"] is None


def test_optimizer_adapter_computes_terminal_reward_and_preserves_runtime_status(
    monkeypatch,
) -> None:
    row = _row()
    row["rubrics"] = [row["rubrics"][0]]
    monkeypatch.setattr(healthbench, "load_row", lambda seed: row)
    calls = iter(
        [
            {
                "text": "Rest and seek care for red flags.",
                "usage": healthbench._usage(
                    "groq", "llama-3.1-8b-instant", {"prompt_tokens": 10, "completion_tokens": 5}
                ),
            },
            {
                "text": '{"explanation":"Appropriate.","criteria_met":true}',
                "usage": healthbench._usage(
                    "openai",
                    "gpt-4.1-mini-2025-04-14",
                    {"prompt_tokens": 20, "completion_tokens": 5},
                ),
            },
        ]
    )
    monkeypatch.setattr(healthbench, "_chat", lambda config, messages: next(calls))
    response = TestClient(create_compat_app("healthbench_chat")).post(
        "/rollout",
        json={
            "rollout_id": "hb-gepa-success",
            "task": {"seed": 0},
            "candidate": {"system_prompt": "Be safe."},
            "policy": {"provider": "groq", "model": "llama-3.1-8b-instant"},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["success_status"] == "success"
    assert body["reward_info"]["outcome_reward"] == 1.0
    assert body["summary"]["reward_status"] == "scored"
    assert body["usage"]["calls"] == 2
    assert body["usage"]["cost_source"] == "mixed_public_price_tables"


def test_optimizer_adapter_derives_openai_policy_connection_defaults(monkeypatch) -> None:
    row = _row()
    row["rubrics"] = [row["rubrics"][0]]
    monkeypatch.setattr(healthbench, "load_row", lambda seed: row)
    configs = []

    def chat(config, messages):
        configs.append(config)
        if len(configs) == 1:
            return {
                "text": "Rest and seek care for red flags.",
                "usage": healthbench._usage(
                    "openai", "gpt-4.1-mini-2025-04-14", {"prompt_tokens": 10, "completion_tokens": 5}
                ),
            }
        return {
            "text": '{"explanation":"Appropriate.","criteria_met":true}',
            "usage": healthbench._usage(
                "openai", healthbench.GRADER_MODEL, {"prompt_tokens": 20, "completion_tokens": 5}
            ),
        }

    monkeypatch.setattr(healthbench, "_chat", chat)
    response = TestClient(create_compat_app("healthbench_chat")).post(
        "/rollout",
        json={
            "rollout_id": "hb-gepa-openai-policy",
            "task": {"seed": 0},
            "candidate": {"system_prompt": "Be safe."},
            "policy": {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"},
        },
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert configs[0]["base_url"] == "https://api.openai.com/v1"
    assert configs[0]["api_key_env"] == "OPENAI_API_KEY"


def test_optimizer_retry_with_same_rollout_id_is_idempotent(monkeypatch) -> None:
    row = _row()
    row["rubrics"] = [row["rubrics"][0]]
    monkeypatch.setattr(healthbench, "load_row", lambda seed: row)
    calls = []

    def chat(config, messages):
        calls.append((config, messages))
        if len(calls) == 1:
            return {
                "text": "Rest and seek care for red flags.",
                "usage": healthbench._usage(
                    "groq",
                    "llama-3.1-8b-instant",
                    {"prompt_tokens": 10, "completion_tokens": 5},
                ),
            }
        if len(calls) == 2:
            return {
                "text": '{"explanation":"Appropriate.","criteria_met":true}',
                "usage": healthbench._usage(
                    "openai",
                    "gpt-4.1-mini-2025-04-14",
                    {"prompt_tokens": 20, "completion_tokens": 5},
                ),
            }
        raise AssertionError("idempotent retry reran paid provider calls")

    monkeypatch.setattr(healthbench, "_chat", chat)
    client = TestClient(create_compat_app("healthbench_chat"))
    request = {
        "rollout_id": "hb-gepa-idempotent",
        "task": {"seed": 0},
        "candidate": {"system_prompt": "Be safe."},
        "policy": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    }
    first = client.post("/rollout", json=request)
    retried = client.post("/rollout", json=request)
    assert first.status_code == retried.status_code == 200
    assert first.json()["reward_info"] == retried.json()["reward_info"]
    assert first.json()["stream"] == retried.json()["stream"]
    assert len(calls) == 2
