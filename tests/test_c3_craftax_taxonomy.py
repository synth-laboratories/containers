"""C-3: Craftax accounting, taxonomy, gold pin, truncation, usage null-not-zero."""

from __future__ import annotations

import io
import urllib.error

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.craftax_taxonomy import (
    GOLD_URL_CONFIG_KEY,
    classify_action,
    classify_completion,
    usage_from_call_identity,
)
from synth_containers.platform.gold_craftax_world import (
    GoldConnectionError,
    GoldCraftaxWorld,
    GoldHTTPError,
)
from synth_containers.platform.react import OpenRouterReAct, ScriptedReAct


def test_usage_from_call_identity_never_fabricates_zero_tokens() -> None:
    omitted = usage_from_call_identity(
        calls=4,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        kind="model",
    )
    assert omitted["llm_calls"] == omitted["calls"] == 4
    assert omitted["prompt_tokens"] is None
    assert omitted["completion_tokens"] is None
    assert omitted["total_tokens"] is None
    assert omitted["cost_usd"] is None
    assert omitted["usage_status"] == "provider_omitted"
    reported = usage_from_call_identity(
        calls=2,
        prompt_tokens=10,
        completion_tokens=3,
        total_tokens=13,
        cost_usd=0.01,
        kind="model",
    )
    assert reported["usage_status"] == "reported"
    scripted = usage_from_call_identity(
        calls=3,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        kind="scripted",
    )
    assert scripted["usage_status"] == "not_applicable"
    assert scripted["prompt_tokens"] is None


def test_classify_action_outcomes() -> None:
    before = {"x": 2, "y": 2, "wood": 0, "valid_actions": ["north", "do", "noop"]}
    moved = classify_action(
        action="east",
        valid_actions=["north", "south", "east", "west", "do", "noop"],
        before=before,
        after={"x": 3, "y": 2, "wood": 0},
    )
    assert moved == {"class": "executed", "outcome": "moved"}
    blocked = classify_action(
        action="west",
        valid_actions=["west"],
        before={"x": 0, "y": 0},
        after={"x": 0, "y": 0},
    )
    assert blocked == {"class": "effective_noop", "outcome": "blocked"}
    harvested = classify_action(
        action="do",
        valid_actions=["do"],
        before={"x": 3, "y": 2, "wood": 0},
        after={"x": 3, "y": 2, "wood": 1},
    )
    assert harvested == {"class": "executed", "outcome": "harvested"}
    missing = classify_action(
        action="do",
        valid_actions=["do"],
        before={"x": 0, "y": 0, "wood": 0},
        after={"x": 0, "y": 0, "wood": 0},
    )
    assert missing == {"class": "infeasible", "outcome": "refused_missing_prerequisite"}
    invalid = classify_action(
        action="fly",
        valid_actions=["do"],
        before=before,
        after=before,
    )
    assert invalid == {"class": "syntactically_invalid", "outcome": "no_effect"}
    infeasible = classify_action(
        action="sleep",
        valid_actions=["do"],
        before=before,
        after=before,
    )
    assert infeasible == {"class": "infeasible", "outcome": "refused_missing_prerequisite"}


def test_classify_completion_kinds() -> None:
    assert classify_completion(terminated=True, truncated=False, env_steps=4, max_steps=8) == (
        "natural_completion"
    )
    assert classify_completion(terminated=False, truncated=True, env_steps=8, max_steps=8) == (
        "truncated"
    )
    assert classify_completion(
        terminated=False, truncated=False, env_steps=3, max_steps=8, infra_failed=True
    ) == "infra_complete"


def test_fixture_episode_emits_taxonomy_truncation_and_reconciled_usage(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "p0"))
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_c3_tax",
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "react", "config": "luna_med"},
            "task_instance_id": "seed:0",
        },
    )
    assert started.status_code == 200, started.text
    body = started.json()
    usage = body["usage"]
    assert usage["llm_calls"] == usage["calls"]
    assert usage["llm_calls"] >= 1
    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["usage_status"] == "not_applicable"
    events = client.get("/rollouts/roll_c3_tax/events", params={"after": 0}).json()["events"]
    actions = [item for item in events if item["kind"] == "action"]
    assert actions
    assert {item["payload"]["class"] for item in actions} <= {
        "syntactically_invalid",
        "infeasible",
        "effective_noop",
        "executed",
    }
    assert {item["payload"]["outcome"] for item in actions} <= {
        "moved",
        "blocked",
        "harvested",
        "crafted",
        "refused_missing_prerequisite",
        "no_effect",
    }
    truncated = [item for item in events if item["kind"] == "plan.truncated"]
    assert truncated
    payload = truncated[0]["payload"]
    assert payload["declared"] == payload["accepted"] + payload["dropped"]
    assert payload["dropped"] >= 1
    closed = next(item for item in events if item["kind"] == "env.episode.closed")
    assert closed["payload"]["completion"] in {"natural_completion", "truncated"}
    assert closed["payload"]["truncated"] is (closed["payload"]["completion"] == "truncated")
    assert closed["payload"]["infra_complete"] is False
    reconciled = next(item for item in events if item["kind"] == "usage.reconciled")
    assert reconciled["payload"]["llm_calls"] == usage["llm_calls"]
    assert reconciled["payload"]["llm_call_events"] == usage["llm_call_events"]
    assert reconciled["payload"]["prompt_tokens"] is None


def test_rollout_max_steps_is_an_enforced_identity_pin(tmp_path) -> None:
    client = TestClient(create_compat_app("craftax_engine", storage_root=tmp_path / "cap"))
    body = {
        "rollout_id": "roll_bounded",
        "max_steps": 3,
        "telemetry": {"enabled": True, "transport": "sse"},
        "policy_ref": {"harness": "react", "config": "luna_med"},
        "task_instance_id": "seed:0",
    }
    completed = client.post("/rollouts", json=body)
    assert completed.status_code == 200, completed.text
    events = client.get("/rollouts/roll_bounded/events", params={"after": 0}).json()[
        "events"
    ]
    closed = next(item for item in events if item["kind"] == "env.episode.closed")
    assert closed["payload"]["steps"] == 3
    assert closed["payload"]["truncated"] is True

    conflict = client.post("/rollouts", json={**body, "max_steps": 4})
    assert conflict.status_code == 409
    assert conflict.json()["error"] == "rollout_identity_conflict"

    rejected = client.post("/rollouts", json={**body, "rollout_id": "roll_invalid", "max_steps": 0})
    assert rejected.status_code == 422
    assert "max_steps must be a positive integer" in rejected.text


def test_openrouter_omitted_usage_is_null_not_zero() -> None:
    planner = OpenRouterReAct(
        config_id="luna_med",
        config={
            "model": "meta/muse-spark-1.1",
            "effort": "medium",
            "max_tokens": 1024,
            "context_token_budget": 16000,
            "compact_at": 0.7,
            "keep_recent_messages": 8,
            "keep_recent_frames": 2,
            "observation_mode": "text",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "parse_retries": 0,
            "system_prompt": "Choose the best valid Craftax action.",
        },
    )
    planner.calls = 3
    usage = planner.usage()
    assert usage["llm_calls"] == 3
    assert usage["usage_status"] == "provider_omitted"
    assert usage["prompt_tokens"] is None
    assert 0 not in (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
    scripted = ScriptedReAct(config_id="luna_med")
    scripted.calls = 2
    assert scripted.usage()["usage_status"] == "not_applicable"


def test_gold_connection_error_names_url_and_config_key() -> None:
    world = GoldCraftaxWorld(max_steps=1, base_url="http://127.0.0.1:1", require_frames=False)
    try:
        world.reset(0)
    except GoldConnectionError as exc:
        assert exc.config_key == GOLD_URL_CONFIG_KEY
        assert "http://127.0.0.1:1/rollouts" in exc.attempted_url
        assert GOLD_URL_CONFIG_KEY in str(exc)
        assert "http://127.0.0.1:1/rollouts" in str(exc)
    else:
        raise AssertionError("expected GoldConnectionError")


def test_gold_http_status_is_an_environment_contract_error(monkeypatch) -> None:
    def reject(request, timeout):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            request.full_url,
            422,
            "wrong service contract",
            {},
            io.BytesIO(b'{"detail":"do not persist provider-shaped bodies"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", reject)
    world = GoldCraftaxWorld(
        max_steps=1,
        base_url="http://127.0.0.1:8800",
        require_frames=False,
    )
    try:
        world.reset(0)
    except GoldHTTPError as exc:
        assert isinstance(exc, GoldConnectionError)
        assert exc.status_code == 422
        assert exc.attempted_url == "http://127.0.0.1:8800/rollouts"
        assert "do not persist" not in str(exc)
    else:
        raise AssertionError("expected GoldHTTPError")
