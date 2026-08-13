"""Unit tests for HTTP-boundary parsers and the plan-outcome classifier."""

from __future__ import annotations

import pytest

from synth_containers.platform.http_requests import (
    RequestParseError,
    parse_create_rollout,
    parse_reward_post,
    to_platform_dict,
)
from synth_containers.platform.reward_plan import PlanOutcome, classify_plan_outcome


def test_classify_plan_outcome_gated_refused_scored() -> None:
    assert classify_plan_outcome("eval:gated") is PlanOutcome.GATED
    assert classify_plan_outcome("eval:craftax.env_sum.gated") is PlanOutcome.GATED
    assert classify_plan_outcome("eval:refused") is PlanOutcome.REFUSED
    assert classify_plan_outcome("eval:craftax.env_sum") is PlanOutcome.SCORED
    assert classify_plan_outcome("eval:deo.heldout_gate") is PlanOutcome.SCORED


def test_parse_create_rollout_accepts_auto_and_defaults_slot() -> None:
    req = parse_create_rollout(
        {
            "telemetry": {"enabled": True, "transport": "auto"},
            "policy_ref": {"harness": "react", "config": "luna_med"},
        }
    )
    assert req.telemetry.transport == "auto"
    assert req.slot == "stream"
    payload = to_platform_dict(req)
    assert payload["slot"] == "stream"
    assert payload["stream_slot"] == "stream"
    assert payload["telemetry"]["transport"] == "auto"
    assert "retention" in payload["telemetry"]
    assert payload["policy_ref"]["harness"] == "react"
    assert payload["policy_ref"]["config"] == "luna_med"


def test_parse_create_rollout_refuses_silent_policy_pin() -> None:
    with pytest.raises(RequestParseError, match="policy_ref.harness") as missing:
        parse_create_rollout({"telemetry": {"enabled": True, "transport": "sse"}})
    assert missing.value.status_code == 422
    with pytest.raises(RequestParseError, match="policy_ref.config") as no_config:
        parse_create_rollout(
            {
                "telemetry": {"enabled": True, "transport": "sse"},
                "policy_ref": {"harness": "react"},
            }
        )
    assert no_config.value.status_code == 422
    isolated = parse_create_rollout(
        {
            "telemetry": {"enabled": True, "transport": "sse"},
            "policy_ref": {"harness": "isolated_policy_process"},
        }
    )
    assert isolated.policy_ref.harness == "isolated_policy_process"
    assert isolated.policy_ref.config is None


def test_parse_create_rollout_rejects_unknown_transport() -> None:
    with pytest.raises(RequestParseError, match="POST /rollouts: telemetry.transport") as caught:
        parse_create_rollout({"telemetry": {"enabled": True, "transport": "ftp"}})
    assert caught.value.status_code == 422
    assert "ftp" in str(caught.value)


def test_parse_reward_post_xor() -> None:
    with pytest.raises(RequestParseError, match="exactly one of rollout_id or evidence"):
        parse_reward_post({})
    with pytest.raises(RequestParseError, match="exactly one of rollout_id or evidence"):
        parse_reward_post({"rollout_id": "r1", "evidence": {"reward": 1}})
    req = parse_reward_post({"rollout_id": "r1"})
    assert req.mode == "terminal"
    assert req.rescore is False
