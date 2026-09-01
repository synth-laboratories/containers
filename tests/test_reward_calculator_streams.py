"""Reward calculation catalog and event streams: code and verifier families."""

from __future__ import annotations

from fastapi.testclient import TestClient

from synth_containers.event_log import RolloutEventLog
from synth_containers.platform import create_compat_app
from synth_containers.platform.reward import (
    REWARD_API_SCHEMA,
    REWARD_CLOSED,
    REWARD_OPENED,
    RewardStreamer,
    reward_api_catalog,
)
from synth_containers.platform.targets import RewardCalculatorFamily


TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}


def test_reward_catalog_distinguishes_code_and_verifier() -> None:
    code = reward_api_catalog(calculator=RewardCalculatorFamily.CODE, authority="environment")
    verifier = reward_api_catalog(
        calculator=RewardCalculatorFamily.VERIFIER,
        authority="healthbench_physician_rubric_grader",
    )
    assert code["schema"] == verifier["schema"] == REWARD_API_SCHEMA
    assert code["calculator"] == "code"
    assert verifier["calculator"] == "verifier"
    assert code["rewritten_by_annotations"] is False
    assert "rubric.grade" in verifier["event_kinds"]
    assert "rubric.grade" not in code["event_kinds"]
    paths = {item["path"] for item in code["endpoints"]}
    assert "/reward/catalog" in paths
    assert "/rollouts/{rollout_id}/reward/stream" in paths


def test_code_reward_streamer_emits_opened_signal_closed() -> None:
    log = RolloutEventLog(rollout_id="roll_code", stream_id="stream_code")
    streamer = RewardStreamer.code(log, authority="environment", kind="classification_accuracy")
    streamer.opened()
    streamer.signal(value=1.0)
    streamer.closed()
    kinds = [item.kind for item in log.after(0)]
    assert kinds == [REWARD_OPENED, "reward_signal", REWARD_CLOSED]
    signal = next(item for item in log.after(0) if item.kind == "reward_signal")
    assert signal.payload["value"] == 1.0
    assert signal.payload["calculator"] == "code"
    assert signal.payload["authority"] == "environment"


def test_verifier_reward_streamer_emits_rubric_grades_then_signal() -> None:
    log = RolloutEventLog(rollout_id="roll_ver", stream_id="stream_ver")
    streamer = RewardStreamer.verifier(
        log,
        authority="healthbench_physician_rubric_grader",
        kind="healthbench_overall_score",
        plan_ref="healthbench_eval.v1",
    )
    streamer.opened()
    streamer.evaluator_opened({"model": "gpt-4.1-2025-04-14"})
    streamer.grade({"index": 0, "criteria_met": True, "points": 2})
    streamer.grade({"index": 1, "criteria_met": False, "points": -1})
    streamer.evaluator_closed({"status": "completed"})
    streamer.signal(value=1.0)
    streamer.closed()
    kinds = [item.kind for item in log.after(0)]
    assert kinds.count("rubric.grade") == 2
    assert kinds[0] == REWARD_OPENED
    assert kinds[-1] == REWARD_CLOSED
    assert next(item for item in log.after(0) if item.kind == "reward_signal").payload["calculator"] == "verifier"


def test_openenv_advertises_code_calculator_and_streams_reward_events() -> None:
    client = TestClient(create_compat_app("openenv_echo"))
    info = client.get("/info").json()
    assert info["reward_calculator"] == "code"
    assert info["reward_authority"] == "environment"
    assert info["reward_api"]["schema"] == REWARD_API_SCHEMA
    catalog = client.get("/reward/catalog").json()
    assert catalog["calculator"] == "code"

    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": "roll_reward_stream", "telemetry": TELEMETRY},
    )
    assert prepared.status_code == 200, prepared.text
    stream = prepared.json()["stream"]
    assert stream["reward"]["events"] == "/rollouts/roll_reward_stream/reward/events"
    assert stream["reward"]["stream"] == "/rollouts/roll_reward_stream/reward/stream"
    started = client.post(
        "/rollouts",
        json={
            "rollout_id": "roll_reward_stream",
            "telemetry": TELEMETRY,
            "policy_ref": {"harness": "gym_loop", "config": "echo"},
        },
    )
    assert started.status_code == 200, started.text
    events = client.get("/rollouts/roll_reward_stream/reward/events", params={"after": 0}).json()["events"]
    kinds = [row["kind"] for row in events if not row.get("control")]
    assert kinds[0] == REWARD_OPENED
    assert "reward_signal" in kinds
    assert kinds[-1] == REWARD_CLOSED
    assert all(
        row["kind"] in {REWARD_OPENED, "reward_signal", REWARD_CLOSED, "stream.subscribed"}
        or row.get("control")
        for row in events
    )
    scored = client.post("/reward", json={"rollout_id": "roll_reward_stream", "mode": "terminal"}).json()
    assert scored["status"] == "scored"
    assert scored["reward"] is not None

    with client.stream("GET", stream["reward"]["stream"]) as response:
        text = "".join(response.iter_text())
    assert response.status_code == 200
    assert "event: reward.calculation.opened" in text
    assert "event: reward_signal" in text
    assert "event: reward.calculation.closed" in text
