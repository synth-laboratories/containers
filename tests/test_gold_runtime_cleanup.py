"""GoldRuntime must return server-side rollout capacity on every terminal path."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import synth_containers.gold_runtime as gold_runtime_module
from synth_containers.event_log import RolloutEventLog
from synth_containers.gold_runtime import GoldRuntime
from synth_containers.platform.state import PolicyConfig, RolloutPin


class FakeWorld:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePlanner:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.closed = False
        self.close_error = close_error

    def usage(self) -> dict[str, int]:
        return {"calls": 1}

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _pin() -> RolloutPin:
    return RolloutPin(
        rollout_id="rollout-cleanup",
        world_ref="world",
        environment_ref="craftax",
        policy_ref={"harness": "test", "config": "policy"},
        evaluation_plan_ref="plan",
        task_instance_id="task:0",
        stream_id="stream",
        engine_generation=1,
        policy_revision_id=None,
        seed=0,
    )


def _platform() -> Any:
    return SimpleNamespace(
        spec=SimpleNamespace(environment_ref="craftax", max_episode_steps=1),
        policy_configs={
            "policy": PolicyConfig(config_id="policy", harness="test", config={})
        },
        checkpoints={},
        step_calls=0,
    )


def _runtime_with(
    monkeypatch: pytest.MonkeyPatch, world: FakeWorld, planner: FakePlanner
) -> GoldRuntime:
    runtime = GoldRuntime(environment_ref="craftax", task_payload=lambda seed, steps: {})
    monkeypatch.setattr(GoldRuntime, "_world_for", lambda *_args, **_kwargs: world)
    monkeypatch.setattr(gold_runtime_module, "build_planner", lambda *_args, **_kwargs: planner)
    return runtime


def test_policy_failure_closes_world_and_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    world = FakeWorld()
    planner = FakePlanner()
    runtime = _runtime_with(monkeypatch, world, planner)
    monkeypatch.setattr(
        gold_runtime_module,
        "run_episode",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("policy failed")),
    )
    pin = _pin()
    log = RolloutEventLog(pin.rollout_id, pin.stream_id)

    runtime.simulate(_platform(), pin, log)

    assert world.closed is True
    assert planner.closed is True
    assert pin.status == "failed"
    assert log.closed is True


def test_world_closes_even_when_planner_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    world = FakeWorld()
    planner = FakePlanner(close_error=RuntimeError("planner close failed"))
    runtime = _runtime_with(monkeypatch, world, planner)
    monkeypatch.setattr(
        gold_runtime_module,
        "run_episode",
        lambda **_kwargs: {
            "steps": 0,
            "reward_signals": [],
            "usage": {"calls": 0},
            "frame_digest": "digest",
            "frames": [],
        },
    )

    with pytest.raises(RuntimeError, match="planner close failed"):
        runtime.simulate(
            _platform(),
            _pin(),
            RolloutEventLog("rollout-cleanup", "stream"),
        )

    assert world.closed is True
    assert planner.closed is True
