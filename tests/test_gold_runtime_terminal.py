from types import SimpleNamespace

from synth_containers.event_log import RolloutEventLog
from synth_containers.gold_runtime import GoldRuntime, _last_environment_step
from synth_containers.policies.nanohorizon import NanoHorizonSamplerFailure


def test_last_environment_step_uses_explicit_environment_events_only() -> None:
    log = RolloutEventLog(rollout_id="roll_failure", stream_id="stream:roll_failure")
    log.append("trace.opened", {"step": 99})
    log.append("action", {"step": 2, "action": "do"})
    log.append("reward_signal", {"step": 2, "value": 0.0})
    log.append("frame", {"step": 3, "digest": "frame-3"})
    log.append("status", {"status": "running", "steps": 100})

    assert _last_environment_step(log) == 3


def test_capability_exhaustion_closes_terminal_partial_journal(monkeypatch) -> None:
    runtime = GoldRuntime(
        environment_ref="env:craftax",
        task_payload=lambda _seed, _steps: {},
    )

    class Planner:
        def run(self, **_kwargs):
            raise NanoHorizonSamplerFailure(
                "workshop_capability_exhausted",
                completion={},
            )

        def usage(self):
            return {"calls": 7, "prompt_tokens": 100, "completion_tokens": 20}

    class World:
        closed = False

        def close(self):
            self.closed = True

    world = World()
    monkeypatch.setattr(GoldRuntime, "_world_for", lambda *_args, **_kwargs: world)
    monkeypatch.setattr(
        GoldRuntime,
        "_nanohorizon_planner",
        lambda *_args, **_kwargs: Planner(),
    )
    platform = SimpleNamespace(
        spec=SimpleNamespace(environment_ref="env:craftax", max_episode_steps=2000),
    )
    pin = SimpleNamespace(
        policy_ref={"harness": "nanohorizon", "config": "test"},
        resume_from_checkpoint_id=None,
        checkpoint_schedule=None,
        seed=780005,
        omit_reward=False,
        status="running",
        terminal=False,
        usage={},
    )
    log = RolloutEventLog(rollout_id="roll_cap", stream_id="stream:roll_cap")
    log.append("action", {"step": 3, "action": "do"})

    runtime.simulate(platform, pin, log)

    assert world.closed is True
    assert pin.status == "failed"
    assert pin.terminal is True
    assert pin.usage["calls"] == 7
    assert log.closed is True
    status = [row for row in log.after(0) if row.kind == "status"][-1]
    assert status.payload == {
        "status": "failed",
        "reason": "capability_exhausted",
        "detail": "workshop_capability_exhausted",
        "steps": 3,
        "error_type": "NanoHorizonSamplerFailure",
        "error": "workshop_capability_exhausted",
    }
    assert [row.kind for row in log.after(0)][-2:] == [
        "capture.high_water",
        "capture.closed",
    ]
