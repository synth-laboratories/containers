from synth_containers.event_log import RolloutEventLog
from synth_containers.gold_runtime import _last_environment_step


def test_last_environment_step_uses_explicit_environment_events_only() -> None:
    log = RolloutEventLog(rollout_id="roll_failure", stream_id="stream:roll_failure")
    log.append("trace.opened", {"step": 99})
    log.append("action", {"step": 2, "action": "do"})
    log.append("reward_signal", {"step": 2, "value": 0.0})
    log.append("frame", {"step": 3, "digest": "frame-3"})
    log.append("status", {"status": "running", "steps": 100})

    assert _last_environment_step(log) == 3
