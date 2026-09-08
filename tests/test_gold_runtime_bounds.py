from types import SimpleNamespace

from synth_containers.gold_runtime import GoldRuntime


def _runtime() -> GoldRuntime:
    return GoldRuntime(environment_ref="env:test", task_payload=lambda seed, steps: {})


def test_request_step_ceiling_tightens_target_default(monkeypatch) -> None:
    monkeypatch.delenv("SYNTH_GOLD_MAX_STEPS", raising=False)
    platform = SimpleNamespace(spec=SimpleNamespace(max_episode_steps=120))
    pin = SimpleNamespace(max_steps=40)
    assert _runtime()._max_steps(platform, pin) == 40


def test_request_step_ceiling_cannot_expand_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("SYNTH_GOLD_MAX_STEPS", "24")
    platform = SimpleNamespace(spec=SimpleNamespace(max_episode_steps=120))
    pin = SimpleNamespace(max_steps=40)
    assert _runtime()._max_steps(platform, pin) == 24


def test_repeated_invalid_plans_close_failed_instead_of_spinning():
    from synth_containers.gold_episode import run_episode
    from synth_containers.gold_http import StepResult
    from synth_containers.event_log import RolloutEventLog
    result = StepResult({}, None, False, ["move"], "", "", 0)
    world = SimpleNamespace(reset=lambda *args, **kw: result)
    class Planner:
        calls = 0
        def plan(self, observation, on_delta=None):
            self.calls += 1
            return ["illegal"]
        def metadata(self): return {}
        def usage(self): return {"calls": self.calls}
    planner = Planner()
    log = RolloutEventLog("bounded", "stream:bounded")
    run_episode(world=world, planner=planner, log=log, seed=0, max_steps=12)
    assert planner.calls == 8
    assert log.closed
    status = next(e for e in log.after(0) if e.kind == "status")
    assert status.payload["status"] == "failed"
    assert status.payload["reason"] == "policy_no_progress"
