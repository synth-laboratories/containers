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
