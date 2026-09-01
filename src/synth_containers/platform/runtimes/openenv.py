"""OpenEnv gym-style wrap. Observation / action / env reward. Not a fold.

World selection is ``environment_ref`` (``env:echo``), not ``target_id``.
``true_checkpoint`` stays unsupported. ``state()`` is not a checkpoint.
``/reward`` is environment authority (env-sum of RewardSignals). This wrap
does not write Harbor ``reward.txt`` or script nodes.

This cut is an in-process Echo-shaped gym world. It is not an unmodified
Echo image (A7).
"""

from __future__ import annotations

from typing import Any

from ...event_log import RolloutEventLog
from ..echo_world import ECHO_ENVIRONMENT, EchoWorld
from ..reward import RewardStreamer
from ..state import CompatPlatform, RolloutPin


_EMPTY_USAGE = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
}


class OpenEnvRuntime:
    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        world = _world_for(platform)
        seed = int(pin.seed or 0)
        harness = str(pin.policy_ref.get("harness") or "").strip()
        if not harness:
            raise ValueError(
                "OpenEnv simulate requires policy_ref.harness; start must not fill a default"
            )

        log.append(
            "env.episode.opened",
            {"seed": seed, "environment_ref": pin.environment_ref},
        )
        opened = world.reset(seed)
        prompt = str(opened.observation.get("text") or "")
        log.append(
            "observation",
            {"text": prompt, "obs": {"text": prompt}, "seed": seed},
        )

        emit_spans = harness == "gym_loop"
        if emit_spans:
            log.append(
                "span.policy.opened",
                {"harness": harness, "config": pin.policy_ref.get("config")},
            )
        action = self._act(platform, pin, opened)
        if emit_spans:
            log.append("span.policy.closed", {"status": "completed"})

        result = world.step(action)
        platform.step_calls += 1
        log.append("action", {"action": action})

        value: float | None = None if pin.omit_reward else result.reward
        reward = RewardStreamer.code(log, authority="environment", kind="env_sum")
        reward.opened()
        reward.signal(value=value)
        reward.closed()
        pin.reward_signals = [value]
        pin.status = "completed"
        pin.terminal = True
        pin.usage = dict(_EMPTY_USAGE)
        log.append(
            "env.episode.closed",
            {"status": "completed", "steps": result.env_steps},
        )
        log.append("status", {"status": "completed"})
        self._seal_capture(log)

    def _act(
        self,
        platform: CompatPlatform,
        pin: RolloutPin,
        opened: Any,
    ) -> str:
        prompt = str(opened.valid_action or opened.observation.get("text") or "")
        config_id = str(pin.policy_ref.get("config") or "").strip()
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        forced = config.get("forced_action")
        if isinstance(forced, str):
            return forced
        harness = str(pin.policy_ref.get("harness") or "").strip()
        if harness != "gym_loop":
            raise ValueError(f"unknown_openenv_harness:{harness}")
        return prompt

    def _seal_capture(self, log: RolloutEventLog) -> None:
        evidence_high_water = log.high_water
        log.append("capture.high_water", {"high_water": evidence_high_water})
        log.append("capture.closed", {"high_water": evidence_high_water})
        log.mark_closed()


def _world_for(platform: CompatPlatform) -> EchoWorld:
    ref = platform.spec.environment_ref
    if ref == ECHO_ENVIRONMENT:
        return EchoWorld()
    raise ValueError(f"unknown_openenv_environment:{ref}")
