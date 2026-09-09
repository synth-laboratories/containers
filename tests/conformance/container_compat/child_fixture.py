"""The nested child a `deo_nested` rollout exposes as a first-class resource.

C4-06 asserts a DEO parent publishes a child rollout that can be scored on its
own, with its own frames, and that the parent's reward is the held-out gate
rather than a copy of the child's env-sum. The child has to be a real rollout
for any of that to be checkable.

The platform is image-agnostic: it holds no game. So the suite brings its own
minimal child — enough to emit a frame, a reward signal, and a terminal seal —
instead of reaching into an evals image for one.
"""

from __future__ import annotations

import hashlib

from typing import Any

from synth_containers.event_log import RolloutEventLog
from synth_containers.platform.state import CompatPlatform, RolloutPin
from synth_containers.platform.targets import (
    PolicySeed,
    RewardKind,
    ScriptNode,
    TargetRuntimeKind,
    TargetSpec,
)
from synth_containers.platform.affordances import AffordanceMap

# A 1x1 PNG. The frame only has to be a real image; nothing reads its pixels.
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

CHILD_STEPS = 3
CHILD_STEP_REWARD = 0.25


class _ChildRuntime:
    """Three steps, one frame each, a fixed env-sum reward."""

    def simulate(
        self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog
    ) -> None:
        log.append("env.episode.opened", {"seed": int(pin.seed or 0), "max_steps": CHILD_STEPS})
        signals: list[float] = []
        digest = hashlib.sha256(PNG_1X1).hexdigest()[:16]
        for step in range(1, CHILD_STEPS + 1):
            platform.artifacts[digest] = {
                "digest": digest,
                "retention": platform.spec.retention,
                "kind": "frame",
                "rollout_id": pin.rollout_id,
                "format": "png",
                "bytes": PNG_1X1,
            }
            log.append(
                "frame",
                {"step": step, "format": "png", "digest": digest, "url": f"/artifacts/{digest}"},
            )
            log.append("action", {"step": step, "action": "noop"})
            signals.append(CHILD_STEP_REWARD)
            log.append(
                "reward_signal",
                {"step": step, "value": CHILD_STEP_REWARD, "authority": "environment"},
            )
        pin.reward_signals = list(signals)
        log.append("terminal", {"status": "completed"})
        log.append("env.episode.closed", {"status": "completed", "steps": CHILD_STEPS})
        log.append("status", {"status": "completed"})
        high_water = log.high_water
        log.append("capture.high_water", {"high_water": high_water})
        log.append("capture.closed", {"high_water": high_water})
        log.mark_closed()
        pin.status = "completed"
        pin.terminal = True


def _env(items: dict[str, str]) -> AffordanceMap:
    return AffordanceMap(by_role={"environment": dict(items), "policy": {}, "evaluator": {}})


NESTED_CHILD = TargetSpec(
    target_id="deo_nested_child",
    runtime_family=TargetRuntimeKind.EXTERNAL,
    adapter_chain=(),
    world_ref="world:conformance_child",
    environment_ref="env:conformance_child",
    evaluation_plan_ref="eval:child.env_sum",
    default_policy_harness="isolated_policy_process",
    scale_leases=1,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=False,
    live_frames="native",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "env.episode.opened",
        "frame",
        "action",
        "reward_signal",
        "terminal",
        "status",
    ),
    script_node=ScriptNode.REWARD_TXT,
    max_episode_steps=CHILD_STEPS,
    affordances=_env(
        {
            "step": "unsupported",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_frames": "native",
            "true_checkpoint": "unsupported",
            "update_policy_code": "native",
        }
    ),
    policy_seeds=(PolicySeed("heuristic", "isolated_policy_process", {}),),
    runtime=_ChildRuntime(),
)


def runtime_config_for(target: str) -> dict[str, Any] | None:
    """The child spec a `deo_nested` app needs; nothing for any other target."""

    return {"nested_child_spec": NESTED_CHILD} if target == "deo_nested" else None
