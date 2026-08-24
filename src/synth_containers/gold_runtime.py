"""Generic gold-engine runtime. World is a baked engine only; no fixture worlds.

``GoldRuntime`` is the ``TargetSpec.runtime`` for every image whose environment
is a gold HTTP engine baked in as a child of PID 1. The game-specific parts are
constructor arguments — the environment ref, the engine's URL env var, the task
payload builder, and the step-budget env override. Everything else (policy
binding, checkpoint schedules, frame artifacts, achievement labels, failure
sealing) is identical across games and lives here once.

The policy half of the split is resolved by ``policies.build_planner``, except
two harnesses that need PUT ``policy.py``: ``isolated_policy_process`` (code
only) and ``nanohorizon`` (code + sampler config). Neither lives in the
factory map.
"""

from __future__ import annotations

import os
from typing import Any

from collections.abc import Callable
from dataclasses import dataclass

from .event_log import RolloutEventLog
from .gold_episode import PNG_MAGIC, run_episode
from .gold_http import GoldHttpWorld
from .platform.policy_process import DEFAULT_HEURISTIC, IsolatedPolicyProcess
from .platform.state import CompatPlatform, RolloutPin
from .policies import NANOHORIZON_HARNESS, build_planner
from .policies.nanohorizon import build_planner as build_nanohorizon


@dataclass(frozen=True)
class GoldRuntime:
    """``TargetSpec.runtime`` for a baked gold HTTP engine.

    ``task_payload(seed, max_steps)`` is the only game-specific input.
    """

    environment_ref: str
    task_payload: Callable[[int, int], dict[str, Any]]
    url_env: str = "SYNTH_GOLD_URL"
    engine: str = "gold"
    max_steps_env: str = "SYNTH_GOLD_MAX_STEPS"
    frame_path: str = "/rollouts/{rollout_id}/frames/{env_steps}.png"

    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        max_steps = self._max_steps(platform)
        world = self._world_for(platform, max_steps=max_steps)
        harness = str(pin.policy_ref.get("harness") or "").strip()
        if not harness:
            raise ValueError(
                "simulate requires policy_ref.harness; start must not fill a default"
            )
        closer = None
        if harness == "isolated_policy_process":
            planner, closer = self._code_policy_planner(platform, pin)
        elif harness == NANOHORIZON_HARNESS:
            planner = self._nanohorizon_planner(platform, pin)
            closer = getattr(planner, "close", None)
        else:
            config_id = str(pin.policy_ref.get("config") or "").strip()
            if not config_id:
                raise ValueError(
                    "simulate requires policy_ref.config; start must not default a model"
                )
            policy = platform.policy_configs.get(config_id)
            config = dict(policy.config) if policy is not None else {}
            if platform.spec.environment_ref != self.environment_ref:
                raise ValueError(f"unknown_environment:{platform.spec.environment_ref}")
            config.setdefault("env_name", self.engine)
            planner = build_planner(harness, config_id=config_id, config=config)
            closer = getattr(planner, "close", None)
        resume_checkpoint = None
        if pin.resume_from_checkpoint_id:
            resume_checkpoint = platform.checkpoints.get(pin.resume_from_checkpoint_id)
            if resume_checkpoint is None:
                raise RuntimeError(f"unknown_checkpoint:{pin.resume_from_checkpoint_id}")

        checkpoint_callback = None
        if pin.checkpoint_schedule is not None:
            mode = str(pin.checkpoint_schedule.get("mode") or "").strip()
            if mode != "per_policy_call":
                raise RuntimeError(f"unsupported_checkpoint_schedule:{mode or 'missing'}")
            prefix = str(pin.checkpoint_schedule.get("checkpoint_id_prefix") or "").strip()
            if not prefix:
                raise RuntimeError("checkpoint_schedule requires checkpoint_id_prefix")

            def checkpoint_callback(
                current_world: Any,
                current_planner: Any,
                result: Any,
                signals: list[float | None],
            ) -> dict[str, Any]:
                capture = getattr(current_world, "checkpoint", None)
                policy_capture = getattr(current_planner, "checkpoint_state", None)
                if not callable(capture) or not callable(policy_capture):
                    raise RuntimeError("runtime does not implement true checkpoint capture")
                environment = capture()
                checkpoint_id = f"{prefix}_{int(current_planner.usage().get('calls') or 0):04d}"
                reward = sum(float(value) for value in signals if isinstance(value, (int, float)))
                achievements = _achievement_labels(result.observation)
                platform.record_checkpoint(
                    {
                        "checkpoint_id": checkpoint_id,
                        "rollout_id": pin.rollout_id,
                        "environment_ref": pin.environment_ref,
                        "world_ref": pin.world_ref,
                        "policy_ref": {
                            key: value for key, value in pin.policy_ref.items() if key != "code"
                        },
                        "policy_llm_call_index": int(
                            current_planner.usage().get("calls") or 0
                        ),
                        "step": int(result.env_steps),
                        "reward": reward,
                        "achievements": achievements,
                        "environment_checkpoint_id": environment["checkpoint_id"],
                        "environment_blob": environment["blob"],
                        "policy_state": policy_capture(),
                        "parent_checkpoint_id": pin.resume_from_checkpoint_id,
                    }
                )
                descriptor = {
                    "checkpoint_id": checkpoint_id,
                    "rollout_id": pin.rollout_id,
                    "policy_llm_call_index": int(current_planner.usage().get("calls") or 0),
                    "step": int(result.env_steps),
                    "reward": reward,
                    "achievements": achievements,
                    "parent_checkpoint_id": pin.resume_from_checkpoint_id,
                    # A snapshot taken after the environment has terminated is
                    # valid terminal evidence, but it is not a state from
                    # which GoEx can continue. Never advertise terminal
                    # snapshots as resumable branches.
                    "restore_eligible": not bool(result.done),
                    "branchable": not bool(result.done),
                    "checkpoint_semantics": "true_environment_snapshot",
                    "resume_blockers": ["terminal_environment"] if result.done else [],
                }
                log.append("rollout.checkpoint", descriptor)
                return descriptor
        try:
            try:
                if harness == NANOHORIZON_HARNESS:
                    if resume_checkpoint is not None or checkpoint_callback is not None:
                        raise RuntimeError("nanohorizon harness does not resume gold checkpoints")
                    outcome = planner.run(
                        world=world,
                        log=log,
                        seed=int(pin.seed or 0),
                        max_steps=max_steps,
                        omit_reward=pin.omit_reward,
                    )
                else:
                    outcome = run_episode(
                        world=world,
                        planner=planner,
                        log=log,
                        seed=int(pin.seed or 0),
                        max_steps=max_steps,
                        omit_reward=pin.omit_reward,
                        emit_policy_spans=True,
                        resume_checkpoint=resume_checkpoint,
                        checkpoint_callback=checkpoint_callback,
                    )
            except Exception as exc:
                # The stream is the durable authority. A provider/configuration
                # failure after span.policy.opened must not leave the rollout
                # looking active forever merely because the HTTP start request
                # returned 500. Persist a secret-free terminal lifecycle and
                # let CompatPlatform seal it normally.
                if not log.closed:
                    log.append(
                        "span.policy.closed",
                        {"status": "failed", "error_type": type(exc).__name__},
                    )
                    log.append(
                        "policy.session.closed",
                        {"status": "failed", "calls": planner.usage().get("calls")},
                    )
                    log.append("env.episode.closed", {"status": "failed", "steps": 0})
                    log.append(
                        "status",
                        {
                            "status": "failed",
                            "reason": "policy_error",
                            "error_type": type(exc).__name__,
                        },
                    )
                    evidence_high_water = log.high_water
                    log.append("capture.high_water", {"high_water": evidence_high_water})
                    log.append("capture.closed", {"high_water": evidence_high_water})
                    log.mark_closed()
                pin.status = "failed"
                pin.terminal = True
                pin.usage = dict(planner.usage())
                return
        finally:
            if closer is not None:
                closer()
        platform.step_calls += int(outcome["steps"])
        pin.reward_signals = list(outcome["reward_signals"])
        pin.status = "completed"
        pin.terminal = True
        pin.usage = dict(outcome["usage"])
        pin.scheduled_checkpoints = list(outcome.get("scheduled_checkpoints") or [])
        digest = str(outcome["frame_digest"])
        records = list(outcome.get("frames") or [])
        if not records:
            records = _frames_from_log(platform, pin, log, fallback_digest=digest)
        for frame in records:
            _put_frame_artifact(platform, pin, frame, fallback_digest=digest)

    def _max_steps(self, platform: CompatPlatform) -> int:
        override = os.environ.get(self.max_steps_env)
        if override:
            return int(override)
        pinned = platform.spec.max_episode_steps
        if pinned is None or pinned <= 0:
            raise ValueError(
                "max_episode_steps must be pinned on the target spec; "
                "do not inherit the engine's silent world default"
            )
        return int(pinned)

    def _world_for(self, platform: CompatPlatform, *, max_steps: int) -> GoldHttpWorld:
        ref = platform.spec.environment_ref
        if ref != self.environment_ref:
            raise ValueError(f"unknown_environment:{ref}")
        return GoldHttpWorld(
            max_steps=max_steps,
            task_payload=self.task_payload,
            url_env=self.url_env,
            engine=self.engine,
            require_frames=platform.spec.live_frames == "native",
            frame_path=self.frame_path if platform.spec.live_frames == "native" else "",
        )

    def _nanohorizon_planner(self, platform: CompatPlatform, pin: RolloutPin) -> Any:
        config_id = str(pin.policy_ref.get("config") or "").strip()
        if not config_id:
            raise ValueError(
                "nanohorizon requires policy_ref.config; start must not default a sampler"
            )
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        if platform.spec.environment_ref != self.environment_ref:
            raise ValueError(f"unknown_environment:{platform.spec.environment_ref}")
        config.setdefault("env_name", self.engine)
        code = b""
        rev = platform.policy_revisions.get(str(pin.policy_revision_id or ""))
        if rev is not None and rev.code:
            code = rev.code
        if not code:
            raise RuntimeError("nanohorizon_missing_policy_code")
        return build_nanohorizon(config_id=config_id, config=config, code=code)

    def _code_policy_planner(self, platform: CompatPlatform, pin: RolloutPin) -> tuple[Any, Any]:
        code = DEFAULT_HEURISTIC.encode("utf-8")
        rev = platform.policy_revisions.get(str(pin.policy_revision_id or ""))
        if rev is not None and rev.code:
            code = rev.code
        process = IsolatedPolicyProcess(code)

        class _Planner:
            def __init__(self, proc: IsolatedPolicyProcess) -> None:
                self.proc = proc
                self.ply = 0

            def plan(self, observation: dict[str, Any], on_delta: Any = None) -> list[str]:
                del on_delta
                self.ply += 1
                return self.proc.choose(observation, ply=self.ply)

            def usage(self) -> dict[str, Any]:
                return {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "calls": self.ply,
                }

            def metadata(self) -> dict[str, Any]:
                return {"harness": "isolated_policy_process", "kind": "code_policy"}

        return _Planner(process), process.close


def _put_frame_artifact(
    platform: CompatPlatform,
    pin: RolloutPin,
    frame: dict[str, Any],
    *,
    fallback_digest: str,
) -> None:
    frame_digest = str(frame.get("digest") or fallback_digest)
    payload = frame.get("bytes")
    blob = bytes(payload) if isinstance(payload, (bytes, bytearray)) else b"ASCII"
    claimed = str(frame.get("format") or "ascii")
    fmt = "png" if claimed == "png" and blob.startswith(PNG_MAGIC) else "ascii"
    existing = platform.artifacts.get(frame_digest)
    if (
        existing is not None
        and existing.get("format") == "png"
        and bytes(existing.get("bytes") or b"").startswith(PNG_MAGIC)
        and fmt != "png"
    ):
        return
    platform.artifacts[frame_digest] = {
        "digest": frame_digest,
        "retention": platform.spec.retention,
        "kind": "frame",
        "rollout_id": pin.rollout_id,
        "format": fmt,
        "bytes": blob,
    }


def _frames_from_log(
    platform: CompatPlatform,
    pin: RolloutPin,
    log: RolloutEventLog,
    *,
    fallback_digest: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in log.after(0):
        if item.kind != "frame":
            continue
        payload = item.payload
        digest = str(payload.get("digest") or fallback_digest)
        claimed = str(payload.get("format") or "ascii")
        url = payload.get("url")
        blob = b"ASCII"
        fmt = "ascii"
        if claimed == "png" and url:
            step = int(payload.get("step") or 0)
            path = RolloutEventLog.frame_asset_path(platform.storage_root, pin.rollout_id, step)
            if path.is_file():
                blob = path.read_bytes()
                if blob.startswith(PNG_MAGIC):
                    fmt = "png"
        records.append({"digest": digest, "bytes": blob, "format": fmt})
    return records


def _achievement_labels(observation: dict[str, Any]) -> list[str]:
    labels: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if "achievement" in lowered:
            if isinstance(value, str) and value.strip():
                labels.add(value.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        labels.add(item.strip())
            elif isinstance(value, dict):
                for name, enabled in value.items():
                    if enabled and str(name).strip():
                        labels.add(str(name).strip())
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)

    visit(observation)
    return sorted(labels)


