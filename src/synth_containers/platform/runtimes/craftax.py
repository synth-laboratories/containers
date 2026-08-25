"""Craftax world + ReAct / isolated policy episode.

World selection is `environment_ref` (fixture vs gold HTTP), not `target_id`.
max_episode_steps is pinned on the TargetSpec; SYNTH_CRAFTAX_MAX_STEPS may override.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from ...event_log import RolloutEventLog
from ..craftax_taxonomy import GOLD_URL_CONFIG_KEY, usage_from_call_identity
from ..craftax_world import CraftaxWorld
from ..episode import PNG_MAGIC, run_episode
from ..policy_process import DEFAULT_HEURISTIC, IsolatedPolicyProcess
from ..gold_craftax_world import GoldConnectionError, GoldCraftaxWorld
from ..react import OpenRouterReAct, ScriptedReAct
from ..state import CompatPlatform, RolloutPin

FIXTURE_ENVIRONMENT = "env:craftax_fixture"
GOLD_ENVIRONMENT = "env:craftax_gold"


class CraftaxRuntime:
    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        max_steps = _max_steps(platform, pin)
        world = _world_for(platform, max_steps=max_steps)
        harness = str(pin.policy_ref.get("harness") or "").strip()
        if not harness:
            raise ValueError(
                "Craftax simulate requires policy_ref.harness; start must not fill a default"
            )
        closer = None
        if harness == "isolated_policy_process":
            planner, closer = self._code_policy_planner(platform, pin)
        else:
            config_id = str(pin.policy_ref.get("config") or "").strip()
            if not config_id:
                raise ValueError(
                    "Craftax simulate requires policy_ref.config; start must not default luna_med"
                )
            policy = platform.policy_for(pin)
            config = dict(policy.config) if policy is not None else {}
            hosted_sampler = bool(
                config.get("training_sampler_endpoint") or config.get("inference_target")
            )
            planner = (
                OpenRouterReAct(config_id=config_id, config=config)
                if platform.spec.environment_ref == GOLD_ENVIRONMENT or hosted_sampler
                else ScriptedReAct(config_id=config_id)
            )
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
                # The event log is durable and shippable, so it carries the
                # exception *class* only — a provider SDK may embed a key in a
                # message. But the class alone is not diagnosable: a missing
                # API key, a sampler path that is not yet servable and a
                # malformed response all arrive as RuntimeError and each needs
                # a different fix. Put the detail on stderr, which is
                # process-local and never shipped.
                traceback.print_exc()
                # The stream is the durable authority. A provider/configuration
                # failure after span.policy.opened must not leave the rollout
                # looking active forever merely because the HTTP start request
                # returned 500. Persist a secret-free terminal lifecycle and
                # let CompatPlatform seal it normally.
                attempted_url = None
                config_key = None
                if isinstance(exc, GoldConnectionError):
                    attempted_url = exc.attempted_url
                    config_key = exc.config_key
                log.append(
                    "span.policy.closed",
                    {"status": "failed", "error_type": type(exc).__name__},
                )
                log.append(
                    "policy.session.closed",
                    {"status": "failed", "calls": planner.usage().get("calls")},
                )
                log.append(
                    "env.episode.closed",
                    {
                        "status": "failed",
                        "steps": 0,
                        "completion": "infra_complete",
                        "natural_completion": False,
                        "truncated": False,
                        "infra_complete": True,
                        # An unreachable world dies in reset(), before the policy
                        # is called. Calling that `policy_error` is what sent two
                        # diagnoses chasing the model instead of the address.
                        "reason": (
                            "environment_error"
                            if isinstance(exc, GoldConnectionError)
                            else "policy_error"
                        ),
                        "error_type": type(exc).__name__,
                    },
                )
                status_payload = {
                    "status": "failed",
                    "reason": "policy_error" if not isinstance(exc, GoldConnectionError) else "gold_connection",
                    "error_type": type(exc).__name__,
                    "completion": "infra_complete",
                }
                if attempted_url is not None:
                    status_payload["attempted_url"] = attempted_url
                    status_payload["config_key"] = config_key
                    status_code = getattr(exc, "status_code", None)
                    if isinstance(status_code, int):
                        status_payload["status_code"] = status_code
                log.append("status", status_payload)
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
                return usage_from_call_identity(
                    calls=self.ply,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    kind="scripted",
                )

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


def _max_steps(platform: CompatPlatform, pin: RolloutPin) -> int:
    if pin.max_steps is not None:
        if pin.max_steps <= 0:
            raise ValueError("rollout max_steps must be a positive immutable pin")
        return int(pin.max_steps)
    override = os.environ.get("SYNTH_CRAFTAX_MAX_STEPS")
    if override:
        return int(override)
    pinned = platform.spec.max_episode_steps
    if pinned is None or pinned <= 0:
        raise ValueError(
            "Craftax max_episode_steps must be pinned on the target spec; "
            "do not inherit gold's silent 120-step world default"
        )
    return int(pinned)


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


def _world_for(platform: CompatPlatform, *, max_steps: int) -> CraftaxWorld | GoldCraftaxWorld:
    ref = platform.spec.environment_ref
    if ref == GOLD_ENVIRONMENT:
        pinned = (
            str(platform.runtime_config.get(GOLD_URL_CONFIG_KEY) or "").strip()
            or str(getattr(platform.spec, "gold_base_url", "") or "").strip()
        )
        if not pinned:
            raise GoldConnectionError(
                attempted_url="",
                config_key=GOLD_URL_CONFIG_KEY,
                cause=ValueError("gold address is not pinned on the target spec or runtime_config"),
            )
        return GoldCraftaxWorld(
            max_steps=max_steps,
            base_url=pinned,
            require_frames=platform.spec.live_frames == "native",
            config_key=GOLD_URL_CONFIG_KEY,
        )
    if ref == FIXTURE_ENVIRONMENT:
        return CraftaxWorld(max_steps=max_steps)
    raise ValueError(f"unknown_craftax_environment:{ref}")
