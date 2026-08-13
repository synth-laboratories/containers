"""Drive one Craftax episode into a durable log. Harness owns plans; world owns steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ..event_log import RolloutEventLog
from .craftax_world import CraftaxWorld, StepResult


class Planner(Protocol):
    def plan(self, observation: dict[str, Any], on_delta: Any = None) -> list[str]: ...

    def usage(self) -> dict[str, Any]: ...

    def metadata(self) -> dict[str, Any]: ...


def _emit_obs(log: RolloutEventLog, result: StepResult, *, seed: int) -> str | None:
    log.append(
        "observation",
        {
            "step": result.env_steps,
            "seed": seed,
            "grid": result.ascii_map,
            "readout": result.observation,
        },
    )
    durable_url = (
        log.persist_frame(result.env_steps, result.frame_bytes)
        if result.frame_bytes is not None
        else None
    )
    frame = {"step": result.env_steps, "digest": result.frame_digest, "format": "ascii"}
    if durable_url is not None:
        frame.update({"url": durable_url, "format": "png"})
    log.append("frame", frame)
    log.append(
        "artifact.declared",
        {
            "kind": "frame",
            "digest": result.frame_digest,
            "step": result.env_steps,
            **({"url": durable_url} if durable_url is not None else {}),
        },
    )
    if durable_url is not None:
        log.append(
            "artifact.available",
            {"kind": "frame", "digest": result.frame_digest, "url": durable_url},
        )
    return durable_url


def _relay_native(world: Any, log: RolloutEventLog) -> None:
    drain = getattr(world, "drain_native_events", None)
    if not callable(drain):
        return
    for event in drain():
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        if not isinstance(kind, str) or not kind:
            continue
        payload = {key: value for key, value in event.items() if key != "kind"}
        log.append(kind, payload)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _frame_record(result: StepResult, durable_url: str | None) -> dict[str, Any]:
    """In-memory artifact row. format=png only when persist_frame returned a url."""
    payload = result.frame_bytes
    if durable_url is not None and payload is not None and payload.startswith(PNG_MAGIC):
        return {"digest": result.frame_digest, "bytes": payload, "format": "png"}
    if payload is not None:
        return {"digest": result.frame_digest, "bytes": payload, "format": "ascii"}
    return {"digest": result.frame_digest, "bytes": b"ASCII", "format": "ascii"}


def run_episode(
    *,
    world: CraftaxWorld,
    planner: Planner,
    log: RolloutEventLog,
    seed: int,
    max_steps: int,
    omit_reward: bool = False,
    emit_policy_spans: bool = True,
    resume_checkpoint: dict[str, Any] | None = None,
    checkpoint_callback: Callable[
        [Any, Planner, StepResult, list[float | None]], dict[str, Any]
    ]
    | None = None,
) -> dict[str, Any]:
    log.append("env.episode.opened", {"seed": seed, "max_steps": max_steps})
    result = world.reset(seed, max_steps=max_steps)
    if resume_checkpoint is not None:
        env_blob = resume_checkpoint.get("environment_blob")
        policy_state = resume_checkpoint.get("policy_state")
        restore_world = getattr(world, "restore", None)
        restore_policy = getattr(planner, "restore_checkpoint_state", None)
        if not isinstance(env_blob, str) or not env_blob:
            raise RuntimeError("resume checkpoint omitted environment_blob")
        if not isinstance(policy_state, dict):
            raise RuntimeError("resume checkpoint omitted policy_state")
        if not callable(restore_world) or not callable(restore_policy):
            raise RuntimeError("runtime does not implement true checkpoint restore")
        result = restore_world(env_blob)
        restore_policy(policy_state)
        log.append(
            "rollout.restored",
            {
                "checkpoint_id": resume_checkpoint.get("checkpoint_id"),
                "parent_rollout_id": resume_checkpoint.get("rollout_id"),
                "step": result.env_steps,
                "checkpoint_semantics": "true_environment_snapshot",
            },
        )
    signals: list[float | None] = []
    actions: list[str] = []
    frames: list[dict[str, Any]] = []
    scheduled_checkpoints: list[dict[str, Any]] = []
    _relay_native(world, log)
    durable_url = _emit_obs(log, result, seed=seed)
    frames.append(_frame_record(result, durable_url))
    session_open = False
    if emit_policy_spans:
        log.append("policy.session.opened", dict(planner.metadata()))
        session_open = True
    # Step zero is a real policy-call boundary. Capturing it guarantees a true
    # branch even when one model plan spans the entire episode and the next
    # boundary is already terminal.
    if checkpoint_callback is not None:
        scheduled_checkpoints.append(checkpoint_callback(world, planner, result, list(signals)))
    while not result.done:
        if emit_policy_spans:
            log.append(
                "span.policy.opened",
                {"harness": planner.metadata().get("harness"), "call": planner.metadata()},
            )

        def on_delta(payload: dict[str, Any]) -> None:
            if emit_policy_spans and isinstance(payload, dict) and payload:
                log.append("span.policy.data", payload)

        plan = planner.plan(result.observation, on_delta=on_delta)
        if emit_policy_spans:
            trace_data = getattr(planner, "trace_data", None)
            if callable(trace_data):
                data = trace_data()
                if data:
                    log.append("span.policy.data", data)
            log.append("span.policy.plan", {"actions": plan, "length": len(plan)})
            log.append("span.policy.closed", {"length": len(plan)})
        if not plan:
            plan = ["noop"]
        for action in plan:
            if result.done:
                break
            log.append("span.step.opened", {"action": action, "step": result.env_steps})
            result = world.step(action)
            actions.append(action)
            _relay_native(world, log)
            log.append("action", {"step": result.env_steps, "action": action})
            value: float | None = result.reward
            if omit_reward and result.env_steps == 2:
                value = None
            signals.append(value)
            log.append(
                "reward_signal",
                {"step": result.env_steps, "value": value, "authority": "environment"},
            )
            durable_url = _emit_obs(log, result, seed=seed)
            frames.append(_frame_record(result, durable_url))
            log.append("span.step.closed", {"action": action, "step": result.env_steps})
        if checkpoint_callback is not None:
            scheduled_checkpoints.append(
                checkpoint_callback(world, planner, result, list(signals))
            )
    if session_open:
        log.append("policy.session.closed", {"calls": planner.usage().get("calls")})
    log.append("env.episode.closed", {"status": "completed", "steps": result.env_steps})
    log.append("status", {"status": "completed", "steps": result.env_steps})
    evidence_high_water = log.high_water
    log.append("capture.high_water", {"high_water": evidence_high_water})
    log.append("capture.closed", {"high_water": evidence_high_water})
    log.mark_closed()
    return {
        "reward_signals": signals,
        "actions": actions,
        "usage": planner.usage(),
        "frame_digest": result.frame_digest,
        "steps": result.env_steps,
        "frames": frames,
        "scheduled_checkpoints": scheduled_checkpoints,
    }
