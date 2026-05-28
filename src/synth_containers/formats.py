from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .capabilities import RuntimeCapabilitySurface, RuntimeMetadata, TaskInfo
from .nouns import CheckpointDescriptor, ExecutionRecord
from .ontology import ResumeSemantics


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ExecutionControlSurface:
    pause_supported: bool = False
    terminate_supported: bool = False
    resume_supported: bool = False
    checkpoint_supported: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionProgress:
    step_count: int = 0
    agent_turn_count: int = 0
    wall_clock_seconds: float | None = None
    reward: float = 0.0
    goal_signals: list[str] | None = None
    stall_signals: list[str] | None = None



def metadata_to_http_payload(metadata: RuntimeMetadata) -> dict[str, Any]:
    return {
        "runtime": {
            "runtime_id": metadata.runtime_id,
            "name": metadata.name,
            "description": metadata.description,
        },
        "capabilities": metadata.capabilities.to_dict(),
        "metadata": dict(metadata.metadata),
    }



def task_info_to_http_payload(task_info: TaskInfo) -> dict[str, Any]:
    return {
        "task": task_info.task.to_dict(),
        "dataset": task_info.dataset.to_dict(),
        "capabilities": task_info.capabilities.to_dict(),
        "limits": dict(task_info.limits),
        "inference": dict(task_info.inference),
        "task_metadata": dict(task_info.task_metadata),
        "environment": task_info.environment,
        "metadata": dict(task_info.metadata),
        "resource_refs": [item.to_dict() for item in task_info.resource_refs],
    }



def checkpoint_to_http_payload(checkpoint: CheckpointDescriptor | None) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    return checkpoint.to_dict()



def _event_rewards(execution: ExecutionRecord) -> list[float]:
    rewards: list[float] = []
    for turn in execution.trajectory.turns:
        if turn.event_rewards:
            rewards.append(float(sum(turn.event_rewards)))
        elif turn.outcome_reward is not None:
            rewards.append(float(turn.outcome_reward))
    return rewards



def execution_to_rollout_payload(execution: ExecutionRecord) -> dict[str, Any]:
    checkpoint_payload = checkpoint_to_http_payload(execution.checkpoint)
    trial_id = execution.trial_id()
    turns = execution.artifact_turns()
    trace_payload = execution.trace_payload()
    artifacts = [item.to_dict() for item in execution.artifacts]
    if execution.trajectory.turns and not any(item.get("kind") == "trajectory" for item in artifacts):
        artifacts.append(
            {
                "artifact_id": f"{execution.execution_id}:trajectory",
                "kind": "trajectory",
                "artifact_type": "trajectory",
                "uri": f"memory://rollouts/{execution.execution_id}/trajectory",
                "trace_correlation_id": execution.trace_correlation_id,
                "trial_id": trial_id,
                "rollout_id": execution.execution_id,
                "outcome_reward": execution.outcome_reward(),
                "step_count": len(execution.trajectory.events) or len(execution.trajectory.turns),
                "turns": turns,
                "trace": trace_payload,
                "metadata": dict(execution.trajectory.metadata),
            }
        )
    summary = {
        "outcome_reward": execution.outcome_reward(),
        **dict(execution.summary),
    }
    if execution.task:
        summary.setdefault("task_id", execution.task.task_id)
    if execution.task_instance and execution.task_instance.split:
        summary.setdefault("dataset_split", execution.task_instance.split)
    if execution.seed() is not None:
        summary.setdefault("seed", execution.seed())
    if trial_id:
        summary.setdefault("trial_id", trial_id)
    if checkpoint_payload is not None:
        summary.setdefault("checkpoint_id", checkpoint_payload.get("checkpoint_id"))
    metadata = dict(execution.metadata)
    if execution.seed() is not None:
        metadata.setdefault("seed", execution.seed())
    if trial_id:
        metadata.setdefault("trial_id", trial_id)
    raw_trace_metadata = trace_payload.get("metadata")
    trace_metadata = dict(raw_trace_metadata) if isinstance(raw_trace_metadata, dict) else {}
    trace_metadata.setdefault("task_id", execution.task.task_id if execution.task else "")
    trace_metadata.setdefault("total_reward", execution.outcome_reward())
    trace_metadata.setdefault("status", execution.status)
    if execution.seed() is not None:
        trace_metadata.setdefault("seed", execution.seed())
    if trial_id:
        trace_metadata.setdefault("trial_id", trial_id)
    trace_payload["metadata"] = trace_metadata
    proposed_action_count = sum(len(turn.actions) for turn in execution.trajectory.turns)
    executed_action_count = sum(len(turn.executed_actions) for turn in execution.trajectory.turns)
    effective_action_count = sum(len(turn.executed_actions or turn.actions) for turn in execution.trajectory.turns)
    status_detail = str(metadata.get("status_detail") or summary.get("status_detail") or execution.status).strip()
    return {
        "rollout_id": execution.execution_id,
        "trace_correlation_id": execution.trace_correlation_id,
        "trial_id": trial_id,
        "status": execution.status,
        "success_status": execution.success_status,
        "status_detail": status_detail,
        "task_id": execution.task.task_id if execution.task else "",
        "seed": execution.seed(),
        "checkpoint_id": checkpoint_payload.get("checkpoint_id") if checkpoint_payload else None,
        "checkpoint": checkpoint_payload,
        "reward_info": {
            "outcome_reward": execution.outcome_reward(),
            "event_rewards": _event_rewards(execution),
            "details": {
                **dict(metadata),
                "seed": execution.seed(),
                "trainable_turn_count": sum(1 for turn in execution.trajectory.turns if turn.trainable),
                "proposed_action_count": proposed_action_count,
                "executed_action_count": executed_action_count,
                "effective_action_count": effective_action_count,
            },
        },
        "summary": summary,
        "usage": dict(execution.usage),
        "artifacts": artifacts,
        "trace": trace_payload,
        "turns": turns,
        "metadata": metadata,
        "created_at": execution.created_at,
        "updated_at": execution.updated_at or execution.created_at,
        "parent_rollout_id": execution.parent_rollout_id,
        "parent_checkpoint_id": execution.parent_checkpoint_id,
    }



def execution_progress(execution: ExecutionRecord) -> ExecutionProgress:
    goal_signals = execution.metadata.get("goal_signals") or execution.summary.get("goal_signals") or []
    stall_signals = execution.metadata.get("stall_signals") or execution.summary.get("stall_signals") or []
    wall_clock_seconds = execution.summary.get("wall_clock_seconds")
    return ExecutionProgress(
        step_count=int(execution.summary.get("step_count") or len(execution.trajectory.events) or len(execution.trajectory.turns)),
        agent_turn_count=len(execution.trajectory.turns),
        wall_clock_seconds=float(wall_clock_seconds) if wall_clock_seconds is not None else None,
        reward=execution.outcome_reward(),
        goal_signals=[str(item) for item in goal_signals],
        stall_signals=[str(item) for item in stall_signals],
    )



def execution_to_state_payload(
    execution: ExecutionRecord,
    *,
    capabilities: RuntimeCapabilitySurface,
    control: ExecutionControlSurface | None = None,
) -> dict[str, Any]:
    checkpoint_payload = checkpoint_to_http_payload(execution.checkpoint)
    progress = execution_progress(execution)
    resolved_control = control or ExecutionControlSurface(
        pause_supported=capabilities.pause_support,
        terminate_supported=capabilities.terminate_support,
        resume_supported=capabilities.resume_support,
        checkpoint_supported=capabilities.checkpoint_support,
    )
    resume_mode = str(capabilities.resume_semantics or ResumeSemantics.UNSUPPORTED)
    return {
        "rollout_id": execution.execution_id,
        "trace_correlation_id": execution.trace_correlation_id,
        "trial_id": execution.trial_id(),
        "status": execution.status,
        "success_status": execution.success_status,
        "status_detail": str(execution.metadata.get("status_detail") or execution.summary.get("status_detail") or execution.status).strip(),
        "created_at": execution.created_at,
        "updated_at": execution.updated_at or execution.created_at,
        "checkpoint_available": checkpoint_payload is not None,
        "checkpoint": checkpoint_payload,
        "termination": execution.metadata.get("termination") or {},
        "control": {
            "pause_supported": resolved_control.pause_supported,
            "terminate_supported": resolved_control.terminate_supported,
            "resume_supported": resolved_control.resume_supported,
            "checkpoint_supported": resolved_control.checkpoint_supported,
        },
        "resume_semantics": {
            "mode": resume_mode,
            "checkpoint_semantics": str(capabilities.checkpoint_semantics),
            "supports_branching": bool(capabilities.supports_branching),
            "true_environment_snapshot": bool(capabilities.true_environment_snapshot),
        },
        "reward_source": execution.metadata.get("reward_source") or capabilities.metadata.get("reward_source") or "",
        "proxy_rewards": bool(execution.metadata.get("proxy_rewards", False)),
        "progress": {
            "step_count": progress.step_count,
            "agent_turn_count": progress.agent_turn_count,
            "wall_clock_seconds": progress.wall_clock_seconds,
            "reward": progress.reward,
            "goal_signals": progress.goal_signals or [],
            "stall_signals": progress.stall_signals or [],
        },
        "artifacts": [item.to_dict() for item in execution.artifacts],
        "env_state": execution.state.to_dict() if execution.state is not None else {},
        "metadata": dict(execution.metadata),
    }
