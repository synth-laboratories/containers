from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ontology import CheckpointSemantics, OutcomeKind, RuntimeKind
from .serde import JsonDataclassMixin, jsonable


@dataclass(slots=True)
class TaskDefinition(JsonDataclassMixin):
    task_id: str
    task_name: str
    task_family: str = ""
    description: str = ""
    version: str = "v1"
    benchmark: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskInstance(JsonDataclassMixin):
    task_instance_id: str
    task_id: str
    split: str | None = None
    seed: int | None = None
    input_payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    asset_refs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Actor(JsonDataclassMixin):
    actor_id: str
    role: str = "agent"
    display_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action(JsonDataclassMixin):
    actor_id: str
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    action_space: str = ""
    tool_name: str = ""
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Observation(JsonDataclassMixin):
    content: Any = None
    channels: dict[str, Any] = field(default_factory=dict)
    actor_id: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StateSnapshot(JsonDataclassMixin):
    state_id: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    checksum: str = ""
    authoritative: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolSpecification(JsonDataclassMixin):
    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallRecord(JsonDataclassMixin):
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool | None = None
    error: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenTrace(JsonDataclassMixin):
    token_ids: list[int] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    logits_ref: str | None = None
    top_logprobs: list[dict[str, float]] = field(default_factory=list)
    response_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceEvent(JsonDataclassMixin):
    event_type: str
    at: str = ""
    event_id: str = ""
    step_index: int | None = None
    actor_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    token_trace: TokenTrace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckpointDescriptor(JsonDataclassMixin):
    checkpoint_id: str
    rollout_id: str = ""
    checkpoint_uri: str | None = None
    created_at: str | None = None
    checkpoint_version: str = "v1"
    parent_checkpoint_id: str | None = None
    parent_rollout_id: str | None = None
    restore_eligible: bool = True
    label: str | None = None
    labels: list[str] = field(default_factory=list)
    source: str | None = None
    actor_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_available: bool = True
    branchable: bool = True
    checkpoint_semantics: CheckpointSemantics | str = CheckpointSemantics.NONE
    restore_semantics: str = ""
    true_environment_snapshot: bool = False

    def resume_descriptor(self) -> dict[str, Any]:
        semantics = str(self.restore_semantics or self.checkpoint_semantics or "unknown")
        return {
            "mode": semantics,
            "checkpoint_semantics": str(self.checkpoint_semantics),
            "supports_branching": bool(self.restore_eligible and self.branchable),
            "true_environment_snapshot": bool(self.true_environment_snapshot),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = JsonDataclassMixin.to_dict(self)
        metadata = dict(payload.get("metadata") or {})
        if self.checkpoint_semantics and not str(metadata.get("checkpoint_semantics") or "").strip():
            metadata["checkpoint_semantics"] = str(self.checkpoint_semantics)
        if self.restore_semantics and not str(metadata.get("restore_semantics") or "").strip():
            metadata["restore_semantics"] = self.restore_semantics
        payload["metadata"] = metadata
        payload["resume_semantics"] = self.resume_descriptor()
        return payload


@dataclass(slots=True)
class ArtifactDescriptor(JsonDataclassMixin):
    artifact_id: str
    kind: str
    uri: str = ""
    path: str = ""
    media_type: str = ""
    digest: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardSignal(JsonDataclassMixin):
    name: str = "outcome_reward"
    value: float = 0.0
    source: str = ""
    at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerifierResult(JsonDataclassMixin):
    verdict: str = ""
    score: float | None = None
    passed: bool | None = None
    rubric_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Outcome(JsonDataclassMixin):
    kind: OutcomeKind | str = OutcomeKind.REWARD
    reward: float | None = None
    score: float | None = None
    grade: float | None = None
    verifier: VerifierResult | None = None
    passed: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def outcome_reward(self) -> float:
        if self.reward is not None:
            return float(self.reward)
        if self.score is not None:
            return float(self.score)
        if self.grade is not None:
            return float(self.grade)
        if self.verifier and self.verifier.score is not None:
            return float(self.verifier.score)
        return 0.0


@dataclass(slots=True)
class TurnRecord(JsonDataclassMixin):
    turn_index: int
    actor_id: str = "agent"
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    actions: list[Any] = field(default_factory=list)
    executed_actions: list[Any] = field(default_factory=list)
    observation: Observation | None = None
    event_rewards: list[float] = field(default_factory=list)
    outcome_reward: float | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    token_trace: TokenTrace | None = None
    trainable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_training_turn(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "turn_index": self.turn_index,
            "actor_id": self.actor_id,
            "prompt_messages": [dict(item) for item in self.prompt_messages if isinstance(item, dict)],
            "assistant_text": self.assistant_text,
            "actions": [jsonable(item) for item in self.actions],
            "executed_actions": [jsonable(item) for item in self.executed_actions],
            "event_rewards": [float(value) for value in self.event_rewards],
            "trainable": bool(self.trainable),
            "metadata": dict(self.metadata),
        }
        if self.outcome_reward is not None:
            payload["outcome_reward"] = float(self.outcome_reward)
        if self.observation is not None:
            payload["observation"] = self.observation.to_dict()
        if self.tool_calls:
            payload["tool_calls"] = [item.to_dict() for item in self.tool_calls]
        if self.token_trace is not None:
            token_trace_payload = self.token_trace.to_dict()
            payload["token_trace"] = token_trace_payload
            if self.token_trace.token_ids:
                payload["token_ids"] = list(self.token_trace.token_ids)
            if self.token_trace.tokens:
                payload["tokens"] = list(self.token_trace.tokens)
            if self.token_trace.logprobs:
                payload["token_logprobs"] = [float(value) for value in self.token_trace.logprobs]
                sequence_logprob = float(sum(self.token_trace.logprobs))
                payload.setdefault("assistant_sequence_logprob", sequence_logprob)
                payload.setdefault("behavior_sequence_logprob", sequence_logprob)
                payload.setdefault("old_logprob", sequence_logprob)
            if self.token_trace.top_logprobs:
                payload["top_logprobs"] = [dict(item) for item in self.token_trace.top_logprobs]
            if self.token_trace.logits_ref:
                payload["logits_ref"] = self.token_trace.logits_ref
        if self.event_rewards:
            payload.setdefault("decision_reward", float(sum(self.event_rewards)))
        elif self.outcome_reward is not None:
            payload.setdefault("decision_reward", float(self.outcome_reward))
        payload.setdefault("proposed_action_count", int(self.metadata.get("proposed_action_count", len(payload["actions"]))))
        payload.setdefault("executed_action_count", int(self.metadata.get("executed_action_count", len(payload["executed_actions"]))))
        payload.setdefault(
            "effective_action_count",
            int(self.metadata.get("effective_action_count", len(payload["executed_actions"] or payload["actions"]))),
        )
        default_effective_rate = (
            float(payload["effective_action_count"]) / max(float(payload["executed_action_count"]), 1.0)
            if payload["executed_action_count"]
            else float(bool(payload["effective_action_count"]))
        )
        payload.setdefault(
            "effective_action_rate",
            float(self.metadata.get("effective_action_rate", default_effective_rate)),
        )
        passthrough_keys = (
            "step_start",
            "step_end",
            "reward_before",
            "reward_after",
            "env_reward_before",
            "env_reward_after",
            "return_to_go",
            "episode_return",
            "route",
            "request_id",
            "usage",
            "reasoning_text",
            "reasoning_token_count_estimate",
            "behavior_model",
            "behavior_version",
            "policy_version",
            "invalid_parse",
            "assistant_sequence_logprob",
            "behavior_sequence_logprob",
            "old_logprob",
        )
        for key in passthrough_keys:
            if key in self.metadata and key not in payload:
                payload[key] = jsonable(self.metadata[key])
        return payload


@dataclass(slots=True)
class Trajectory(JsonDataclassMixin):
    turns: list[TurnRecord] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def event_rewards(self) -> list[float]:
        rewards: list[float] = []
        for turn in self.turns:
            if turn.event_rewards:
                rewards.append(float(sum(turn.event_rewards)))
            elif turn.outcome_reward is not None:
                rewards.append(float(turn.outcome_reward))
        return rewards


@dataclass(slots=True)
class ExecutionRecord(JsonDataclassMixin):
    execution_id: str
    trace_correlation_id: str
    status: str = "completed"
    success_status: str = "success"
    created_at: str = ""
    updated_at: str = ""
    runtime_kind: RuntimeKind | str = RuntimeKind.ENVIRONMENT
    task: TaskDefinition | None = None
    task_instance: TaskInstance | None = None
    actors: list[Actor] = field(default_factory=list)
    trajectory: Trajectory = field(default_factory=Trajectory)
    outcome: Outcome = field(default_factory=Outcome)
    checkpoint: CheckpointDescriptor | None = None
    parent_rollout_id: str | None = None
    parent_checkpoint_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactDescriptor] = field(default_factory=list)
    state: StateSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rollout_id(self) -> str:
        return self.execution_id

    def seed(self) -> int | None:
        if self.task_instance and self.task_instance.seed is not None:
            return int(self.task_instance.seed)
        seed = self.summary.get("seed")
        try:
            return int(seed) if seed is not None and str(seed).strip() else None
        except (TypeError, ValueError):
            return None

    def trial_id(self) -> str | None:
        for source in (self.metadata, self.summary):
            value = source.get("trial_id") if isinstance(source, dict) else None
            text = str(value or "").strip()
            if text:
                return text
        return None

    def outcome_reward(self) -> float:
        summary_value = self.summary.get("outcome_reward")
        if summary_value is not None:
            try:
                return float(summary_value)
            except (TypeError, ValueError):
                pass
        return self.outcome.outcome_reward()

    def artifact_turns(self) -> list[dict[str, Any]]:
        return [turn.to_training_turn() for turn in self.trajectory.turns]

    def trace_payload(self) -> dict[str, Any]:
        turns = self.artifact_turns()
        events: list[dict[str, Any]] = []
        for event in self.trajectory.events:
            row = event.to_dict()
            payload = row.get("payload")
            if isinstance(payload, dict):
                for key, value in payload.items():
                    row.setdefault(key, value)
            if event.step_index is not None:
                row.setdefault("step_idx", event.step_index)
            events.append(row)
        metadata = dict(self.trajectory.metadata)
        metadata.setdefault("trace_correlation_id", self.trace_correlation_id)
        metadata.setdefault("rollout_id", self.execution_id)
        if self.trial_id():
            metadata.setdefault("trial_id", self.trial_id())
        if self.seed() is not None:
            metadata.setdefault("seed", self.seed())
        metadata.setdefault("step_count", len(events) or len(turns))
        metadata.setdefault("outcome_reward", self.outcome_reward())
        return {
            "schema_version": "2026-04-23",
            "turns": turns,
            "events": events,
            "event_history": events,
            "inference": {"turns": turns},
            "metadata": metadata,
        }
