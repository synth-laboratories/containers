from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from inspect import isawaitable
from typing import Any, TypeAlias

from ..capabilities import RuntimeMetadata, TaskCatalog, TaskInfo
from ..formats import utc_now_iso
from ..nouns import (
    Actor,
    ArtifactDescriptor,
    CheckpointDescriptor,
    ExecutionRecord,
    Observation,
    Outcome,
    StateSnapshot,
    TaskDefinition,
    TaskInstance,
    TraceEvent,
    Trajectory,
    TurnRecord,
    VerifierResult,
)
from ..ontology import OutcomeKind, RuntimeKind
from ..resources import ResourceKind, ResourceRef
from ..serde import JsonObject, jsonable

ServiceResult: TypeAlias = ExecutionRecord | Mapping[str, Any]
RolloutHandler: TypeAlias = Callable[[JsonObject], ServiceResult | Awaitable[ServiceResult]]


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def normalize_resource_refs(
    values: list[ResourceRef | Mapping[str, Any]] | None,
) -> list[ResourceRef]:
    refs: list[ResourceRef] = []
    for index, value in enumerate(values or []):
        if isinstance(value, ResourceRef):
            refs.append(value)
            continue
        payload = dict(value)
        refs.append(
            ResourceRef(
                resource_id=str(
                    payload.get("resource_id") or payload.get("id") or f"resource:{index}"
                ),
                kind=ResourceKind.parse(payload.get("kind")),
                name=str(payload.get("name") or ""),
                subtype=str(payload.get("subtype") or ""),
                uri=str(payload.get("uri") or ""),
                path=str(payload.get("path") or ""),
                digest=str(payload.get("digest") or ""),
                version=str(payload.get("version") or ""),
                media_type=str(payload.get("media_type") or ""),
                size_bytes=(
                    payload.get("size_bytes")
                    if isinstance(payload.get("size_bytes"), int)
                    else None
                ),
                required=bool(payload.get("required", True)),
                labels=[str(item) for item in payload.get("labels") or []],
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    return refs


def _first_numeric(values: list[Any]) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_reward(payload: Mapping[str, Any]) -> float:
    reward_info = _as_dict(payload.get("reward_info"))
    metrics = _as_dict(payload.get("metrics"))
    summary = _as_dict(payload.get("summary"))
    value = _first_numeric(
        [
            payload.get("score"),
            payload.get("reward"),
            metrics.get("outcome_reward"),
            metrics.get("outcome_score"),
            reward_info.get("outcome_reward"),
            reward_info.get("outcome_score"),
            summary.get("outcome_reward"),
            summary.get("total_reward"),
        ]
    )
    return float(value or 0.0)


def _artifact_descriptors(
    payload: Mapping[str, Any],
    *,
    rollout_id: str,
) -> list[ArtifactDescriptor]:
    rows = payload.get("artifacts")
    if rows is None:
        rows = payload.get("artifact")
    artifacts: list[ArtifactDescriptor] = []
    if isinstance(rows, list):
        for index, item in enumerate(rows):
            item_payload = _as_dict(item)
            if not item_payload:
                continue
            artifacts.append(
                ArtifactDescriptor(
                    artifact_id=str(
                        item_payload.get("artifact_id") or f"{rollout_id}:artifact:{index}"
                    ),
                    kind=str(
                        item_payload.get("kind")
                        or item_payload.get("artifact_type")
                        or "artifact"
                    ),
                    uri=str(item_payload.get("uri") or ""),
                    path=str(item_payload.get("path") or ""),
                    media_type=str(item_payload.get("media_type") or ""),
                    digest=str(item_payload.get("digest") or ""),
                    metadata=_as_dict(item_payload.get("metadata")) or item_payload,
                )
            )
    if not artifacts:
        artifacts.append(
            ArtifactDescriptor(
                artifact_id=f"{rollout_id}:service_result",
                kind="service_result",
                uri=f"memory://rollouts/{rollout_id}/service_result",
                metadata={"payload": jsonable(dict(payload))},
            )
        )
    return artifacts


def _turns_from_payload(payload: Mapping[str, Any]) -> list[TurnRecord]:
    raw_turns = payload.get("turns")
    trace_payload = _as_dict(payload.get("trace"))
    if raw_turns is None:
        inference = _as_dict(trace_payload.get("inference"))
        if inference:
            raw_turns = inference.get("turns")
    turns: list[TurnRecord] = []
    if not isinstance(raw_turns, list):
        return turns
    for index, item in enumerate(raw_turns):
        item_payload = _as_dict(item)
        if not item_payload:
            continue
        decision_reward = item_payload.get("decision_reward")
        outcome_reward = item_payload.get("outcome_reward")
        turns.append(
            TurnRecord(
                turn_index=int(item_payload.get("turn_index") or index),
                actor_id=str(item_payload.get("actor_id") or "agent"),
                prompt_messages=[
                    dict(row)
                    for row in _as_list(item_payload.get("prompt_messages"))
                    if isinstance(row, Mapping)
                ],
                assistant_text=str(item_payload.get("assistant_text") or ""),
                actions=_as_list(item_payload.get("actions")),
                executed_actions=(
                    _as_list(item_payload.get("executed_actions"))
                    or _as_list(item_payload.get("actions"))
                ),
                observation=(
                    Observation(content=item_payload.get("observation"))
                    if "observation" in item_payload
                    else None
                ),
                event_rewards=(
                    [float(decision_reward)]
                    if isinstance(decision_reward, (int, float))
                    else []
                ),
                outcome_reward=(
                    float(outcome_reward)
                    if isinstance(outcome_reward, (int, float))
                    else None
                ),
                trainable=bool(item_payload.get("trainable", True)),
                metadata=_as_dict(item_payload.get("metadata")),
            )
        )
    return turns


def _event_step_index(item: Mapping[str, Any], index: int) -> int:
    value = item.get("step_idx") if "step_idx" in item else item.get("step_index")
    if isinstance(value, int):
        return value
    return index


def _events_from_payload(payload: Mapping[str, Any]) -> list[TraceEvent]:
    raw_events: Any = None
    trace_payload = _as_dict(payload.get("trace"))
    if trace_payload:
        raw_events = trace_payload.get("events") or trace_payload.get("event_history")
    if raw_events is None:
        raw_events = payload.get("events") or payload.get("event_history")
    events: list[TraceEvent] = []
    if not isinstance(raw_events, list):
        return events
    for index, item in enumerate(raw_events):
        item_payload = _as_dict(item)
        if not item_payload:
            continue
        events.append(
            TraceEvent(
                event_type=str(
                    item_payload.get("event_type") or item_payload.get("type") or "event"
                ),
                at=str(item_payload.get("at") or ""),
                event_id=str(item_payload.get("event_id") or ""),
                step_index=_event_step_index(item_payload, index),
                actor_id=str(item_payload.get("actor_id") or "") or None,
                payload=item_payload,
                metadata=_as_dict(item_payload.get("metadata")),
            )
        )
    return events


def _task_instance_from_request(
    *,
    request: Mapping[str, Any],
    task: TaskDefinition,
    task_catalog: TaskCatalog,
) -> TaskInstance | None:
    target = str(request.get("task_instance_id") or "").strip()
    if target:
        found = task_catalog.get_instance(target)
        if found is not None:
            return found
    env = _as_dict(request.get("env"))
    seed_value = env.get("seed")
    seed = seed_value if isinstance(seed_value, int) else None
    return TaskInstance(
        task_instance_id=target or f"{task.task_id}:{seed or 0}",
        task_id=task.task_id,
        seed=seed,
    )


def execution_from_service_result(
    *,
    result: Mapping[str, Any],
    request: Mapping[str, Any],
    task_info: TaskInfo,
    task_catalog: TaskCatalog,
) -> ExecutionRecord:
    rollout_id = str(
        result.get("rollout_id")
        or result.get("execution_id")
        or request.get("rollout_id")
        or request.get("trace_correlation_id")
        or f"rollout_{uuid.uuid4().hex[:10]}"
    )
    trace_id = str(
        result.get("trace_correlation_id") or request.get("trace_correlation_id") or rollout_id
    )
    now = utc_now_iso()
    reward = _extract_reward(result)
    status = str(result.get("status") or ("completed" if not result.get("error") else "failed"))
    success = bool(
        result.get("success", status not in {"failed", "error", "cancelled", "canceled"})
    )
    summary = dict(result.get("summary") or result.get("metrics") or {})
    summary.setdefault("outcome_reward", reward)
    metadata = dict(result.get("metadata") or {})
    metadata.setdefault("status_detail", str(result.get("status_detail") or status))
    trace_payload = _as_dict(result.get("trace"))
    trace_metadata = _as_dict(trace_payload.get("metadata"))
    state_payload = _as_dict(result.get("state"))
    state_metadata = _as_dict(result.get("state_metadata"))
    return ExecutionRecord(
        execution_id=rollout_id,
        trace_correlation_id=trace_id,
        status=status,
        success_status=str(result.get("success_status") or ("success" if success else "failed")),
        created_at=str(result.get("created_at") or now),
        updated_at=str(result.get("updated_at") or now),
        runtime_kind=RuntimeKind.ENVIRONMENT,
        task=task_info.task,
        task_instance=_task_instance_from_request(
            request=request,
            task=task_info.task,
            task_catalog=task_catalog,
        ),
        actors=[Actor(actor_id="service", role="runtime", display_name="Wrapped Service")],
        trajectory=Trajectory(
            turns=_turns_from_payload(result),
            events=_events_from_payload(result),
            metadata=dict(trace_metadata or {}),
        ),
        outcome=Outcome(
            kind=OutcomeKind.REWARD,
            reward=reward,
            passed=success,
            verifier=VerifierResult(
                verdict="success" if success else "failed",
                score=reward,
                passed=success,
            ),
        ),
        summary=summary,
        usage=dict(result.get("usage") or {}),
        artifacts=_artifact_descriptors(result, rollout_id=rollout_id),
        state=StateSnapshot(
            state_id=f"{rollout_id}:state",
            values=state_payload,
            created_at=str(result.get("updated_at") or now),
            metadata=state_metadata,
        ),
        metadata=metadata,
    )


class ThinCompatRuntime:
    def __init__(
        self,
        *,
        metadata: RuntimeMetadata,
        task_info: TaskInfo,
        task_catalog: TaskCatalog,
        rollout_handler: RolloutHandler | None = None,
    ) -> None:
        self._metadata = metadata
        self._task_info = task_info
        self._task_catalog = task_catalog
        self._rollout_handler = rollout_handler
        self._executions: dict[str, ExecutionRecord] = {}
        self._checkpoints: dict[str, CheckpointDescriptor] = {}

    def metadata(self) -> RuntimeMetadata:
        return self._metadata

    def task_info(self) -> TaskInfo:
        return self._task_info

    def task_catalog(self) -> TaskCatalog:
        return self._task_catalog

    async def submit_rollout(self, request: JsonObject) -> ExecutionRecord:
        if self._rollout_handler is None:
            raise NotImplementedError("no rollout handler configured")
        value = self._rollout_handler(request)
        if isawaitable(value):
            value = await value
        if isinstance(value, ExecutionRecord):
            execution = value
        else:
            execution = execution_from_service_result(
                result=_as_dict(value),
                request=request,
                task_info=self.task_info(),
                task_catalog=self.task_catalog(),
            )
        self._executions[execution.execution_id] = execution
        return execution

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | None:
        return self._executions.get(str(rollout_id))

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def pause_execution(self, rollout_id: str, request: JsonObject) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def terminate_execution(
        self,
        rollout_id: str,
        request: JsonObject,
    ) -> ExecutionRecord | None:
        execution = await self.get_execution(rollout_id)
        if execution is not None and execution.status not in {"completed", "failed", "cancelled"}:
            execution.status = "cancelled"
            execution.success_status = "cancelled"
            execution.metadata["status_detail"] = str(request.get("reason") or "terminated")
        return execution

    async def create_checkpoint(
        self,
        rollout_id: str,
        request: JsonObject,
    ) -> CheckpointDescriptor | None:
        return None

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointDescriptor | None:
        return self._checkpoints.get(str(checkpoint_id))

    async def list_checkpoints(self, rollout_id: str | None = None) -> list[CheckpointDescriptor]:
        rows = list(self._checkpoints.values())
        if rollout_id is None:
            return rows
        return [item for item in rows if item.rollout_id == str(rollout_id)]

    async def get_rollout_checkpoint(
        self,
        rollout_id: str,
        checkpoint_id: str,
    ) -> CheckpointDescriptor | None:
        checkpoint = await self.get_checkpoint(checkpoint_id)
        if checkpoint is None or checkpoint.rollout_id != str(rollout_id):
            return None
        return checkpoint

    async def update_checkpoint_labels(
        self,
        checkpoint_id: str,
        request: JsonObject,
    ) -> CheckpointDescriptor | None:
        checkpoint = await self.get_checkpoint(checkpoint_id)
        if checkpoint is None:
            return None
        labels = [
            str(item).strip()
            for item in _as_list(request.get("labels"))
            if str(item).strip()
        ]
        if labels:
            checkpoint.labels = labels
            checkpoint.label = labels[0]
        checkpoint.metadata.update(_as_dict(request.get("metadata")))
        checkpoint.annotations.update(_as_dict(request.get("annotations")))
        return checkpoint

    async def resume_execution(
        self,
        rollout_id: str,
        request: JsonObject,
    ) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)
