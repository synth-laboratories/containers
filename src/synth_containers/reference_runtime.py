from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Mapping
from uuid import uuid4

from .capabilities import RuntimeMetadata, TaskCatalog, TaskInfo
from .formats import utc_now_iso
from .nouns import (
    Action,
    Actor,
    CheckpointDescriptor,
    ExecutionRecord,
    Observation,
    Outcome,
    StateSnapshot,
    TaskDefinition,
    TaskInstance,
    ToolSpecification,
    TraceEvent,
    Trajectory,
    TurnRecord,
)
from .ontology import CheckpointSemantics, OutcomeKind
from .wire import RolloutState, SubmissionMode, lifecycle_projection, resolve_submission_mode, state_from_status


CounterAction = str
_ALLOWED_ACTIONS: tuple[CounterAction, ...] = ("increment", "decrement", "stop")


def _coerce_request(request: Mapping[str, Any]) -> dict[str, Any]:
    return dict(request)


def _task_from_info(task_info: TaskInfo | None) -> TaskDefinition:
    if task_info is None:
        return TaskDefinition(
            task_id="counter.default",
            task_name="Counter Runtime",
            task_family="reference",
            description="Simple integer-counter reference runtime.",
            version="v1",
            benchmark="reference",
            metadata={},
        )
    return TaskDefinition(
        task_id=task_info.task.task_id,
        task_name=task_info.task.task_name,
        task_family=task_info.task.task_family,
        description=task_info.task.description,
        version=task_info.task.version,
        benchmark=task_info.task.benchmark,
        metadata=deepcopy(task_info.task.metadata),
    )


class CounterRuntime:
    """Small in-memory runtime that exercises the full normalized contract.

    The environment holds a single integer counter. Actions are:
    - increment: count += 1
    - decrement: count -= 1
    - stop: mark episode done
    """

    def __init__(self, *, target: int = 3) -> None:
        self.target = int(target)
        self._count = 0
        self._done = False
        self._last_reward = 0.0
        self._events: list[TraceEvent] = []
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._actors = [Actor(actor_id="agent", role="agent", display_name="Agent")]

    def reset(self, task_instance: TaskInstance | None = None, *, seed: int | None = None) -> Observation:
        del task_instance, seed
        self._count = 0
        self._done = False
        self._last_reward = 0.0
        self._events = []
        return self.observe()

    def step(self, action: Action | str, *, actor_id: str | None = None) -> Observation:
        if self._done:
            return Observation(
                content="episode_already_done",
                channels={"count": self._count, "reward": 0.0, "done": True},
                actor_id=actor_id or "agent",
                created_at=utc_now_iso(),
            )
        action_name = action.name if isinstance(action, Action) else str(action or "").strip().lower()
        if action_name not in _ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {action_name!r}")
        previous = self._count
        if action_name == "increment":
            self._count += 1
        elif action_name == "decrement":
            self._count -= 1
        elif action_name == "stop":
            self._done = True
        reward = 1.0 if self._count == self.target else 0.0
        if action_name == "stop" and self._count != self.target:
            reward = -0.25
        if self._count == self.target:
            self._done = True
        self._last_reward = reward
        now = utc_now_iso()
        observation = Observation(
            content=f"count={self._count}",
            channels={
                "count": self._count,
                "delta": self._count - previous,
                "reward": reward,
                "done": self._done,
            },
            actor_id=actor_id or "agent",
            created_at=now,
        )
        self._events.append(
            TraceEvent(
                event_type="counter_step",
                at=now,
                step_index=len(self._events) + 1,
                actor_id=actor_id or "agent",
                payload={
                    "action": action_name,
                    "count": self._count,
                    "reward": reward,
                    "done": self._done,
                },
            )
        )
        return observation

    def observe(self, actor_id: str | None = None) -> Observation:
        return Observation(
            content=f"count={self._count}",
            channels={"count": self._count, "reward": self._last_reward, "done": self._done},
            actor_id=actor_id or "agent",
            created_at=utc_now_iso(),
        )

    def read_state(self) -> StateSnapshot:
        return StateSnapshot(
            state_id=f"counter:{self._count}",
            values={"count": self._count, "done": self._done},
            created_at=utc_now_iso(),
            authoritative=True,
            metadata={"runtime": "counter"},
        )

    def checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointDescriptor:
        checkpoint_key = str(checkpoint_id or f"ckpt_{uuid4().hex[:10]}")
        now = utc_now_iso()
        self._checkpoints[checkpoint_key] = {
            "count": self._count,
            "done": self._done,
            "reward": self._last_reward,
            "events": [event.to_dict() for event in self._events],
            "created_at": now,
            "label": label,
            "metadata": dict(metadata or {}),
        }
        return CheckpointDescriptor(
            checkpoint_id=checkpoint_key,
            created_at=now,
            label=label,
            metadata={**dict(metadata or {}), "count": self._count, "done": self._done},
            restore_eligible=True,
            branchable=True,
            checkpoint_semantics=CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT,
            restore_semantics="true_environment_snapshot",
            true_environment_snapshot=True,
        )

    def restore(self, checkpoint: CheckpointDescriptor | str | Mapping[str, Any]) -> StateSnapshot:
        if isinstance(checkpoint, CheckpointDescriptor):
            checkpoint_id = checkpoint.checkpoint_id
        elif isinstance(checkpoint, Mapping):
            checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
        else:
            checkpoint_id = str(checkpoint)
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f"unknown_checkpoint:{checkpoint_id}")
        payload = self._checkpoints[checkpoint_id]
        self._count = int(payload.get("count") or 0)
        self._done = bool(payload.get("done"))
        self._last_reward = float(payload.get("reward") or 0.0)
        self._events = [
            TraceEvent(
                event_type=str(item.get("event_type") or "counter_step"),
                at=str(item.get("at") or ""),
                event_id=str(item.get("event_id") or ""),
                step_index=item.get("step_index"),
                actor_id=item.get("actor_id"),
                payload=dict(item.get("payload") or {}),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in payload.get("events") or []
            if isinstance(item, Mapping)
        ]
        return self.read_state()

    def fork(self, checkpoint: CheckpointDescriptor | str | Mapping[str, Any] | None = None) -> "CounterRuntime":
        clone = CounterRuntime(target=self.target)
        clone._checkpoints = deepcopy(self._checkpoints)
        clone._events = deepcopy(self._events)
        clone._count = self._count
        clone._done = self._done
        clone._last_reward = self._last_reward
        if checkpoint is not None:
            clone.restore(checkpoint)
        return clone

    def trace(self, execution_id: str | None = None) -> dict[str, Any]:
        payload = [event.to_dict() for event in self._events]
        return {
            "trace_id": execution_id or "counter_trace",
            "events": payload,
            "event_history": payload,
            "metadata": {
                "step_count": len(payload),
                "final_count": self._count,
                "target": self.target,
            },
        }

    def outcome(self, execution_id: str | None = None) -> Outcome:
        del execution_id
        return Outcome(
            kind=OutcomeKind.REWARD,
            reward=self._last_reward,
            passed=self._done and self._count == self.target,
            details={"count": self._count, "target": self.target, "done": self._done},
        )

    def tools(self) -> list[ToolSpecification]:
        return [
            ToolSpecification(
                tool_name="counter_action",
                description="Apply one of increment, decrement, or stop to the counter.",
                input_schema={"type": "object", "properties": {"action": {"type": "string", "enum": list(_ALLOWED_ACTIONS)}}},
                output_schema={"type": "object", "properties": {"count": {"type": "integer"}, "done": {"type": "boolean"}}},
                metadata={"stateful": True},
            )
        ]

    def actors(self) -> list[Actor]:
        return list(self._actors)

    def run_rollout(
        self,
        request: Mapping[str, Any],
        *,
        trace_correlation_id: str | None = None,
        task_info: TaskInfo | None = None,
    ) -> ExecutionRecord:
        payload = _coerce_request(request)
        trace_id = str(trace_correlation_id or payload.get("trace_correlation_id") or f"rollout_{uuid4().hex[:10]}")
        env_payload = payload.get("env")
        env = dict(env_payload) if isinstance(env_payload, Mapping) else {}
        seed = env.get("seed")
        self.reset(seed=seed if isinstance(seed, int) else None)

        config = env.get("config")
        config_dict = dict(config) if isinstance(config, Mapping) else {}
        actions = config_dict.get("actions")
        if not isinstance(actions, list) or not actions:
            max_steps = int(config_dict.get("max_steps") or max(self.target, 1))
            actions = ["increment"] * max_steps

        turns: list[TurnRecord] = []
        for index, raw_action in enumerate(actions, start=1):
            observation = self.step(str(raw_action), actor_id="agent")
            reward = float(observation.channels.get("reward") or self._last_reward)
            turns.append(
                TurnRecord(
                    turn_index=index,
                    actor_id="agent",
                    assistant_text=str(raw_action),
                    actions=[{"name": str(raw_action)}],
                    executed_actions=[{"name": str(raw_action)}],
                    observation=observation,
                    event_rewards=[reward],
                    outcome_reward=reward,
                    metadata={
                        "step_idx": index,
                        "count": self._count,
                        "done": self._done,
                    },
                )
            )
            if self._done:
                break

        task = _task_from_info(task_info)
        trial_id = str(payload.get("trial_id") or "").strip() or None
        task_instance_id = str(payload.get("task_instance_id") or f"{task.task_id}:default")
        task_instance = TaskInstance(
            task_instance_id=task_instance_id,
            task_id=task.task_id,
            seed=int(seed) if isinstance(seed, int) else None,
        )
        success = bool(self._done and self._count == self.target)
        now = utc_now_iso()
        trajectory = Trajectory(
            turns=turns,
            events=deepcopy(self._events),
            metadata={
                "trace_correlation_id": trace_id,
                "target": self.target,
                "final_count": self._count,
                "step_count": len(turns),
            },
        )
        summary = {
            "final_count": self._count,
            "target": self.target,
            "step_count": len(turns),
            "task_id": task.task_id,
            "seed": task_instance.seed,
            **({"trial_id": trial_id} if trial_id else {}),
        }
        metadata = {
            "status_detail": "target_reached" if success else "target_not_reached",
            "reference_runtime": "counter",
            "reward_source": "environment",
            **({"trial_id": trial_id} if trial_id else {}),
        }
        return ExecutionRecord(
            execution_id=trace_id,
            trace_correlation_id=trace_id,
            status="completed",
            success_status="success" if success else "failed",
            created_at=now,
            updated_at=now,
            task=task,
            task_instance=task_instance,
            actors=self.actors(),
            trajectory=trajectory,
            outcome=self.outcome(),
            summary=summary,
            usage={},
            metadata=metadata,
            state=self.read_state(),
        )


@dataclass(slots=True)
class _StoredRollout:
    request: dict[str, Any]
    state: RolloutState
    execution: ExecutionRecord
    error: str = ""


class InMemoryAsyncRolloutExecutor:
    """Queued async executor that keeps lifecycle transitions explicit."""

    def __init__(
        self,
        *,
        task_info: TaskInfo | None = None,
        runtime_factory: Callable[[], CounterRuntime] | None = None,
        async_max_workers: int = 4,
    ) -> None:
        self._task_info = task_info
        self._runtime_factory = runtime_factory or CounterRuntime
        self._lock = Lock()
        self._records: dict[str, _StoredRollout] = {}
        self._inflight: set[str] = set()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(async_max_workers)),
            thread_name_prefix="synth_containers_rollout",
        )

    def _task_instance_id(self, payload: Mapping[str, Any], task: TaskDefinition) -> str:
        return str(payload.get("task_instance_id") or f"{task.task_id}:default")

    def _failed_execution(
        self,
        *,
        rollout_id: str,
        payload: Mapping[str, Any],
        error: Exception,
    ) -> ExecutionRecord:
        now = utc_now_iso()
        task = _task_from_info(self._task_info)
        error_text = f"{type(error).__name__}: {error}"
        return ExecutionRecord(
            execution_id=rollout_id,
            trace_correlation_id=rollout_id,
            status=RolloutState.FAILED.value,
            success_status=lifecycle_projection(RolloutState.FAILED)["success_status"],
            created_at=now,
            updated_at=now,
            task=task,
            task_instance=TaskInstance(
                task_instance_id=self._task_instance_id(payload, task),
                task_id=task.task_id,
            ),
            actors=[Actor(actor_id="agent", role="agent", display_name="Agent")],
            trajectory=Trajectory(),
            outcome=Outcome(kind=OutcomeKind.REWARD, reward=0.0, passed=False),
            summary={"error": error_text},
            metadata={
                "status_detail": "failed",
                "error": error_text,
                "error_kind": type(error).__name__,
            },
            usage={},
        )

    def _normalize_terminal_execution(
        self,
        *,
        rollout_id: str,
        execution: ExecutionRecord,
    ) -> tuple[RolloutState, ExecutionRecord]:
        raw_state = state_from_status(execution.status, default=RolloutState.RUNNING)
        if raw_state in {RolloutState.COMPLETED, RolloutState.FAILED, RolloutState.CANCELLED}:
            terminal = raw_state
        else:
            terminal = RolloutState.COMPLETED if str(execution.success_status or "").lower() == "success" else RolloutState.FAILED
        execution.status = terminal.value
        execution.success_status = lifecycle_projection(terminal)["success_status"]
        execution.updated_at = utc_now_iso()
        metadata = dict(execution.metadata or {})
        metadata.setdefault(
            "status_detail",
            "completed" if terminal is RolloutState.COMPLETED else terminal.value,
        )
        execution.metadata = metadata
        return terminal, execution

    def _execute_rollout(self, rollout_id: str) -> None:
        with self._lock:
            record = self._records.get(str(rollout_id))
            if record is None or record.state is not RolloutState.QUEUED:
                self._inflight.discard(str(rollout_id))
                return
            record.state = RolloutState.RUNNING
            record.execution.status = RolloutState.RUNNING.value
            record.execution.success_status = lifecycle_projection(RolloutState.RUNNING)["success_status"]
            record.execution.updated_at = utc_now_iso()
            record.execution.metadata["status_detail"] = "running"
            payload = dict(record.request)
        final_state = RolloutState.FAILED
        try:
            runtime = self._runtime_factory()
            result = runtime.run_rollout(payload, trace_correlation_id=str(rollout_id), task_info=self._task_info)
            final_state, execution = self._normalize_terminal_execution(
                rollout_id=str(rollout_id),
                execution=result,
            )
        except Exception as exc:
            execution = self._failed_execution(
                rollout_id=str(rollout_id),
                payload=payload,
                error=exc,
            )
            final_state = RolloutState.FAILED
        with self._lock:
            row = self._records.get(str(rollout_id))
            if row is not None:
                row.state = final_state
                row.execution = execution
            self._inflight.discard(str(rollout_id))

    def _dispatch_rollout(self, rollout_id: str) -> None:
        rollout_key = str(rollout_id)
        with self._lock:
            record = self._records.get(rollout_key)
            if record is None or record.state is not RolloutState.QUEUED or rollout_key in self._inflight:
                return
            self._inflight.add(rollout_key)
        self._pool.submit(self._execute_rollout, rollout_key)

    def submit_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord:
        payload = _coerce_request(request)
        mode = resolve_submission_mode(payload)
        trace_id = str(payload.get("trace_correlation_id") or f"rollout_{uuid4().hex[:10]}")
        payload["trace_correlation_id"] = trace_id
        if mode is SubmissionMode.SYNC:
            runtime = self._runtime_factory()
            execution = runtime.run_rollout(payload, trace_correlation_id=trace_id, task_info=self._task_info)
            return execution
        now = utc_now_iso()
        task = _task_from_info(self._task_info)
        queued = ExecutionRecord(
            execution_id=trace_id,
            trace_correlation_id=trace_id,
            status=RolloutState.QUEUED.value,
            success_status=lifecycle_projection(RolloutState.QUEUED)["success_status"],
            created_at=now,
            updated_at=now,
            task=task,
            task_instance=TaskInstance(
                task_instance_id=str(payload.get("task_instance_id") or f"{task.task_id}:default"),
                task_id=task.task_id,
            ),
            actors=[Actor(actor_id="agent", role="agent", display_name="Agent")],
            trajectory=Trajectory(),
            outcome=Outcome(kind=OutcomeKind.REWARD, reward=0.0, passed=False),
            summary={},
            metadata={"status_detail": "queued"},
            usage={},
        )
        with self._lock:
            self._records[trace_id] = _StoredRollout(
                request=payload,
                state=RolloutState.QUEUED,
                execution=queued,
            )
        self._dispatch_rollout(trace_id)
        return queued

    def run_pending(self, rollout_id: str | None = None) -> None:
        with self._lock:
            pending_ids = (
                [rollout_id]
                if rollout_id is not None
                else [key for key, row in self._records.items() if row.state is RolloutState.QUEUED]
            )
        for current_id in pending_ids:
            with self._lock:
                if current_id in self._inflight:
                    continue
                self._inflight.add(current_id)
            self._execute_rollout(current_id)

    def rollout_status(self, rollout_id: str) -> ExecutionRecord | None:
        with self._lock:
            row = self._records.get(str(rollout_id))
            return deepcopy(row.execution) if row is not None else None

    def all_rollouts(self) -> list[ExecutionRecord]:
        with self._lock:
            return [deepcopy(row.execution) for row in self._records.values()]


class ReferenceManagedRuntime:
    """Reference managed runtime for the HTTP adapter.

    Combines a concrete runtime (`CounterRuntime`) with async submission semantics
    and lifecycle-safe checkpoint/pause/resume/terminate controls.
    """

    def __init__(
        self,
        *,
        metadata: RuntimeMetadata,
        task_info: TaskInfo,
        task_catalog: TaskCatalog | None = None,
        runtime_factory: Callable[[], CounterRuntime] | None = None,
    ) -> None:
        self._metadata = metadata
        self._task_info = task_info
        self._task_catalog = task_catalog or TaskCatalog(
            catalog_id=f"{task_info.task.task_family or task_info.task.task_id}:catalog",
            tasks=[_task_from_info(task_info)],
            instances=[],
            metadata={"source": "reference_runtime"},
        )
        self._executor = InMemoryAsyncRolloutExecutor(
            task_info=task_info,
            runtime_factory=runtime_factory,
        )
        self._executions: dict[str, ExecutionRecord] = {}
        self._checkpoints: dict[str, CheckpointDescriptor] = {}

    def metadata(self) -> RuntimeMetadata:
        return self._metadata

    def task_info(self) -> TaskInfo:
        return self._task_info

    def task_catalog(self) -> TaskCatalog:
        return self._task_catalog

    def _store_execution(self, execution: ExecutionRecord) -> None:
        self._executions[execution.execution_id] = execution
        if execution.checkpoint is not None:
            self._checkpoints[execution.checkpoint.checkpoint_id] = execution.checkpoint

    def run_pending(self, rollout_id: str | None = None) -> None:
        self._executor.run_pending(rollout_id)
        rows = [self._executor.rollout_status(rollout_id)] if rollout_id else self._executor.all_rollouts()
        for row in rows:
            if row is None:
                continue
            self._store_execution(row)

    async def submit_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord:
        execution = self._executor.submit_rollout(request)
        self._store_execution(execution)
        return execution

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | None:
        status = self._executor.rollout_status(rollout_id)
        if status is not None:
            self._store_execution(status)
            return status
        return self._executions.get(rollout_id)

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | None:
        return await self.get_execution(rollout_id)

    async def pause_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | None:
        execution = await self.get_execution(rollout_id)
        if execution is None:
            return None
        state = state_from_status(execution.status, default=RolloutState.RUNNING)
        if state in {RolloutState.COMPLETED, RolloutState.FAILED, RolloutState.CANCELLED}:
            return execution
        execution.status = RolloutState.PAUSED.value
        execution.success_status = lifecycle_projection(RolloutState.PAUSED)["success_status"]
        execution.updated_at = utc_now_iso()
        execution.metadata["termination"] = {
            "reason": request.get("reason") or "paused",
            "stop_action": "pause",
            "triggered_by": "operator",
            "at": execution.updated_at,
        }
        execution.metadata["status_detail"] = "paused"
        if execution.checkpoint is None:
            checkpoint = await self.create_checkpoint(
                rollout_id,
                {
                    "label": "pause_snapshot",
                    "metadata": {"pause_reason": request.get("reason") or "paused"},
                },
            )
            execution.checkpoint = checkpoint
        self._store_execution(execution)
        return execution

    async def terminate_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | None:
        execution = await self.get_execution(rollout_id)
        if execution is None:
            return None
        execution.status = RolloutState.CANCELLED.value
        execution.success_status = lifecycle_projection(RolloutState.CANCELLED)["success_status"]
        execution.updated_at = utc_now_iso()
        execution.metadata["termination"] = {
            "reason": request.get("reason") or "terminated",
            "stop_action": "terminate",
            "triggered_by": "operator",
            "at": execution.updated_at,
        }
        execution.metadata["status_detail"] = "terminated"
        self._store_execution(execution)
        return execution

    async def create_checkpoint(self, rollout_id: str, request: Mapping[str, Any]) -> CheckpointDescriptor | None:
        execution = await self.get_execution(rollout_id)
        if execution is None:
            return None
        checkpoint = CheckpointDescriptor(
            checkpoint_id=str(request.get("checkpoint_id") or f"ckpt_{uuid4().hex[:10]}"),
            rollout_id=execution.execution_id,
            checkpoint_uri=str(request.get("checkpoint_uri") or f"memory://checkpoints/{execution.execution_id}"),
            created_at=utc_now_iso(),
            checkpoint_version=str(request.get("checkpoint_version") or "v1"),
            label=request.get("label"),
            labels=list(request.get("labels") or []),
            source=request.get("source"),
            actor_ids=list(request.get("actor_ids") or []),
            metadata=dict(request.get("metadata") or {}),
            annotations=dict(request.get("annotations") or {}),
            restore_eligible=bool(request.get("restore_eligible", True)),
            branchable=bool(self._metadata.capabilities.supports_branching),
            checkpoint_semantics=str(self._metadata.capabilities.checkpoint_semantics),
            restore_semantics=str(
                self._metadata.capabilities.restore_semantics
                or self._metadata.capabilities.checkpoint_semantics
            ),
            true_environment_snapshot=bool(self._metadata.capabilities.true_environment_snapshot),
        )
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        execution.checkpoint = checkpoint
        execution.updated_at = utc_now_iso()
        self._store_execution(execution)
        return checkpoint

    async def list_checkpoints(self, rollout_id: str | None = None) -> list[CheckpointDescriptor]:
        if rollout_id is None:
            return list(self._checkpoints.values())
        target = str(rollout_id)
        return [
            checkpoint
            for checkpoint in self._checkpoints.values()
            if checkpoint.rollout_id == target
        ]

    async def get_rollout_checkpoint(self, rollout_id: str, checkpoint_id: str) -> CheckpointDescriptor | None:
        checkpoint = self._checkpoints.get(str(checkpoint_id))
        if checkpoint is None:
            return None
        if checkpoint.rollout_id and checkpoint.rollout_id != str(rollout_id):
            return None
        return checkpoint

    async def update_checkpoint_labels(
        self,
        checkpoint_id: str,
        request: Mapping[str, Any],
    ) -> CheckpointDescriptor | None:
        checkpoint = self._checkpoints.get(str(checkpoint_id))
        if checkpoint is None:
            return None
        checkpoint.labels = list(request.get("labels") or checkpoint.labels)
        checkpoint.annotations = {**checkpoint.annotations, **dict(request.get("annotations") or {})}
        checkpoint.metadata = {**checkpoint.metadata, **dict(request.get("metadata") or {})}
        return checkpoint

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointDescriptor | None:
        return self._checkpoints.get(str(checkpoint_id))

    async def resume_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | None:
        source = await self.get_execution(rollout_id)
        if source is None:
            return None
        requested_checkpoint_id = str(request.get("checkpoint_id") or "").strip()
        checkpoint = (
            self._checkpoints.get(requested_checkpoint_id)
            if requested_checkpoint_id
            else source.checkpoint
        )
        if checkpoint is None:
            return None
        target_rollout_id = str(request.get("target_rollout_id") or f"{rollout_id}_resume_{uuid4().hex[:8]}")
        clone = deepcopy(source)
        clone.execution_id = target_rollout_id
        clone.trace_correlation_id = target_rollout_id
        clone.created_at = utc_now_iso()
        clone.updated_at = clone.created_at
        clone.parent_rollout_id = source.execution_id
        clone.parent_checkpoint_id = checkpoint.checkpoint_id
        clone.metadata = {
            **dict(source.metadata),
            **dict(request.get("branch_metadata") or {}),
            "resumed_from_rollout_id": source.execution_id,
            "resumed_from_checkpoint_id": checkpoint.checkpoint_id,
            "status_detail": "resumed",
        }
        mode = resolve_submission_mode(dict(request))
        if mode is SubmissionMode.ASYNC:
            clone.status = RolloutState.QUEUED.value
            clone.success_status = lifecycle_projection(RolloutState.QUEUED)["success_status"]
        self._store_execution(clone)
        return clone
