from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .capabilities import RuntimeCapabilitySurface, RuntimeMetadata, TaskCatalog, TaskInfo
from .nouns import (
    Action,
    Actor,
    CheckpointDescriptor,
    ExecutionRecord,
    Observation,
    StateSnapshot,
    TaskInstance,
    ToolSpecification,
    VerifierResult,
)


@runtime_checkable
class CatalogBacked(Protocol):
    def task_catalog(self) -> TaskCatalog: ...


@runtime_checkable
class Resettable(Protocol):
    def reset(self, task_instance: TaskInstance | None = None, *, seed: int | None = None) -> Observation: ...


@runtime_checkable
class Steppable(Protocol):
    def step(self, action: Action) -> Observation: ...


@runtime_checkable
class Observable(Protocol):
    def observe(self, actor_id: str | None = None) -> Observation: ...


@runtime_checkable
class StateReadable(Protocol):
    def read_state(self) -> StateSnapshot: ...


@runtime_checkable
class Checkpointable(Protocol):
    def checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CheckpointDescriptor: ...


@runtime_checkable
class Restorable(Protocol):
    def restore(self, checkpoint: CheckpointDescriptor | str) -> StateSnapshot: ...


@runtime_checkable
class Forkable(Protocol):
    def fork(self, checkpoint: CheckpointDescriptor | str, *, execution_id: str | None = None) -> str: ...


@runtime_checkable
class RolloutRunnable(Protocol):
    async def run_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord: ...


@runtime_checkable
class AsyncRolloutRunnable(Protocol):
    async def launch_rollout(self, request: Mapping[str, Any]) -> str: ...

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None: ...

    async def get_execution_state(self, execution_id: str) -> Mapping[str, Any] | None: ...


@runtime_checkable
class TraceEmitting(Protocol):
    def read_trace(self, execution_id: str | None = None) -> Mapping[str, Any]: ...


@runtime_checkable
class RewardEmitting(Protocol):
    def reward_info(self, execution_id: str | None = None) -> Mapping[str, Any]: ...


@runtime_checkable
class VerifierBacked(Protocol):
    def verifier_result(self, execution_id: str | None = None) -> VerifierResult: ...


@runtime_checkable
class ToolCallable(Protocol):
    def tools(self) -> Sequence[ToolSpecification]: ...


@runtime_checkable
class TokenTraceEmitting(Protocol):
    def token_trace(self, execution_id: str | None = None) -> Mapping[str, Any]: ...


@runtime_checkable
class MultiActor(Protocol):
    def actors(self) -> Sequence[Actor]: ...


@runtime_checkable
class ProxiedInferenceBacked(Protocol):
    def inference_target(self) -> Mapping[str, Any]: ...


@runtime_checkable
class MetadataReadable(Protocol):
    def metadata(self) -> RuntimeMetadata: ...


@runtime_checkable
class CapabilitiesReadable(Protocol):
    def capabilities(self) -> RuntimeCapabilitySurface: ...


@runtime_checkable
class TaskInfoReadable(Protocol):
    def task_info(self) -> TaskInfo: ...
