from __future__ import annotations

from pathlib import Path
from typing import Any

from ..capabilities import RuntimeCapabilitySurface, RuntimeMetadata, TaskCatalog, TaskInfo
from ..ontology import (
    CapabilityLevel,
    CheckpointSemantics,
    ExecutionProfile,
    PrimitiveProtocol,
    ResumeSemantics,
    RolloutMode,
    RuntimeKind,
    StatefulnessTier,
)
from ..resources import ResourceKind, ResourceRef
from .base import RolloutHandler, ThinCompatRuntime


def openenv_capability_surface(
    *,
    checkpointable: bool = True,
    metadata: dict[str, Any] | None = None,
) -> RuntimeCapabilitySurface:
    protocol_fidelity: dict[PrimitiveProtocol | str, CapabilityLevel | str] = {
        PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.NATIVE,
        PrimitiveProtocol.RESETTABLE: CapabilityLevel.NATIVE,
        PrimitiveProtocol.STEPPABLE: CapabilityLevel.NATIVE,
        PrimitiveProtocol.OBSERVABLE: CapabilityLevel.NATIVE,
        PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
        PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.NATIVE,
        PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.NATIVE,
    }
    if checkpointable:
        protocol_fidelity.update(
            {
                PrimitiveProtocol.STATE_READABLE: CapabilityLevel.NATIVE,
                PrimitiveProtocol.CHECKPOINTABLE: CapabilityLevel.NATIVE,
                PrimitiveProtocol.RESTORABLE: CapabilityLevel.NATIVE,
                PrimitiveProtocol.FORKABLE: CapabilityLevel.NATIVE,
                PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
            }
        )
    return RuntimeCapabilitySurface(
        runtime_kind=RuntimeKind.ENVIRONMENT,
        profiles=[
            ExecutionProfile.GYM_STYLE_ENVIRONMENT,
            ExecutionProfile.CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT,
        ],
        rollout_modes=(
            [RolloutMode.BLOCKING, RolloutMode.ASYNC]
            if checkpointable
            else [RolloutMode.BLOCKING]
        ),
        statefulness_tier=(
            StatefulnessTier.LONG_HORIZON if checkpointable else StatefulnessTier.EPISODIC
        ),
        protocol_fidelity=protocol_fidelity,
        checkpoint_semantics=(
            CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT
            if checkpointable
            else CheckpointSemantics.NONE
        ),
        restore_semantics=(
            CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT.value if checkpointable else ""
        ),
        resume_semantics=(
            ResumeSemantics.TRUE_ENVIRONMENT_SNAPSHOT
            if checkpointable
            else ResumeSemantics.UNSUPPORTED
        ),
        checkpoint_support=checkpointable,
        pause_support=checkpointable,
        resume_support=checkpointable,
        terminate_support=checkpointable,
        state_support=checkpointable,
        trace_support=True,
        reward_support=True,
        artifact_support=True,
        supports_branching=checkpointable,
        true_environment_snapshot=checkpointable,
        metadata=dict(metadata or {}),
    )


def openenv_registry_resource_refs(
    *,
    registry_path: Path,
    environment_name: str,
    dataset_id: str,
) -> list[ResourceRef]:
    registry = Path(registry_path)
    return [
        ResourceRef(
            resource_id=f"{dataset_id}:registry",
            kind=ResourceKind.DATA,
            name=f"{environment_name} task registry",
            subtype="task_registry",
            path=str(registry),
            metadata={"framework": "openenv", "environment": environment_name},
        ),
        ResourceRef(
            resource_id=f"{environment_name}:runtime",
            kind=ResourceKind.RUNTIME,
            name=f"{environment_name} runtime",
            subtype="gymnasium_environment",
            metadata={"framework": "openenv"},
        ),
        ResourceRef(
            resource_id=f"{environment_name}:reward",
            kind=ResourceKind.EVALUATION,
            name=f"{environment_name} reward/evaluation",
            subtype="environment_reward",
            metadata={"framework": "openenv"},
        ),
        ResourceRef(
            resource_id=f"{environment_name}:config",
            kind=ResourceKind.CONFIG,
            name=f"{environment_name} container config",
            subtype="runtime_config",
            metadata={"framework": "openenv"},
        ),
    ]


class OpenEnvCompatRuntime(ThinCompatRuntime):
    def __init__(
        self,
        *,
        metadata: RuntimeMetadata,
        task_info: TaskInfo,
        task_catalog: TaskCatalog,
        rollout_handler: RolloutHandler | None = None,
    ) -> None:
        super().__init__(
            metadata=metadata,
            task_info=task_info,
            task_catalog=task_catalog,
            rollout_handler=rollout_handler,
        )
