from __future__ import annotations

from pathlib import Path
from typing import Any

from ..capabilities import RuntimeCapabilitySurface, RuntimeMetadata, TaskCatalog, TaskInfo
from ..ontology import (
    CapabilityLevel,
    ExecutionProfile,
    PrimitiveProtocol,
    RolloutMode,
    RuntimeKind,
    StatefulnessTier,
)
from ..resources import ResourceKind, ResourceRef
from .base import RolloutHandler, ThinCompatRuntime


def harbor_capability_surface(
    *,
    metadata: dict[str, Any] | None = None,
) -> RuntimeCapabilitySurface:
    return RuntimeCapabilitySurface(
        runtime_kind=RuntimeKind.ENVIRONMENT,
        profiles=[ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT],
        rollout_modes=[RolloutMode.BLOCKING],
        statefulness_tier=StatefulnessTier.EPISODIC,
        protocol_fidelity={
            PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.DERIVED,
            PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.DERIVED,
            PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.DERIVED,
        },
        trace_support=True,
        reward_support=True,
        artifact_support=True,
        metadata=dict(metadata or {}),
    )


def harbor_resource_refs(
    *,
    container_root: Path,
    dataset_id: str,
    runner_path: Path | None = None,
) -> list[ResourceRef]:
    root = Path(container_root)
    refs = [
        ResourceRef(
            resource_id=f"{dataset_id}:dataset",
            kind=ResourceKind.DATA,
            name=dataset_id,
            subtype="dataset",
            uri=f"hf://datasets/{dataset_id}",
            metadata={"framework": "harbor"},
        ),
        ResourceRef(
            resource_id="harbor:build_context",
            kind=ResourceKind.CODE,
            name="Harbor build context",
            subtype="build_context",
            path=str(root),
            metadata={"framework": "harbor"},
        ),
        ResourceRef(
            resource_id="harbor:evaluation",
            kind=ResourceKind.EVALUATION,
            name="Harbor verifier/evaluation",
            subtype="packaging_smoke",
            metadata={"framework": "harbor"},
        ),
    ]
    if runner_path is not None:
        refs.append(
            ResourceRef(
                resource_id="harbor:runner",
                kind=ResourceKind.CODE,
                name="Harbor runner",
                subtype="entrypoint",
                path=str(runner_path),
                metadata={"framework": "harbor"},
            )
        )
    return refs


class HarborCompatRuntime(ThinCompatRuntime):
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
