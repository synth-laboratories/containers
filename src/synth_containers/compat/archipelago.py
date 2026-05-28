from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..capabilities import (
    DatasetDescriptor,
    RuntimeCapabilitySurface,
    RuntimeMetadata,
    TaskCatalog,
    TaskInfo,
)
from ..nouns import ExecutionRecord, TaskDefinition, TaskInstance
from ..ontology import (
    CapabilityLevel,
    ExecutionProfile,
    PrimitiveProtocol,
    RolloutMode,
    RuntimeKind,
    StatefulnessTier,
)
from ..resources import ResourceKind, ResourceRef
from .base import ThinCompatRuntime, execution_from_service_result
from .service_http import (
    execute_service_rollout,
    fetch_service_health,
    fetch_service_info,
    fetch_service_metadata,
    fetch_service_task_info,
)

_SUPPORTED_INTERFACE_MODES = {"synth_http"}
_DEFAULT_INTERFACE_MODE = "synth_http"
_DEFAULT_PROVIDER = "archipelago"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


@dataclass(slots=True)
class ArchipelagoTarget:
    service_url: str
    provider: str = _DEFAULT_PROVIDER
    interface_mode: str = _DEFAULT_INTERFACE_MODE
    world_id: str | None = None
    dataset_name: str | None = None
    limits: dict[str, Any] = field(default_factory=dict)
    auth_token: str | None = None


def _lookup_payload_value(payloads: Sequence[Mapping[str, Any]], *keys: str) -> str | None:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _lookup_payload_object(
    payloads: Sequence[Mapping[str, Any]],
    *keys: str,
) -> dict[str, Any] | None:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, Mapping):
                return dict(value)
    return None


def archipelago_task_requires_binding(task_payload: Mapping[str, Any]) -> bool:
    return bool(
        _lookup_payload_value(
            [task_payload],
            "service_url",
            "container_url",
            "provider_url",
            "base_url",
        )
        or _lookup_payload_value(
            [task_payload],
            "world_id",
            "environment_id",
            "dataset_name",
            "dataset",
            "benchmark",
        )
    )


async def materialize_archipelago_task_config(task_config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(task_config)
    service_url = _lookup_payload_value(
        [task_config],
        "service_url",
        "container_url",
        "provider_url",
        "base_url",
    )
    if service_url:
        normalized = service_url.rstrip("/")
        updated["service_url"] = normalized
        updated["container_url"] = normalized
    updated["provider"] = (
        _lookup_payload_value([task_config], "provider", "execution_provider") or _DEFAULT_PROVIDER
    )
    updated["interface_mode"] = (
        _lookup_payload_value([task_config], "interface_mode") or _DEFAULT_INTERFACE_MODE
    )
    return updated


def resolve_archipelago_target(
    *,
    pool_config: Mapping[str, Any],
    task_config: Mapping[str, Any] | None,
) -> ArchipelagoTarget:
    payloads: list[Mapping[str, Any]] = [dict(pool_config or {})]
    if task_config is not None:
        payloads.insert(0, dict(task_config or {}))

    service_url = _lookup_payload_value(
        payloads,
        "service_url",
        "container_url",
        "provider_url",
        "base_url",
    )
    if not service_url:
        raise ValueError(
            "Archipelago tasks require service_url/container_url for the wrapped service."
        )
    interface_mode = _lookup_payload_value(payloads, "interface_mode") or _DEFAULT_INTERFACE_MODE
    if interface_mode not in _SUPPORTED_INTERFACE_MODES:
        raise ValueError(
            f"Unsupported Archipelago interface_mode {interface_mode!r}. "
            f"Supported modes: {sorted(_SUPPORTED_INTERFACE_MODES)}"
        )

    return ArchipelagoTarget(
        service_url=service_url.rstrip("/"),
        provider=_lookup_payload_value(payloads, "provider", "execution_provider")
        or _DEFAULT_PROVIDER,
        interface_mode=interface_mode,
        world_id=_lookup_payload_value(payloads, "world_id", "environment_id"),
        dataset_name=_lookup_payload_value(payloads, "dataset_name", "dataset", "benchmark"),
        limits=_lookup_payload_object(payloads, "limits") or {},
        auth_token=_lookup_payload_value(
            payloads,
            "worker_token",
            "container_worker_token",
            "auth_token",
        ),
    )


def _archipelago_capabilities() -> RuntimeCapabilitySurface:
    return RuntimeCapabilitySurface(
        runtime_kind=RuntimeKind.ENVIRONMENT,
        profiles=[ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT],
        rollout_modes=[RolloutMode.BLOCKING],
        statefulness_tier=StatefulnessTier.LONG_HORIZON,
        protocol_fidelity={
            PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.DERIVED,
            PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.DERIVED,
            PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.DERIVED,
        },
        trace_support=True,
        reward_support=True,
        artifact_support=True,
        metadata={"framework": "archipelago"},
    )


def _resource_refs_for_target(
    target: ArchipelagoTarget,
    *,
    task_path: str = "",
) -> list[ResourceRef]:
    refs = [
        ResourceRef(
            resource_id="archipelago:service",
            kind=ResourceKind.RUNTIME,
            name="Archipelago service",
            subtype="synth_http_service",
            uri=target.service_url,
            metadata={"provider": target.provider, "interface_mode": target.interface_mode},
        ),
        ResourceRef(
            resource_id="archipelago:evaluation",
            kind=ResourceKind.EVALUATION,
            name="Archipelago evaluator",
            subtype="rollout_reward",
            metadata={"provider": target.provider},
        ),
    ]
    if task_path:
        refs.append(
            ResourceRef(
                resource_id="archipelago:task_bundle",
                kind=ResourceKind.DATA,
                name="Archipelago task bundle",
                subtype="task_bundle",
                path=task_path,
                metadata={"world_id": target.world_id, "dataset_name": target.dataset_name},
            )
        )
    return refs


class ArchipelagoProxyRuntime(ThinCompatRuntime):
    def __init__(
        self,
        *,
        target: ArchipelagoTarget,
        task_path: str = "",
        task_id: str = "archipelago.proxy_eval",
        task_name: str = "Archipelago Proxy Eval",
    ) -> None:
        self.target = target
        self.task_path = task_path
        resources = _resource_refs_for_target(target, task_path=task_path)
        capabilities = _archipelago_capabilities()
        task = TaskDefinition(
            task_id=task_id,
            task_name=task_name,
            task_family="archipelago",
            description=(
                "Proxy a Synth HTTP Archipelago service through the synth-containers contract."
            ),
            benchmark=target.dataset_name or target.world_id or "archipelago",
            metadata={"provider": target.provider, "interface_mode": target.interface_mode},
            resource_refs=resources,
        )
        task_info = TaskInfo(
            task=task,
            dataset=DatasetDescriptor(
                dataset_id=target.dataset_name or target.world_id or "archipelago",
                source="archipelago",
                path=task_path,
                metadata={"world_id": target.world_id, "provider": target.provider},
            ),
            capabilities=capabilities,
            environment="archipelago",
            metadata={"service_url": target.service_url},
            resource_refs=resources,
        )
        super().__init__(
            metadata=RuntimeMetadata(
                runtime_id="archipelago.proxy.synth_containers",
                name="Archipelago synth-containers proxy",
                description="Thin proxy over a Synth HTTP Archipelago runtime.",
                capabilities=capabilities,
                metadata={"service_url": target.service_url, "provider": target.provider},
            ),
            task_info=task_info,
            task_catalog=TaskCatalog(
                catalog_id="archipelago:catalog",
                tasks=[task],
                instances=[
                    TaskInstance(
                        task_instance_id=f"{task_id}:default",
                        task_id=task_id,
                        input_payload={"task_path": task_path} if task_path else {},
                        metadata={"world_id": target.world_id, "dataset_name": target.dataset_name},
                        resource_refs=resources,
                    )
                ],
                metadata={"provider": target.provider},
                resource_refs=resources,
            ),
        )

    async def fetch_health(self) -> dict[str, Any]:
        return await fetch_service_health(target=self.target, error_label="Archipelago")

    async def fetch_info(self) -> dict[str, Any]:
        return await fetch_service_info(target=self.target, error_label="Archipelago")

    async def fetch_metadata(self) -> dict[str, Any]:
        return await fetch_service_metadata(target=self.target, error_label="Archipelago")

    async def fetch_task_info(
        self,
        *,
        seeds: list[int] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return await fetch_service_task_info(
            target=self.target,
            seeds=list(seeds or [0]),
            error_label="Archipelago",
        )

    async def submit_rollout(self, request: dict[str, Any]) -> ExecutionRecord:
        rollout_id = str(
            request.get("rollout_id")
            or request.get("trace_correlation_id")
            or f"archipelago_{uuid.uuid4().hex[:10]}"
        )
        payload = dict(request)
        payload.setdefault("trace_correlation_id", rollout_id)
        env = payload.get("env") if isinstance(payload.get("env"), dict) else {}
        env_payload = dict(env) if isinstance(env, dict) else {}
        env_config = _as_dict(env_payload.get("config"))
        if self.task_path:
            env_config.setdefault("task_path", self.task_path)
        if self.target.world_id:
            env_config.setdefault("world_id", self.target.world_id)
        if self.target.dataset_name:
            env_config.setdefault("dataset_name", self.target.dataset_name)
        payload["env"] = {**env_payload, "config": env_config}
        result = await execute_service_rollout(
            target=self.target,
            rollout_id=rollout_id,
            input_payload=payload,
            org_api_key=str(payload.get("container_api_key") or payload.get("api_key") or "")
            or None,
            error_label="Archipelago",
        )
        execution = execution_from_service_result(
            result=result,
            request=payload,
            task_info=self.task_info(),
            task_catalog=self.task_catalog(),
        )
        self._executions[execution.execution_id] = execution
        return execution
