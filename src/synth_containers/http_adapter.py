from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException

from .capabilities import RuntimeMetadata
from .compatibility import compatibility_matrix, evaluate_consumer_support
from .formats import (
    ExecutionControlSurface,
    execution_to_rollout_payload,
    execution_to_state_payload,
    metadata_to_http_payload,
    task_info_to_http_payload,
)
from .http_models import (
    CheckpointLabelsRequestModel,
    CreateCheckpointRequestModel,
    PauseRequestModel,
    ResumeRequestModel,
    RolloutRequestModel,
    TerminateRequestModel,
)
from .nouns import CheckpointDescriptor, ExecutionRecord
from .ontology import CONTRACT_VERSION


@runtime_checkable
class ManagedRuntime(Protocol):
    def metadata(self) -> RuntimeMetadata: ...

    def task_info(self) -> Any: ...

    async def submit_rollout(self, request: Mapping[str, Any]) -> ExecutionRecord | Mapping[str, Any]: ...

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | Mapping[str, Any] | None: ...

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | Mapping[str, Any] | None: ...

    async def pause_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | Mapping[str, Any] | None: ...

    async def terminate_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | Mapping[str, Any] | None: ...

    async def create_checkpoint(self, rollout_id: str, request: Mapping[str, Any]) -> CheckpointDescriptor | Mapping[str, Any] | None: ...

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointDescriptor | Mapping[str, Any] | None: ...

    async def list_checkpoints(self, rollout_id: str | None = None) -> list[CheckpointDescriptor | Mapping[str, Any]]: ...

    async def get_rollout_checkpoint(
        self, rollout_id: str, checkpoint_id: str
    ) -> CheckpointDescriptor | Mapping[str, Any] | None: ...

    async def update_checkpoint_labels(
        self, checkpoint_id: str, request: Mapping[str, Any]
    ) -> CheckpointDescriptor | Mapping[str, Any] | None: ...

    async def resume_execution(self, rollout_id: str, request: Mapping[str, Any]) -> ExecutionRecord | Mapping[str, Any] | None: ...


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _call(runtime: Any, *names: str, **kwargs: Any) -> Any:
    for name in names:
        method = getattr(runtime, name, None)
        if method is None:
            continue
        return await _maybe_await(method(**kwargs) if kwargs else method())
    available = ", ".join(names)
    raise HTTPException(status_code=501, detail=f"runtime_missing_method:{available}")



def _metadata(runtime: Any) -> RuntimeMetadata:
    value = runtime.metadata()
    if isinstance(value, RuntimeMetadata):
        return value
    raise TypeError("runtime.metadata() must return RuntimeMetadata")



def _task_info(runtime: Any) -> Any:
    return runtime.task_info()


async def _task_catalog(runtime: Any) -> Any:
    method = getattr(runtime, "task_catalog", None)
    if method is None:
        raise HTTPException(status_code=501, detail="runtime_missing_method:task_catalog")
    return await _maybe_await(method())



def _coerce_rollout_payload(value: ExecutionRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, ExecutionRecord):
        return execution_to_rollout_payload(value)
    return dict(value)



def _coerce_state_payload(runtime: Any, value: ExecutionRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, ExecutionRecord):
        capabilities = _metadata(runtime).capabilities
        return execution_to_state_payload(
            value,
            capabilities=capabilities,
            control=ExecutionControlSurface(
                pause_supported=capabilities.pause_support,
                terminate_supported=capabilities.terminate_support,
                resume_supported=capabilities.resume_support,
                checkpoint_supported=capabilities.checkpoint_support,
            ),
        )
    return dict(value)



def _coerce_checkpoint_payload(value: CheckpointDescriptor | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, CheckpointDescriptor):
        return value.to_dict()
    return dict(value)


def _coerce_checkpoint_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("runtime.list_checkpoints() must return a list")
    rows: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, CheckpointDescriptor):
            rows.append(item.to_dict())
        elif isinstance(item, Mapping):
            rows.append(dict(item))
        else:
            raise TypeError("checkpoint rows must be mapping-like")
    return rows



def create_reference_app(runtime: ManagedRuntime, *, title: str = "synth-containers-reference") -> FastAPI:
    app = FastAPI(title=title)

    @app.get("/")
    async def root() -> dict[str, Any]:
        metadata = _metadata(runtime)
        return {
            "status": "ok",
            "contract_version": CONTRACT_VERSION,
            "runtime": metadata_to_http_payload(metadata),
            "task_info": task_info_to_http_payload(_task_info(runtime)),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "contract_version": CONTRACT_VERSION}

    @app.get("/metadata")
    @app.get("/info")
    async def metadata() -> dict[str, Any]:
        return metadata_to_http_payload(_metadata(runtime))

    @app.get("/task_info")
    async def task_info() -> dict[str, Any]:
        return task_info_to_http_payload(_task_info(runtime))

    @app.get("/task_catalog")
    async def task_catalog() -> dict[str, Any]:
        value = await _task_catalog(runtime)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("runtime.task_catalog() must return a mapping-like payload")

    @app.get("/compatibility")
    async def compatibility(target: str | None = None) -> dict[str, Any]:
        metadata = _metadata(runtime)
        if target is None or not str(target).strip():
            return compatibility_matrix(metadata)
        normalized_target = str(target).strip()
        try:
            return evaluate_consumer_support(metadata, normalized_target).to_dict()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid_compatibility_target:{normalized_target}:{exc}") from exc

    @app.post("/rollout")
    @app.post("/rollouts")
    async def rollout(request: RolloutRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="python", exclude_none=True)
        result = await _call(runtime, "submit_rollout", request=payload)
        return _coerce_rollout_payload(result)

    @app.get("/rollouts/{rollout_id}")
    async def get_rollout(rollout_id: str) -> dict[str, Any]:
        result = await _call(runtime, "get_execution", rollout_id=rollout_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_rollout_payload(result)

    @app.get("/rollouts/{rollout_id}/state")
    async def get_rollout_state(rollout_id: str) -> dict[str, Any]:
        result = await _call(runtime, "get_execution_state", rollout_id=rollout_id)
        if result is None:
            result = await _call(runtime, "get_execution", rollout_id=rollout_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.get("/rollouts/{rollout_id}/summary")
    async def get_rollout_summary(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        return {
            "rollout_id": rollout_id,
            "trace_correlation_id": payload.get("trace_correlation_id"),
            "summary": payload.get("summary") or {},
            "outcome_reward": ((payload.get("reward_info") or {}).get("outcome_reward")),
            "parent_rollout_id": payload.get("parent_rollout_id"),
            "parent_checkpoint_id": payload.get("parent_checkpoint_id"),
        }

    @app.get("/rollouts/{rollout_id}/usage")
    async def get_rollout_usage(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        return {
            "rollout_id": rollout_id,
            "trace_correlation_id": payload.get("trace_correlation_id"),
            "usage": payload.get("usage") or {},
        }

    @app.get("/rollouts/{rollout_id}/artifacts")
    async def get_rollout_artifacts(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        return {"rollout_id": rollout_id, "artifact": payload.get("artifact") or []}

    @app.get("/rollouts/{rollout_id}/events")
    async def get_rollout_events(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
        return {"rollout_id": rollout_id, "events": trace.get("events") or trace.get("event_history") or []}

    @app.get("/rollouts/{rollout_id}/trace")
    async def get_rollout_trace(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
        return {"rollout_id": rollout_id, **trace}

    @app.post("/rollouts/{rollout_id}/pause")
    async def pause_rollout(rollout_id: str, request: PauseRequestModel) -> dict[str, Any]:
        result = await _call(runtime, "pause_execution", "pause", rollout_id=rollout_id, request=request.model_dump(mode="python", exclude_none=True))
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.post("/rollouts/{rollout_id}/terminate")
    async def terminate_rollout(rollout_id: str, request: TerminateRequestModel) -> dict[str, Any]:
        result = await _call(runtime, "terminate_execution", "terminate", rollout_id=rollout_id, request=request.model_dump(mode="python", exclude_none=True))
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.post("/rollouts/{rollout_id}/checkpoints")
    async def create_checkpoint(rollout_id: str, request: CreateCheckpointRequestModel) -> dict[str, Any]:
        result = await _call(runtime, "create_checkpoint", "checkpoint", rollout_id=rollout_id, request=request.model_dump(mode="python", exclude_none=True))
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_checkpoint_payload(result)

    @app.get("/rollouts/{rollout_id}/checkpoints")
    async def list_rollout_checkpoints(rollout_id: str) -> dict[str, Any]:
        rows = await _call(runtime, "list_checkpoints", rollout_id=rollout_id)
        return {
            "rollout_id": rollout_id,
            "checkpoints": _coerce_checkpoint_list(rows),
        }

    @app.get("/rollouts/{rollout_id}/checkpoints/{checkpoint_id}")
    async def get_rollout_checkpoint(rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
        result = await _call(
            runtime,
            "get_rollout_checkpoint",
            rollout_id=rollout_id,
            checkpoint_id=checkpoint_id,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.get("/checkpoints")
    async def list_checkpoints() -> dict[str, Any]:
        rows = await _call(runtime, "list_checkpoints", rollout_id=None)
        return {"checkpoints": _coerce_checkpoint_list(rows)}

    @app.get("/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str) -> dict[str, Any]:
        result = await _call(runtime, "get_checkpoint", checkpoint_id=checkpoint_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.post("/checkpoints/{checkpoint_id}/labels")
    async def update_checkpoint_labels(
        checkpoint_id: str,
        request: CheckpointLabelsRequestModel,
    ) -> dict[str, Any]:
        result = await _call(
            runtime,
            "update_checkpoint_labels",
            checkpoint_id=checkpoint_id,
            request=request.model_dump(mode="python", exclude_none=True),
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.post("/rollouts/{rollout_id}/resume")
    async def resume_rollout(rollout_id: str, request: ResumeRequestModel) -> dict[str, Any]:
        result = await _call(runtime, "resume_execution", "resume", rollout_id=rollout_id, request=request.model_dump(mode="python", exclude_none=True))
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_rollout_payload(result)

    return app
