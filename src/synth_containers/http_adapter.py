from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import uuid
from inspect import isawaitable
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .capabilities import RuntimeMetadata, TaskCatalog, TaskInfo
from .compatibility import compatibility_matrix, evaluate_consumer_support
from .formats import (
    ExecutionControlSurface,
    execution_to_rollout_payload,
    execution_to_state_payload,
    metadata_to_http_payload,
    task_info_to_http_payload,
)
from .event_log import (
    CONTROL_SUBSCRIBED,
    RolloutEventLog,
    stream_descriptor,
)
from .http_models import (
    CheckpointLabelsRequestModel,
    CreateCheckpointRequestModel,
    PauseRequestModel,
    ResumeRequestModel,
    RewardRequestModel,
    RolloutRequestModel,
    TerminateRequestModel,
)
from .annotations import (
    RolloutAnnotationList,
    coerce_annotation_list,
    derive_annotations_from_execution,
)
from .nouns import CheckpointDescriptor, ExecutionRecord
from .ontology import CONTRACT_VERSION
from .prompt_programs import gepa_optimizer_contract
from .serde import JsonObject


@runtime_checkable
class ManagedRuntime(Protocol):
    def metadata(self) -> RuntimeMetadata: ...

    def task_info(self) -> TaskInfo: ...

    def task_catalog(self) -> TaskCatalog: ...

    async def submit_rollout(self, request: JsonObject) -> ExecutionRecord: ...

    async def get_execution(self, rollout_id: str) -> ExecutionRecord | None: ...

    async def get_execution_state(self, rollout_id: str) -> ExecutionRecord | None: ...

    async def pause_execution(
        self, rollout_id: str, request: JsonObject
    ) -> ExecutionRecord | None: ...

    async def terminate_execution(
        self, rollout_id: str, request: JsonObject
    ) -> ExecutionRecord | None: ...

    async def create_checkpoint(
        self, rollout_id: str, request: JsonObject
    ) -> CheckpointDescriptor | None: ...

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointDescriptor | None: ...

    async def list_checkpoints(
        self, rollout_id: str | None = None
    ) -> list[CheckpointDescriptor]: ...

    async def get_rollout_checkpoint(
        self, rollout_id: str, checkpoint_id: str
    ) -> CheckpointDescriptor | None: ...

    async def update_checkpoint_labels(
        self, checkpoint_id: str, request: JsonObject
    ) -> CheckpointDescriptor | None: ...

    async def resume_execution(
        self, rollout_id: str, request: JsonObject
    ) -> ExecutionRecord | None: ...


def _metadata(runtime: ManagedRuntime) -> RuntimeMetadata:
    value = runtime.metadata()
    if isinstance(value, RuntimeMetadata):
        return value
    raise TypeError("runtime.metadata() must return RuntimeMetadata")


def _task_info(runtime: ManagedRuntime) -> TaskInfo:
    return runtime.task_info()


async def _task_info_for_request(runtime: ManagedRuntime, query: dict[str, Any]) -> TaskInfo:
    handler = getattr(runtime, "task_info_for_request", None)
    if callable(handler):
        value = handler(query)
        if isawaitable(value):
            value = await value
        if isinstance(value, TaskInfo):
            return value
        raise TypeError("runtime.task_info_for_request() must return TaskInfo")
    return _task_info(runtime)


def _task_catalog(runtime: ManagedRuntime) -> TaskCatalog:
    return runtime.task_catalog()


def _gepa_optimizer_route_contract(runtime: ManagedRuntime) -> dict[str, Any] | None:
    route_methods = {
        "program_route": "program",
        "taskset_route": "taskset_info",
        "taskset_tasks_route": "taskset_tasks",
    }
    if not all(callable(getattr(runtime, method, None)) for method in route_methods.values()):
        return None
    return gepa_optimizer_contract()


def _with_optimizer_contracts(payload: dict[str, Any], runtime: ManagedRuntime) -> dict[str, Any]:
    value = dict(payload)
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    gepa_contract = _gepa_optimizer_route_contract(runtime)
    if gepa_contract is not None:
        optimizer_contracts = metadata.get("optimizer_contracts")
        if not isinstance(optimizer_contracts, dict):
            optimizer_contracts = {}
        else:
            optimizer_contracts = dict(optimizer_contracts)
        optimizer_contracts["gepa"] = {
            **gepa_contract,
            **dict(optimizer_contracts.get("gepa") or {}),
        }
        metadata["optimizer_contracts"] = optimizer_contracts
    value["metadata"] = metadata
    return value


async def _optional_runtime_contract_call(
    runtime: ManagedRuntime,
    method_name: str,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    handler = getattr(runtime, method_name, None)
    if not callable(handler):
        raise HTTPException(status_code=404, detail=f"container_route_not_supported:{method_name}")
    value = handler(dict(request or {})) if request is not None else handler()
    if isawaitable(value):
        value = await value
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"runtime.{method_name}() must return a mapping")


def _coerce_rollout_payload(value: ExecutionRecord) -> dict[str, Any]:
    return execution_to_rollout_payload(value)


def _coerce_state_payload(runtime: ManagedRuntime, value: ExecutionRecord) -> dict[str, Any]:
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


def _coerce_checkpoint_payload(value: CheckpointDescriptor) -> dict[str, Any]:
    return value.to_dict()


def _coerce_checkpoint_list(value: list[CheckpointDescriptor]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("runtime.list_checkpoints() must return a list")
    return [item.to_dict() for item in value]


async def _resolve_rollout_annotations(
    runtime: ManagedRuntime, rollout_id: str
) -> RolloutAnnotationList:
    execution = await runtime.get_execution(rollout_id=rollout_id)
    if execution is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    handler = getattr(runtime, "get_rollout_annotations", None)
    if callable(handler):
        value = handler(rollout_id)
        if isawaitable(value):
            value = await value
        coerced = coerce_annotation_list(
            value,
            rollout_id=rollout_id,
            trace_correlation_id=execution.trace_correlation_id,
        )
        if coerced is not None:
            return coerced
    return derive_annotations_from_execution(execution)


def create_reference_app(
    runtime: ManagedRuntime,
    *,
    title: str = "synth-containers-reference",
    storage_root: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title=title)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    telemetry_by_rollout: dict[str, dict[str, Any]] = {}
    event_logs: dict[str, RolloutEventLog] = {}
    start_requests: dict[str, dict[str, Any]] = {}
    start_responses: dict[str, dict[str, Any]] = {}
    start_lock = asyncio.Lock()
    reward_executions: dict[str, dict[str, Any]] = {}
    event_store_root = (
        Path(storage_root)
        if storage_root is not None
        else Path(tempfile.mkdtemp(prefix="synth-containers-reference-events-"))
    )

    def _new_event_log(rollout_id: str, stream_id: str) -> RolloutEventLog:
        journal_name = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()
        return RolloutEventLog(
            rollout_id=rollout_id,
            stream_id=stream_id,
            journal_path=event_store_root / "event_logs" / f"{journal_name}.jsonl",
        )

    async def live_snapshot(rollout_id: str) -> dict[str, Any] | None:
        result = await runtime.get_execution_state(rollout_id=rollout_id)
        if result is None:
            result = await runtime.get_execution(rollout_id=rollout_id)
        return _coerce_state_payload(runtime, result) if result is not None else None

    def _append_snapshot(log: RolloutEventLog, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        if log.last_snapshot_key == encoded:
            return
        state = str(payload.get("status") or payload.get("state") or "running")
        terminal = state.lower() in {"completed", "failed", "cancelled", "terminated", "stopped"}
        kind = "eval.run.terminal" if terminal else "snapshot"
        log.append(kind, payload)
        log.last_snapshot_key = encoded
        if terminal:
            log.mark_closed()

    async def _ensure_log_current(rollout_id: str) -> RolloutEventLog | None:
        log = event_logs.get(rollout_id)
        if log is None or log.closed:
            return log
        snapshot = await live_snapshot(rollout_id)
        if snapshot is None:
            return log
        _append_snapshot(log, snapshot)
        return log

    def live_event_from_envelope(rollout_id: str, envelope: Any) -> dict[str, Any]:
        row = envelope.to_dict()
        row["rollout_id"] = rollout_id
        row["run_id"] = envelope.payload.get("run_id") or rollout_id
        row["lane"] = rollout_id
        if envelope.sequence is not None:
            row["event_id"] = envelope.sequence
        return row

    @app.get("/")
    async def root() -> dict[str, Any]:
        metadata = _metadata(runtime)
        return {
            "status": "ok",
            "contract_version": CONTRACT_VERSION,
            "runtime": _with_optimizer_contracts(metadata_to_http_payload(metadata), runtime),
            "task_info": _with_optimizer_contracts(
                task_info_to_http_payload(await _task_info_for_request(runtime, {})),
                runtime,
            ),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "contract_version": CONTRACT_VERSION}

    @app.get("/metadata")
    @app.get("/info")
    async def metadata() -> dict[str, Any]:
        return _with_optimizer_contracts(metadata_to_http_payload(_metadata(runtime)), runtime)

    @app.get("/task_info")
    async def task_info(request: Request) -> dict[str, Any]:
        query = {key: value for key, value in request.query_params.multi_items()}
        return _with_optimizer_contracts(
            task_info_to_http_payload(await _task_info_for_request(runtime, query)),
            runtime,
        )

    @app.get("/program")
    async def program() -> dict[str, Any]:
        return await _optional_runtime_contract_call(runtime, "program")

    @app.get("/taskset")
    async def taskset() -> dict[str, Any]:
        return await _optional_runtime_contract_call(runtime, "taskset_info")

    @app.post("/taskset/tasks")
    async def taskset_tasks(request: Request) -> dict[str, Any]:
        payload = await request.json()
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="taskset_tasks_request_must_be_object")
        return await _optional_runtime_contract_call(runtime, "taskset_tasks", payload)

    @app.get("/task_catalog")
    async def task_catalog() -> dict[str, Any]:
        return _task_catalog(runtime).to_dict()

    @app.get("/compatibility")
    async def compatibility(target: str | None = None) -> dict[str, Any]:
        metadata = _metadata(runtime)
        if target is None or not str(target).strip():
            return compatibility_matrix(metadata)
        normalized_target = str(target).strip()
        try:
            return evaluate_consumer_support(metadata, normalized_target).to_dict()
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"invalid_compatibility_target:{normalized_target}:{exc}"
            ) from exc

    @app.post("/rollout", include_in_schema=False)
    @app.post("/rollouts")
    async def rollout(request: RolloutRequestModel) -> dict[str, Any]:
        async with start_lock:
            return await _rollout_locked(request)

    async def _rollout_locked(request: RolloutRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        # The prepare response allocates the public rollout id. Existing managed
        # runtimes key execution by trace_correlation_id, so carry that id through
        # when the caller did not provide a separate correlation id.
        if payload.get("rollout_id") and not payload.get("trace_correlation_id"):
            payload["trace_correlation_id"] = payload["rollout_id"]
        telemetry = payload.get("telemetry")
        requested_rollout_id = payload.get("rollout_id")
        if requested_rollout_id is not None:
            identity = str(requested_rollout_id)
            previous_request = start_requests.get(identity)
            if previous_request is not None:
                if previous_request != payload:
                    raise HTTPException(
                        status_code=409,
                        detail={"error": "rollout_identity_conflict", "rollout_id": identity},
                    )
                replay = dict(start_responses[identity])
                replay["replayed"] = True
                return replay
        prepared_config = (
            telemetry_by_rollout.get(str(requested_rollout_id))
            if requested_rollout_id is not None
            else None
        )
        if isinstance(telemetry, dict) and prepared_config is not None:
            requested_binding = (
                str(telemetry.get("transport") or "sse"),
                str(telemetry.get("retention") or "run"),
            )
            prepared_binding = (
                str(prepared_config.get("transport") or "sse"),
                str(prepared_config.get("retention") or "run"),
            )
            if requested_binding != prepared_binding:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "stream_binding_mismatch",
                        "prepared": {
                            "transport": prepared_binding[0],
                            "retention": prepared_binding[1],
                        },
                        "requested": {
                            "transport": requested_binding[0],
                            "retention": requested_binding[1],
                        },
                    },
                )
        result = await runtime.submit_rollout(request=payload)
        response = _coerce_rollout_payload(result)
        if isinstance(telemetry, dict) and telemetry.get("enabled"):
            rollout_id = str(
                response.get("rollout_id")
                or payload.get("rollout_id")
                or result.execution_id
            )
            bound = str(telemetry.get("transport") or "sse")
            stream_id = f"stream:{rollout_id}"
            log = event_logs.get(rollout_id)
            if log is None:
                log = _new_event_log(rollout_id, stream_id)
                event_logs[rollout_id] = log
                telemetry_by_rollout[rollout_id] = telemetry
                log.append_control(CONTROL_SUBSCRIBED, log.subscribed_payload())
            snapshot = await live_snapshot(rollout_id)
            if snapshot is not None:
                _append_snapshot(log, snapshot)
            descriptor = stream_descriptor(
                rollout_id=rollout_id,
                stream_id=stream_id,
                bound_transport=bound,
                retention=str(telemetry.get("retention") or "run"),
            )
            response["stream"] = descriptor
        response_rollout_id = str(
            response.get("rollout_id")
            or payload.get("rollout_id")
            or result.execution_id
        )
        request_identity = (
            str(requested_rollout_id)
            if requested_rollout_id is not None
            else response_rollout_id
        )
        start_requests[request_identity] = dict(payload)
        start_responses[request_identity] = dict(response)
        if response_rollout_id != request_identity:
            start_requests[response_rollout_id] = dict(payload)
            start_responses[response_rollout_id] = dict(response)
        return response

    @app.post("/rollouts/prepare")
    async def prepare_rollout(request: RolloutRequestModel) -> dict[str, Any]:
        telemetry = request.telemetry
        if telemetry is None or not telemetry.enabled:
            raise HTTPException(status_code=400, detail="prepare_requires_telemetry")
        rollout_id = request.rollout_id or f"roll_{uuid.uuid4().hex[:12]}"
        if rollout_id in event_logs:
            prepared = telemetry_by_rollout[rollout_id]
            requested_binding = (telemetry.transport, telemetry.retention)
            prepared_binding = (
                str(prepared.get("transport") or "sse"),
                str(prepared.get("retention") or "run"),
            )
            if requested_binding != prepared_binding:
                raise HTTPException(
                    status_code=409,
                    detail=f"rollout_prepare_identity_conflict:{rollout_id}",
                )
            return {
                "rollout_id": rollout_id,
                "stream": stream_descriptor(
                    rollout_id=rollout_id,
                    stream_id=event_logs[rollout_id].stream_id,
                    bound_transport=telemetry.transport,
                    retention=telemetry.retention,
                ),
                "replayed": True,
            }
        bound = telemetry.transport
        stream_id = f"stream:{rollout_id}"
        log = _new_event_log(rollout_id, stream_id)
        event_logs[rollout_id] = log
        telemetry_by_rollout[rollout_id] = telemetry.model_dump(mode="json")
        log.append_control(CONTROL_SUBSCRIBED, log.subscribed_payload())
        descriptor = stream_descriptor(
            rollout_id=rollout_id,
            stream_id=stream_id,
            bound_transport=bound,
            retention=telemetry.retention,
        )
        return {"rollout_id": rollout_id, "stream": descriptor}

    @app.get("/rollouts/{rollout_id}/stream")
    async def rollout_stream(rollout_id: str, request: Request) -> StreamingResponse:
        config = telemetry_by_rollout.get(rollout_id)
        log = event_logs.get(rollout_id)
        bound = str((config or {}).get("transport") or "")
        if config is None or log is None or bound not in {"sse", "websocket"}:
            raise HTTPException(status_code=404, detail=f"telemetry_not_enabled:{rollout_id}")
        raw_last = request.headers.get("last-event-id", "0")
        try:
            after = int(raw_last)
        except ValueError:
            after = 0
        interval = max(0.1, int(config.get("poll_interval_ms", 500)) / 1000)

        async def generate():
            nonlocal after
            while not await request.is_disconnected():
                await _ensure_log_current(rollout_id)
                for envelope in log.after(after):
                    event = live_event_from_envelope(rollout_id, envelope)
                    sse_id = envelope.sequence if envelope.sequence is not None else 0
                    if envelope.sequence is not None:
                        after = envelope.sequence
                    yield (
                        f"id: {sse_id}\n"
                        f"event: {event['kind']}\n"
                        f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                    )
                if log.closed:
                    break
                yield ": heartbeat\n\n"
                await asyncio.sleep(interval)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.websocket("/rollouts/{rollout_id}/ws")
    async def rollout_websocket(websocket: WebSocket, rollout_id: str) -> None:
        config = telemetry_by_rollout.get(rollout_id)
        log = event_logs.get(rollout_id)
        bound = str((config or {}).get("transport") or "")
        if config is None or log is None or bound != "websocket":
            await websocket.close(code=4404, reason="telemetry_not_enabled")
            return
        await websocket.accept()
        interval = max(0.1, int(config.get("poll_interval_ms", 500)) / 1000)
        after = 0
        try:
            while True:
                await _ensure_log_current(rollout_id)
                emitted = False
                for envelope in log.after(after):
                    event = live_event_from_envelope(rollout_id, envelope)
                    if envelope.sequence is not None:
                        after = envelope.sequence
                    await websocket.send_json(event)
                    emitted = True
                    if envelope.kind == "eval.run.terminal":
                        await websocket.close(code=1000)
                        return
                if log.closed:
                    await websocket.close(code=1000)
                    return
                if not emitted:
                    await asyncio.sleep(interval)
        except WebSocketDisconnect:
            return

    @app.get("/rollouts/{rollout_id}")
    async def get_rollout(rollout_id: str) -> dict[str, Any]:
        result = await runtime.get_execution(rollout_id=rollout_id)
        if result is None:
            if rollout_id in start_responses:
                return {**start_responses[rollout_id], "started": True}
            if rollout_id in event_logs:
                return {
                    "rollout_id": rollout_id,
                    "status": "prepared",
                    "started": False,
                    "terminated": False,
                    "stream": stream_descriptor(
                        rollout_id=rollout_id,
                        stream_id=event_logs[rollout_id].stream_id,
                        bound_transport=str(
                            telemetry_by_rollout[rollout_id].get("transport") or "sse"
                        ),
                        retention=str(
                            telemetry_by_rollout[rollout_id].get("retention") or "run"
                        ),
                    ),
                }
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        payload = _coerce_rollout_payload(result)
        payload.setdefault("started", True)
        return payload

    @app.get("/rollouts/{rollout_id}/state")
    async def get_rollout_state(rollout_id: str) -> dict[str, Any]:
        result = await runtime.get_execution_state(rollout_id=rollout_id)
        if result is None:
            result = await runtime.get_execution(rollout_id=rollout_id)
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
        return {"rollout_id": rollout_id, "artifacts": payload["artifacts"]}

    @app.get("/rollouts/{rollout_id}/annotations")
    async def get_rollout_annotations(rollout_id: str) -> dict[str, Any]:
        return (await _resolve_rollout_annotations(runtime, rollout_id)).to_dict()

    @app.get("/rollouts/{rollout_id}/events")
    async def get_rollout_events(
        rollout_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=10_000),
    ) -> dict[str, Any]:
        log = event_logs.get(rollout_id)
        if log is not None:
            await _ensure_log_current(rollout_id)
            available = log.after(after)
            controls = [item for item in available if item.sequence is None]
            evidence = [item for item in available if item.sequence is not None]
            envelopes = [*controls, *evidence[:limit]]
            return {
                "rollout_id": rollout_id,
                "stream_id": log.stream_id,
                "cursor": {
                    "kind": "sequence",
                    "after": after,
                    "high_water": log.high_water,
                    "closed": log.closed,
                    "next": max(
                        [after, *(item.sequence for item in envelopes if item.sequence is not None)]
                    ),
                    "has_more": len(evidence) > limit,
                },
                "events": [live_event_from_envelope(rollout_id, item) for item in envelopes],
            }
        payload = await get_rollout(rollout_id)
        raw_trace = payload.get("trace")
        trace = dict(raw_trace) if isinstance(raw_trace, dict) else {}
        return {
            "rollout_id": rollout_id,
            "cursor": {"kind": "sequence", "after": after, "high_water": None, "closed": True},
            "events": trace.get("events") or trace.get("event_history") or [],
        }

    def _reward_record(rollout_id: str) -> dict[str, Any]:
        existing = reward_executions.get(rollout_id)
        if existing is None:
            return {"status": "absent", "reward": None, "rollout_id": rollout_id}
        return existing

    @app.get("/reward")
    async def get_reward_query(rollout_id: str = Query(...)) -> dict[str, Any]:
        return _reward_record(rollout_id)

    @app.get("/rollouts/{rollout_id}/reward")
    async def get_reward_path(rollout_id: str) -> dict[str, Any]:
        return _reward_record(rollout_id)

    @app.post("/reward")
    async def post_reward(request: RewardRequestModel) -> dict[str, Any]:
        if request.evidence is not None:
            raise HTTPException(status_code=501, detail="provided_evidence_not_implemented")
        rollout_id = str(request.rollout_id)
        if request.rescore is False and rollout_id in reward_executions:
            return reward_executions[rollout_id]
        if request.mode == "provisional":
            raise HTTPException(
                status_code=409,
                detail={"status": "refused", "reason": "live_reward_unsupported"},
            )
        execution = await runtime.get_execution(rollout_id=rollout_id)
        if execution is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        terminal = str(execution.status).lower() in {
            "completed",
            "failed",
            "cancelled",
            "terminated",
            "stopped",
        }
        if not terminal:
            raise HTTPException(
                status_code=409,
                detail={"status": "incomplete", "missing_evidence": ["terminal_status"]},
            )
        reward = execution.outcome_reward()
        record = {
            "execution_id": f"eval_{rollout_id}",
            "rollout_id": rollout_id,
            "status": "scored" if reward is not None else "absent",
            "reward": reward,
            "node_results": [
                {
                    "node_id": "env_reward",
                    "kind": "env_reward",
                    "authority": "environment",
                    "status": "scored" if reward is not None else "skipped",
                    "value": reward,
                }
            ],
        }
        reward_executions[rollout_id] = record
        return record

    @app.get("/rollouts/{rollout_id}/trace")
    async def get_rollout_trace(rollout_id: str) -> dict[str, Any]:
        payload = await get_rollout(rollout_id)
        raw_trace = payload.get("trace")
        trace = dict(raw_trace) if isinstance(raw_trace, dict) else {}
        return {"rollout_id": rollout_id, **trace}

    @app.post("/rollouts/{rollout_id}/pause")
    async def pause_rollout(rollout_id: str, request: PauseRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.pause_execution(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.post("/rollouts/{rollout_id}/terminate")
    async def terminate_rollout(rollout_id: str, request: TerminateRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.terminate_execution(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_state_payload(runtime, result)

    @app.post("/rollouts/{rollout_id}/checkpoints")
    async def create_checkpoint(
        rollout_id: str, request: CreateCheckpointRequestModel
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.create_checkpoint(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_checkpoint_payload(result)

    @app.get("/rollouts/{rollout_id}/checkpoints")
    async def list_rollout_checkpoints(rollout_id: str) -> dict[str, Any]:
        rows = await runtime.list_checkpoints(rollout_id=rollout_id)
        return {
            "rollout_id": rollout_id,
            "checkpoints": _coerce_checkpoint_list(rows),
        }

    @app.get("/rollouts/{rollout_id}/checkpoints/{checkpoint_id}")
    async def get_rollout_checkpoint(rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
        result = await runtime.get_rollout_checkpoint(
            rollout_id=rollout_id,
            checkpoint_id=checkpoint_id,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.get("/checkpoints")
    async def list_checkpoints() -> dict[str, Any]:
        rows = await runtime.list_checkpoints(rollout_id=None)
        return {"checkpoints": _coerce_checkpoint_list(rows)}

    @app.get("/checkpoints/{checkpoint_id}/export")
    async def export_checkpoint(checkpoint_id: str) -> dict[str, Any]:
        handler = getattr(runtime, "export_checkpoint", None)
        if not callable(handler):
            raise HTTPException(status_code=404, detail="container_route_not_supported:export_checkpoint")
        result = handler(checkpoint_id=checkpoint_id)
        if isawaitable(result):
            result = await result
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        if not isinstance(result, Mapping):
            raise TypeError("runtime.export_checkpoint() must return a mapping")
        return dict(result)

    @app.post("/checkpoints/import")
    async def import_checkpoint(request: Request) -> dict[str, Any]:
        handler = getattr(runtime, "import_checkpoint", None)
        if not callable(handler):
            raise HTTPException(status_code=404, detail="container_route_not_supported:import_checkpoint")
        payload = await request.json()
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="checkpoint_import_request_must_be_object")
        try:
            result = handler(payload)
            if isawaitable(result):
                result = await result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"checkpoint_import_failed:{exc}") from exc
        if result is None:
            raise HTTPException(status_code=400, detail="checkpoint_import_failed")
        if not isinstance(result, CheckpointDescriptor):
            raise TypeError("runtime.import_checkpoint() must return CheckpointDescriptor")
        return _coerce_checkpoint_payload(result)

    @app.get("/checkpoints/{checkpoint_id}")
    async def get_checkpoint(checkpoint_id: str) -> dict[str, Any]:
        result = await runtime.get_checkpoint(checkpoint_id=checkpoint_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.post("/checkpoints/{checkpoint_id}/labels")
    async def update_checkpoint_labels(
        checkpoint_id: str,
        request: CheckpointLabelsRequestModel,
    ) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.update_checkpoint_labels(
            checkpoint_id=checkpoint_id,
            request=payload,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}")
        return _coerce_checkpoint_payload(result)

    @app.post("/rollouts/{rollout_id}/resume")
    @app.post("/rollouts/{rollout_id}/resume_async")
    @app.post("/rollouts/{rollout_id}/fork")
    async def resume_rollout(rollout_id: str, request: ResumeRequestModel) -> dict[str, Any]:
        payload = request.model_dump(mode="json", exclude_none=True)
        result = await runtime.resume_execution(rollout_id=rollout_id, request=payload)
        if result is None:
            raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
        return _coerce_rollout_payload(result)

    return app
