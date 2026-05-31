from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .ontology import CONTRACT_VERSION
from .prompt_programs import gepa_optimizer_contract
from .serde import JsonObject, jsonable
from .wire import SubmissionMode


ContainerCallback = Callable[..., JsonObject | Awaitable[JsonObject]]


class JsonObjectPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_value(cls, value: Any, *, label: str) -> JsonObject:
        payload = jsonable(value)
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must return a JSON object, got {type(payload).__name__}")
        return cls.model_validate(payload).to_payload()

    def to_payload(self) -> JsonObject:
        return self.model_dump(mode="json")


class RolloutSubmissionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    rollout_id: str = Field(
        default_factory=lambda: f"rollout_{uuid.uuid4().hex[:12]}", min_length=1
    )
    submission_mode: SubmissionMode | None = None

    def resolved_submission_mode(self, container_default: SubmissionMode) -> SubmissionMode:
        if self.submission_mode is None:
            return container_default
        return self.submission_mode

    def to_payload(self) -> JsonObject:
        return JsonObjectPayload.from_value(
            self.model_dump(mode="json", exclude_none=True), label="rollout_submission"
        )


class CompletedRolloutPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    rollout_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    success_status: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)

    def record_for(self, rollout_id: str) -> "AsyncRolloutRecord":
        if self.rollout_id != rollout_id:
            raise ValueError(
                f"rollout_id mismatch: request={rollout_id!r} response={self.rollout_id!r}"
            )
        return AsyncRolloutRecord(
            payload=JsonObjectPayload.from_value(
                self.model_dump(mode="json"), label="completed_rollout"
            )
        )


class RolloutStatePayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rollout_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    success_status: str = Field(min_length=1)
    status_detail: str = ""
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
    summary: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> JsonObject:
        return JsonObjectPayload.from_value(self.model_dump(mode="json"), label="rollout_state")


class TasksetTasksPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    split: str = "train"
    task_ids: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)

    def to_payload(self) -> JsonObject:
        return JsonObjectPayload.from_value(self.model_dump(mode="json"), label="taskset_tasks")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tcp_port_ready(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise ValueError(f"container URL is missing a host: {url!r}")
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    elif parsed.scheme == "http":
        port = 80
    else:
        raise ValueError(f"container URL must use http or https: {url!r}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((parsed.hostname, port)) == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _call(callback: ContainerCallback, *args: object, label: str) -> JsonObject:
    value = callback(*args)
    if asyncio.iscoroutine(value):
        value = await value
    return JsonObjectPayload.from_value(value, label=label)


@dataclass(frozen=True, slots=True)
class ContainerConnection:
    """URL-only value passed from container lifecycle code to optimizers."""

    url: str

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("ContainerConnection.url is required")

    def to_dict(self) -> dict[str, str]:
        return {"url": self.url}


@dataclass(frozen=True, slots=True)
class ActionableSideInfo:
    """Optimizer-facing rollout evidence that is not itself an objective score."""

    payload: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return dict(self.payload)


@dataclass(frozen=True, slots=True)
class ObjectiveScore:
    objective: str
    value: float
    source: str = "container"
    rationale: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        payload: JsonObject = {
            "objective": self.objective,
            "value": float(self.value),
            "source": self.source,
        }
        if self.rationale is not None:
            payload["rationale"] = self.rationale
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class RolloutResult:
    rollout_id: str
    reward: float
    task_id: str = ""
    status: str = "completed"
    success_status: str = "succeeded"
    objective: str = "outcome_reward"
    objective_scores: list[ObjectiveScore] = field(default_factory=list)
    actionable_side_info: ActionableSideInfo | JsonObject | None = None
    reward_details: JsonObject = field(default_factory=dict)
    summary: JsonObject = field(default_factory=dict)
    usage: JsonObject = field(default_factory=dict)
    trace: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        objective_scores = self.objective_scores or [
            ObjectiveScore(
                objective=self.objective,
                value=self.reward,
                source="container.reward_info",
            )
        ]
        payload: JsonObject = {
            "rollout_id": self.rollout_id,
            "status": self.status,
            "success_status": self.success_status,
            "task_id": self.task_id,
            "reward_info": {
                "outcome_reward": float(self.reward),
                "event_rewards": [float(self.reward)],
                "details": {
                    "objective": self.objective,
                    **dict(self.reward_details),
                },
            },
            "objective_scores": [score.to_dict() for score in objective_scores],
            "summary": dict(self.summary),
            "usage": dict(self.usage),
            "trace": dict(self.trace),
            "metadata": dict(self.metadata),
            "created_at": _now(),
            "updated_at": _now(),
        }
        if self.actionable_side_info is not None:
            if isinstance(self.actionable_side_info, ActionableSideInfo):
                payload["actionable_side_info"] = self.actionable_side_info.to_dict()
            else:
                payload["actionable_side_info"] = dict(self.actionable_side_info)
        return JsonObjectPayload.from_value(payload, label="rollout_result")


@dataclass(frozen=True, slots=True)
class AsyncRolloutRecord:
    payload: JsonObject

    def state_payload(self) -> JsonObject:
        return RolloutStatePayload.model_validate(self.payload).to_payload()


class ContainerHandle:
    def __init__(
        self,
        *,
        url: str,
        stop: Callable[[], None] | None = None,
        log_reader: Callable[[], str] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self._stop = stop
        self._log_reader = log_reader
        self._closed = False

    def connection(self) -> ContainerConnection:
        return ContainerConnection(url=self.url)

    def health(self, *, timeout_seconds: float = 5.0) -> JsonObject:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(f"{self.url}/health")
            response.raise_for_status()
            return JsonObjectPayload.from_value(response.json(), label="container /health")

    def logs(self) -> str:
        if self._log_reader is None:
            return ""
        return self._log_reader()

    def down(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._stop is not None:
            self._stop()

    def __enter__(self) -> "ContainerHandle":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.down()


class ContainerRunner:
    def __init__(
        self,
        target: "Container | FastAPI | None" = None,
        *,
        app: FastAPI | None = None,
        command: list[str] | None = None,
        url: str | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout_seconds: float = 30.0,
    ) -> None:
        self.target = target
        self.app = app
        self.command = [] if command is None else list(command)
        self.url = url
        self.cwd = Path(cwd) if cwd is not None else None
        self.env = {} if env is None else dict(env)
        self.host = host
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds

    def serve(self) -> ContainerHandle:
        if self.url:
            handle = ContainerHandle(url=self.url)
            if not self._wait_for_health(handle):
                raise TimeoutError(
                    f"container did not become healthy at {handle.url} within "
                    f"{self.startup_timeout_seconds:.1f}s"
                )
            return handle
        if self.command:
            return self._serve_command()
        app = self.app
        if app is None and isinstance(self.target, FastAPI):
            app = self.target
        if app is None and isinstance(self.target, Container):
            app = self.target.fastapi()
        if app is None:
            raise ValueError("ContainerRunner requires a Container, FastAPI app, command, or url")
        return self._serve_app(app)

    def _serve_app(self, app: FastAPI) -> ContainerHandle:
        port = _free_port() if self.port is None else self.port
        config = uvicorn.Config(
            app,
            host=self.host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, name=f"synth-container:{port}", daemon=True)
        thread.start()

        def stop() -> None:
            server.should_exit = True
            thread.join(timeout=5.0)

        handle = ContainerHandle(url=f"http://{self.host}:{port}", stop=stop)
        if not self._wait_for_health(handle):
            stop()
            raise TimeoutError(
                f"container did not become healthy at {handle.url} within "
                f"{self.startup_timeout_seconds:.1f}s"
            )
        return handle

    def _serve_command(self) -> ContainerHandle:
        if self.port is None and not self.url:
            raise ValueError("ContainerRunner(command=...) requires an explicit port or url")
        url = f"http://{self.host}:{self.port}" if self.url is None else self.url
        process = subprocess.Popen(
            self.command,
            cwd=str(self.cwd) if self.cwd is not None else None,
            env={**os.environ, **self.env} if self.env else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log_lines: list[str] = []

        def drain() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                log_lines.append(line)
                if len(log_lines) > 1000:
                    del log_lines[: len(log_lines) - 1000]

        drain_thread = threading.Thread(target=drain, name="synth-container-logs", daemon=True)
        drain_thread.start()

        def stop() -> None:
            if process.poll() is None:
                process.terminate()
                deadline = time.monotonic() + 5.0
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5.0)

        handle = ContainerHandle(url=url, stop=stop, log_reader=lambda: "".join(log_lines))
        if not self._wait_for_health(handle):
            stop()
            raise TimeoutError(
                f"container did not become healthy at {handle.url} within "
                f"{self.startup_timeout_seconds:.1f}s"
            )
        return handle

    def _wait_for_health(self, handle: ContainerHandle) -> bool:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if _tcp_port_ready(handle.url):
                handle.health(timeout_seconds=1.0)
                return True
            time.sleep(0.1)
        return False


class Container:
    """Author a Synth task container from Python callables."""

    def __init__(
        self,
        name: str,
        *,
        runtime_id: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
        policy_ready: bool = True,
        default_submission_mode: str = "async",
    ) -> None:
        if not name.strip():
            raise ValueError("Container.name is required")
        self.name = name
        self.runtime_id = name if runtime_id is None else runtime_id
        self.description = description
        self.metadata = {} if metadata is None else dict(metadata)
        self.policy_ready = policy_ready
        self.default_submission_mode = SubmissionMode(default_submission_mode)
        self._metadata: ContainerCallback | None = None
        self._task_info: ContainerCallback | None = None
        self._taskset: ContainerCallback | None = None
        self._taskset_tasks: ContainerCallback | None = None
        self._rollout: ContainerCallback | None = None
        self._program: ContainerCallback | None = None

    def metadata_route(self, callback: ContainerCallback) -> ContainerCallback:
        self._metadata = callback
        return callback

    def task_info(self, callback: ContainerCallback) -> ContainerCallback:
        self._task_info = callback
        return callback

    def taskset(self, callback: ContainerCallback) -> ContainerCallback:
        self._taskset = callback
        return callback

    def taskset_tasks(self, callback: ContainerCallback) -> ContainerCallback:
        self._taskset_tasks = callback
        return callback

    def rollout(self, callback: ContainerCallback) -> ContainerCallback:
        self._rollout = callback
        return callback

    def program(self, callback: ContainerCallback) -> ContainerCallback:
        self._program = callback
        return callback

    def beta_program(self, callback: ContainerCallback) -> ContainerCallback:
        return self.program(callback)

    def fastapi(self, *, title: str | None = None) -> FastAPI:
        app = FastAPI(title=title or self.name)
        records: dict[str, AsyncRolloutRecord] = {}
        tasks: dict[str, asyncio.Task[None]] = {}
        records_lock = asyncio.Lock()

        async def metadata_payload() -> dict[str, Any]:
            if self._metadata is not None:
                payload = await _call(self._metadata, label="metadata")
            else:
                payload = {}
            base = {
                "runtime": {
                    "runtime_id": self.runtime_id,
                    "name": self.name,
                    "description": self.description,
                },
                "capabilities": {
                    "contract_version": CONTRACT_VERSION,
                    "rollout_modes": ["blocking", "async"],
                    "route_hints": {
                        "metadata_routes": ["/metadata", "/info"],
                        "task_info_routes": ["/task_info"],
                        "program_routes": ["/program"],
                        "taskset_routes": ["/taskset", "/taskset/tasks"],
                        "rollout_routes": ["/rollout", "/rollouts"],
                        "state_routes": ["/rollouts/{rollout_id}/state"],
                    },
                    "metadata": {
                        "policy_ready": self.policy_ready,
                        "program_ready": self._program is not None,
                    },
                },
                "metadata": _deep_merge(
                    {"optimizer_contracts": {"gepa": gepa_optimizer_contract()}},
                    dict(self.metadata),
                ),
            }
            return _deep_merge(base, payload)

        async def task_info_payload() -> dict[str, Any]:
            if self._task_info is None:
                return {
                    "task": {
                        "task_id": self.runtime_id,
                        "task_name": self.name,
                        "description": self.description,
                    },
                    "metadata": {},
                }
            return await _call(self._task_info, label="task_info")

        async def taskset_payload() -> dict[str, Any]:
            if self._taskset is None:
                raise HTTPException(status_code=404, detail="container_route_not_supported:taskset")
            return await _call(self._taskset, label="taskset")

        async def taskset_tasks_payload(payload: JsonObject) -> JsonObject:
            if self._taskset_tasks is None:
                raise HTTPException(
                    status_code=404, detail="container_route_not_supported:taskset_tasks"
                )
            return await _call(self._taskset_tasks, payload, label="taskset_tasks")

        async def program_payload() -> dict[str, Any]:
            if self._program is None:
                raise HTTPException(status_code=404, detail="container_route_not_supported:program")
            return await _call(self._program, label="program")

        async def run_rollout(payload: JsonObject) -> JsonObject:
            if self._rollout is None:
                raise HTTPException(status_code=404, detail="container_route_not_supported:rollout")
            return await _call(self._rollout, payload, label="rollout")

        async def complete_async_rollout(rollout_id: str, payload: JsonObject) -> None:
            result = CompletedRolloutPayload.model_validate(await run_rollout(payload)).record_for(
                rollout_id
            )
            async with records_lock:
                records[rollout_id] = result
                tasks.pop(rollout_id, None)

        def observe_async_task(rollout_id: str, task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                running = RolloutStatePayload.model_validate(records[rollout_id].payload)
                records[rollout_id] = AsyncRolloutRecord(
                    payload={
                        "rollout_id": rollout_id,
                        "status": "failed",
                        "success_status": "error",
                        "status_detail": f"{type(exc).__name__}: {exc}",
                        "created_at": running.created_at,
                        "updated_at": _now(),
                        "summary": {},
                        "metadata": {
                            "error_code": "async_rollout_execution_failed",
                            "error_type": type(exc).__name__,
                        },
                    }
                )
                tasks.pop(rollout_id, None)

        @app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "status": "ok",
                "contract_version": CONTRACT_VERSION,
                "runtime": await metadata_payload(),
                "task_info": await task_info_payload(),
            }

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok", "contract_version": CONTRACT_VERSION}

        @app.get("/metadata")
        @app.get("/info")
        async def metadata() -> dict[str, Any]:
            return await metadata_payload()

        @app.get("/task_info")
        async def task_info() -> dict[str, Any]:
            return await task_info_payload()

        @app.get("/program")
        async def program() -> dict[str, Any]:
            return await program_payload()

        @app.get("/taskset")
        async def taskset() -> dict[str, Any]:
            return await taskset_payload()

        @app.post("/taskset/tasks")
        async def taskset_tasks(request: TasksetTasksPayload) -> dict[str, Any]:
            return await taskset_tasks_payload(request.to_payload())

        @app.post("/rollout")
        @app.post("/rollouts")
        async def rollout(submission: RolloutSubmissionPayload) -> dict[str, Any]:
            payload = submission.to_payload()
            mode = submission.resolved_submission_mode(self.default_submission_mode)
            if mode == SubmissionMode.SYNC:
                return (
                    CompletedRolloutPayload.model_validate(await run_rollout(payload))
                    .record_for(submission.rollout_id)
                    .payload
                )
            now = _now()
            initial = AsyncRolloutRecord(
                payload={
                    "rollout_id": submission.rollout_id,
                    "status": "running",
                    "success_status": "running",
                    "created_at": now,
                    "updated_at": now,
                    "summary": {},
                    "metadata": {"submission_mode": "async"},
                }
            )
            async with records_lock:
                records[submission.rollout_id] = initial
                task = asyncio.create_task(complete_async_rollout(submission.rollout_id, payload))
                task.add_done_callback(lambda done: observe_async_task(submission.rollout_id, done))
                tasks[submission.rollout_id] = task
            return initial.payload

        @app.get("/rollouts/{rollout_id}")
        async def get_rollout(rollout_id: str) -> dict[str, Any]:
            async with records_lock:
                if rollout_id not in records:
                    raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
                record = records[rollout_id]
            return dict(record.payload)

        @app.get("/rollouts/{rollout_id}/state")
        async def get_rollout_state(rollout_id: str) -> dict[str, Any]:
            async with records_lock:
                if rollout_id not in records:
                    raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
                record = records[rollout_id]
            return record.state_payload()

        @app.post("/rollouts/{rollout_id}/terminate")
        async def terminate_rollout(rollout_id: str) -> dict[str, Any]:
            async with records_lock:
                task = tasks.pop(rollout_id, None)
                if rollout_id not in records:
                    raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
                record = records[rollout_id]
                if task is not None:
                    task.cancel()
                updated = {
                    **record.payload,
                    "status": "terminated",
                    "success_status": "failed",
                    "status_detail": "terminated",
                    "updated_at": _now(),
                }
                record = AsyncRolloutRecord(payload=updated)
                records[rollout_id] = record
            return record.state_payload()

        return app

    def serve(
        self,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        startup_timeout_seconds: float = 30.0,
    ) -> ContainerHandle:
        return ContainerRunner(
            self,
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
        ).serve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
