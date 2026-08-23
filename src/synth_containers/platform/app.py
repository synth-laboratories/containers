"""HTTP façade for the containers-compat platform."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from ..nested import NestedError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..event_log import RolloutEventLog
from .http_requests import (
    RequestParseError,
    parse_combine_reward,
    parse_create_rollout,
    parse_policy_config,
    parse_prepare_rollout,
    parse_put_policy,
    parse_reward_post,
    to_policy_config_dict,
    to_put_policy_dict,
)
from .state import CompatPlatform
from .targets import TARGETS, TargetSpec


def _raise_platform(result: dict[str, Any]) -> dict[str, Any]:
    if "error" in result:
        status = int(result["status_code"]) if "status_code" in result else 400
        raise HTTPException(status_code=status, detail=result)
    return result


def _platform_response(result: dict[str, Any], *, default_status: int = 400) -> Any:
    if "error" in result:
        status = int(result["status_code"]) if "status_code" in result else default_status
        return JSONResponse(status_code=status, content=result)
    return result


def _http_from_parse(exc: BaseException) -> HTTPException:
    status = exc.status_code if isinstance(exc, RequestParseError) else 422
    return HTTPException(status_code=status, detail=str(exc))


def create_compat_app(
    target: str | TargetSpec = "openenv_echo",
    *,
    storage_root: str | Path | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> FastAPI:
    spec = TARGETS[target] if isinstance(target, str) else target
    platform = CompatPlatform(
        spec,
        storage_root=storage_root,
        runtime_config=runtime_config,
    )
    app = FastAPI(title=f"synth-containers-compat:{spec.target_id}")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.state.platform = platform
    app.state.spec = spec

    def _sse_event(rollout_id: str, envelope: Any) -> dict[str, Any]:
        row = envelope.to_dict()
        row["rollout_id"] = rollout_id
        return row

    @app.get("/health")
    async def health() -> dict[str, Any]:
        payload = {
            "status": "ok",
            "target": spec.target_id,
            "runtime_family": spec.runtime_family.value,
            "environment_ref": spec.environment_ref,
        }
        if spec.health_probe is not None:
            extra = spec.health_probe()
            if isinstance(extra, dict):
                payload.update(extra)
            if str(payload.get("status") or "ok") != "ok":
                return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/metadata")
    @app.get("/info")
    async def metadata() -> dict[str, Any]:
        payload = platform.metadata_payload()
        if spec.metadata_extra is not None:
            extra = spec.metadata_extra(payload)
            if isinstance(extra, dict):
                payload = extra
        return payload

    @app.post("/rollouts/prepare")
    async def prepare(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
            req = parse_prepare_rollout(body)
        except (RequestParseError, ValueError) as exc:
            # Translate once at the HTTP edge: parse failures are 422, not 500.
            raise _http_from_parse(exc) from exc
        if not req.telemetry.enabled:
            raise HTTPException(status_code=400, detail="prepare_requires_telemetry")
        if req.telemetry.enabled:
            bad = platform._refuse_transport(req.telemetry.transport)
            if bad:
                return _raise_platform(bad)
        rollout_id = req.rollout_id
        if not rollout_id:
            rollout_id = f"roll_{uuid.uuid4().hex[:12]}"
        retention = (
            req.telemetry.retention if req.telemetry.retention is not None else spec.retention
        )
        if rollout_id in platform.logs:
            if platform.logs[rollout_id].closed:
                raise HTTPException(status_code=409, detail=f"event_log_sealed:{rollout_id}")
            bound_transport, bound_retention = platform.stream_bindings.get(
                rollout_id, (req.telemetry.transport, retention)
            )
            if (bound_transport, bound_retention) != (req.telemetry.transport, retention):
                raise HTTPException(
                    status_code=409, detail=f"rollout_prepare_identity_conflict:{rollout_id}"
                )
            return {
                "rollout_id": rollout_id,
                "stream": platform.stream_descriptor_for(rollout_id),
                "replayed": True,
            }
        try:
            descriptor = platform.prepare(rollout_id, req.telemetry.transport, retention)
        except RuntimeError as exc:
            detail = str(exc)
            if detail.startswith(("event_log_sealed:", "event_log_unrecoverable:")):
                raise HTTPException(status_code=409, detail=detail) from exc
            raise
        return {"rollout_id": rollout_id, "stream": descriptor}

    @app.post("/rollouts")
    async def start(request: Request) -> Any:
        try:
            body = await request.json()
            req = parse_create_rollout(body)
        except (RequestParseError, ValueError) as exc:
            # Translate once at the HTTP edge: parse failures are 422, not 500.
            raise _http_from_parse(exc) from exc
        try:
            result = await asyncio.to_thread(platform.start_rollout, req)
        except NestedError as exc:
            # A rollout that named a trial the platform does not carry is a bad
            # request, not a platform fault. A 500 tells the caller nothing; the
            # known ids are in the message, so hand them back.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _platform_response(result)

    @app.post("/rollout", include_in_schema=False)
    async def optimizer_rollout(request: Request) -> Any:
        if spec.compat_rollout is not None:
            return await spec.compat_rollout(request, platform, spec)
        return await start(request)

    @app.post("/rollouts/{rollout_id}/complete")
    async def complete(rollout_id: str) -> Any:
        result = await asyncio.to_thread(platform.complete_rollout, rollout_id)
        return _platform_response(result)

    @app.get("/rollouts/{rollout_id}")
    async def rollout_status(rollout_id: str) -> Any:
        return _platform_response(platform.rollout_status(rollout_id), default_status=404)

    @app.get("/rollouts/{rollout_id}/events")
    async def poll_events(
        rollout_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=10_000),
    ) -> Any:
        result = platform.events_payload(rollout_id, after, limit)
        return _platform_response(result, default_status=404)

    @app.get("/rollouts/{rollout_id}/frames/{step}.png")
    async def frame_asset(rollout_id: str, step: int) -> FileResponse:
        try:
            frame_path = RolloutEventLog.frame_asset_path(platform.storage_root, rollout_id, step)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="frame_not_found") from exc
        if not frame_path.is_file():
            raise HTTPException(status_code=404, detail="frame_not_found")
        return FileResponse(frame_path, media_type="image/png")

    @app.get("/rollouts/{rollout_id}/stream")
    async def sse(rollout_id: str, request: Request) -> StreamingResponse:
        log = platform.logs.get(rollout_id)
        if log is None or not platform.transport_is_bound(rollout_id, "sse"):
            raise HTTPException(status_code=404, detail=f"telemetry_not_enabled:{rollout_id}")
        raw_last = request.headers.get("last-event-id", "0")
        try:
            after = int(raw_last)
        except ValueError:
            # Invalid Last-Event-ID is a cursor reset, not a failed request: resume from sequence 0.
            after = 0

        async def generate():
            nonlocal after
            while not await request.is_disconnected():
                emitted = False
                for envelope in log.after(after):
                    sse_id = envelope.sequence if envelope.sequence is not None else 0
                    if envelope.sequence is not None:
                        after = envelope.sequence
                    event = _sse_event(rollout_id, envelope)
                    yield (
                        f"id: {sse_id}\n"
                        f"event: {event['kind']}\n"
                        f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                    )
                    emitted = True
                if log.closed:
                    break
                # Heartbeats must not end the stream. Luna plan calls are idle
                # for seconds; cutting SSE after 1s dropped span.policy.data.
                if not emitted:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.websocket("/rollouts/{rollout_id}/ws")
    async def rollout_websocket(websocket: WebSocket, rollout_id: str) -> None:
        log = platform.logs.get(rollout_id)
        if log is None or not platform.transport_is_bound(rollout_id, "websocket"):
            await websocket.close(code=4404, reason="telemetry_not_enabled")
            return
        await websocket.accept()
        after = 0
        try:
            while True:
                emitted = False
                for envelope in log.after(after):
                    if envelope.sequence is not None:
                        after = envelope.sequence
                    await websocket.send_json(_sse_event(rollout_id, envelope))
                    emitted = True
                if log.closed:
                    await websocket.close(code=1000)
                    return
                if not emitted:
                    await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    @app.get("/reward")
    async def get_reward(rollout_id: str = Query(...)) -> dict[str, Any]:
        return platform.get_reward(rollout_id)

    @app.get("/rollouts/{rollout_id}/reward")
    async def get_reward_path(rollout_id: str) -> dict[str, Any]:
        return platform.get_reward(rollout_id)

    @app.post("/reward")
    async def post_reward(request: Request) -> Any:
        try:
            body = await request.json()
            req = parse_reward_post(body)
        except (RequestParseError, ValueError) as exc:
            # Translate once at the HTTP edge: parse failures are 422, not 500.
            raise _http_from_parse(exc) from exc
        result = platform.compute_reward(
            rollout_id=req.rollout_id,
            evidence=req.evidence,
            mode=req.mode,
            rescore=req.rescore,
            plan_ref=req.evaluation_plan_ref,
            after_sequence=req.after_sequence,
        )
        if "error" in result:
            status = int(result["status_code"]) if "status_code" in result else 400
            return JSONResponse(status_code=status, content=result)
        if "http_status" in result and result["http_status"] == 202:
            return JSONResponse(status_code=202, content=result)
        return result

    @app.post("/reward/combine")
    async def combine(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
            req = parse_combine_reward(body)
        except (RequestParseError, ValueError) as exc:
            # Translate once at the HTTP edge: parse failures are 422, not 500.
            raise _http_from_parse(exc) from exc
        return platform.product_combiner(req.bases, req.required)

    @app.get("/evaluations/{evaluation_id}")
    async def get_evaluation(evaluation_id: str) -> dict[str, Any]:
        row = platform.evaluations.get(evaluation_id) or platform.reward_by_execution_id.get(
            evaluation_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown_evaluation:{evaluation_id}")
        return row

    @app.get("/evaluations/{evaluation_id}/events")
    async def evaluation_events(evaluation_id: str) -> dict[str, Any]:
        events = platform.evaluation_logs.get(evaluation_id)
        row = platform.evaluations.get(evaluation_id) or platform.reward_by_execution_id.get(
            evaluation_id
        )
        if events is None and row is None:
            raise HTTPException(status_code=404, detail=f"unknown_evaluation:{evaluation_id}")
        status = row["status"] if row is not None and "status" in row else None
        return {
            "evaluation_id": evaluation_id,
            "status": status,
            "events": events or [],
        }

    @app.post("/policy-configs")
    @app.post("/policy-configs/{config_id}")
    async def policy_configs(request: Request, config_id: str | None = None) -> Any:
        try:
            body = await request.json()
            req = parse_policy_config(body, path_config_id=config_id)
        except (RequestParseError, ValueError) as exc:
            # Translate once at the HTTP edge: parse failures are 422, not 500.
            raise _http_from_parse(exc) from exc
        result = platform.register_policy_config(req.config_id, to_policy_config_dict(req))
        return _platform_response(result)

    @app.put("/policy")
    async def put_policy(request: Request) -> Any:
        try:
            body = await request.json()
            req = parse_put_policy(body)
        except (RequestParseError, ValueError) as exc:
            # Translate once at the HTTP edge: parse failures are 422, not 500.
            raise _http_from_parse(exc) from exc
        result = platform.put_policy(to_put_policy_dict(req))
        return _platform_response(result)

    @app.post("/policy/restart")
    async def restart_policy() -> dict[str, Any]:
        return platform.restart_policy()

    @app.post("/world/stop")
    async def world_stop() -> dict[str, Any]:
        return platform.world_stop()

    @app.post("/world/restart")
    async def world_restart() -> Any:
        return _platform_response(platform.restart_world())

    @app.get("/artifacts/{artifact_id}")
    async def get_artifact(artifact_id: str) -> dict[str, Any]:
        row = platform.artifact(artifact_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"unknown_artifact:{artifact_id}")
        payload = {k: v for k, v in row.items() if k != "bytes"}
        blob = row["bytes"] if "bytes" in row and row["bytes"] is not None else b""
        payload["size"] = len(blob)
        return payload

    @app.get("/rollouts/{rollout_id}/trace")
    async def get_trace(rollout_id: str) -> dict[str, Any]:
        seal = platform.seals.get(rollout_id)
        if seal is None:
            raise HTTPException(status_code=404, detail=f"trace_not_sealed:{rollout_id}")
        return seal

    @app.post("/rollouts/{rollout_id}/drop_session")
    async def drop_session(rollout_id: str) -> Any:
        result = platform.drop_session(rollout_id)
        return _platform_response(result)

    if spec.mount_routes is not None:
        spec.mount_routes(app, platform, spec)

    return app
