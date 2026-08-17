"""HTTP façade for the containers-compat platform."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..event_log import (
    STREAM_HEARTBEAT_INTERVAL_S,
    STREAM_TERMINAL_GRACE_S,
    RolloutEventLog,
)
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
from .state import CompatPlatform, PolicyConfig
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
    if result.get("accepted") is True:
        return JSONResponse(status_code=202, content=result)
    return result


def _http_from_parse(exc: BaseException) -> HTTPException:
    status = exc.status_code if isinstance(exc, RequestParseError) else 422
    return HTTPException(status_code=status, detail=str(exc))


def create_compat_app(
    target: str | TargetSpec = "craftax_engine",
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
        return platform.health_payload()

    @app.get("/metadata")
    @app.get("/info")
    async def metadata() -> dict[str, Any]:
        payload = platform.metadata_payload()
        if spec.runtime_family.value == "healthbench":
            existing = payload.get("capabilities")
            overlay = {
                "contract_version": "container_contract.v1",
                "rollout_modes": ["blocking"],
                "metadata": {"policy_ready": True},
            }
            if isinstance(existing, dict):
                payload["capabilities"] = {**existing, **overlay}
            else:
                payload["capabilities"] = overlay
            payload["metadata"] = {
                "optimizer_contracts": {
                    "gepa": {
                        "version": "synth_optimizers.gepa.v2",
                        "program_route": "/program",
                        "taskset_route": "/taskset",
                        "taskset_tasks_route": "/taskset/tasks",
                        "rollout_route": "/rollout",
                        "trace_route": "/rollouts/{rollout_id}/events",
                    }
                }
            }
        return payload

    @app.get("/task_info")
    async def task_info() -> Any:
        if spec.runtime_family.value != "healthbench":
            raise HTTPException(status_code=404, detail="task_info_not_supported")
        return {
            "task_id": "healthbench.response",
            "name": "HealthBench physician-rubric response",
            "objective": "maximize physician-rubric score",
            "output_kind": "open_text",
            "literal_training_targets": "forbidden",
            "proposer_hints": {
                "do_not_memorize": True,
                "focus": "safe, accurate, relevant, and appropriately escalated health responses",
            },
        }

    @app.get("/program")
    async def program() -> Any:
        if spec.runtime_family.value != "healthbench":
            raise HTTPException(status_code=404, detail="program_not_supported")
        seed_prompt = (
            "You are a careful health assistant. Give accurate, relevant, and safe guidance. "
            "State uncertainty, ask useful follow-up questions, and recommend appropriate "
            "professional or emergency care when warranted."
        )
        return {
            "version": "prompt_program.v1",
            "program_id": "healthbench.response.v1",
            "modules": [
                {
                    "module_id": "system_prompt",
                    "role": "system",
                    "mutable": True,
                    "candidate_field": "system_prompt",
                    "content": seed_prompt,
                }
            ],
            "target_modules": [
                {
                    "module_id": "system_prompt",
                    "candidate_field": "system_prompt",
                    "objective": "healthbench_physician_rubric_score",
                }
            ],
            "seed_candidate": {"system_prompt": seed_prompt},
            "rollout_task_id": "healthbench.response",
            "rollout_overlay_schema": {"system_prompt": "policy.system_message"},
        }

    @app.get("/taskset")
    async def taskset() -> Any:
        if spec.runtime_family.value != "healthbench":
            raise HTTPException(status_code=404, detail="taskset_not_supported")
        from .healthbench_world import rows

        count = len(rows())
        return {
            "taskset_id": "openai.healthbench.2025-05-07",
            "splits": {"train": count, "heldout": count},
            "source": "openai/healthbench",
            "metadata": {"gold_visibility": "environment_only", "rubric_authority": "physician"},
        }

    @app.post("/taskset/tasks")
    async def taskset_tasks(request: Request) -> Any:
        if spec.runtime_family.value != "healthbench":
            raise HTTPException(status_code=404, detail="taskset_tasks_not_supported")
        from .healthbench_world import load_row, public_observation

        body = await request.json()
        tasks = []
        for task_id in body.get("task_ids") or []:
            try:
                seed = int(str(task_id).rsplit(":", 1)[-1])
            except ValueError as exc:
                raise HTTPException(status_code=422, detail="invalid_healthbench_task_id") from exc
            row = load_row(seed)
            if row is None:
                raise HTTPException(status_code=404, detail=f"unknown_healthbench_task:{task_id}")
            tasks.append(
                {
                    "task_id": str(task_id),
                    "task_instance_id": f"seed:{seed}",
                    "seed": seed,
                    "split": str(body.get("split") or "train"),
                    "input": public_observation(row, seed),
                    "objective": "healthbench physician-rubric score",
                }
            )
        return {"tasks": tasks, "metadata": {"requested": len(tasks)}}

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
        result = await asyncio.to_thread(platform.start_rollout, req)
        return _platform_response(result)

    @app.post("/rollout", include_in_schema=False)
    async def optimizer_rollout(request: Request) -> Any:
        if spec.runtime_family.value != "healthbench":
            return await start(request)
        body = await request.json()
        task = body.get("task") if isinstance(body.get("task"), dict) else {}
        seed = int(task.get("seed") or 0)
        candidate = body.get("candidate") if isinstance(body.get("candidate"), dict) else {}
        policy = body.get("policy") if isinstance(body.get("policy"), dict) else {}
        policy_config = {
            "provider": policy.get("provider") or "groq",
            "model": policy.get("model") or "llama-3.1-8b-instant",
            "base_url": policy.get("base_url")
            or policy.get("api_base")
            or "https://api.groq.com/openai/v1",
            "api_key_env": policy.get("api_key_env") or "GROQ_API_KEY",
            "max_tokens": policy.get("max_tokens") or 1536,
            "system_prompt": candidate.get("system_prompt"),
        }
        config_digest = hashlib.sha256(
            json.dumps(policy_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        config_id = f"gepa_healthbench_{config_digest}"
        # The optimizer adapter may dispatch several immutable candidates at
        # once. Registering an additional future pin is not a mid-trial mutation
        # of any active pin; each rollout retains its own config id.
        platform.policy_configs.setdefault(
            config_id,
            PolicyConfig(
                config_id=config_id,
                harness="chat_completion",
                config=policy_config,
            ),
        )
        rollout_id = str(
            body.get("rollout_id")
            or body.get("trace_correlation_id")
            or f"roll_{uuid.uuid4().hex[:12]}"
        )
        telemetry = {"enabled": True, "transport": "poll", "retention": "run"}
        if rollout_id not in platform.logs:
            platform.prepare(rollout_id, "poll", "run")
        req = parse_create_rollout(
            {
                "rollout_id": rollout_id,
                "submission_mode": "sync",
                "slot": "stream",
                "telemetry": telemetry,
                "task_instance_id": f"seed:{seed}",
                "world_ref": "world:healthbench@eval",
                "evaluation_plan_ref": "healthbench_eval.v1",
                "policy_ref": {"harness": "chat_completion", "config": config_id},
            }
        )
        result = await asyncio.to_thread(platform.start_rollout, req)
        if "error" in result:
            return _platform_response(result)
        reward_record = platform.compute_reward(
            rollout_id=rollout_id,
            evidence=None,
            mode="terminal",
            rescore=False,
            plan_ref="healthbench_eval.v1",
            after_sequence=None,
        )
        reward = reward_record.get("reward")
        events = platform.events_payload(rollout_id, 0, 10_000).get("events", [])
        return {
            **result,
            "status": result.get("status"),
            "success_status": "success" if reward is not None else "failed",
            "reward_info": {"outcome_reward": reward, "metrics": {"healthbench_score": reward}},
            "trace": {"events": events, "prompt_assertions": body.get("prompt_assertions")},
            "events": events,
            "summary": {
                "seed": seed,
                "rubric_authority": "physician",
                "reward_status": reward_record.get("status"),
                "reward_reasons": reward_record.get("reasons", []),
            },
        }

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
    async def sse(rollout_id: str, request: Request) -> Any:
        log = platform.logs.get(rollout_id)
        if log is None or not platform.transport_is_bound(rollout_id, "sse"):
            raise HTTPException(status_code=404, detail=f"telemetry_not_enabled:{rollout_id}")
        busy = platform.acquire_stream(rollout_id)
        if busy:
            return JSONResponse(
                status_code=429,
                content=busy,
                headers={"Retry-After": str(int(busy["retry_after"]))},
            )
        raw_last = request.headers.get("last-event-id", "0")
        try:
            after = int(raw_last)
        except ValueError:
            # Invalid Last-Event-ID is a cursor reset, not a failed request: resume from sequence 0.
            after = 0

        async def generate():
            nonlocal after
            terminal_since: float | None = None
            try:
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
                    pin = platform.pins.get(rollout_id)
                    terminal = bool((pin is not None and pin.terminal) or log.closed)
                    now = time.monotonic()
                    if terminal and terminal_since is None:
                        terminal_since = now
                    if log.closed:
                        break
                    if (
                        terminal
                        and terminal_since is not None
                        and (now - terminal_since) >= STREAM_TERMINAL_GRACE_S
                    ):
                        break
                    if not emitted:
                        yield ": heartbeat\n\n"
                    await asyncio.sleep(STREAM_HEARTBEAT_INTERVAL_S)
            finally:
                platform.release_stream(rollout_id)

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
        busy = platform.acquire_stream(rollout_id)
        if busy:
            await websocket.close(code=1013, reason="stream_backpressure")
            return
        await websocket.accept()
        after = 0
        terminal_since: float | None = None
        try:
            while True:
                emitted = False
                for envelope in log.after(after):
                    if envelope.sequence is not None:
                        after = envelope.sequence
                    await websocket.send_json(_sse_event(rollout_id, envelope))
                    emitted = True
                pin = platform.pins.get(rollout_id)
                terminal = bool((pin is not None and pin.terminal) or log.closed)
                now = time.monotonic()
                if terminal and terminal_since is None:
                    terminal_since = now
                if log.closed:
                    await websocket.close(code=1000)
                    return
                if (
                    terminal
                    and terminal_since is not None
                    and (now - terminal_since) >= STREAM_TERMINAL_GRACE_S
                ):
                    await websocket.close(code=1000)
                    return
                if not emitted:
                    await websocket.send_json({"kind": "stream.heartbeat", "control": True})
                await asyncio.sleep(STREAM_HEARTBEAT_INTERVAL_S)
        except WebSocketDisconnect:
            return
        finally:
            platform.release_stream(rollout_id)

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

    @app.get("/rollouts/{rollout_id}/trace/bundle")
    async def get_trace_bundle(rollout_id: str) -> Any:
        result = platform.get_trace_bundle(rollout_id)
        if "error" in result:
            return _platform_response(result, default_status=404)
        return FileResponse(
            result["path"],
            media_type=result["media_type"],
            filename=f"{rollout_id}.trace-bundle.zip",
        )

    @app.get("/rollouts/{rollout_id}/manifest")
    async def get_manifest(rollout_id: str) -> Any:
        return _platform_response(platform.get_execution_manifest(rollout_id), default_status=404)

    @app.post("/rollouts/{rollout_id}/drop_session")
    async def drop_session(rollout_id: str) -> Any:
        result = platform.drop_session(rollout_id)
        return _platform_response(result)

    @app.post("/rollouts/{rollout_id}/cancel")
    async def cancel_rollout(rollout_id: str, request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            body = {}
        owner_id = None
        if isinstance(body, dict):
            owner_id = body.get("owner_id") or (body.get("metadata") or {}).get("owner_id")
        result = platform.cancel_rollout(
            rollout_id,
            owner_id=str(owner_id) if owner_id else None,
        )
        return _platform_response(result, default_status=404)

    @app.post("/cleanup")
    async def cleanup(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:
            body = {}
        owner_id = body.get("owner_id") if isinstance(body, dict) else None
        result = platform.cleanup_owned(str(owner_id) if owner_id else "")
        return _platform_response(result, default_status=422)

    return app
