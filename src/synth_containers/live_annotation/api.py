"""HTTP surface: protocol install/read and per-rollout annotation stream routes."""

from __future__ import annotations

from typing import Any

import asyncio
import json

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from ..event_log import SSE_HEADERS, iter_sse
from .service import LiveAnnotationService


def _response(result: dict[str, Any], *, default_status: int = 400) -> Any:
    if "error" in result:
        status = int(result.get("status_code") or default_status)
        return JSONResponse(status_code=status, content=result)
    return result


def mount_live_annotation(app: FastAPI, service: LiveAnnotationService, *, platform: Any) -> None:
    @app.get("/annotation-protocol")
    async def get_annotation_protocol() -> dict[str, Any]:
        return service.protocol_state_payload()

    @app.put("/annotation-protocol")
    async def put_annotation_protocol(request: Request) -> Any:
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="PUT /annotation-protocol expects a JSON object")
        return _response(service.put_protocol(body))

    @app.get("/rollouts/{rollout_id}/annotations/events")
    async def annotation_events(
        rollout_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=1000, ge=1, le=10_000),
    ) -> Any:
        return _response(service.events_payload(rollout_id, after, limit), default_status=404)

    @app.get("/rollouts/{rollout_id}/annotations/stream")
    async def annotation_sse(rollout_id: str, request: Request) -> StreamingResponse:
        log = service.log_for(rollout_id)
        if log is None or not platform.transport_is_bound(rollout_id, "sse"):
            raise HTTPException(status_code=404, detail=f"annotation_stream_not_found:{rollout_id}")
        raw_last = request.headers.get("last-event-id", "0")
        try:
            after = int(raw_last)
        except ValueError:
            after = 0
        return StreamingResponse(
            iter_sse(
                log,
                request,
                after=after,
                extra={"rollout_id": rollout_id, "stream_id": log.stream_id},
            ),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/rollouts/{rollout_id}/annotations/control")
    async def annotation_control(rollout_id: str, request: Request) -> Any:
        body = await request.json()
        result = await asyncio.to_thread(service.control, rollout_id, body)
        if "error" in result:
            return _response(result, default_status=404)
        return JSONResponse(status_code=202 if result.get("accepted") else 422, content=result)

    @app.websocket("/rollouts/{rollout_id}/annotations/ws")
    async def annotation_websocket(websocket: WebSocket, rollout_id: str) -> None:
        """Duplex: envelopes flow out, control messages flow in, acks answer inline.

        The durable acknowledgement still lands on the stream (and therefore on
        this socket) so a consumer that only listens sees the same history as
        the one that spoke.
        """

        log = service.log_for(rollout_id)
        if log is None or not platform.transport_is_bound(rollout_id, "websocket"):
            await websocket.close(code=4404, reason="annotation_stream_not_found")
            return
        await websocket.accept()
        after = 0
        emitted_controls: set[str] = set()
        receiver = asyncio.create_task(websocket.receive_text())
        try:
            while True:
                emitted = False
                for envelope in log.after(after):
                    if envelope.sequence is not None:
                        after = envelope.sequence
                    elif envelope.control:
                        # Control records are unsequenced; send each exactly once.
                        key = f"{envelope.kind}:{envelope.digest}"
                        if key in emitted_controls:
                            continue
                        emitted_controls.add(key)
                    row = envelope.to_dict()
                    row["rollout_id"] = rollout_id
                    row["stream_id"] = log.stream_id
                    await websocket.send_json(row)
                    emitted = True
                if receiver.done():
                    text = receiver.result()
                    try:
                        body = json.loads(text)
                    except json.JSONDecodeError:
                        body = {"op": "invalid_json"}
                    ack = await asyncio.to_thread(service.control, rollout_id, body)
                    await websocket.send_json({"type": "control.ack", **ack})
                    receiver = asyncio.create_task(websocket.receive_text())
                    emitted = True
                if log.closed:
                    await websocket.close(code=1000)
                    return
                if not emitted:
                    await asyncio.wait({receiver}, timeout=0.05)
        except WebSocketDisconnect:
            return
        finally:
            receiver.cancel()
