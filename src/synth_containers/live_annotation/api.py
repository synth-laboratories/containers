"""HTTP surface: protocol install/read and per-rollout annotation stream routes."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
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
