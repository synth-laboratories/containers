"""OpenAI Responses WebSocket relay with one in-flight response per connection."""

from __future__ import annotations

import json
import asyncio
import threading
from collections.abc import Callable, Mapping
from typing import Any

import websockets.asyncio.client
import websockets.asyncio.server

from ..canonical import bytes_digest, canonical_bytes, canonical_text
from ..models.capture_data import CapturedBodyRefV1, RawCaptureDisposition
from ..models.identity import TraceContextV1
from .envelope import RawRecordType
from .redaction import redact_payload
from .session import CaptureSession


RESPONSES_WEBSOCKET_URL = "wss://api.openai.com/v1/responses"


class ResponsesWebSocketRelay:
    def __init__(
        self,
        session: CaptureSession,
        *,
        upstream_url: str = RESPONSES_WEBSOCKET_URL,
        authorization: str | None = None,
        open_timeout: float = 30.0,
    ) -> None:
        self.session = session
        self.upstream_url = upstream_url
        self.authorization = authorization
        self.open_timeout = open_timeout

    async def relay(
        self,
        downstream: Any,
        *,
        actor_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        headers = _upstream_headers(
            getattr(getattr(downstream, "request", None), "headers", {}),
            authorization=self.authorization,
        )
        async with websockets.asyncio.client.connect(
            self.upstream_url,
            additional_headers=headers or None,
            open_timeout=self.open_timeout,
        ) as upstream:
            in_flight = False
            call_id: str | None = None
            call_index = 0
            while True:
                request = await downstream.recv()
                wire_request = (
                    request
                    if isinstance(request, bytes)
                    else str(request).encode("utf-8")
                )
                body = json.loads(wire_request.decode("utf-8"))
                if body.get("type") != "response.create":
                    raise ValueError("Responses WebSocket client messages must be response.create")
                response = body.get("response") or {}
                if not isinstance(response, Mapping):
                    raise ValueError("Responses WebSocket response.create requires an object")
                if response.get("background"):
                    raise ValueError("background mode is unsupported over Responses WebSocket")
                if in_flight:
                    raise ValueError("only one response may be in flight per WebSocket")
                in_flight = True
                call_id = self.session.mint(
                    "call",
                    kind="responses_websocket",
                    key=call_index,
                )
                inline_request, request_ref, request_redaction = self._capture_json(
                    body,
                    wire=wire_request,
                    media_type="application/json",
                )
                self.session.append(
                    RawRecordType.MODEL_CALL_STARTED,
                    actor_id=actor_id,
                    session_id=session_id,
                    call_id=call_id,
                    payload={
                        "call_index": call_index,
                        "route": RESPONSES_WEBSOCKET_URL,
                        "provider_adapter": "openai_responses",
                        "stream": True,
                        "model": response.get("model"),
                        "request_digest": bytes_digest(wire_request),
                        "request_body": inline_request or {},
                        "request_body_ref": request_ref.to_dict(),
                        "redaction": request_redaction,
                        "transport": "websocket",
                    },
                )
                await upstream.send(request)
                frame_index = 0
                terminal = False
                async for message in upstream:
                    await downstream.send(message)
                    wire_message = (
                        message
                        if isinstance(message, bytes)
                        else str(message).encode("utf-8")
                    )
                    try:
                        loaded = json.loads(wire_message.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        loaded = {"type": "unsupported.websocket_frame"}
                    event = dict(loaded) if isinstance(loaded, Mapping) else {
                        "type": "unsupported.websocket_frame",
                        "value": loaded,
                    }
                    frame_payload = self._capture_frame(
                        event,
                        wire=wire_message,
                        frame_index=frame_index,
                    )
                    self.session.append(
                        RawRecordType.RESPONSE_FRAME,
                        actor_id=actor_id,
                        session_id=session_id,
                        call_id=call_id,
                        sequence_in_call=frame_index,
                        payload=frame_payload,
                    )
                    frame_index += 1
                    event_type = str(event.get("type") or "")
                    if event_type in {"response.completed", "response.failed", "error"}:
                        response_payload = event.get("response")
                        usage = (
                            response_payload.get("usage")
                            if isinstance(response_payload, Mapping)
                            and isinstance(response_payload.get("usage"), Mapping)
                            else None
                        )
                        self.session.append(
                            RawRecordType.MODEL_CALL_FINISHED,
                            actor_id=actor_id,
                            session_id=session_id,
                            call_id=call_id,
                            payload={
                                "http_status": 200 if event_type == "response.completed" else 502,
                                "provider_adapter": "openai_responses",
                                "provider_terminal_observed": True,
                                "provider_status": event_type,
                                "usage": dict(usage) if usage is not None else None,
                                "usage_observed": usage is not None,
                                "frames": frame_index,
                                "transport": "websocket",
                            },
                        )
                        in_flight = False
                        terminal = True
                        break
                if not terminal:
                    self.session.append(
                        RawRecordType.MODEL_CALL_FINISHED,
                        actor_id=actor_id,
                        session_id=session_id,
                        call_id=call_id,
                        payload={
                            "http_status": 502,
                            "provider_adapter": "openai_responses",
                            "provider_terminal_observed": False,
                            "provider_status": "connection_closed",
                            "usage_observed": False,
                            "frames": frame_index,
                            "transport": "websocket",
                        },
                    )
                    raise RuntimeError(
                        "Responses WebSocket upstream closed before a terminal event"
                    )
                call_index += 1

    def _capture_json(
        self,
        payload: Mapping[str, Any],
        *,
        wire: bytes,
        media_type: str,
    ) -> tuple[dict[str, Any] | None, CapturedBodyRefV1, dict[str, Any]]:
        redacted, report = redact_payload(payload)
        safe = canonical_bytes(redacted)
        if len(safe) <= self.session.binding.policy.max_inline_bytes:
            return (
                dict(redacted),
                CapturedBodyRefV1(
                    wire_digest=bytes_digest(wire),
                    wire_byte_size=len(wire),
                    disposition=RawCaptureDisposition.REDACTED_INLINE,
                    media_type=media_type,
                    inline=redacted,
                    redaction_profile=report.profile,
                ),
                report.to_dict(),
            )
        stored_digest, uri = self.session.store_blob(safe)
        return (
            None,
            CapturedBodyRefV1(
                wire_digest=bytes_digest(wire),
                wire_byte_size=len(wire),
                disposition=RawCaptureDisposition.REDACTED_ARTIFACT,
                media_type=media_type,
                stored_digest=stored_digest,
                uri=uri,
                redaction_profile=report.profile,
            ),
            report.to_dict(),
        )

    def _capture_frame(
        self,
        event: Mapping[str, Any],
        *,
        wire: bytes,
        frame_index: int,
    ) -> dict[str, Any]:
        redacted, report = redact_payload(event)
        frame = f"data: {canonical_text(redacted)}\n\n"
        safe = frame.encode("utf-8")
        frame_ref = None
        if len(safe) > self.session.binding.policy.max_inline_bytes:
            stored_digest, uri = self.session.store_blob(safe)
            frame_ref = CapturedBodyRefV1(
                wire_digest=bytes_digest(wire),
                wire_byte_size=len(wire),
                disposition=RawCaptureDisposition.REDACTED_ARTIFACT,
                media_type="text/event-stream",
                stored_digest=stored_digest,
                uri=uri,
                redaction_profile=report.profile,
            ).to_dict()
        return {
            "frame_index": frame_index,
            "byte_size": len(wire),
            "digest": bytes_digest(wire),
            "frame": frame if frame_ref is None else "",
            "frame_ref": frame_ref,
            "redaction": report.to_dict(),
            "capture_boundary": "complete_websocket_message",
        }


class ResponsesWebSocketServer:
    def __init__(
        self,
        relay: ResponsesWebSocketRelay,
        *,
        context_resolver: Callable[[Any], TraceContextV1 | None],
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.relay = relay
        self.host = host
        self.requested_port = port
        self.context_resolver = context_resolver
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._server: Any = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def start() -> None:
            self._server = await websockets.asyncio.server.serve(
                self._handle_connection,
                self.host,
                self.requested_port,
            )
            self.port = int(self._server.sockets[0].getsockname()[1])
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    async def _handle_connection(self, connection: Any) -> None:
        if connection.request.path != "/v1/responses":
            await connection.close(code=1008, reason="unsupported path")
            return
        context = self.context_resolver(connection.request.headers)
        if context is None:
            await connection.close(code=1008, reason="capture context required")
            return
        await self.relay.relay(
            connection,
            actor_id=context.actor_id,
            session_id=context.actor_session_id,
        )

    def start(self) -> "ResponsesWebSocketServer":
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("Responses WebSocket server did not become ready")
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        async def shutdown() -> None:
            self._server.close()
            await self._server.wait_closed()

        future = asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        future.result(timeout=10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._server = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/v1/responses"


def _upstream_headers(
    headers: Mapping[str, str],
    *,
    authorization: str | None,
) -> dict[str, str]:
    excluded = {
        "connection",
        "host",
        "sec-websocket-accept",
        "sec-websocket-extensions",
        "sec-websocket-key",
        "sec-websocket-protocol",
        "sec-websocket-version",
        "upgrade",
    }
    forwarded = {
        str(name): str(value)
        for name, value in headers.items()
        if str(name).lower() not in excluded
        and not str(name).lower().startswith("x-synth-")
    }
    if authorization is not None:
        for name in tuple(forwarded):
            if name.lower() == "authorization":
                forwarded.pop(name)
        forwarded["authorization"] = authorization
    return forwarded


__all__ = [
    "RESPONSES_WEBSOCKET_URL",
    "ResponsesWebSocketRelay",
    "ResponsesWebSocketServer",
]
