"""OpenAI Responses WebSocket relay with one in-flight response per connection."""

from __future__ import annotations

import json
import asyncio
import threading
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import parse_qs, urlsplit

import websockets.asyncio.client
import websockets.asyncio.server

from ..adapters.openai_responses import (
    NORMALIZER_VERSION as RESPONSES_ADAPTER_VERSION,
)
from ..canonical import bytes_digest, canonical_bytes, canonical_text
from ...gen_ai import request_observation
from ..models.capture_data import CapturedBodyRefV1, RawCaptureDisposition
from ..models.identity import TraceContextV1
from .envelope import RawRecordType
from .redaction import redact_payload
from .session import CaptureSession
from .proxy import ProxyStats


RESPONSES_WEBSOCKET_URL = "wss://api.openai.com/v1/responses"


class ResponsesWebSocketRelay:
    def __init__(
        self,
        session: CaptureSession,
        *,
        upstream_url: str = RESPONSES_WEBSOCKET_URL,
        authorization: str | None = None,
        open_timeout: float = 30.0,
        stats: ProxyStats | None = None,
    ) -> None:
        self.session = session
        self.upstream_url = upstream_url
        self.authorization = authorization
        self.open_timeout = open_timeout
        self.stats = stats

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
        in_flight = False
        call_id: str | None = None
        frame_index = 0
        try:
            async with websockets.asyncio.client.connect(
                self.upstream_url,
                additional_headers=headers or None,
                open_timeout=self.open_timeout,
            ) as upstream:
                while True:
                    request = await downstream.recv()
                    wire_request = (
                        request
                        if isinstance(request, bytes)
                        else str(request).encode("utf-8")
                    )
                    body = json.loads(wire_request.decode("utf-8"))
                    if body.get("type") != "response.create":
                        raise ValueError(
                            "Responses WebSocket client messages must be response.create"
                        )
                    response = body.get("response") or {}
                    if not isinstance(response, Mapping):
                        raise ValueError(
                            "Responses WebSocket response.create requires an object"
                        )
                    if response.get("background"):
                        raise ValueError(
                            "background mode is unsupported over Responses WebSocket"
                        )
                    if in_flight:
                        raise ValueError(
                            "only one response may be in flight per WebSocket"
                        )
                    call_id, call_index = self.session.mint_call(
                        kind="responses_websocket",
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
                            "route": "/v1/responses",
                            "upstream_host": urlsplit(self.upstream_url).netloc,
                            "upstream_path": urlsplit(self.upstream_url).path,
                            "provider_adapter": "openai_responses",
                            "provider_adapter_version": RESPONSES_ADAPTER_VERSION,
                            "stream": True,
                            "model": response.get("model"),
                            "request_digest": bytes_digest(wire_request),
                            "request_body": inline_request or {},
                            "request_body_ref": request_ref.to_dict(),
                            "redaction": request_redaction,
                            "transport": "websocket",
                            **request_observation(
                                response if isinstance(response, Mapping) else body
                            ),
                        },
                    )
                    in_flight = True
                    if self.stats is not None:
                        self.stats.increment(calls_accepted=1)
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
                        event = (
                            dict(loaded)
                            if isinstance(loaded, Mapping)
                            else {
                                "type": "unsupported.websocket_frame",
                                "value": loaded,
                            }
                        )
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
                        if event_type in {
                            "response.completed",
                            "response.failed",
                            "error",
                        }:
                            response_payload = event.get("response")
                            usage = (
                                response_payload.get("usage")
                                if isinstance(response_payload, Mapping)
                                and isinstance(
                                    response_payload.get("usage"),
                                    Mapping,
                                )
                                else None
                            )
                            self.session.append(
                                RawRecordType.MODEL_CALL_FINISHED,
                                actor_id=actor_id,
                                session_id=session_id,
                                call_id=call_id,
                                payload={
                                    "http_status": (
                                        200
                                        if event_type == "response.completed"
                                        else 502
                                    ),
                                    "provider_adapter": "openai_responses",
                                    "provider_terminal_observed": True,
                                    "provider_status": event_type,
                                    "usage": (
                                        dict(usage)
                                        if usage is not None
                                        else None
                                    ),
                                    "usage_observed": usage is not None,
                                    "frames": frame_index,
                                    "transport": "websocket",
                                },
                            )
                            if self.stats is not None:
                                if event_type == "response.completed":
                                    completed = 1
                                    errored = 0
                                else:
                                    completed = 0
                                    errored = 1
                                self.stats.increment(
                                    calls_completed=completed,
                                    calls_errored=errored,
                                    calls_normalized=1,
                                    frames=frame_index,
                                )
                            in_flight = False
                            terminal = True
                            break
                    if not terminal:
                        self._finish_interrupted(
                            call_id=call_id,
                            actor_id=actor_id,
                            session_id=session_id,
                            frame_index=frame_index,
                            provider_status="connection_closed",
                        )
                        in_flight = False
                        raise RuntimeError(
                            "Responses WebSocket upstream closed before a terminal event"
                        )
        except asyncio.CancelledError:
            if in_flight and call_id is not None:
                self._finish_interrupted(
                    call_id=call_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    frame_index=frame_index,
                    provider_status="capture_shutdown",
                )
            raise
        except Exception:
            if in_flight and call_id is not None:
                self._finish_interrupted(
                    call_id=call_id,
                    actor_id=actor_id,
                    session_id=session_id,
                    frame_index=frame_index,
                    provider_status="transport_error",
                )
            raise

    def _finish_interrupted(
        self,
        *,
        call_id: str,
        actor_id: str | None,
        session_id: str | None,
        frame_index: int,
        provider_status: str,
    ) -> None:
        """Persist a terminal call fact before an interrupted relay releases."""

        self.session.append(
            RawRecordType.MODEL_CALL_FINISHED,
            actor_id=actor_id,
            session_id=session_id,
            call_id=call_id,
            payload={
                "http_status": 502,
                "provider_adapter": "openai_responses",
                "provider_terminal_observed": False,
                "provider_status": provider_status,
                "usage_observed": False,
                "frames": frame_index,
                "transport": "websocket",
            },
        )
        if self.stats is not None:
            self.stats.increment(
                calls_errored=1,
                calls_normalized=1,
                frames=frame_index,
            )

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
        query_context_resolver: Callable[[str], TraceContextV1 | None] | None = None,
        context_activity_begin: Callable[
            [TraceContextV1],
            Callable[[], None],
        ]
        | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.relay = relay
        self.host = host
        self.requested_port = port
        self.context_resolver = context_resolver
        self.query_context_resolver = query_context_resolver
        self.context_activity_begin = context_activity_begin
        self.port = 0
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._server: Any = None
        self._connection_tasks: set[asyncio.Task[Any]] = set()
        self._connections: dict[int, Any] = {}
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

        try:
            self._loop.run_until_complete(start())
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    async def _handle_connection(self, connection: Any) -> None:
        current_task = asyncio.current_task()
        if current_task is not None:
            self._connection_tasks.add(current_task)
        self._connections[id(connection)] = connection
        try:
            request_target = urlsplit(connection.request.path)
            if request_target.path != "/v1/responses":
                await connection.close(code=1008, reason="unsupported path")
                return
            context = self.context_resolver(connection.request.headers)
            declares_header_context = any(
                str(name).lower().startswith("x-synth-")
                for name in connection.request.headers
            )
            if (
                context is None
                and not declares_header_context
                and self.query_context_resolver is not None
            ):
                tokens = parse_qs(
                    request_target.query,
                    keep_blank_values=True,
                ).get("synth_trace_token", ())
                if len(tokens) == 1:
                    context = self.query_context_resolver(tokens[0])
            if context is None:
                await connection.close(
                    code=1008,
                    reason="capture context required",
                )
                return
            release_activity = None
            if self.context_activity_begin is not None:
                try:
                    release_activity = self.context_activity_begin(context)
                except (RuntimeError, ValueError):
                    await connection.close(
                        code=1008,
                        reason="capture context is terminal",
                    )
                    return
            try:
                await self.relay.relay(
                    connection,
                    actor_id=context.actor_id,
                    session_id=context.actor_session_id,
                )
            finally:
                if release_activity is not None:
                    release_activity()
        finally:
            if current_task is not None:
                self._connection_tasks.discard(current_task)
            self._connections.pop(id(connection), None)

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
            connections = tuple(self._connections.values())
            if connections:
                await asyncio.gather(
                    *(
                        connection.close(
                            code=1001,
                            reason="capture finalizing",
                        )
                        for connection in connections
                    ),
                    return_exceptions=True,
                )
            current_task = asyncio.current_task()
            connection_tasks = tuple(
                task
                for task in self._connection_tasks
                if task is not current_task
            )
            for task in connection_tasks:
                task.cancel()
            if connection_tasks:
                await asyncio.gather(
                    *connection_tasks,
                    return_exceptions=True,
                )
            await self._server.wait_closed()

        future = asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        shutdown_error: BaseException | None = None
        try:
            future.result(timeout=10)
        except BaseException as exc:
            shutdown_error = exc
        finally:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)
            self._server = None
        if self._thread.is_alive():
            raise RuntimeError("Responses WebSocket event loop did not stop")
        if shutdown_error is not None:
            raise RuntimeError(
                "Responses WebSocket server did not drain"
            ) from shutdown_error

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
