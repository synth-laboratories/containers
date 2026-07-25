"""Binding-scoped HTTP collector for events emitted across a container boundary."""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlparse
import ipaddress

from .collector import LocalCollector
from ..models.identity import TraceContextV1


class CollectorServer:
    def __init__(
        self,
        collector: LocalCollector,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = 32 * 1024 * 1024,
        collector_token: str | None = None,
        on_register_context: Callable[[TraceContextV1, dict[str, Any], dict[str, Any]], None]
        | None = None,
    ) -> None:
        self.collector = collector
        self.max_request_bytes = max_request_bytes
        self.collector_token = collector_token
        self.on_register_context = on_register_context
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host == "localhost"
        if not is_loopback and not collector_token:
            raise ValueError("a collector token is required for non-loopback collector hosts")
        self.is_loopback = is_loopback
        self._server = ThreadingHTTPServer((host, port), _handler(self))
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"synth-trace-collector-{collector.binding.capture_id}",
            daemon=True,
        )
        self._started = False
        self._stopped = False
        binding = collector.binding
        self._contexts: dict[str, TraceContextV1] = {
            binding.capture_id: binding.context_for_child()
        }

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> "CollectorServer":
        if self._started:
            return self
        self._thread.start()
        self._started = True
        return self

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # BaseServer.shutdown deadlocks when serve_forever was never started.
        if self._started:
            self._server.shutdown()
        self._server.server_close()

    def register_context(self, context: TraceContextV1) -> None:
        if context.trace_id != self.collector.binding.trace_id:
            raise ValueError("child context must join the collector trace")
        if not context.parent_actor_id or not context.delegation_id:
            raise ValueError("child context requires parent_actor_id and delegation_id")
        existing = self._contexts.get(context.capture_id)
        if existing is not None and existing != context:
            raise ValueError("child capture_id is already registered with different context")
        self._contexts[context.capture_id] = context


def _handler(server: CollectorServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            binding = server.collector.binding
            capture_id = self.headers.get("x-synth-capture-id") or ""
            context = server._contexts.get(capture_id)
            identity_matches = bool(
                context and self.headers.get("x-synth-trace-id") == binding.trace_id
            )
            if not identity_matches:
                return False
            if server.collector_token is None:
                return True
            return self.headers.get("authorization") == f"Bearer {server.collector_token}"

        def _context(self) -> TraceContextV1 | None:
            return server._contexts.get(self.headers.get("x-synth-capture-id") or "")

        def do_GET(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/healthz":
                self._json(404, {"error": "not_found"})
                return
            if server.is_loopback or self._authorized():
                self._json(200, {"ok": True})
                return
            self._json(403, {"error": "capture_binding_mismatch"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                self._json(403, {"error": "capture_binding_mismatch"})
                return
            length = int(self.headers.get("content-length") or "0")
            if length <= 0 or length > server.max_request_bytes:
                self._json(413, {"error": "invalid_request_size"})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": "invalid_json"})
                return
            path = urlparse(self.path).path
            if path == "/v1/contexts":
                try:
                    context = TraceContextV1(**dict(payload["context"]))
                    requester = self._context()
                    if requester is None or context.parent_actor_id != requester.actor_id:
                        raise ValueError("child parent_actor_id must match registering actor")
                    if context.parent_actor_session_id not in {
                        None,
                        requester.actor_session_id,
                    }:
                        raise ValueError(
                            "child parent_actor_session_id must match registering session"
                        )
                    actor = dict(payload["actor"])
                    session = dict(payload["session"])
                    if server.on_register_context is not None:
                        server.on_register_context(context, actor, session)
                    server.register_context(context)
                except (KeyError, TypeError, ValueError) as exc:
                    self._json(400, {"error": "invalid_child_context", "message": str(exc)})
                    return
                self._json(200, {"capture_id": context.capture_id, "registered": True})
                return
            context = self._context()
            if context is None:
                self._json(403, {"error": "unregistered_capture"})
                return
            if payload.get("actor_id") not in {None, context.actor_id} or payload.get(
                "session_id"
            ) not in {None, context.actor_session_id}:
                self._json(403, {"error": "child_identity_mismatch"})
                return
            try:
                if path == "/v1/events":
                    envelope_id = server.collector.event(
                        event_type=str(payload["event_type"]),
                        payload=dict(payload.get("payload") or {}),
                        actor_id=payload.get("actor_id"),
                        session_id=payload.get("session_id"),
                        occurred_at=payload.get("occurred_at"),
                        caused_by=tuple(payload.get("caused_by") or ()),
                        structural=payload.get("structural"),
                    )
                    self._json(200, {"envelope_id": envelope_id})
                    return
                if path == "/v1/artifacts":
                    content = base64.b64decode(str(payload["content_base64"]), validate=True)
                    artifact_id = server.collector.artifact(
                        role=str(payload["role"]),
                        media_type=str(payload["media_type"]),
                        content=content,
                        logical_name=str(payload["logical_name"]),
                        visibility=str(payload.get("visibility") or "private"),
                        actor_id=payload.get("actor_id"),
                        session_id=payload.get("session_id"),
                    )
                    self._json(200, {"artifact_id": artifact_id})
                    return
            except (KeyError, TypeError, ValueError) as exc:
                self._json(400, {"error": "invalid_payload", "message": str(exc)})
                return
            self._json(404, {"error": "not_found"})

    return Handler


__all__ = ["CollectorServer"]
