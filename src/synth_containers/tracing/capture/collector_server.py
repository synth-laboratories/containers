"""Binding-scoped HTTP collector for events emitted across a container boundary."""

from __future__ import annotations

import base64
from datetime import datetime
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
import ipaddress
import secrets

from .collector import LocalCollector
from .live import page_from_spool, parse_live_cursor, sse_frame, status_from_spool
from ..canonical import utc_now
from ..models.actors import SessionStatus
from ..models.events import EventType
from ..models.identity import TraceContextV1


class SessionTerminalError(RuntimeError):
    """Raised when a delegated session writes after terminalization."""


class SessionActivityError(RuntimeError):
    """Raised when terminalization races active or unterminated descendants."""


class CollectorServer:
    def __init__(
        self,
        collector: LocalCollector,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_bytes: int = 32 * 1024 * 1024,
        collector_token: str | None = None,
        on_register_context: Callable[
            [TraceContextV1, dict[str, Any], dict[str, Any]],
            str | None,
        ]
        | None = None,
        on_finish_context: Callable[
            [TraceContextV1, SessionStatus | str, str | None],
            tuple[str, str, str],
        ]
        | None = None,
    ) -> None:
        self.collector = collector
        self.max_request_bytes = max_request_bytes
        self.collector_token = collector_token
        self.on_register_context = on_register_context
        self.on_finish_context = on_finish_context
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host == "localhost"
        if not is_loopback and not collector_token:
            raise ValueError("a collector token is required for non-loopback collector hosts")
        self.is_loopback = is_loopback
        self._server = ThreadingHTTPServer((host, port), _handler(self))
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.05),
            name=f"synth-trace-collector-{collector.binding.capture_id}",
            daemon=True,
        )
        self._started = False
        self._stopped = False
        binding = collector.binding
        self._contexts: dict[str, TraceContextV1] = {
            binding.capture_id: binding.context_for_child()
        }
        self._context_started_at: dict[str, str] = {}
        self._terminal_contexts: dict[str, tuple[str, str, str]] = {}
        self._active_contexts: dict[str, int] = {}
        self._context_tokens: dict[str, str | None] = {
            binding.capture_id: collector_token
        }
        self._accepting_mutations = True
        self._state_lock = threading.RLock()

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
        # CollectorServer is also used with small protocol fakes in tests and
        # adapters. Wake live spool readers when the concrete collector exposes
        # them, without making shutdown depend on that implementation detail.
        session = getattr(self.collector, "session", None)
        spool = getattr(session, "spool", None)
        wake_readers = getattr(spool, "wake_readers", None)
        if callable(wake_readers):
            wake_readers()
        # BaseServer.shutdown deadlocks when serve_forever was never started.
        if self._started:
            self._server.shutdown()
        self._server.server_close()

    def register_context(
        self,
        context: TraceContextV1,
        *,
        started_at: str | None = None,
        capability_token: str | None = None,
    ) -> str:
        if context.trace_id != self.collector.binding.trace_id:
            raise ValueError("child context must join the collector trace")
        if not context.parent_actor_id or not context.delegation_id:
            raise ValueError("child context requires parent_actor_id and delegation_id")
        if started_at is not None:
            _parse_timestamp(started_at, field="started_at")
        with self._state_lock:
            self._assert_mutable()
            existing = self._contexts.get(context.capture_id)
            if existing is not None and existing != context:
                raise ValueError(
                    "child capture_id is already registered with different context"
                )
            prior_started_at = self._context_started_at.get(context.capture_id)
            if (
                prior_started_at is not None
                and started_at is not None
                and prior_started_at != started_at
            ):
                raise ValueError(
                    "child capture_id is already registered with a different "
                    "session start"
                )
            prior_token = self._context_tokens.get(context.capture_id)
            if (
                prior_token is not None
                and capability_token is not None
                and not secrets.compare_digest(prior_token, capability_token)
            ):
                raise ValueError(
                    "child capture_id is already registered with a different capability"
                )
            token = prior_token or capability_token or secrets.token_urlsafe(32)
            self._contexts[context.capture_id] = context
            if started_at is not None:
                self._context_started_at[context.capture_id] = started_at
            self._context_tokens[context.capture_id] = token
            return token

    def unregister_context(self, context: TraceContextV1) -> None:
        """Roll back a just-authorized child whose durable registration failed."""

        with self._state_lock:
            if self._contexts.get(context.capture_id) != context:
                return
            if self._active_contexts.get(context.capture_id, 0):
                raise SessionActivityError("cannot unregister an active child context")
            self._contexts.pop(context.capture_id, None)
            self._context_started_at.pop(context.capture_id, None)
            self._context_tokens.pop(context.capture_id, None)
            self._terminal_contexts.pop(context.capture_id, None)

    def context_token(self, capture_id: str) -> str:
        """Return the ephemeral capability for an already registered context."""

        with self._state_lock:
            token = self._context_tokens.get(capture_id)
            if token is None:
                raise ValueError("registered context has no collector capability")
            return token

    def token_authorizes(self, capture_id: str, token: str) -> bool:
        with self._state_lock:
            expected = self._context_tokens.get(capture_id)
            return bool(expected and secrets.compare_digest(expected, token))

    def is_context_terminal(self, capture_id: str) -> bool:
        with self._state_lock:
            return capture_id in self._terminal_contexts

    def terminal_context_fact(
        self,
        capture_id: str,
    ) -> tuple[str, str, str] | None:
        with self._state_lock:
            return self._terminal_contexts.get(capture_id)

    def freeze(self) -> None:
        """Reject newly admitted writes while already leased calls drain."""

        with self._state_lock:
            self._accepting_mutations = False

    def begin_context_activity(
        self,
        context: TraceContextV1,
    ) -> Callable[[], None]:
        """Lease one child activity atomically against terminalization."""

        with self._state_lock:
            self._assert_context_open(context)
            self._active_contexts[context.capture_id] = (
                self._active_contexts.get(context.capture_id, 0) + 1
            )
        released = False
        release_lock = threading.Lock()

        def release() -> None:
            nonlocal released
            with release_lock:
                if released:
                    return
                released = True
            with self._state_lock:
                remaining = self._active_contexts.get(context.capture_id, 0) - 1
                if remaining <= 0:
                    self._active_contexts.pop(context.capture_id, None)
                else:
                    self._active_contexts[context.capture_id] = remaining

        return release

    def finish_context(
        self,
        context: TraceContextV1,
        *,
        status: SessionStatus | str,
        ended_at: str | None = None,
    ) -> tuple[str, str, str]:
        """Append or replay one authenticated terminal child-session fact."""

        normalized = str(status)
        if normalized not in {
            str(SessionStatus.COMPLETED),
            str(SessionStatus.FAILED),
            str(SessionStatus.INTERRUPTED),
        }:
            raise ValueError("child session status must be terminal")
        with self._state_lock:
            self._assert_mutable()
            registered = self._contexts.get(context.capture_id)
            if registered != context:
                raise ValueError("child context is not registered")
            if context.capture_id == self.collector.binding.capture_id:
                raise ValueError(
                    "root session lifecycle is owned by CaptureSupervisor.finalize"
                )
            if self._active_contexts.get(context.capture_id, 0):
                raise SessionActivityError(
                    "child session has in-flight capture activity"
                )
            unfinished_descendants = [
                item.actor_session_id
                for item in self._descendant_contexts(context)
                if item.capture_id not in self._terminal_contexts
            ]
            if unfinished_descendants:
                raise SessionActivityError(
                    "child session has unterminated descendants: "
                    + ", ".join(sorted(unfinished_descendants))
                )
            existing = self._terminal_contexts.get(context.capture_id)
            if existing is not None:
                prior_status, prior_ended_at, envelope_id = existing
                if prior_status != normalized or (
                    ended_at is not None and prior_ended_at != ended_at
                ):
                    raise ValueError(
                        "child session is already finished with a conflicting "
                        "terminal fact"
                    )
                return envelope_id, prior_status, prior_ended_at
            terminal_at = ended_at or utc_now()
            terminal_moment = _parse_timestamp(terminal_at, field="ended_at")
            started_at = self._context_started_at.get(context.capture_id)
            if started_at is not None and terminal_moment < _parse_timestamp(
                started_at,
                field="started_at",
            ):
                raise ValueError("child session ended_at precedes started_at")
            for descendant in self._descendant_contexts(context):
                descendant_terminal = self._terminal_contexts.get(
                    descendant.capture_id
                )
                if descendant_terminal is None:
                    continue
                descendant_ended_at = descendant_terminal[1]
                if _parse_timestamp(
                    descendant_ended_at,
                    field="descendant ended_at",
                ) > terminal_moment:
                    raise ValueError(
                        "child session ended_at precedes a descendant terminal fact"
                    )
            envelope_id, terminal_at = self.collector.finish_session(
                status=normalized,
                actor_id=context.actor_id,
                session_id=context.actor_session_id,
                ended_at=terminal_at,
            )
            self._terminal_contexts[context.capture_id] = (
                normalized,
                terminal_at,
                envelope_id,
            )
            return envelope_id, normalized, terminal_at

    def restore_terminal_context(
        self,
        context: TraceContextV1,
        *,
        status: SessionStatus | str,
        ended_at: str,
        envelope_id: str,
    ) -> None:
        """Rebuild the volatile terminal index from durable raw facts."""

        normalized = str(status)
        if normalized not in {
            str(SessionStatus.COMPLETED),
            str(SessionStatus.FAILED),
            str(SessionStatus.INTERRUPTED),
        }:
            raise ValueError("restored child session status must be terminal")
        _parse_timestamp(ended_at, field="ended_at")
        with self._state_lock:
            self._assert_context_open(context)
            unfinished_descendants = [
                item.actor_session_id
                for item in self._descendant_contexts(context)
                if item.capture_id not in self._terminal_contexts
            ]
            if unfinished_descendants:
                raise ValueError(
                    "restored child terminal precedes descendants: "
                    + ", ".join(sorted(unfinished_descendants))
                )
            terminal_moment = _parse_timestamp(ended_at, field="ended_at")
            started_at = self._context_started_at.get(context.capture_id)
            if started_at is not None and terminal_moment < _parse_timestamp(
                started_at,
                field="started_at",
            ):
                raise ValueError("restored child ended_at precedes started_at")
            for descendant in self._descendant_contexts(context):
                descendant_terminal = self._terminal_contexts.get(
                    descendant.capture_id
                )
                if descendant_terminal is None:
                    continue
                if _parse_timestamp(
                    descendant_terminal[1],
                    field="descendant ended_at",
                ) > terminal_moment:
                    raise ValueError(
                        "restored child ended_at precedes a descendant terminal"
                    )
            self._terminal_contexts[context.capture_id] = (
                normalized,
                ended_at,
                envelope_id,
            )

    def event(self, context: TraceContextV1, **values: Any) -> str:
        with self._state_lock:
            self._assert_context_open(context)
            return self.collector.event(
                **values,
                actor_id=context.actor_id,
                session_id=context.actor_session_id,
            )

    def artifact(self, context: TraceContextV1, **values: Any) -> str:
        with self._state_lock:
            self._assert_context_open(context)
            return self.collector.artifact(
                **values,
                actor_id=context.actor_id,
                session_id=context.actor_session_id,
            )

    def _assert_context_open(self, context: TraceContextV1) -> None:
        self._assert_mutable()
        if self._contexts.get(context.capture_id) != context:
            raise ValueError("child context is not registered")
        if context.capture_id in self._terminal_contexts:
            raise SessionTerminalError("child session is already terminal")
        for ancestor in self._ancestor_contexts(context):
            if ancestor.capture_id in self._terminal_contexts:
                raise SessionTerminalError("ancestor child session is already terminal")

    def _assert_mutable(self) -> None:
        if not self._accepting_mutations:
            raise SessionActivityError("capture finalization has begun")

    def _ancestor_contexts(
        self,
        context: TraceContextV1,
    ) -> tuple[TraceContextV1, ...]:
        by_session = {
            item.actor_session_id: item
            for item in self._contexts.values()
        }
        ancestors: list[TraceContextV1] = []
        parent_id = context.parent_actor_session_id
        seen: set[str] = set()
        while parent_id:
            if parent_id in seen:
                raise ValueError("child context ancestry contains a cycle")
            seen.add(parent_id)
            parent = by_session.get(parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            parent_id = parent.parent_actor_session_id
        return tuple(ancestors)

    def _descendant_contexts(
        self,
        context: TraceContextV1,
    ) -> tuple[TraceContextV1, ...]:
        descendants: list[TraceContextV1] = []
        frontier = [context.actor_session_id]
        seen: set[str] = set()
        while frontier:
            parent_id = frontier.pop()
            if parent_id in seen:
                raise ValueError("child context descendants contain a cycle")
            seen.add(parent_id)
            children = [
                item
                for item in self._contexts.values()
                if item.parent_actor_session_id == parent_id
            ]
            descendants.extend(children)
            frontier.extend(item.actor_session_id for item in children)
        return tuple(descendants)


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


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
            with server._state_lock:
                context = server._contexts.get(capture_id)
                expected_token = server._context_tokens.get(capture_id)
            actor_id = self.headers.get("x-synth-actor-id")
            session_id = self.headers.get("x-synth-session-id")
            # Pre-v5 emitters did not send the explicit identity pair. The
            # capture-scoped bearer capability already selects exactly one
            # registered context, so infer both only when both are absent.
            # A partial or conflicting declaration remains an auth failure.
            if context is not None and actor_id is None and session_id is None:
                actor_id = context.actor_id
                session_id = context.actor_session_id
            identity_matches = bool(
                context
                and self.headers.get("x-synth-trace-id") == binding.trace_id
                and actor_id == context.actor_id
                and session_id == context.actor_session_id
            )
            if not identity_matches:
                return False
            if expected_token is None:
                return True
            provided = self.headers.get("authorization") or ""
            return secrets.compare_digest(provided, f"Bearer {expected_token}")

        def _context(self) -> TraceContextV1 | None:
            with server._state_lock:
                return server._contexts.get(
                    self.headers.get("x-synth-capture-id") or ""
                )

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                if server.is_loopback or self._authorized():
                    self._json(200, {"ok": True})
                    return
                self._json(403, {"error": "capture_binding_mismatch"})
                return
            if parsed.path not in {
                "/v1/events",
                "/v1/events/stream",
                "/v1/live-manifest",
            }:
                self._json(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._json(403, {"error": "capture_binding_mismatch"})
                return
            query = parse_qs(parsed.query)
            try:
                cursor_value = (
                    query.get("after_ordinal", [None])[0]
                    or self.headers.get("last-event-id")
                )
                after_ordinal = parse_live_cursor(
                    cursor_value,
                    capture_id=server.collector.binding.capture_id,
                )
                limit = int(query.get("limit", ["256"])[0])
                if limit < 1 or limit > 10_000:
                    raise ValueError("limit must be between 1 and 10000")
            except (TypeError, ValueError) as exc:
                self._json(400, {"error": "invalid_live_cursor", "message": str(exc)})
                return
            spool = server.collector.session.spool
            if parsed.path == "/v1/live-manifest":
                self._json(200, status_from_spool(spool).to_dict())
                return
            if parsed.path == "/v1/events":
                self._json(
                    200,
                    page_from_spool(
                        spool,
                        after_ordinal=after_ordinal,
                        limit=limit,
                    ).to_dict(),
                )
                return
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.send_header("x-accel-buffering", "no")
            self.end_headers()
            cursor = after_ordinal
            try:
                while not server._stopped:
                    page = page_from_spool(
                        spool,
                        after_ordinal=cursor,
                        limit=limit,
                    )
                    for envelope in page.records:
                        self.wfile.write(sse_frame(envelope))
                        cursor = envelope.ordinal
                    if page.records:
                        self.wfile.flush()
                    if page.closed and cursor >= page.high_water_ordinal:
                        return
                    snapshot = spool.wait_for_change(
                        cursor,
                        timeout_seconds=15.0,
                    )
                    if (
                        snapshot.high_water_ordinal <= cursor
                        and not snapshot.closed
                    ):
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

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
                release_activity: Callable[[], None] | None = None
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
                    release_activity = server.begin_context_activity(requester)
                    capability: str | None = None
                    if server.on_register_context is not None:
                        capability = server.on_register_context(
                            context,
                            actor,
                            session,
                        )
                    if capability is None:
                        capability = server.register_context(
                            context,
                            started_at=(
                                str(session["started_at"])
                                if session.get("started_at") is not None
                                else None
                            )
                        )
                except (SessionActivityError, SessionTerminalError) as exc:
                    self._json(409, {"error": "session_terminal", "message": str(exc)})
                    return
                except (KeyError, TypeError, ValueError) as exc:
                    self._json(400, {"error": "invalid_child_context", "message": str(exc)})
                    return
                finally:
                    if release_activity is not None:
                        release_activity()
                self._json(
                    200,
                    {
                        "capture_id": context.capture_id,
                        "collector_token": capability,
                        "registered": True,
                    },
                )
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
                if path == "/v1/sessions/finish":
                    requested_status = str(payload["status"])
                    requested_ended_at = (
                        str(payload["ended_at"])
                        if payload.get("ended_at") is not None
                        else None
                    )
                    if server.on_finish_context is not None:
                        envelope_id, status, ended_at = server.on_finish_context(
                            context,
                            requested_status,
                            requested_ended_at,
                        )
                    else:
                        envelope_id, status, ended_at = server.finish_context(
                            context,
                            status=requested_status,
                            ended_at=requested_ended_at,
                        )
                    self._json(
                        200,
                        {
                            "envelope_id": envelope_id,
                            "session_id": context.actor_session_id,
                            "status": status,
                            "ended_at": ended_at,
                        },
                    )
                    return
                if path == "/v1/events":
                    if str(payload["event_type"]) == str(EventType.SESSION_FINISHED):
                        raise ValueError(
                            "session.finished must use /v1/sessions/finish"
                        )
                    envelope_id = server.event(
                        context,
                        event_type=str(payload["event_type"]),
                        payload=dict(payload.get("payload") or {}),
                        occurred_at=payload.get("occurred_at"),
                        caused_by=tuple(payload.get("caused_by") or ()),
                        structural=payload.get("structural"),
                    )
                    self._json(200, {"envelope_id": envelope_id})
                    return
                if path == "/v1/artifacts":
                    content = base64.b64decode(str(payload["content_base64"]), validate=True)
                    artifact_id = server.artifact(
                        context,
                        role=str(payload["role"]),
                        media_type=str(payload["media_type"]),
                        content=content,
                        logical_name=str(payload["logical_name"]),
                        visibility=str(payload.get("visibility") or "private"),
                    )
                    self._json(200, {"artifact_id": artifact_id})
                    return
            except (SessionActivityError, SessionTerminalError) as exc:
                self._json(409, {"error": "session_not_writable", "message": str(exc)})
                return
            except (KeyError, TypeError, ValueError) as exc:
                self._json(400, {"error": "invalid_payload", "message": str(exc)})
                return
            self._json(404, {"error": "not_found"})

    return Handler


__all__ = [
    "CollectorServer",
    "SessionActivityError",
    "SessionTerminalError",
]
