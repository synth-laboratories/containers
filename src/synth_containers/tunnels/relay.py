"""SynthTunnel relay agent and lease provider owned by synth-containers."""

from __future__ import annotations

from email.message import Message

import base64
import http.client
import logging
import socket
import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urljoin, urlparse, urlunparse

from .paths import (
    confined_origin_url,
    confined_forward_path as _confined_forward_path,  # noqa: F401 - compatibility re-export
)
from .protocol import DataFrame, DataKind, SUBPROTOCOL, CHUNK_BYTES
from .attachment import (
    Attachment,
    LeaseAttachment,
    ConnectorAttachment,
    LocalTarget as _LocalTarget,
)
from .connector_control import ConnectorAttachmentProvider, PreparedAttachment
from .targets import LocalTargetRegistry, TargetError
from .origin_connections import OriginConnections, OriginPoolScope


_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_LOCAL_ONLY_AUTH_HEADERS = {"authorization", "x-api-key", "x-api-keys"}
# Per-hop handshake headers. The agent handshakes with the origin using its own
# key and negotiates its own extensions, so none of these are forwarded.
# ``sec-websocket-protocol`` is deliberately absent from this set: it *is*
# forwarded, as offered subprotocols, and whatever the origin selects comes back
# in WS_OPEN_ACK.
_WEBSOCKET_HANDSHAKE_HEADERS = {
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-accept",
    "sec-websocket-extensions",
}
_WEBSOCKET_SUBPROTOCOL_HEADER = "sec-websocket-protocol"
# Response headers the agent owns rather than the origin. Everything else in a
# WS_OPEN_REJECT is forwarded verbatim -- notably ``upgrade`` and
# ``sec-websocket-version``, which are what tell a client to stop retrying the
# handshake against the Trace V5 capture proxy.
_WEBSOCKET_REJECT_FRAMING_HEADERS = {"connection", "content-length", "transfer-encoding"}

# One agent socket carries every stream for a lease, so a stream that blocks on a
# slow origin must never block the shared receive loop. Each open WebSocket gets
# this many queued outbound messages; on overflow the stream -- and only that
# stream -- is closed with 1013.
_WEBSOCKET_STREAM_QUEUE_FRAMES = 64
# Default ceiling on concurrently open tunnelled WebSockets. Counted separately
# from HTTP in-flight requests so a long-lived socket cannot consume the request
# concurrency budget.
_DEFAULT_MAX_WEBSOCKET_STREAMS = 64

_WS_CLOSE_NORMAL = 1000
_WS_CLOSE_GOING_AWAY = 1001  # lease expiry / deliberate agent shutdown
_WS_CLOSE_PROTOCOL_ERROR = 1002
_WS_CLOSE_MESSAGE_TOO_BIG = 1009
_WS_CLOSE_INTERNAL_ERROR = 1011
_WS_CLOSE_SERVICE_RESTART = 1012  # the agent lost its relay connection
_WS_CLOSE_TRY_AGAIN_LATER = 1013  # per-stream queue overflow


class SynthTunnelRelayError(RuntimeError):
    """The SynthTunnel relay could not attach or proxy a request safely."""


@runtime_checkable
class SynthTunnelControlPlane(Protocol):
    """Public control-plane operations needed by the container-owned provider."""

    def create_synth_lease(
        self,
        *,
        client_instance_id: str,
        local_host: str,
        local_port: int,
        requested_ttl_seconds: int,
        metadata: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> Mapping[str, Any]: ...

    def close_synth_lease(self, lease_id: str) -> object: ...


@dataclass(slots=True)
class _PendingRequest:
    method: str
    path: str
    query: str
    headers: list[tuple[str, str]]
    deadline_ms: int
    connection_generation: int
    received_monotonic: float
    body: bytearray = field(default_factory=bytearray)
    local_target: _LocalTarget | None = None
    origin_pool_scope: OriginPoolScope = None
    origin_headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)


class _OriginUpgradeRefused(Exception):
    """The local origin answered the WebSocket handshake with an HTTP response."""

    def __init__(self, status: int, headers: list[list[str]], body: bytes) -> None:
        super().__init__(f"origin refused websocket upgrade with HTTP {status}")
        self.status = status
        self.headers = headers
        self.body = body


@dataclass(slots=True)
class _WebSocketStream:
    rid: str
    path: str
    query: str
    headers: list[tuple[str, str]]
    deadline_ms: int
    connection_generation: int
    outbound: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=_WEBSOCKET_STREAM_QUEUE_FRAMES)
    )
    lock: threading.Lock = field(default_factory=threading.Lock)
    origin: Any | None = None
    local_target: _LocalTarget | None = None
    origin_headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    pending: bytearray = field(default_factory=bytearray)
    pending_opcode: str | None = None
    closed: bool = False


@dataclass(slots=True)
class AttachedSynthTunnelLease:
    """An attached relay agent plus the credentials used by remote workers."""

    lease_id: str
    public_url: str
    worker_token: str = field(repr=False)
    expires_at: str | None
    connector_mode: str
    _control_plane: SynthTunnelControlPlane = field(repr=False)
    _agent: "SynthTunnelRelayAgent" = field(repr=False)
    _local_health_url: str = field(repr=False)
    _attach_timeout_seconds: float = field(repr=False)
    _ready_timeout_seconds: float = field(repr=False)
    route_token: str | None = None
    diagnostics_hint: str | None = None
    _close_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _ready_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closing: threading.Event = field(default_factory=threading.Event, repr=False)
    _local_ready: bool = field(default=False, repr=False)
    _agent_started: bool = field(default=False, repr=False)
    _agent_stopped: bool = field(default=False, repr=False)
    _control_plane_closed: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, repr=False)

    def wait_ready(self, timeout_seconds: float | None = None) -> None:
        ready_timeout = self._ready_timeout_seconds if timeout_seconds is None else timeout_seconds
        if ready_timeout <= 0:
            raise ValueError("SynthTunnel readiness timeout must be positive")
        attach_timeout = (
            self._attach_timeout_seconds
            if timeout_seconds is None
            else min(self._attach_timeout_seconds, ready_timeout)
        )
        with self._ready_lock:
            if self._closed or self._closing.is_set():
                raise SynthTunnelRelayError("SynthTunnel lease is closed")
            local_ready = self._local_ready
        if not local_ready:
            _wait_for_http_ok(
                self._local_health_url,
                timeout_seconds=min(10.0, ready_timeout),
                cancel_event=self._closing,
            )
            with self._ready_lock:
                if self._closed or self._closing.is_set():
                    raise SynthTunnelRelayError("SynthTunnel lease is closed")
                self._local_ready = True
        with self._ready_lock:
            if self._closed or self._closing.is_set():
                raise SynthTunnelRelayError("SynthTunnel lease is closed")
            agent_started = self._agent_started
        if not agent_started:
            self._agent.start(
                timeout_seconds=attach_timeout,
                cancel_event=self._closing,
            )
            with self._ready_lock:
                if self._closed or self._closing.is_set():
                    raise SynthTunnelRelayError("SynthTunnel lease is closed")
                self._agent_started = True
        _wait_for_http_ok(
            _join_health_url(self.public_url),
            headers={"Authorization": f"Bearer {self.worker_token}"},
            timeout_seconds=ready_timeout,
            cancel_event=self._closing,
        )
        with self._ready_lock:
            if self._closed or self._closing.is_set():
                raise SynthTunnelRelayError("SynthTunnel lease is closed")

    def update_credentials(
        self,
        *,
        worker_token: str,
        expires_at: str | None,
        agent_token: str,
    ) -> None:
        """Adopt heartbeat-rotated credentials without replacing the relay."""

        normalized_worker_token = _required_text(worker_token, "worker token")
        normalized_agent_token = _required_text(agent_token, "agent token")
        with self._ready_lock:
            if self._closed or self._closing.is_set():
                raise SynthTunnelRelayError("SynthTunnel lease is closed")
            self.worker_token = normalized_worker_token
            self.expires_at = expires_at or self.expires_at
            self._agent.update_agent_token(normalized_agent_token)

    def close(self) -> None:
        self._closing.set()
        with self._close_lock:
            with self._ready_lock:
                if self._closed:
                    return
            errors: list[Exception] = []
            if not self._agent_stopped:
                try:
                    self._agent.stop()
                except Exception as error:
                    errors.append(error)
                else:
                    self._agent_stopped = True
            if not self._control_plane_closed:
                try:
                    self._control_plane.close_synth_lease(self.lease_id)
                except Exception as error:
                    errors.append(error)
                else:
                    self._control_plane_closed = True
            with self._ready_lock:
                self._closed = self._agent_stopped and self._control_plane_closed
            if errors:
                raise SynthTunnelRelayError(
                    "SynthTunnel lease cleanup failed: "
                    + ", ".join(type(error).__name__ for error in errors)
                ) from errors[0]


class SynthTunnelProvider:
    """Open fully attached SynthTunnel leases for a local container."""

    def __init__(
        self,
        *,
        control_plane: SynthTunnelControlPlane,
        client_instance_id: str | None = None,
        attach_timeout_seconds: float = 30.0,
        ready_timeout_seconds: float = 60.0,
        max_in_flight_requests: int = 256,
        max_request_body_bytes: int = 64 * 1024 * 1024,
        max_websocket_streams: int = _DEFAULT_MAX_WEBSOCKET_STREAMS,
    ) -> None:
        if attach_timeout_seconds <= 0 or ready_timeout_seconds <= 0:
            raise ValueError("SynthTunnel timeouts must be positive")
        if max_in_flight_requests <= 0:
            raise ValueError("max_in_flight_requests must be positive")
        if max_request_body_bytes <= 0:
            raise ValueError("max_request_body_bytes must be positive")
        if max_websocket_streams <= 0:
            raise ValueError("max_websocket_streams must be positive")
        self._control_plane = control_plane
        self._client_instance_id = client_instance_id or f"synth-containers-{uuid.uuid4().hex[:24]}"
        self._attach_timeout_seconds = attach_timeout_seconds
        self._ready_timeout_seconds = ready_timeout_seconds
        self._max_in_flight_requests = max_in_flight_requests
        self._max_request_body_bytes = max_request_body_bytes
        self._max_websocket_streams = max_websocket_streams

    def open_synth_tunnel(
        self,
        local_url: str,
        *,
        requested_ttl_seconds: int,
        metadata: Mapping[str, object],
        capabilities: Mapping[str, object],
        wait_ready: bool = True,
    ) -> AttachedSynthTunnelLease:
        target = _parse_local_target(local_url)
        local_health_url = _join_health_url(target.base_url)
        if wait_ready:
            _wait_for_http_ok(
                local_health_url,
                timeout_seconds=min(10.0, self._ready_timeout_seconds),
            )
        response = self._control_plane.create_synth_lease(
            client_instance_id=self._client_instance_id,
            local_host=target.host,
            local_port=target.port,
            requested_ttl_seconds=requested_ttl_seconds,
            metadata=dict(metadata),
            capabilities=dict(capabilities),
        )
        lease_id = _required_text(response.get("lease_id"), "lease_id")
        try:
            public_url = _required_text(response.get("public_url"), "public_url").rstrip("/")
            worker_token = _required_text(response.get("worker_token"), "worker_token")
            agent_connect = response.get("agent_connect")
            if not isinstance(agent_connect, Mapping):
                raise SynthTunnelRelayError("SynthTunnel lease response omitted agent_connect")
            agent = SynthTunnelRelayAgent(
                lease_id=lease_id,
                local_target=target,
                agent_connect=agent_connect,
                max_in_flight_requests=self._max_in_flight_requests,
                max_request_body_bytes=self._max_request_body_bytes,
                max_websocket_streams=self._max_websocket_streams,
            )
        except Exception as response_error:
            try:
                self._control_plane.close_synth_lease(lease_id)
            except Exception as cleanup_error:
                raise SynthTunnelRelayError(
                    "SynthTunnel lease response was invalid and cleanup also failed: "
                    f"{type(response_error).__name__}; {type(cleanup_error).__name__}"
                ) from response_error
            raise
        lease = AttachedSynthTunnelLease(
            lease_id=lease_id,
            public_url=public_url,
            worker_token=worker_token,
            expires_at=_optional_text(response.get("expires_at")),
            connector_mode=_optional_text(response.get("connector_mode")) or "synth_tunnel_agent",
            _local_health_url=local_health_url,
            _attach_timeout_seconds=self._attach_timeout_seconds,
            _ready_timeout_seconds=self._ready_timeout_seconds,
            _local_ready=wait_ready,
            route_token=_optional_text(response.get("route_token")),
            diagnostics_hint=_optional_text(response.get("diagnostics_hint")),
            _control_plane=self._control_plane,
            _agent=agent,
        )
        if wait_ready:
            try:
                lease.wait_ready()
            except Exception as startup_error:
                try:
                    lease.close()
                except Exception as cleanup_error:
                    raise SynthTunnelRelayError(
                        "SynthTunnel startup failed and its lease cleanup also failed: "
                        f"{type(startup_error).__name__}; {type(cleanup_error).__name__}"
                    ) from startup_error
                raise
        return lease


class _FatalAttachError(SynthTunnelRelayError):
    pass


class _RetryAttachError(SynthTunnelRelayError):
    """A prepared owner was superseded; reacquire without admitting its socket."""


class _SynthTunnelAgent:
    """Physical byte engine shared by explicit attachment modes."""

    def __init__(
        self,
        *,
        attachment: Attachment,
        agent_connect: Mapping[str, Any] | None,
        max_in_flight_requests: int,
        max_request_body_bytes: int,
        max_websocket_streams: int = _DEFAULT_MAX_WEBSOCKET_STREAMS,
    ) -> None:
        transport = (
            "ws"
            if agent_connect is None
            else _required_text(agent_connect.get("transport"), "agent transport")
        )
        if transport != "ws":
            raise SynthTunnelRelayError(f"unsupported SynthTunnel agent transport {transport!r}")
        self._protocol = (
            SUBPROTOCOL
            if agent_connect is None
            else str(agent_connect.get("protocol", "synthtunnel"))
        )
        if self._protocol not in {"synthtunnel", SUBPROTOCOL}:
            raise SynthTunnelRelayError("unsupported agent protocol")
        self._attachment_generation = 0
        self._upload_lock = threading.RLock()
        self._origin_connections = OriginConnections()
        self._uploads: dict[
            str,
            tuple[
                queue.Queue[bytes | None],
                threading.Event,
                http.client.HTTPConnection,
                threading.Thread,
            ],
        ] = {}
        self._attachment = attachment
        if isinstance(attachment, ConnectorAttachment) and self._protocol != SUBPROTOCOL:
            raise SynthTunnelRelayError("connector attachment requires the binary v2 protocol")
        if agent_connect is not None:
            self._url = _required_text(agent_connect.get("url"), "agent url")
            self._agent_token = _required_text(agent_connect.get("agent_token"), "agent token")
        self._attachment_provider: ConnectorAttachmentProvider | None = None
        self._prepared: PreparedAttachment | None = None
        self._prepare_operation = uuid.uuid4()
        self._max_in_flight_requests = max_in_flight_requests
        self._max_request_body_bytes = max_request_body_bytes
        self._max_websocket_message_bytes = (
            min(max_request_body_bytes, 2 * 1024 * 1024)
            if isinstance(attachment, ConnectorAttachment)
            else max_request_body_bytes
        )
        if max_websocket_streams <= 0:
            raise ValueError("max_websocket_streams must be positive")
        self._max_websocket_streams = max_websocket_streams
        self._request_slots = threading.BoundedSemaphore(max_in_flight_requests)
        # Separate ceiling: a long-lived socket must not consume the HTTP
        # request concurrency budget.
        self._websocket_slots = threading.BoundedSemaphore(max_websocket_streams)
        self._ready = threading.Event()
        self._fatal = threading.Event()
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._requests_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._requests: dict[str, _PendingRequest] = {}
        self._streams_lock = threading.Lock()
        self._streams: dict[str, _WebSocketStream] = {}
        self._thread: threading.Thread | None = None
        self._websocket: Any | None = None
        self._connection_generation = 0
        self._startup_error: str | None = None

    def update_agent_token(self, token: str) -> None:
        """Use a rotated agent token for the next relay connection."""

        normalized = _required_text(token, "agent token")
        with self._lifecycle_lock:
            self._agent_token = normalized
    def _identity_fields(self) -> dict[str, str]:
        match self._attachment:
            case LeaseAttachment(lease_id=lease):
                return {"lease_id": lease}
            case ConnectorAttachment(connector=connector):
                return {"connector": str(connector)}
        raise AssertionError("undeclared attachment scope")

    def _attach_message(self) -> dict[str, Any]:
        match self._attachment:
            case LeaseAttachment(lease_id=lease):
                return {
                    "type": "ATTACH",
                    "leases": [{"lease_id": lease}],
                    "capabilities": {"websocket": True},
                }
            case ConnectorAttachment(connector=connector, client_instance=client):
                return {
                    "type": "CONNECTOR_ATTACH",
                    "connector": str(connector),
                    "client_instance": str(client),
                    "capabilities": {"websocket": True},
                }
        raise AssertionError("undeclared attachment scope")

    def _origin_target(self, snapshot: _LocalTarget | None) -> _LocalTarget:
        if snapshot is not None:
            return snapshot
        match self._attachment:
            case LeaseAttachment(target=target):
                return target
            case ConnectorAttachment():
                raise TargetError("missing_origin_snapshot")
        raise AssertionError("undeclared attachment scope")

    def _bind_origin(
        self, payload: Mapping[str, Any]
    ) -> tuple[_LocalTarget, tuple[tuple[str, str], ...], OriginPoolScope]:
        match self._attachment:
            case LeaseAttachment(target=target):
                return target, (), None
            case ConnectorAttachment(connector=connector, targets=targets):
                binding = payload.get("route_binding")
                if not isinstance(binding, Mapping) or binding.get("connector") != str(connector):
                    raise TargetError("missing_connector_route_binding")
                if binding.get("attachment_generation") != self._attachment_generation:
                    raise TargetError("stale_origin_binding")
                lease = uuid.UUID(str(payload.get("lease_id", "")))
                route = uuid.UUID(str(binding.get("route", "")))
                if lease.int == 0 or route.int == 0:
                    raise TargetError("nil_origin_binding")
                key = binding.get("target_key")
                service = binding.get("service_identity")
                if not isinstance(key, str) or not isinstance(service, str):
                    raise TargetError("invalid_origin_binding")
                bound = targets.bind(key, service_identity=service)
                return (
                    _parse_local_target(bound.registration.origin.base_url),
                    bound.headers,
                    (key, service, bound.registration.generation),
                )
        raise AssertionError("undeclared attachment scope")

    def start(
        self,
        *,
        timeout_seconds: float,
        cancel_event: threading.Event | None = None,
    ) -> None:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        if not self._start_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise SynthTunnelRelayError("SynthTunnel agent attach wait exceeded its deadline")
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise SynthTunnelRelayError("SynthTunnel agent attach was cancelled")
            with self._lifecycle_lock:
                if cancel_event is not None and cancel_event.is_set():
                    raise SynthTunnelRelayError("SynthTunnel agent attach was cancelled")
                if self._thread is None or not self._thread.is_alive():
                    if self._attachment_provider is not None:
                        reset = getattr(self._attachment_provider, "reset", None)
                        if reset is not None:
                            reset()
                    self._stop.clear()
                    self._ready.clear()
                    self._fatal.clear()
                    self._startup_error = None
                    self._thread = threading.Thread(
                        target=self._run,
                        name="synth-containers-tunnel-agent",
                        daemon=True,
                    )
                    self._thread.start()
            while time.monotonic() < deadline:
                if self._ready.wait(timeout=0.05):
                    return
                if cancel_event is not None and cancel_event.is_set():
                    self.stop()
                    raise SynthTunnelRelayError("SynthTunnel agent attach was cancelled")
                if self._fatal.is_set():
                    break
                if self._thread is None or not self._thread.is_alive():
                    break
            detail = self._startup_error or "agent did not attach before the readiness deadline"
            self.stop()
            raise SynthTunnelRelayError(f"SynthTunnel agent attach failed: {detail}")
        finally:
            self._start_lock.release()

    def stop(self) -> None:
        # Retire admission before cleanup. A prepare/dial racing shutdown must
        # never create another connector socket after this close began.
        self._stop.set()
        self._origin_connections.retire()
        if self._attachment_provider is not None:
            cancel = getattr(self._attachment_provider, "cancel", None)
            if cancel is not None:
                cancel()
        with self._upload_lock:
            uploads = list(self._uploads.items())
        for rid, _ in uploads:
            self._cancel_upload(rid)
        for _, upload in uploads:
            upload[3].join(timeout=5)
            if upload[3].is_alive():
                raise SynthTunnelRelayError("origin upload thread did not stop")
        # Deliberate lease teardown. Tell the relay 1001 while the agent socket
        # is still up; the 1012 path in _run() only covers unplanned loss.
        self._close_all_websocket_streams(
            _WS_CLOSE_GOING_AWAY,
            "lease closed",
            notify_relay=True,
        )
        with self._lifecycle_lock:
            self._stop.set()
            with self._connection_lock:
                websocket = self._websocket
            thread = self._thread
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise SynthTunnelRelayError(
                    "SynthTunnel agent thread did not stop within five seconds"
                )
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None
            with self._connection_lock:
                self._websocket = None
        with self._requests_lock:
            self._requests.clear()

    def _cancel_upload(self, request_id: str) -> None:
        with self._upload_lock:
            upload = self._uploads.get(request_id)
            if upload is not None:
                _, stopped, connection, _ = upload
                stopped.set()
                if connection.sock is not None:
                    try:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass  # shutdown may race a socket that has already closed
                # Serialize retirement against returning this exact connection
                # to the pool; a late cancel must not close a sibling's borrow.
                connection.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            websocket = None
            connection_generation = 0
            retry_after_cleanup = False
            try:
                if self._attachment_provider is not None:
                    scope = self._attachment
                    if not isinstance(scope, ConnectorAttachment):
                        raise _FatalAttachError("provider requires connector scope")
                    prepared = self._attachment_provider.prepare(
                        connector=scope.connector,
                        client_instance=scope.client_instance,
                        operation=self._prepare_operation,
                    )
                    self._prepared = prepared
                    self._url = prepared.url
                    self._agent_token = prepared.token
                if self._stop.is_set():
                    break
                connect_options = (
                    {"subprotocols": [SUBPROTOCOL]} if self._protocol == SUBPROTOCOL else {}
                )
                websocket = _connect_websocket(
                    self._url,
                    headers={"Authorization": f"Bearer {self._agent_token}"},
                    max_message_bytes=_websocket_message_limit(self._max_request_body_bytes),
                    **connect_options,
                )
                if self._stop.is_set():
                    break
                if self._protocol == SUBPROTOCOL and websocket.subprotocol != SUBPROTOCOL:
                    raise _FatalAttachError("relay did not negotiate requested protocol")
                with self._connection_lock:
                    self._connection_generation += 1
                    connection_generation = self._connection_generation
                    self._websocket = websocket
                self._send_frame(
                    self._attach_message(),
                    expected_generation=connection_generation,
                )
                while not self._stop.is_set():
                    raw = websocket.recv()
                    if raw in (None, b"", ""):
                        raise SynthTunnelRelayError("SynthTunnel websocket closed")
                    if isinstance(raw, bytes) and self._protocol == SUBPROTOCOL:
                        frame = DataFrame.decode(raw)
                        if frame.generation != self._attachment_generation:
                            raise _FatalAttachError("stale binary attachment generation")
                        request_id = frame.stream.hex
                        if frame.kind == DataKind.REQUEST_BODY:
                            with self._upload_lock:
                                upload = self._uploads.get(request_id)
                            if upload is not None:
                                try:
                                    upload[0].put_nowait(frame.payload)
                                except queue.Full:
                                    self._cancel_upload(request_id)
                                    self._send_request_error(
                                        request_id, "UPLOAD_QUEUE_OVERFLOW", connection_generation
                                    )
                        elif frame.kind in {DataKind.WEBSOCKET_TEXT, DataKind.WEBSOCKET_BINARY}:
                            self._accept_websocket_frame(
                                request_id,
                                {
                                    "opcode": "text"
                                    if frame.kind == DataKind.WEBSOCKET_TEXT
                                    else "binary",
                                    "data_b64": _encode_bytes(frame.payload),
                                    "fin": frame.final_chunk,
                                },
                            )
                        else:
                            raise _FatalAttachError("unexpected binary frame direction")
                        continue
                    payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                    if isinstance(payload, Mapping):
                        self._handle_frame(payload, connection_generation)
            except _FatalAttachError as error:
                self._startup_error = _startup_error_detail(error)
                self._fatal.set()
                break
            except Exception as error:
                if not self._ready.is_set():
                    self._startup_error = _startup_error_detail(error)
                retry_after_cleanup = True
            finally:
                self._ready.clear()
                self._origin_connections.retire()
                with self._upload_lock:
                    upload_ids = list(self._uploads)
                for rid in upload_ids:
                    self._cancel_upload(rid)
                with self._connection_lock:
                    if self._websocket is websocket:
                        self._websocket = None
                with self._requests_lock:
                    stale = [
                        rid
                        for rid, request in self._requests.items()
                        if request.connection_generation == connection_generation
                    ]
                    for rid in stale:
                        self._requests.pop(rid, None)
                with self._streams_lock:
                    stale_streams = [
                        rid
                        for rid, stream in self._streams.items()
                        if stream.connection_generation == connection_generation
                    ]
                for rid in stale_streams:
                    self._close_websocket_stream(
                        rid,
                        _WS_CLOSE_SERVICE_RESTART,
                        "agent disconnected",
                        notify_relay=False,
                    )
                # Retired origin transports must close before waiting for the
                # shared connector's close handshake. That handshake can use
                # its full timeout when the gateway disappeared.
                if websocket is not None:
                    try:
                        websocket.close()
                    except Exception:
                        pass
            # Reconnect pacing cannot keep retired origin sockets alive or
            # leave readiness set after the attachment has already failed.
            if retry_after_cleanup and self._stop.wait(0.5):
                break

    def _handle_frame(
        self,
        payload: Mapping[str, Any],
        connection_generation: int,
    ) -> None:
        message_type = str(payload.get("type") or "")
        if message_type in {"ATTACH_ACK", "CONNECTOR_ATTACH_ACK"}:
            match self._attachment:
                case LeaseAttachment(lease_id=lease):
                    if message_type != "ATTACH_ACK" or lease not in {
                        str(item) for item in payload.get("accepted_leases", [])
                    }:
                        raise _FatalAttachError("SynthTunnel relay rejected the lease")
                case ConnectorAttachment(connector=connector):
                    if message_type != "CONNECTOR_ATTACH_ACK" or payload.get("connector") != str(
                        connector
                    ):
                        raise _FatalAttachError("SynthTunnel relay rejected connector identity")
                    session = payload.get("session")
                    if not isinstance(session, str) or uuid.UUID(session).int == 0:
                        raise _FatalAttachError("invalid connector session identity")
            if self._protocol == SUBPROTOCOL:
                config = payload.get("server_config")
                if not isinstance(config, Mapping) or config.get("protocol") != SUBPROTOCOL:
                    raise _FatalAttachError("invalid negotiated server protocol")
                generation = config.get("attachment_generation")
                if not isinstance(generation, int) or generation <= 0:
                    raise _FatalAttachError("missing attachment generation")
                if self._prepared is not None:
                    if payload.get("session") != str(self._prepared.session):
                        raise _FatalAttachError("connector acknowledgement lineage mismatch")
                    if generation < self._prepared.generation:
                        raise _FatalAttachError("connector acknowledgement lineage mismatch")
                    if generation > self._prepared.generation:
                        # A lifecycle transition raced prepare -> WebSocket.
                        # Never trust the unmatched ACK. Close this socket and
                        # obtain a new independently bound control receipt.
                        self._prepare_operation = uuid.uuid4()
                        raise _RetryAttachError("connector preparation superseded")
                self._attachment_generation = generation
            self._prepare_operation = uuid.uuid4()
            self._startup_error = None
            self._ready.set()
            return

        request_id = str(payload.get("rid") or "")
        if not request_id:
            return
        if message_type == "REQ_HEADERS":
            try:
                local_target, origin_headers, origin_pool_scope = self._bind_origin(payload)
            except (TargetError, ValueError):
                self._send_request_error(
                    request_id, "ORIGIN_BINDING_REJECTED", connection_generation
                )
                return
            with self._requests_lock:
                if len(self._requests) >= self._max_in_flight_requests:
                    self._send_request_error(
                        request_id,
                        "TOO_MANY_IN_FLIGHT_REQUESTS",
                        connection_generation,
                    )
                    return
                self._requests[request_id] = _PendingRequest(
                    method=str(payload.get("method") or "GET").upper(),
                    path=_request_path(payload.get("path")),
                    query=str(payload.get("query") or ""),
                    headers=_header_pairs(payload.get("headers")),
                    deadline_ms=max(1000, int(payload.get("deadline_ms") or 120000)),
                    connection_generation=connection_generation,
                    received_monotonic=time.monotonic(),
                    local_target=local_target,
                    origin_headers=origin_headers,
                    origin_pool_scope=origin_pool_scope,
                )
            if self._protocol == SUBPROTOCOL:
                with self._requests_lock:
                    request = self._requests.pop(request_id)
                self._start_upload(request_id, request)
            return
        if message_type == "REQ_CANCEL":
            self._cancel_upload(request_id)
            with self._requests_lock:
                self._requests.pop(request_id, None)
            return
        if message_type == "REQ_BODY":
            try:
                chunk = _decode_bytes(str(payload.get("chunk_b64") or ""))
            except ValueError:
                with self._requests_lock:
                    self._requests.pop(request_id, None)
                self._send_request_error(
                    request_id,
                    "INVALID_REQUEST_BODY",
                    connection_generation,
                )
                return
            with self._requests_lock:
                request = self._requests.get(request_id)
                if request is None:
                    return
                if len(request.body) + len(chunk) > self._max_request_body_bytes:
                    self._requests.pop(request_id, None)
                    self._send_request_error(
                        request_id,
                        "REQUEST_BODY_TOO_LARGE",
                        connection_generation,
                    )
                    return
                request.body.extend(chunk)
            return
        if message_type == "REQ_END":
            if self._protocol == SUBPROTOCOL:
                with self._upload_lock:
                    upload = self._uploads.get(request_id)
                if upload is not None:
                    try:
                        upload[0].put_nowait(None)
                    except queue.Full:
                        self._cancel_upload(request_id)
                        self._send_request_error(
                            request_id, "UPLOAD_QUEUE_OVERFLOW", connection_generation
                        )
                return
            with self._requests_lock:
                request = self._requests.pop(request_id, None)
            if request is not None:
                if not self._request_slots.acquire(blocking=False):
                    self._send_request_error(
                        request_id,
                        "TOO_MANY_ACTIVE_REQUESTS",
                        connection_generation,
                    )
                    return
                threading.Thread(
                    target=self._serve_request,
                    args=(request_id, request),
                    name="synth-containers-tunnel-request",
                    daemon=True,
                ).start()
            return
        if message_type == "WS_OPEN":
            self._open_websocket_stream(request_id, payload, connection_generation)
            return
        if message_type == "WS_FRAME":
            self._accept_websocket_frame(request_id, payload)
            return
        if message_type == "WS_CLOSE":
            # The relay closed this stream. Stop sending for the rid and drop it;
            # closing is idempotent, and the relay does not need an echo.
            self._close_websocket_stream(
                request_id,
                _websocket_close_code(payload.get("code")),
                str(payload.get("reason") or ""),
                notify_relay=False,
            )
            return

    # ------------------------------------------------------------------
    # WebSocket streams
    # ------------------------------------------------------------------

    def _open_websocket_stream(
        self,
        request_id: str,
        payload: Mapping[str, Any],
        connection_generation: int,
    ) -> None:
        if not self._websocket_slots.acquire(blocking=False):
            self._send_websocket_open_reject(
                request_id,
                connection_generation,
                503,
                [["content-type", "text/plain; charset=utf-8"]],
                b"synth-tunnel: too many open websocket streams",
            )
            return
        try:
            local_target, origin_headers, _origin_pool_scope = self._bind_origin(payload)
        except (TargetError, ValueError):
            self._websocket_slots.release()
            self._send_websocket_open_reject(
                request_id, connection_generation, 503, [], b"origin binding rejected"
            )
            return
        stream = _WebSocketStream(
            rid=request_id,
            path=_request_path(payload.get("path")),
            query=str(payload.get("query") or ""),
            headers=_header_pairs(payload.get("headers")),
            deadline_ms=max(1000, int(payload.get("deadline_ms") or 120000)),
            connection_generation=connection_generation,
            local_target=local_target,
            origin_headers=origin_headers,
        )
        with self._streams_lock:
            if request_id in self._streams:
                self._websocket_slots.release()
                return
            self._streams[request_id] = stream
        try:
            threading.Thread(
                target=self._serve_websocket,
                args=(stream,),
                name="synth-containers-tunnel-ws",
                daemon=True,
            ).start()
        except Exception:
            with self._streams_lock:
                if self._streams.get(request_id) is stream:
                    del self._streams[request_id]
            self._websocket_slots.release()
            self._send_websocket_open_reject(
                request_id,
                connection_generation,
                503,
                [["content-type", "text/plain; charset=utf-8"]],
                b"synth-tunnel: could not start a websocket stream",
            )

    def _accept_websocket_frame(
        self,
        request_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Queue one relay frame for the origin. Never blocks the receive loop."""
        with self._streams_lock:
            stream = self._streams.get(request_id)
        if stream is None:
            return
        try:
            data = _decode_bytes(str(payload.get("data_b64") or ""))
        except ValueError:
            self._close_websocket_stream(
                request_id,
                _WS_CLOSE_PROTOCOL_ERROR,
                "invalid websocket frame payload",
                notify_relay=True,
            )
            return
        opcode = "text" if str(payload.get("opcode") or "binary") == "text" else "binary"
        final = bool(payload.get("fin", True))
        with stream.lock:
            if stream.closed:
                return
            if len(stream.pending) + len(data) > self._max_request_body_bytes:
                oversize = True
            else:
                oversize = False
                if not final:
                    if stream.pending_opcode is None:
                        stream.pending_opcode = opcode
                    stream.pending.extend(data)
                    return
                if stream.pending:
                    opcode = stream.pending_opcode or opcode
                    data = bytes(stream.pending) + data
                    stream.pending = bytearray()
                    stream.pending_opcode = None
        if oversize:
            self._close_websocket_stream(
                request_id,
                _WS_CLOSE_MESSAGE_TOO_BIG,
                "websocket frame exceeded max_request_bytes",
                notify_relay=True,
            )
            return
        try:
            stream.outbound.put_nowait(("frame", opcode, data))
        except queue.Full:
            self._close_websocket_stream(
                request_id,
                _WS_CLOSE_TRY_AGAIN_LATER,
                "websocket stream queue overflowed",
                notify_relay=True,
            )

    def _close_websocket_stream(
        self,
        request_id: str,
        code: int,
        reason: str,
        *,
        notify_relay: bool,
    ) -> None:
        """Drop one stream. Idempotent: a second call for the same rid is a no-op."""
        with self._streams_lock:
            stream = self._streams.pop(request_id, None)
        if stream is None:
            return
        with stream.lock:
            stream.closed = True
            stream.pending = bytearray()
            stream.pending_opcode = None
        if notify_relay:
            try:
                self._send_frame(
                    {
                        "type": "WS_CLOSE",
                        **self._identity_fields(),
                        "rid": request_id,
                        "code": int(code),
                        "reason": _close_reason_text(reason),
                    },
                    expected_generation=stream.connection_generation,
                )
            except SynthTunnelRelayError:
                pass
        while True:
            try:
                stream.outbound.get_nowait()
            except queue.Empty:
                break
        try:
            stream.outbound.put_nowait(("close", int(code), _close_reason_text(reason)))
        except queue.Full:  # pragma: no cover - the queue was just drained
            pass

    def _close_all_websocket_streams(
        self,
        code: int,
        reason: str,
        *,
        notify_relay: bool,
    ) -> None:
        with self._streams_lock:
            request_ids = list(self._streams)
        for request_id in request_ids:
            self._close_websocket_stream(
                request_id,
                code,
                reason,
                notify_relay=notify_relay,
            )

    def _serve_websocket(self, stream: _WebSocketStream) -> None:
        origin: Any | None = None
        try:
            try:
                origin = self._dial_origin_websocket(stream)
            except _OriginUpgradeRefused as refused:
                # The path that matters most: the origin answered the upgrade with
                # an ordinary HTTP response and the client must see it intact.
                self._send_websocket_open_reject(
                    stream.rid,
                    stream.connection_generation,
                    refused.status,
                    refused.headers,
                    refused.body,
                )
                return
            except Exception:
                self._send_websocket_open_reject(
                    stream.rid,
                    stream.connection_generation,
                    502,
                    [["content-type", "text/plain; charset=utf-8"]],
                    b"synth-tunnel: local websocket dial failed",
                )
                return
            with stream.lock:
                already_closed = stream.closed
                if not already_closed:
                    stream.origin = origin
            if already_closed:
                _close_origin(origin, _WS_CLOSE_GOING_AWAY, "stream closed")
                return
            subprotocol = getattr(origin, "subprotocol", None)
            ack_headers = [[_WEBSOCKET_SUBPROTOCOL_HEADER, str(subprotocol)]] if subprotocol else []
            try:
                self._send_frame(
                    {
                        "type": "WS_OPEN_ACK",
                        **self._identity_fields(),
                        "rid": stream.rid,
                        "headers": ack_headers,
                    },
                    expected_generation=stream.connection_generation,
                )
            except SynthTunnelRelayError:
                _close_origin(origin, _WS_CLOSE_SERVICE_RESTART, "agent disconnected")
                return
            threading.Thread(
                target=self._pump_origin_to_relay,
                args=(stream, origin),
                name="synth-containers-tunnel-ws-reader",
                daemon=True,
            ).start()
            self._pump_relay_to_origin(stream, origin)
        finally:
            with self._streams_lock:
                if self._streams.get(stream.rid) is stream:
                    del self._streams[stream.rid]
            if origin is not None:
                _close_origin(origin, _WS_CLOSE_NORMAL, "")
            # Released exactly once per acquired slot, by the owning thread.
            self._websocket_slots.release()

    def _pump_relay_to_origin(self, stream: _WebSocketStream, origin: Any) -> None:
        while True:
            try:
                item = stream.outbound.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    _close_origin(origin, _WS_CLOSE_GOING_AWAY, "lease closed")
                    return
                with stream.lock:
                    if stream.closed:
                        _close_origin(origin, _WS_CLOSE_GOING_AWAY, "stream closed")
                        return
                continue
            if item[0] == "close":
                _close_origin(origin, int(item[1]), str(item[2]))
                return
            _, opcode, data = item
            try:
                origin.send(data.decode("utf-8", "replace") if opcode == "text" else data)
            except Exception:
                self._close_websocket_stream(
                    stream.rid,
                    _WS_CLOSE_INTERNAL_ERROR,
                    "origin websocket send failed",
                    notify_relay=True,
                )
                return

    def _pump_origin_to_relay(self, stream: _WebSocketStream, origin: Any) -> None:
        while True:
            try:
                message = origin.recv()
            except Exception as error:
                code, reason = _origin_close_reason(origin, error)
                self._close_websocket_stream(
                    stream.rid,
                    code,
                    reason,
                    notify_relay=True,
                )
                return
            if isinstance(message, str):
                opcode, data = "text", message.encode("utf-8")
            else:
                opcode, data = "binary", bytes(message)
            if len(data) > self._max_request_body_bytes:
                self._close_websocket_stream(
                    stream.rid,
                    _WS_CLOSE_MESSAGE_TOO_BIG,
                    "websocket frame exceeded max_request_bytes",
                    notify_relay=True,
                )
                return
            try:
                self._send_frame(
                    {
                        "type": "WS_FRAME",
                        **self._identity_fields(),
                        "rid": stream.rid,
                        "opcode": opcode,
                        "data_b64": _encode_bytes(data),
                        "fin": True,
                    },
                    expected_generation=stream.connection_generation,
                )
            except SynthTunnelRelayError:
                self._close_websocket_stream(
                    stream.rid,
                    _WS_CLOSE_SERVICE_RESTART,
                    "agent disconnected",
                    notify_relay=False,
                )
                return

    def _dial_origin_websocket(self, stream: _WebSocketStream) -> Any:
        from websockets.exceptions import InvalidStatus

        headers: list[tuple[str, str]] = []
        subprotocols: list[str] = []
        for key, value in stream.headers:
            name = key.strip().lower()
            if name == _WEBSOCKET_SUBPROTOCOL_HEADER:
                subprotocols.extend(part.strip() for part in value.split(",") if part.strip())
                continue
            if (
                name in _HOP_BY_HOP_HEADERS
                or name in _LOCAL_ONLY_AUTH_HEADERS
                or name in _WEBSOCKET_HANDSHAKE_HEADERS
            ):
                continue
            headers.append((key, value))
        headers.extend(stream.origin_headers)
        try:
            return _connect_origin_websocket(
                _local_websocket_url(
                    self._origin_target(stream.local_target), stream.path, stream.query
                ),
                headers=headers,
                subprotocols=subprotocols,
                max_message_bytes=self._max_websocket_message_bytes,
                open_timeout=max(1.0, stream.deadline_ms / 1000.0),
            )
        except InvalidStatus as refused:
            response = refused.response
            raise _OriginUpgradeRefused(
                int(response.status_code),
                _reject_header_pairs(response.headers),
                bytes(response.body or b""),
            ) from refused

    def _send_websocket_open_reject(
        self,
        request_id: str,
        connection_generation: int,
        status: int,
        headers: list[list[str]],
        body: bytes,
    ) -> None:
        try:
            self._send_frame(
                {
                    "type": "WS_OPEN_REJECT",
                    **self._identity_fields(),
                    "rid": request_id,
                    "status": int(status),
                    "headers": headers,
                    "body_b64": _encode_bytes(body),
                },
                expected_generation=connection_generation,
            )
        except SynthTunnelRelayError:
            return

    def _start_upload(self, request_id: str, request: _PendingRequest) -> None:
        try:
            upstream = urlparse(_local_upstream_url(self._origin_target(request.local_target), request.path, request.query))
        except ValueError:
            self._send_request_error(request_id, "INVALID_ORIGIN_PATH", request.connection_generation)
            return
        if not self._request_slots.acquire(blocking=False):
            self._send_request_error(
                request_id, "TOO_MANY_ACTIVE_REQUESTS", request.connection_generation
            )
            return
        connection_key = (
            request.connection_generation,
            request.origin_pool_scope,
            upstream.scheme,
            upstream.hostname or "",
            upstream.port,
        )
        connection, pool_epoch = self._origin_connections.borrow(
            connection_key, timeout_seconds=max(1, request.deadline_ms / 1000)
        )
        chunks: queue.Queue[bytes | None] = queue.Queue(maxsize=32)
        stopped = threading.Event()
        thread = threading.Thread(
            target=self._serve_upload,
            args=(request_id, request, upstream, chunks, stopped, connection, pool_epoch),
            name="synth-tunnel-upload",
            daemon=True,
        )
        with self._upload_lock:
            self._uploads[request_id] = (chunks, stopped, connection, thread)
        thread.start()

    def _serve_upload(
        self,
        request_id: str,
        request: _PendingRequest,
        upstream: Any,
        chunks: queue.Queue[bytes | None],
        stopped: threading.Event,
        connection: http.client.HTTPConnection,
        pool_epoch: int,
    ) -> None:
        reusable = False
        thread_started = time.monotonic()
        try:
            path = upstream.path or "/"
            if upstream.query:
                path += "?" + upstream.query
            connection.putrequest(request.method, path)
            for key, value in request.headers:
                if key.lower() not in _HOP_BY_HOP_HEADERS | _LOCAL_ONLY_AUTH_HEADERS:
                    connection.putheader(key, value)
            for key, value in request.origin_headers:
                connection.putheader(key, value)
            connection.putheader("Transfer-Encoding", "chunked")
            connection.endheaders()
            self._send_frame(
                {"type": "REQ_CREDIT", "rid": request_id, "slots": 31},
                expected_generation=request.connection_generation,
            )
            uploaded = 0
            deadline = request.received_monotonic + request.deadline_ms / 1000
            while not stopped.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("origin upload deadline exceeded")
                try:
                    chunk = chunks.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if chunk is None:
                    connection.send(b"0\r\n\r\n")
                    break
                uploaded += len(chunk)
                if uploaded > self._max_request_body_bytes:
                    raise ValueError("origin upload exceeds body limit")
                if connection.sock is not None:
                    connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
                connection.send(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self._send_frame(
                    {"type": "REQ_CREDIT", "rid": request_id, "slots": 1},
                    expected_generation=request.connection_generation,
                )
            if not stopped.is_set():
                if connection.sock is not None:
                    connection.sock.settimeout(max(0.001, deadline - time.monotonic()))
                with connection.getresponse() as response:
                    head_received = time.monotonic()
                    if head_received - request.received_monotonic > .02:
                        logging.getLogger(__name__).warning(
                            "slow origin response head request_id=%s generation=%s queue_ms=%.3f origin_ms=%.3f",
                            request_id, request.connection_generation,
                            (thread_started - request.received_monotonic) * 1000,
                            (head_received - thread_started) * 1000,
                        )
                    self._send_response(
                        request_id, response.status, response.headers, response, request
                    )
                    reusable = not response.will_close
        except Exception as error:
            logging.getLogger(__name__).warning(
                "origin stream failed request_id=%s error_kind=%s", request_id, type(error).__name__
            )
            if not stopped.is_set():
                try:
                    self._send_request_error(
                        request_id, "LOCAL_REQUEST_FAILED", request.connection_generation
                    )
                except SynthTunnelRelayError:
                    stopped.set()
        finally:
            with self._upload_lock:
                current = self._uploads.get(request_id)
                if current is not None and current[3] is threading.current_thread():
                    self._uploads.pop(request_id, None)
                if (
                    reusable
                    and not stopped.is_set()
                    and not self._stop.is_set()
                    and self._ready.is_set()
                    and request.connection_generation == self._connection_generation
                ):
                    self._origin_connections.release(
                        (
                            request.connection_generation,
                            request.origin_pool_scope,
                            upstream.scheme,
                            upstream.hostname,
                            upstream.port,
                        ),
                        connection,
                        pool_epoch,
                    )
                else:
                    connection.close()
            self._request_slots.release()

    def _serve_request(self, request_id: str, request: _PendingRequest) -> None:
        try:
            elapsed = time.monotonic() - request.received_monotonic
            timeout = max(1.0, request.deadline_ms / 1000.0 - elapsed)
            try:
                upstream_url = _local_upstream_url(self._origin_target(request.local_target), request.path, request.query)
            except ValueError:
                self._send_request_error(request_id, "INVALID_ORIGIN_PATH", request.connection_generation)
                return
            headers = {
                key: value
                for key, value in request.headers
                if key.strip().lower() not in _HOP_BY_HOP_HEADERS | _LOCAL_ONLY_AUTH_HEADERS
            }
            headers.update(request.origin_headers)
            try:
                upstream_request = urllib.request.Request(
                    upstream_url,
                    data=bytes(request.body) if request.body else None,
                    headers=headers,
                    method=request.method,
                )
                with _open_upstream(upstream_request, timeout=timeout) as response:
                    self._send_response(
                        request_id,
                        response.status,
                        response.headers,
                        response,
                        request,
                    )
            except urllib.error.HTTPError as error:
                self._send_response(
                    request_id,
                    error.code,
                    error.headers,
                    error,
                    request,
                )
            except Exception:
                try:
                    self._send_request_error(
                        request_id,
                        "LOCAL_REQUEST_FAILED",
                        request.connection_generation,
                    )
                except SynthTunnelRelayError:
                    return
        finally:
            self._request_slots.release()

    def _send_response(
        self,
        request_id: str,
        status_code: int,
        headers: Mapping[str, Any] | Message,
        response: Any,
        request: _PendingRequest,
    ) -> None:
        header_list = [
            [str(key), str(value)]
            for key, value in headers.items()
            if key.lower() not in {"connection", "content-length", "transfer-encoding"}
        ]
        self._send_frame(
            {
                "type": "RESP_HEADERS",
                **self._identity_fields(),
                "rid": request_id,
                "status": int(status_code),
                "headers": header_list,
            },
            expected_generation=request.connection_generation,
        )
        # read() blocks until it has the full request size or the stream ends,
        # so an SSE origin writing a few bytes every few hundred ms produced one
        # chunk at the very end: the agent, not the relay, was what stopped
        # streaming from streaming. read1() returns whatever has arrived.
        # Measured on a 6-event SSE origin: read() gave 1 chunk at 1.51s,
        # read1() gave 6 chunks at 0.00, 0.30, 0.60, 0.91, 1.22, 1.52.
        read_available = getattr(response, "read1", None) or response.read
        while True:
            chunk = read_available(65536)
            if not chunk:
                break
            if self._protocol == SUBPROTOCOL:
                self._send_binary(
                    request_id, DataKind.RESPONSE_BODY, chunk, request.connection_generation
                )
            else:
                self._send_frame(
                    {
                        "type": "RESP_BODY",
                        **self._identity_fields(),
                        "rid": request_id,
                        "chunk_b64": _encode_bytes(chunk),
                        "eof": False,
                    },
                    expected_generation=request.connection_generation,
                )
        self._send_frame(
            {"type": "RESP_END", **self._identity_fields(), "rid": request_id},
            expected_generation=request.connection_generation,
        )

    def _send_request_error(
        self,
        request_id: str,
        code: str,
        connection_generation: int,
    ) -> None:
        self._send_frame(
            {
                "type": "RESP_ERROR",
                **self._identity_fields(),
                "rid": request_id,
                "code": code,
                "message": code,
            },
            expected_generation=connection_generation,
        )

    def _send_binary(
        self, request_id: str, data_kind: DataKind, data: bytes, generation: int, final: bool = True
    ) -> None:
        with self._send_lock:
            with self._connection_lock:
                websocket = self._websocket
                if websocket is None or generation != self._connection_generation:
                    raise SynthTunnelRelayError("SynthTunnel websocket generation changed")
            pieces = [data[i : i + CHUNK_BYTES] for i in range(0, len(data), CHUNK_BYTES)] or [b""]
            for index, piece in enumerate(pieces):
                websocket.send(
                    DataFrame(
                        data_kind,
                        uuid.UUID(request_id),
                        self._attachment_generation,
                        final and index == len(pieces) - 1,
                        piece,
                    ).encode()
                )

    def _send_frame(
        self,
        payload: Mapping[str, Any],
        *,
        expected_generation: int,
    ) -> None:
        with self._send_lock:
            with self._connection_lock:
                websocket = self._websocket
                current_generation = self._connection_generation
            if websocket is None or current_generation != expected_generation:
                raise SynthTunnelRelayError("SynthTunnel websocket generation changed")
            kind = payload.get("type")
            if self._protocol == SUBPROTOCOL and kind in {"RESP_BODY", "WS_FRAME"}:
                encoded = (
                    payload.get("chunk_b64") if kind == "RESP_BODY" else payload.get("data_b64")
                )
                data = _decode_bytes(str(encoded or ""))
                data_kind = (
                    DataKind.RESPONSE_BODY
                    if kind == "RESP_BODY"
                    else (
                        DataKind.WEBSOCKET_TEXT
                        if payload.get("opcode") == "text"
                        else DataKind.WEBSOCKET_BINARY
                    )
                )
                pieces = [data[i : i + CHUNK_BYTES] for i in range(0, len(data), CHUNK_BYTES)] or [
                    b""
                ]
                for index, piece in enumerate(pieces):
                    final = bool(payload.get("fin", True)) and index == len(pieces) - 1
                    websocket.send(
                        DataFrame(
                            data_kind,
                            uuid.UUID(str(payload["rid"])),
                            self._attachment_generation,
                            final,
                            piece,
                        ).encode()
                    )
            else:
                websocket.send(json.dumps(dict(payload)))


def _connect_websocket(
    url: str,
    *,
    headers: Mapping[str, str],
    max_message_bytes: int,
    subprotocols: Sequence[str] | None = None,
) -> Any:
    from websockets.sync.client import connect
    from websockets.typing import Subprotocol

    if urlparse(url).scheme not in {"ws", "wss"}:
        raise SynthTunnelRelayError("SynthTunnel agent URL must use ws or wss")
    return connect(
        url,
        additional_headers=dict(headers),
        subprotocols=[Subprotocol(value) for value in subprotocols] if subprotocols is not None else None,
        open_timeout=3,
        # recv() reports remote closure only after the TCP closing handshake.
        # A retired gateway may not finish that handshake while cancelling a
        # backpressured stream. Leave at most 100ms before retiring origins;
        # a socket close never asserts that application effects stopped.
        close_timeout=0.1,
        max_size=max_message_bytes,
    )


def _connect_origin_websocket(
    url: str,
    *,
    headers: Sequence[tuple[str, str]],
    subprotocols: Sequence[str],
    max_message_bytes: int,
    open_timeout: float,
) -> Any:
    from websockets.sync.client import connect
    from websockets.typing import Subprotocol

    offered = [Subprotocol(value) for value in subprotocols] or None
    return connect(
        url,
        additional_headers=list(headers),
        subprotocols=offered,
        open_timeout=open_timeout,
        close_timeout=5,
        # max_size bounds a single origin message; websockets closes the socket
        # with 1009 itself when the origin exceeds it.
        max_size=max_message_bytes,
        # The origin hop is loopback and the agent re-frames every message, so no
        # extension is negotiated toward the origin.
        compression=None,
        # Never let a local WebSocket dial be redirected through a proxy.
        proxy=None,
    )


def _local_websocket_url(target: _LocalTarget, path: str, query: str) -> str:
    parsed = urlparse(_local_upstream_url(target, path, query))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, ""))


def _reject_header_pairs(headers: Any) -> list[list[str]]:
    """Origin response headers for WS_OPEN_REJECT, forwarded verbatim.

    Only the framing headers the relay must own are dropped. ``upgrade`` and
    ``sec-websocket-version`` are kept: they are what makes a 426 from the Trace
    V5 capture proxy stop a client from retrying the handshake.
    """
    raw_items = getattr(headers, "raw_items", None)
    items = raw_items() if callable(raw_items) else headers.items()
    return [
        [str(key), str(value)]
        for key, value in items
        if str(key).strip().lower() not in _WEBSOCKET_REJECT_FRAMING_HEADERS
    ]


def _websocket_close_code(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return _WS_CLOSE_NORMAL
    try:
        code = int(value)
    except (TypeError, ValueError):
        return _WS_CLOSE_NORMAL
    return code if 1000 <= code <= 4999 else _WS_CLOSE_NORMAL


def _close_reason_text(reason: str) -> str:
    """A close frame carries at most 125 bytes, two of which are the code."""
    return reason.encode("utf-8", "replace")[:120].decode("utf-8", "ignore")


def _origin_close_reason(origin: Any, error: Exception) -> tuple[int, str]:
    """Why the origin socket ended, as a close code to report to the relay.

    ``ConnectionClosed`` carries the close frames that were received and sent.
    The connection's own ``close_code`` is only 1006 when the peer never echoed
    the close -- which is exactly what happens when the agent itself closes with
    1009 for an oversize origin message -- so the frames are preferred.
    """
    for close in (getattr(error, "rcvd", None), getattr(error, "sent", None)):
        code = getattr(close, "code", None)
        if isinstance(code, int) and 1000 <= code <= 4999:
            return code, str(getattr(close, "reason", "") or "")
    code = getattr(origin, "close_code", None)
    if isinstance(code, int) and 1000 <= code <= 4999:
        return code, str(getattr(origin, "close_reason", None) or "")
    return _WS_CLOSE_INTERNAL_ERROR, type(error).__name__


def _close_origin(origin: Any, code: int, reason: str) -> None:
    try:
        origin.close(int(code), _close_reason_text(reason))
    except Exception:
        try:
            origin.close_socket()
        except Exception:
            pass


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


_ALLOWED_UPSTREAM_SCHEMES = frozenset({"http", "https"})
class _OriginHTTPSHandler(urllib.request.HTTPSHandler):
    """Construct TLS authority only when an HTTPS connection is actually used."""

    def __init__(self) -> None:
        # HTTPSHandler eagerly parses the trust store even for HTTP origins.
        # Parallel parsing contends inside OpenSSL and can stall plain HTTP
        # requests for seconds. HTTPSConnection retains normal certificate and
        # hostname verification when its default context is created on demand.
        urllib.request.AbstractHTTPHandler.__init__(self)

    def https_open(self, req: urllib.request.Request) -> Any:
        return self.do_open(http.client.HTTPSConnection, req)


def _open_upstream(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    scheme = (request.type or "").lower()
    if scheme not in _ALLOWED_UPSTREAM_SCHEMES:
        raise SynthTunnelRelayError(
            f"SynthTunnel upstream scheme {scheme!r} is not permitted"
        )
    # Build an opener that only speaks http(s): no FileHandler/FTPHandler/
    # UnknownHandler, so a ``file://``/``ftp://`` request can never be served
    # even if one reached this far. Redirects are refused, and no proxy is used.
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPHandler(),
        _OriginHTTPSHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        _NoRedirectHandler(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener.open(request, timeout=timeout)


def _parse_local_target(local_url: str) -> _LocalTarget:
    parsed = urlparse(local_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("local_url must be absolute HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("local_url cannot include credentials, query, or fragment")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    netloc = (
        f"[{parsed.hostname}]:{port}" if ":" in parsed.hostname else f"{parsed.hostname}:{port}"
    )
    base_url = urlunparse((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "", ""))
    return _LocalTarget(base_url=base_url, host=parsed.hostname, port=port)


def _wait_for_http_ok(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float,
    cancel_event: threading.Event | None = None,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise SynthTunnelRelayError("SynthTunnel readiness wait was cancelled")
        request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
        try:
            request_timeout = min(5.0, max(0.05, deadline - time.monotonic()))
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            if 200 <= error.code < 300:
                return
            last_error = f"HTTP {error.code}"
        except urllib.error.URLError as error:
            last_error = type(error.reason).__name__
        except TimeoutError:
            last_error = "TimeoutError"
        if cancel_event is not None:
            if cancel_event.wait(0.25):
                raise SynthTunnelRelayError("SynthTunnel readiness wait was cancelled")
        else:
            time.sleep(0.25)
    raise SynthTunnelRelayError(f"timed out waiting for tunnel health: {last_error}")


def _join_health_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "health")


def _local_upstream_url(target: _LocalTarget, path: str, query: str) -> str:
    return confined_origin_url(target.base_url, path, query)


def _request_path(value: object) -> str:
    path = str(value or "/").strip() or "/"
    return path if path.startswith("/") else f"/{path}"


def _header_pairs(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    headers: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) < 2:
            continue
        name = str(item[0])
        if name.strip():
            headers.append((name, str(item[1])))
    return headers


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise SynthTunnelRelayError(f"SynthTunnel lease response omitted required {field_name}")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: str) -> bytes:
    if not data:
        return b""
    try:
        return base64.b64decode(data.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as error:
        raise ValueError("invalid base64 request body") from error


def _websocket_message_limit(max_request_body_bytes: int) -> int:
    base64_bytes = ((max_request_body_bytes + 2) // 3) * 4
    return max(1024 * 1024, base64_bytes + 64 * 1024)


def _startup_error_detail(error: Exception) -> str:
    if isinstance(error, ModuleNotFoundError):
        missing_module = str(error.name or "").strip()
        if missing_module and all(
            character.isalnum() or character in {"_", ".", "-"} for character in missing_module
        ):
            return f"ModuleNotFoundError({missing_module})"
    return type(error).__name__


class SynthTunnelRelayAgent(_SynthTunnelAgent):
    """Explicit legacy lease attachment, including its negotiated v2 byte codec."""

    def __init__(
        self,
        *,
        lease_id: str,
        local_target: _LocalTarget,
        agent_connect: Mapping[str, Any],
        max_in_flight_requests: int,
        max_request_body_bytes: int,
        max_websocket_streams: int = _DEFAULT_MAX_WEBSOCKET_STREAMS,
    ) -> None:
        super().__init__(
            attachment=LeaseAttachment(lease_id, local_target),
            agent_connect=agent_connect,
            max_in_flight_requests=max_in_flight_requests,
            max_request_body_bytes=max_request_body_bytes,
            max_websocket_streams=max_websocket_streams,
        )


class SynthTunnelConnectorAgent(_SynthTunnelAgent):
    """One physical connector attachment serving separately bound lease streams."""

    def __init__(
        self,
        *,
        connector_id: uuid.UUID,
        client_instance_id: uuid.UUID,
        targets: LocalTargetRegistry,
        agent_connect: Mapping[str, Any] | None = None,
        provider: ConnectorAttachmentProvider | None = None,
        max_in_flight_requests: int,
        max_request_body_bytes: int,
        max_websocket_streams: int = _DEFAULT_MAX_WEBSOCKET_STREAMS,
    ) -> None:
        if (agent_connect is None) == (provider is None):
            raise ValueError(
                "declare exactly one connector attachment provider or static transport"
            )
        super().__init__(
            attachment=ConnectorAttachment(connector_id, client_instance_id, targets),
            agent_connect=agent_connect,
            max_in_flight_requests=max_in_flight_requests,
            max_request_body_bytes=max_request_body_bytes,
            max_websocket_streams=max_websocket_streams,
        )
        self._attachment_provider = provider
