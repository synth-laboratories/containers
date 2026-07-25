"""SynthTunnel relay agent and lease provider owned by synth-containers."""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urljoin, urlparse, urlunparse


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


@dataclass(frozen=True, slots=True)
class _LocalTarget:
    base_url: str
    host: str
    port: int


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
    _close_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _agent_stopped: bool = field(default=False, repr=False)
    _control_plane_closed: bool = field(default=False, repr=False)
    _closed: bool = field(default=False, repr=False)

    def close(self) -> None:
        with self._close_lock:
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
    ) -> None:
        if attach_timeout_seconds <= 0 or ready_timeout_seconds <= 0:
            raise ValueError("SynthTunnel timeouts must be positive")
        if max_in_flight_requests <= 0:
            raise ValueError("max_in_flight_requests must be positive")
        if max_request_body_bytes <= 0:
            raise ValueError("max_request_body_bytes must be positive")
        self._control_plane = control_plane
        self._client_instance_id = (
            client_instance_id or f"synth-containers-{uuid.uuid4().hex[:24]}"
        )
        self._attach_timeout_seconds = attach_timeout_seconds
        self._ready_timeout_seconds = ready_timeout_seconds
        self._max_in_flight_requests = max_in_flight_requests
        self._max_request_body_bytes = max_request_body_bytes

    def open_synth_tunnel(
        self,
        local_url: str,
        *,
        requested_ttl_seconds: int,
        metadata: Mapping[str, object],
        capabilities: Mapping[str, object],
    ) -> AttachedSynthTunnelLease:
        target = _parse_local_target(local_url)
        _wait_for_http_ok(
            _join_health_url(target.base_url),
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
                raise SynthTunnelRelayError(
                    "SynthTunnel lease response omitted agent_connect"
                )
            agent = SynthTunnelRelayAgent(
                lease_id=lease_id,
                local_target=target,
                agent_connect=agent_connect,
                max_in_flight_requests=self._max_in_flight_requests,
                max_request_body_bytes=self._max_request_body_bytes,
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
            connector_mode=_optional_text(response.get("connector_mode"))
            or "synth_tunnel_agent",
            _control_plane=self._control_plane,
            _agent=agent,
        )
        try:
            agent.start(timeout_seconds=self._attach_timeout_seconds)
            _wait_for_http_ok(
                _join_health_url(public_url),
                headers={"Authorization": f"Bearer {worker_token}"},
                timeout_seconds=self._ready_timeout_seconds,
            )
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


class SynthTunnelRelayAgent:
    """Attach to the hosted relay and proxy requests into one local container."""

    def __init__(
        self,
        *,
        lease_id: str,
        local_target: _LocalTarget,
        agent_connect: Mapping[str, Any],
        max_in_flight_requests: int,
        max_request_body_bytes: int,
    ) -> None:
        transport = _required_text(agent_connect.get("transport"), "agent transport")
        if transport != "ws":
            raise SynthTunnelRelayError(
                f"unsupported SynthTunnel agent transport {transport!r}"
            )
        self._lease_id = lease_id
        self._local_target = local_target
        self._url = _required_text(agent_connect.get("url"), "agent url")
        self._agent_token = _required_text(agent_connect.get("agent_token"), "agent token")
        self._max_in_flight_requests = max_in_flight_requests
        self._max_request_body_bytes = max_request_body_bytes
        self._request_slots = threading.BoundedSemaphore(max_in_flight_requests)
        self._ready = threading.Event()
        self._fatal = threading.Event()
        self._stop = threading.Event()
        self._send_lock = threading.Lock()
        self._requests_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._requests: dict[str, _PendingRequest] = {}
        self._thread: threading.Thread | None = None
        self._websocket: Any | None = None
        self._connection_generation = 0
        self._startup_error: str | None = None

    def start(self, *, timeout_seconds: float) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
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
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while time.monotonic() < deadline:
            if self._ready.wait(timeout=0.05):
                return
            if self._fatal.is_set():
                break
            if self._thread is None or not self._thread.is_alive():
                break
        detail = self._startup_error or "agent did not attach before the readiness deadline"
        self.stop()
        raise SynthTunnelRelayError(f"SynthTunnel agent attach failed: {detail}")

    def stop(self) -> None:
        self._stop.set()
        with self._connection_lock:
            websocket = self._websocket
        if websocket is not None:
            try:
                websocket.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise SynthTunnelRelayError(
                    "SynthTunnel agent thread did not stop within five seconds"
                )
        self._thread = None
        with self._connection_lock:
            self._websocket = None
        with self._requests_lock:
            self._requests.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            websocket = None
            connection_generation = 0
            try:
                websocket = _connect_websocket(
                    self._url,
                    headers={"Authorization": f"Bearer {self._agent_token}"},
                    max_message_bytes=_websocket_message_limit(
                        self._max_request_body_bytes
                    ),
                )
                with self._connection_lock:
                    self._connection_generation += 1
                    connection_generation = self._connection_generation
                    self._websocket = websocket
                self._send_frame(
                    {"type": "ATTACH", "leases": [{"lease_id": self._lease_id}]},
                    expected_generation=connection_generation,
                )
                while not self._stop.is_set():
                    raw = websocket.recv()
                    if raw in (None, b"", ""):
                        raise SynthTunnelRelayError("SynthTunnel websocket closed")
                    payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                    if isinstance(payload, Mapping):
                        self._handle_frame(payload, connection_generation)
            except _FatalAttachError as error:
                self._startup_error = type(error).__name__
                self._fatal.set()
                break
            except Exception as error:
                if not self._ready.is_set():
                    self._startup_error = type(error).__name__
                if self._stop.wait(0.5):
                    break
            finally:
                if websocket is not None:
                    try:
                        websocket.close()
                    except Exception:
                        pass
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

    def _handle_frame(
        self,
        payload: Mapping[str, Any],
        connection_generation: int,
    ) -> None:
        message_type = str(payload.get("type") or "")
        if message_type == "ATTACH_ACK":
            accepted = payload.get("accepted_leases") or []
            if self._lease_id not in {str(item) for item in accepted}:
                raise _FatalAttachError("SynthTunnel relay rejected the lease")
            self._ready.set()
            return

        request_id = str(payload.get("rid") or "")
        if not request_id:
            return
        if message_type == "REQ_HEADERS":
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
                )
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

    def _serve_request(self, request_id: str, request: _PendingRequest) -> None:
        try:
            elapsed = time.monotonic() - request.received_monotonic
            timeout = max(1.0, request.deadline_ms / 1000.0 - elapsed)
            upstream_url = _local_upstream_url(
                self._local_target,
                request.path,
                request.query,
            )
            headers = {
                key: value
                for key, value in request.headers
                if key.strip().lower() not in _HOP_BY_HOP_HEADERS | _LOCAL_ONLY_AUTH_HEADERS
            }
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
        headers: Mapping[str, Any],
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
                "lease_id": self._lease_id,
                "rid": request_id,
                "status": int(status_code),
                "headers": header_list,
            },
            expected_generation=request.connection_generation,
        )
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            self._send_frame(
                {
                    "type": "RESP_BODY",
                    "lease_id": self._lease_id,
                    "rid": request_id,
                    "chunk_b64": _encode_bytes(chunk),
                    "eof": False,
                },
                expected_generation=request.connection_generation,
            )
        self._send_frame(
            {"type": "RESP_END", "lease_id": self._lease_id, "rid": request_id},
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
                "lease_id": self._lease_id,
                "rid": request_id,
                "code": code,
                "message": code,
            },
            expected_generation=connection_generation,
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
            websocket.send(json.dumps(dict(payload)))


def _connect_websocket(
    url: str,
    *,
    headers: Mapping[str, str],
    max_message_bytes: int,
) -> Any:
    from websockets.sync.client import connect

    if urlparse(url).scheme not in {"ws", "wss"}:
        raise SynthTunnelRelayError("SynthTunnel agent URL must use ws or wss")
    return connect(
        url,
        additional_headers=dict(headers),
        open_timeout=10,
        close_timeout=5,
        max_size=max_message_bytes,
    )


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: Any,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> None:
        return None


def _open_upstream(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    return urllib.request.build_opener(_NoRedirectHandler()).open(
        request,
        timeout=timeout,
    )


def _parse_local_target(local_url: str) -> _LocalTarget:
    parsed = urlparse(local_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("local_url must be absolute HTTP(S)")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("local_url cannot include credentials, query, or fragment")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    netloc = (
        f"[{parsed.hostname}]:{port}"
        if ":" in parsed.hostname
        else f"{parsed.hostname}:{port}"
    )
    base_url = urlunparse(
        (parsed.scheme, netloc, parsed.path.rstrip("/"), "", "", "")
    )
    return _LocalTarget(base_url=base_url, host=parsed.hostname, port=port)


def _wait_for_http_ok(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        request = urllib.request.Request(url, headers=dict(headers or {}), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as error:
            if 200 <= error.code < 300:
                return
            last_error = f"HTTP {error.code}"
        except urllib.error.URLError as error:
            last_error = type(error.reason).__name__
        time.sleep(0.25)
    raise SynthTunnelRelayError(f"timed out waiting for tunnel health: {last_error}")


def _join_health_url(base_url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", "health")


def _local_upstream_url(target: _LocalTarget, path: str, query: str) -> str:
    upstream_url = urljoin(target.base_url.rstrip("/") + "/", path.lstrip("/"))
    return f"{upstream_url}?{query}" if query else upstream_url


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
        raise SynthTunnelRelayError(
            f"SynthTunnel lease response omitted required {field_name}"
        )
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
