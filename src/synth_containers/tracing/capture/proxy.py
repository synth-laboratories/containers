"""Explicit base-URL capture proxy for OpenAI-compatible Chat Completions traffic.

The proxy is a tee, not a reserialization path: the bytes returned to the caller are
the upstream bytes, in upstream order, with upstream status and content type. Capture
writes raw envelopes first and normalizes afterwards.

Push 1 supports exactly the routes the two acceptance consumers exercise. An unknown
route fails with a typed unsupported-protocol error in required mode instead of
silently direct-connecting.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx

from ..canonical import bytes_digest, record_id, text_digest, utc_now
from .binding import CaptureMode
from .coverage import CaptureCoverageReceiptV1
from .envelope import RawRecordType
from .redaction import redact_headers, redact_payload
from .session import CaptureSession


PROXY_VERSION = "synth-trace-proxy/1"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
HEALTH_PATH = "/healthz"

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


class UnsupportedProtocol(RuntimeError):
    """Raised when a required-mode capture sees a route it cannot observe."""


@dataclass(slots=True)
class ProxyStats:
    calls_accepted: int = 0
    calls_completed: int = 0
    calls_errored: int = 0
    calls_normalized: int = 0
    upstream_retries: int = 0
    unsupported_routes: list[str] = field(default_factory=list)
    redacted_headers: set[str] = field(default_factory=set)
    frames: int = 0
    truncated_records: int = 0


class CaptureProxy:
    """Local HTTP capture proxy bound to one capture session."""

    def __init__(
        self,
        session: CaptureSession,
        *,
        upstream_base_url: str,
        upstream_api_key: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        request_timeout: float = 600.0,
    ) -> None:
        self.session = session
        self.binding = session.binding
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_api_key = upstream_api_key
        self.stats = ProxyStats()
        self.request_timeout = request_timeout
        self._lock = threading.Lock()
        self._call_index = 0
        self._client = httpx.Client(timeout=request_timeout)
        self._server = ThreadingHTTPServer((host, port), _build_handler(self))
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"synth-trace-proxy-{self.binding.capture_id}",
            daemon=True,
        )
        self._max_inline = int(self.binding.policy.max_inline_bytes)

    # -- lifecycle ---------------------------------------------------------------

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}{CHAT_COMPLETIONS_PATH}"

    @property
    def openai_base_url(self) -> str:
        return f"{self.base_url}/v1"

    def start(self) -> "CaptureProxy":
        self._thread.start()
        self._append(
            RawRecordType.CAPTURE_STARTED,
            payload={
                "proxy_version": PROXY_VERSION,
                "upstream_host": urlparse(self.upstream_base_url).netloc,
                "base_url": self.base_url,
                "routes": [CHAT_COMPLETIONS_PATH, MODELS_PATH],
                "credential_mode": "proxy" if self.upstream_api_key else "passthrough",
            },
        )
        return self

    def stop(self, *, reason: str = "normal") -> None:
        self._append(
            RawRecordType.CAPTURE_FINISHED,
            payload={"reason": reason, "calls_accepted": self.stats.calls_accepted},
        )
        self._server.shutdown()
        self._server.server_close()
        self._client.close()

    def __enter__(self) -> "CaptureProxy":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # -- raw spool ---------------------------------------------------------------

    def _append(
        self,
        record_type: RawRecordType,
        *,
        payload: dict[str, Any],
        call_id: str | None = None,
        upstream_attempt_id: str | None = None,
        sequence_in_call: int | None = None,
    ) -> None:
        self.session.append(
            record_type,
            payload=payload,
            call_id=call_id,
            upstream_attempt_id=upstream_attempt_id,
            sequence_in_call=sequence_in_call,
            producer_version=PROXY_VERSION,
        )

    def _next_call(self) -> tuple[str, int]:
        with self._lock:
            self._call_index += 1
            index = self._call_index
        return self.session.mint("call", kind="model_call", key=index), index

    def _bounded(self, payload: bytes) -> tuple[Any, bool]:
        """Return an inline-safe body plus whether it was truncated."""

        if len(payload) > self._max_inline:
            return {"truncated_bytes": len(payload), "digest": bytes_digest(payload)}, True
        try:
            return json.loads(payload.decode("utf-8")), False
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"text": payload.decode("utf-8", errors="replace")}, False

    # -- request handling --------------------------------------------------------

    def handle_chat_completions(
        self,
        *,
        headers: Mapping[str, str],
        body: bytes,
    ) -> "ProxyResponse":
        call_id, call_index = self._next_call()
        self.stats.calls_accepted += 1
        request_headers, header_report = redact_headers(headers)
        self.stats.redacted_headers.update(header_report.removed_headers)
        try:
            request_payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.stats.calls_errored += 1
            self._append(
                RawRecordType.ERROR,
                payload={"stage": "request_parse", "message": str(exc)},
                call_id=call_id,
            )
            raise
        streaming = bool(request_payload.get("stream"))
        redacted_request, request_report = redact_payload(request_payload)
        started_at = utc_now()
        self._append(
            RawRecordType.MODEL_CALL_STARTED,
            payload={
                "call_index": call_index,
                "route": CHAT_COMPLETIONS_PATH,
                "provider_adapter": "openai_chat_completions",
                "model": request_payload.get("model"),
                "stream": streaming,
                "request_headers": request_headers,
                "request_body": redacted_request,
                "request_digest": bytes_digest(body),
                "redaction": request_report.merged(header_report).to_dict(),
                "started_at": started_at,
            },
            call_id=call_id,
        )

        attempt_id = record_id("att", kind="upstream_attempt", scope=(call_id,), key=1)
        upstream_url = f"{self.upstream_base_url}/chat/completions"
        forward_headers = _forward_headers(headers, self.upstream_api_key)
        self._append(
            RawRecordType.UPSTREAM_ATTEMPT_STARTED,
            payload={
                "attempt": 1,
                "upstream_host": urlparse(upstream_url).netloc,
                "upstream_path": urlparse(upstream_url).path,
            },
            call_id=call_id,
            upstream_attempt_id=attempt_id,
        )
        if streaming:
            return self._stream_call(
                call_id=call_id,
                call_index=call_index,
                attempt_id=attempt_id,
                url=upstream_url,
                headers=forward_headers,
                body=body,
                started_at=started_at,
            )
        return self._unary_call(
            call_id=call_id,
            call_index=call_index,
            attempt_id=attempt_id,
            url=upstream_url,
            headers=forward_headers,
            body=body,
            started_at=started_at,
        )

    def _unary_call(
        self,
        *,
        call_id: str,
        call_index: int,
        attempt_id: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        started_at: str,
    ) -> "ProxyResponse":
        response = self._client.post(url, content=body, headers=headers)
        payload = response.content
        response_headers, header_report = redact_headers(response.headers)
        self.stats.redacted_headers.update(header_report.removed_headers)
        bounded, truncated = self._bounded(payload)
        if truncated:
            self.stats.truncated_records += 1
        redacted_body, body_report = redact_payload(bounded)
        self._append(
            RawRecordType.UPSTREAM_ATTEMPT_FINISHED,
            payload={"attempt": 1, "http_status": response.status_code},
            call_id=call_id,
            upstream_attempt_id=attempt_id,
        )
        self._append(
            RawRecordType.RESPONSE_BODY,
            payload={
                "http_status": response.status_code,
                "response_headers": response_headers,
                "response_body": redacted_body,
                "response_digest": bytes_digest(payload),
                "truncated": truncated,
                "redaction": body_report.merged(header_report).to_dict(),
            },
            call_id=call_id,
            sequence_in_call=0,
        )
        usage = _usage_from_response(bounded)
        self._finish_call(
            call_id=call_id,
            call_index=call_index,
            status=response.status_code,
            started_at=started_at,
            usage=usage,
            streaming=False,
            frames=0,
        )
        return ProxyResponse(
            status_code=response.status_code,
            headers=_client_headers(response.headers, decoded=True),
            body=payload,
            stream=None,
        )

    def _stream_call(
        self,
        *,
        call_id: str,
        call_index: int,
        attempt_id: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        started_at: str,
    ) -> "ProxyResponse":
        proxy = self

        def generate(sink: Any) -> None:
            frames = 0
            status = 0
            usage: dict[str, Any] | None = None
            with proxy._client.stream("POST", url, content=body, headers=headers) as response:
                status = response.status_code
                sink.begin(status, _client_headers(response.headers))
                response_headers, header_report = redact_headers(response.headers)
                proxy.stats.redacted_headers.update(header_report.removed_headers)
                proxy._append(
                    RawRecordType.UPSTREAM_ATTEMPT_FINISHED,
                    payload={
                        "attempt": 1,
                        "http_status": status,
                        "response_headers": response_headers,
                    },
                    call_id=call_id,
                    upstream_attempt_id=attempt_id,
                )
                for chunk in response.iter_raw():
                    if not chunk:
                        continue
                    sink.write(chunk)
                    frames += 1
                    text = chunk.decode("utf-8", errors="replace")
                    redacted, _ = redact_payload({"frame": text})
                    proxy._append(
                        RawRecordType.RESPONSE_FRAME,
                        payload={
                            "frame_index": frames - 1,
                            "byte_size": len(chunk),
                            "digest": bytes_digest(chunk),
                            "frame": redacted["frame"][: proxy._max_inline],
                        },
                        call_id=call_id,
                        sequence_in_call=frames - 1,
                    )
                    parsed = _usage_from_sse_chunk(text)
                    if parsed is not None:
                        usage = parsed
            proxy.stats.frames += frames
            proxy._finish_call(
                call_id=call_id,
                call_index=call_index,
                status=status,
                started_at=started_at,
                usage=usage,
                streaming=True,
                frames=frames,
            )

        return ProxyResponse(status_code=0, headers={}, body=b"", stream=generate)

    def _finish_call(
        self,
        *,
        call_id: str,
        call_index: int,
        status: int,
        started_at: str,
        usage: dict[str, Any] | None,
        streaming: bool,
        frames: int,
    ) -> None:
        ended_at = utc_now()
        if 200 <= status < 300:
            self.stats.calls_completed += 1
        else:
            self.stats.calls_errored += 1
        self._append(
            RawRecordType.MODEL_CALL_FINISHED,
            payload={
                "call_index": call_index,
                "http_status": status,
                "usage": usage,
                "usage_observed": usage is not None,
                "streaming": streaming,
                "frames": frames,
                "started_at": started_at,
                "ended_at": ended_at,
            },
            call_id=call_id,
        )
        self.stats.calls_normalized += 1

    def handle_unsupported(self, path: str) -> None:
        self.stats.unsupported_routes.append(path)
        self._append(
            RawRecordType.ERROR,
            payload={"stage": "route", "code": "unsupported_protocol", "path": path},
        )
        mode = str(self.binding.capture.mode)
        if mode in {CaptureMode.REQUIRED, CaptureMode.REQUIRED_EGRESS_ASSERTED}:
            raise UnsupportedProtocol(f"capture mode {mode} cannot observe route {path}")

    def apply_to_receipt(self, receipt: CaptureCoverageReceiptV1) -> CaptureCoverageReceiptV1:
        """Fold observed proxy counters into a coverage receipt."""

        from dataclasses import replace

        return replace(
            receipt,
            calls_accepted=self.stats.calls_accepted,
            calls_completed=self.stats.calls_completed,
            calls_errored=self.stats.calls_errored,
            calls_normalized=self.stats.calls_normalized,
            upstream_retries=self.stats.upstream_retries,
            truncated_records=self.stats.truncated_records,
            redacted_headers=tuple(sorted(self.stats.redacted_headers)),
            unsupported_routes=tuple(self.stats.unsupported_routes),
            first_observed_at=self.session.first_observed_at,
            last_observed_at=self.session.last_observed_at,
            provider_adapters=("openai_chat_completions",),
            routes_enabled=(CHAT_COMPLETIONS_PATH, MODELS_PATH),
            endpoint_identity_digest=text_digest(self.upstream_base_url),
        )


@dataclass(slots=True)
class ProxyResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    stream: Any | None


def _forward_headers(headers: Mapping[str, str], api_key: str | None) -> dict[str, str]:
    """Build upstream headers. Credentials pass through the process, never to disk."""

    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP and not name.lower().startswith("x-synth-trace")
    }
    if api_key:
        forwarded["Authorization"] = f"Bearer {api_key}"
    forwarded.setdefault("Content-Type", "application/json")
    return forwarded


def _client_headers(headers: Mapping[str, str], *, decoded: bool = False) -> dict[str, str]:
    """Headers to return to the caller.

    ``decoded=True`` is used when the body was already decompressed for the caller, so
    the upstream ``content-encoding`` no longer describes the bytes being sent.
    """

    dropped = set(_HOP_BY_HOP)
    if decoded:
        dropped.add("content-encoding")
    return {name: value for name, value in headers.items() if name.lower() not in dropped}


def _usage_from_response(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping):
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            return dict(usage)
    return None


def _usage_from_sse_chunk(text: str) -> dict[str, Any] | None:
    """Pull provider usage out of SSE data frames without reassembling the stream."""

    found: dict[str, Any] | None = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = parsed.get("usage") if isinstance(parsed, Mapping) else None
        if isinstance(usage, Mapping):
            found = dict(usage)
    return found


class _StreamSink:
    """Adapts a BaseHTTPRequestHandler into the begin/write sink the proxy streams to."""

    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._handler = handler

    def begin(self, status: int, headers: Mapping[str, str]) -> None:
        # A streamed body has no known length, so the connection boundary ends it.
        self._handler.close_connection = True
        self._handler.send_response(status)
        for name, value in headers.items():
            self._handler.send_header(name, value)
        self._handler.send_header("Connection", "close")
        self._handler.end_headers()

    def write(self, chunk: bytes) -> None:
        self._handler.wfile.write(chunk)
        self._handler.wfile.flush()


def _build_handler(proxy: CaptureProxy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
            self.send_response(status)
            for name, value in headers.items():
                if name.lower() == "content-length":
                    continue
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send(status, {"Content-Type": "application/json"}, body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == HEALTH_PATH:
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "proxy": PROXY_VERSION,
                        "capture_id": proxy.binding.capture_id,
                        "trace_id": proxy.binding.trace_id,
                    },
                )
                return
            if path == MODELS_PATH:
                response = proxy._client.get(
                    f"{proxy.upstream_base_url}/models",
                    headers=_forward_headers(self.headers, proxy.upstream_api_key),
                )
                self._send(
                    response.status_code,
                    _client_headers(response.headers, decoded=True),
                    response.content,
                )
                return
            self._unsupported(path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != CHAT_COMPLETIONS_PATH:
                self._unsupported(path)
                return
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length else b"{}"
            try:
                result = proxy.handle_chat_completions(headers=self.headers, body=body)
            except httpx.HTTPError as exc:
                proxy.stats.calls_errored += 1
                self._send_json(502, {"error": {"type": "upstream_error", "message": str(exc)}})
                return
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._send_json(400, {"error": {"type": "bad_request", "message": str(exc)}})
                return
            if result.stream is not None:
                result.stream(_StreamSink(self))
                return
            self._send(result.status_code, result.headers, result.body)

        def _unsupported(self, path: str) -> None:
            try:
                proxy.handle_unsupported(path)
            except UnsupportedProtocol as exc:
                self._send_json(
                    501,
                    {"error": {"type": "unsupported_protocol", "message": str(exc)}},
                )
                return
            self._send_json(
                404,
                {"error": {"type": "not_found", "message": f"route {path} is not captured"}},
            )

    return Handler


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "HEALTH_PATH",
    "MODELS_PATH",
    "PROXY_VERSION",
    "CaptureProxy",
    "ProxyResponse",
    "ProxyStats",
    "UnsupportedProtocol",
]
