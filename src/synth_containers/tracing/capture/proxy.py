"""Explicit base-URL capture proxy for OpenAI-compatible Chat Completions traffic.

The proxy is a tee, not a reserialization path: the bytes returned to the caller are
the upstream bytes, in upstream order, with upstream status and content type. Capture
writes raw envelopes first and normalizes afterwards.

Push 1 supports exactly the routes the two acceptance consumers exercise. An unknown
route fails with a typed unsupported-protocol error in required mode instead of
silently direct-connecting.
"""

from __future__ import annotations

import io
import json
import threading
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping
from urllib.parse import urlparse, urlsplit

import httpx
import zstandard

from ..adapters import provider_adapters
from ..adapters.base import ProviderAdapterRegistry
from ..adapters.sse import SSEDecoder, SSEEvent
from ..canonical import bytes_digest, canonical_bytes, record_id, text_digest, utc_now
from ..models.capture_data import CapturedBodyRefV1, RawCaptureDisposition
from ..models.identity import TraceContextV1
from .binding import CaptureMode
from .coverage import CaptureCoverageReceiptV1
from .envelope import RawRecordType
from .redaction import redact_headers, redact_payload
from .routes import (
    ProviderEndpointConfig,
    ProviderRouteRegistry,
    UpstreamAuthKind,
)
from .session import CaptureSession


PROXY_VERSION = "synth-trace-proxy/1"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
RESPONSES_PATH = "/v1/responses"
RESPONSES_COMPACT_PATH = "/v1/responses/compact"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
MODELS_PATH = "/v1/models"
HEALTH_PATH = "/healthz"
MAX_DECODED_REQUEST_BYTES = 64 * 1024 * 1024

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


class CaptureContextError(RuntimeError):
    """A request declared a trace context that the supervisor did not authorize."""


class UnsupportedContentEncoding(ValueError):
    """Raised when a provider request uses an encoding the proxy cannot inspect."""


class DecodedRequestTooLarge(ValueError):
    """Raised before an encoded request can expand beyond the inspection limit."""


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
        upstream_base_url: str | None = None,
        upstream_api_key: str | None = None,
        provider_endpoints: tuple[ProviderEndpointConfig, ...] | None = None,
        adapter_registry: ProviderAdapterRegistry | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        request_timeout: float = 600.0,
        context_resolver: Callable[[Mapping[str, str]], TraceContextV1 | None] | None = None,
    ) -> None:
        self.session = session
        self.binding = session.binding
        if provider_endpoints is None:
            if not upstream_base_url:
                raise ValueError("upstream_base_url or provider_endpoints is required")
            provider_endpoints = (
                ProviderEndpointConfig(
                    route=CHAT_COMPLETIONS_PATH,
                    adapter_name="openai_chat_completions",
                    upstream_base_url=upstream_base_url,
                    upstream_path="/chat/completions",
                    auth_kind=(
                        UpstreamAuthKind.BEARER
                        if upstream_api_key
                        else UpstreamAuthKind.PASSTHROUGH
                    ),
                    api_key=upstream_api_key,
                ),
            )
        self.routes = ProviderRouteRegistry(provider_endpoints)
        self.adapters = adapter_registry or provider_adapters()
        for endpoint in provider_endpoints:
            if self.adapters.by_name(endpoint.adapter_name) is None:
                raise ValueError(f"unknown provider adapter: {endpoint.adapter_name}")
        self.upstream_base_url = (
            upstream_base_url or provider_endpoints[0].upstream_base_url
        ).rstrip("/")
        self.upstream_api_key = upstream_api_key
        self.stats = ProxyStats()
        self.request_timeout = request_timeout
        self.context_resolver = context_resolver
        self._lock = threading.Lock()
        self._call_contexts: dict[str, tuple[str, str]] = {}
        # This client is the registered upstream boundary. It must never inherit
        # child-facing HTTP_PROXY/HTTPS_PROXY settings, especially when the scoped
        # MITM chains decrypted provider traffic into this proxy.
        self._client = httpx.Client(timeout=request_timeout, trust_env=False)
        self._server = ThreadingHTTPServer((host, port), _build_handler(self))
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"synth-trace-proxy-{self.binding.capture_id}",
            daemon=True,
        )
        self._max_inline = int(self.binding.policy.max_inline_bytes)
        self._started = False
        self._stopped = False

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
        if self._started:
            return self
        self._thread.start()
        self._started = True
        self._append(
            RawRecordType.CAPTURE_STARTED,
            payload={
                "proxy_version": PROXY_VERSION,
                "upstream_host": urlparse(self.upstream_base_url).netloc,
                "base_url": self.base_url,
                "routes": list(self.routes.routes),
                "credential_mode": "memory_only_or_passthrough",
            },
        )
        return self

    def stop(self, *, reason: str = "normal") -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._started:
            # BaseServer.shutdown deadlocks when serve_forever was never started.
            self._server.shutdown()
            self._server.server_close()
            # ThreadingHTTPServer.server_close waits for request threads. Emit the
            # terminal record only after every accepted call has drained.
            self._append(
                RawRecordType.CAPTURE_FINISHED,
                payload={"reason": reason, "calls_accepted": self.stats.calls_accepted},
            )
        else:
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
        actor_id = None
        session_id = None
        if call_id is not None:
            with self._lock:
                context = self._call_contexts.get(call_id)
            if context is not None:
                actor_id, session_id = context
        self.session.append(
            record_type,
            payload=payload,
            actor_id=actor_id,
            session_id=session_id,
            call_id=call_id,
            upstream_attempt_id=upstream_attempt_id,
            sequence_in_call=sequence_in_call,
            producer_version=PROXY_VERSION,
        )

    def _next_call(self) -> tuple[str, int]:
        return self.session.mint_call(kind="model_call")

    def _bounded(self, payload: bytes) -> tuple[Any, bool]:
        """Return an inline-safe body plus whether it was truncated."""

        if len(payload) > self._max_inline:
            return {"truncated_bytes": len(payload), "digest": bytes_digest(payload)}, True
        try:
            return json.loads(payload.decode("utf-8")), False
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"text": payload.decode("utf-8", errors="replace")}, False

    def _capture_body(
        self,
        payload: bytes,
        *,
        media_type: str,
        wire_payload: bytes | None = None,
    ) -> tuple[Any | None, CapturedBodyRefV1]:
        """Retain a safe representation while preserving the original wire digest."""

        wire = payload if wire_payload is None else wire_payload
        wire_digest = bytes_digest(wire)
        try:
            parsed: Any = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = {"text": payload.decode("utf-8", errors="replace")}
        redacted, report = redact_payload(parsed)
        safe = canonical_bytes(redacted)
        if len(safe) <= self._max_inline:
            return redacted, CapturedBodyRefV1(
                wire_digest=wire_digest,
                wire_byte_size=len(wire),
                disposition=RawCaptureDisposition.REDACTED_INLINE,
                media_type=media_type,
                inline=redacted,
                redaction_profile=report.profile,
            )
        stored_digest, uri = self.session.store_blob(safe)
        return None, CapturedBodyRefV1(
            wire_digest=wire_digest,
            wire_byte_size=len(wire),
            disposition=RawCaptureDisposition.REDACTED_ARTIFACT,
            media_type=media_type,
            stored_digest=stored_digest,
            uri=uri,
            redaction_profile=report.profile,
            truncated=False,
        )

    # -- request handling --------------------------------------------------------

    def handle_chat_completions(
        self,
        *,
        headers: Mapping[str, str],
        body: bytes,
    ) -> "ProxyResponse":
        return self.handle_provider_request(
            path=CHAT_COMPLETIONS_PATH,
            headers=headers,
            body=body,
        )

    def handle_provider_request(
        self,
        *,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        query: str = "",
    ) -> "ProxyResponse":
        endpoint = self.routes.resolve(path)
        if endpoint is None:
            self.handle_unsupported(path)
            raise UnsupportedProtocol(f"no provider route configured for {path}")
        adapter = self.adapters.by_name(endpoint.adapter_name)
        if adapter is None:
            raise UnsupportedProtocol(f"no provider adapter configured for {endpoint.adapter_name}")
        actor_id, session_id = self._request_identity(headers)
        decoded_body, content_encodings = _decode_request_body(headers, body)
        call_id, call_index = self._next_call()
        with self._lock:
            self._call_contexts[call_id] = (actor_id, session_id)
        self.stats.calls_accepted += 1
        request_headers, header_report = redact_headers(headers)
        self.stats.redacted_headers.update(header_report.removed_headers)
        try:
            request_payload = json.loads(decoded_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.stats.calls_errored += 1
            self._append(
                RawRecordType.ERROR,
                payload={"stage": "request_parse", "message": str(exc)},
                call_id=call_id,
            )
            with self._lock:
                self._call_contexts.pop(call_id, None)
            raise
        streaming = bool(request_payload.get("stream")) and path != RESPONSES_COMPACT_PATH
        redacted_request, request_report = redact_payload(request_payload)
        inline_request, request_body_ref = self._capture_body(
            decoded_body,
            media_type=str(headers.get("content-type") or "application/json"),
            wire_payload=body,
        )
        started_at = utc_now()
        self._append(
            RawRecordType.MODEL_CALL_STARTED,
            payload={
                "call_index": call_index,
                "route": path,
                "provider_adapter": endpoint.adapter_name,
                "provider_adapter_version": adapter.version,
                "model": request_payload.get("model"),
                "stream": streaming,
                "request_headers": request_headers,
                "request_body": inline_request or {},
                "request_body_ref": request_body_ref.to_dict(),
                "request_digest": bytes_digest(body),
                "decoded_request_digest": bytes_digest(decoded_body),
                "content_encodings": content_encodings,
                "redaction": request_report.merged(header_report).to_dict(),
                "started_at": started_at,
            },
            call_id=call_id,
        )

        attempt_id = record_id("att", kind="upstream_attempt", scope=(call_id,), key=1)
        upstream_url = _append_query(endpoint.upstream_url(), query)
        forward_headers = _forward_headers(headers, endpoint)
        self._append(
            RawRecordType.UPSTREAM_ATTEMPT_STARTED,
            payload={
                "attempt": 1,
                "upstream_host": urlparse(upstream_url).netloc,
                "upstream_path": urlparse(upstream_url).path,
                "provider_adapter": endpoint.adapter_name,
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
                adapter_name=endpoint.adapter_name,
            )
        return self._unary_call(
            call_id=call_id,
            call_index=call_index,
            attempt_id=attempt_id,
            url=upstream_url,
            headers=forward_headers,
            body=body,
            started_at=started_at,
            adapter_name=endpoint.adapter_name,
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
        adapter_name: str,
    ) -> "ProxyResponse":
        response = self._client.post(url, content=body, headers=headers)
        payload = response.content
        response_headers, header_report = redact_headers(response.headers)
        self.stats.redacted_headers.update(header_report.removed_headers)
        inline_body, body_ref = self._capture_body(
            payload,
            media_type=str(response.headers.get("content-type") or "application/json"),
        )
        truncated = bool(body_ref.truncated)
        if truncated:
            self.stats.truncated_records += 1
        redacted_body, body_report = redact_payload(inline_body or {})
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
                "response_body_ref": body_ref.to_dict(),
                "response_digest": bytes_digest(payload),
                "truncated": truncated,
                "redaction": body_report.merged(header_report).to_dict(),
            },
            call_id=call_id,
            sequence_in_call=0,
        )
        usage_payload = inline_body
        if usage_payload is None:
            try:
                usage_payload = json.loads(payload.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                usage_payload = None
        usage = _usage_from_response(usage_payload)
        self._finish_call(
            call_id=call_id,
            call_index=call_index,
            status=response.status_code,
            started_at=started_at,
            usage=usage,
            streaming=False,
            frames=0,
            adapter_name=adapter_name,
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
        adapter_name: str,
    ) -> "ProxyResponse":
        proxy = self

        def generate(sink: Any) -> None:
            frames = 0
            status = 0
            usage: dict[str, Any] | None = None
            decoder = SSEDecoder()
            wire_chunks = 0
            wire_bytes = 0
            with proxy._client.stream("POST", url, content=body, headers=headers) as response:
                status = response.status_code
                sink.begin(status, _client_headers(response.headers))
                content_decoder = _StreamingContentDecoder(response.headers)
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
                    wire_chunks += 1
                    wire_bytes += len(chunk)
                    for event in decoder.feed(content_decoder.feed(chunk)):
                        frames += 1
                        parsed = proxy._capture_sse_event(
                            event=event,
                            call_id=call_id,
                            frame_index=frames - 1,
                        )
                        if parsed is not None:
                            usage = parsed
                decoded_tail = content_decoder.finish()
                if decoded_tail:
                    for event in decoder.feed(decoded_tail):
                        frames += 1
                        parsed = proxy._capture_sse_event(
                            event=event,
                            call_id=call_id,
                            frame_index=frames - 1,
                        )
                        if parsed is not None:
                            usage = parsed
                for event in decoder.finish():
                    frames += 1
                    parsed = proxy._capture_sse_event(
                        event=event,
                        call_id=call_id,
                        frame_index=frames - 1,
                    )
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
                adapter_name=adapter_name,
                transport_detail={
                    "wire_chunks": wire_chunks,
                    "wire_bytes": wire_bytes,
                    "capture_boundary": "complete_sse_event",
                },
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
        adapter_name: str,
        transport_detail: Mapping[str, Any] | None = None,
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
                "provider_adapter": adapter_name,
                "transport_detail": dict(transport_detail or {}),
                "started_at": started_at,
                "ended_at": ended_at,
            },
            call_id=call_id,
        )
        with self._lock:
            self._call_contexts.pop(call_id, None)
        self.stats.calls_normalized += 1

    def _request_identity(self, headers: Mapping[str, str]) -> tuple[str, str]:
        """Resolve an authenticated child context, or use the root workload."""

        if self.context_resolver is None:
            return (
                self.binding.workload.root_actor_id,
                self.binding.workload.actor_session_id,
            )
        context = self.context_resolver(headers)
        declares_context = any(
            name.lower()
            in {
                "x-synth-trace-id",
                "x-synth-capture-id",
                "x-synth-actor-id",
                "x-synth-session-id",
            }
            for name in headers
        )
        if context is None:
            if declares_context:
                raise CaptureContextError("provider request trace context is not authorized")
            return (
                self.binding.workload.root_actor_id,
                self.binding.workload.actor_session_id,
            )
        if context.trace_id != self.binding.trace_id:
            raise CaptureContextError("provider request belongs to a different trace")
        return context.actor_id, context.actor_session_id

    def _capture_sse_event(
        self,
        *,
        event: SSEEvent,
        call_id: str,
        frame_index: int,
    ) -> dict[str, Any] | None:
        """Capture one complete SSE event, never a partially decoded wire chunk."""

        lines: list[str] = []
        if event.event is not None:
            lines.append(f"event: {event.event}")
        if event.event_id is not None:
            lines.append(f"id: {event.event_id}")
        lines.extend(f"data: {line}" for line in event.data.split("\n"))
        frame = "\n".join(lines) + "\n\n"
        redacted, report = redact_payload({"frame": frame})
        safe_frame = str(redacted["frame"]).encode("utf-8")
        frame_ref = None
        if len(safe_frame) > self._max_inline:
            stored_digest, uri = self.session.store_blob(safe_frame)
            frame_ref = CapturedBodyRefV1(
                wire_digest=bytes_digest(frame.encode("utf-8")),
                wire_byte_size=len(frame.encode("utf-8")),
                disposition=RawCaptureDisposition.REDACTED_ARTIFACT,
                media_type="text/event-stream",
                stored_digest=stored_digest,
                uri=uri,
                redaction_profile=report.profile,
            ).to_dict()
        self._append(
            RawRecordType.RESPONSE_FRAME,
            payload={
                "frame_index": frame_index,
                "byte_size": len(frame.encode("utf-8")),
                "digest": bytes_digest(frame.encode("utf-8")),
                "frame": redacted["frame"] if frame_ref is None else "",
                "frame_ref": frame_ref,
                "capture_boundary": "complete_sse_event",
            },
            call_id=call_id,
            sequence_in_call=frame_index,
        )
        return _usage_from_sse_chunk(frame)

    def handle_unsupported(self, path: str) -> None:
        self.stats.unsupported_routes.append(path)
        self._append(
            RawRecordType.ERROR,
            payload={"stage": "route", "code": "unsupported_protocol", "path": path},
        )
        mode = str(self.binding.capture.mode)
        if mode in {
            CaptureMode.REQUIRED,
            CaptureMode.REQUIRED_EGRESS_ASSERTED,
            CaptureMode.OBSERVE_AND_TRANSFORM,
        }:
            raise UnsupportedProtocol(f"capture mode {mode} cannot observe route {path}")

    def passthrough_unsupported(
        self,
        *,
        method: str,
        request_target: str,
        headers: Mapping[str, str],
        body: bytes = b"",
    ) -> "ProxyResponse":
        """Forward an unobserved route only for explicit best-effort capture.

        The request target is resolved against the configured upstream boundary.
        Absolute-form targets and authority-form targets are rejected so a child
        cannot turn this fallback into an arbitrary-host forward proxy.
        """

        if str(self.binding.capture.mode) != CaptureMode.BEST_EFFORT:
            raise UnsupportedProtocol(
                "unsupported-route passthrough requires best_effort capture mode"
            )
        parsed = urlsplit(request_target)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
            raise UnsupportedProtocol("unsupported-route passthrough requires an origin-form path")
        endpoint = ProviderEndpointConfig(
            route=parsed.path,
            adapter_name="best_effort_passthrough",
            upstream_base_url=self.upstream_base_url,
            upstream_path=parsed.path,
            auth_kind=(
                UpstreamAuthKind.BEARER
                if self.upstream_api_key
                else UpstreamAuthKind.PASSTHROUGH
            ),
            api_key=self.upstream_api_key,
        )
        upstream_url = endpoint.upstream_url()
        if parsed.query:
            upstream_url = f"{upstream_url}?{parsed.query}"
        request = self._client.build_request(
            method,
            upstream_url,
            headers=_forward_headers(
                headers,
                endpoint,
                ensure_json_content_type=False,
            ),
            content=body,
        )
        response = self._client.send(request, stream=True)

        def generate(sink: Any) -> None:
            try:
                sink.begin(
                    response.status_code,
                    _client_headers(response.headers),
                )
                if response.is_stream_consumed:
                    if response.content:
                        sink.write(response.content)
                else:
                    for chunk in response.iter_raw():
                        if chunk:
                            sink.write(chunk)
            finally:
                response.close()

        return ProxyResponse(
            status_code=response.status_code,
            headers={},
            body=b"",
            stream=generate,
        )

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
            provider_adapters=self.routes.adapter_names,
            routes_enabled=(*self.routes.routes, MODELS_PATH),
            endpoint_identity_digest=text_digest(self.upstream_base_url),
        )


@dataclass(slots=True)
class ProxyResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes
    stream: Any | None


def _forward_headers(
    headers: Mapping[str, str],
    endpoint: ProviderEndpointConfig,
    *,
    ensure_json_content_type: bool = True,
) -> dict[str, str]:
    """Build upstream headers. Credentials pass through the process, never to disk."""

    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP and not name.lower().startswith("x-synth-")
    }
    for name, value in endpoint.static_headers.items():
        for actual in list(forwarded):
            if actual.lower() == str(name).lower():
                forwarded.pop(actual)
        forwarded[str(name)] = str(value)
    if endpoint.auth_kind == UpstreamAuthKind.BEARER and endpoint.api_key:
        for actual in list(forwarded):
            if actual.lower() == endpoint.auth_header.lower():
                forwarded.pop(actual)
        forwarded[endpoint.auth_header] = (
            f"{endpoint.auth_scheme} {endpoint.api_key}".strip()
        )
    elif endpoint.auth_kind == UpstreamAuthKind.HEADER and endpoint.api_key:
        for actual in list(forwarded):
            if actual.lower() == endpoint.auth_header.lower():
                forwarded.pop(actual)
        forwarded[endpoint.auth_header] = endpoint.api_key
    elif endpoint.auth_kind == UpstreamAuthKind.NONE:
        for name in ("authorization", "x-api-key"):
            for actual in list(forwarded):
                if actual.lower() == name:
                    forwarded.pop(actual)
    if ensure_json_content_type and not any(
        name.lower() == "content-type" for name in forwarded
    ):
        forwarded["Content-Type"] = "application/json"
    return forwarded


def _decode_request_body(
    headers: Mapping[str, str],
    body: bytes,
) -> tuple[bytes, tuple[str, ...]]:
    """Decode provider request content for inspection while retaining wire bytes.

    Encodings are removed in reverse application order per HTTP semantics. The
    original body and headers continue upstream unchanged; only the capture parser
    sees these bounded decoded bytes.
    """

    declared = next(
        (
            str(value)
            for name, value in headers.items()
            if str(name).lower() == "content-encoding"
        ),
        "",
    )
    encodings = tuple(
        item.strip().lower()
        for item in declared.split(",")
        if item.strip() and item.strip().lower() != "identity"
    )
    decoded = body
    for encoding in reversed(encodings):
        if encoding in {"gzip", "x-gzip"}:
            decoded = _bounded_zlib_decompress(decoded, 16 + zlib.MAX_WBITS)
        elif encoding == "deflate":
            try:
                decoded = _bounded_zlib_decompress(decoded, zlib.MAX_WBITS)
            except zlib.error:
                decoded = _bounded_zlib_decompress(decoded, -zlib.MAX_WBITS)
        elif encoding == "zstd":
            with zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(decoded)
            ) as reader:
                decoded = _read_bounded_stream(reader)
        else:
            raise UnsupportedContentEncoding(
                f"provider request content-encoding {encoding!r} is unsupported"
            )
        if len(decoded) > MAX_DECODED_REQUEST_BYTES:
            raise DecodedRequestTooLarge(
                "decoded provider request exceeds "
                f"{MAX_DECODED_REQUEST_BYTES} bytes"
            )
    return decoded, encodings


class _StreamingContentDecoder:
    """Incrementally decode response bytes for capture while forwarding wire bytes."""

    def __init__(self, headers: Mapping[str, str]) -> None:
        declared = next(
            (
                str(value)
                for name, value in headers.items()
                if str(name).lower() == "content-encoding"
            ),
            "",
        )
        encodings = tuple(
            item.strip().lower()
            for item in declared.split(",")
            if item.strip() and item.strip().lower() != "identity"
        )
        self._decoders: list[Any] = []
        for encoding in reversed(encodings):
            if encoding in {"gzip", "x-gzip"}:
                self._decoders.append(zlib.decompressobj(16 + zlib.MAX_WBITS))
            elif encoding == "deflate":
                self._decoders.append(zlib.decompressobj(zlib.MAX_WBITS))
            elif encoding == "zstd":
                self._decoders.append(
                    zstandard.ZstdDecompressor().decompressobj()
                )
            else:
                raise UnsupportedContentEncoding(
                    f"provider response content-encoding {encoding!r} is unsupported"
                )

    def feed(self, chunk: bytes) -> bytes:
        decoded = chunk
        for decoder in self._decoders:
            decoded = decoder.decompress(decoded)
        return decoded

    def finish(self) -> bytes:
        decoded_tail = b""
        for index, decoder in enumerate(self._decoders):
            flushed = decoder.flush()
            for downstream in self._decoders[index + 1 :]:
                flushed = downstream.decompress(flushed)
            decoded_tail += flushed
        return decoded_tail


def _bounded_zlib_decompress(payload: bytes, window_bits: int) -> bytes:
    decoder = zlib.decompressobj(window_bits)
    decoded = decoder.decompress(payload, MAX_DECODED_REQUEST_BYTES + 1)
    if len(decoded) > MAX_DECODED_REQUEST_BYTES or decoder.unconsumed_tail:
        raise DecodedRequestTooLarge(
            "decoded provider request exceeds "
            f"{MAX_DECODED_REQUEST_BYTES} bytes"
        )
    remaining = MAX_DECODED_REQUEST_BYTES + 1 - len(decoded)
    decoded += decoder.flush(remaining)
    if len(decoded) > MAX_DECODED_REQUEST_BYTES:
        raise DecodedRequestTooLarge(
            "decoded provider request exceeds "
            f"{MAX_DECODED_REQUEST_BYTES} bytes"
        )
    return decoded


def _read_bounded_stream(reader: Any) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = reader.read(
            min(1024 * 1024, MAX_DECODED_REQUEST_BYTES + 1 - total)
        )
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_DECODED_REQUEST_BYTES:
            raise DecodedRequestTooLarge(
                "decoded provider request exceeds "
                f"{MAX_DECODED_REQUEST_BYTES} bytes"
            )


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


def _append_query(url: str, query: str) -> str:
    """Append an origin-form query without decoding or normalizing its bytes."""

    if not query:
        return url
    parsed = urlsplit(url)
    combined = f"{parsed.query}&{query}" if parsed.query else query
    return parsed._replace(query=combined).geturl()


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
                    },
                )
                return
            if (
                path == RESPONSES_PATH
                and proxy.routes.resolve(RESPONSES_PATH) is not None
                and str(self.headers.get("Upgrade") or "").lower() == "websocket"
            ):
                # Current Codex clients attempt Responses-over-WebSocket first and
                # switch the session to HTTP only when the handshake explicitly
                # reports Upgrade Required. The capture proxy is intentionally an
                # HTTP/SSE observer, so advertise that boundary instead of letting
                # BaseHTTPRequestHandler return its generic 501 response.
                self._send(
                    426,
                    {
                        "Connection": "close",
                        "Upgrade": "websocket",
                        "Sec-WebSocket-Version": "13",
                    },
                    b"",
                )
                return
            if path == MODELS_PATH:
                response = proxy._client.get(
                    f"{proxy.upstream_base_url}/models",
                    headers=_forward_headers(
                        self.headers,
                        ProviderEndpointConfig(
                            route=MODELS_PATH,
                            adapter_name="models_passthrough",
                            upstream_base_url=proxy.upstream_base_url,
                            auth_kind=(
                                UpstreamAuthKind.BEARER
                                if proxy.upstream_api_key
                                else UpstreamAuthKind.PASSTHROUGH
                            ),
                            api_key=proxy.upstream_api_key,
                        ),
                    ),
                )
                self._send(
                    response.status_code,
                    _client_headers(response.headers, decoded=True),
                    response.content,
                )
                return
            self._unsupported(method="GET", request_target=self.path)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if proxy.routes.resolve(path) is None:
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length) if length else b""
                self._unsupported(
                    method="POST",
                    request_target=self.path,
                    body=body,
                )
                return
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length else b"{}"
            try:
                result = proxy.handle_provider_request(
                    path=path,
                    headers=self.headers,
                    body=body,
                    query=urlparse(self.path).query,
                )
            except CaptureContextError as exc:
                self._send_json(
                    403,
                    {"error": {"type": "capture_context_error", "message": str(exc)}},
                )
                return
            except UnsupportedContentEncoding as exc:
                self._send_json(
                    415,
                    {
                        "error": {
                            "type": "unsupported_content_encoding",
                            "message": str(exc),
                        }
                    },
                )
                return
            except DecodedRequestTooLarge as exc:
                self._send_json(
                    413,
                    {
                        "error": {
                            "type": "decoded_request_too_large",
                            "message": str(exc),
                        }
                    },
                )
                return
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

        def _unsupported(
            self,
            *,
            method: str,
            request_target: str,
            body: bytes = b"",
        ) -> None:
            path = urlparse(request_target).path
            try:
                proxy.handle_unsupported(path)
            except UnsupportedProtocol as exc:
                self._send_json(
                    501,
                    {"error": {"type": "unsupported_protocol", "message": str(exc)}},
                )
                return
            if str(proxy.binding.capture.mode) == CaptureMode.BEST_EFFORT:
                try:
                    result = proxy.passthrough_unsupported(
                        method=method,
                        request_target=request_target,
                        headers=self.headers,
                        body=body,
                    )
                    if result.stream is not None:
                        result.stream(_StreamSink(self))
                    else:
                        self._send(result.status_code, result.headers, result.body)
                except (UnsupportedProtocol, httpx.HTTPError) as exc:
                    self._send_json(
                        502,
                        {
                            "error": {
                                "type": "best_effort_passthrough_error",
                                "message": str(exc),
                            }
                        },
                    )
                return
            self._send_json(
                404,
                {"error": {"type": "not_found", "message": f"route {path} is not captured"}},
            )

    return Handler


__all__ = [
    "CHAT_COMPLETIONS_PATH",
    "CaptureContextError",
    "HEALTH_PATH",
    "MODELS_PATH",
    "PROXY_VERSION",
    "CaptureProxy",
    "ProxyResponse",
    "ProxyStats",
    "UnsupportedProtocol",
]
