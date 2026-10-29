"""Origin-side I/O for the SynthTunnel agent: the local target, HTTP and WebSocket.

Everything here talks to the confined local origin, never to the relay. Split out of
``relay.py`` (file-size cap) with behavior unchanged; ``relay`` re-exports these names.
"""

from __future__ import annotations

import http.client
import urllib.request
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse, urlunparse

from .attachment import LocalTarget as _LocalTarget
from .paths import confined_origin_url
from .relay_errors import SynthTunnelRelayError

# Response headers the agent owns rather than the origin. Everything else in a
# WS_OPEN_REJECT is forwarded verbatim -- notably ``upgrade`` and
# ``sec-websocket-version``, which are what tell a client to stop retrying the
# handshake against the Trace V5 capture proxy.
_WEBSOCKET_REJECT_FRAMING_HEADERS = {"connection", "content-length", "transfer-encoding"}

_WS_CLOSE_NORMAL = 1000
_WS_CLOSE_INTERNAL_ERROR = 1011


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


def _local_upstream_url(target: _LocalTarget, path: str, query: str) -> str:
    return confined_origin_url(target.base_url, path, query)
