"""mitmproxy addon for the scoped Trace V5 provider interceptor.

The supervisor launches ``mitmdump`` with an exact provider-authority allowlist.
This addon is the second enforcement layer: only configured provider routes are
rewritten to the local ``CaptureProxy``. An allowlisted provider host with an
unknown path fails closed instead of reaching the provider directly.

The module intentionally imports mitmproxy lazily so its routing logic remains
unit-testable without making mitmproxy a runtime dependency of synth-containers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


MITM_ADDON_CONFIG_ENV = "SYNTH_TRACE_MITM_CONFIG"
MITM_ADDON_CONFIG_SCHEMA_VERSION = "synth.trace-mitm-addon-config.v1"


@dataclass(frozen=True, slots=True)
class AddonRoute:
    provider_host: str
    provider_port: int
    provider_path: str
    capture_route: str


@dataclass(frozen=True, slots=True)
class AddonConfig:
    config_digest: str
    allowed_hosts: tuple[str, ...]
    routes: tuple[AddonRoute, ...]
    capture_proxy_host: str
    capture_proxy_port: int
    event_log: Path


def load_addon_config(path: Path) -> AddonConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != MITM_ADDON_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported scoped MITM addon config schema")
    allowed_hosts = tuple(
        sorted({_normalize_host(str(item)) for item in payload.get("allowed_hosts") or ()})
    )
    if not allowed_hosts:
        raise ValueError("scoped MITM addon requires a non-empty provider allowlist")
    proxy = dict(payload.get("capture_proxy") or {})
    proxy_host = _normalize_host(str(proxy.get("host") or ""))
    proxy_port = int(proxy.get("port") or 0)
    if not proxy_host or not (1 <= proxy_port <= 65535):
        raise ValueError("scoped MITM addon capture proxy endpoint is invalid")
    routes = tuple(
        AddonRoute(
            provider_host=_normalize_host(str(item["provider_host"])),
            provider_port=int(item["provider_port"]),
            provider_path=_normalize_path(str(item["provider_path"])),
            capture_route=_normalize_path(str(item["capture_route"])),
        )
        for item in payload.get("routes") or ()
    )
    if not routes:
        raise ValueError("scoped MITM addon requires at least one provider route")
    for route in routes:
        if route.provider_host not in allowed_hosts:
            raise ValueError("scoped MITM route host is outside the declared allowlist")
        if not (1 <= route.provider_port <= 65535):
            raise ValueError("scoped MITM route port is invalid")
    return AddonConfig(
        config_digest=str(payload["config_digest"]),
        allowed_hosts=allowed_hosts,
        routes=routes,
        capture_proxy_host=proxy_host,
        capture_proxy_port=proxy_port,
        event_log=Path(str(payload["event_log"])),
    )


def resolve_addon_route(
    config: AddonConfig,
    *,
    host: str,
    port: int,
    path: str,
) -> AddonRoute | None:
    normalized_host = _normalize_host(host)
    normalized_path = _normalize_path(urlsplit(path).path)
    matches = tuple(
        route
        for route in config.routes
        if route.provider_host == normalized_host
        and route.provider_port == int(port)
        and route.provider_path == normalized_path
    )
    if len(matches) > 1:
        raise RuntimeError("ambiguous scoped MITM provider route")
    return matches[0] if matches else None


class ScopedProviderAddon:
    """Rewrite recognized provider requests into the local capture proxy."""

    def __init__(
        self,
        config: AddonConfig,
        *,
        response_factory: Callable[[int, bytes, Mapping[str, str]], Any] | None = None,
    ) -> None:
        self.config = config
        self._response_factory = response_factory or _mitmproxy_response
        self._record_event(
            "addon_ready",
            config_digest=config.config_digest,
            allowed_host_count=len(config.allowed_hosts),
            route_count=len(config.routes),
        )

    def request(self, flow: Any) -> None:
        request = flow.request
        original_host = _normalize_host(str(request.host))
        original_port = int(request.port)
        original_path = str(request.path)
        is_decrypted_tls = str(request.scheme).lower() == "https"

        if original_host not in self.config.allowed_hosts:
            # ``--allow-hosts`` should prevent this branch for TLS. If mitmproxy
            # ever hands us a decrypted non-provider request anyway, deny it and
            # leave a credential-free receipt event.
            if is_decrypted_tls:
                flow.response = self._response(
                    421,
                    b"scoped trace MITM refused an undeclared TLS authority",
                )
                self._record_event(
                    "unexpected_tls_host",
                    authority_digest=_authority_digest(original_host, original_port),
                )
            return

        route = resolve_addon_route(
            self.config,
            host=original_host,
            port=original_port,
            path=original_path,
        )
        if route is None:
            flow.response = self._response(
                421,
                b"scoped trace MITM has no capture route for this provider request",
            )
            self._record_event(
                "unmapped_provider_route",
                authority_digest=_authority_digest(original_host, original_port),
                path_digest=_path_digest(original_path),
            )
            return

        query = urlsplit(original_path).query
        if query:
            # CaptureProxy currently registers exact provider paths. Refuse a
            # query-bearing request instead of silently forwarding it without the
            # query and changing provider semantics.
            flow.response = self._response(
                421,
                b"scoped trace MITM does not support provider query parameters",
            )
            self._record_event(
                "unmapped_provider_route",
                authority_digest=_authority_digest(original_host, original_port),
                path_digest=_path_digest(original_path),
                reason="query_parameters_unsupported",
            )
            return
        metadata = getattr(flow, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            flow.metadata = metadata
        metadata["synth_trace_mitm_routed"] = True
        request.scheme = "http"
        request.host = self.config.capture_proxy_host
        request.port = self.config.capture_proxy_port
        request.path = route.capture_route
        request.headers["host"] = (
            f"{self.config.capture_proxy_host}:{self.config.capture_proxy_port}"
        )
        request.headers["x-synth-mitm-route"] = route.capture_route
        self._record_event(
            "provider_routed",
            authority_digest=_authority_digest(original_host, original_port),
            capture_route=route.capture_route,
            query_present=False,
        )

    def responseheaders(self, flow: Any) -> None:
        """Preserve SSE/unary byte flow instead of letting mitmproxy buffer it."""

        metadata = getattr(flow, "metadata", None)
        if isinstance(metadata, dict) and metadata.get("synth_trace_mitm_routed"):
            flow.response.stream = True

    def _response(self, status: int, body: bytes) -> Any:
        return self._response_factory(
            status,
            body,
            {
                "content-type": "text/plain; charset=utf-8",
                "cache-control": "no-store",
            },
        )

    def _record_event(self, event: str, **values: Any) -> None:
        payload = {"event": event, **values}
        with self.config.event_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()


def _mitmproxy_response(
    status: int,
    body: bytes,
    headers: Mapping[str, str],
) -> Any:
    from mitmproxy import http

    return http.Response.make(status, body, dict(headers))


def _normalize_host(value: str) -> str:
    normalized = value.strip().lower().rstrip(".")
    if not normalized:
        raise ValueError("provider host must not be empty")
    if "/" in normalized or "\\" in normalized or any(char.isspace() for char in normalized):
        raise ValueError("provider host is invalid")
    return normalized


def _normalize_path(value: str) -> str:
    path = value.strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def _authority_digest(host: str, port: int) -> str:
    body = f"{_normalize_host(host)}:{int(port)}".encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _path_digest(path: str) -> str:
    return "sha256:" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _load_runtime_addon() -> ScopedProviderAddon | None:
    config_path = os.environ.get(MITM_ADDON_CONFIG_ENV)
    if not config_path:
        return None
    return ScopedProviderAddon(load_addon_config(Path(config_path)))


# mitmproxy discovers this module-level list when loaded through ``mitmdump -s``.
# The manager proves the addon actually loaded by waiting for its ``addon_ready``
# event; an absent/malformed environment therefore fails supervisor readiness.
_runtime_addon = _load_runtime_addon()
addons = [_runtime_addon] if _runtime_addon is not None else []


__all__ = [
    "AddonConfig",
    "AddonRoute",
    "MITM_ADDON_CONFIG_ENV",
    "MITM_ADDON_CONFIG_SCHEMA_VERSION",
    "ScopedProviderAddon",
    "load_addon_config",
    "resolve_addon_route",
]
