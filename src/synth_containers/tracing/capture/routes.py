"""Non-secret provider route configuration for the capture proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class UpstreamAuthKind(StrEnum):
    PASSTHROUGH = "passthrough"
    BEARER = "bearer"
    HEADER = "header"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class ProviderEndpointConfig:
    route: str
    adapter_name: str
    upstream_base_url: str
    auth_kind: UpstreamAuthKind | str = UpstreamAuthKind.PASSTHROUGH
    auth_header: str = "authorization"
    auth_scheme: str = "Bearer"
    api_key: str | None = field(default=None, repr=False)
    static_headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    upstream_path: str | None = None

    def upstream_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        route = self.upstream_path or self.route
        if not route.startswith("/"):
            route = f"/{route}"
        if base.endswith("/v1") and route.startswith("/v1/"):
            return base + route[len("/v1") :]
        return base + route


class ProviderRouteRegistry:
    def __init__(self, endpoints: tuple[ProviderEndpointConfig, ...]) -> None:
        self._endpoints: dict[str, ProviderEndpointConfig] = {}
        for endpoint in endpoints:
            if endpoint.route in self._endpoints:
                raise ValueError(f"duplicate provider capture route: {endpoint.route}")
            self._endpoints[endpoint.route] = endpoint

    def resolve(self, route: str) -> ProviderEndpointConfig | None:
        return self._endpoints.get(route)

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(sorted(self._endpoints))

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(sorted({item.adapter_name for item in self._endpoints.values()}))


__all__ = [
    "ProviderEndpointConfig",
    "ProviderRouteRegistry",
    "UpstreamAuthKind",
]
