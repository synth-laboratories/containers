"""Provider-normalization contracts shared by HTTP, WebSocket, and MITM capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..models.messages import MessagePartV5
from ..models.spans import UsageV5
from ..models.tokens import TokenCaptureV5


@dataclass(slots=True)
class NormalizedMessage:
    role: str
    parts: list[MessagePartV5] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class NormalizedProviderResult:
    messages: list[NormalizedMessage] = field(default_factory=list)
    usage: UsageV5 | None = None
    token_capture: TokenCaptureV5 | None = None
    diagnostics: list[str] = field(default_factory=list)
    provider_ids: dict[str, str] = field(default_factory=dict)
    terminal_observed: bool = False
    raw_events: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class ProviderStreamAssembler(Protocol):
    def feed(self, chunk: bytes) -> None: ...

    def finish(self) -> NormalizedProviderResult: ...


@runtime_checkable
class ProviderAdapter(Protocol):
    name: str
    version: str
    routes: tuple[str, ...]

    def normalize_request(self, body: Mapping[str, Any]) -> list[NormalizedMessage]: ...

    def normalize_unary(self, body: Mapping[str, Any]) -> NormalizedProviderResult: ...

    def new_stream(self) -> ProviderStreamAssembler: ...

    def usage(self, payload: Mapping[str, Any] | None) -> UsageV5: ...


class ProviderAdapterRegistry:
    def __init__(self) -> None:
        self._by_name: dict[str, ProviderAdapter] = {}
        self._by_route: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter) -> None:
        if adapter.name in self._by_name:
            raise ValueError(f"provider adapter already registered: {adapter.name}")
        self._by_name[adapter.name] = adapter
        for route in adapter.routes:
            if route in self._by_route:
                raise ValueError(f"provider route already registered: {route}")
            self._by_route[route] = adapter

    def by_name(self, name: str) -> ProviderAdapter | None:
        return self._by_name.get(name)

    def by_route(self, route: str) -> ProviderAdapter | None:
        return self._by_route.get(route)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_route))


__all__ = [
    "NormalizedMessage",
    "NormalizedProviderResult",
    "ProviderAdapter",
    "ProviderAdapterRegistry",
    "ProviderStreamAssembler",
]
