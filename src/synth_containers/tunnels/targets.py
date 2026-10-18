"""Local origin configuration; remote workers supply approved keys, never URLs.

Target snapshots are immutable so a live configuration update cannot redirect
an already admitted stream. Secret references remain local to the connector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import threading
from typing import Protocol
from urllib.parse import urlparse, urlunparse

MAX_TARGETS = 64
MAX_SECRET_HEADERS = 8
_HEADER = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")


class TargetError(ValueError):
    """Stable local binding refusal; messages never contain secret values."""


class SecretResolver(Protocol):
    def resolve(self, name: str) -> str: ...


class EnvironmentSecrets:
    """Resolve explicitly named environment values; no credential registry."""

    def resolve(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise TargetError("origin_secret_unavailable")
        return value


@dataclass(frozen=True, slots=True)
class OriginSecretHeader:
    header: str
    secret_name: str

    def __post_init__(self) -> None:
        if not _HEADER.fullmatch(self.header) or self.header.lower() in {
            "host",
            "connection",
            "content-length",
            "transfer-encoding",
            "upgrade",
            "sec-websocket-key",
            "sec-websocket-accept",
            "sec-websocket-version",
        }:
            raise TargetError("invalid_origin_secret_header")
        if (
            not self.secret_name
            or self.secret_name != self.secret_name.strip()
            or len(self.secret_name.encode()) > 128
        ):
            raise TargetError("invalid_origin_secret_reference")


@dataclass(frozen=True, slots=True)
class LocalOrigin:
    base_url: str
    service_identity: str
    credentials: tuple[OriginSecretHeader, ...] = ()

    def __post_init__(self) -> None:
        try:
            parsed = urlparse(self.base_url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError:
            raise TargetError("invalid_origin_address") from None
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise TargetError("invalid_origin_address")
        if (
            not self.service_identity
            or self.service_identity != self.service_identity.strip()
            or len(self.service_identity.encode()) > 128
        ):
            raise TargetError("invalid_origin_service_identity")
        if len(self.credentials) > MAX_SECRET_HEADERS or len(
            {item.header.lower() for item in self.credentials}
        ) != len(self.credentials):
            raise TargetError("invalid_origin_credential_bindings")
        netloc = (
            f"[{parsed.hostname}]:{port}" if ":" in parsed.hostname else f"{parsed.hostname}:{port}"
        )
        object.__setattr__(
            self,
            "base_url",
            urlunparse((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "", "")),
        )


@dataclass(frozen=True, slots=True)
class TargetRegistration:
    key: str
    generation: int
    origin: LocalOrigin


@dataclass(frozen=True, slots=True)
class BoundOrigin:
    registration: TargetRegistration
    headers: tuple[tuple[str, str], ...] = field(repr=False)


class LocalTargetRegistry:
    def __init__(self, *, secrets: SecretResolver | None = None) -> None:
        self._secrets = secrets or EnvironmentSecrets()
        self._lock = threading.Lock()
        self._targets: dict[str, TargetRegistration] = {}
        # A process-wide local version does not grow a tombstone map. Reusing
        # a key receives a new version, fencing stale local cleanup callbacks.
        self._next_generation = 0

    def register(
        self, key: str, origin: LocalOrigin, *, expected_generation: int
    ) -> TargetRegistration:
        if (
            not key.strip()
            or key != key.strip()
            or len(key.encode()) > 256
            or type(expected_generation) is not int
            or expected_generation < 0
        ):
            raise TargetError("invalid_target_registration")
        with self._lock:
            active = self._targets.get(key)
            previous = active.generation if active is not None else 0
            if previous != expected_generation:
                raise TargetError("stale_target_generation")
            if key not in self._targets and len(self._targets) >= MAX_TARGETS:
                raise TargetError("target_capacity")
            if self._next_generation >= 9_007_199_254_740_991:
                raise TargetError("target_generation_exhausted")
            self._next_generation += 1
            registered = TargetRegistration(key, self._next_generation, origin)
            self._targets[key] = registered
            return registered

    def unregister(self, key: str, *, expected_generation: int) -> None:
        if type(expected_generation) is not int or expected_generation <= 0:
            raise TargetError("invalid_target_generation")
        with self._lock:
            active = self._targets.get(key)
            if active is None:
                return
            if active.generation != expected_generation:
                raise TargetError("stale_target_generation")
            self._targets.pop(key, None)

    def bind(self, key: str, *, service_identity: str) -> BoundOrigin:
        with self._lock:
            registration = self._targets.get(key)
        if registration is None:
            raise TargetError("target_unavailable")
        if registration.origin.service_identity != service_identity:
            raise TargetError("origin_service_identity_mismatch")
        headers = []
        for credential in registration.origin.credentials:
            value = self._secrets.resolve(credential.secret_name)
            if (
                not value
                or len(value.encode()) > 8192
                or "\r" in value
                or "\n" in value
                or "\x00" in value
            ):
                raise TargetError("invalid_origin_secret_value")
            headers.append((credential.header, value))
        return BoundOrigin(registration, tuple(headers))
