"""Own and recover SynthTunnel leases around restartable container operations.

# See: Jstack/.jstack/records/decisions/containers/2026-07-25-synthtunnel-runtime-ownership.md
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, runtime_checkable


ResultT = TypeVar("ResultT")
_AGENT_OFFLINE_PATTERN = re.compile(r"(?<![A-Z0-9_])AGENT_OFFLINE(?![A-Z0-9_])")


class SynthTunnelSupervisorState(StrEnum):
    NEW = "new"
    ATTACHED = "attached"
    REPLACING = "replacing"
    CLOSED = "closed"


class SynthTunnelAgentOffline(RuntimeError):
    """The control plane reports that the local tunnel agent is detached."""


class SynthTunnelRecoveryExhausted(RuntimeError):
    """A restartable operation remained offline after all lease generations."""


@runtime_checkable
class SynthTunnelLease(Protocol):
    @property
    def lease_id(self) -> str: ...

    @property
    def public_url(self) -> str: ...

    @property
    def worker_token(self) -> str: ...

    def close(self) -> None: ...


@runtime_checkable
class SynthTunnelLeaseProvider(Protocol):
    def open_synth_tunnel(
        self,
        local_url: str,
        *,
        requested_ttl_seconds: int,
        metadata: Mapping[str, object],
        capabilities: Mapping[str, object],
    ) -> SynthTunnelLease: ...


@dataclass(frozen=True, slots=True)
class SynthTunnelCredentials:
    lease_id: str
    public_url: str
    worker_token: str
    generation: int


@dataclass(frozen=True, slots=True)
class SynthTunnelEvent:
    kind: str
    monotonic_seconds: float
    generation: int
    operation_name: str | None = None
    detail: str | None = None


@runtime_checkable
class SynthTunnelOperation(Protocol, Generic[ResultT]):
    def __call__(self, credentials: SynthTunnelCredentials) -> ResultT: ...


class SynthTunnelSupervisor:
    """Keep one attached lease generation around restartable operations.

    Operations must persist their own progress before raising
    ``SynthTunnelAgentOffline``. The supervisor replaces the lease and reruns
    only that operation; it never guesses whether arbitrary side effects are
    safe to replay.
    """

    def __init__(
        self,
        *,
        provider: SynthTunnelLeaseProvider,
        local_url: str,
        requested_ttl_seconds: int = 3600,
        metadata: Mapping[str, object] | None = None,
        capabilities: Mapping[str, object] | None = None,
        max_replacements_per_operation: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not local_url.startswith(("http://", "https://")):
            raise ValueError("local_url must be absolute HTTP(S)")
        if requested_ttl_seconds <= 0:
            raise ValueError("requested_ttl_seconds must be positive")
        if max_replacements_per_operation < 0:
            raise ValueError("max_replacements_per_operation cannot be negative")
        self._provider = provider
        self._local_url = local_url.rstrip("/")
        self._requested_ttl_seconds = requested_ttl_seconds
        self._metadata = dict(metadata or {})
        self._capabilities = dict(capabilities or {})
        self._max_replacements_per_operation = max_replacements_per_operation
        self._clock = clock
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._state = SynthTunnelSupervisorState.NEW
        self._lease: SynthTunnelLease | None = None
        self._generation = 0
        self._events: list[SynthTunnelEvent] = []

    @property
    def state(self) -> SynthTunnelSupervisorState:
        with self._lock:
            return self._state

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def events(self) -> tuple[SynthTunnelEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def start(self) -> SynthTunnelCredentials:
        with self._lock:
            if self._state == SynthTunnelSupervisorState.CLOSED:
                raise RuntimeError("SynthTunnel supervisor is closed")
            if self._lease is None:
                self._attach(kind="lease_attached")
            return self._credentials()

    def run_restartable(
        self,
        operation_name: str,
        operation: SynthTunnelOperation[ResultT],
    ) -> ResultT:
        if not operation_name.strip():
            raise ValueError("operation_name is required")
        with self._operation_lock:
            credentials = self.start()
            replacements = 0
            while True:
                with self._lock:
                    self._record("operation_started", operation_name=operation_name)
                try:
                    result = operation(credentials)
                except SynthTunnelAgentOffline as error:
                    with self._lock:
                        self._record(
                            "agent_offline",
                            operation_name=operation_name,
                            detail=type(error).__name__,
                        )
                    if replacements >= self._max_replacements_per_operation:
                        with self._lock:
                            self._record(
                                "recovery_exhausted",
                                operation_name=operation_name,
                            )
                            self._detach(suppress_close_error=True)
                        raise SynthTunnelRecoveryExhausted(
                            f"operation {operation_name!r} exhausted "
                            f"{replacements} lease replacements"
                        ) from error
                    replacements += 1
                    credentials = self._replace()
                    continue
                except Exception as error:
                    with self._lock:
                        self._record(
                            "operation_failed",
                            operation_name=operation_name,
                            detail=type(error).__name__,
                        )
                    raise
                with self._lock:
                    self._record("operation_completed", operation_name=operation_name)
                return result

    def close(self) -> None:
        with self._lock:
            if self._state == SynthTunnelSupervisorState.CLOSED:
                return
            lease = self._lease
            self._lease = None
            self._state = SynthTunnelSupervisorState.CLOSED
            if lease is not None:
                try:
                    lease.close()
                except Exception as error:
                    self._record("lease_close_failed", detail=type(error).__name__)
                    self._record("supervisor_closed")
                    raise
            self._record("supervisor_closed")

    def __enter__(self) -> "SynthTunnelSupervisor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _replace(self) -> SynthTunnelCredentials:
        with self._lock:
            if self._state == SynthTunnelSupervisorState.CLOSED:
                raise RuntimeError("SynthTunnel supervisor is closed")
            self._state = SynthTunnelSupervisorState.REPLACING
            self._detach(suppress_close_error=True)
            self._attach(kind="lease_replaced")
            return self._credentials()

    def _detach(self, *, suppress_close_error: bool) -> None:
        previous = self._lease
        self._lease = None
        close_error: Exception | None = None
        if previous is not None:
            try:
                previous.close()
            except Exception as error:
                close_error = error
                self._record("lease_close_failed", detail=type(error).__name__)
        if self._state != SynthTunnelSupervisorState.CLOSED:
            self._state = SynthTunnelSupervisorState.NEW
        self._record("lease_detached")
        if close_error is not None and not suppress_close_error:
            raise close_error

    def _attach(self, *, kind: str) -> None:
        try:
            lease = self._provider.open_synth_tunnel(
                self._local_url,
                requested_ttl_seconds=self._requested_ttl_seconds,
                metadata=self._metadata,
                capabilities=self._capabilities,
            )
        except Exception as error:
            self._state = SynthTunnelSupervisorState.NEW
            self._record("lease_attach_failed", detail=type(error).__name__)
            raise
        try:
            complete = bool(lease.public_url and lease.worker_token and lease.lease_id)
        except Exception as error:
            self._record("lease_rejected", detail=type(error).__name__)
            self._close_rejected_lease(lease)
            self._state = SynthTunnelSupervisorState.NEW
            raise RuntimeError("SynthTunnel provider returned an unreadable lease") from error
        if not complete:
            self._record("lease_rejected", detail="incomplete")
            self._close_rejected_lease(lease)
            self._state = SynthTunnelSupervisorState.NEW
            raise RuntimeError("SynthTunnel provider returned an incomplete lease")
        self._lease = lease
        self._generation += 1
        self._state = SynthTunnelSupervisorState.ATTACHED
        self._record(kind)

    def _close_rejected_lease(self, lease: SynthTunnelLease) -> None:
        try:
            lease.close()
        except Exception as error:
            self._record("lease_close_failed", detail=type(error).__name__)

    def _credentials(self) -> SynthTunnelCredentials:
        lease = self._lease
        if lease is None or self._state != SynthTunnelSupervisorState.ATTACHED:
            raise RuntimeError("SynthTunnel has no attached lease")
        return SynthTunnelCredentials(
            lease_id=lease.lease_id,
            public_url=lease.public_url,
            worker_token=lease.worker_token,
            generation=self._generation,
        )

    def _record(
        self,
        kind: str,
        *,
        operation_name: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._events.append(
            SynthTunnelEvent(
                kind=kind,
                monotonic_seconds=self._clock(),
                generation=self._generation,
                operation_name=operation_name,
                detail=detail,
            )
        )


def is_synth_tunnel_agent_offline(*, status_code: int, detail: str | bytes) -> bool:
    """Classify the relay's terminal holder-loss response without broad 503 matching."""

    if status_code != 503:
        return False
    text = detail.decode("utf-8", errors="replace") if isinstance(detail, bytes) else detail
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return bool(_AGENT_OFFLINE_PATTERN.search(text.upper()))
    return _payload_contains_agent_offline(payload)


def raise_for_synth_tunnel_agent_offline(*, status_code: int, detail: str | bytes) -> None:
    """Raise the typed recovery signal for an exact relay holder-loss response."""

    if is_synth_tunnel_agent_offline(status_code=status_code, detail=detail):
        raise SynthTunnelAgentOffline("SynthTunnel relay reports that its local agent is offline")


def _payload_contains_agent_offline(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_payload_contains_agent_offline(item) for item in value.values())
    if isinstance(value, list):
        return any(_payload_contains_agent_offline(item) for item in value)
    return isinstance(value, str) and value.upper() == "AGENT_OFFLINE"
