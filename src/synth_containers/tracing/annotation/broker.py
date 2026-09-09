"""Paid-compute reservations: the only way a paid annotator gets to run.

Containers never decides whether money may be spent and never trusts a caller's
claim that it may. The host's approval broker issues an opaque reservation
receipt; Containers *claims* it (single use, atomically), binds it to one job,
runs under its cap, and *reconciles* what actually happened. Reservation ids are
the only thing that crosses the HTTP/MCP boundary.

``PaidComputeBroker`` is the contract Workshop implements. ``LocalReservationBroker``
is a file-backed reference implementation for development and tests.
``DenyAllBroker`` is the default: with no broker configured, no paid job starts.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterator, Protocol

from synth_containers.serde import JsonDataclassMixin

from ..canonical import record_id, utc_now

RESERVATION_SCHEMA_VERSION = "synth.paid-compute-reservation.v1"
USD_MICROS = 1_000_000


class ReservationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ReservationBindingV1(JsonDataclassMixin):
    """What a reservation may be spent on. Every field must match the claiming job exactly.

    ``model`` and ``session_id`` are typed optional only so a *binding built from a
    job* can carry "unknown"; a reservation itself is never issued with either
    missing unless the issuer explicitly opts into a wildcard, which is a
    separately audited capability (see ``LocalReservationBroker.issue``).
    """

    trace_digest: str
    annotator_id: str
    model: str | None
    session_id: str | None = None

    def is_exact(self) -> bool:
        return bool(self.trace_digest and self.annotator_id and self.model and self.session_id)


@dataclass(frozen=True, slots=True)
class PaidComputeReservationV1(JsonDataclassMixin):
    """One bounded, single-use approval issued by the host broker."""

    reservation_id: str
    issued_at: str
    cap_usd_micros: int
    binding: ReservationBindingV1
    approver: str = ""
    expires_at: str | None = None
    claimed_by_job_id: str | None = None
    claimed_at: str | None = None
    reconciled_at: str | None = None
    actual_cost_usd_micros: int | None = None
    outcome: str | None = None
    schema_version: str = RESERVATION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cap_usd(self) -> float:
        return self.cap_usd_micros / USD_MICROS


class PaidComputeBroker(Protocol):
    """Host-owned.

    ``claim`` must be atomic and single-use *per job id*: a second claim for the
    same ``job_id`` returns the same reservation (so a crashed preparation can
    resume), any other job id is refused. ``reconcile`` must be idempotent and
    must raise ``ReservationError`` (not swallow) when it cannot record the
    outcome, so the caller's outbox retries it.
    """

    def claim(self, reservation_id: str, *, binding: ReservationBindingV1, job_id: str) -> PaidComputeReservationV1: ...

    def reconcile(self, reservation_id: str, *, job_id: str, outcome: str, actual_cost_usd_micros: int | None) -> None: ...


class DenyAllBroker:
    """No broker means no paid execution. Deterministic jobs are unaffected."""

    def claim(self, reservation_id: str, *, binding: ReservationBindingV1, job_id: str) -> PaidComputeReservationV1:
        raise ReservationError("reservation_broker_unavailable", "no paid-compute broker is configured; paid annotators cannot run")

    def reconcile(self, reservation_id: str, *, job_id: str, outcome: str, actual_cost_usd_micros: int | None) -> None:
        raise ReservationError("reservation_broker_unavailable", "no paid-compute broker is configured; reconciliation stays pending")


class FlakyBroker:
    """Test double: wraps a broker and fails ``reconcile`` until ``fail_reconciles`` is exhausted."""

    def __init__(self, inner: PaidComputeBroker, *, fail_reconciles: int = 1) -> None:
        self.inner = inner
        self.fail_reconciles = fail_reconciles
        self.reconcile_calls = 0

    def claim(self, reservation_id: str, *, binding: ReservationBindingV1, job_id: str) -> PaidComputeReservationV1:
        return self.inner.claim(reservation_id, binding=binding, job_id=job_id)

    def reconcile(self, reservation_id: str, *, job_id: str, outcome: str, actual_cost_usd_micros: int | None) -> None:
        self.reconcile_calls += 1
        if self.fail_reconciles > 0:
            self.fail_reconciles -= 1
            raise ReservationError("broker_unreachable", "simulated broker outage")
        self.inner.reconcile(reservation_id, job_id=job_id, outcome=outcome, actual_cost_usd_micros=actual_cost_usd_micros)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class LocalReservationBroker:
    """File-backed reference broker: issue → claim (once) → reconcile.

    Not for production: it exists so the claim/reconcile contract has a runnable
    implementation and so tests can prove forged or replayed ids are refused.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.root / ".lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _path(self, reservation_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in reservation_id)
        return self.root / f"{safe}.json"

    def _read(self, reservation_id: str) -> PaidComputeReservationV1 | None:
        path = self._path(reservation_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("reservation_id") != reservation_id:
            return None
        binding = ReservationBindingV1(**payload.pop("binding"))
        return PaidComputeReservationV1(binding=binding, **payload)

    def _write(self, reservation: PaidComputeReservationV1) -> None:
        path = self._path(reservation.reservation_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reservation.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def issue(self, *, cap_usd_micros: int, binding: ReservationBindingV1, approver: str = "", expires_at: str | None = None, allow_wildcard: bool = False) -> PaidComputeReservationV1:
        if cap_usd_micros <= 0:
            raise ReservationError("reservation_cap_invalid", "cap must be a positive number of USD micros")
        if not binding.is_exact() and not allow_wildcard:
            raise ReservationError(
                "reservation_binding_incomplete",
                "reservations must bind an exact trace digest, annotator, model, and session; "
                "wildcards require allow_wildcard=True and are recorded as such",
            )
        issued_at = utc_now()
        reservation = PaidComputeReservationV1(
            reservation_id=record_id("rsv", kind="paid_compute_reservation", key={"binding": binding.to_dict(), "issued_at": issued_at, "cap": cap_usd_micros, "nonce": os.urandom(8).hex()}),
            issued_at=issued_at,
            cap_usd_micros=int(cap_usd_micros),
            binding=binding,
            approver=approver,
            expires_at=expires_at,
            metadata={"wildcard": not binding.is_exact()},
        )
        with self._lock():
            self._write(reservation)
        return reservation

    def claim(self, reservation_id: str, *, binding: ReservationBindingV1, job_id: str) -> PaidComputeReservationV1:
        with self._lock():
            reservation = self._read(reservation_id)
            if reservation is None:
                raise ReservationError("reservation_unknown", f"reservation {reservation_id} was not issued by this broker")
            if reservation.claimed_by_job_id is not None:
                if reservation.claimed_by_job_id == job_id and reservation.reconciled_at is None:
                    return reservation  # idempotent resume of the same preparation
                raise ReservationError("reservation_consumed", f"reservation {reservation_id} was already claimed by {reservation.claimed_by_job_id}")
            if reservation.expires_at is not None and utc_now() > reservation.expires_at:
                raise ReservationError("reservation_expired", f"reservation {reservation_id} expired at {reservation.expires_at}")
            expected = reservation.binding
            wildcard = bool(reservation.metadata.get("wildcard"))
            if expected.trace_digest != binding.trace_digest:
                raise ReservationError("reservation_binding_mismatch", "reservation is bound to a different trace digest")
            if expected.annotator_id != binding.annotator_id:
                raise ReservationError("reservation_binding_mismatch", "reservation is bound to a different annotator")
            if expected.model != binding.model and not (wildcard and expected.model is None):
                raise ReservationError("reservation_binding_mismatch", "reservation is bound to a different model")
            if expected.session_id != binding.session_id and not (wildcard and expected.session_id is None):
                raise ReservationError("reservation_binding_mismatch", "reservation is bound to a different session")
            claimed = replace(reservation, claimed_by_job_id=job_id, claimed_at=utc_now())
            self._write(claimed)
            return claimed

    def reconcile(self, reservation_id: str, *, job_id: str, outcome: str, actual_cost_usd_micros: int | None) -> None:
        with self._lock():
            reservation = self._read(reservation_id)
            if reservation is None:
                raise ReservationError("reservation_unknown", f"reservation {reservation_id} is unknown; cannot reconcile")
            if reservation.claimed_by_job_id != job_id:
                raise ReservationError("reservation_binding_mismatch", f"reservation {reservation_id} is not claimed by {job_id}")
            if reservation.reconciled_at is not None:
                return  # idempotent
            self._write(replace(reservation, reconciled_at=utc_now(), outcome=outcome, actual_cost_usd_micros=actual_cost_usd_micros))

    def get(self, reservation_id: str) -> PaidComputeReservationV1 | None:
        with self._lock():
            return self._read(reservation_id)


def usd_to_micros(value: float) -> int:
    return int(round(float(value) * USD_MICROS))


__all__ = [
    "RESERVATION_SCHEMA_VERSION",
    "USD_MICROS",
    "DenyAllBroker",
    "FlakyBroker",
    "LocalReservationBroker",
    "PaidComputeBroker",
    "PaidComputeReservationV1",
    "ReservationBindingV1",
    "ReservationError",
    "usd_to_micros",
]
