"""Signed reservations: a host issues them, a container claims them without calling home.

A container has no route to the host's IPC, so the reservation itself carries
the authority: a JSON payload (binding, cap in USD micros, expiry, issuer nonce)
plus an HMAC-SHA256 over its canonical bytes under a secret the host injects at
launch (``SYNTH_ANNOTATION_BROKER_SECRET``). Claiming verifies the signature,
the binding, and the expiry, then records the reservation id as consumed in a
lock-protected local register (idempotent for the same job id). Reconciliation
is posted to the host when ``reconcile_url`` is configured; otherwise it is
recorded locally and served for the host to pull, so nothing is lost either way.
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import hmac
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from ..canonical import canonical_bytes, record_id, utc_now
from .broker import PaidComputeReservationV1, ReservationBindingV1, ReservationError

SIGNED_RESERVATION_VERSION = "synth.signed-reservation.v1"


def _sign(secret: bytes, payload: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(hmac.new(secret, canonical_bytes(payload), hashlib.sha256).digest()).decode("ascii").rstrip("=")


def issue_signed_reservation(*, secret: bytes, cap_usd_micros: int, binding: ReservationBindingV1, approver: str = "", expires_at: str | None = None, issuer: str = "workshop") -> str:
    """Host side: mint a self-contained reservation token (base64 JSON with an HMAC)."""

    if cap_usd_micros <= 0:
        raise ReservationError("reservation_cap_invalid", "cap must be positive USD micros")
    if not binding.is_exact():
        raise ReservationError("reservation_binding_incomplete", "signed reservations must bind trace, annotator, model, and session")
    issued_at = utc_now()
    payload = {
        "version": SIGNED_RESERVATION_VERSION,
        "reservation_id": record_id("rsv", kind="signed_reservation", key={"binding": binding.to_dict(), "issued_at": issued_at, "cap": cap_usd_micros, "nonce": os.urandom(8).hex()}),
        "issued_at": issued_at,
        "cap_usd_micros": int(cap_usd_micros),
        "binding": binding.to_dict(),
        "approver": approver,
        "expires_at": expires_at,
        "issuer": issuer,
    }
    envelope = {"payload": payload, "signature": _sign(secret, payload)}
    return base64.urlsafe_b64encode(json.dumps(envelope, separators=(",", ":"), sort_keys=True).encode("utf-8")).decode("ascii")


def decode_signed_reservation(token: str, *, secret: bytes | None = None) -> dict[str, Any]:
    try:
        envelope = json.loads(base64.urlsafe_b64decode(token.encode("ascii") + b"=" * (-len(token) % 4)))
    except (ValueError, TypeError) as error:
        raise ReservationError("reservation_unknown", "reservation token is not decodable") from error
    payload = envelope.get("payload") if isinstance(envelope, dict) else None
    if not isinstance(payload, dict) or payload.get("version") != SIGNED_RESERVATION_VERSION:
        raise ReservationError("reservation_unknown", "reservation token has an unknown shape")
    if secret is not None and not hmac.compare_digest(_sign(secret, payload), str(envelope.get("signature") or "")):
        raise ReservationError("reservation_signature_invalid", "reservation signature does not verify")
    return payload


class SignedReservationBroker:
    """Container side ``PaidComputeBroker`` for host-signed reservations."""

    def __init__(self, root: Path, *, secret: bytes, reconcile_url: str | None = None, reconcile_token: str | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.secret = secret
        self.reconcile_url = reconcile_url.rstrip("/") if reconcile_url else None
        self.reconcile_token = reconcile_token

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
        data = json.loads(path.read_text(encoding="utf-8"))
        return PaidComputeReservationV1(binding=ReservationBindingV1(**data.pop("binding")), **data)

    def _write(self, reservation: PaidComputeReservationV1) -> None:
        path = self._path(reservation.reservation_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reservation.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def get(self, reservation_id: str) -> PaidComputeReservationV1 | None:
        with self._lock():
            return self._read(reservation_id)

    def claim(self, token: str, *, binding: ReservationBindingV1, job_id: str) -> PaidComputeReservationV1:
        payload = decode_signed_reservation(token, secret=self.secret)
        reservation_id = str(payload["reservation_id"])
        expected = ReservationBindingV1(**payload["binding"])
        with self._lock():
            existing = self._read(reservation_id)
            if existing is not None:
                if existing.claimed_by_job_id == job_id and existing.reconciled_at is None:
                    return existing
                raise ReservationError("reservation_consumed", f"reservation {reservation_id} was already claimed by {existing.claimed_by_job_id}")
            if payload.get("expires_at") and utc_now() > str(payload["expires_at"]):
                raise ReservationError("reservation_expired", f"reservation {reservation_id} expired")
            if expected != binding:
                raise ReservationError("reservation_binding_mismatch", "reservation is bound to a different trace/annotator/model/session")
            claimed = PaidComputeReservationV1(
                reservation_id=reservation_id,
                issued_at=str(payload["issued_at"]),
                cap_usd_micros=int(payload["cap_usd_micros"]),
                binding=expected,
                approver=str(payload.get("approver") or ""),
                expires_at=payload.get("expires_at"),
                claimed_by_job_id=job_id,
                claimed_at=utc_now(),
                metadata={"issuer": payload.get("issuer"), "signed": True},
            )
            self._write(claimed)
            return claimed

    def reconcile(self, reservation_id: str, *, job_id: str, outcome: str, actual_cost_usd_micros: int | None) -> None:
        with self._lock():
            reservation = self._read(reservation_id)
            if reservation is None:
                raise ReservationError("reservation_unknown", f"reservation {reservation_id} was never claimed here")
            if reservation.claimed_by_job_id != job_id:
                raise ReservationError("reservation_binding_mismatch", f"reservation {reservation_id} belongs to {reservation.claimed_by_job_id}")
            if reservation.reconciled_at is not None:
                return
        if self.reconcile_url:
            import httpx

            headers = {"Content-Type": "application/json"}
            if self.reconcile_token:
                headers["Authorization"] = f"Bearer {self.reconcile_token}"
            try:
                response = httpx.post(f"{self.reconcile_url}/reservations/{reservation_id}/reconcile", json={"job_id": job_id, "outcome": outcome, "actual_cost_usd_micros": actual_cost_usd_micros}, headers=headers, timeout=10.0)
            except httpx.HTTPError as error:
                raise ReservationError("broker_unreachable", f"reconcile failed: {error}") from error
            if response.status_code >= 400:
                raise ReservationError("reconcile_rejected", f"{response.status_code}: {response.text[:200]}")
        with self._lock():
            reservation = self._read(reservation_id)
            if reservation is not None and reservation.reconciled_at is None:
                self._write(replace(reservation, reconciled_at=utc_now(), outcome=outcome, actual_cost_usd_micros=actual_cost_usd_micros, metadata={**reservation.metadata, "reconciled_via": "host" if self.reconcile_url else "local"}))

    def reconciled(self) -> list[dict[str, Any]]:
        """What the host pulls when it did not receive reconciliations by push."""

        with self._lock():
            rows = []
            for path in sorted(self.root.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("reconciled_at"):
                    rows.append(data)
            return rows


__all__ = ["SIGNED_RESERVATION_VERSION", "SignedReservationBroker", "decode_signed_reservation", "issue_signed_reservation"]
