"""HTTP client for a host-owned paid-compute broker (Workshop's approval broker).

Wire contract (JSON, bearer-authenticated by the host)::

    POST {base}/reservations/{reservation_id}/claim
         body   {"job_id", "binding": {"trace_digest","annotator_id","model","session_id"}}
         200    PaidComputeReservationV1 (claimed_by_job_id == job_id)
         409    {"code": "reservation_consumed" | "reservation_binding_mismatch" | "reservation_expired"}
         404    {"code": "reservation_unknown"}
    POST {base}/reservations/{reservation_id}/reconcile
         body   {"job_id", "outcome", "actual_cost_usd_micros"}
         200/204 acknowledged (idempotent)
         4xx/5xx -> ReservationError (the caller's ledger retries)

Anything but a 2xx raises ``ReservationError`` so the ledger records the
attempt; nothing is ever swallowed.
"""

from __future__ import annotations

from typing import Any

import httpx

from .broker import PaidComputeReservationV1, ReservationBindingV1, ReservationError


class HttpReservationBroker:
    def __init__(self, base_url: str, *, token: str | None = None, timeout_seconds: float = 10.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._client = client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        try:
            if self._client is not None:
                return self._client.post(f"{self.base_url}{path}", json=body, headers=self._headers(), timeout=self.timeout_seconds)
            with httpx.Client(timeout=self.timeout_seconds) as client:
                return client.post(f"{self.base_url}{path}", json=body, headers=self._headers())
        except httpx.HTTPError as error:
            raise ReservationError("broker_unreachable", f"broker request failed: {error}") from error

    @staticmethod
    def _error(response: httpx.Response, default: str) -> ReservationError:
        code = default
        message = response.text[:300]
        try:
            payload = response.json()
            if isinstance(payload, dict):
                code = str(payload.get("code") or payload.get("reason") or default)
                message = str(payload.get("message") or message)
        except ValueError:
            pass
        return ReservationError(code, f"{response.status_code}: {message}")

    def claim(self, reservation_id: str, *, binding: ReservationBindingV1, job_id: str) -> PaidComputeReservationV1:
        response = self._post(f"/reservations/{reservation_id}/claim", {"job_id": job_id, "binding": binding.to_dict()})
        if response.status_code == 404:
            raise self._error(response, "reservation_unknown")
        if response.status_code == 409:
            raise self._error(response, "reservation_consumed")
        if response.status_code >= 400:
            raise self._error(response, "reservation_rejected")
        payload = response.json()
        payload_binding = payload.pop("binding", None) or binding.to_dict()
        reservation = PaidComputeReservationV1(binding=ReservationBindingV1(**payload_binding), **{k: v for k, v in payload.items() if k in PaidComputeReservationV1.__dataclass_fields__})
        if reservation.claimed_by_job_id != job_id:
            raise ReservationError("reservation_binding_mismatch", "broker returned a reservation claimed by a different job")
        return reservation

    def reconcile(self, reservation_id: str, *, job_id: str, outcome: str, actual_cost_usd_micros: int | None) -> None:
        response = self._post(f"/reservations/{reservation_id}/reconcile", {"job_id": job_id, "outcome": outcome, "actual_cost_usd_micros": actual_cost_usd_micros})
        if response.status_code >= 400:
            raise self._error(response, "reconcile_rejected")


__all__ = ["HttpReservationBroker"]
