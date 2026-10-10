"""Callback cancellation is separate from origin outcome and resource cleanup.

See docs/specs/tunnel-v2-connector.md. Cancelling an awaiter cannot establish
that its external provider, thread or subprocess stopped.
"""
from __future__ import annotations

from hashlib import sha256
import json

from .serde import JsonObject

TERMINAL_ROLLOUT_STATES = frozenset({"completed", "failed", "cancelled", "terminated", "stopped"})


def request_identity(payload: JsonObject) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def request_cancellation(payload: JsonObject, now: str) -> tuple[JsonObject, bool]:
    if payload["status"] in TERMINAL_ROLLOUT_STATES:
        return payload, False
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("cancel_requested"):
        return payload, False
    return {
        **payload, "status": "cancelling", "success_status": "running",
        "status_detail": "cancellation_requested", "updated_at": now,
        "metadata": {**metadata, "cancel_requested": True,
                     "callback_task": "running", "origin_outcome": "unverified",
                     "resource_cleanup": "unverified"},
    }, True


def callback_cancelled(payload: JsonObject, now: str) -> JsonObject:
    if payload["status"] in TERMINAL_ROLLOUT_STATES:
        return payload
    return {
        **payload, "status": "in_doubt", "success_status": "unknown",
        "status_detail": "callback_cancelled_origin_outcome_unverified", "updated_at": now,
        "metadata": {**dict(payload.get("metadata") or {}), "callback_task": "cancelled",
                     "origin_outcome": "unverified", "resource_cleanup": "unverified"},
    }
