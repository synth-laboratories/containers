"""Lite Trace V5 seal from a durable rollout log. High-water must match live."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..event_log import RolloutEventLog

SCHEMA = "synth.trace.v5"


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def seal_rollout_log(log: RolloutEventLog, *, pin: dict[str, Any] | None = None) -> dict[str, Any]:
    events = []
    for item in log.after(0):
        if item.control:
            continue
        events.append(
            {
                "event_id": str(item.sequence),
                "event_type": item.kind,
                "occurred_at": item.ts,
                "order": {"chronological_sequence": item.sequence},
                "payload": item.payload,
                "digest": item.digest,
            }
        )
    body = {
        "schema_version": SCHEMA,
        "trace_id": log.rollout_id,
        "rollout_id": log.rollout_id,
        "stream.id": log.stream_id,
        "high_water": log.high_water,
        "closed": log.closed,
        "events": events,
        "pin": pin or {},
        "capture.closed": True,
    }
    sealed = {**body, "content_digest": _digest(body)}
    return sealed


def validate_rollout_seal(seal: dict[str, Any]) -> None:
    """Fail closed unless a persisted lite Trace V5 seal is self-consistent."""
    if seal.get("schema_version") != SCHEMA:
        raise ValueError("trace_seal_schema_mismatch")
    supplied = seal.get("content_digest")
    body = {key: value for key, value in seal.items() if key != "content_digest"}
    if supplied != _digest(body):
        raise ValueError("trace_seal_digest_mismatch")
    events = seal.get("events")
    high_water = seal.get("high_water")
    if not isinstance(events, list) or isinstance(high_water, bool) or not isinstance(high_water, int):
        raise ValueError("trace_seal_shape_invalid")
    sequences = [((row.get("order") or {}).get("chronological_sequence")) for row in events]
    if sequences != list(range(1, high_water + 1)):
        raise ValueError("trace_seal_sequence_mismatch")
    if seal.get("closed") is not True or seal.get("capture.closed") is not True:
        raise ValueError("trace_seal_not_closed")
