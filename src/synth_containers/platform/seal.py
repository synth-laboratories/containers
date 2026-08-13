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
