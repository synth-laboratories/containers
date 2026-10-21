"""Pure rollout journal retention projection, shared by platform readbacks."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from ..event_log import RolloutEventLog


def retention_ttl_seconds(runtime_config: dict[str, Any]) -> int:
    raw = (runtime_config or {}).get("journal_retention_ttl_seconds")
    try:
        ttl = int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        ttl = 0
    return ttl if ttl > 0 else 604_800


def journal_retention(log: RolloutEventLog, *, ttl_seconds: int, now: float | None = None) -> dict[str, Any]:
    ttl = ttl_seconds
    moment = time.time() if now is None else now
    expires_at = None
    released = False
    reason = None
    if log.closed:
        if log.last_acked >= log.high_water:
            released = True
            reason = "acked"
        if log.closed_at:
            closed_epoch = datetime.fromisoformat(
                log.closed_at.replace("Z", "+00:00")
            ).timestamp()
            expires_epoch = closed_epoch + ttl
            expires_at = (
                datetime.fromtimestamp(expires_epoch, tz=timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
            if not released and moment >= expires_epoch:
                released = True
                reason = "ttl_expired"
    return {
        "policy": "until-acked-or-ttl",
        "ttl_seconds": ttl,
        "acked": log.last_acked,
        "high_water": log.high_water,
        "closed": log.closed,
        "released": released,
        "released_reason": reason,
        "expires_at": expires_at,
    }
