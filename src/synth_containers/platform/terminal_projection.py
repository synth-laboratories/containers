"""Truthful terminal facts projected from a rollout's durable journal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..event_log import RolloutEventLog


_TERMINAL_STATUSES = {
    "completed",
    "failed",
    "truncated",
    "cancelled",
    "terminated",
    "stopped",
    "game_over",
}
_STEP_KINDS = {
    "action",
    "action_applied",
    "env.episode.closed",
    "frame",
    "observation",
    "reward_signal",
    "span.step.closed",
    "state_transition",
    "status",
    "terminal",
}


def terminal_journal_facts(log: RolloutEventLog | None) -> dict[str, Any]:
    """Return lifecycle facts without confusing sequence with env steps."""

    if log is None:
        return {}
    facts: dict[str, Any] = {}
    steps: int | None = None
    for envelope in log.after(0):
        payload = envelope.payload
        if envelope.kind in _STEP_KINDS:
            for key in ("steps", "step"):
                value = payload.get(key)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    steps = value if steps is None else max(steps, value)
        if envelope.kind not in {"env.episode.closed", "status", "terminal"}:
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status not in _TERMINAL_STATUSES:
            continue
        facts["status"] = status
        for key in ("reason", "detail", "error"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                facts[key] = value.strip()
    if steps is not None:
        facts["steps"] = steps
    return facts
