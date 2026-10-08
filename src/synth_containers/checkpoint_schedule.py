"""Configured gold checkpoint cadence and restore-eligible ring buffer.

Parse once per rollout. The episode loop still invokes the capture callback at
step 0 and after every plan; due-check and ``keep_last`` live here so a skipped
boundary is a no-op (``None``) rather than a different call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_MODES = frozenset(
    {"per_policy_call", "every_n_policy_calls", "every_n_env_steps"}
)
_CADENCE_MODES = frozenset({"every_n_policy_calls", "every_n_env_steps"})


@dataclass(frozen=True)
class CheckpointSchedule:
    mode: str
    checkpoint_id_prefix: str
    n: int | None = None
    keep_last: int | None = None


def parse_checkpoint_schedule(raw: dict[str, Any]) -> CheckpointSchedule:
    """Validate a pin ``checkpoint_schedule`` object. Fail closed on bad input."""

    mode = str(raw.get("mode") or "").strip()
    if mode not in SUPPORTED_MODES:
        raise RuntimeError(f"unsupported_checkpoint_schedule:{mode or 'missing'}")
    prefix = str(raw.get("checkpoint_id_prefix") or "").strip()
    if not prefix:
        raise RuntimeError("checkpoint_schedule requires checkpoint_id_prefix")
    n = _require_positive_int(raw.get("n"), "n") if mode in _CADENCE_MODES else None
    keep_last = None
    if "keep_last" in raw and raw["keep_last"] is not None:
        keep_last = _require_positive_int(raw["keep_last"], "keep_last")
    return CheckpointSchedule(
        mode=mode,
        checkpoint_id_prefix=prefix,
        n=n,
        keep_last=keep_last,
    )


def checkpoint_is_due(
    schedule: CheckpointSchedule,
    *,
    calls: int,
    env_steps: int,
    last_captured_env_steps: int | None,
) -> bool:
    """True when this policy-call boundary should capture a checkpoint.

    The first boundary of an episode (``last_captured_env_steps is None``) is
    always due: GoEx needs a true branch even when one plan spans the rest of
    the episode. ``per_policy_call`` ignores ``n``. Env-step cadence uses a
    delta so a plan that jumps past the boundary still captures.
    """

    if last_captured_env_steps is None:
        return True
    if schedule.mode == "per_policy_call":
        return True
    if schedule.mode == "every_n_policy_calls":
        n = schedule.n
        if n is None:
            raise RuntimeError("checkpoint_schedule requires n")
        return calls > 0 and calls % n == 0
    if schedule.mode == "every_n_env_steps":
        n = schedule.n
        if n is None:
            raise RuntimeError("checkpoint_schedule requires n")
        return (env_steps - last_captured_env_steps) >= n
    raise RuntimeError(f"unsupported_checkpoint_schedule:{schedule.mode}")


def apply_checkpoint_keep_last(
    platform: Any,
    schedule: CheckpointSchedule,
    *,
    retain_checkpoint_id: str | None = None,
) -> None:
    """Drop oldest restore-eligible checkpoints sharing this prefix.

    Terminal snapshots (``restore_eligible: false``) are evidence and are never
    ring-evicted. ``retain_checkpoint_id`` (the pin's resume parent) is never
    dropped even when it shares the prefix.
    """

    if schedule.keep_last is None:
        return
    prefix = schedule.checkpoint_id_prefix
    eligible_ids = [
        checkpoint_id
        for checkpoint_id, record in platform.checkpoints.items()
        if _shares_prefix(str(checkpoint_id), prefix)
        and (not isinstance(record, dict) or record.get("restore_eligible") is not False)
    ]
    overflow = len(eligible_ids) - schedule.keep_last
    if overflow <= 0:
        return
    dropped = 0
    for checkpoint_id in eligible_ids:
        if dropped >= overflow:
            break
        if retain_checkpoint_id and checkpoint_id == retain_checkpoint_id:
            continue
        platform.drop_checkpoint(checkpoint_id)
        dropped += 1


def _shares_prefix(checkpoint_id: str, prefix: str) -> bool:
    return checkpoint_id == prefix or checkpoint_id.startswith(f"{prefix}_")


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"checkpoint_schedule requires {field}")
    return value
