"""Headless visual reducer. Missing reward/usage/score stay null, never 0."""

from __future__ import annotations

from typing import Any

FORBIDDEN_BLOBS = ("collector", "capability_blob", "capabilities_blob")


def project_envelopes(
    envelopes: list[dict[str, Any]],
    *,
    cutoff_sequence: int | None = None,
    usage: dict[str, Any] | None = None,
    reward: float | None = None,
    reward_status: str = "absent",
) -> dict[str, Any]:
    rows = []
    for item in envelopes:
        if item.get("control"):
            continue
        seq = item.get("sequence")
        if cutoff_sequence is not None and seq is not None and int(seq) > cutoff_sequence:
            continue
        rows.append(item)
    kinds = [str(item.get("kind")) for item in rows]
    has_frames = "frame" in kinds
    has_reward_txt = any(
        "reward.txt" in json_keys(item.get("payload")) for item in rows
    )
    return {
        "events": rows,
        "kinds": kinds,
        "has_live_frames": has_frames,
        "has_reward_txt": has_reward_txt,
        "reward": reward,
        "reward_status": reward_status,
        "usage": usage,
        "cutoff_sequence": cutoff_sequence,
    }


def json_keys(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    keys: set[str] = {str(key) for key in payload}
    for value in payload.values():
        keys |= json_keys(value)
    return keys


def assert_honest_projection(projection: dict[str, Any]) -> list[str]:
    """Return defect strings. Missing must not become 0."""
    defects: list[str] = []
    if projection.get("reward_status") in {"absent", "gated", "refused"} and projection.get("reward") == 0:
        defects.append("missing_reward_coerced_to_zero")
    usage = projection.get("usage")
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if key in usage and usage[key] == 0 and usage.get(f"{key}_missing"):
                defects.append(f"missing_usage_{key}_coerced_to_zero")
    blob = json_dumps_safe(projection)
    for name in FORBIDDEN_BLOBS:
        if name in blob:
            defects.append(f"forbidden_blob:{name}")
    return defects


def json_dumps_safe(value: Any) -> str:
    import json

    return json.dumps(value, default=str)
