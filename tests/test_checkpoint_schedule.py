"""Configured gold checkpoint cadence and restore-eligible ring buffer."""

from __future__ import annotations

from typing import Any

import pytest

from synth_containers.checkpoint_schedule import (
    apply_checkpoint_keep_last,
    checkpoint_is_due,
    parse_checkpoint_schedule,
)

PREFIX = "luna_med_lantern_s11"


def _schedule(mode: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"mode": mode, "checkpoint_id_prefix": PREFIX}
    payload.update(extra)
    return payload


def test_parse_per_policy_call_ok() -> None:
    parsed = parse_checkpoint_schedule(_schedule("per_policy_call", n=0, keep_last=3))
    assert parsed.mode == "per_policy_call"
    assert parsed.checkpoint_id_prefix == PREFIX
    assert parsed.n is None
    assert parsed.keep_last == 3


def test_parse_unknown_mode_raises() -> None:
    with pytest.raises(RuntimeError, match="unsupported_checkpoint_schedule:per_llm_call"):
        parse_checkpoint_schedule(_schedule("per_llm_call"))
    with pytest.raises(RuntimeError, match="unsupported_checkpoint_schedule:missing"):
        parse_checkpoint_schedule({"checkpoint_id_prefix": PREFIX})


def test_parse_every_n_policy_calls_requires_n() -> None:
    with pytest.raises(RuntimeError, match="checkpoint_schedule requires n"):
        parse_checkpoint_schedule(_schedule("every_n_policy_calls"))


def test_parse_n_zero_raises() -> None:
    with pytest.raises(RuntimeError, match="checkpoint_schedule requires n"):
        parse_checkpoint_schedule(_schedule("every_n_env_steps", n=0))


def test_parse_keep_last_zero_raises() -> None:
    with pytest.raises(RuntimeError, match="checkpoint_schedule requires keep_last"):
        parse_checkpoint_schedule(_schedule("per_policy_call", keep_last=0))


def test_due_per_policy_call_always() -> None:
    schedule = parse_checkpoint_schedule(_schedule("per_policy_call"))
    assert checkpoint_is_due(
        schedule, calls=0, env_steps=0, last_captured_env_steps=None
    )
    assert checkpoint_is_due(schedule, calls=1, env_steps=3, last_captured_env_steps=0)
    assert checkpoint_is_due(schedule, calls=7, env_steps=20, last_captured_env_steps=18)


def test_due_every_n_policy_calls() -> None:
    schedule = parse_checkpoint_schedule(_schedule("every_n_policy_calls", n=4))
    assert checkpoint_is_due(
        schedule, calls=0, env_steps=0, last_captured_env_steps=None
    )
    assert not checkpoint_is_due(
        schedule, calls=1, env_steps=2, last_captured_env_steps=0
    )
    assert not checkpoint_is_due(
        schedule, calls=3, env_steps=6, last_captured_env_steps=0
    )
    assert checkpoint_is_due(schedule, calls=4, env_steps=8, last_captured_env_steps=0)
    assert not checkpoint_is_due(
        schedule, calls=5, env_steps=10, last_captured_env_steps=8
    )
    assert checkpoint_is_due(schedule, calls=8, env_steps=16, last_captured_env_steps=8)


def test_due_every_n_env_steps_jump_past_boundary() -> None:
    schedule = parse_checkpoint_schedule(_schedule("every_n_env_steps", n=4))
    assert checkpoint_is_due(
        schedule, calls=0, env_steps=0, last_captured_env_steps=None
    )
    assert not checkpoint_is_due(
        schedule, calls=1, env_steps=3, last_captured_env_steps=0
    )
    assert checkpoint_is_due(schedule, calls=1, env_steps=4, last_captured_env_steps=0)
    # A plan that jumps 0 → 5 still captures: delta, not modulo.
    assert checkpoint_is_due(schedule, calls=1, env_steps=5, last_captured_env_steps=0)
    assert not checkpoint_is_due(
        schedule, calls=2, env_steps=8, last_captured_env_steps=5
    )
    assert checkpoint_is_due(schedule, calls=2, env_steps=9, last_captured_env_steps=5)


class FakePlatform:
    def __init__(self) -> None:
        self.checkpoints: dict[str, dict[str, Any]] = {}

    def record_checkpoint(self, record: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(record["checkpoint_id"])
        stored = dict(record)
        self.checkpoints[checkpoint_id] = stored
        return stored

    def drop_checkpoint(self, checkpoint_id: str) -> None:
        self.checkpoints.pop(checkpoint_id, None)


def _eligible_ids(platform: FakePlatform) -> list[str]:
    return [
        checkpoint_id
        for checkpoint_id, record in platform.checkpoints.items()
        if record.get("restore_eligible") is not False
    ]


def test_keep_last_retains_ring_parent_and_terminal() -> None:
    schedule = parse_checkpoint_schedule(_schedule("per_policy_call", keep_last=3))
    platform = FakePlatform()
    parent_id = f"{PREFIX}_parent"
    platform.record_checkpoint({"checkpoint_id": parent_id, "restore_eligible": True})
    apply_checkpoint_keep_last(platform, schedule, retain_checkpoint_id=parent_id)

    for index in range(6):
        platform.record_checkpoint(
            {"checkpoint_id": f"{PREFIX}_{index:04d}", "restore_eligible": True}
        )
        apply_checkpoint_keep_last(platform, schedule, retain_checkpoint_id=parent_id)

    terminal_id = f"{PREFIX}_terminal"
    platform.record_checkpoint({"checkpoint_id": terminal_id, "restore_eligible": False})
    apply_checkpoint_keep_last(platform, schedule, retain_checkpoint_id=parent_id)

    eligible = _eligible_ids(platform)
    assert len(eligible) == 3
    assert parent_id in eligible
    assert parent_id in platform.checkpoints
    assert terminal_id in platform.checkpoints
    assert platform.checkpoints[terminal_id]["restore_eligible"] is False
    assert f"{PREFIX}_0000" not in platform.checkpoints
    assert f"{PREFIX}_0001" not in platform.checkpoints
    assert f"{PREFIX}_0002" not in platform.checkpoints
    assert f"{PREFIX}_0005" in platform.checkpoints
