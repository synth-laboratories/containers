"""Shared structural decoding for native Harbor trial results.

This reader does not assign benchmark validity, harvest exceptions, certify a
trusted scorer, or seal traces. Consumers retain those scientific policies. The
staged accessors preserve scorer-failure precedence when a verifier did not write
its independent result. Missing rewards are errors, never implicit zero scores.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

MAX_RESULT_BYTES = 16 * 1024 * 1024


class HarborResultError(ValueError):
    """A result is unavailable, exceeds the read bound, or violates its schema."""


def read_result_object(path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise HarborResultError(f"{label} is missing")
    try:
        with path.open("rb") as handle:
            encoded = handle.read(MAX_RESULT_BYTES + 1)
        if len(encoded) > MAX_RESULT_BYTES:
            raise HarborResultError(f"{label} exceeds {MAX_RESULT_BYTES} bytes")
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise HarborResultError(f"{label} is not valid JSON ({type(error).__name__})") from error
    if not isinstance(value, dict):
        raise HarborResultError(f"{label} must be a JSON object")
    return value


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarborResultError(f"{field} must be numeric")
    try:
        parsed = float(value)
    except OverflowError as error:
        raise HarborResultError(f"{field} must be finite") from error
    if not math.isfinite(parsed):
        raise HarborResultError(f"{field} must be finite")
    return parsed


@dataclass(frozen=True)
class HarborTrialRecord:
    """A decoded result with explicit structural-validation stages."""

    payload: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> HarborTrialRecord:
        if not isinstance(payload, Mapping):
            raise HarborResultError("native Harbor trial result must be a JSON object")
        for field in ("task_name", "trial_name"):
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                raise HarborResultError(
                    f"native Harbor trial result.{field} must be a non-empty string"
                )
        if "exception_info" not in payload:
            raise HarborResultError("native Harbor trial result.exception_info is missing")
        return cls(payload=dict(payload))

    def exception_info(self) -> dict[str, Any] | None:
        value = self.payload["exception_info"]
        if value is not None and not isinstance(value, dict):
            raise HarborResultError(
                "native Harbor trial result.exception_info must be an object or null"
            )
        return dict(value) if value is not None else None

    def agent_result(self) -> dict[str, Any]:
        value = self.payload.get("agent_result")
        if not isinstance(value, dict):
            raise HarborResultError("native Harbor trial result.agent_result must be an object")
        return dict(value)

    def require_reward(self) -> float:
        verifier = self.payload.get("verifier_result")
        if not isinstance(verifier, dict):
            raise HarborResultError("native Harbor trial result.verifier_result must be an object")
        rewards = verifier.get("rewards")
        if not isinstance(rewards, dict):
            raise HarborResultError(
                "native Harbor trial result.verifier_result.rewards must be an object"
            )
        if "reward" not in rewards:
            raise HarborResultError(
                "native Harbor trial result.verifier_result.rewards.reward is missing"
            )
        return finite_number(rewards["reward"], field="verifier_result.rewards.reward")
