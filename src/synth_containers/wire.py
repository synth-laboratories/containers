from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class RolloutState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubmissionMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


@dataclass(frozen=True, slots=True)
class RequestValidationIssue:
    code: str
    message: str
    fields: tuple[str, ...] = field(default_factory=tuple)


_ALLOWED_TRANSITIONS: dict[RolloutState, set[RolloutState]] = {
    RolloutState.QUEUED: {RolloutState.RUNNING, RolloutState.PAUSED, RolloutState.CANCELLED},
    RolloutState.RUNNING: {RolloutState.PAUSED, RolloutState.COMPLETED, RolloutState.FAILED, RolloutState.CANCELLED},
    RolloutState.PAUSED: {RolloutState.RUNNING, RolloutState.COMPLETED, RolloutState.FAILED, RolloutState.CANCELLED},
    RolloutState.COMPLETED: set(),
    RolloutState.FAILED: set(),
    RolloutState.CANCELLED: set(),
}

_SUCCESS_STATUS_BY_STATE: dict[RolloutState, str] = {
    RolloutState.QUEUED: "pending",
    RolloutState.RUNNING: "running",
    RolloutState.PAUSED: "paused",
    RolloutState.COMPLETED: "success",
    RolloutState.FAILED: "failed",
    RolloutState.CANCELLED: "cancelled",
}


def state_from_status(raw_status: Any, *, default: RolloutState = RolloutState.COMPLETED) -> RolloutState:
    text = str(raw_status or "").strip().lower()
    if text in {"queued", "pending"}:
        return RolloutState.QUEUED
    if text in {"running", "in_progress"}:
        return RolloutState.RUNNING
    if text in {"paused", "pause"}:
        return RolloutState.PAUSED
    if text in {"failed", "error"}:
        return RolloutState.FAILED
    if text in {"cancelled", "canceled", "terminated"}:
        return RolloutState.CANCELLED
    if text in {"completed", "success", "succeeded", "ok"}:
        return RolloutState.COMPLETED
    return default


def assert_valid_transition(current: RolloutState, target: RolloutState) -> None:
    if current == target:
        return
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if target in allowed:
        return
    allowed_text = ", ".join(sorted(item.value for item in allowed)) or "<none>"
    raise ValueError(f"invalid_rollout_state_transition:{current.value}->{target.value} (allowed={allowed_text})")


def lifecycle_projection(state: RolloutState) -> dict[str, str]:
    return {
        "status": state.value,
        "success_status": _SUCCESS_STATUS_BY_STATE[state],
    }


def project_lifecycle_fields(payload: dict[str, Any], state: RolloutState) -> dict[str, Any]:
    projected = deepcopy(payload)
    projected.update(lifecycle_projection(state))
    return projected


def validate_submission_mode_request(request_payload: dict[str, Any]) -> RequestValidationIssue | None:
    fields: list[str] = []
    if "async_submit" in request_payload:
        fields.append("async_submit")
    if "async" in request_payload:
        fields.append("async")
    metadata = request_payload.get("metadata")
    if isinstance(metadata, dict):
        if "async_submit" in metadata:
            fields.append("metadata.async_submit")
        if "async" in metadata:
            fields.append("metadata.async")
    if fields:
        return RequestValidationIssue(
            code="legacy_async_flags_not_supported",
            message="Use submission_mode='sync|async'; legacy async flags are not supported.",
            fields=tuple(fields),
        )

    mode_raw = request_payload.get("submission_mode")
    if mode_raw is None:
        return None
    normalized_mode = str(mode_raw).strip().lower()
    if normalized_mode not in {item.value for item in SubmissionMode}:
        return RequestValidationIssue(
            code="invalid_submission_mode",
            message="submission_mode must be one of: sync, async.",
            fields=("submission_mode",),
        )
    return None


def resolve_submission_mode(request_payload: dict[str, Any]) -> SubmissionMode:
    mode_raw = request_payload.get("submission_mode")
    if mode_raw is None:
        return SubmissionMode.SYNC
    return SubmissionMode(str(mode_raw).strip().lower())
