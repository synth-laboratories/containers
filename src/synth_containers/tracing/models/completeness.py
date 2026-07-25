"""Lifecycle and completeness: what was captured, what was missed, and why."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import DIGEST_ALGORITHM
from .actors import CoverageState


class TraceStatus(StrEnum):
    LIVE = "live"
    SEALING = "sealing"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CaptureStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    TRUNCATED = "truncated"
    STREAMING = "streaming"
    FAILED_FINALIZATION = "failed_finalization"


class ValidityState(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class TerminationV5(JsonDataclassMixin):
    reason: str
    exit_code: int | None = None
    signal: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TraceLifecycleV5(JsonDataclassMixin):
    status: TraceStatus | str
    started_at: str
    ended_at: str | None = None
    termination: TerminationV5 | None = None


@dataclass(frozen=True, slots=True)
class TraceCompletenessV5(JsonDataclassMixin):
    capture_status: CaptureStatus | str
    terminal_event_observed: bool
    model_calls: CoverageState | str = CoverageState.NOT_CAPTURED
    raw_provider: CoverageState | str = CoverageState.NOT_CAPTURED
    agent_events: CoverageState | str = CoverageState.NOT_CAPTURED
    environment_events: CoverageState | str = CoverageState.NOT_CAPTURED
    tool_events: CoverageState | str = CoverageState.NOT_CAPTURED
    usage: CoverageState | str = CoverageState.NOT_CAPTURED
    expected_record_count: int | None = None
    captured_record_count: int | None = None
    high_water_ordinal: int | None = None
    missing_ranges: tuple[str, ...] = ()
    truncation_reasons: tuple[str, ...] = ()
    artifact_finalization: str = "complete"
    repair_receipt_ids: tuple[str, ...] = ()
    validator_result_id: str | None = None
    digest_algorithm: str = DIGEST_ALGORITHM
    reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CaptureStatus",
    "TerminationV5",
    "TraceCompletenessV5",
    "TraceLifecycleV5",
    "TraceStatus",
    "ValidityState",
]
