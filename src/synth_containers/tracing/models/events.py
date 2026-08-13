"""Events: immutable occurrence facts with chronological and structural ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .identity import AliasV1


class EventType(StrEnum):
    """Event types Push 1 emits. Unknown producer types are preserved verbatim."""

    ACTOR_STARTED = "actor.started"
    ACTOR_FINISHED = "actor.finished"
    SESSION_STARTED = "session.started"
    SESSION_FINISHED = "session.finished"
    MODEL_CALL_STARTED = "model_call.started"
    MODEL_CALL_FINISHED = "model_call.finished"
    UPSTREAM_ATTEMPT_STARTED = "upstream_attempt.started"
    UPSTREAM_ATTEMPT_FINISHED = "upstream_attempt.finished"
    PROVIDER_USAGE = "provider.usage"
    TOOL_CALL_PROPOSED = "tool.call_proposed"
    TOOL_CALL_EXECUTED = "tool.call_executed"
    TOOL_RESULT = "tool.result"
    ENV_OBSERVATION = "environment.observation"
    ENV_ACTION_PROPOSED = "environment.action_proposed"
    ENV_ACTION_EXECUTED = "environment.action_executed"
    ENV_TRANSITION = "environment.transition"
    ENV_REWARD = "environment.reward"
    ENV_TERMINAL = "environment.terminal"
    APPLICATION = "application.event"
    ARTIFACT_PRODUCED = "artifact.produced"
    CONTEXT_COMPACTION = "context.compaction"
    CAPTURE_LIFECYCLE = "capture.lifecycle"
    ERROR = "error"


class EventStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    REJECTED = "rejected"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class StructuralAddressV1(JsonDataclassMixin):
    """Workflow position, which is replay identity for deterministic producers."""

    workflow_id: str
    node_path: tuple[str, ...] = ()
    iteration: int = 0
    local_sequence: int = 0


@dataclass(frozen=True, slots=True)
class EventOrderV1(JsonDataclassMixin):
    chronological_sequence: int | None = None
    actor_sequence: int | None = None
    source_order_id: str | None = None
    structural: StructuralAddressV1 | None = None


@dataclass(frozen=True, slots=True)
class EventV5(JsonDataclassMixin):
    event_id: str
    event_type: EventType | str
    actor_id: str
    session_id: str
    occurred_at: str
    span_id: str | None = None
    turn_id: str | None = None
    message_id: str | None = None
    order: EventOrderV1 = field(default_factory=EventOrderV1)
    caused_by_event_ids: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    status: EventStatus | str = EventStatus.OK
    error_id: str | None = None
    raw_source_ref: str | None = None
    artifact_ids: tuple[str, ...] = ()
    aliases: tuple[AliasV1, ...] = ()
    content_digest: str = ""

    def sealed(self) -> "EventV5":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class TraceErrorV1(JsonDataclassMixin):
    error_id: str
    stage: str
    component: str
    code: str
    message: str
    retryable: bool = False
    retry_count: int = 0
    partial_success: bool = False
    terminal: bool = False
    actor_id: str | None = None
    session_id: str | None = None
    span_id: str | None = None
    event_id: str | None = None
    artifact_id: str | None = None
    caused_by_error_id: str | None = None
    observed_at: str | None = None
    resolved_at: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EventOrderV1",
    "EventStatus",
    "EventType",
    "EventV5",
    "StructuralAddressV1",
    "TraceErrorV1",
]
