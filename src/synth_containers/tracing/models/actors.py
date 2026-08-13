"""Actors and sessions: who acted, under which attempt, with which capture coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .identity import AliasV1


class ActorKind(StrEnum):
    AGENT = "agent"
    HUMAN = "human"
    ENVIRONMENT = "environment"
    TOOL = "tool"
    VERIFIER = "verifier"
    ORCHESTRATOR = "orchestrator"
    EVALUATOR = "evaluator"


class Visibility(StrEnum):
    """Classification ceiling for a record and everything it contains."""

    PUBLIC = "public"
    OPERATOR = "operator"
    PRIVATE = "private"


class SessionStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class CoverageState(StrEnum):
    """How much of one capture surface a session actually observed."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    AGGREGATE_ONLY = "aggregate_only"
    UNAVAILABLE = "unavailable"
    NOT_CAPTURED = "not_captured"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ActorV5(JsonDataclassMixin):
    actor_id: str
    kind: ActorKind | str
    display_name: str
    role: str = ""
    subtype: str = ""
    parent_actor_id: str | None = None
    actor_path: Optional[str] = None
    origin_interaction_id: Optional[str] = None
    harness: str | None = None
    runtime: str | None = None
    model: str | None = None
    provider: str | None = None
    policy_id: str | None = None
    task_id: str | None = None
    workflow_id: str | None = None
    external_trace_refs: tuple[str, ...] = ()
    visibility: Visibility | str = Visibility.PRIVATE
    aliases: tuple[AliasV1, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "ActorV5":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class SessionCoverageV5(JsonDataclassMixin):
    """Per-surface capture coverage for one session; never inferred, always declared."""

    model_calls: CoverageState | str = CoverageState.NOT_CAPTURED
    agent_events: CoverageState | str = CoverageState.NOT_CAPTURED
    environment_events: CoverageState | str = CoverageState.NOT_CAPTURED
    tool_events: CoverageState | str = CoverageState.NOT_CAPTURED
    usage: CoverageState | str = CoverageState.NOT_CAPTURED
    raw_provider: CoverageState | str = CoverageState.NOT_CAPTURED
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionV5(JsonDataclassMixin):
    session_id: str
    actor_id: str
    started_at: str
    attempt_id: str | None = None
    thread_id: str | None = None
    workflow_id: str | None = None
    capture_id: str | None = None
    parent_session_id: str | None = None
    branch_head_id: str | None = None
    started_sequence: Optional[int] = None
    ended_sequence: Optional[int] = None
    status: SessionStatus | str = SessionStatus.RUNNING
    ended_at: str | None = None
    harness: str | None = None
    provider: str | None = None
    coverage: SessionCoverageV5 = field(default_factory=SessionCoverageV5)
    aliases: tuple[AliasV1, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "SessionV5":
        return seal_record(self)


__all__ = [
    "ActorKind",
    "ActorV5",
    "CoverageState",
    "SessionCoverageV5",
    "SessionStatus",
    "SessionV5",
    "Visibility",
]
