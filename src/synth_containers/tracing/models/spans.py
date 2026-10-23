"""Spans: operations, containment, duration, status, and process topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .identity import AliasV1
from .tokens import TokenCaptureV5


class SpanKind(StrEnum):
    AGENT_SESSION = "agent_session"
    AGENT_TURN = "agent_turn"
    MODEL_CALL = "model_call"
    TOOL_EXECUTION = "tool_execution"
    ENVIRONMENT_STEP = "environment_step"
    JOINT_TURN = "joint_turn"
    WORKFLOW_NODE = "workflow_node"
    DELEGATION = "delegation"
    CONTEXT_MANAGEMENT = "context_management"
    CHECKPOINT = "checkpoint"
    VERIFIER_EXECUTION = "verifier_execution"
    EVALUATOR_EXECUTION = "evaluator_execution"
    ARTIFACT_OPERATION = "artifact_operation"
    APPLICATION = "application"


class SpanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    CANCELED = "canceled"
    TRUNCATED = "truncated"
    RUNNING = "running"


class UsageProvenance(StrEnum):
    """Where a usage number came from. ``unavailable`` is never silently zero."""

    OBSERVED_PROVIDER = "observed_provider"
    OBSERVED_HARNESS = "observed_harness"
    DERIVED = "derived"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class UsageV5(JsonDataclassMixin):
    """Token/cost accounting with explicit provenance for every observation."""

    provenance: UsageProvenance | str = UsageProvenance.UNAVAILABLE
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None
    requests: int | None = None
    wall_time_seconds: float | None = None
    cost_usd: float | None = None
    unavailable_fields: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def merged(self, other: "UsageV5") -> "UsageV5":
        """Sum disjoint observations, preserving unknowns rather than assuming zero.

        This is not a snapshot/delta reducer: callers must deduplicate observations.
        Optional fields absent on both sides stay null. A field known on only one
        side is unavailable for the aggregate and makes its provenance partial.
        Initialize reductions from their first observation, not a synthetic zero.
        """
        unavailable = set(self.unavailable_fields) | set(other.unavailable_fields)
        partial = bool(unavailable) or bool(
            {self.provenance, other.provenance}
            & {UsageProvenance.UNAVAILABLE, UsageProvenance.PARTIAL}
        )

        def add(name: str) -> Any:
            nonlocal partial
            left, right = getattr(self, name), getattr(other, name)
            if (left is None) != (right is None):
                unavailable.add(name)
                partial = True
            if name in unavailable or left is None or right is None:
                return None
            return left + right

        values = {
            name: add(name)
            for name in (
                "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens",
                "cache_write_tokens", "total_tokens", "requests", "wall_time_seconds", "cost_usd",
            )
        }
        return UsageV5(
            provenance=UsageProvenance.PARTIAL if partial else UsageProvenance.DERIVED,
            **values,
            unavailable_fields=tuple(sorted(unavailable)),
            source_refs=tuple(sorted(set(self.source_refs) | set(other.source_refs))),
        )


@dataclass(frozen=True, slots=True)
class TransformationRecordV1(JsonDataclassMixin):
    """How raw provider bytes became canonical entities, including what was lost."""

    name: str
    version: str
    config_digest: str | None = None
    input_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    losses: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    deterministic: bool = True
    reversible: bool = False


@dataclass(frozen=True, slots=True)
class SpanV5(JsonDataclassMixin):
    span_id: str
    span_kind: SpanKind | str
    actor_id: str
    session_id: str
    started_at: str
    parent_span_id: str | None = None
    turn_id: str | None = None
    branch_id: str | None = None
    workflow_address: str | None = None
    context_epoch_id: Optional[str] = None
    caused_by_span_ids: tuple[str, ...] = ()
    ended_at: str | None = None
    status: SpanStatus | str = SpanStatus.OK
    error_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    input_message_ids: tuple[str, ...] = ()
    output_message_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    usage: UsageV5 | None = None
    token_capture: TokenCaptureV5 | None = None
    transformations: tuple[TransformationRecordV1, ...] = ()
    aliases: tuple[AliasV1, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "SpanV5":
        return seal_record(self)


__all__ = [
    "SpanKind",
    "SpanStatus",
    "SpanV5",
    "TransformationRecordV1",
    "UsageProvenance",
    "UsageV5",
]
