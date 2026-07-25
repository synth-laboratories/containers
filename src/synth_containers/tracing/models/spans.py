"""Spans: operations, containment, duration, status, and process topology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .identity import AliasV1


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
    total_tokens: int | None = None
    requests: int | None = None
    wall_time_seconds: float | None = None
    unavailable_fields: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def merged(self, other: "UsageV5") -> "UsageV5":
        """Sum two observed usages; any unavailable component keeps the result partial."""

        def add(left: int | None, right: int | None) -> int | None:
            if left is None and right is None:
                return None
            return int(left or 0) + int(right or 0)

        provenance = UsageProvenance.DERIVED
        if UsageProvenance.UNAVAILABLE in {self.provenance, other.provenance}:
            provenance = UsageProvenance.PARTIAL
        return UsageV5(
            provenance=provenance,
            prompt_tokens=add(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=add(self.completion_tokens, other.completion_tokens),
            reasoning_tokens=add(self.reasoning_tokens, other.reasoning_tokens),
            cached_tokens=add(self.cached_tokens, other.cached_tokens),
            total_tokens=add(self.total_tokens, other.total_tokens),
            requests=add(self.requests, other.requests),
            wall_time_seconds=(
                None
                if self.wall_time_seconds is None and other.wall_time_seconds is None
                else float(self.wall_time_seconds or 0.0) + float(other.wall_time_seconds or 0.0)
            ),
            unavailable_fields=tuple(
                sorted(set(self.unavailable_fields) | set(other.unavailable_fields))
            ),
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
    caused_by_span_ids: tuple[str, ...] = ()
    ended_at: str | None = None
    status: SpanStatus | str = SpanStatus.OK
    error_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    input_message_ids: tuple[str, ...] = ()
    output_message_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    usage: UsageV5 | None = None
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
