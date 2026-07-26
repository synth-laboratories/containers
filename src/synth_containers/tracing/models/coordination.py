"""First-class multi-agent coordination records for Trace V5.

Messages and events remain the execution facts. This graph records the typed
relationships between those facts: teams, information-flow and control edges,
model-visible context epochs, and shared-environment turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Optional

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .identity import AliasV1


COORDINATION_SCHEMA_VERSION = "synth.trace-coordination.v1"


class ActorGroupKind(StrEnum):
    TEAM = "team"
    SWARM = "swarm"
    PARTY = "party"
    WORKFLOW = "workflow"
    COHORT = "cohort"


class AnchorBasis(StrEnum):
    CANONICAL = "canonical"
    NATIVE_ALIAS = "native_alias"
    RAW_SOURCE = "raw_source"


class InteractionKind(StrEnum):
    SPAWN_AGENT = "spawn_agent"
    ASSIGN_TASK = "assign_task"
    SEND_MESSAGE = "send_message"
    AGENT_RESULT = "agent_result"
    WAIT_AGENT = "wait_agent"
    CLOSE_AGENT = "close_agent"
    REVIEW_REQUEST = "review_request"
    REVIEW_RESULT = "review_result"
    DEPENDENCY = "dependency"
    CONTEXT_TRANSFER = "context_transfer"
    SHARED_STATE_READ = "shared_state_read"
    SHARED_STATE_WRITE = "shared_state_write"


class InteractionStatus(StrEnum):
    CREATED = "created"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    IN_DOUBT = "in_doubt"


class CoordinationEvidenceBasis(StrEnum):
    OBSERVED = "observed"
    DECLARED = "declared"
    DERIVED = "derived"


class ParticipantState(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DONE = "done"


ACTOR_GROUP_DECLARED_EVENT = "coordination.actor_group.declared"
CONTEXT_EPOCH_OBSERVED_EVENT = "coordination.context_epoch.observed"
JOINT_TURN_OBSERVED_EVENT = "coordination.joint_turn.observed"

_EVENT_BY_INTERACTION_KIND = {
    InteractionKind.SPAWN_AGENT: "coordination.agent.spawned",
    InteractionKind.ASSIGN_TASK: "coordination.task.assigned",
    InteractionKind.SEND_MESSAGE: "coordination.message.sent",
    InteractionKind.AGENT_RESULT: "coordination.agent.result",
    InteractionKind.WAIT_AGENT: "coordination.agent.waited",
    InteractionKind.CLOSE_AGENT: "coordination.agent.closed",
    InteractionKind.REVIEW_REQUEST: "coordination.review.requested",
    InteractionKind.REVIEW_RESULT: "coordination.review.completed",
    InteractionKind.DEPENDENCY: "coordination.dependency.observed",
    InteractionKind.CONTEXT_TRANSFER: "coordination.context.transferred",
    InteractionKind.SHARED_STATE_READ: "coordination.shared_state.read",
    InteractionKind.SHARED_STATE_WRITE: "coordination.shared_state.written",
}
INTERACTION_KIND_BY_EVENT = {
    event_type: interaction_kind
    for interaction_kind, event_type in _EVENT_BY_INTERACTION_KIND.items()
}
COORDINATION_EVENT_TYPES = frozenset(
    {
        ACTOR_GROUP_DECLARED_EVENT,
        CONTEXT_EPOCH_OBSERVED_EVENT,
        JOINT_TURN_OBSERVED_EVENT,
        *INTERACTION_KIND_BY_EVENT,
    }
)


def coordination_event_type(kind: InteractionKind | str) -> str:
    """Return the one canonical application-event name for an interaction kind."""

    try:
        normalized = InteractionKind(str(kind))
    except ValueError as exc:
        raise ValueError(f"unsupported coordination interaction kind: {kind!r}") from exc
    return _EVENT_BY_INTERACTION_KIND[normalized]


@dataclass(frozen=True, slots=True)
class TraceAnchorV1(JsonDataclassMixin):
    """One endpoint resolved by canonical identity, native alias, or raw evidence."""

    basis: AnchorBasis | str
    entity_kind: Optional[str] = None
    entity_id: Optional[str] = None
    alias_namespace: Optional[str] = None
    alias_value: Optional[str] = None
    raw_source_ref: Optional[str] = None

    def __post_init__(self) -> None:
        basis = str(self.basis)
        if basis == str(AnchorBasis.CANONICAL):
            if not self.entity_kind or not self.entity_id:
                raise ValueError("canonical trace anchor requires entity_kind and entity_id")
            if self.alias_namespace or self.alias_value or self.raw_source_ref:
                raise ValueError("canonical trace anchor cannot carry native or raw identity")
            return
        if basis == str(AnchorBasis.NATIVE_ALIAS):
            if not self.alias_namespace or not self.alias_value:
                raise ValueError("native trace anchor requires alias_namespace and alias_value")
            if self.entity_id or self.raw_source_ref:
                raise ValueError("native trace anchor cannot carry canonical or raw identity")
            return
        if basis == str(AnchorBasis.RAW_SOURCE):
            if not self.raw_source_ref:
                raise ValueError("raw trace anchor requires raw_source_ref")
            if self.entity_id or self.alias_namespace or self.alias_value:
                raise ValueError("raw trace anchor cannot carry canonical or native identity")
            return
        raise ValueError(f"unsupported trace anchor basis: {self.basis!r}")

    @classmethod
    def canonical(cls, entity_kind: str, entity_id: str) -> "TraceAnchorV1":
        return cls(
            basis=AnchorBasis.CANONICAL,
            entity_kind=entity_kind,
            entity_id=entity_id,
        )

    @classmethod
    def native_alias(
        cls,
        alias_namespace: str,
        alias_value: str,
        *,
        entity_kind: Optional[str] = None,
    ) -> "TraceAnchorV1":
        return cls(
            basis=AnchorBasis.NATIVE_ALIAS,
            entity_kind=entity_kind,
            alias_namespace=alias_namespace,
            alias_value=alias_value,
        )

    @classmethod
    def raw(
        cls,
        raw_source_ref: str,
        *,
        entity_kind: Optional[str] = None,
    ) -> "TraceAnchorV1":
        return cls(
            basis=AnchorBasis.RAW_SOURCE,
            entity_kind=entity_kind,
            raw_source_ref=raw_source_ref,
        )


@dataclass(frozen=True, slots=True)
class ActorGroupV1(JsonDataclassMixin):
    group_id: str
    kind: ActorGroupKind | str
    display_name: str
    member_actor_ids: tuple[str, ...]
    leader_actor_ids: tuple[str, ...] = ()
    environment_actor_id: Optional[str] = None
    parent_group_id: Optional[str] = None
    purpose: Optional[str] = None
    formed_sequence: Optional[int] = None
    dissolved_sequence: Optional[int] = None
    formed_at: Optional[str] = None
    dissolved_at: Optional[str] = None
    aliases: tuple[AliasV1, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "ActorGroupV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class InteractionEdgeV1(JsonDataclassMixin):
    """A typed transport or control edge; carried semantic records remain separate."""

    interaction_id: str
    kind: InteractionKind | str
    source: TraceAnchorV1
    target: TraceAnchorV1
    started_sequence: int
    started_at: str
    status: InteractionStatus | str
    ended_sequence: Optional[int] = None
    ended_at: Optional[str] = None
    correlation_id: Optional[str] = None
    transport: Optional[str] = None
    carried_message_ids: tuple[str, ...] = ()
    carried_artifact_ids: tuple[str, ...] = ()
    carried_event_ids: tuple[str, ...] = ()
    carried_raw_refs: tuple[str, ...] = ()
    delivery_receipt_ids: tuple[str, ...] = ()
    evidence_basis: CoordinationEvidenceBasis | str = CoordinationEvidenceBasis.OBSERVED
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "InteractionEdgeV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class ContextEpochV1(JsonDataclassMixin):
    """Exact model-visible context kept distinct from runtime evidence."""

    context_epoch_id: str
    actor_id: str
    session_id: str
    started_sequence: int
    started_at: str
    model_visible_message_ids: tuple[str, ...]
    model_call_span_ids: tuple[str, ...] = ()
    runtime_evidence_event_ids: tuple[str, ...] = ()
    parent_context_epoch_id: Optional[str] = None
    transfer_interaction_id: Optional[str] = None
    context_digest: Optional[str] = None
    ended_sequence: Optional[int] = None
    ended_at: Optional[str] = None
    evidence_basis: CoordinationEvidenceBasis | str = CoordinationEvidenceBasis.OBSERVED
    losses: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "ContextEpochV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class JointTurnParticipantV1(JsonDataclassMixin):
    actor_id: str
    session_id: str
    state: ParticipantState | str
    action_event_ids: tuple[str, ...] = ()
    observation_event_ids: tuple[str, ...] = ()
    message_interaction_ids: tuple[str, ...] = ()
    reward_event_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JointTurnV1(JsonDataclassMixin):
    """One shared environment step whose participant facts remain canonical events."""

    joint_turn_id: str
    environment_actor_id: str
    environment_step: int
    started_sequence: int
    ended_sequence: int
    started_at: str
    ended_at: str
    participants: tuple[JointTurnParticipantV1, ...]
    actor_group_id: Optional[str] = None
    environment_session_id: Optional[str] = None
    shared_transition_event_ids: tuple[str, ...] = ()
    shared_reward_event_ids: tuple[str, ...] = ()
    status: InteractionStatus | str = InteractionStatus.COMPLETED
    evidence_basis: CoordinationEvidenceBasis | str = CoordinationEvidenceBasis.OBSERVED
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "JointTurnV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class CoordinationGraphV1(JsonDataclassMixin):
    actor_groups: tuple[ActorGroupV1, ...] = ()
    interaction_edges: tuple[InteractionEdgeV1, ...] = ()
    context_epochs: tuple[ContextEpochV1, ...] = ()
    joint_turns: tuple[JointTurnV1, ...] = ()
    schema_version: str = COORDINATION_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "CoordinationGraphV1":
        return seal_record(self)

    def interaction(self, interaction_id: str) -> Optional[InteractionEdgeV1]:
        return next(
            (
                item
                for item in self.interaction_edges
                if item.interaction_id == interaction_id
            ),
            None,
        )


__all__ = [
    "ACTOR_GROUP_DECLARED_EVENT",
    "CONTEXT_EPOCH_OBSERVED_EVENT",
    "COORDINATION_EVENT_TYPES",
    "COORDINATION_SCHEMA_VERSION",
    "INTERACTION_KIND_BY_EVENT",
    "JOINT_TURN_OBSERVED_EVENT",
    "ActorGroupKind",
    "ActorGroupV1",
    "AnchorBasis",
    "ContextEpochV1",
    "CoordinationEvidenceBasis",
    "CoordinationGraphV1",
    "InteractionEdgeV1",
    "InteractionKind",
    "InteractionStatus",
    "JointTurnParticipantV1",
    "JointTurnV1",
    "ParticipantState",
    "TraceAnchorV1",
    "coordination_event_type",
]
