"""Deterministically reduce raw coordination facts into the Trace V5 graph.

The reducer consumes append-only envelopes after the ordinary V5 messages, spans,
events, and artifacts exist. It never writes live state and never infers task
assignment, message delivery, or results from actor ancestry alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence

from ..canonical import content_digest, record_id
from ..models.actors import ActorV5, SessionV5
from ..models.artifacts import ArtifactRefV5
from ..models.coordination import (
    ACTOR_GROUP_DECLARED_EVENT,
    CONTEXT_EPOCH_OBSERVED_EVENT,
    COORDINATION_EVENT_TYPES,
    INTERACTION_KIND_BY_EVENT,
    JOINT_TURN_OBSERVED_EVENT,
    ActorGroupV1,
    AnchorBasis,
    ContextEpochV1,
    CoordinationEvidenceBasis,
    CoordinationGraphV1,
    InteractionEdgeV1,
    InteractionKind,
    InteractionStatus,
    JointTurnParticipantV1,
    JointTurnV1,
    TraceAnchorV1,
    coordination_event_type,
)
from ..models.events import EventV5
from ..models.messages import MessageNodeV5
from ..models.spans import SpanV5
from .envelope import RawRecordType


@dataclass(frozen=True, slots=True)
class CoordinationReductionV1:
    graph: Optional[CoordinationGraphV1]
    actors: tuple[ActorV5, ...]
    sessions: tuple[SessionV5, ...]
    spans: tuple[SpanV5, ...]


def reduce_coordination(
    *,
    trace_id: str,
    records: Sequence[Mapping[str, Any]],
    actors: Sequence[ActorV5],
    sessions: Sequence[SessionV5],
    messages: Sequence[MessageNodeV5],
    spans: Sequence[SpanV5],
    events: Sequence[EventV5],
    artifacts: Sequence[ArtifactRefV5],
) -> CoordinationReductionV1:
    """Build the optional coordination graph from observed and declared raw facts."""

    del artifacts  # Artifact references are validated after document assembly.
    groups: list[ActorGroupV1] = []
    explicit_interactions: list[InteractionEdgeV1] = []
    derived_spawns: list[InteractionEdgeV1] = []
    context_epochs: list[ContextEpochV1] = []
    joint_turns: list[JointTurnV1] = []
    sequence_by_session: dict[str, list[int]] = {}

    for record in records:
        session_id = _required_str(record, "session_id", context="raw envelope")
        ordinal = _required_int(record, "ordinal", context="raw envelope")
        sequence_by_session.setdefault(session_id, []).append(ordinal)
        record_type = str(record.get("record_type") or "")
        if record_type == str(RawRecordType.CHILD_REGISTERED):
            derived_spawns.append(_spawn_from_child_registration(trace_id, record))
            continue
        if record_type != str(RawRecordType.APPLICATION_EVENT):
            continue
        payload = _required_mapping(record, "payload", context="application envelope")
        event_type = _required_str(payload, "event_type", context="application payload")
        if event_type not in COORDINATION_EVENT_TYPES:
            continue
        body = _required_mapping(payload, "body", context=event_type)
        if event_type == ACTOR_GROUP_DECLARED_EVENT:
            groups.append(_actor_group(record, body))
        elif event_type == CONTEXT_EPOCH_OBSERVED_EVENT:
            context_epochs.append(_context_epoch(trace_id, record, body))
        elif event_type == JOINT_TURN_OBSERVED_EVENT:
            joint_turns.append(_joint_turn(trace_id, record, body))
        else:
            explicit_interactions.append(
                _interaction(
                    trace_id,
                    record,
                    body,
                    kind=INTERACTION_KIND_BY_EVENT[event_type],
                )
            )

    explicit_context_span_ids = {
        span_id
        for epoch in context_epochs
        for span_id in epoch.model_call_span_ids
    }
    context_epochs.extend(
        _derived_model_context_epochs(
            trace_id=trace_id,
            spans=spans,
            messages=messages,
            events=events,
            excluded_span_ids=explicit_context_span_ids,
        )
    )
    context_epochs = [
        _with_context_digest(epoch, messages=messages)
        for epoch in context_epochs
    ]

    explicitly_spawned_actor_ids = {
        interaction.target.entity_id
        for interaction in explicit_interactions
        if interaction.kind == InteractionKind.SPAWN_AGENT
        and interaction.target.basis == AnchorBasis.CANONICAL
        and interaction.target.entity_kind == "actor"
    }
    interactions = [
        *(
            interaction
            for interaction in derived_spawns
            if interaction.target.entity_id not in explicitly_spawned_actor_ids
        ),
        *explicit_interactions,
    ]
    interactions.sort(key=lambda item: (item.started_sequence, item.interaction_id))
    groups.sort(key=lambda item: item.group_id)
    context_epochs.sort(key=lambda item: (item.started_sequence, item.context_epoch_id))
    joint_turns.sort(key=lambda item: (item.started_sequence, item.joint_turn_id))

    if not groups and not interactions and not context_epochs and not joint_turns:
        return CoordinationReductionV1(
            graph=None,
            actors=tuple(actors),
            sessions=tuple(sessions),
            spans=tuple(spans),
        )

    origin_interaction_by_actor = {
        interaction.target.entity_id: interaction.interaction_id
        for interaction in interactions
        if interaction.kind == InteractionKind.SPAWN_AGENT
        and interaction.target.basis == AnchorBasis.CANONICAL
        and interaction.target.entity_kind == "actor"
        and interaction.target.entity_id is not None
    }
    reduced_actors = tuple(
        replace(
            actor,
            origin_interaction_id=origin_interaction_by_actor.get(
                actor.actor_id,
                actor.origin_interaction_id,
            ),
            content_digest="",
        ).sealed()
        if actor.actor_id in origin_interaction_by_actor
        else actor
        for actor in actors
    )
    reduced_actors = _actors_with_paths(reduced_actors)
    reduced_sessions = tuple(
        _session_with_sequences(session, sequence_by_session.get(session.session_id))
        for session in sessions
    )
    context_by_span = {
        span_id: epoch.context_epoch_id
        for epoch in context_epochs
        for span_id in epoch.model_call_span_ids
    }
    reduced_spans = tuple(
        replace(
            span,
            context_epoch_id=context_by_span[span.span_id],
            content_digest="",
        ).sealed()
        if span.span_id in context_by_span
        else span
        for span in spans
    )
    graph = CoordinationGraphV1(
        actor_groups=tuple(groups),
        interaction_edges=tuple(interactions),
        context_epochs=tuple(context_epochs),
        joint_turns=tuple(joint_turns),
    ).sealed()
    return CoordinationReductionV1(
        graph=graph,
        actors=reduced_actors,
        sessions=reduced_sessions,
        spans=reduced_spans,
    )


def _derived_model_context_epochs(
    *,
    trace_id: str,
    spans: Sequence[SpanV5],
    messages: Sequence[MessageNodeV5],
    events: Sequence[EventV5],
    excluded_span_ids: set[str],
) -> tuple[ContextEpochV1, ...]:
    messages_by_id = {message.message_id: message for message in messages}
    events_by_span: dict[str, list[EventV5]] = {}
    for event in events:
        if event.span_id is not None:
            events_by_span.setdefault(event.span_id, []).append(event)
    epochs: list[ContextEpochV1] = []
    for span in spans:
        if str(span.span_kind) != "model_call" or span.span_id in excluded_span_ids:
            continue
        span_events = events_by_span.get(span.span_id, [])
        sequences = [
            event.order.chronological_sequence
            for event in span_events
            if event.order.chronological_sequence is not None
        ]
        if not sequences:
            continue
        missing_messages = [
            message_id
            for message_id in span.input_message_ids
            if message_id not in messages_by_id
        ]
        if missing_messages:
            raise ValueError(
                f"model call {span.span_id} has unknown input messages: "
                + ", ".join(missing_messages)
            )
        epoch_id = record_id(
            "ctx",
            kind="model_visible_context",
            scope=(trace_id,),
            key=span.span_id,
        )
        epochs.append(
            ContextEpochV1(
                context_epoch_id=epoch_id,
                actor_id=span.actor_id,
                session_id=span.session_id,
                started_sequence=min(sequences),
                started_at=span.started_at,
                ended_sequence=max(sequences),
                ended_at=span.ended_at,
                model_visible_message_ids=span.input_message_ids,
                model_call_span_ids=(span.span_id,),
                runtime_evidence_event_ids=tuple(
                    event.event_id
                    for event in sorted(
                        span_events,
                        key=lambda item: (
                            item.order.chronological_sequence
                            if item.order.chronological_sequence is not None
                            else -1
                        ),
                    )
                ),
                context_digest=content_digest(
                    tuple(
                        messages_by_id[message_id].content_digest
                        for message_id in span.input_message_ids
                    )
                ),
                evidence_basis=CoordinationEvidenceBasis.OBSERVED,
                metadata={
                    "visibility_basis": "provider_model_input",
                    "source_span_id": span.span_id,
                },
            ).sealed()
        )
    return tuple(epochs)


def _with_context_digest(
    epoch: ContextEpochV1,
    *,
    messages: Sequence[MessageNodeV5],
) -> ContextEpochV1:
    messages_by_id = {message.message_id: message for message in messages}
    missing = [
        message_id
        for message_id in epoch.model_visible_message_ids
        if message_id not in messages_by_id
    ]
    if missing:
        raise ValueError(
            f"context epoch {epoch.context_epoch_id} has unknown visible messages: "
            + ", ".join(missing)
        )
    expected_digest = content_digest(
        tuple(
            messages_by_id[message_id].content_digest
            for message_id in epoch.model_visible_message_ids
        )
    )
    if epoch.context_digest is not None:
        if epoch.context_digest != expected_digest:
            raise ValueError(
                f"context epoch {epoch.context_epoch_id} digest does not match "
                "its model-visible messages"
            )
        return epoch
    return replace(
        epoch,
        context_digest=expected_digest,
        content_digest="",
    ).sealed()


def _session_with_sequences(
    session: SessionV5,
    sequences: Optional[Sequence[int]],
) -> SessionV5:
    if not sequences:
        return session
    return replace(
        session,
        started_sequence=min(sequences),
        ended_sequence=max(sequences),
        content_digest="",
    ).sealed()


def _actors_with_paths(actors: Sequence[ActorV5]) -> tuple[ActorV5, ...]:
    children_by_parent: dict[str, list[ActorV5]] = {}
    roots: list[ActorV5] = []
    for actor in actors:
        if actor.parent_actor_id is None:
            roots.append(actor)
        else:
            children_by_parent.setdefault(actor.parent_actor_id, []).append(actor)
    assigned: dict[str, str] = {}

    def visit(actor: ActorV5, path: str) -> None:
        if actor.actor_id in assigned:
            return
        assigned[actor.actor_id] = actor.actor_path or path
        for index, child in enumerate(
            sorted(
                children_by_parent.get(actor.actor_id, ()),
                key=lambda item: item.actor_id,
            )
        ):
            visit(child, f"{assigned[actor.actor_id]}/{index}")

    for index, root in enumerate(roots):
        visit(root, "/root" if len(roots) == 1 else f"/root/{index}")
    return tuple(
        replace(
            actor,
            actor_path=assigned[actor.actor_id],
            content_digest="",
        ).sealed()
        if actor.actor_id in assigned and actor.actor_path != assigned[actor.actor_id]
        else actor
        for actor in actors
    )


def _spawn_from_child_registration(
    trace_id: str,
    record: Mapping[str, Any],
) -> InteractionEdgeV1:
    payload = _required_mapping(record, "payload", context="child.registered")
    actor_payload = _required_mapping(payload, "actor", context="child.registered")
    session_payload = _required_mapping(payload, "session", context="child.registered")
    child_actor_id = _required_str(actor_payload, "actor_id", context="child actor")
    parent_actor_id = _required_str(
        actor_payload,
        "parent_actor_id",
        context="child actor",
    )
    context_payload = payload.get("context")
    correlation_id: Optional[str] = None
    if context_payload is not None:
        if not isinstance(context_payload, Mapping):
            raise ValueError("child.registered context must be an object")
        correlation_id = _optional_str(context_payload, "delegation_id")
    envelope_id = _required_str(record, "envelope_id", context="child.registered")
    return InteractionEdgeV1(
        interaction_id=record_id(
            "ixn",
            kind="child_registration_spawn",
            scope=(trace_id,),
            key=envelope_id,
        ),
        kind=InteractionKind.SPAWN_AGENT,
        source=TraceAnchorV1.canonical("actor", parent_actor_id),
        target=TraceAnchorV1.canonical("actor", child_actor_id),
        started_sequence=_required_int(record, "ordinal", context="child.registered"),
        started_at=_required_str(record, "occurred_at", context="child.registered"),
        status=InteractionStatus.COMPLETED,
        correlation_id=correlation_id,
        carried_raw_refs=(envelope_id,),
        evidence_basis=CoordinationEvidenceBasis.DERIVED,
        metadata={
            "topology_basis": "authenticated_child_registration",
            "child_session_id": _required_str(
                session_payload,
                "session_id",
                context="child session",
            ),
        },
    ).sealed()


def _actor_group(
    record: Mapping[str, Any],
    body: Mapping[str, Any],
) -> ActorGroupV1:
    payload = _required_mapping(body, "actor_group", context=ACTOR_GROUP_DECLARED_EVENT)
    aliases = tuple(_alias(item) for item in _mapping_sequence(payload, "aliases"))
    return ActorGroupV1(
        group_id=_required_str(payload, "group_id", context="actor group"),
        kind=_required_str(payload, "kind", context="actor group"),
        display_name=_required_str(payload, "display_name", context="actor group"),
        member_actor_ids=_string_tuple(payload, "member_actor_ids", required=True),
        leader_actor_ids=_string_tuple(payload, "leader_actor_ids"),
        environment_actor_id=_optional_str(payload, "environment_actor_id"),
        parent_group_id=_optional_str(payload, "parent_group_id"),
        purpose=_optional_str(payload, "purpose"),
        formed_sequence=_required_int(
            record,
            "ordinal",
            context="actor group envelope",
        ),
        dissolved_sequence=_optional_int(payload, "dissolved_sequence"),
        formed_at=_optional_str(payload, "formed_at")
        or _required_str(
            record,
            "occurred_at",
            context="actor group envelope",
        ),
        dissolved_at=_optional_str(payload, "dissolved_at"),
        aliases=aliases,
        metadata=_metadata(payload),
    ).sealed()


def _interaction(
    trace_id: str,
    record: Mapping[str, Any],
    body: Mapping[str, Any],
    *,
    kind: InteractionKind,
) -> InteractionEdgeV1:
    payload = _required_mapping(body, "interaction", context=coordination_event_type(kind))
    envelope_id = _required_str(record, "envelope_id", context="interaction envelope")
    declared_kind = _optional_str(payload, "kind")
    if declared_kind is not None and declared_kind != str(kind):
        raise ValueError(
            f"interaction event kind {kind!r} conflicts with payload kind {declared_kind!r}"
        )
    interaction_id = _optional_str(payload, "interaction_id") or record_id(
        "ixn",
        kind=str(kind),
        scope=(trace_id,),
        key=envelope_id,
    )
    carried_raw_refs = _string_tuple(payload, "carried_raw_refs")
    if envelope_id not in carried_raw_refs:
        carried_raw_refs = (*carried_raw_refs, envelope_id)
    return InteractionEdgeV1(
        interaction_id=interaction_id,
        kind=kind,
        source=_anchor(_required_mapping(payload, "source", context="interaction")),
        target=_anchor(_required_mapping(payload, "target", context="interaction")),
        started_sequence=_required_int(record, "ordinal", context="interaction envelope"),
        started_at=_required_str(record, "occurred_at", context="interaction envelope"),
        status=_required_str(payload, "status", context="interaction"),
        ended_sequence=_optional_int(payload, "ended_sequence"),
        ended_at=_optional_str(payload, "ended_at"),
        correlation_id=_optional_str(payload, "correlation_id"),
        transport=_optional_str(payload, "transport"),
        carried_message_ids=_string_tuple(payload, "carried_message_ids"),
        carried_artifact_ids=_string_tuple(payload, "carried_artifact_ids"),
        carried_event_ids=_string_tuple(payload, "carried_event_ids"),
        carried_raw_refs=carried_raw_refs,
        delivery_receipt_ids=_string_tuple(payload, "delivery_receipt_ids"),
        evidence_basis=_required_str(payload, "evidence_basis", context="interaction"),
        metadata=_metadata(payload),
    ).sealed()


def _context_epoch(
    trace_id: str,
    record: Mapping[str, Any],
    body: Mapping[str, Any],
) -> ContextEpochV1:
    payload = _required_mapping(body, "context_epoch", context=CONTEXT_EPOCH_OBSERVED_EVENT)
    envelope_id = _required_str(record, "envelope_id", context="context epoch envelope")
    return ContextEpochV1(
        context_epoch_id=_optional_str(payload, "context_epoch_id")
        or record_id(
            "ctx",
            kind="context_epoch",
            scope=(trace_id,),
            key=envelope_id,
        ),
        actor_id=_required_str(record, "actor_id", context="context epoch envelope"),
        session_id=_required_str(record, "session_id", context="context epoch envelope"),
        started_sequence=_required_int(record, "ordinal", context="context epoch envelope"),
        started_at=_required_str(record, "occurred_at", context="context epoch envelope"),
        model_visible_message_ids=_string_tuple(
            payload,
            "model_visible_message_ids",
            required=True,
        ),
        model_call_span_ids=_string_tuple(payload, "model_call_span_ids"),
        runtime_evidence_event_ids=_string_tuple(
            payload,
            "runtime_evidence_event_ids",
        ),
        parent_context_epoch_id=_optional_str(payload, "parent_context_epoch_id"),
        transfer_interaction_id=_optional_str(payload, "transfer_interaction_id"),
        context_digest=_optional_str(payload, "context_digest"),
        ended_sequence=_optional_int(payload, "ended_sequence"),
        ended_at=_optional_str(payload, "ended_at"),
        evidence_basis=_required_str(payload, "evidence_basis", context="context epoch"),
        losses=_string_tuple(payload, "losses"),
        metadata=_metadata(payload),
    ).sealed()


def _joint_turn(
    trace_id: str,
    record: Mapping[str, Any],
    body: Mapping[str, Any],
) -> JointTurnV1:
    payload = _required_mapping(body, "joint_turn", context=JOINT_TURN_OBSERVED_EVENT)
    envelope_id = _required_str(record, "envelope_id", context="joint turn envelope")
    ended_sequence = _required_int(record, "ordinal", context="joint turn envelope")
    ended_at = _required_str(record, "occurred_at", context="joint turn envelope")
    started_sequence = _optional_int(payload, "started_sequence")
    started_at = _optional_str(payload, "started_at")
    return JointTurnV1(
        joint_turn_id=_optional_str(payload, "joint_turn_id")
        or record_id(
            "joint",
            kind="joint_turn",
            scope=(trace_id,),
            key=envelope_id,
        ),
        environment_actor_id=_required_str(
            record,
            "actor_id",
            context="joint turn envelope",
        ),
        environment_session_id=_required_str(
            record,
            "session_id",
            context="joint turn envelope",
        ),
        actor_group_id=_optional_str(payload, "actor_group_id"),
        environment_step=_required_int(payload, "environment_step", context="joint turn"),
        started_sequence=(
            started_sequence if started_sequence is not None else ended_sequence
        ),
        ended_sequence=ended_sequence,
        started_at=started_at or ended_at,
        ended_at=ended_at,
        participants=tuple(
            _joint_participant(item)
            for item in _mapping_sequence(payload, "participants", required=True)
        ),
        shared_transition_event_ids=_string_tuple(
            payload,
            "shared_transition_event_ids",
        ),
        shared_reward_event_ids=_string_tuple(payload, "shared_reward_event_ids"),
        status=_required_str(payload, "status", context="joint turn"),
        evidence_basis=_required_str(payload, "evidence_basis", context="joint turn"),
        metadata=_metadata(payload),
    ).sealed()


def _joint_participant(payload: Mapping[str, Any]) -> JointTurnParticipantV1:
    return JointTurnParticipantV1(
        actor_id=_required_str(payload, "actor_id", context="joint turn participant"),
        session_id=_required_str(payload, "session_id", context="joint turn participant"),
        state=_required_str(payload, "state", context="joint turn participant"),
        action_event_ids=_string_tuple(payload, "action_event_ids"),
        observation_event_ids=_string_tuple(payload, "observation_event_ids"),
        message_interaction_ids=_string_tuple(payload, "message_interaction_ids"),
        reward_event_ids=_string_tuple(payload, "reward_event_ids"),
        metadata=_metadata(payload),
    )


def _anchor(payload: Mapping[str, Any]) -> TraceAnchorV1:
    basis = _required_str(payload, "basis", context="trace anchor")
    return TraceAnchorV1(
        basis=basis,
        entity_kind=_optional_str(payload, "entity_kind"),
        entity_id=_optional_str(payload, "entity_id"),
        alias_namespace=_optional_str(payload, "alias_namespace"),
        alias_value=_optional_str(payload, "alias_value"),
        raw_source_ref=_optional_str(payload, "raw_source_ref"),
    )


def _alias(payload: Mapping[str, Any]) -> Any:
    from ..models.identity import AliasV1

    return AliasV1(
        namespace=_required_str(payload, "namespace", context="alias"),
        value=_required_str(payload, "value", context="alias"),
        target_id=_required_str(payload, "target_id", context="alias"),
        target_kind=_required_str(payload, "target_kind", context="alias"),
        provenance=_required_str(payload, "provenance", context="alias"),
        confidence=_required_float(payload, "confidence", context="alias"),
    )


def _metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("metadata")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object")
    return dict(value)


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} requires object field {key!r}")
    return value


def _mapping_sequence(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f"{key!r} is required")
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be an array")
    output: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{key!r} entries must be objects")
        output.append(item)
    return tuple(output)


def _string_tuple(
    payload: Mapping[str, Any],
    key: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        if required:
            raise ValueError(f"{key!r} is required")
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be an array")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{key!r} entries must be non-empty strings")
    return tuple(value)


def _required_str(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires non-empty string field {key!r}")
    return value


def _optional_str(payload: Mapping[str, Any], key: str) -> Optional[str]:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} must be a non-empty string when present")
    return value


def _required_int(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} requires integer field {key!r}")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> Optional[int]:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer when present")
    return value


def _required_float(payload: Mapping[str, Any], key: str, *, context: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{context} requires numeric field {key!r}")
    return float(value)


__all__ = [
    "CoordinationReductionV1",
    "reduce_coordination",
]
