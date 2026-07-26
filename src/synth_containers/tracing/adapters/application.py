"""Application-event adapters for ReAct, Codex, Jesterky, and delegated agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from ..models.coordination import (
    ACTOR_GROUP_DECLARED_EVENT,
    CONTEXT_EPOCH_OBSERVED_EVENT,
    JOINT_TURN_OBSERVED_EVENT,
    ActorGroupV1,
    ContextEpochV1,
    InteractionEdgeV1,
    JointTurnV1,
    coordination_event_type,
)

if TYPE_CHECKING:
    from ..capture.collector import LocalCollector


@dataclass(frozen=True, slots=True)
class ApplicationEvent:
    event_type: str
    payload: dict[str, Any]
    actor_id: str | None = None
    session_id: str | None = None
    caused_by: tuple[str, ...] = ()
    structural: dict[str, Any] | None = None


class ApplicationTraceAssembler:
    """Maps common agent runtime facts onto the collector's canonical vocabulary."""

    def __init__(self, collector: "LocalCollector") -> None:
        self.collector = collector

    def emit(self, event: ApplicationEvent) -> str:
        return self.collector.event(
            event_type=event.event_type,
            payload=event.payload,
            actor_id=event.actor_id,
            session_id=event.session_id,
            caused_by=event.caused_by,
            structural=event.structural,
        )

    def react_step(self, payload: Mapping[str, Any], **identity: Any) -> str:
        return self.emit(ApplicationEvent("react.step", dict(payload), **identity))

    def codex_item(self, payload: Mapping[str, Any], **identity: Any) -> str:
        kind = str(payload.get("type") or "item")
        return self.emit(ApplicationEvent(f"codex.{kind}", dict(payload), **identity))

    def jesterky_transition(self, payload: Mapping[str, Any], **identity: Any) -> str:
        return self.emit(ApplicationEvent("jesterky.transition", dict(payload), **identity))

    def delegation(
        self,
        payload: Mapping[str, Any],
        *,
        parent_actor_id: str,
        child_actor_id: str,
        **identity: Any,
    ) -> str:
        body = dict(payload)
        body.update(
            {
                "parent_actor_id": parent_actor_id,
                "child_actor_id": child_actor_id,
            }
        )
        return self.emit(ApplicationEvent("agent.delegated", body, **identity))

    def actor_group(self, group: ActorGroupV1) -> str:
        return self.emit(
            ApplicationEvent(
                ACTOR_GROUP_DECLARED_EVENT,
                {"actor_group": group.to_dict()},
            )
        )

    def interaction(self, interaction: InteractionEdgeV1) -> str:
        return self.emit(
            ApplicationEvent(
                coordination_event_type(interaction.kind),
                {"interaction": interaction.to_dict()},
            )
        )

    def context_epoch(self, context_epoch: ContextEpochV1) -> str:
        return self.emit(
            ApplicationEvent(
                CONTEXT_EPOCH_OBSERVED_EVENT,
                {"context_epoch": context_epoch.to_dict()},
                actor_id=context_epoch.actor_id,
                session_id=context_epoch.session_id,
            )
        )

    def joint_turn(self, joint_turn: JointTurnV1) -> str:
        return self.emit(
            ApplicationEvent(
                JOINT_TURN_OBSERVED_EVENT,
                {"joint_turn": joint_turn.to_dict()},
                actor_id=joint_turn.environment_actor_id,
                session_id=joint_turn.environment_session_id,
            )
        )


__all__ = ["ApplicationEvent", "ApplicationTraceAssembler"]
