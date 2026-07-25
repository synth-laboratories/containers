"""Application-event adapters for ReAct, Codex, Jesterky, and delegated agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

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


__all__ = ["ApplicationEvent", "ApplicationTraceAssembler"]
