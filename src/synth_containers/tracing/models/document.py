"""``TraceDocumentV5`` — the sealed, content-addressed execution-fact authority."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .actors import ActorV5, SessionV5, Visibility
from .artifacts import ArtifactRefV5
from .completeness import TraceCompletenessV5, TraceLifecycleV5
from .events import EventV5, TraceErrorV1
from .identity import (
    TRACE_SCHEMA_VERSION,
    AliasV1,
    TraceIdentityV5,
    TraceKind,
    TraceProvenanceV5,
)
from .messages import BranchV5, MessageNodeV5
from .spans import SpanV5, UsageV5


@dataclass(frozen=True, slots=True)
class TraceCaptureSummaryV5(JsonDataclassMixin):
    """The capture session that produced this document."""

    capture_id: str
    binding_id: str
    binding_digest: str
    capture_profile: str
    interception: str
    mode: str
    proxy_config_digest: str | None = None
    coverage_receipt_id: str | None = None
    segment_digests: tuple[str, ...] = ()
    segment_count: int = 0
    raw_record_count: int = 0


@dataclass(frozen=True, slots=True)
class TraceLinkV5(JsonDataclassMixin):
    """A typed edge to another trace, projection, or external record."""

    relation: str
    target_id: str
    target_digest: str | None = None
    target_kind: str = "trace"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceDocumentV5(JsonDataclassMixin):
    """Immutable core capture. Sealed bytes never change once ``content_digest`` is set."""

    trace_id: str
    trace_kind: TraceKind | str
    identity: TraceIdentityV5
    lifecycle: TraceLifecycleV5
    capture: TraceCaptureSummaryV5
    provenance: TraceProvenanceV5
    completeness: TraceCompletenessV5
    actors: tuple[ActorV5, ...] = ()
    sessions: tuple[SessionV5, ...] = ()
    messages: tuple[MessageNodeV5, ...] = ()
    branches: tuple[BranchV5, ...] = ()
    spans: tuple[SpanV5, ...] = ()
    events: tuple[EventV5, ...] = ()
    artifacts: tuple[ArtifactRefV5, ...] = ()
    errors: tuple[TraceErrorV1, ...] = ()
    usage: UsageV5 = field(default_factory=UsageV5)
    aliases: tuple[AliasV1, ...] = ()
    links: tuple[TraceLinkV5, ...] = ()
    visibility: Visibility | str = Visibility.PRIVATE
    extensions: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "TraceDocumentV5":
        """Return the sealed document; sealing is the only way to set ``content_digest``."""

        return replace(self, content_digest=content_digest(self))

    def actor(self, actor_id: str) -> ActorV5 | None:
        return next((item for item in self.actors if item.actor_id == actor_id), None)

    def session(self, session_id: str) -> SessionV5 | None:
        return next((item for item in self.sessions if item.session_id == session_id), None)

    def span(self, span_id: str) -> SpanV5 | None:
        return next((item for item in self.spans if item.span_id == span_id), None)

    def event(self, event_id: str) -> EventV5 | None:
        return next((item for item in self.events if item.event_id == event_id), None)

    def message(self, message_id: str) -> MessageNodeV5 | None:
        return next((item for item in self.messages if item.message_id == message_id), None)

    def artifact(self, artifact_id: str) -> ArtifactRefV5 | None:
        return next((item for item in self.artifacts if item.artifact_id == artifact_id), None)

    def events_of_type(self, event_type: str) -> tuple[EventV5, ...]:
        return tuple(item for item in self.events if str(item.event_type) == str(event_type))

    def spans_of_kind(self, span_kind: str) -> tuple[SpanV5, ...]:
        return tuple(item for item in self.spans if str(item.span_kind) == str(span_kind))


__all__ = ["TraceCaptureSummaryV5", "TraceDocumentV5", "TraceLinkV5"]
