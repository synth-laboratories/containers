"""Immutable message graph: each semantic message is stored once and referenced by ID."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .identity import AliasV1
from .actors import Visibility


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    ORCHESTRATOR = "orchestrator"


class PartType(StrEnum):
    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STRUCTURED = "structured"
    MEDIA = "media"
    ARTIFACT = "artifact"
    OBSERVATION = "observation"
    UNSUPPORTED = "unsupported"


class ReasoningAvailability(StrEnum):
    CAPTURED = "captured"
    REDACTED = "redacted"
    SUMMARIZED = "summarized"
    REFERENCE_ONLY = "reference_only"
    UNAVAILABLE = "unavailable"
    PROVIDER_DID_NOT_EXPOSE = "provider_did_not_expose"


class BranchReason(StrEnum):
    CONTINUATION = "continuation"
    ALTERNATE_SAMPLE = "alternate_sample"
    COMPACTION = "compaction"
    DELEGATION = "delegation"
    SUBAGENT = "subagent"
    CHECKPOINT_RESUME = "checkpoint_resume"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class MessagePartV5(JsonDataclassMixin):
    """One typed content part. Unknown provider content stays ``unsupported``."""

    part_id: str
    type: PartType | str
    text: str | None = None
    reasoning_availability: ReasoningAvailability | str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_json: str | None = None
    is_error: bool | None = None
    structured: dict[str, Any] | None = None
    artifact_id: str | None = None
    media_type: str | None = None
    raw_kind: str | None = None
    conversion_diagnostics: tuple[str, ...] = ()
    visibility: Visibility | str = Visibility.PRIVATE


@dataclass(frozen=True, slots=True)
class MessageNodeV5(JsonDataclassMixin):
    message_id: str
    role: MessageRole | str
    parts: tuple[MessagePartV5, ...]
    sender_actor_id: str
    session_id: str
    predecessor_message_ids: tuple[str, ...] = ()
    recipient_actor_ids: tuple[str, ...] = ()
    turn_id: str | None = None
    thread_id: str | None = None
    produced_by_span_id: str | None = None
    produced_by_event_id: str | None = None
    occurred_at: str | None = None
    visibility: Visibility | str = Visibility.PRIVATE
    aliases: tuple[AliasV1, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "MessageNodeV5":
        return seal_record(self)

    def text(self) -> str:
        return "".join(part.text or "" for part in self.parts if part.type == PartType.TEXT)


@dataclass(frozen=True, slots=True)
class BranchV5(JsonDataclassMixin):
    branch_id: str
    head_message_id: str | None
    actor_id: str
    session_id: str
    reason: BranchReason | str = BranchReason.CONTINUATION
    parent_branch_id: str | None = None
    fork_point_message_id: str | None = None
    retained_message_ids: tuple[str, ...] = ()
    summarized_message_ids: tuple[str, ...] = ()
    removed_message_ids: tuple[str, ...] = ()
    external_context_refs: tuple[str, ...] = ()
    source_content_digest: str | None = None
    result_content_digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BranchReason",
    "BranchV5",
    "MessageNodeV5",
    "MessagePartV5",
    "MessageRole",
    "PartType",
    "ReasoningAvailability",
]
