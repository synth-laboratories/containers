"""Shared live and sealed read model for trace visualization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .selectors import TraceSelectorV1


TRACE_VISUAL_SCHEMA_VERSION = "synth.trace-visual.v1"


class TraceVisualState(StrEnum):
    PROVISIONAL = "provisional"
    SEALED = "sealed"


@dataclass(frozen=True, slots=True)
class TraceVisualLaneV1(JsonDataclassMixin):
    lane_id: str
    actor_id: str
    session_id: str
    display_name: str
    actor_kind: str
    role: str = ""
    parent_actor_id: str | None = None
    visibility: str = "private"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceVisualItemV1(JsonDataclassMixin):
    item_id: str
    kind: str
    occurred_at: str
    title: str
    actor_id: str | None = None
    session_id: str | None = None
    lane_id: str | None = None
    sequence: int | None = None
    status: str = ""
    source_envelope_id: str | None = None
    source_ordinal: int | None = None
    source_selector: TraceSelectorV1 | None = None
    source_digest: str | None = None
    visibility: str = "private"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceVisualProjectionV1(JsonDataclassMixin):
    capture_id: str
    state: TraceVisualState | str
    lanes: tuple[TraceVisualLaneV1, ...]
    items: tuple[TraceVisualItemV1, ...]
    trace_id: str | None = None
    trace_digest: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    high_water_ordinal: int = -1
    visibility_ceiling: str = "private"
    usage: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    losses: tuple[str, ...] = ()
    schema_version: str = TRACE_VISUAL_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "TraceVisualProjectionV1":
        return replace(self, content_digest=content_digest(self))


__all__ = [
    "TRACE_VISUAL_SCHEMA_VERSION",
    "TraceVisualItemV1",
    "TraceVisualLaneV1",
    "TraceVisualProjectionV1",
    "TraceVisualState",
]
