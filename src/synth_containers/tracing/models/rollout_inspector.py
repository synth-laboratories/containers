"""Versioned viewer packet composed from the generic sealed visual model."""

from __future__ import annotations

from dataclasses import dataclass, replace

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .visual import TraceVisualProjectionV1


ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION = (
    "synth.trace-projection.rollout-inspector.v1"
)


@dataclass(frozen=True, slots=True)
class RolloutInspectorProjectionV1(JsonDataclassMixin):
    """Consumer packet whose items retain stable selectors into one sealed trace."""

    trace_id: str
    trace_digest: str
    capture_id: str
    visual: TraceVisualProjectionV1
    evidence_digest: str | None = None
    schema_version: str = ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "RolloutInspectorProjectionV1":
        return replace(self, content_digest=content_digest(self))


__all__ = [
    "ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION",
    "RolloutInspectorProjectionV1",
]
