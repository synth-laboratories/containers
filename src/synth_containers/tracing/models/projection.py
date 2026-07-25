"""``ProjectionManifestV1`` — every export names its source digest and its loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .actors import Visibility


PROJECTION_MANIFEST_SCHEMA_VERSION = "synth.projection-manifest.v1"


@dataclass(frozen=True, slots=True)
class ProjectionLossV1(JsonDataclassMixin):
    """One fact the projection format cannot express."""

    field_path: str
    reason: str
    record_count: int = 0


@dataclass(frozen=True, slots=True)
class ProjectionManifestV1(JsonDataclassMixin):
    projection_id: str
    format: str
    source_trace_id: str
    source_trace_digest: str
    producer: str
    producer_version: str
    created_at: str
    config_digest: str | None = None
    requested_view: dict[str, Any] = field(default_factory=dict)
    redaction_profile: str = "default"
    included_layers: tuple[str, ...] = ()
    omitted_layers: tuple[str, ...] = ()
    losses: tuple[ProjectionLossV1, ...] = ()
    target_digest: str | None = None
    target_media_type: str = "application/json"
    visibility: Visibility | str = Visibility.PRIVATE
    schema_version: str = PROJECTION_MANIFEST_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "ProjectionManifestV1":
        return seal_record(self)


__all__ = [
    "PROJECTION_MANIFEST_SCHEMA_VERSION",
    "ProjectionLossV1",
    "ProjectionManifestV1",
]
