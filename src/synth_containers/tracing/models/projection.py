"""``ProjectionManifestV1`` — every export names its source digest and its loss."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .actors import Visibility

if TYPE_CHECKING:
    from ..capture.binding import TraceCaptureBindingV1


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
    source_binding_id: str | None = None
    source_binding_digest: str | None = None
    capture_policy: dict[str, Any] = field(default_factory=dict)
    capture_policy_digest: str | None = None
    requested_view: dict[str, Any] = field(default_factory=dict)
    redaction_profile: str = "unknown"
    redaction_provenance: str = "unknown"
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


def bind_projection_manifest(
    manifest: ProjectionManifestV1,
    binding: "TraceCaptureBindingV1",
) -> ProjectionManifestV1:
    """Attach the exact, non-secret capture authority used for this projection."""

    return replace(
        manifest,
        config_digest=binding.capture.proxy_config_digest,
        source_binding_id=binding.binding_id,
        source_binding_digest=binding.content_digest,
        capture_policy=binding.policy.to_dict(),
        capture_policy_digest=binding.policy.digest(),
        redaction_profile=binding.policy.redaction_profile,
        redaction_provenance="source_capture_binding.policy.redaction_profile",
        content_digest="",
    )


__all__ = [
    "PROJECTION_MANIFEST_SCHEMA_VERSION",
    "ProjectionLossV1",
    "ProjectionManifestV1",
    "bind_projection_manifest",
]
