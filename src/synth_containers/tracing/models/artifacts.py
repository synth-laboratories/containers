"""Digest-addressed artifact references for large or sensitive material."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from .actors import Visibility


class ArtifactRole(StrEnum):
    RAW_PROVIDER_PAYLOAD = "raw_provider_payload"
    ROLLOUT_RESPONSE = "rollout_response"
    TOOL_OUTPUT = "tool_output"
    SOURCE_CODE = "source_code"
    EVALUATION_OUTPUT = "evaluation_output"
    RESEARCH_LOG = "research_log"
    PROXY_AUDIT = "proxy_audit"
    PROJECTION = "projection"
    RECEIPT = "receipt"
    SCREENSHOT = "screenshot"
    OTHER = "other"


class ArtifactCompleteness(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    REFERENCE_ONLY = "reference_only"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class ArtifactRefV5(JsonDataclassMixin):
    artifact_id: str
    digest: str
    media_type: str
    size_bytes: int
    role: ArtifactRole | str = ArtifactRole.OTHER
    uri: str | None = None
    producer: str | None = None
    source_authority: str | None = None
    visibility: Visibility | str = Visibility.PRIVATE
    completeness: ArtifactCompleteness | str = ArtifactCompleteness.COMPLETE
    produced_at: str | None = None
    observed_at: str | None = None
    ingested_at: str | None = None
    logical_name: str | None = None
    supersedes_artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["ArtifactCompleteness", "ArtifactRefV5", "ArtifactRole"]
