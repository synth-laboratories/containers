"""Security disposition for provider bodies retained by Trace V5 capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin


class RawCaptureDisposition(StrEnum):
    NONE = "none"
    REDACTED_INLINE = "redacted_inline"
    REDACTED_ARTIFACT = "redacted_artifact"
    ENCRYPTED_ARTIFACT = "encrypted_artifact"
    DIGEST_ONLY = "digest_only"


@dataclass(frozen=True, slots=True)
class CapturedBodyRefV1(JsonDataclassMixin):
    """A wire body and the safe representation that capture retained."""

    wire_digest: str
    wire_byte_size: int
    disposition: RawCaptureDisposition | str
    media_type: str = "application/octet-stream"
    stored_digest: str | None = None
    uri: str | None = None
    inline: Any | None = None
    redaction_profile: str | None = None
    encryption_profile: str | None = None
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["CapturedBodyRefV1", "RawCaptureDisposition"]
