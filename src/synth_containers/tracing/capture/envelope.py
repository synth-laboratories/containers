"""``RawCaptureEnvelopeV1`` — the append-only raw record every capture writes first.

Raw envelopes are the durable substrate. Canonical V5 entities and the V4 projection
are both derived from them, so a normalizer bug is repairable without re-running the
workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now


RAW_ENVELOPE_SCHEMA_VERSION = "synth.capture.raw.v1"


class RawRecordType(StrEnum):
    CAPTURE_STARTED = "capture.started"
    CAPTURE_FINISHED = "capture.finished"
    MODEL_CALL_STARTED = "model_call.started"
    UPSTREAM_ATTEMPT_STARTED = "upstream_attempt.started"
    UPSTREAM_ATTEMPT_FINISHED = "upstream_attempt.finished"
    RESPONSE_FRAME = "response.frame"
    RESPONSE_BODY = "response.body"
    MODEL_CALL_FINISHED = "model_call.finished"
    NORMALIZER_RESULT = "normalizer.result"
    APPLICATION_EVENT = "application.event"
    ARTIFACT = "artifact"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RawCaptureEnvelopeV1(JsonDataclassMixin):
    """One immutable raw record. ``ordinal`` is monotonic within a capture session."""

    envelope_id: str
    capture_id: str
    ordinal: int
    record_type: RawRecordType | str
    occurred_at: str
    actor_id: str
    session_id: str
    call_id: str | None = None
    upstream_attempt_id: str | None = None
    sequence_in_call: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    producer: str = "synth-trace"
    producer_version: str = "1"
    schema_version: str = RAW_ENVELOPE_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "RawCaptureEnvelopeV1":
        return replace(self, content_digest=content_digest(self))


def make_envelope(
    *,
    capture_id: str,
    ordinal: int,
    record_type: RawRecordType | str,
    actor_id: str,
    session_id: str,
    payload: dict[str, Any] | None = None,
    call_id: str | None = None,
    upstream_attempt_id: str | None = None,
    sequence_in_call: int | None = None,
    occurred_at: str | None = None,
    producer_version: str = "1",
) -> RawCaptureEnvelopeV1:
    envelope_id = record_id(
        "raw",
        kind="raw_envelope",
        scope=(capture_id,),
        key={"ordinal": ordinal, "type": str(record_type)},
    )
    envelope = RawCaptureEnvelopeV1(
        envelope_id=envelope_id,
        capture_id=capture_id,
        ordinal=ordinal,
        record_type=record_type,
        occurred_at=occurred_at or utc_now(),
        actor_id=actor_id,
        session_id=session_id,
        call_id=call_id,
        upstream_attempt_id=upstream_attempt_id,
        sequence_in_call=sequence_in_call,
        payload=dict(payload or {}),
        producer_version=producer_version,
    )
    return envelope.sealed()


__all__ = [
    "RAW_ENVELOPE_SCHEMA_VERSION",
    "RawCaptureEnvelopeV1",
    "RawRecordType",
    "make_envelope",
]
