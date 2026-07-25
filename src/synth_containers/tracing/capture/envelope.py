"""``RawCaptureEnvelopeV1`` — the append-only raw record every capture writes first.

Raw envelopes are the durable substrate. Canonical V5 entities and the V4 projection
are both derived from them, so a normalizer bug is repairable without re-running the
workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now


RAW_ENVELOPE_SCHEMA_VERSION = "synth.capture.raw.v1"


class RawRecordType(StrEnum):
    CAPTURE_STARTED = "capture.started"
    CAPTURE_FINISHED = "capture.finished"
    ACTOR_DECLARED = "actor.declared"
    ALIAS_DECLARED = "alias.declared"
    CHILD_REGISTERED = "child.registered"
    SESSION_FINISHED = "session.finished"
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


def validate_envelope_payload(
    payload: Mapping[str, Any],
    *,
    expected_capture_id: str | None = None,
    previous_ordinal: int | None = None,
) -> RawCaptureEnvelopeV1:
    """Rehydrate and prove the identity, digest, and ordering of a raw envelope."""

    try:
        envelope = RawCaptureEnvelopeV1(
            envelope_id=str(payload["envelope_id"]),
            capture_id=str(payload["capture_id"]),
            ordinal=int(payload["ordinal"]),
            record_type=str(payload["record_type"]),
            occurred_at=str(payload["occurred_at"]),
            actor_id=str(payload["actor_id"]),
            session_id=str(payload["session_id"]),
            call_id=str(payload["call_id"]) if payload.get("call_id") is not None else None,
            upstream_attempt_id=(
                str(payload["upstream_attempt_id"])
                if payload.get("upstream_attempt_id") is not None
                else None
            ),
            sequence_in_call=(
                int(payload["sequence_in_call"])
                if payload.get("sequence_in_call") is not None
                else None
            ),
            payload=dict(payload.get("payload") or {}),
            producer=str(payload.get("producer") or "synth-trace"),
            producer_version=str(payload.get("producer_version") or "1"),
            schema_version=str(
                payload.get("schema_version") or RAW_ENVELOPE_SCHEMA_VERSION
            ),
            content_digest=str(payload.get("content_digest") or ""),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid raw envelope shape: {exc}") from exc
    if envelope.schema_version != RAW_ENVELOPE_SCHEMA_VERSION:
        raise ValueError(f"unsupported raw envelope schema: {envelope.schema_version}")
    if expected_capture_id and envelope.capture_id != expected_capture_id:
        raise ValueError(
            f"raw envelope capture_id {envelope.capture_id!r} does not match "
            f"{expected_capture_id!r}"
        )
    if envelope.ordinal < 0:
        raise ValueError("raw envelope ordinal must be non-negative")
    if previous_ordinal is not None and envelope.ordinal <= previous_ordinal:
        raise ValueError(
            f"raw envelope ordinal {envelope.ordinal} is not above {previous_ordinal}"
        )
    if not envelope.actor_id or not envelope.session_id:
        raise ValueError("raw envelope actor_id and session_id are required")
    expected_id = record_id(
        "raw",
        kind="raw_envelope",
        scope=(envelope.capture_id,),
        key={"ordinal": envelope.ordinal, "type": str(envelope.record_type)},
    )
    if envelope.envelope_id != expected_id:
        raise ValueError(
            f"raw envelope id mismatch: expected {expected_id}, found "
            f"{envelope.envelope_id}"
        )
    if not envelope.content_digest:
        raise ValueError("raw envelope is not sealed")
    actual_digest = content_digest(envelope)
    if actual_digest != envelope.content_digest:
        raise ValueError(
            f"raw envelope digest mismatch: expected {envelope.content_digest}, "
            f"found {actual_digest}"
        )
    return envelope


__all__ = [
    "RAW_ENVELOPE_SCHEMA_VERSION",
    "RawCaptureEnvelopeV1",
    "RawRecordType",
    "make_envelope",
    "validate_envelope_payload",
]
