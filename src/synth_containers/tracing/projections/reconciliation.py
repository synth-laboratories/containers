"""Build the proof that live raw facts converge to sealed Trace V5."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..canonical import content_digest, utc_now
from ..capture.envelope import RawCaptureEnvelopeV1, RawRecordType, validate_envelope_payload
from ..models.document import TraceDocumentV5
from ..models.reconciliation import (
    LiveReconciliationDisposition,
    LiveReconciliationEntryV1,
    LiveReconciliationTargetV1,
    TraceLiveReconciliationReceiptV1,
)
from ..models.selectors import SelectorKind, selector_for


def build_live_reconciliation(
    document: TraceDocumentV5,
    records: Iterable[Mapping[str, Any] | RawCaptureEnvelopeV1],
    *,
    envelope_to_event: Mapping[str, str],
    call_index_to_span: Mapping[int, str],
) -> TraceLiveReconciliationReceiptV1:
    """Map every durable live ordinal to retained, merged, or dropped final truth."""

    if not document.content_digest:
        raise ValueError("live reconciliation requires a sealed trace")
    envelopes = _envelopes(records, capture_id=document.capture.capture_id)
    entries = tuple(
        _entry(
            document,
            envelope,
            envelope_to_event=envelope_to_event,
            call_index_to_span=call_index_to_span,
        )
        for envelope in envelopes
    )
    return TraceLiveReconciliationReceiptV1(
        capture_id=document.capture.capture_id,
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        high_water_ordinal=envelopes[-1].ordinal if envelopes else -1,
        entries=entries,
        generated_at=utc_now(),
    ).sealed()


def _entry(
    document: TraceDocumentV5,
    envelope: RawCaptureEnvelopeV1,
    *,
    envelope_to_event: Mapping[str, str],
    call_index_to_span: Mapping[int, str],
) -> LiveReconciliationEntryV1:
    event_id = envelope_to_event.get(envelope.envelope_id)
    if event_id:
        event = document.event(event_id)
        if event is None:
            raise ValueError(
                f"reconciliation event {event_id!r} is absent from the sealed trace"
            )
        return _resolved_entry(
            document,
            envelope,
            kind=SelectorKind.EVENT,
            entity_id=event_id,
            entity_digest=event.content_digest,
            disposition=LiveReconciliationDisposition.RETAINED,
            reason="raw_envelope_retained_as_event",
        )

    record_type = str(envelope.record_type)
    if record_type in {
        str(RawRecordType.CAPTURE_STARTED),
        str(RawRecordType.CAPTURE_FINISHED),
        str(RawRecordType.ALIAS_DECLARED),
    }:
        return _resolved_entry(
            document,
            envelope,
            kind=SelectorKind.TRACE,
            entity_id=None,
            entity_digest=document.content_digest,
            disposition=LiveReconciliationDisposition.MERGED,
            reason="raw_capture_fact_merged_into_trace",
        )
    if record_type == str(RawRecordType.ACTOR_DECLARED):
        actor = document.actor(envelope.actor_id)
        if actor is not None:
            return _resolved_entry(
                document,
                envelope,
                kind=SelectorKind.ACTOR,
                entity_id=actor.actor_id,
                entity_digest=actor.content_digest,
                disposition=LiveReconciliationDisposition.RETAINED,
                reason="raw_actor_fact_retained",
            )
    if record_type in {
        str(RawRecordType.CHILD_REGISTERED),
        str(RawRecordType.SESSION_FINISHED),
    }:
        session = document.session(envelope.session_id)
        if session is not None:
            return _resolved_entry(
                document,
                envelope,
                kind=SelectorKind.SESSION,
                entity_id=session.session_id,
                entity_digest=session.content_digest,
                disposition=LiveReconciliationDisposition.RETAINED,
                reason="raw_session_fact_retained",
            )
    if record_type == str(RawRecordType.ARTIFACT):
        artifact_id = str(envelope.payload.get("artifact_id") or "")
        artifact = document.artifact(artifact_id)
        if artifact is not None:
            return _resolved_entry(
                document,
                envelope,
                kind=SelectorKind.ARTIFACT,
                entity_id=artifact.artifact_id,
                entity_digest=artifact.digest,
                disposition=LiveReconciliationDisposition.RETAINED,
                reason="raw_artifact_fact_retained",
            )
    call_index = _call_index(envelope)
    if call_index is not None and call_index in call_index_to_span:
        span_id = call_index_to_span[call_index]
        span = document.span(span_id)
        if span is None:
            raise ValueError(
                f"reconciliation span {span_id!r} is absent from the sealed trace"
            )
        return _resolved_entry(
            document,
            envelope,
            kind=SelectorKind.SPAN,
            entity_id=span.span_id,
            entity_digest=span.content_digest,
            disposition=LiveReconciliationDisposition.MERGED,
            reason="raw_provider_fact_merged_into_model_call_span",
        )
    return LiveReconciliationEntryV1(
        ordinal=envelope.ordinal,
        envelope_id=envelope.envelope_id,
        record_type=record_type,
        disposition=LiveReconciliationDisposition.DROPPED,
        reason="no_final_entity",
        losses=("raw_fact_has_no_canonical_v5_entity",),
    )


def _resolved_entry(
    document: TraceDocumentV5,
    envelope: RawCaptureEnvelopeV1,
    *,
    kind: SelectorKind,
    entity_id: str | None,
    entity_digest: str,
    disposition: LiveReconciliationDisposition,
    reason: str,
) -> LiveReconciliationEntryV1:
    return LiveReconciliationEntryV1(
        ordinal=envelope.ordinal,
        envelope_id=envelope.envelope_id,
        record_type=str(envelope.record_type),
        disposition=disposition,
        targets=(
            LiveReconciliationTargetV1(
                selector=selector_for(
                    document,
                    kind=kind,
                    entity_id=entity_id,
                ),
                entity_digest=entity_digest,
            ),
        ),
        reason=reason,
    )


def _envelopes(
    records: Iterable[Mapping[str, Any] | RawCaptureEnvelopeV1],
    *,
    capture_id: str,
) -> tuple[RawCaptureEnvelopeV1, ...]:
    found: list[RawCaptureEnvelopeV1] = []
    previous = -1
    for record in records:
        envelope = (
            record
            if isinstance(record, RawCaptureEnvelopeV1)
            else validate_envelope_payload(
                record,
                expected_capture_id=capture_id,
                previous_ordinal=previous,
            )
        )
        if envelope.capture_id != capture_id:
            raise ValueError("reconciliation records mix capture ids")
        if envelope.ordinal <= previous:
            raise ValueError("reconciliation records are not strictly ordered")
        if envelope.content_digest != content_digest(envelope):
            raise ValueError("reconciliation envelope digest mismatch")
        found.append(envelope)
        previous = envelope.ordinal
    return tuple(found)


def _call_index(envelope: RawCaptureEnvelopeV1) -> int | None:
    value = envelope.payload.get("call_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw call_index must be an integer") from exc


__all__ = ["build_live_reconciliation"]
