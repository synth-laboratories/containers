"""Receipt binding live raw envelopes to final sealed Trace V5 entities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest
from .selectors import TraceSelectorV1


TRACE_LIVE_RECONCILIATION_SCHEMA_VERSION = (
    "synth.trace-live-reconciliation.v1"
)


class LiveReconciliationDisposition(StrEnum):
    RETAINED = "retained"
    MERGED = "merged"
    REDACTED = "redacted"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class LiveReconciliationTargetV1(JsonDataclassMixin):
    selector: TraceSelectorV1
    entity_digest: str


@dataclass(frozen=True, slots=True)
class LiveReconciliationEntryV1(JsonDataclassMixin):
    ordinal: int
    envelope_id: str
    record_type: str
    disposition: LiveReconciliationDisposition | str
    targets: tuple[LiveReconciliationTargetV1, ...] = ()
    reason: str = ""
    losses: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceLiveReconciliationReceiptV1(JsonDataclassMixin):
    capture_id: str
    trace_id: str
    trace_digest: str
    high_water_ordinal: int
    entries: tuple[LiveReconciliationEntryV1, ...]
    generated_at: str
    schema_version: str = TRACE_LIVE_RECONCILIATION_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "TraceLiveReconciliationReceiptV1":
        return replace(self, content_digest=content_digest(self))


__all__ = [
    "TRACE_LIVE_RECONCILIATION_SCHEMA_VERSION",
    "LiveReconciliationDisposition",
    "LiveReconciliationEntryV1",
    "LiveReconciliationTargetV1",
    "TraceLiveReconciliationReceiptV1",
]
