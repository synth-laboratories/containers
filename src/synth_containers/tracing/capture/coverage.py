"""``CaptureCoverageReceiptV1`` — four separate claims about one capture attempt.

Configured, reachable, observed, and complete are distinct. Only ``complete`` for the
declared scope supports a complete-trace claim, and the receipt says which scope that
is. A capture with zero calls still produces a receipt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now
from ..models.actors import CoverageState


COVERAGE_RECEIPT_SCHEMA_VERSION = "synth.capture-coverage-receipt.v1"


class CaptureScope(StrEnum):
    """What the capture claims to cover. Naming it prevents overclaim."""

    MODEL_CALLS_ONLY = "model_calls_only"
    MODEL_CALLS_AND_APPLICATION = "model_calls_and_application"
    APPLICATION_ONLY = "application_only"
    IMPORTED_AGENT_EVENTS = "imported_agent_events"


class Completeness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CaptureCoverageReceiptV1(JsonDataclassMixin):
    receipt_id: str
    binding_id: str
    binding_digest: str
    capture_id: str
    scope: CaptureScope | str
    requested_mode: str
    resolved_mode: str
    interception: str
    proxy_config_digest: str
    started_at: str
    registration_ok: bool = False
    readiness_ok: bool = False
    reachability_detail: str = ""
    direct_egress_asserted: bool = False
    provider_adapters: tuple[str, ...] = ()
    routes_enabled: tuple[str, ...] = ()
    injected_variables: tuple[str, ...] = ()
    endpoint_identity_digest: str | None = None
    calls_accepted: int = 0
    calls_completed: int = 0
    calls_errored: int = 0
    calls_canceled: int = 0
    calls_normalized: int = 0
    upstream_retries: int = 0
    application_events: int = 0
    artifacts_recorded: int = 0
    dropped_records: int = 0
    truncated_records: int = 0
    redacted_headers: tuple[str, ...] = ()
    unsupported_routes: tuple[str, ...] = ()
    malformed_records: int = 0
    segment_count: int = 0
    segment_bytes: int = 0
    segment_digests: tuple[str, ...] = ()
    first_observed_at: str | None = None
    last_observed_at: str | None = None
    child_exit_code: int | None = None
    finalization_status: str = "pending"
    ended_at: str | None = None
    model_calls: CoverageState | str = CoverageState.NOT_CAPTURED
    raw_provider: CoverageState | str = CoverageState.NOT_CAPTURED
    agent_events: CoverageState | str = CoverageState.NOT_CAPTURED
    environment_events: CoverageState | str = CoverageState.NOT_CAPTURED
    tool_events: CoverageState | str = CoverageState.NOT_CAPTURED
    usage: CoverageState | str = CoverageState.NOT_CAPTURED
    completeness: Completeness | str = Completeness.INCOMPLETE
    completeness_reasons: tuple[str, ...] = ()
    projection_refs: tuple[str, ...] = ()
    schema_version: str = COVERAGE_RECEIPT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "CaptureCoverageReceiptV1":
        return replace(self, content_digest=content_digest(self))

    @property
    def configured(self) -> bool:
        return self.registration_ok

    @property
    def reachable(self) -> bool:
        return self.readiness_ok

    @property
    def observed(self) -> bool:
        return self.calls_accepted > 0 or self.application_events > 0


def new_coverage_receipt(
    *,
    binding_id: str,
    binding_digest: str,
    capture_id: str,
    scope: CaptureScope | str,
    requested_mode: str,
    resolved_mode: str,
    interception: str,
    proxy_config_digest: str,
) -> CaptureCoverageReceiptV1:
    receipt_id = record_id("rcpt", kind="capture_coverage", scope=(capture_id,), key=binding_digest)
    return CaptureCoverageReceiptV1(
        receipt_id=receipt_id,
        binding_id=binding_id,
        binding_digest=binding_digest,
        capture_id=capture_id,
        scope=scope,
        requested_mode=requested_mode,
        resolved_mode=resolved_mode,
        interception=interception,
        proxy_config_digest=proxy_config_digest,
        started_at=utc_now(),
    )


__all__ = [
    "COVERAGE_RECEIPT_SCHEMA_VERSION",
    "CaptureCoverageReceiptV1",
    "CaptureScope",
    "Completeness",
    "new_coverage_receipt",
]
