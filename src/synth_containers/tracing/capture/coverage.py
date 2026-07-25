"""``CaptureCoverageReceiptV1`` — four separate claims about one capture attempt.

Configured, reachable, observed, and complete are distinct. Only ``complete`` for the
declared scope supports a complete-trace claim, and the receipt says which scope that
is. A capture with zero calls still produces a receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now
from ..models.actors import ActorKind, CoverageState
from ..models.completeness import TerminationV5, TraceStatus
from ..models.identity import (
    AliasV1,
    TraceIdentityV5,
    TraceProvenanceV5,
)


COVERAGE_RECEIPT_SCHEMA_VERSION = "synth.capture-coverage-receipt.v1"
CAPTURE_FINALIZATION_SCHEMA_VERSION = "synth.capture-finalization.v1"


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


@dataclass(frozen=True, slots=True)
class CaptureFinalizationV1(JsonDataclassMixin):
    """Durable, secret-free authority for deterministic terminal resealing."""

    status: TraceStatus | str
    captured_at: str
    coverage_seed: CaptureCoverageReceiptV1
    provenance: TraceProvenanceV5
    identity: TraceIdentityV5
    root_actor_name: str
    root_actor_kind: str
    finalizer_name: str
    finalizer_version: str
    aliases: tuple[AliasV1, ...] = ()
    termination: TerminationV5 | None = None
    child_exit_code: int | None = None
    egress_assertion: dict[str, Any] | None = None
    mitm_lifecycle: dict[str, Any] | None = None
    egress_failure: str | None = None
    mitm_failure: str | None = None
    schema_version: str = CAPTURE_FINALIZATION_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "CaptureFinalizationV1":
        return replace(self, content_digest=content_digest(self))


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
    started_at: str | None = None,
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
        started_at=started_at or utc_now(),
    )


_COVERAGE_TUPLE_FIELDS = (
    "provider_adapters",
    "routes_enabled",
    "injected_variables",
    "redacted_headers",
    "unsupported_routes",
    "segment_digests",
    "completeness_reasons",
    "projection_refs",
)


def coverage_receipt_from_dict(
    payload: dict[str, Any],
    *,
    require_digest: bool = True,
) -> CaptureCoverageReceiptV1:
    """Rehydrate a coverage receipt and reject schema or digest drift."""

    values = dict(payload)
    for name in _COVERAGE_TUPLE_FIELDS:
        values[name] = tuple(values.get(name) or ())
    receipt = CaptureCoverageReceiptV1(**values)
    if receipt.schema_version != COVERAGE_RECEIPT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported capture coverage schema: {receipt.schema_version}"
        )
    if require_digest and receipt.content_digest != content_digest(receipt):
        raise ValueError("capture coverage receipt digest mismatch")
    return receipt


def finalization_from_dict(payload: dict[str, Any]) -> CaptureFinalizationV1:
    """Rehydrate and verify one typed terminal capture fact."""

    values = dict(payload)
    coverage_payload = values.get("coverage_seed")
    if not isinstance(coverage_payload, dict):
        raise ValueError("capture finalization coverage seed is invalid")
    values["coverage_seed"] = coverage_receipt_from_dict(coverage_payload)
    provenance_payload = values.get("provenance")
    if not isinstance(provenance_payload, dict):
        raise ValueError("capture finalization provenance is invalid")
    provenance_values = dict(provenance_payload)
    provenance_values["transformation_chain"] = tuple(
        provenance_values.get("transformation_chain") or ()
    )
    provenance_values["aliases"] = _aliases_from_payload(
        provenance_values.get("aliases"),
        field="capture finalization provenance aliases",
    )
    values["provenance"] = TraceProvenanceV5(**provenance_values)
    identity_payload = values.get("identity")
    if not isinstance(identity_payload, dict):
        raise ValueError("capture finalization identity is invalid")
    values["identity"] = TraceIdentityV5(**identity_payload)
    values["aliases"] = _aliases_from_payload(
        values.get("aliases"),
        field="capture finalization aliases",
    )
    termination_payload = values.get("termination")
    if termination_payload is not None:
        if not isinstance(termination_payload, dict):
            raise ValueError("capture finalization termination is invalid")
        values["termination"] = TerminationV5(**termination_payload)
    finalization = CaptureFinalizationV1(**values)
    if finalization.schema_version != CAPTURE_FINALIZATION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported capture finalization schema: {finalization.schema_version}"
        )
    if finalization.content_digest != content_digest(finalization):
        raise ValueError("capture finalization digest mismatch")
    if str(finalization.status) not in {
        str(TraceStatus.COMPLETED),
        str(TraceStatus.FAILED),
        str(TraceStatus.INTERRUPTED),
    }:
        raise ValueError("capture finalization status must be terminal")
    captured_at = _parse_timestamp(
        finalization.captured_at,
        field="capture finalization captured_at",
    )
    if finalization.coverage_seed.finalization_status != "captured":
        raise ValueError(
            "capture finalization coverage must have captured status"
        )
    seed = finalization.coverage_seed
    if (
        seed.segment_count
        or seed.segment_bytes
        or seed.segment_digests
        or seed.application_events
        or seed.artifacts_recorded
        or seed.projection_refs
    ):
        raise ValueError(
            "capture finalization coverage seed contains derived bundle facts"
        )
    if any(
        str(value) != str(CoverageState.NOT_CAPTURED)
        for value in (
            seed.model_calls,
            seed.raw_provider,
            seed.agent_events,
            seed.environment_events,
            seed.tool_events,
            seed.usage,
        )
    ) or str(seed.completeness) != str(Completeness.INCOMPLETE):
        raise ValueError(
            "capture finalization coverage seed contains derived coverage states"
        )
    if finalization.coverage_seed.ended_at != finalization.captured_at:
        raise ValueError(
            "capture finalization coverage ended_at must equal captured_at"
        )
    if finalization.provenance.captured_at != finalization.captured_at:
        raise ValueError(
            "capture finalization provenance captured_at must equal captured_at"
        )
    if finalization.coverage_seed.child_exit_code != finalization.child_exit_code:
        raise ValueError(
            "capture finalization child exit code is inconsistent"
        )
    for field_name, timestamp in (
        ("coverage started_at", finalization.coverage_seed.started_at),
        ("coverage first_observed_at", finalization.coverage_seed.first_observed_at),
        ("coverage last_observed_at", finalization.coverage_seed.last_observed_at),
    ):
        if timestamp is not None and _parse_timestamp(
            timestamp,
            field=field_name,
        ) > captured_at:
            raise ValueError(f"{field_name} follows capture finalization")
    if finalization.egress_assertion is not None:
        passed = bool(finalization.egress_assertion.get("passed"))
        if passed != finalization.coverage_seed.direct_egress_asserted:
            raise ValueError(
                "capture finalization egress result is inconsistent"
            )
    elif finalization.coverage_seed.direct_egress_asserted:
        raise ValueError(
            "capture finalization lacks its successful egress assertion"
        )
    if (
        not isinstance(finalization.root_actor_name, str)
        or not finalization.root_actor_name.strip()
    ):
        raise ValueError("capture finalization root actor name is empty")
    if not isinstance(finalization.root_actor_kind, str) or (
        finalization.root_actor_kind not in {str(kind) for kind in ActorKind}
    ):
        raise ValueError("capture finalization root actor kind is invalid")
    if (
        not isinstance(finalization.finalizer_name, str)
        or not finalization.finalizer_name.strip()
    ):
        raise ValueError("capture finalization finalizer name is empty")
    if (
        not isinstance(finalization.finalizer_version, str)
        or not finalization.finalizer_version.strip()
    ):
        raise ValueError("capture finalization finalizer version is empty")
    return finalization


def _aliases_from_payload(value: Any, *, field: str) -> tuple[AliasV1, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    aliases: list[AliasV1] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{field} contains a non-object alias")
        alias = AliasV1(**dict(item))
        if (
            not str(alias.namespace).strip()
            or not alias.value
            or not alias.target_id
            or not alias.target_kind
        ):
            raise ValueError(f"{field} contains an incomplete alias")
        aliases.append(alias)
    return tuple(aliases)


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


__all__ = [
    "CAPTURE_FINALIZATION_SCHEMA_VERSION",
    "COVERAGE_RECEIPT_SCHEMA_VERSION",
    "CaptureCoverageReceiptV1",
    "CaptureFinalizationV1",
    "CaptureScope",
    "Completeness",
    "coverage_receipt_from_dict",
    "finalization_from_dict",
    "new_coverage_receipt",
]
