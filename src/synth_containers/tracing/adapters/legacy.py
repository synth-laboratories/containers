"""Legacy-format dispatcher with conservative coverage declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..canonical import bytes_digest, canonical_bytes
from .atif import import_atif, inspect_atif_import
from .experiments_v4 import import_experiments_trace_v4
from .optimizer_event_history import import_optimizer_event_history
from .v4 import import_rollout_trace_v4


@dataclass(frozen=True, slots=True)
class LegacyImport:
    source_format: str
    source_digest: str
    canonical: Any | None
    payload: Mapping[str, Any]
    coverage: str
    losses: tuple[str, ...]


def import_legacy(payload: Mapping[str, Any], *, source_format: str) -> LegacyImport:
    normalized = source_format.lower().strip()
    digest = bytes_digest(canonical_bytes(payload))
    safe_payload = _redacted_payload(payload)
    canonical = None
    losses = ("raw provider transport was not captured",)
    coverage = "imported_opaque"
    if normalized in {"experiments.trace.v4", "executionrecord.trace_payload"}:
        canonical = import_experiments_trace_v4(safe_payload)
        coverage = "partial_agent_and_environment"
    elif normalized in {"containers.rollout_trace.v4", "trajectory"}:
        canonical = import_rollout_trace_v4(safe_payload)
        coverage = "model_calls_from_projection"
    elif normalized in {"harbor-atif", "atif"}:
        assessment = inspect_atif_import(safe_payload)
        canonical = import_atif(safe_payload)
        losses = tuple(assessment["losses"])
        coverage = "atif_steps_only"
    elif normalized == "optimizer.event_history":
        canonical = import_optimizer_event_history(safe_payload)
        coverage = "partial_model_and_application_events"
        losses = tuple(canonical.completeness.reasons)
    elif normalized in {"smr.transcript", "opaque"}:
        losses = (
            "source preserved as an opaque import artifact",
            "semantic entities require a source-specific assembler",
            "raw provider transport was not captured",
        )
    else:
        raise ValueError(f"unsupported legacy source format: {source_format}")
    return LegacyImport(
        source_format=source_format,
        source_digest=digest,
        canonical=canonical,
        payload=safe_payload,
        coverage=coverage,
        losses=losses,
    )


def _redacted_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    # Redaction is a runtime transformation boundary. Importing capture.redaction at
    # module load would create adapters -> capture.finalizer -> adapters.
    from ..capture.redaction import assert_no_secrets, redact_payload

    redacted, _ = redact_payload(payload)
    if not isinstance(redacted, Mapping):
        raise TypeError("redacted legacy source must remain a mapping")
    assert_no_secrets(redacted, where="legacy trace import")
    return redacted


__all__ = ["LegacyImport", "import_legacy"]
