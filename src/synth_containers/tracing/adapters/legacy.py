"""Legacy-format dispatcher with conservative coverage declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..canonical import bytes_digest, canonical_bytes
from .atif import import_atif, inspect_atif_import
from .experiments_v4 import import_experiments_trace_v4
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
    canonical = None
    losses = ("raw provider transport was not captured",)
    coverage = "imported_opaque"
    if normalized in {"experiments.trace.v4", "executionrecord.trace_payload"}:
        canonical = import_experiments_trace_v4(payload)
        coverage = "partial_agent_and_environment"
    elif normalized in {"containers.rollout_trace.v4", "trajectory"}:
        canonical = import_rollout_trace_v4(payload)
        coverage = "model_calls_from_projection"
    elif normalized in {"harbor-atif", "atif"}:
        assessment = inspect_atif_import(payload)
        canonical = import_atif(payload)
        losses = tuple(assessment["losses"])
        coverage = "atif_steps_only"
    elif normalized in {"optimizer.event_history", "smr.transcript", "opaque"}:
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
        payload=payload,
        coverage=coverage,
        losses=losses,
    )


__all__ = ["LegacyImport", "import_legacy"]
