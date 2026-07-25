"""Trace representation negotiation without introducing another write authority."""

from __future__ import annotations

from typing import Any

from .adapters.atif import export_atif
from .models.document import TraceDocumentV5
from .projections.v4 import project_v4


SUPPORTED_TRACE_FORMATS = ("synth-v5", "synth-v4", "harbor-atif")


def negotiate_trace(
    document: TraceDocumentV5,
    accept: str | None,
) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    requested = (accept or "synth-v5").lower().strip()
    if requested in {"synth-v5", "application/vnd.synth.trace.v5+json"}:
        return "synth-v5", document.to_dict(), ()
    if requested in {"synth-v4", "application/vnd.synth.trace.v4+json"}:
        projected, manifest = project_v4(document)
        return (
            "synth-v4",
            projected.to_dict(),
            tuple(f"{item.field_path}: {item.reason}" for item in manifest.losses),
        )
    if requested in {"harbor-atif", "application/vnd.harbor.atif+json"}:
        payload = export_atif(document)
        return (
            "harbor-atif",
            payload,
            tuple(payload["extra"]["projection_losses"]),
        )
    raise ValueError(
        f"unsupported trace representation {accept!r}; "
        f"supported={','.join(SUPPORTED_TRACE_FORMATS)}"
    )


__all__ = ["SUPPORTED_TRACE_FORMATS", "negotiate_trace"]
