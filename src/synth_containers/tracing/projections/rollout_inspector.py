"""Build the common rollout-inspector packet from sealed Trace V5 authority."""

from __future__ import annotations

from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.rollout_inspector import RolloutInspectorProjectionV1
from .visual import visual_from_sealed


def rollout_inspector_from_sealed(
    document: TraceDocumentV5,
    evidence: TraceEvidenceBundleV5 | None = None,
    *,
    visibility_ceiling: str = "private",
) -> RolloutInspectorProjectionV1:
    """Return a versioned viewer packet bound to the sealed trace digest."""

    visual = visual_from_sealed(
        document,
        evidence,
        visibility_ceiling=visibility_ceiling,
    )
    return RolloutInspectorProjectionV1(
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        capture_id=document.capture.capture_id,
        evidence_digest=evidence.content_digest if evidence is not None else None,
        visual=visual,
    ).sealed()


__all__ = ["rollout_inspector_from_sealed"]
