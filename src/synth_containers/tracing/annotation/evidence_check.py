"""Validate an evidence revision at the cost of what it appends.

``validate_evidence`` is the authority: it re-resolves every citation, re-digests
every record, and cross-checks every reference in a bundle. Run on a growing
evidence head it is quadratic over a campaign — the thousandth job re-validates
the 999 jobs before it. Sealed traces and sealed records are immutable, so a
revision built by ``attach_many`` on a head that already validated clean against
this trace only needs its *appended* records checked, plus the references those
records make into the head.

That is exactly what ``validate_appended_evidence`` does:

* the trace lookups go through ``SealedTraceIndex.view`` (O(1) entity lookup,
  identical semantics — the real ``resolve_selector`` runs on it);
* if the prior head is known-clean for this trace and the candidate extends it
  record-for-record (identity, not equality), the authority runs on a *delta
  bundle*: every definition collection whole, the appended annotations and
  receipts, and every prior annotation an appended annotation refers to
  (derivation sources, dissent, supersession — followed transitively);
* otherwise the authority runs on the whole candidate, and the result is
  remembered so the next revision is incremental.

Nothing is skipped for appended records: every selector still resolves and every
quote still matches, through the same code path as before.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..models.evidence import TraceEvidenceBundleV5
from ..models.standards import AnnotationV1
from ..validation.validator import Severity, ValidationFindingV1, validate_evidence
from .trace_index import SealedTraceIndex

# Collections this package appends per job; every other collection is small
# (definitions, rubrics, criteria) or is cross-referenced by kinds this package
# never appends (reward/evaluation records), so those stay whole in the delta.
_COLLECTIONS = (
    "criteria",
    "rubrics",
    "verifier_definitions",
    "annotator_definitions",
    "reward_definitions",
    "annotations",
    "verifier_results",
    "reward_records",
    "reward_aggregations",
    "evaluation_results",
    "benchmark_verdicts",
    "receipts",
    "artifacts",
)


def _extends(prior: TraceEvidenceBundleV5, candidate: TraceEvidenceBundleV5) -> bool:
    """True when every prior record is the same object at the same position in the candidate."""

    for name in _COLLECTIONS:
        before = getattr(prior, name)
        after = getattr(candidate, name)
        if len(after) < len(before):
            return False
        for old, new in zip(before, after):
            if old is not new:
                return False
    return True


def _referenced_prior_annotations(
    new_annotations: tuple[AnnotationV1, ...],
    prior_by_id: dict[str, AnnotationV1],
) -> list[AnnotationV1]:
    """Prior annotations reachable from the appended ones, in head order."""

    wanted: set[str] = set()
    frontier: list[AnnotationV1] = list(new_annotations)
    while frontier:
        item = frontier.pop()
        refs: list[str] = []
        if item.supersedes_id:
            refs.append(item.supersedes_id)
        derivation = item.derivation
        if derivation is not None:
            refs.extend(derivation.source_annotation_ids)
            refs.extend(derivation.dissenting_annotation_ids)
        for ref in refs:
            if ref in wanted:
                continue
            found = prior_by_id.get(ref)
            if found is not None:
                wanted.add(ref)
                frontier.append(found)
    return [item for item in prior_by_id.values() if item.annotation_id in wanted]


def _error(code: str, message: str, entity_id: str | None = None) -> ValidationFindingV1:
    return ValidationFindingV1(code=code, severity=Severity.ERROR, message=message, entity_id=entity_id)


def validate_appended_evidence(
    index: SealedTraceIndex,
    candidate: TraceEvidenceBundleV5,
    *,
    prior: TraceEvidenceBundleV5 | None,
) -> tuple[list[ValidationFindingV1], dict[str, Any]]:
    """Every check ``validate_evidence`` makes on ``candidate``, paying only for what is new.

    Returns the findings and a small report (``mode`` is ``"full"`` or
    ``"incremental"``) for receipts and tests.
    """

    if candidate.trace_ref.content_digest != index.trace_digest or candidate.trace_ref.trace_id != index.trace_id:
        # Wrong trace: let the authority say so in its own words.
        findings, _, _ = validate_evidence(index.view, candidate)
        return findings, {"mode": "full", "reason": "trace_mismatch"}

    incremental = (
        prior is not None
        and index.bundle_verified(prior.content_digest)
        and candidate.metadata.get("supersedes_bundle_digest") == prior.content_digest
        and _extends(prior, candidate)
    )
    if not incremental:
        findings, _, _ = validate_evidence(index.view, candidate)
        if not any(str(item.severity) == Severity.ERROR for item in findings):
            index.mark_bundle_verified(candidate.content_digest)
        return findings, {"mode": "full", "reason": "no_verified_prior" if prior is None or not index.bundle_verified(prior.content_digest) else "not_an_extension"}

    assert prior is not None
    findings: list[ValidationFindingV1] = []
    if not candidate.content_digest:
        findings.append(_error("evidence_not_sealed", "evidence bundle has no content digest"))

    new_annotations = candidate.annotations[len(prior.annotations) :]
    new_receipts = candidate.receipts[len(prior.receipts) :]

    # Identity collisions across the boundary; attach_many refuses these too.
    prior_annotation_ids = {item.annotation_id: item for item in prior.annotations}
    for item in new_annotations:
        if item.annotation_id in prior_annotation_ids:
            findings.append(_error("duplicate_evidence_record_id", "duplicate annotation id in evidence bundle", item.annotation_id))
    prior_receipt_ids = {item.receipt_id for item in prior.receipts}
    for item in new_receipts:
        if item.receipt_id in prior_receipt_ids:
            findings.append(_error("duplicate_evidence_record_id", "duplicate receipt id in evidence bundle", item.receipt_id))

    carried = _referenced_prior_annotations(new_annotations, prior_annotation_ids)
    delta = replace(
        candidate,
        annotations=tuple(carried) + tuple(new_annotations),
        receipts=tuple(new_receipts),
        content_digest="",
    ).sealed()
    delta_findings, _, _ = validate_evidence(index.view, delta)
    findings.extend(delta_findings)
    report = {
        "mode": "incremental",
        "new_annotations": len(new_annotations),
        "new_receipts": len(new_receipts),
        "carried_annotations": len(carried),
        "prior_annotations": len(prior.annotations),
    }
    if not any(str(item.severity) == Severity.ERROR for item in findings):
        index.mark_bundle_verified(candidate.content_digest)
    return findings, report


__all__ = ["validate_appended_evidence"]
