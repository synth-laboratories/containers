"""Append-only evidence attachment and evaluator execution helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable

from .canonical import content_digest, record_id, utc_now
from .models.artifacts import ArtifactRefV5
from .models.document import TraceDocumentV5
from .models.evidence import TraceEvidenceBundleV5, TraceRefV5
from .models.selectors import TraceSelectorV1
from .models.standards import (
    AnnotationV1,
    BenchmarkVerdictV1,
    CriterionDefinitionV1,
    EvaluationResultV1,
    ReceiptV1,
    RewardAggregationV1,
    RewardDefinitionV1,
    RewardRecordV1,
    RubricDefinitionV2,
    TraceAnnotatorDefinitionV1,
    VerifierDefinitionV1,
    VerifierResultV2,
)


_COLLECTIONS: dict[str, tuple[str, type[Any], str]] = {
    "criterion": ("criteria", CriterionDefinitionV1, "criterion_id"),
    "rubric": ("rubrics", RubricDefinitionV2, "rubric_id"),
    "verifier_definition": (
        "verifier_definitions",
        VerifierDefinitionV1,
        "verifier_id",
    ),
    "annotator_definition": (
        "annotator_definitions",
        TraceAnnotatorDefinitionV1,
        "annotator_id",
    ),
    "reward_definition": ("reward_definitions", RewardDefinitionV1, "reward_id"),
    "annotation": ("annotations", AnnotationV1, "annotation_id"),
    "verifier_result": (
        "verifier_results",
        VerifierResultV2,
        "verifier_result_id",
    ),
    "reward_record": ("reward_records", RewardRecordV1, "reward_record_id"),
    "reward_aggregation": (
        "reward_aggregations",
        RewardAggregationV1,
        "aggregation_id",
    ),
    "evaluation_result": (
        "evaluation_results",
        EvaluationResultV1,
        "evaluation_id",
    ),
    "benchmark_verdict": (
        "benchmark_verdicts",
        BenchmarkVerdictV1,
        "verdict_id",
    ),
    "receipt": ("receipts", ReceiptV1, "receipt_id"),
    "artifact": ("artifacts", ArtifactRefV5, "artifact_id"),
}


def new_evidence_bundle(document: TraceDocumentV5) -> TraceEvidenceBundleV5:
    _require_sealed_trace(document)
    created_at = utc_now()
    return TraceEvidenceBundleV5(
        bundle_id=record_id(
            "evb",
            kind="trace_evidence",
            scope=(document.trace_id,),
            key={"trace_digest": document.content_digest, "created_at": created_at},
        ),
        trace_ref=TraceRefV5(
            trace_id=document.trace_id,
            content_digest=document.content_digest,
        ),
        created_at=created_at,
    ).sealed()


def attach(
    bundle: TraceEvidenceBundleV5,
    *,
    kind: str,
    record: Any,
) -> TraceEvidenceBundleV5:
    """Return a new sealed evidence bundle; the prior bundle remains immutable."""

    return attach_many(bundle, records=((kind, record),))


def attach_many(
    bundle: TraceEvidenceBundleV5,
    *,
    records: tuple[tuple[str, Any], ...],
) -> TraceEvidenceBundleV5:
    """Append an ordered batch as one immutable evidence revision.

    A batch keeps a native evaluator import atomic: readers either select the old
    evidence head or the revision containing every linked definition and result.
    """

    if not records:
        raise ValueError("evidence revision must append at least one record")
    _require_sealed_bundle(bundle)
    updated: dict[str, list[Any]] = {}
    revision_records: list[dict[str, str]] = []
    for kind, record in records:
        collection = _COLLECTIONS.get(kind)
        if collection is None:
            raise ValueError(f"unsupported evidence kind: {kind}")
        field_name, expected_type, id_field = collection
        if not isinstance(record, expected_type):
            raise TypeError(
                f"evidence kind {kind!r} requires {expected_type.__name__}, "
                f"got {type(record).__name__}"
            )
        _require_sealed_record(record)
        _require_selector_trace(bundle, record)
        current = updated.setdefault(field_name, list(getattr(bundle, field_name)))
        record_identity = str(getattr(record, id_field))
        if any(str(getattr(item, id_field)) == record_identity for item in current):
            raise ValueError(f"duplicate {kind} id: {record_identity}")
        current.append(record)
        revision_records.append(
            {
                "kind": kind,
                "record_id": record_identity,
                "record_digest": content_digest(record),
            }
        )

    return replace(
        bundle,
        bundle_id=record_id(
            "evb",
            kind="trace_evidence_revision",
            scope=(bundle.trace_ref.trace_id,),
            key={
                "prior": bundle.content_digest,
                "records": revision_records,
            },
        ),
        created_at=utc_now(),
        content_digest="",
        metadata={
            **bundle.metadata,
            "supersedes_bundle_id": bundle.bundle_id,
            "supersedes_bundle_digest": bundle.content_digest,
            "revision_records": revision_records,
        },
        **{field_name: tuple(items) for field_name, items in updated.items()},
    ).sealed()


def _require_sealed_bundle(bundle: TraceEvidenceBundleV5) -> None:
    if not bundle.content_digest:
        raise ValueError("evidence attachment requires a sealed bundle")
    recomputed = content_digest(bundle)
    if recomputed != bundle.content_digest:
        raise ValueError("evidence bundle content digest does not match its content")
    if not bundle.trace_ref.trace_id or not bundle.trace_ref.content_digest:
        raise ValueError("evidence bundle must identify one sealed trace")


def _require_sealed_trace(document: TraceDocumentV5) -> None:
    if not document.content_digest:
        raise ValueError("evidence can only attach to a sealed trace")
    if content_digest(document) != document.content_digest:
        raise ValueError("trace content digest does not match its content")


def _require_sealed_record(record: Any) -> None:
    if isinstance(record, ArtifactRefV5):
        if not record.digest:
            raise ValueError("artifact evidence requires a content digest")
        return
    stored = getattr(record, "content_digest", "")
    if not stored:
        raise ValueError(f"{type(record).__name__} must be sealed before attachment")
    if content_digest(record) != stored:
        raise ValueError(f"{type(record).__name__} content digest does not match its content")


def _require_selector_trace(bundle: TraceEvidenceBundleV5, record: Any) -> None:
    for selector in _selectors(record):
        if selector.trace_id != bundle.trace_ref.trace_id:
            raise ValueError("evidence record selector refers to a different trace id")
        if selector.trace_digest != bundle.trace_ref.content_digest:
            raise ValueError("evidence record selector refers to a different trace digest")


def _selectors(value: Any) -> tuple[TraceSelectorV1, ...]:
    found: list[TraceSelectorV1] = []

    def visit(item: Any) -> None:
        if isinstance(item, TraceSelectorV1):
            found.append(item)
        elif is_dataclass(item):
            for field_info in fields(item):
                visit(getattr(item, field_info.name))
        elif isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return tuple(found)


def evaluate(
    document: TraceDocumentV5,
    evaluator: Callable[[TraceDocumentV5], Any],
    *,
    bundle: TraceEvidenceBundleV5 | None = None,
    result_kind: str = "evaluation_result",
) -> TraceEvidenceBundleV5:
    """Run an evaluator against the sealed authority and append its typed result."""

    _require_sealed_trace(document)
    evidence = bundle or new_evidence_bundle(document)
    if evidence.trace_ref.trace_id != document.trace_id:
        raise ValueError("evidence bundle refers to a different trace id")
    if evidence.trace_ref.content_digest != document.content_digest:
        raise ValueError("evidence bundle refers to a different trace digest")
    return attach(evidence, kind=result_kind, record=evaluator(document))


__all__ = ["attach", "attach_many", "evaluate", "new_evidence_bundle"]
