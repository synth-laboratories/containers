"""Agreement, consensus, and adjudication over immutable source annotations.

Repeated jobs produce independent ``AnnotationV1`` records. Nothing here edits
them: consensus and adjudication append *derived* records that name every source
they consumed, so disagreement stays visible instead of being averaged away.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Sequence

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now
from ..models.actors import Visibility
from ..models.selectors import GroundingStatus, TraceSelectorV1
from ..models.standards import (
    AnnotationDerivationKind,
    AnnotationDerivationV1,
    AnnotationInspectionSource,
    AnnotationInspectionV1,
    AnnotationReviewState,
    AnnotationStatus,
    AnnotationV1,
    ProducerKind,
    ProducerRefV1,
    TraceAnnotatorDefinitionV1,
)


AGREEMENT_SCHEMA_VERSION = "synth.annotation-agreement.v1"


def _group_key(annotation: AnnotationV1) -> tuple[str, str]:
    return content_digest(annotation.target), annotation.annotation_type


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass(frozen=True, slots=True)
class AgreementGroupV1(JsonDataclassMixin):
    target_digest: str
    annotation_type: str
    annotation_ids: tuple[str, ...]
    applied_count: int
    abstained_count: int
    label_agreement: float | None
    majority_labels: tuple[str, ...]
    dissenting_annotation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgreementReportV1(JsonDataclassMixin):
    annotator_id: str
    annotation_count: int
    group_count: int
    mean_label_agreement: float | None
    abstention_rate: float
    groups: tuple[AgreementGroupV1, ...] = ()
    schema_version: str = AGREEMENT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


def agreement(
    annotations: Sequence[AnnotationV1],
    *,
    majority_threshold: float = 0.5,
) -> AgreementReportV1:
    """Pairwise label agreement per (target, annotation_type) group."""

    if not annotations:
        raise ValueError("agreement requires at least one annotation")
    annotator_ids = {item.annotator_id for item in annotations}
    if len(annotator_ids) != 1:
        raise ValueError("agreement compares repeats of one annotator; got " + ", ".join(sorted(annotator_ids)))
    grouped: dict[tuple[str, str], list[AnnotationV1]] = defaultdict(list)
    for item in annotations:
        grouped[_group_key(item)].append(item)
    groups: list[AgreementGroupV1] = []
    agreements: list[float] = []
    abstained_total = 0
    for (target_digest, annotation_type), members in sorted(grouped.items()):
        applied = [
            item
            for item in members
            if (item.status is None or str(item.status) == AnnotationStatus.APPLIED)
        ]
        abstained = [item for item in members if item not in applied]
        abstained_total += len(abstained)
        label_agreement: float | None = None
        if len(applied) >= 2:
            pair_scores = [
                _jaccard(left.labels, right.labels)
                for left, right in combinations(applied, 2)
            ]
            label_agreement = sum(pair_scores) / len(pair_scores)
            agreements.append(label_agreement)
        counts: dict[str, int] = defaultdict(int)
        for item in applied:
            for label in item.labels:
                counts[label] += 1
        majority = tuple(
            sorted(
                label
                for label, count in counts.items()
                if applied and count / len(applied) > majority_threshold
            )
        )
        dissent = tuple(
            item.annotation_id
            for item in applied
            if set(item.labels) != set(majority)
        )
        groups.append(
            AgreementGroupV1(
                target_digest=target_digest,
                annotation_type=annotation_type,
                annotation_ids=tuple(item.annotation_id for item in members),
                applied_count=len(applied),
                abstained_count=len(abstained),
                label_agreement=label_agreement,
                majority_labels=majority,
                dissenting_annotation_ids=dissent,
            )
        )
    return AgreementReportV1(
        annotator_id=next(iter(annotator_ids)),
        annotation_count=len(annotations),
        group_count=len(groups),
        mean_label_agreement=(sum(agreements) / len(agreements)) if agreements else None,
        abstention_rate=abstained_total / len(annotations),
        groups=tuple(groups),
    )


def _derived_base(
    *,
    definition: TraceAnnotatorDefinitionV1,
    sources: Sequence[AnnotationV1],
    producer: ProducerRefV1,
    annotation_id: str,
    target: TraceSelectorV1,
    annotation_type: str,
    labels: tuple[str, ...],
    payload: dict[str, Any],
    confidence: float | None,
    rationale: str,
    evidence: tuple[TraceSelectorV1, ...],
    derivation: AnnotationDerivationV1,
    visibility: Visibility | str,
) -> AnnotationV1:
    return AnnotationV1(
        annotation_id=annotation_id,
        annotator_id=definition.annotator_id,
        annotator_version=definition.version,
        annotator_digest=definition.content_digest,
        target=target,
        annotation_type=annotation_type,
        labels=labels,
        author_kind=str(producer.kind),
        producer=producer,
        created_at=utc_now(),
        grounding=GroundingStatus.GROUNDED if evidence else GroundingStatus.UNINSPECTED,
        payload=payload,
        confidence=confidence,
        rationale=rationale,
        evidence=evidence,
        visibility=visibility,
        status=AnnotationStatus.APPLIED,
        review_state=AnnotationReviewState.UNREVIEWED,
        inspection=AnnotationInspectionV1(
            source=AnnotationInspectionSource.TRACE_AUTHORITY,
            trace_body_read=True,
        ),
        derivation=derivation,
    ).sealed()


def _consensus_payload(
    definition: TraceAnnotatorDefinitionV1, applied: Sequence[AnnotationV1]
) -> dict[str, Any]:
    """Aggregate the sources' typed payload fields so the record passes the
    annotator's own output contract (required fields must be present).

    Per schema field: numeric kinds take the median (integer-preserving),
    everything else takes the modal value (ties -> earliest source). Fields no
    source provided are omitted; unknown fields are never copied.
    """

    contract = definition.output_contract
    schema = contract.payload_schema if contract is not None else None
    if schema is None:
        return {}
    payload: dict[str, Any] = {}
    for spec in schema.fields:
        values = [item.payload[spec.field_name] for item in applied if spec.field_name in item.payload]
        values = [value for value in values if value is not None]
        if not values:
            continue
        kind = str(spec.value_kind)
        if kind in ("integer", "number") and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            ordered = sorted(values)
            mid = len(ordered) // 2
            median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
            if kind == "integer" or all(isinstance(v, int) for v in values):
                median = int(round(median))
            payload[spec.field_name] = median
            continue
        counts: dict[str, int] = defaultdict(int)
        first: dict[str, Any] = {}
        order: list[str] = []
        for value in values:
            key = json.dumps(value, sort_keys=True, default=str)
            if key not in first:
                first[key] = value
                order.append(key)
            counts[key] += 1
        best = max(order, key=lambda key: (counts[key], -order.index(key)))
        payload[spec.field_name] = first[best]
    return payload


def consensus_annotation(
    sources: Sequence[AnnotationV1],
    *,
    definition: TraceAnnotatorDefinitionV1,
    majority_threshold: float = 0.5,
    producer: ProducerRefV1 | None = None,
) -> AnnotationV1 | None:
    """Majority-label consensus over applied repeats of one target.

    Returns ``None`` when fewer than two applied sources exist or no label clears
    the threshold; consensus is never manufactured from a single voice.
    """

    applied = [
        item
        for item in sources
        if (item.status is None or str(item.status) == AnnotationStatus.APPLIED)
    ]
    if len(applied) < 2:
        return None
    keys = {_group_key(item) for item in applied}
    if len(keys) != 1:
        raise ValueError("consensus sources must share one target and annotation type")
    if any(item.annotator_id != definition.annotator_id for item in applied):
        raise ValueError("consensus sources must come from the given annotator definition")
    counts: dict[str, int] = defaultdict(int)
    for item in applied:
        for label in item.labels:
            counts[label] += 1
    majority = tuple(
        sorted(label for label, count in counts.items() if count / len(applied) > majority_threshold)
    )
    if not majority:
        return None
    contract = definition.output_contract
    if contract is not None and contract.allowed_producer_kinds and not any(
        str(kind) == ProducerKind.COMPOSITE for kind in contract.allowed_producer_kinds
    ):
        raise ValueError(
            f"annotator {definition.annotator_id} does not allow composite producers; "
            "consensus records cannot be attached under it"
        )
    report = agreement(applied, majority_threshold=majority_threshold)
    group = report.groups[0]
    evidence: dict[str, TraceSelectorV1] = {}
    for item in applied:
        for selector in item.evidence:
            evidence.setdefault(content_digest(selector), selector)
    source_ids = tuple(item.annotation_id for item in applied)
    derivation = AnnotationDerivationV1(
        kind=AnnotationDerivationKind.CONSENSUS,
        source_annotation_ids=source_ids,
        method=f"majority_labels>{majority_threshold:g}",
        agreement=group.label_agreement,
        dissenting_annotation_ids=group.dissenting_annotation_ids,
    )
    return _derived_base(
        definition=definition,
        sources=applied,
        producer=producer
        or ProducerRefV1(kind=ProducerKind.COMPOSITE, name="synth.annotation.consensus", version="1"),
        annotation_id=record_id(
            "ann",
            kind="annotation_consensus",
            scope=(applied[0].target.trace_id,),
            key={"sources": sorted(source_ids), "threshold": majority_threshold},
        ),
        target=applied[0].target,
        annotation_type=applied[0].annotation_type,
        labels=majority,
        payload=_consensus_payload(definition, applied),
        # Agreement is the natural confidence of a majority record, but a
        # deterministic-semantics annotator may only carry confidence 1.0.
        confidence=(None if str(definition.confidence_semantics) == "deterministic" else group.label_agreement),
        rationale=(
            f"majority of {len(applied)} independent annotations; label counts "
            + ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
        ),
        evidence=tuple(evidence.values()),
        derivation=derivation,
        visibility=applied[0].visibility,
    )


def adjudication_annotation(
    sources: Sequence[AnnotationV1],
    *,
    definition: TraceAnnotatorDefinitionV1,
    producer: ProducerRefV1,
    labels: tuple[str, ...],
    rationale: str,
    evidence: tuple[TraceSelectorV1, ...],
    payload: dict[str, Any] | None = None,
    confidence: float | None = None,
    method: str = "arbiter",
    annotation_id: str | None = None,
) -> AnnotationV1:
    """A derived record that resolves disagreement by naming what it overruled."""

    if not sources:
        raise ValueError("adjudication requires at least one source annotation")
    keys = {_group_key(item) for item in sources}
    if len(keys) != 1:
        raise ValueError("adjudication sources must share one target and annotation type")
    source_ids = tuple(item.annotation_id for item in sources)
    dissent = tuple(item.annotation_id for item in sources if set(item.labels) != set(labels))
    derivation = AnnotationDerivationV1(
        kind=AnnotationDerivationKind.ADJUDICATION,
        source_annotation_ids=source_ids,
        method=method,
        dissenting_annotation_ids=dissent,
    )
    return _derived_base(
        definition=definition,
        sources=sources,
        producer=producer,
        annotation_id=annotation_id
        or record_id(
            "ann",
            kind="annotation_adjudication",
            scope=(sources[0].target.trace_id,),
            key={"sources": sorted(source_ids), "producer": producer.to_dict(), "labels": sorted(labels)},
        ),
        target=sources[0].target,
        annotation_type=sources[0].annotation_type,
        labels=tuple(labels),
        payload=dict(payload or {}),
        confidence=confidence,
        rationale=rationale,
        evidence=evidence,
        derivation=derivation,
        visibility=sources[0].visibility,
    )


__all__ = [
    "AGREEMENT_SCHEMA_VERSION",
    "AgreementGroupV1",
    "AgreementReportV1",
    "adjudication_annotation",
    "agreement",
    "consensus_annotation",
]
