"""One generic inspector that works unchanged against any V5 bundle.

The inspector reads only the sealed document and evidence bundle — never a
consumer-specific record — which is how Push 1 proves the two very different
acceptance bundles share one authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical import content_digest, readable_json
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.selectors import resolve_selector
from ..models.spans import SpanKind
from ..store.bundle import LocalTraceBundle
from ..validation.rehydrate import evidence_bundle_from_payload, rehydrate_trace


@dataclass(frozen=True, slots=True)
class InspectedBundle:
    trace: TraceDocumentV5
    evidence: TraceEvidenceBundleV5 | None


def load_bundle(root: Path) -> list[InspectedBundle]:
    """Load every sealed trace in a bundle together with its evidence."""

    bundle = LocalTraceBundle(root)
    manifest = bundle.read_manifest()
    evidence_revisions: dict[str, list[TraceEvidenceBundleV5]] = {}
    for entry in manifest.get("evidence") or []:
        payload = bundle.read_evidence(entry["bundle_digest"])
        record = evidence_bundle_from_payload(payload)
        if record.bundle_id != entry["bundle_id"]:
            raise ValueError(
                "evidence manifest entry bundle_id does not match its sealed record"
            )
        if record.content_digest != entry["bundle_digest"]:
            raise ValueError(
                "evidence manifest entry digest does not match its sealed record"
            )
        if record.trace_ref.content_digest != entry["trace_digest"]:
            raise ValueError(
                "evidence manifest entry trace digest does not match its sealed record"
            )
        evidence_revisions.setdefault(record.trace_ref.content_digest, []).append(record)
    evidence_by_digest = {
        trace_digest: select_evidence_head(tuple(records))
        for trace_digest, records in evidence_revisions.items()
    }
    inspected: list[InspectedBundle] = []
    for entry in manifest.get("traces") or []:
        document = rehydrate_trace(bundle.read_trace(entry["trace_digest"]))
        inspected.append(
            InspectedBundle(
                trace=document,
                evidence=evidence_by_digest.get(document.content_digest),
            )
        )
    return inspected


def select_evidence_head(
    records: tuple[TraceEvidenceBundleV5, ...],
) -> TraceEvidenceBundleV5:
    """Select the unique append-only revision head without trusting list order."""

    if not records:
        raise ValueError("cannot select an evidence head from an empty revision set")
    by_digest: dict[str, TraceEvidenceBundleV5] = {}
    by_id: dict[str, str] = {}
    trace_refs = {
        (record.trace_ref.trace_id, record.trace_ref.content_digest)
        for record in records
    }
    if len(trace_refs) != 1:
        raise ValueError("evidence revision set spans more than one sealed trace")
    for record in records:
        if not record.content_digest:
            raise ValueError(f"evidence revision {record.bundle_id!r} is not sealed")
        if content_digest(record) != record.content_digest:
            raise ValueError(
                f"evidence revision {record.bundle_id!r} has a content digest mismatch"
            )
        if record.content_digest in by_digest:
            raise ValueError(
                f"duplicate evidence revision digest: {record.content_digest}"
            )
        prior_digest = by_id.get(record.bundle_id)
        if prior_digest is not None:
            raise ValueError(
                f"duplicate evidence revision id {record.bundle_id!r} names "
                "different revisions"
            )
        by_digest[record.content_digest] = record
        by_id[record.bundle_id] = record.content_digest

    parent_by_child: dict[str, str] = {}
    children: dict[str, list[str]] = {digest: [] for digest in by_digest}
    for digest, record in by_digest.items():
        parent_id = _metadata_text(record, "supersedes_bundle_id")
        parent_digest = _metadata_text(record, "supersedes_bundle_digest")
        if bool(parent_id) != bool(parent_digest):
            raise ValueError(
                f"evidence revision {record.bundle_id!r} has an incomplete parent link"
            )
        if not parent_digest:
            continue
        if parent_digest == digest:
            raise ValueError(
                f"evidence revision {record.bundle_id!r} supersedes itself"
            )
        parent = by_digest.get(parent_digest)
        if parent is None:
            raise ValueError(
                f"evidence revision {record.bundle_id!r} names a missing parent "
                f"{parent_digest!r}"
            )
        if parent.bundle_id != parent_id:
            raise ValueError(
                f"evidence revision {record.bundle_id!r} parent id/digest disagree"
            )
        _require_append_only_revision(parent, record)
        parent_by_child[digest] = parent_digest
        children[parent_digest].append(digest)

    roots = sorted(set(by_digest) - set(parent_by_child))
    if len(roots) != 1:
        raise ValueError(
            f"evidence revision graph has {len(roots)} roots; exactly one is required"
        )
    forked = sorted(digest for digest, child_ids in children.items() if len(child_ids) > 1)
    if forked:
        raise ValueError(
            f"evidence revision graph forks at {', '.join(forked)}"
        )

    visited: set[str] = set()
    cursor = roots[0]
    while True:
        if cursor in visited:
            raise ValueError("evidence revision graph contains a cycle")
        visited.add(cursor)
        successors = children[cursor]
        if not successors:
            break
        cursor = successors[0]
    if visited != set(by_digest):
        raise ValueError("evidence revision graph is disconnected or cyclic")
    return by_digest[cursor]


_EVIDENCE_COLLECTIONS = (
    ("criteria", "criterion", "criterion_id"),
    ("rubrics", "rubric", "rubric_id"),
    ("verifier_definitions", "verifier_definition", "verifier_id"),
    ("annotator_definitions", "annotator_definition", "annotator_id"),
    ("reward_definitions", "reward_definition", "reward_id"),
    ("annotations", "annotation", "annotation_id"),
    ("verifier_results", "verifier_result", "verifier_result_id"),
    ("reward_records", "reward_record", "reward_record_id"),
    ("reward_aggregations", "reward_aggregation", "aggregation_id"),
    ("evaluation_results", "evaluation_result", "evaluation_id"),
    ("benchmark_verdicts", "benchmark_verdict", "verdict_id"),
    ("receipts", "receipt", "receipt_id"),
    ("artifacts", "artifact", "artifact_id"),
)


def _metadata_text(record: TraceEvidenceBundleV5, key: str) -> str:
    value = record.metadata.get(key)
    return str(value).strip() if value is not None else ""


def _require_append_only_revision(
    parent: TraceEvidenceBundleV5,
    child: TraceEvidenceBundleV5,
) -> None:
    added = 0
    appended: set[tuple[str, str, str]] = set()
    for field_name, kind, id_field in _EVIDENCE_COLLECTIONS:
        before = tuple(getattr(parent, field_name))
        after = tuple(getattr(child, field_name))
        if after[: len(before)] != before:
            raise ValueError(
                f"evidence revision {child.bundle_id!r} rewrites {field_name}"
            )
        new_records = after[len(before) :]
        added += len(new_records)
        appended.update(
            (
                kind,
                str(getattr(record, id_field)),
                content_digest(record),
            )
            for record in new_records
        )
    if added <= 0:
        raise ValueError(
            f"evidence revision {child.bundle_id!r} appends no evidence records"
        )
    parent_metadata = {
        key: value
        for key, value in parent.metadata.items()
        if key
        not in {
            "supersedes_bundle_id",
            "supersedes_bundle_digest",
            "revision_records",
        }
    }
    child_metadata = {
        key: value
        for key, value in child.metadata.items()
        if key
        not in {
            "supersedes_bundle_id",
            "supersedes_bundle_digest",
            "revision_records",
        }
    }
    if any(child_metadata.get(key) != value for key, value in parent_metadata.items()):
        raise ValueError(
            f"evidence revision {child.bundle_id!r} rewrites inherited metadata"
        )
    if child.schema_version != parent.schema_version:
        raise ValueError(
            f"evidence revision {child.bundle_id!r} changes the evidence schema"
        )
    declared = child.metadata.get("revision_records")
    if not isinstance(declared, list):
        raise ValueError(
            f"evidence revision {child.bundle_id!r} has no append receipt"
        )
    declared_records = {
        (
            str(item.get("kind") or ""),
            str(item.get("record_id") or ""),
            str(item.get("record_digest") or ""),
        )
        for item in declared
        if isinstance(item, dict)
    }
    if len(declared) != added or declared_records != appended:
        raise ValueError(
            f"evidence revision {child.bundle_id!r} append receipt is incorrect"
        )


def summarize(inspected: InspectedBundle) -> dict[str, Any]:
    """The generic view: actors, calls, events, criteria, rewards, artifacts, coverage."""

    document = inspected.trace
    evidence = inspected.evidence
    model_calls = document.spans_of_kind(SpanKind.MODEL_CALL.value)
    event_types: dict[str, int] = {}
    for event in document.events:
        key = str(event.event_type)
        event_types[key] = event_types.get(key, 0) + 1

    summary: dict[str, Any] = {
        "trace_id": document.trace_id,
        "trace_digest": document.content_digest,
        "trace_kind": str(document.trace_kind),
        "schema_version": document.schema_version,
        "lifecycle": {
            "status": str(document.lifecycle.status),
            "started_at": document.lifecycle.started_at,
            "ended_at": document.lifecycle.ended_at,
        },
        "actors": [
            {
                "actor_id": item.actor_id,
                "kind": str(item.kind),
                "display_name": item.display_name,
                "role": item.role,
            }
            for item in document.actors
        ],
        "sessions": [
            {
                "session_id": item.session_id,
                "actor_id": item.actor_id,
                "status": str(item.status),
                "coverage": item.coverage.to_dict(),
            }
            for item in document.sessions
        ],
        "model_calls": [
            {
                "span_id": item.span_id,
                "call_index": item.detail.get("call_index"),
                "model": item.detail.get("model"),
                "streaming": item.detail.get("streaming"),
                "http_status": item.detail.get("http_status"),
                "status": str(item.status),
                "usage": item.usage.to_dict() if item.usage else None,
                "input_messages": len(item.input_message_ids),
                "output_messages": len(item.output_message_ids),
            }
            for item in sorted(
                model_calls, key=lambda span: int(span.detail.get("call_index") or 0)
            )
        ],
        "event_counts": dict(sorted(event_types.items())),
        "environment_events": [
            {
                "event_id": item.event_id,
                "event_type": str(item.event_type),
                "sequence": item.order.chronological_sequence,
                "caused_by": list(item.caused_by_event_ids),
                "payload_keys": sorted(item.payload.keys()),
            }
            for item in document.events
            if str(item.event_type).startswith("environment.")
        ],
        "artifacts": [
            {
                "artifact_id": item.artifact_id,
                "role": str(item.role),
                "media_type": item.media_type,
                "size_bytes": item.size_bytes,
                "digest": item.digest,
                "logical_name": item.logical_name,
            }
            for item in document.artifacts
        ],
        "usage": document.usage.to_dict(),
        "completeness": document.completeness.to_dict(),
        "aliases": [
            {"namespace": str(item.namespace), "value": item.value, "target": item.target_id}
            for item in document.aliases
        ],
    }

    if evidence is None:
        summary["evidence"] = None
        return summary

    resolutions = [resolve_selector(document, selector) for selector in evidence.selectors()]
    unavailable_resolutions = [
        resolve_selector(document, selector)
        for selector in evidence.unavailable_selectors()
    ]
    summary["evidence"] = {
        "bundle_id": evidence.bundle_id,
        "bundle_digest": evidence.content_digest,
        "criteria": [
            {"criterion_id": item.criterion_id, "role": str(item.role), "weight": item.weight}
            for rubric in evidence.rubrics
            for item in rubric.criteria
        ],
        "verifier_results": [
            {
                "verifier_result_id": item.verifier_result_id,
                "verifier_id": item.verifier_id,
                "execution_status": str(item.execution_status),
                "verification_status": str(item.verification_status),
                "grounding": str(item.grounding),
                "score": item.score,
                "passed": item.passed,
                "criteria": [
                    {
                        "criterion_id": criterion.criterion_id,
                        "score": criterion.score,
                        "verdict": criterion.verdict,
                        "evidence": len(criterion.evidence),
                    }
                    for criterion in item.criterion_results
                ],
            }
            for item in evidence.verifier_results
        ],
        "rewards": [
            {
                "reward_record_id": item.reward_record_id,
                "reward_id": item.reward_id,
                "value": item.value,
                "components": item.components,
                "provenance": item.provenance,
                "position": item.position,
            }
            for item in evidence.reward_records
        ],
        "reward_aggregations": [
            {
                "aggregation_id": item.aggregation_id,
                "reward_id": item.reward_id,
                "value": item.value,
                "inputs": len(item.input_reward_record_ids),
            }
            for item in evidence.reward_aggregations
        ],
        "annotator_definitions": [
            {
                "annotator_id": item.annotator_id,
                "name": item.name,
                "version": item.version,
                "purpose": item.purpose,
                "taxonomy": list(item.taxonomy),
                "grounding_requirement": str(item.grounding_requirement),
                "unavailable_evidence_behavior": str(
                    item.unavailable_evidence_behavior
                ),
                "confidence_semantics": str(item.confidence_semantics),
                "task_kind": (
                    str(item.output_contract.task_kind)
                    if item.output_contract is not None
                    else None
                ),
                "annotation_types": (
                    list(item.output_contract.annotation_types)
                    if item.output_contract is not None
                    else []
                ),
            }
            for item in evidence.annotator_definitions
        ],
        "annotations": [
            {
                "annotation_id": item.annotation_id,
                "annotator_id": item.annotator_id,
                "annotation_type": item.annotation_type,
                "labels": list(item.labels),
                "grounding": str(item.grounding),
                "confidence": item.confidence,
                "status": (
                    str(item.status) if item.status is not None else None
                ),
                "review_state": (
                    str(item.review_state)
                    if item.review_state is not None
                    else None
                ),
                "producer": {
                    "kind": str(item.producer.kind),
                    "name": item.producer.name,
                    "version": item.producer.version,
                    "model": item.producer.model,
                    "config_digest": item.producer.config_digest,
                },
                "trace_body_read": (
                    item.inspection.trace_body_read
                    if item.inspection is not None
                    else None
                ),
                "inspected_projection": (
                    item.inspection.projection_id
                    if item.inspection is not None
                    else item.inspected_projection
                ),
                "projection_losses": (
                    len(item.inspection.losses)
                    if item.inspection is not None
                    else 0
                ),
                "annotator_execution_trace": (
                    {
                        "trace_id": item.annotator_execution_trace_id,
                        "trace_digest": item.annotator_execution_trace_digest,
                    }
                    if item.annotator_execution_trace_id is not None
                    else None
                ),
                "evidence_gaps": (
                    len(item.unavailable_evidence.gaps)
                    if item.unavailable_evidence is not None
                    else 0
                ),
                "derivation": (
                    {
                        "kind": str(item.derivation.kind),
                        "method": item.derivation.method,
                        "source_annotation_ids": list(
                            item.derivation.source_annotation_ids
                        ),
                        "agreement": item.derivation.agreement,
                        "dissenting_annotation_ids": list(
                            item.derivation.dissenting_annotation_ids
                        ),
                    }
                    if item.derivation is not None
                    else None
                ),
                "revision": item.revision,
                "supersedes_id": item.supersedes_id,
                "target": {
                    "kind": str(item.target.kind),
                    "entity_id": item.target.entity_id,
                },
            }
            for item in evidence.annotations
        ],
        "evaluation_results": [
            {
                "evaluation_id": item.evaluation_id,
                "aggregate_score": item.aggregate_score,
                "metrics": item.objective_metrics,
                "status": str(item.execution_status),
            }
            for item in evidence.evaluation_results
        ],
        "selectors_resolved": sum(1 for item in resolutions if item.resolved),
        "selectors_failed": sum(1 for item in resolutions if not item.resolved),
        "unavailable_selectors": len(unavailable_resolutions),
        "unavailable_selectors_now_resolved": sum(
            1 for item in unavailable_resolutions if item.resolved
        ),
    }
    return summary


def render(root: Path) -> str:
    """Human-readable inspection of every trace in a bundle."""

    return readable_json([summarize(item) for item in load_bundle(root)])


__all__ = [
    "InspectedBundle",
    "load_bundle",
    "render",
    "select_evidence_head",
    "summarize",
]
