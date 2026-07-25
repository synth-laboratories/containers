"""One generic inspector that works unchanged against any V5 bundle.

The inspector reads only the sealed document and evidence bundle — never a
consumer-specific record — which is how Push 1 proves the two very different
acceptance bundles share one authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..canonical import readable_json
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
    evidence_by_digest: dict[str, TraceEvidenceBundleV5] = {}
    for entry in manifest.get("evidence") or []:
        payload = bundle.read_evidence(entry["bundle_digest"])
        record = evidence_bundle_from_payload(payload)
        evidence_by_digest[record.trace_ref.content_digest] = record
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
        "annotations": [
            {
                "annotation_id": item.annotation_id,
                "annotator_id": item.annotator_id,
                "labels": list(item.labels),
                "grounding": str(item.grounding),
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
    }
    return summary


def render(root: Path) -> str:
    """Human-readable inspection of every trace in a bundle."""

    return readable_json([summarize(item) for item in load_bundle(root)])


__all__ = ["InspectedBundle", "load_bundle", "render", "summarize"]
