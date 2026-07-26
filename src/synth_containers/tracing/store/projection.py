"""SQL-neutral catalog rows derived from sealed Trace V5 authorities.

The returned rows match the local SQLite and managed Turso catalog schemas. They
contain only rebuildable search facts; the sealed trace and evidence objects remain
the authority.
"""

from __future__ import annotations

from typing import Any

from ..canonical import canonical_text
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5


CATALOG_PROJECTION_SCHEMA_VERSION = "synth.trace-catalog-projection.v2"


def catalog_projection(
    record: TraceDocumentV5 | TraceEvidenceBundleV5,
) -> dict[str, Any]:
    """Project one sealed trace or evidence bundle into portable catalog rows."""

    if isinstance(record, TraceDocumentV5):
        return _trace_projection(record)
    if isinstance(record, TraceEvidenceBundleV5):
        return _evidence_projection(record)
    raise TypeError(f"unsupported catalog projection record: {type(record).__name__}")


def _trace_projection(document: TraceDocumentV5) -> dict[str, Any]:
    if not document.content_digest:
        raise ValueError("only sealed trace documents can be projected")
    digest = document.content_digest
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []

    def entity(
        entity_id: str,
        kind: str,
        *,
        owner_actor_id: str | None = None,
        owner_session_id: str | None = None,
        source_order: int | None = None,
        occurred_at: str | None = None,
        content_digest: str | None = None,
        facts: Any,
    ) -> None:
        entities.append(
            {
                "trace_digest": digest,
                "entity_id": entity_id,
                "kind": kind,
                "owner_actor_id": owner_actor_id,
                "owner_session_id": owner_session_id,
                "source_order": source_order,
                "occurred_at": occurred_at,
                "content_digest": content_digest,
                "facts": canonical_text(facts),
            }
        )

    def relationship(
        source: str,
        relation: str,
        target: str,
        source_order: int | None = None,
    ) -> None:
        relationships.append(
            {
                "trace_digest": digest,
                "source_entity_id": source,
                "relation": relation,
                "target_entity_id": target,
                "source_order": source_order,
            }
        )

    def add_alias(item: Any) -> None:
        aliases.append(
            {
                "trace_digest": digest,
                "namespace": str(item.namespace),
                "value": item.value,
                "target_id": item.target_id,
                "target_kind": item.target_kind,
            }
        )

    for actor in document.actors:
        entity(
            actor.actor_id,
            "actor",
            owner_actor_id=actor.actor_id,
            content_digest=actor.content_digest,
            facts={
                "kind": str(actor.kind),
                "name": actor.display_name,
                "role": actor.role,
                "subtype": actor.subtype,
                "model": actor.model,
                "provider": actor.provider,
                "task_id": actor.task_id,
                "workflow_id": actor.workflow_id,
                "visibility": str(actor.visibility),
                "trace_visibility": str(document.visibility),
            },
        )
        if actor.parent_actor_id:
            relationship(actor.parent_actor_id, "parent_of", actor.actor_id)
        for alias in actor.aliases:
            add_alias(alias)

    for session in document.sessions:
        entity(
            session.session_id,
            "session",
            owner_actor_id=session.actor_id,
            owner_session_id=session.session_id,
            occurred_at=session.started_at,
            content_digest=session.content_digest,
            facts={
                "status": str(session.status),
                "coverage": session.coverage.to_dict(),
                "attempt_id": session.attempt_id,
                "thread_id": session.thread_id,
                "workflow_id": session.workflow_id,
                "provider": session.provider,
                "trace_visibility": str(document.visibility),
            },
        )
        relationship(session.actor_id, "owns_session", session.session_id)
        if session.parent_session_id:
            relationship(session.parent_session_id, "parent_of", session.session_id)
        for alias in session.aliases:
            add_alias(alias)

    for index, span in enumerate(document.spans):
        entity(
            span.span_id,
            "span",
            owner_actor_id=span.actor_id,
            owner_session_id=span.session_id,
            source_order=index,
            occurred_at=span.started_at,
            content_digest=span.content_digest,
            facts={
                "span_kind": str(span.span_kind),
                "status": str(span.status),
                "turn_id": span.turn_id,
                "branch_id": span.branch_id,
                "workflow_address": span.workflow_address,
                "model": span.detail.get("model"),
                "provider": (
                    span.detail.get("provider")
                    or span.detail.get("provider_adapter")
                ),
                "detail": span.detail,
                "usage": span.usage.to_dict() if span.usage else None,
                "trace_visibility": str(document.visibility),
            },
        )
        if span.parent_span_id:
            relationship(span.parent_span_id, "parent_of", span.span_id, index)
        for parent in span.caused_by_span_ids:
            relationship(parent, "caused", span.span_id, index)
        for message_id in span.input_message_ids:
            relationship(message_id, "input_to", span.span_id, index)
        for message_id in span.output_message_ids:
            relationship(span.span_id, "produced_message", message_id, index)
        for artifact_id in span.artifact_ids:
            relationship(span.span_id, "produced_artifact", artifact_id, index)
        if span.workflow_address:
            aliases.append(
                {
                    "trace_digest": digest,
                    "namespace": "workflow_address",
                    "value": span.workflow_address,
                    "target_id": span.span_id,
                    "target_kind": "span",
                }
            )
        for alias in span.aliases:
            add_alias(alias)

    for index, event in enumerate(document.events):
        order = event.order.chronological_sequence
        source_order = index if order is None else order
        entity(
            event.event_id,
            "event",
            owner_actor_id=event.actor_id,
            owner_session_id=event.session_id,
            source_order=source_order,
            occurred_at=event.occurred_at,
            content_digest=event.content_digest,
            facts={
                "event_type": str(event.event_type),
                "status": str(event.status),
                "payload": event.payload,
                "trace_visibility": str(document.visibility),
            },
        )
        for parent in event.caused_by_event_ids:
            relationship(parent, "caused", event.event_id, source_order)
        if event.span_id:
            relationship(event.span_id, "contains", event.event_id, source_order)
        if event.message_id:
            relationship(event.event_id, "references_message", event.message_id, source_order)
        for artifact_id in event.artifact_ids:
            relationship(event.event_id, "references_artifact", artifact_id, source_order)
        for alias in event.aliases:
            add_alias(alias)

    for index, message in enumerate(document.messages):
        entity(
            message.message_id,
            "message",
            owner_actor_id=message.sender_actor_id,
            owner_session_id=message.session_id,
            source_order=index,
            occurred_at=message.occurred_at,
            content_digest=message.content_digest,
            facts={
                "role": str(message.role),
                "part_types": [str(part.type) for part in message.parts],
                "text_preview": message.text()[:512],
                "turn_id": message.turn_id,
                "thread_id": message.thread_id,
                "visibility": str(message.visibility),
                "trace_visibility": str(document.visibility),
            },
        )
        for predecessor in message.predecessor_message_ids:
            relationship(predecessor, "precedes", message.message_id, index)
        if message.produced_by_span_id:
            relationship(message.produced_by_span_id, "produced_message", message.message_id, index)
        if message.produced_by_event_id:
            relationship(message.produced_by_event_id, "produced_message", message.message_id, index)
        for recipient in message.recipient_actor_ids:
            relationship(message.message_id, "addressed_to", recipient, index)
        for alias in message.aliases:
            add_alias(alias)

    for index, branch in enumerate(document.branches):
        entity(
            branch.branch_id,
            "branch",
            owner_actor_id=branch.actor_id,
            owner_session_id=branch.session_id,
            source_order=index,
            facts={
                "reason": str(branch.reason),
                "head_message_id": branch.head_message_id,
                "fork_point_message_id": branch.fork_point_message_id,
                "trace_visibility": str(document.visibility),
            },
        )
        if branch.parent_branch_id:
            relationship(branch.parent_branch_id, "parent_of", branch.branch_id, index)
        if branch.head_message_id:
            relationship(branch.branch_id, "has_head", branch.head_message_id, index)

    for index, artifact in enumerate(document.artifacts):
        entity(
            artifact.artifact_id,
            "artifact",
            source_order=index,
            occurred_at=artifact.observed_at or artifact.produced_at,
            content_digest=artifact.digest,
            facts={
                "role": str(artifact.role),
                "media_type": artifact.media_type,
                "size_bytes": artifact.size_bytes,
                "logical_name": artifact.logical_name,
                "uri": artifact.uri,
                "completeness": str(artifact.completeness),
                "visibility": str(artifact.visibility),
                "trace_visibility": str(document.visibility),
            },
        )

    for index, error in enumerate(document.errors):
        entity(
            error.error_id,
            "error",
            owner_actor_id=error.actor_id,
            owner_session_id=error.session_id,
            source_order=index,
            occurred_at=error.observed_at,
            facts={
                "stage": error.stage,
                "component": error.component,
                "code": error.code,
                "message": error.message,
                "retryable": error.retryable,
                "terminal": error.terminal,
                "trace_visibility": str(document.visibility),
            },
        )
        if error.caused_by_error_id:
            relationship(error.caused_by_error_id, "caused", error.error_id, index)

    for alias in document.aliases:
        add_alias(alias)
    for link in document.links:
        relationship(document.trace_id, str(link.relation), link.target_id)

    return {
        "schema_version": CATALOG_PROJECTION_SCHEMA_VERSION,
        "documents": [
            {
                "trace_id": document.trace_id,
                "trace_digest": digest,
                "schema_version": document.schema_version,
                "trace_kind": str(document.trace_kind),
                "capture_id": document.capture.capture_id,
                "binding_digest": document.capture.binding_digest,
                "lifecycle_status": str(document.lifecycle.status),
                "capture_status": str(document.completeness.capture_status),
                "started_at": document.lifecycle.started_at,
                "ended_at": document.lifecycle.ended_at,
                "actor_count": len(document.actors),
                "span_count": len(document.spans),
                "event_count": len(document.events),
                "message_count": len(document.messages),
                "artifact_count": len(document.artifacts),
                "prompt_tokens": document.usage.prompt_tokens,
                "completion_tokens": document.usage.completion_tokens,
                "usage_provenance": str(document.usage.provenance),
                "task_id": document.identity.task_id,
                "run_id": document.identity.run_id,
                "correlation_id": document.identity.correlation_id,
            }
        ],
        "entities": entities,
        "relationships": relationships,
        "aliases": aliases,
        "evidence": [],
    }


def _evidence_projection(bundle: TraceEvidenceBundleV5) -> dict[str, Any]:
    if not bundle.content_digest:
        raise ValueError("only sealed evidence bundles can be projected")
    trace_digest = bundle.trace_ref.content_digest
    rows: list[dict[str, Any]] = []

    def evidence(
        record_kind: str,
        record_id: str,
        *,
        definition_id: str | None = None,
        subject_entity_id: str | None = None,
        grounding: str | None = None,
        value: float | None = None,
        verdict: str | None = None,
        facts: Any,
    ) -> None:
        rows.append(
            {
                "trace_digest": trace_digest,
                "bundle_id": bundle.bundle_id,
                "record_kind": record_kind,
                "record_id": record_id,
                "definition_id": definition_id,
                "subject_entity_id": subject_entity_id,
                "grounding": grounding,
                "value": value,
                "verdict": verdict,
                "facts": canonical_text(facts),
            }
        )

    for definition in bundle.annotator_definitions:
        output_contract = definition.output_contract
        evidence(
            "annotator_definition",
            definition.annotator_id,
            definition_id=definition.annotator_id,
            verdict=(
                str(output_contract.task_kind)
                if output_contract is not None
                else None
            ),
            facts={
                "name": definition.name,
                "purpose": definition.purpose,
                "version": definition.version,
                "taxonomy": list(definition.taxonomy),
                "supported_trace_schemas": list(
                    definition.supported_trace_schemas
                ),
                "required_subject_scope": definition.required_subject_scope,
                "grounding_requirement": str(definition.grounding_requirement),
                "minimum_evidence": definition.minimum_evidence,
                "unavailable_evidence_behavior": str(
                    definition.unavailable_evidence_behavior
                ),
                "confidence_semantics": str(definition.confidence_semantics),
                "output_contract": (
                    output_contract.to_dict()
                    if output_contract is not None
                    else None
                ),
            },
        )
    for annotation in bundle.annotations:
        inspection = annotation.inspection
        derivation = annotation.derivation
        evidence(
            "annotation",
            annotation.annotation_id,
            definition_id=annotation.annotator_id,
            subject_entity_id=annotation.target.entity_id,
            grounding=str(annotation.grounding),
            value=annotation.confidence,
            verdict=",".join(annotation.labels),
            facts={
                "annotation_type": annotation.annotation_type,
                "target_kind": str(annotation.target.kind),
                "labels": list(annotation.labels),
                "payload": annotation.payload,
                "rationale": annotation.rationale,
                "status": (
                    str(annotation.status)
                    if annotation.status is not None
                    else None
                ),
                "review_state": (
                    str(annotation.review_state)
                    if annotation.review_state is not None
                    else None
                ),
                "author_kind": str(annotation.author_kind),
                "producer": annotation.producer.to_dict(),
                "visibility": str(annotation.visibility),
                "abstention_reason": annotation.abstention_reason,
                "unavailable_evidence": (
                    annotation.unavailable_evidence.to_dict()
                    if annotation.unavailable_evidence is not None
                    else None
                ),
                "inspection": (
                    inspection.to_dict() if inspection is not None else None
                ),
                "derivation": (
                    derivation.to_dict() if derivation is not None else None
                ),
                "annotator_execution_trace_id": (
                    annotation.annotator_execution_trace_id
                ),
                "annotator_execution_trace_digest": (
                    annotation.annotator_execution_trace_digest
                ),
                "revision": annotation.revision,
                "supersedes_id": annotation.supersedes_id,
            },
        )
    for result in bundle.verifier_results:
        evidence(
            "verifier_result",
            result.verifier_result_id,
            definition_id=result.verifier_id,
            subject_entity_id=result.subject.entity_id,
            grounding=str(result.grounding),
            value=result.score,
            verdict=result.verdict,
            facts={
                "rubric_id": result.rubric_id,
                "execution_status": str(result.execution_status),
                "verification_status": str(result.verification_status),
                "passed": result.passed,
                "criterion_results": [
                    item.to_dict() for item in result.criterion_results
                ],
                "failure_modes": list(result.failure_modes),
            },
        )
    for reward in bundle.reward_records:
        evidence(
            "reward_record",
            reward.reward_record_id,
            definition_id=reward.reward_id,
            subject_entity_id=reward.subject.entity_id,
            grounding=str(reward.grounding),
            value=reward.value,
            verdict=reward.validity,
            facts={
                "components": reward.components,
                "position": reward.position,
                "actor_id": reward.actor_id,
                "session_id": reward.session_id,
                "source_result_ids": list(reward.source_result_ids),
            },
        )
    for aggregation in bundle.reward_aggregations:
        evidence(
            "reward_aggregation",
            aggregation.aggregation_id,
            definition_id=aggregation.reward_id,
            value=aggregation.value,
            verdict=aggregation.grouping,
            facts={
                "components": aggregation.components,
                "inputs": list(aggregation.input_reward_record_ids),
                "input_digests": list(aggregation.input_digests),
                "calculation": aggregation.calculation,
            },
        )
    for evaluation in bundle.evaluation_results:
        evidence(
            "evaluation_result",
            evaluation.evaluation_id,
            definition_id=evaluation.task_id,
            subject_entity_id=evaluation.subject.entity_id,
            value=evaluation.aggregate_score,
            verdict=str(evaluation.execution_status),
            facts={
                "suite": evaluation.suite,
                "benchmark": evaluation.benchmark,
                "task_id": evaluation.task_id,
                "instance_id": evaluation.instance_id,
                "objective_metrics": evaluation.objective_metrics,
                "verifier_result_ids": list(evaluation.verifier_result_ids),
                "rubric_ids": list(evaluation.rubric_ids),
            },
        )
    for verdict in bundle.benchmark_verdicts:
        evidence(
            "benchmark_verdict",
            verdict.verdict_id,
            definition_id=verdict.benchmark_authority,
            value=verdict.threshold,
            verdict=verdict.decision,
            facts={
                "score_source": verdict.score_source,
                "required_evaluation_ids": list(verdict.required_evaluation_ids),
                "required_gates": list(verdict.required_gates),
                "failure_reasons": list(verdict.failure_reasons),
            },
        )
    for receipt in bundle.receipts:
        evidence(
            "receipt",
            receipt.receipt_id,
            definition_id=receipt.operation,
            verdict=receipt.status,
            facts={
                "target_ids": list(receipt.target_ids),
                "return_code": receipt.return_code,
                "wall_time_seconds": receipt.wall_time_seconds,
                "input_digests": list(receipt.input_digests),
                "output_digests": list(receipt.output_digests),
                "completeness": receipt.completeness,
                "errors": list(receipt.errors),
            },
        )

    return {
        "schema_version": CATALOG_PROJECTION_SCHEMA_VERSION,
        "documents": [],
        "entities": [],
        "relationships": [],
        "aliases": [],
        "evidence": rows,
    }


__all__ = ["CATALOG_PROJECTION_SCHEMA_VERSION", "catalog_projection"]
