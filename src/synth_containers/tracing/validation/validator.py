"""Structural and evidence invariants for sealed traces and evidence bundles.

Validation is what separates "a file exists" from "this trace can be cited". A
violation is reported as a typed finding rather than an exception so one pass can
report everything wrong with a bundle at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.selectors import resolve_selector
from ..models.spans import UsageProvenance


VALIDATION_RECEIPT_SCHEMA_VERSION = "synth.validation-receipt.v1"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationFindingV1(JsonDataclassMixin):
    code: str
    severity: Severity | str
    message: str
    entity_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationReceiptV1(JsonDataclassMixin):
    receipt_id: str
    trace_id: str
    trace_digest: str
    validated_at: str
    valid: bool
    checks_run: tuple[str, ...] = ()
    findings: tuple[ValidationFindingV1, ...] = ()
    evidence_bundle_digest: str | None = None
    selectors_resolved: int = 0
    selectors_failed: int = 0
    schema_version: str = VALIDATION_RECEIPT_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "ValidationReceiptV1":
        return replace(self, content_digest=content_digest(self))


_CHECKS = (
    "sealed_digest",
    "unique_ids",
    "cross_references",
    "message_graph",
    "span_graph",
    "event_order",
    "tool_result_pairing",
    "usage_consistency",
    "completeness_consistency",
    "evidence_selectors",
    "evidence_definitions",
)


def validate_trace(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    """Run every structural invariant over a sealed trace document."""

    findings: list[ValidationFindingV1] = []

    if not document.content_digest:
        findings.append(
            ValidationFindingV1(
                code="trace_not_sealed",
                severity=Severity.ERROR,
                message="trace document has no content digest",
            )
        )
    else:
        recomputed = content_digest(document)
        if recomputed != document.content_digest:
            findings.append(
                ValidationFindingV1(
                    code="sealed_digest_mismatch",
                    severity=Severity.ERROR,
                    message="stored content digest does not match the document content",
                    detail={"stored": document.content_digest, "recomputed": recomputed},
                )
            )

    findings.extend(_check_unique(document))
    findings.extend(_check_references(document))
    findings.extend(_check_message_graph(document))
    findings.extend(_check_span_graph(document))
    findings.extend(_check_event_order(document))
    findings.extend(_check_tool_results(document))
    findings.extend(_check_usage(document))
    findings.extend(_check_completeness(document))
    return findings


def _check_unique(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    groups = {
        "actor": [item.actor_id for item in document.actors],
        "session": [item.session_id for item in document.sessions],
        "span": [item.span_id for item in document.spans],
        "event": [item.event_id for item in document.events],
        "message": [item.message_id for item in document.messages],
        "artifact": [item.artifact_id for item in document.artifacts],
        "branch": [item.branch_id for item in document.branches],
    }
    for kind, ids in groups.items():
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        for duplicate in duplicates:
            findings.append(
                ValidationFindingV1(
                    code="duplicate_id",
                    severity=Severity.ERROR,
                    message=f"duplicate {kind} id",
                    entity_id=duplicate,
                )
            )
    part_ids = [part.part_id for message in document.messages for part in message.parts]
    duplicates = sorted({item for item in part_ids if part_ids.count(item) > 1})
    for duplicate in duplicates:
        findings.append(
            ValidationFindingV1(
                code="duplicate_part_id",
                severity=Severity.ERROR,
                message="duplicate message part id",
                entity_id=duplicate,
            )
        )
    return findings


def _check_references(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    actors = {item.actor_id for item in document.actors}
    sessions = {item.session_id for item in document.sessions}
    spans = {item.span_id for item in document.spans}
    messages = {item.message_id for item in document.messages}
    artifacts = {item.artifact_id for item in document.artifacts}

    for session in document.sessions:
        if session.actor_id not in actors:
            findings.append(
                ValidationFindingV1(
                    code="dangling_actor_ref",
                    severity=Severity.ERROR,
                    message="session references an unknown actor",
                    entity_id=session.session_id,
                )
            )
    for span in document.spans:
        if span.actor_id not in actors:
            findings.append(
                ValidationFindingV1(
                    code="dangling_actor_ref",
                    severity=Severity.ERROR,
                    message="span references an unknown actor",
                    entity_id=span.span_id,
                )
            )
        if span.session_id not in sessions:
            findings.append(
                ValidationFindingV1(
                    code="dangling_session_ref",
                    severity=Severity.ERROR,
                    message="span references an unknown session",
                    entity_id=span.span_id,
                )
            )
        for message_id in (*span.input_message_ids, *span.output_message_ids):
            if message_id not in messages:
                findings.append(
                    ValidationFindingV1(
                        code="dangling_message_ref",
                        severity=Severity.ERROR,
                        message="span references an unknown message",
                        entity_id=span.span_id,
                        detail={"message_id": message_id},
                    )
                )
        for artifact_id in span.artifact_ids:
            if artifact_id not in artifacts:
                findings.append(
                    ValidationFindingV1(
                        code="dangling_artifact_ref",
                        severity=Severity.ERROR,
                        message="span references an unknown artifact",
                        entity_id=span.span_id,
                    )
                )
    for event in document.events:
        if event.span_id and event.span_id not in spans:
            findings.append(
                ValidationFindingV1(
                    code="dangling_span_ref",
                    severity=Severity.ERROR,
                    message="event references an unknown span",
                    entity_id=event.event_id,
                )
            )
        for parent in event.caused_by_event_ids:
            if parent not in {item.event_id for item in document.events}:
                findings.append(
                    ValidationFindingV1(
                        code="dangling_event_ref",
                        severity=Severity.ERROR,
                        message="event causal parent is not in this trace",
                        entity_id=event.event_id,
                        detail={"caused_by": parent},
                    )
                )
    return findings


def _check_message_graph(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    known = {item.message_id for item in document.messages}
    for message in document.messages:
        for predecessor in message.predecessor_message_ids:
            if predecessor not in known:
                findings.append(
                    ValidationFindingV1(
                        code="dangling_predecessor",
                        severity=Severity.ERROR,
                        message="message predecessor is not in this trace",
                        entity_id=message.message_id,
                    )
                )
            if predecessor == message.message_id:
                findings.append(
                    ValidationFindingV1(
                        code="message_graph_cycle",
                        severity=Severity.ERROR,
                        message="message is its own predecessor",
                        entity_id=message.message_id,
                    )
                )
    for branch in document.branches:
        if branch.head_message_id and branch.head_message_id not in known:
            findings.append(
                ValidationFindingV1(
                    code="dangling_branch_head",
                    severity=Severity.ERROR,
                    message="branch head is not in this trace",
                    entity_id=branch.branch_id,
                )
            )
    return findings


def _check_span_graph(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    parents = {item.span_id: item.parent_span_id for item in document.spans}
    for span_id in parents:
        seen: set[str] = set()
        current: str | None = span_id
        while current is not None:
            if current in seen:
                findings.append(
                    ValidationFindingV1(
                        code="span_graph_cycle",
                        severity=Severity.ERROR,
                        message="span parent chain contains a cycle",
                        entity_id=span_id,
                    )
                )
                break
            seen.add(current)
            current = parents.get(current)
            if current is not None and current not in parents:
                findings.append(
                    ValidationFindingV1(
                        code="dangling_parent_span",
                        severity=Severity.ERROR,
                        message="span parent is not in this trace",
                        entity_id=span_id,
                    )
                )
                break
    return findings


def _check_event_order(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    previous: int | None = None
    for event in document.events:
        sequence = event.order.chronological_sequence
        if sequence is None:
            continue
        if previous is not None and sequence < previous:
            findings.append(
                ValidationFindingV1(
                    code="event_order_not_monotonic",
                    severity=Severity.ERROR,
                    message="chronological sequence decreased within the trace",
                    entity_id=event.event_id,
                    detail={"previous": previous, "current": sequence},
                )
            )
        previous = sequence
    addresses: list[tuple[Any, ...]] = []
    for event in document.events:
        structural = event.order.structural
        if structural is None:
            continue
        key = (
            structural.workflow_id,
            structural.node_path,
            structural.iteration,
            structural.local_sequence,
        )
        if key in addresses:
            findings.append(
                ValidationFindingV1(
                    code="duplicate_structural_address",
                    severity=Severity.ERROR,
                    message="two events share one workflow structural address",
                    entity_id=event.event_id,
                )
            )
        addresses.append(key)
    return findings


def _check_tool_results(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    call_ids: set[str] = set()
    for message in document.messages:
        for part in message.parts:
            if str(part.type) == "tool_call" and part.tool_call_id:
                call_ids.add(part.tool_call_id)
    for message in document.messages:
        for part in message.parts:
            if str(part.type) != "tool_result":
                continue
            if not part.tool_call_id:
                findings.append(
                    ValidationFindingV1(
                        code="orphan_tool_result",
                        severity=Severity.ERROR,
                        message="tool result has no tool call id",
                        entity_id=message.message_id,
                    )
                )
            elif part.tool_call_id not in call_ids:
                findings.append(
                    ValidationFindingV1(
                        code="orphan_tool_result",
                        severity=Severity.WARNING,
                        message="tool result cites a tool call not present in this trace",
                        entity_id=message.message_id,
                        detail={"tool_call_id": part.tool_call_id},
                    )
                )
    return findings


def _check_usage(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    observed = [
        span.usage
        for span in document.spans
        if span.usage is not None
        and str(span.usage.provenance) == UsageProvenance.OBSERVED_PROVIDER
    ]
    if not observed:
        return findings
    total_prompt = sum(int(item.prompt_tokens or 0) for item in observed)
    total_completion = sum(int(item.completion_tokens or 0) for item in observed)
    root = document.usage
    if str(root.provenance) in {UsageProvenance.OBSERVED_PROVIDER, UsageProvenance.DERIVED}:
        if root.prompt_tokens is not None and int(root.prompt_tokens) != total_prompt:
            findings.append(
                ValidationFindingV1(
                    code="usage_aggregate_mismatch",
                    severity=Severity.ERROR,
                    message="root prompt tokens do not equal the sum of observed span usage",
                    detail={"root": root.prompt_tokens, "spans": total_prompt},
                )
            )
        if root.completion_tokens is not None and int(root.completion_tokens) != total_completion:
            findings.append(
                ValidationFindingV1(
                    code="usage_aggregate_mismatch",
                    severity=Severity.ERROR,
                    message="root completion tokens do not equal the sum of observed span usage",
                    detail={"root": root.completion_tokens, "spans": total_completion},
                )
            )
    return findings


def _check_completeness(document: TraceDocumentV5) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    completeness = document.completeness
    model_call_spans = document.spans_of_kind("model_call")
    if str(completeness.model_calls) == "complete" and not model_call_spans:
        findings.append(
            ValidationFindingV1(
                code="completeness_overclaim",
                severity=Severity.ERROR,
                message="completeness claims complete model-call coverage with no model-call spans",
            )
        )
    if str(document.lifecycle.status) == "completed" and not completeness.terminal_event_observed:
        findings.append(
            ValidationFindingV1(
                code="terminal_event_missing",
                severity=Severity.ERROR,
                message="lifecycle is completed but no terminal capture record was observed",
            )
        )
    return findings


def validate_evidence(
    document: TraceDocumentV5,
    bundle: TraceEvidenceBundleV5,
) -> tuple[list[ValidationFindingV1], int, int]:
    """Check that evidence cites this exact trace and that every selector resolves."""

    findings: list[ValidationFindingV1] = []
    resolved = 0
    failed = 0
    if bundle.trace_ref.trace_id != document.trace_id:
        findings.append(
            ValidationFindingV1(
                code="evidence_trace_mismatch",
                severity=Severity.ERROR,
                message="evidence bundle references a different trace id",
            )
        )
    if bundle.trace_ref.content_digest != document.content_digest:
        findings.append(
            ValidationFindingV1(
                code="evidence_digest_mismatch",
                severity=Severity.ERROR,
                message="evidence bundle references a different trace digest",
            )
        )
    for selector in bundle.selectors():
        resolution = resolve_selector(document, selector)
        if resolution.resolved:
            resolved += 1
            continue
        failed += 1
        findings.append(
            ValidationFindingV1(
                code="selector_unresolved",
                severity=Severity.ERROR,
                message=f"evidence selector did not resolve: {resolution.reason}",
                entity_id=selector.entity_id,
                detail={"kind": str(selector.kind)},
            )
        )

    rubric_digests = {item.rubric_id: item.content_digest for item in bundle.rubrics}
    for result in bundle.verifier_results:
        expected = rubric_digests.get(result.rubric_id)
        if expected is None:
            findings.append(
                ValidationFindingV1(
                    code="verifier_rubric_missing",
                    severity=Severity.ERROR,
                    message="verifier result cites a rubric absent from the bundle",
                    entity_id=result.verifier_result_id,
                )
            )
        elif expected != result.rubric_digest:
            findings.append(
                ValidationFindingV1(
                    code="verifier_rubric_digest_mismatch",
                    severity=Severity.ERROR,
                    message="verifier result cites a different rubric version",
                    entity_id=result.verifier_result_id,
                )
            )
    reward_digests = {item.reward_id: item.content_digest for item in bundle.reward_definitions}
    for record in bundle.reward_records:
        expected = reward_digests.get(record.reward_id)
        if expected is None:
            findings.append(
                ValidationFindingV1(
                    code="reward_definition_missing",
                    severity=Severity.ERROR,
                    message="reward record cites a definition absent from the bundle",
                    entity_id=record.reward_record_id,
                )
            )
        elif expected != record.reward_digest:
            findings.append(
                ValidationFindingV1(
                    code="reward_definition_digest_mismatch",
                    severity=Severity.ERROR,
                    message="reward record cites a different reward definition version",
                    entity_id=record.reward_record_id,
                )
            )
    annotator_digests = {
        item.annotator_id: item.content_digest for item in bundle.annotator_definitions
    }
    for annotation in bundle.annotations:
        expected = annotator_digests.get(annotation.annotator_id)
        if expected is None:
            findings.append(
                ValidationFindingV1(
                    code="annotator_definition_missing",
                    severity=Severity.ERROR,
                    message="annotation cites an annotator absent from the bundle",
                    entity_id=annotation.annotation_id,
                )
            )
        elif expected != annotation.annotator_digest:
            findings.append(
                ValidationFindingV1(
                    code="annotator_definition_digest_mismatch",
                    severity=Severity.ERROR,
                    message="annotation cites a different annotator version",
                    entity_id=annotation.annotation_id,
                )
            )
    return findings, resolved, failed


def validate(
    document: TraceDocumentV5,
    evidence: TraceEvidenceBundleV5 | None = None,
) -> ValidationReceiptV1:
    """Validate a trace and its evidence and return a sealed validation receipt."""

    findings = validate_trace(document)
    resolved = 0
    failed = 0
    if evidence is not None:
        evidence_findings, resolved, failed = validate_evidence(document, evidence)
        findings.extend(evidence_findings)
    errors = [item for item in findings if str(item.severity) == Severity.ERROR]
    receipt = ValidationReceiptV1(
        receipt_id=record_id(
            "vrcpt",
            kind="validation",
            scope=(document.trace_id,),
            key=document.content_digest,
        ),
        trace_id=document.trace_id,
        trace_digest=document.content_digest,
        validated_at=utc_now(),
        valid=not errors,
        checks_run=_CHECKS,
        findings=tuple(findings),
        evidence_bundle_digest=evidence.content_digest if evidence else None,
        selectors_resolved=resolved,
        selectors_failed=failed,
    )
    return receipt.sealed()


__all__ = [
    "VALIDATION_RECEIPT_SCHEMA_VERSION",
    "Severity",
    "ValidationFindingV1",
    "ValidationReceiptV1",
    "validate",
    "validate_evidence",
    "validate_trace",
]
