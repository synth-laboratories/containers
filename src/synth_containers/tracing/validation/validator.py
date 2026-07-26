"""Structural and evidence invariants for sealed traces and evidence bundles.

Validation is what separates "a file exists" from "this trace can be cited". A
violation is reported as a typed finding rather than an exception so one pass can
report everything wrong with a bundle at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
import math
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import content_digest, record_id, utc_now
from ..models.actors import Visibility
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.selectors import GroundingStatus, resolve_selector
from ..models.spans import UsageProvenance
from ..models.standards import (
    AnnotationDerivationKind,
    AnnotationInspectionSource,
    AnnotationReviewState,
    AnnotationStatus,
    AnnotationTaskKind,
    AnnotationValueKind,
    AnnotatorGroundingRequirement,
    ConfidenceSemantics,
    ExecutionStatus,
    ProducerKind,
    RecordState,
    UnavailableEvidenceBehavior,
    VerificationStatus,
    aggregate_reward_values,
    aggregate_rubric_score,
)


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
    "alias_integrity",
    "message_graph",
    "span_graph",
    "event_order",
    "tool_result_pairing",
    "usage_consistency",
    "session_lifecycle",
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
    findings.extend(validate_alias_integrity(document))
    findings.extend(_check_message_graph(document))
    findings.extend(_check_span_graph(document))
    findings.extend(_check_event_order(document))
    findings.extend(_check_tool_results(document))
    findings.extend(_check_usage(document))
    findings.extend(_check_session_lifecycle(document))
    findings.extend(_check_completeness(document))
    return findings


_LOCAL_ALIAS_TARGET_KINDS = frozenset(
    {
        "trace",
        "actor",
        "session",
        "span",
        "event",
        "message",
        "part",
        "artifact",
        "branch",
        "error",
    }
)
_EXTERNAL_ALIAS_TARGET_KINDS = frozenset({"external_trace"})


def validate_alias_integrity(
    document: TraceDocumentV5,
) -> list[ValidationFindingV1]:
    """Validate every alias target against the canonical entity inventory."""

    targets = {
        "trace": {document.trace_id},
        "actor": {item.actor_id for item in document.actors},
        "session": {item.session_id for item in document.sessions},
        "span": {item.span_id for item in document.spans},
        "event": {item.event_id for item in document.events},
        "message": {item.message_id for item in document.messages},
        "part": {
            part.part_id
            for message in document.messages
            for part in message.parts
        },
        "artifact": {item.artifact_id for item in document.artifacts},
        "branch": {item.branch_id for item in document.branches},
        "error": {item.error_id for item in document.errors},
    }
    alias_groups = [
        ("trace", document.trace_id, document.aliases),
        ("provenance", document.trace_id, document.provenance.aliases),
        *(
            ("actor", item.actor_id, item.aliases)
            for item in document.actors
        ),
        *(
            ("session", item.session_id, item.aliases)
            for item in document.sessions
        ),
        *(
            ("span", item.span_id, item.aliases)
            for item in document.spans
        ),
        *(
            ("event", item.event_id, item.aliases)
            for item in document.events
        ),
        *(
            ("message", item.message_id, item.aliases)
            for item in document.messages
        ),
    ]
    supported = _LOCAL_ALIAS_TARGET_KINDS | _EXTERNAL_ALIAS_TARGET_KINDS
    findings: list[ValidationFindingV1] = []
    for owner_kind, owner_id, aliases in alias_groups:
        for alias_item in aliases:
            target_kind = str(alias_item.target_kind)
            detail = {
                "alias_namespace": str(alias_item.namespace),
                "alias_value": alias_item.value,
                "owner_kind": owner_kind,
                "target_kind": target_kind,
                "target_id": alias_item.target_id,
            }
            if target_kind not in supported:
                findings.append(
                    ValidationFindingV1(
                        code="unsupported_alias_target_kind",
                        severity=Severity.ERROR,
                        message=(
                            "alias target_kind is not a supported canonical "
                            f"entity kind: {target_kind!r}"
                        ),
                        entity_id=owner_id,
                        detail=detail,
                    )
                )
                continue
            if not alias_item.target_id or (
                target_kind in _LOCAL_ALIAS_TARGET_KINDS
                and alias_item.target_id not in targets[target_kind]
            ):
                findings.append(
                    ValidationFindingV1(
                        code="dangling_alias_target",
                        severity=Severity.ERROR,
                        message=(
                            f"alias target {target_kind}:{alias_item.target_id} "
                            "is not present in the trace"
                        ),
                        entity_id=owner_id,
                        detail=detail,
                    )
                )
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

    for actor in document.actors:
        if actor.parent_actor_id and actor.parent_actor_id not in actors:
            findings.append(
                ValidationFindingV1(
                    code="dangling_parent_actor_ref",
                    severity=Severity.ERROR,
                    message="actor references an unknown parent actor",
                    entity_id=actor.actor_id,
                )
            )
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
        if (
            session.parent_session_id
            and session.parent_session_id not in sessions
        ):
            findings.append(
                ValidationFindingV1(
                    code="dangling_parent_session_ref",
                    severity=Severity.ERROR,
                    message="session references an unknown parent session",
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
        if event.actor_id not in actors:
            findings.append(
                ValidationFindingV1(
                    code="dangling_actor_ref",
                    severity=Severity.ERROR,
                    message="event references an unknown actor",
                    entity_id=event.event_id,
                )
            )
        if event.session_id not in sessions:
            findings.append(
                ValidationFindingV1(
                    code="dangling_session_ref",
                    severity=Severity.ERROR,
                    message="event references an unknown session",
                    entity_id=event.event_id,
                )
            )
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
    for message in document.messages:
        if message.sender_actor_id and message.sender_actor_id not in actors:
            findings.append(
                ValidationFindingV1(
                    code="dangling_actor_ref",
                    severity=Severity.ERROR,
                    message="message references an unknown sender actor",
                    entity_id=message.message_id,
                )
            )
        if message.session_id and message.session_id not in sessions:
            findings.append(
                ValidationFindingV1(
                    code="dangling_session_ref",
                    severity=Severity.ERROR,
                    message="message references an unknown session",
                    entity_id=message.message_id,
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


def _check_session_lifecycle(
    document: TraceDocumentV5,
) -> list[ValidationFindingV1]:
    findings: list[ValidationFindingV1] = []
    trace_status = str(document.lifecycle.status)
    terminal_trace_statuses = {
        "completed",
        "failed",
        "interrupted",
    }
    trace_terminal = trace_status in terminal_trace_statuses
    terminal_session_statuses = {"completed", "failed", "interrupted"}
    if trace_status not in {
        "live",
        "sealing",
        *terminal_trace_statuses,
    }:
        findings.append(
            ValidationFindingV1(
                code="unknown_trace_status",
                severity=Severity.ERROR,
                message=f"unknown trace lifecycle status {trace_status!r}",
                entity_id=document.trace_id,
            )
        )
    trace_started = _validated_timestamp(
        document.lifecycle.started_at,
        code="invalid_trace_timestamp",
        entity_id=document.trace_id,
        field="started_at",
        findings=findings,
    )
    trace_ended = (
        _validated_timestamp(
            document.lifecycle.ended_at,
            code="invalid_trace_timestamp",
            entity_id=document.trace_id,
            field="ended_at",
            findings=findings,
        )
        if document.lifecycle.ended_at is not None
        else None
    )
    if (
        trace_started is not None
        and trace_ended is not None
        and trace_ended < trace_started
    ):
        findings.append(
            ValidationFindingV1(
                code="trace_lifecycle_order_invalid",
                severity=Severity.ERROR,
                message="trace ended_at precedes started_at",
                entity_id=document.trace_id,
            )
        )
    if trace_terminal and document.lifecycle.ended_at is None:
        findings.append(
            ValidationFindingV1(
                code="terminal_trace_missing_ended_at",
                severity=Severity.ERROR,
                message="terminal trace has no ended_at timestamp",
                entity_id=document.trace_id,
            )
        )
    parsed_sessions: dict[str, tuple[datetime | None, datetime | None]] = {}
    for session in document.sessions:
        status = str(session.status)
        if status not in {"running", *terminal_session_statuses}:
            findings.append(
                ValidationFindingV1(
                    code="unknown_session_status",
                    severity=Severity.ERROR,
                    message=f"unknown session lifecycle status {status!r}",
                    entity_id=session.session_id,
                )
            )
        started = _validated_timestamp(
            session.started_at,
            code="invalid_session_timestamp",
            entity_id=session.session_id,
            field="started_at",
            findings=findings,
        )
        ended = (
            _validated_timestamp(
                session.ended_at,
                code="invalid_session_timestamp",
                entity_id=session.session_id,
                field="ended_at",
                findings=findings,
            )
            if session.ended_at is not None
            else None
        )
        parsed_sessions[session.session_id] = (started, ended)
        if started is not None and ended is not None and ended < started:
            findings.append(
                ValidationFindingV1(
                    code="session_lifecycle_order_invalid",
                    severity=Severity.ERROR,
                    message="session ended_at precedes started_at",
                    entity_id=session.session_id,
                )
            )
        if started is not None and trace_started is not None and started < trace_started:
            findings.append(
                ValidationFindingV1(
                    code="session_starts_before_trace",
                    severity=Severity.ERROR,
                    message="session started_at is before trace started_at",
                    entity_id=session.session_id,
                )
            )
        if started is not None and trace_ended is not None and started > trace_ended:
            findings.append(
                ValidationFindingV1(
                    code="session_starts_after_trace",
                    severity=Severity.ERROR,
                    message="session started_at is after trace ended_at",
                    entity_id=session.session_id,
                )
            )
        if ended is not None and trace_ended is not None and ended > trace_ended:
            findings.append(
                ValidationFindingV1(
                    code="session_outlives_trace",
                    severity=Severity.ERROR,
                    message="session ended_at is after trace ended_at",
                    entity_id=session.session_id,
                )
            )
        if trace_terminal and status == "running":
            findings.append(
                ValidationFindingV1(
                    code="session_non_terminal_in_sealed_trace",
                    severity=Severity.ERROR,
                    message="terminal trace contains a running session",
                    entity_id=session.session_id,
                )
            )
        if status == "running" and session.ended_at is not None:
            findings.append(
                ValidationFindingV1(
                    code="running_session_has_ended_at",
                    severity=Severity.ERROR,
                    message="running session carries a terminal timestamp",
                    entity_id=session.session_id,
                )
            )
        if status in terminal_session_statuses and session.ended_at is None:
            findings.append(
                ValidationFindingV1(
                    code="terminal_session_missing_ended_at",
                    severity=Severity.ERROR,
                    message="terminal session has no ended_at timestamp",
                    entity_id=session.session_id,
                )
            )
    sessions_by_id = {session.session_id: session for session in document.sessions}
    actors_by_id = {actor.actor_id: actor for actor in document.actors}
    for child in document.sessions:
        if not child.parent_session_id:
            continue
        parent = sessions_by_id.get(child.parent_session_id)
        if parent is None:
            continue
        child_actor = actors_by_id.get(child.actor_id)
        if (
            child_actor is not None
            and child_actor.parent_actor_id != parent.actor_id
        ):
            findings.append(
                ValidationFindingV1(
                    code="actor_session_parent_disagreement",
                    severity=Severity.ERROR,
                    message=(
                        "child actor parent does not match the parent session actor"
                    ),
                    entity_id=child.session_id,
                    detail={
                        "actor_parent_id": child_actor.parent_actor_id,
                        "parent_session_actor_id": parent.actor_id,
                    },
                )
            )
        if (
            str(parent.status) in terminal_session_statuses
            and str(child.status) == "running"
        ):
            findings.append(
                ValidationFindingV1(
                    code="live_child_of_terminal_parent",
                    severity=Severity.ERROR,
                    message="running child session has a terminal parent",
                    entity_id=child.session_id,
                    detail={"parent_session_id": parent.session_id},
                )
            )
        child_started, child_ended = parsed_sessions[child.session_id]
        parent_started, parent_ended = parsed_sessions[parent.session_id]
        if (
            child_started is not None
            and parent_started is not None
            and child_started < parent_started
        ):
            findings.append(
                ValidationFindingV1(
                    code="child_session_starts_before_parent",
                    severity=Severity.ERROR,
                    message="child session starts before its parent session",
                    entity_id=child.session_id,
                    detail={"parent_session_id": parent.session_id},
                )
            )
        if (
            child_ended is not None
            and parent_ended is not None
            and child_ended > parent_ended
        ):
            findings.append(
                ValidationFindingV1(
                    code="child_session_outlives_parent",
                    severity=Severity.ERROR,
                    message="child session ends after its parent session",
                    entity_id=child.session_id,
                    detail={"parent_session_id": parent.session_id},
                )
            )
    parents = {
        session.session_id: session.parent_session_id
        for session in document.sessions
    }
    for session_id in parents:
        seen: set[str] = set()
        current: str | None = session_id
        while current is not None and current in parents:
            if current in seen:
                findings.append(
                    ValidationFindingV1(
                        code="session_parent_cycle",
                        severity=Severity.ERROR,
                        message="session parent chain contains a cycle",
                        entity_id=session_id,
                    )
                )
                break
            seen.add(current)
            current = parents[current]
    return findings


def _validated_timestamp(
    value: Any,
    *,
    code: str,
    entity_id: str,
    field: str,
    findings: list[ValidationFindingV1],
) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        findings.append(
            ValidationFindingV1(
                code=code,
                severity=Severity.ERROR,
                message=f"{field} must be an RFC3339 timestamp with timezone",
                entity_id=entity_id,
                detail={"field": field, "value": value},
            )
        )
        return None
    return parsed


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


def _annotation_payload_value_matches(value: Any, value_kind: str) -> bool:
    if value_kind == AnnotationValueKind.STRING:
        return isinstance(value, str)
    if value_kind == AnnotationValueKind.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if value_kind == AnnotationValueKind.NUMBER:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if value_kind == AnnotationValueKind.BOOLEAN:
        return isinstance(value, bool)
    if value_kind == AnnotationValueKind.OBJECT:
        return isinstance(value, dict)
    if value_kind == AnnotationValueKind.ARRAY:
        return isinstance(value, list)
    return False


def validate_evidence(
    document: TraceDocumentV5,
    bundle: TraceEvidenceBundleV5,
) -> tuple[list[ValidationFindingV1], int, int]:
    """Check that evidence cites this exact trace and that every selector resolves."""

    findings: list[ValidationFindingV1] = []
    resolved = 0
    failed = 0
    if not bundle.content_digest:
        findings.append(
            ValidationFindingV1(
                code="evidence_not_sealed",
                severity=Severity.ERROR,
                message="evidence bundle has no content digest",
            )
        )
    elif content_digest(bundle) != bundle.content_digest:
        findings.append(
            ValidationFindingV1(
                code="evidence_sealed_digest_mismatch",
                severity=Severity.ERROR,
                message="stored evidence bundle digest does not match its content",
            )
        )
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

    def finding(code: str, message: str, entity_id: str | None = None) -> None:
        findings.append(
            ValidationFindingV1(
                code=code,
                severity=Severity.ERROR,
                message=message,
                entity_id=entity_id,
            )
        )

    def same_number(left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return left is right
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left),
            float(right),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

    def record_identity(record: Any) -> str:
        for attribute in (
            "criterion_id",
            "rubric_id",
            "verifier_id",
            "annotator_id",
            "reward_id",
            "annotation_id",
            "verifier_result_id",
            "reward_record_id",
            "aggregation_id",
            "evaluation_id",
            "verdict_id",
            "receipt_id",
        ):
            value = getattr(record, attribute, None)
            if value:
                return str(value)
        return type(record).__name__

    def check_digest(record: Any) -> None:
        identity = record_identity(record)
        stored = getattr(record, "content_digest", "")
        if not stored:
            finding(
                "evidence_record_not_sealed",
                f"{type(record).__name__} has no content digest",
                identity,
            )
        elif content_digest(record) != stored:
            finding(
                "evidence_record_digest_mismatch",
                f"{type(record).__name__} content digest does not match its content",
                identity,
            )

    standards_collections = (
        bundle.criteria,
        bundle.rubrics,
        bundle.verifier_definitions,
        bundle.annotator_definitions,
        bundle.reward_definitions,
        bundle.annotations,
        bundle.verifier_results,
        bundle.reward_records,
        bundle.reward_aggregations,
        bundle.evaluation_results,
        bundle.benchmark_verdicts,
        bundle.receipts,
    )
    for collection in standards_collections:
        for record in collection:
            check_digest(record)
    for rubric in bundle.rubrics:
        for criterion in rubric.criteria:
            check_digest(criterion)

    def index(records: tuple[Any, ...], id_field: str, kind: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for record in records:
            identity = str(getattr(record, id_field))
            if identity in indexed:
                finding(
                    "duplicate_evidence_record_id",
                    f"duplicate {kind} id in evidence bundle",
                    identity,
                )
            indexed[identity] = record
        return indexed

    criteria = index(bundle.criteria, "criterion_id", "criterion")
    rubrics = index(bundle.rubrics, "rubric_id", "rubric")
    verifier_definitions = index(
        bundle.verifier_definitions, "verifier_id", "verifier definition"
    )
    annotator_definitions = index(
        bundle.annotator_definitions, "annotator_id", "annotator definition"
    )
    annotations = index(bundle.annotations, "annotation_id", "annotation")
    reward_definitions = index(
        bundle.reward_definitions, "reward_id", "reward definition"
    )
    verifier_results = index(
        bundle.verifier_results, "verifier_result_id", "verifier result"
    )
    reward_records = index(bundle.reward_records, "reward_record_id", "reward record")
    reward_aggregations = index(
        bundle.reward_aggregations, "aggregation_id", "reward aggregation"
    )
    evaluation_results = index(
        bundle.evaluation_results, "evaluation_id", "evaluation result"
    )
    source_result_ids = {
        *criteria,
        *verifier_results,
        *reward_records,
        *reward_aggregations,
        *evaluation_results,
    }

    for criterion in bundle.criteria:
        if criterion.max_score <= criterion.min_score:
            finding(
                "criterion_score_range_invalid",
                "criterion maximum score must be greater than its minimum",
                criterion.criterion_id,
            )
        if not criterion.min_score <= criterion.pass_threshold <= criterion.max_score:
            finding(
                "criterion_threshold_out_of_bounds",
                "criterion pass threshold is outside its declared score range",
                criterion.criterion_id,
            )
        if criterion.weight < 0.0:
            finding(
                "criterion_weight_invalid",
                "criterion weight cannot be negative",
                criterion.criterion_id,
            )
        for value in (
            criterion.weight,
            criterion.min_score,
            criterion.max_score,
            criterion.pass_threshold,
        ):
            if not math.isfinite(float(value)):
                finding(
                    "criterion_numeric_value_nonfinite",
                    "criterion weights, bounds, and threshold must be finite",
                    criterion.criterion_id,
                )
                break

    for definition in bundle.reward_definitions:
        if (
            definition.lower_bound is not None
            and definition.upper_bound is not None
            and definition.lower_bound > definition.upper_bound
        ):
            finding(
                "reward_definition_bounds_invalid",
                "reward definition lower bound exceeds its upper bound",
                definition.reward_id,
            )
        for value in (definition.lower_bound, definition.upper_bound):
            if value is not None and not math.isfinite(float(value)):
                finding(
                    "reward_definition_bound_nonfinite",
                    "reward definition bounds must be finite",
                    definition.reward_id,
                )
        if len(definition.vector_components) != len(set(definition.vector_components)):
            finding(
                "reward_definition_duplicate_component",
                "reward definition declares a duplicate vector component",
                definition.reward_id,
            )

    for definition in bundle.annotator_definitions:
        if definition.minimum_evidence < 0:
            finding(
                "annotator_minimum_evidence_invalid",
                "annotator minimum evidence cannot be negative",
                definition.annotator_id,
            )
        if str(definition.grounding_requirement).lower() not in {
            str(value) for value in AnnotatorGroundingRequirement
        }:
            finding(
                "annotator_grounding_requirement_invalid",
                "annotator declares an unsupported grounding requirement",
                definition.annotator_id,
            )
        if str(definition.unavailable_evidence_behavior).lower() not in {
            str(value) for value in UnavailableEvidenceBehavior
        }:
            finding(
                "annotator_unavailable_evidence_behavior_invalid",
                "annotator declares an unsupported unavailable-evidence behavior",
                definition.annotator_id,
            )
        if str(definition.confidence_semantics).lower() not in {
            str(value) for value in ConfidenceSemantics
        }:
            finding(
                "annotator_confidence_semantics_invalid",
                "annotator declares unsupported confidence semantics",
                definition.annotator_id,
            )
        if (
            str(definition.confidence_semantics)
            == ConfidenceSemantics.CALIBRATED_PROBABILITY
            and not definition.confidence_calibration_ref
        ):
            finding(
                "annotator_confidence_calibration_missing",
                "calibrated probability confidence must cite a calibration reference",
                definition.annotator_id,
            )
        if len(definition.taxonomy) != len(set(definition.taxonomy)):
            finding(
                "annotator_taxonomy_duplicate_label",
                "annotator taxonomy contains a duplicate label",
                definition.annotator_id,
            )
        output_contract = definition.output_contract
        if output_contract is None:
            continue
        if str(output_contract.task_kind).lower() not in {
            str(value) for value in AnnotationTaskKind
        }:
            finding(
                "annotator_task_kind_invalid",
                "annotator output contract declares an unsupported task kind",
                definition.annotator_id,
            )
        if not output_contract.annotation_types:
            finding(
                "annotator_annotation_types_missing",
                "annotator output contract must declare at least one annotation type",
                definition.annotator_id,
            )
        if any(not value for value in output_contract.annotation_types):
            finding(
                "annotator_annotation_type_empty",
                "annotator output contract contains an empty annotation type",
                definition.annotator_id,
            )
        if len(output_contract.annotation_types) != len(
            set(output_contract.annotation_types)
        ):
            finding(
                "annotator_annotation_type_duplicate",
                "annotator output contract contains a duplicate annotation type",
                definition.annotator_id,
            )
        producer_kinds = tuple(
            str(value).lower() for value in output_contract.allowed_producer_kinds
        )
        if len(producer_kinds) != len(set(producer_kinds)):
            finding(
                "annotator_producer_kind_duplicate",
                "annotator output contract contains a duplicate producer kind",
                definition.annotator_id,
            )
        if any(
            value not in {str(kind) for kind in ProducerKind}
            for value in producer_kinds
        ):
            finding(
                "annotator_producer_kind_invalid",
                "annotator output contract contains an unsupported producer kind",
                definition.annotator_id,
            )
        taxonomy_labels = tuple(item.label for item in output_contract.taxonomy)
        if any(not label for label in taxonomy_labels):
            finding(
                "annotator_typed_taxonomy_empty_label",
                "annotator typed taxonomy contains an empty canonical label",
                definition.annotator_id,
            )
        if len(taxonomy_labels) != len(set(taxonomy_labels)):
            finding(
                "annotator_typed_taxonomy_duplicate_label",
                "annotator typed taxonomy contains a duplicate canonical label",
                definition.annotator_id,
            )
        if output_contract.taxonomy and set(taxonomy_labels) != set(
            definition.taxonomy
        ):
            finding(
                "annotator_typed_taxonomy_mismatch",
                "annotator typed taxonomy must describe every flat taxonomy label exactly",
                definition.annotator_id,
            )
        taxonomy_label_set = set(taxonomy_labels)
        taxonomy_aliases: set[str] = set()
        parent_by_label: dict[str, str] = {}
        for taxon in output_contract.taxonomy:
            if taxon.parent_label is not None:
                if taxon.parent_label not in taxonomy_label_set:
                    finding(
                        "annotator_taxonomy_parent_missing",
                        "annotator taxonomy label cites an absent parent label",
                        definition.annotator_id,
                    )
                elif taxon.parent_label == taxon.label:
                    finding(
                        "annotator_taxonomy_parent_cycle",
                        "annotator taxonomy label cannot parent itself",
                        definition.annotator_id,
                    )
                else:
                    parent_by_label[taxon.label] = taxon.parent_label
            if len(taxon.aliases) != len(set(taxon.aliases)):
                finding(
                    "annotator_taxonomy_duplicate_alias",
                    "annotator taxonomy label contains a duplicate alias",
                    definition.annotator_id,
                )
            for alias in taxon.aliases:
                if not alias:
                    finding(
                        "annotator_taxonomy_empty_alias",
                        "annotator taxonomy contains an empty alias",
                        definition.annotator_id,
                    )
                if alias in taxonomy_label_set or alias in taxonomy_aliases:
                    finding(
                        "annotator_taxonomy_alias_collision",
                        "annotator taxonomy alias collides with another label or alias",
                        definition.annotator_id,
                    )
                taxonomy_aliases.add(alias)
        for label in parent_by_label:
            seen: set[str] = set()
            cursor = label
            while cursor in parent_by_label:
                if cursor in seen:
                    finding(
                        "annotator_taxonomy_parent_cycle",
                        "annotator taxonomy parent hierarchy contains a cycle",
                        definition.annotator_id,
                    )
                    break
                seen.add(cursor)
                cursor = parent_by_label[cursor]
        payload_schema = output_contract.payload_schema
        if payload_schema is None:
            continue
        field_names = tuple(item.field_name for item in payload_schema.fields)
        if not payload_schema.schema_id or not payload_schema.version:
            finding(
                "annotator_payload_schema_identity_missing",
                "annotator payload schema must declare an id and version",
                definition.annotator_id,
            )
        if any(not name for name in field_names):
            finding(
                "annotator_payload_field_name_missing",
                "annotator payload schema contains an empty field name",
                definition.annotator_id,
            )
        if len(field_names) != len(set(field_names)):
            finding(
                "annotator_payload_field_duplicate",
                "annotator payload schema contains a duplicate field name",
                definition.annotator_id,
            )
        for payload_field in payload_schema.fields:
            value_kind = str(payload_field.value_kind).lower()
            if value_kind not in {
                str(value) for value in AnnotationValueKind
            }:
                finding(
                    "annotator_payload_field_kind_invalid",
                    "annotator payload field declares an unsupported value kind",
                    definition.annotator_id,
                )
                continue
            if any(
                not _annotation_payload_value_matches(value, value_kind)
                for value in payload_field.allowed_values
            ):
                finding(
                    "annotator_payload_allowed_value_type_mismatch",
                    "annotator payload allowed value differs from its field value kind",
                    definition.annotator_id,
                )
            if len(payload_field.allowed_values) != len(
                {
                    content_digest(value)
                    for value in payload_field.allowed_values
                }
            ):
                finding(
                    "annotator_payload_allowed_value_duplicate",
                    "annotator payload field repeats an allowed value",
                    definition.annotator_id,
                )

    for rubric in bundle.rubrics:
        seen: set[str] = set()
        for criterion in rubric.criteria:
            if criterion.criterion_id in seen:
                finding(
                    "rubric_duplicate_criterion",
                    "rubric contains a duplicate criterion id",
                    rubric.rubric_id,
                )
            seen.add(criterion.criterion_id)
            expected = criteria.get(criterion.criterion_id)
            if expected is None:
                finding(
                    "rubric_criterion_missing",
                    "rubric criterion is absent from the bundle criterion registry",
                    criterion.criterion_id,
                )
            elif expected.content_digest != criterion.content_digest:
                finding(
                    "rubric_criterion_digest_mismatch",
                    "rubric embeds a different criterion definition version",
                    criterion.criterion_id,
                )
        if not math.isfinite(float(rubric.aggregation.pass_threshold)):
            finding(
                "rubric_pass_threshold_nonfinite",
                "rubric pass threshold must be finite",
                rubric.rubric_id,
            )
        try:
            aggregate_rubric_score(rubric, ())
        except ValueError as error:
            finding(
                "rubric_aggregation_policy_invalid",
                str(error),
                rubric.rubric_id,
            )

    for definition in bundle.verifier_definitions:
        rubric = rubrics.get(definition.rubric_id)
        if rubric is None:
            finding(
                "verifier_definition_rubric_missing",
                "verifier definition cites a rubric absent from the bundle",
                definition.verifier_id,
            )
        else:
            if rubric.version != definition.rubric_version:
                finding(
                    "verifier_definition_rubric_version_mismatch",
                    "verifier definition cites a different rubric version",
                    definition.verifier_id,
                )
            if rubric.content_digest != definition.rubric_digest:
                finding(
                    "verifier_definition_rubric_digest_mismatch",
                    "verifier definition cites a different rubric digest",
                    definition.verifier_id,
                )

    for result in bundle.verifier_results:
        definition = verifier_definitions.get(result.verifier_id)
        if definition is None:
            finding(
                "verifier_definition_missing",
                "verifier result cites a definition absent from the bundle",
                result.verifier_result_id,
            )
        elif definition.version != result.verifier_version:
            finding(
                "verifier_definition_version_mismatch",
                "verifier result cites a different verifier definition version",
                result.verifier_result_id,
            )
        rubric = rubrics.get(result.rubric_id)
        if rubric is None:
            finding(
                "verifier_rubric_missing",
                "verifier result cites a rubric absent from the bundle",
                result.verifier_result_id,
            )
            continue
        if rubric.content_digest != result.rubric_digest:
            finding(
                "verifier_rubric_digest_mismatch",
                "verifier result cites a different rubric version",
                result.verifier_result_id,
            )
        if definition is not None and (
            definition.rubric_id != result.rubric_id
            or definition.rubric_digest != result.rubric_digest
        ):
            finding(
                "verifier_result_definition_rubric_mismatch",
                "verifier result rubric does not match its verifier definition",
                result.verifier_result_id,
            )
        result_ids = [item.criterion_id for item in result.criterion_results]
        if len(result_ids) != len(set(result_ids)):
            finding(
                "verifier_duplicate_criterion_result",
                "verifier result contains duplicate criterion ids",
                result.verifier_result_id,
            )
        known_criteria = {item.criterion_id for item in rubric.criteria}
        for criterion_id in sorted(set(result_ids) - known_criteria):
            finding(
                "verifier_unknown_criterion",
                "verifier result cites a criterion absent from its rubric",
                criterion_id,
            )
        execution_status = str(result.execution_status).lower()
        verification_status = str(result.verification_status).lower()
        if execution_status not in {item.value for item in ExecutionStatus}:
            finding(
                "verifier_execution_status_invalid",
                "verifier result declares an unsupported execution status",
                result.verifier_result_id,
            )
        if verification_status not in {item.value for item in VerificationStatus}:
            finding(
                "verifier_verification_status_invalid",
                "verifier result declares an unsupported verification status",
                result.verifier_result_id,
            )
        if result.score is not None and not math.isfinite(float(result.score)):
            finding(
                "verifier_score_nonfinite",
                "verifier result score must be finite",
                result.verifier_result_id,
            )
        if (
            execution_status != ExecutionStatus.COMPLETED
            and verification_status == VerificationStatus.VALID
        ):
            finding(
                "verifier_status_inconsistent",
                "an incomplete verifier execution cannot be scientifically valid",
                result.verifier_result_id,
            )
        if execution_status != ExecutionStatus.COMPLETED and result.passed is not None:
            finding(
                "verifier_execution_pass_inconsistent",
                "an incomplete verifier execution cannot make a pass decision",
                result.verifier_result_id,
            )
        if verification_status != VerificationStatus.VALID and result.passed is not None:
            finding(
                "verifier_validity_pass_inconsistent",
                "an invalid or inconclusive verification cannot make a pass decision",
                result.verifier_result_id,
            )
        if (
            str(result.state) == RecordState.INVALIDATED
            and not result.invalidation_reason
        ):
            finding(
                "verifier_invalidation_reason_missing",
                "an invalidated verifier result must state why it was invalidated",
                result.verifier_result_id,
            )
        rubric_by_id = {item.criterion_id: item for item in rubric.criteria}
        for criterion_result in result.criterion_results:
            criterion = rubric_by_id.get(criterion_result.criterion_id)
            if criterion is None:
                continue
            if criterion_result.confidence is not None and not (
                math.isfinite(float(criterion_result.confidence))
                and 0.0 <= criterion_result.confidence <= 1.0
            ):
                finding(
                    "verifier_criterion_confidence_invalid",
                    "criterion confidence must be finite and between zero and one",
                    criterion_result.criterion_id,
                )
            if criterion_result.score is not None:
                score = float(criterion_result.score)
                if not math.isfinite(score):
                    finding(
                        "verifier_criterion_score_nonfinite",
                        "criterion result score must be finite",
                        criterion_result.criterion_id,
                    )
                elif not criterion.min_score <= score <= criterion.max_score:
                    finding(
                        "verifier_criterion_score_out_of_bounds",
                        "criterion result score is outside its declared range",
                        criterion_result.criterion_id,
                    )
                expected_criterion_pass = (
                    score >= criterion.pass_threshold
                    if criterion.higher_is_better
                    else score <= criterion.pass_threshold
                )
                if (
                    criterion_result.passed is not None
                    and criterion_result.passed != expected_criterion_pass
                ):
                    finding(
                        "verifier_criterion_pass_mismatch",
                        "criterion pass decision disagrees with its score threshold",
                        criterion_result.criterion_id,
                    )
            verdict = str(criterion_result.verdict).lower()
            if verdict in {"pass", "passed"} and criterion_result.passed is not True:
                finding(
                    "verifier_criterion_verdict_mismatch",
                    "passing criterion verdict must carry passed=true",
                    criterion_result.criterion_id,
                )
            if (
                verdict in {"fail", "failed", "failure"}
                and criterion_result.passed is not False
            ):
                finding(
                    "verifier_criterion_verdict_mismatch",
                    "failing criterion verdict must carry passed=false",
                    criterion_result.criterion_id,
                )
            if (
                verdict in {
                    "invalid",
                    "inconclusive",
                    "abstain",
                    "abstained",
                    "not_applicable",
                }
                and criterion_result.passed is not None
            ):
                finding(
                    "verifier_criterion_verdict_mismatch",
                    "non-decisive criterion verdict cannot carry a pass decision",
                    criterion_result.criterion_id,
                )
            if verdict == "not_applicable" and not criterion.allows_not_applicable:
                finding(
                    "verifier_criterion_not_applicable_disallowed",
                    "criterion result is not-applicable but its definition disallows it",
                    criterion_result.criterion_id,
                )
            if (
                verdict in {"inconclusive", "abstain", "abstained"}
                and not criterion.allows_abstention
            ):
                finding(
                    "verifier_criterion_abstention_disallowed",
                    "criterion result abstains but its definition disallows abstention",
                    criterion_result.criterion_id,
                )
        if result.pass_threshold is not None and not same_number(
            result.pass_threshold,
            rubric.aggregation.pass_threshold,
        ):
            finding(
                "verifier_threshold_mismatch",
                "verifier result threshold differs from its rubric aggregation threshold",
                result.verifier_result_id,
            )
        if (
            execution_status == ExecutionStatus.COMPLETED
            and verification_status == VerificationStatus.VALID
            and len(result_ids) == len(set(result_ids))
            and set(result_ids) <= known_criteria
        ):
            try:
                expected_score, expected_passed, _ = aggregate_rubric_score(
                    rubric,
                    result.criterion_results,
                )
            except ValueError as error:
                finding(
                    "verifier_rubric_aggregation_invalid",
                    str(error),
                    result.verifier_result_id,
                )
            else:
                if not same_number(result.score, expected_score):
                    finding(
                        "verifier_score_mismatch",
                        "verifier result score does not match rubric recomputation",
                        result.verifier_result_id,
                    )
                if result.passed is None or bool(result.passed) != expected_passed:
                    finding(
                        "verifier_pass_mismatch",
                        "verifier pass decision does not match rubric recomputation",
                        result.verifier_result_id,
                    )

    for record in bundle.reward_records:
        definition = reward_definitions.get(record.reward_id)
        if definition is None:
            finding(
                "reward_definition_missing",
                "reward record cites a definition absent from the bundle",
                record.reward_record_id,
            )
        else:
            if definition.version != record.reward_version:
                finding(
                    "reward_definition_version_mismatch",
                    "reward record cites a different reward definition version",
                    record.reward_record_id,
                )
            if definition.content_digest != record.reward_digest:
                finding(
                    "reward_definition_digest_mismatch",
                    "reward record cites a different reward definition digest",
                    record.reward_record_id,
                )
            for field_name in ("value", "raw_value", "normalized_value"):
                value = getattr(record, field_name)
                if value is not None and not math.isfinite(float(value)):
                    finding(
                        "reward_value_nonfinite",
                        "reward scalar values must be finite",
                        record.reward_record_id,
                    )
            if record.value is not None:
                if (
                    definition.lower_bound is not None
                    and record.value < definition.lower_bound
                ):
                    finding(
                        "reward_value_out_of_bounds",
                        "reward value is below its definition lower bound",
                        record.reward_record_id,
                    )
                if (
                    definition.upper_bound is not None
                    and record.value > definition.upper_bound
                ):
                    finding(
                        "reward_value_out_of_bounds",
                        "reward value is above its definition upper bound",
                        record.reward_record_id,
                    )
            if record.normalized_value is not None and (
                (
                    definition.lower_bound is not None
                    and record.normalized_value < definition.lower_bound
                )
                or (
                    definition.upper_bound is not None
                    and record.normalized_value > definition.upper_bound
                )
            ):
                finding(
                    "reward_normalized_value_out_of_bounds",
                    "normalized reward value is outside its definition bounds",
                    record.reward_record_id,
                )
            if any(
                not math.isfinite(float(value))
                for value in record.components.values()
            ):
                finding(
                    "reward_component_nonfinite",
                    "reward vector components must be finite",
                    record.reward_record_id,
                )
            unknown_components = set(record.components) - set(
                definition.vector_components
            )
            if definition.vector_components and unknown_components:
                finding(
                    "reward_component_unknown",
                    "reward record contains a component absent from its definition",
                    record.reward_record_id,
                )
            missing_components = set(definition.vector_components) - set(
                record.components
            )
            if missing_components and definition.missing_behavior == "fail":
                finding(
                    "reward_component_missing",
                    "reward record omits a required vector component",
                    record.reward_record_id,
                )
        if len(record.source_result_ids) != len(set(record.source_result_ids)):
            finding(
                "reward_source_result_duplicate",
                "reward record cites a source result more than once",
                record.reward_record_id,
            )
        for source_result_id in record.source_result_ids:
            if source_result_id == record.reward_record_id:
                finding(
                    "reward_source_result_cycle",
                    "reward record cannot cite itself as a source result",
                    record.reward_record_id,
                )
            elif source_result_id not in source_result_ids:
                finding(
                    "reward_source_result_missing",
                    "reward record cites a source result absent from the bundle",
                    source_result_id,
                )

    for aggregation in bundle.reward_aggregations:
        definition = reward_definitions.get(aggregation.reward_id)
        if definition is None:
            finding(
                "reward_aggregation_definition_missing",
                "reward aggregation cites a definition absent from the bundle",
                aggregation.aggregation_id,
            )
        elif definition.content_digest != aggregation.definition_digest:
            finding(
                "reward_aggregation_definition_digest_mismatch",
                "reward aggregation cites a different reward definition digest",
                aggregation.aggregation_id,
            )
        if len(aggregation.input_reward_record_ids) != len(aggregation.input_digests):
            finding(
                "reward_aggregation_input_count_mismatch",
                "reward aggregation input ids and digests have different lengths",
                aggregation.aggregation_id,
            )
        if aggregation.value is not None and not math.isfinite(
            float(aggregation.value)
        ):
            finding(
                "reward_aggregation_value_nonfinite",
                "reward aggregation value must be finite",
                aggregation.aggregation_id,
            )
        if any(
            not math.isfinite(float(value))
            for value in aggregation.components.values()
        ):
            finding(
                "reward_aggregation_component_nonfinite",
                "reward aggregation components must be finite",
                aggregation.aggregation_id,
            )
        if len(aggregation.input_reward_record_ids) != len(
            set(aggregation.input_reward_record_ids)
        ):
            finding(
                "reward_aggregation_duplicate_input",
                "reward aggregation consumes the same input record more than once",
                aggregation.aggregation_id,
            )
        resolved_inputs: list[Any] = []
        for index_value, reward_record_id in enumerate(
            aggregation.input_reward_record_ids
        ):
            input_record = reward_records.get(reward_record_id)
            if input_record is None:
                finding(
                    "reward_aggregation_input_missing",
                    "reward aggregation input record is absent from the bundle",
                    reward_record_id,
                )
            elif (
                index_value >= len(aggregation.input_digests)
                or aggregation.input_digests[index_value] != input_record.content_digest
            ):
                finding(
                    "reward_aggregation_input_digest_mismatch",
                    "reward aggregation input digest does not match its record",
                    reward_record_id,
                )
            else:
                resolved_inputs.append(input_record)
        if definition is not None and len(resolved_inputs) == len(
            aggregation.input_reward_record_ids
        ):
            try:
                expected_value, expected_components = aggregate_reward_values(
                    definition,
                    tuple(resolved_inputs),
                    aggregation,
                )
            except ValueError as error:
                finding(
                    "reward_aggregation_calculation_invalid",
                    str(error),
                    aggregation.aggregation_id,
                )
            else:
                if not same_number(aggregation.value, expected_value):
                    finding(
                        "reward_aggregation_value_mismatch",
                        "reward aggregation value does not match its declared calculation",
                        aggregation.aggregation_id,
                    )
                if set(aggregation.components) != set(expected_components) or any(
                    not same_number(aggregation.components.get(component), value)
                    for component, value in expected_components.items()
                ):
                    finding(
                        "reward_aggregation_components_mismatch",
                        "reward aggregation components do not match their declared calculation",
                        aggregation.aggregation_id,
                    )
                if aggregation.value is not None:
                    if (
                        definition.lower_bound is not None
                        and aggregation.value < definition.lower_bound
                    ) or (
                        definition.upper_bound is not None
                        and aggregation.value > definition.upper_bound
                    ):
                        finding(
                            "reward_aggregation_value_out_of_bounds",
                            "reward aggregation value is outside its definition bounds",
                            aggregation.aggregation_id,
                        )

    reward_dependencies: dict[str, set[str]] = {
        record.reward_record_id: {
            source_id
            for source_id in record.source_result_ids
            if source_id in reward_records or source_id in reward_aggregations
        }
        for record in bundle.reward_records
    }
    reward_dependencies.update(
        {
            aggregation.aggregation_id: set(
                aggregation.input_reward_record_ids
            )
            for aggregation in bundle.reward_aggregations
        }
    )
    reward_cycle_reported: set[str] = set()
    for start in reward_dependencies:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit_reward(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(
                visit_reward(dependency)
                for dependency in reward_dependencies.get(node, set())
            ):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        if visit_reward(start) and start not in reward_cycle_reported:
            reward_cycle_reported.add(start)
            finding(
                "reward_source_result_cycle",
                "reward source and aggregation dependencies contain a cycle",
                start,
            )

    annotation_dependencies: dict[str, set[str]] = {}
    for annotation in bundle.annotations:
        definition = annotator_definitions.get(annotation.annotator_id)
        if definition is None:
            finding(
                "annotator_definition_missing",
                "annotation cites an annotator absent from the bundle",
                annotation.annotation_id,
            )
        else:
            if definition.version != annotation.annotator_version:
                finding(
                    "annotator_definition_version_mismatch",
                    "annotation cites a different annotator definition version",
                    annotation.annotation_id,
                )
            if definition.content_digest != annotation.annotator_digest:
                finding(
                    "annotator_definition_digest_mismatch",
                    "annotation cites a different annotator definition digest",
                    annotation.annotation_id,
                )
            producer_kind = str(annotation.producer.kind).strip().lower()
            author_kind = str(annotation.author_kind).strip().lower()
            output_contract = definition.output_contract
            typed_annotation = (
                output_contract is not None or annotation.status is not None
            )
            if (
                typed_annotation
                and producer_kind not in {str(value) for value in ProducerKind}
            ):
                finding(
                    "annotation_producer_kind_invalid",
                    "annotation producer kind is unsupported",
                    annotation.annotation_id,
                )
            if typed_annotation and author_kind != producer_kind:
                finding(
                    "annotation_author_producer_mismatch",
                    "annotation author kind must match its typed producer kind",
                    annotation.annotation_id,
                )
            if (
                typed_annotation
                and producer_kind == ProducerKind.AGENTIC
                and not (
                    annotation.annotator_execution_trace_id
                    and annotation.annotator_execution_trace_digest
                )
            ):
                finding(
                    "annotation_agentic_execution_trace_missing",
                    "agentic annotation producer must cite its execution trace id and digest",
                    annotation.annotation_id,
                )
            if bool(annotation.annotator_execution_trace_id) != bool(
                annotation.annotator_execution_trace_digest
            ):
                finding(
                    "annotation_execution_trace_reference_incomplete",
                    "annotation execution trace reference must include both id and digest",
                    annotation.annotation_id,
                )
            if (
                definition.model
                and annotation.producer.model
                and definition.model != annotation.producer.model
            ):
                finding(
                    "annotation_producer_model_mismatch",
                    "annotation producer model differs from its annotator definition",
                    annotation.annotation_id,
                )
            if output_contract is not None:
                if annotation.annotation_type not in output_contract.annotation_types:
                    finding(
                        "annotation_type_unsupported",
                        "annotation type is absent from its annotator output contract",
                        annotation.annotation_id,
                    )
                allowed_producers = {
                    str(value).lower()
                    for value in output_contract.allowed_producer_kinds
                }
                if allowed_producers and producer_kind not in allowed_producers:
                    finding(
                        "annotation_producer_disallowed",
                        "annotation producer kind is disallowed by its output contract",
                        annotation.annotation_id,
                    )
            if len(annotation.labels) != len(set(annotation.labels)):
                finding(
                    "annotation_duplicate_label",
                    "annotation contains a duplicate taxonomy label",
                    annotation.annotation_id,
                )
            unknown_labels = set(annotation.labels) - set(definition.taxonomy)
            if unknown_labels:
                finding(
                    "annotation_taxonomy_mismatch",
                    "annotation contains a label absent from its annotator taxonomy",
                    annotation.annotation_id,
                )
            status = (
                str(annotation.status).strip().lower()
                if annotation.status is not None
                else str(AnnotationStatus.APPLIED)
            )
            if status not in {str(value) for value in AnnotationStatus}:
                finding(
                    "annotation_status_invalid",
                    "annotation status is unsupported",
                    annotation.annotation_id,
                )
            if output_contract is not None and annotation.status is None:
                finding(
                    "annotation_status_missing",
                    "typed annotator output must declare an annotation status",
                    annotation.annotation_id,
                )
            if annotation.review_state is not None and str(
                annotation.review_state
            ).lower() not in {str(value) for value in AnnotationReviewState}:
                finding(
                    "annotation_review_state_invalid",
                    "annotation review state is unsupported",
                    annotation.annotation_id,
                )
            if str(annotation.visibility).lower() not in {
                str(value) for value in Visibility
            }:
                finding(
                    "annotation_visibility_invalid",
                    "annotation visibility is unsupported",
                    annotation.annotation_id,
                )
            else:
                visibility_rank = {
                    Visibility.PUBLIC: 0,
                    Visibility.OPERATOR: 1,
                    Visibility.PRIVATE: 2,
                }
                document_visibility = str(document.visibility)
                if (
                    document_visibility in visibility_rank
                    and visibility_rank[str(annotation.visibility)]
                    < visibility_rank[document_visibility]
                ):
                    finding(
                        "annotation_visibility_exceeds_trace",
                        "annotation cannot be less restricted than its source trace",
                        annotation.annotation_id,
                    )
            if typed_annotation and annotation.review_state is None:
                finding(
                    "annotation_review_state_missing",
                    "new annotation outcomes must declare their review state",
                    annotation.annotation_id,
                )
            required_scope = str(definition.required_subject_scope).strip().lower()
            if required_scope and required_scope != str(annotation.target.kind).lower():
                finding(
                    "annotation_subject_scope_mismatch",
                    "annotation target does not match the annotator subject scope",
                    annotation.annotation_id,
                )
            unique_evidence = {
                content_digest(selector) for selector in annotation.evidence
            }
            if len(unique_evidence) != len(annotation.evidence):
                finding(
                    "annotation_duplicate_evidence",
                    "annotation cites the same evidence selector more than once",
                    annotation.annotation_id,
                )
            if (
                status == AnnotationStatus.APPLIED
                and len(unique_evidence) < definition.minimum_evidence
            ):
                finding(
                    "annotation_minimum_evidence_unmet",
                    "annotation cites less evidence than its annotator requires",
                    annotation.annotation_id,
                )
            grounding_requirement = str(
                definition.grounding_requirement
            ).strip().lower()
            grounding = str(annotation.grounding).strip().lower()
            if (
                status == AnnotationStatus.APPLIED
                and grounding_requirement == AnnotatorGroundingRequirement.EXACT_SELECTOR
                and grounding != str(GroundingStatus.GROUNDED)
            ):
                finding(
                    "annotation_grounding_requirement_unmet",
                    "exact-selector annotator output must be fully grounded",
                    annotation.annotation_id,
                )
            inspected_projection = (
                annotation.inspection.projection_id
                if annotation.inspection is not None
                else None
            )
            if grounding == str(GroundingStatus.SUMMARY_ONLY) and not (
                annotation.inspected_projection
                or annotation.target.source_projection
                or inspected_projection
            ):
                finding(
                    "annotation_summary_projection_missing",
                    "summary-only annotation must name the projection it inspected",
                    annotation.annotation_id,
                )
            if (
                status == AnnotationStatus.APPLIED
                and grounding == str(GroundingStatus.GROUNDED)
                and not annotation.evidence
            ):
                finding(
                    "annotation_grounded_without_evidence",
                    "fully grounded annotation must cite at least one evidence selector",
                    annotation.annotation_id,
                )
            if status == AnnotationStatus.APPLIED and grounding in {
                GroundingStatus.SOURCE_UNAVAILABLE,
                GroundingStatus.INVALID,
            }:
                finding(
                    "annotation_applied_grounding_invalid",
                    "applied annotation cannot use unavailable or invalid grounding",
                    annotation.annotation_id,
                )
            if status == AnnotationStatus.ABSTAINED:
                if annotation.labels or annotation.payload:
                    finding(
                        "annotation_abstention_has_output",
                        "abstained annotation cannot emit labels or a structured payload",
                        annotation.annotation_id,
                    )
                if not annotation.abstention_reason:
                    finding(
                        "annotation_abstention_reason_missing",
                        "abstained annotation must name its reason",
                        annotation.annotation_id,
                    )
                if str(definition.unavailable_evidence_behavior) not in {
                    UnavailableEvidenceBehavior.ABSTAIN,
                    UnavailableEvidenceBehavior.EMIT_UNAVAILABLE,
                }:
                    finding(
                        "annotation_abstention_disallowed",
                        "annotator definition requires unavailable evidence to fail",
                        annotation.annotation_id,
                    )
            elif status == AnnotationStatus.SOURCE_UNAVAILABLE:
                if annotation.labels or annotation.payload:
                    finding(
                        "annotation_source_unavailable_has_output",
                        "source-unavailable annotation cannot emit labels or a structured payload",
                        annotation.annotation_id,
                    )
                if grounding != GroundingStatus.SOURCE_UNAVAILABLE:
                    finding(
                        "annotation_source_unavailable_grounding_mismatch",
                        "source-unavailable annotation must use source-unavailable grounding",
                        annotation.annotation_id,
                    )
                if (
                    str(definition.unavailable_evidence_behavior)
                    != UnavailableEvidenceBehavior.EMIT_UNAVAILABLE
                ):
                    finding(
                        "annotation_source_unavailable_disallowed",
                        "annotator definition does not permit an unavailable output",
                        annotation.annotation_id,
                    )
                if (
                    annotation.unavailable_evidence is None
                    or not annotation.unavailable_evidence.gaps
                ):
                    finding(
                        "annotation_unavailable_evidence_missing",
                        "source-unavailable annotation must record at least one evidence gap",
                        annotation.annotation_id,
                    )
            elif annotation.abstention_reason:
                finding(
                    "annotation_abstention_reason_unexpected",
                    "applied annotation cannot carry an abstention reason",
                    annotation.annotation_id,
                )
            if (
                annotation.unavailable_evidence is not None
                and status == AnnotationStatus.APPLIED
            ):
                finding(
                    "annotation_applied_with_evidence_gaps",
                    "applied annotation cannot claim unavailable required evidence",
                    annotation.annotation_id,
                )
            if annotation.unavailable_evidence is not None:
                if not annotation.unavailable_evidence.gaps:
                    finding(
                        "annotation_empty_evidence_gaps",
                        "annotation evidence-gap record must not be empty",
                        annotation.annotation_id,
                    )
                for gap in annotation.unavailable_evidence.gaps:
                    if not gap.requirement or not gap.reason:
                        finding(
                            "annotation_evidence_gap_incomplete",
                            "annotation evidence gap must name its requirement and reason",
                            annotation.annotation_id,
                        )
                    if gap.attempted_selector is None:
                        continue
                    gap_resolution = resolve_selector(
                        document, gap.attempted_selector
                    )
                    if gap_resolution.resolved:
                        finding(
                            "annotation_unavailable_evidence_resolved",
                            "annotation claims unavailable evidence that resolves successfully",
                            annotation.annotation_id,
                        )
                    elif gap.reason != gap_resolution.reason:
                        finding(
                            "annotation_evidence_gap_reason_mismatch",
                            "annotation evidence-gap reason differs from selector resolution",
                            annotation.annotation_id,
                        )
            if annotation.inspection is not None:
                inspection = annotation.inspection
                inspection_source = str(inspection.source).strip().lower()
                if inspection_source not in {
                    str(value) for value in AnnotationInspectionSource
                }:
                    finding(
                        "annotation_inspection_source_invalid",
                        "annotation inspection source is unsupported",
                        annotation.annotation_id,
                    )
                if (
                    inspection_source == AnnotationInspectionSource.TRACE_AUTHORITY
                    and not inspection.trace_body_read
                ):
                    finding(
                        "annotation_trace_authority_not_read",
                        "trace-authority inspection must confirm the trace body was read",
                        annotation.annotation_id,
                    )
                if inspection_source == AnnotationInspectionSource.PROJECTION:
                    if not (
                        inspection.projection_id
                        and inspection.projection_digest
                        and inspection.projection_manifest_digest
                    ):
                        finding(
                            "annotation_projection_provenance_incomplete",
                            "projection inspection must cite its id, body digest, and manifest digest",
                            annotation.annotation_id,
                        )
                if (
                    annotation.inspected_projection
                    and inspection.projection_id
                    and annotation.inspected_projection != inspection.projection_id
                ):
                    finding(
                        "annotation_projection_identity_mismatch",
                        "legacy and typed inspected-projection identities disagree",
                        annotation.annotation_id,
                    )
                loss_keys = {
                    (item.field_path, item.reason, item.record_count)
                    for item in inspection.losses
                }
                if len(loss_keys) != len(inspection.losses):
                    finding(
                        "annotation_projection_loss_duplicate",
                        "annotation inspection repeats a projection loss",
                        annotation.annotation_id,
                    )
            elif typed_annotation:
                finding(
                    "annotation_inspection_missing",
                    "new annotation outcomes must record what trace authority or projection was inspected",
                    annotation.annotation_id,
                )
            if (
                annotation.inspection is not None
                and grounding == GroundingStatus.GROUNDED
                and not annotation.inspection.trace_body_read
            ):
                finding(
                    "annotation_grounded_without_trace_read",
                    "grounded annotation must confirm the trace body was read",
                    annotation.annotation_id,
                )
            payload_schema = (
                output_contract.payload_schema
                if output_contract is not None
                else None
            )
            if payload_schema is not None and status == AnnotationStatus.APPLIED:
                fields_by_name = {
                    item.field_name: item for item in payload_schema.fields
                }
                missing_payload_fields = {
                    name
                    for name, payload_field in fields_by_name.items()
                    if payload_field.required and name not in annotation.payload
                }
                if missing_payload_fields:
                    finding(
                        "annotation_payload_required_field_missing",
                        "annotation payload omits a required typed field",
                        annotation.annotation_id,
                    )
                unknown_payload_fields = set(annotation.payload) - set(fields_by_name)
                if (
                    unknown_payload_fields
                    and not payload_schema.additional_fields_allowed
                ):
                    finding(
                        "annotation_payload_unknown_field",
                        "annotation payload contains a field absent from its schema",
                        annotation.annotation_id,
                    )
                for field_name, payload_value in annotation.payload.items():
                    payload_field = fields_by_name.get(field_name)
                    if payload_field is None:
                        continue
                    if not _annotation_payload_value_matches(
                        payload_value, str(payload_field.value_kind).lower()
                    ):
                        finding(
                            "annotation_payload_field_type_mismatch",
                            "annotation payload field does not match its declared value kind",
                            annotation.annotation_id,
                        )
                    if payload_field.allowed_values and not any(
                        payload_value == allowed
                        for allowed in payload_field.allowed_values
                    ):
                        finding(
                            "annotation_payload_field_value_disallowed",
                            "annotation payload field value is absent from its allowed values",
                            annotation.annotation_id,
                        )
            if annotation.confidence is not None and not (
                math.isfinite(float(annotation.confidence))
                and 0.0 <= annotation.confidence <= 1.0
            ):
                finding(
                    "annotation_confidence_invalid",
                    "annotation confidence must be finite and between zero and one",
                    annotation.annotation_id,
                )
            confidence_semantics = str(definition.confidence_semantics).lower()
            if (
                confidence_semantics == ConfidenceSemantics.NONE
                and annotation.confidence is not None
            ):
                finding(
                    "annotation_confidence_unexpected",
                    "annotator declares no confidence semantics but output has confidence",
                    annotation.annotation_id,
                )
            if (
                confidence_semantics == ConfidenceSemantics.DETERMINISTIC
                and annotation.confidence is not None
                and annotation.confidence != 1.0
            ):
                finding(
                    "annotation_deterministic_confidence_invalid",
                    "deterministic confidence, when present, must be one",
                    annotation.annotation_id,
                )
            if (
                confidence_semantics
                == ConfidenceSemantics.INTER_ANNOTATOR_AGREEMENT
                and annotation.derivation is None
            ):
                finding(
                    "annotation_agreement_derivation_missing",
                    "inter-annotator agreement confidence must cite its derivation",
                    annotation.annotation_id,
                )
            if annotation.derivation is not None:
                derivation = annotation.derivation
                source_ids = derivation.source_annotation_ids
                annotation_dependencies[annotation.annotation_id] = set(source_ids)
                if str(derivation.kind).lower() not in {
                    str(value) for value in AnnotationDerivationKind
                }:
                    finding(
                        "annotation_derivation_kind_invalid",
                        "annotation derivation kind is unsupported",
                        annotation.annotation_id,
                    )
                if len(source_ids) != len(set(source_ids)):
                    finding(
                        "annotation_derivation_duplicate_source",
                        "annotation derivation cites a source annotation more than once",
                        annotation.annotation_id,
                    )
                minimum_sources = (
                    2
                    if str(derivation.kind)
                    == AnnotationDerivationKind.CONSENSUS
                    else 1
                )
                if len(source_ids) < minimum_sources:
                    finding(
                        "annotation_derivation_sources_insufficient",
                        "annotation derivation cites too few source annotations",
                        annotation.annotation_id,
                    )
                if not derivation.method:
                    finding(
                        "annotation_derivation_method_missing",
                        "annotation derivation must name its method",
                        annotation.annotation_id,
                    )
                if derivation.agreement is not None and not (
                    math.isfinite(float(derivation.agreement))
                    and 0.0 <= derivation.agreement <= 1.0
                ):
                    finding(
                        "annotation_derivation_agreement_invalid",
                        "annotation derivation agreement must be between zero and one",
                        annotation.annotation_id,
                    )
                if (
                    str(derivation.kind) == AnnotationDerivationKind.CONSENSUS
                    and derivation.agreement is None
                ):
                    finding(
                        "annotation_consensus_agreement_missing",
                        "consensus annotation must report agreement",
                        annotation.annotation_id,
                    )
                if not set(derivation.dissenting_annotation_ids).issubset(
                    set(source_ids)
                ):
                    finding(
                        "annotation_derivation_dissent_source_missing",
                        "dissenting annotation ids must be a subset of derivation sources",
                        annotation.annotation_id,
                    )
                if len(derivation.dissenting_annotation_ids) != len(
                    set(derivation.dissenting_annotation_ids)
                ):
                    finding(
                        "annotation_derivation_duplicate_dissent",
                        "annotation derivation repeats a dissenting annotation id",
                        annotation.annotation_id,
                    )
                if status != AnnotationStatus.APPLIED:
                    finding(
                        "annotation_derivation_not_applied",
                        "consensus or adjudication must produce an applied annotation",
                        annotation.annotation_id,
                    )
                for source_id in source_ids:
                    source = annotations.get(source_id)
                    if source is None:
                        finding(
                            "annotation_derivation_source_missing",
                            "annotation derivation cites an annotation absent from the bundle",
                            annotation.annotation_id,
                        )
                    elif (
                        source.target != annotation.target
                        or source.annotation_type != annotation.annotation_type
                    ):
                        finding(
                            "annotation_derivation_subject_mismatch",
                            "annotation derivation crosses target or annotation type",
                            annotation.annotation_id,
                        )

    annotation_cycle_reported: set[str] = set()
    for start in annotation_dependencies:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit_annotation(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(
                visit_annotation(dependency)
                for dependency in annotation_dependencies.get(node, set())
            ):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        if (
            visit_annotation(start)
            and start not in annotation_cycle_reported
        ):
            annotation_cycle_reported.add(start)
            finding(
                "annotation_derivation_cycle",
                "annotation derivation dependencies contain a cycle",
                start,
            )

    def score_value(record: Any) -> float | None:
        for field_name in ("aggregate_score", "score", "value"):
            value = getattr(record, field_name, None)
            if value is not None:
                return float(value)
        return None

    score_sources: dict[str, Any] = {
        **evaluation_results,
        **verifier_results,
        **reward_records,
        **reward_aggregations,
    }

    for evaluation in bundle.evaluation_results:
        if len(evaluation.environment_reward_record_ids) != len(
            set(evaluation.environment_reward_record_ids)
        ):
            finding(
                "evaluation_duplicate_reward_input",
                "evaluation cites a reward input more than once",
                evaluation.evaluation_id,
            )
        if len(evaluation.verifier_result_ids) != len(
            set(evaluation.verifier_result_ids)
        ):
            finding(
                "evaluation_duplicate_verifier_input",
                "evaluation cites a verifier input more than once",
                evaluation.evaluation_id,
            )
        if len(evaluation.rubric_ids) != len(set(evaluation.rubric_ids)):
            finding(
                "evaluation_duplicate_rubric_input",
                "evaluation cites a rubric more than once",
                evaluation.evaluation_id,
            )
        linked_inputs: dict[str, Any] = {}
        for reward_record_id in evaluation.environment_reward_record_ids:
            reward_record = reward_records.get(reward_record_id)
            if reward_record is None:
                finding(
                    "evaluation_reward_input_missing",
                    "evaluation cites a reward record absent from the bundle",
                    evaluation.evaluation_id,
                )
            else:
                linked_inputs[reward_record_id] = reward_record
        for verifier_result_id in evaluation.verifier_result_ids:
            verifier_result = verifier_results.get(verifier_result_id)
            if verifier_result is None:
                finding(
                    "evaluation_verifier_input_missing",
                    "evaluation cites a verifier result absent from the bundle",
                    evaluation.evaluation_id,
                )
            else:
                linked_inputs[verifier_result_id] = verifier_result
                if verifier_result.rubric_id not in evaluation.rubric_ids:
                    finding(
                        "evaluation_verifier_rubric_unlinked",
                        "evaluation omits the rubric used by one of its verifier results",
                        evaluation.evaluation_id,
                    )
        for rubric_id in evaluation.rubric_ids:
            if rubric_id not in rubrics:
                finding(
                    "evaluation_rubric_input_missing",
                    "evaluation cites a rubric absent from the bundle",
                    evaluation.evaluation_id,
                )

        execution_status = str(evaluation.execution_status).lower()
        if execution_status not in {item.value for item in ExecutionStatus}:
            finding(
                "evaluation_execution_status_invalid",
                "evaluation declares an unsupported execution status",
                evaluation.evaluation_id,
            )
        if execution_status != ExecutionStatus.COMPLETED and evaluation.aggregate_score is not None:
            finding(
                "evaluation_status_score_inconsistent",
                "an incomplete evaluation cannot publish an aggregate score",
                evaluation.evaluation_id,
            )
        if execution_status == ExecutionStatus.COMPLETED and evaluation.error:
            finding(
                "evaluation_status_error_inconsistent",
                "a completed evaluation cannot carry an execution error",
                evaluation.evaluation_id,
            )
        if evaluation.aggregate_score is not None and not math.isfinite(
            float(evaluation.aggregate_score)
        ):
            finding(
                "evaluation_aggregate_nonfinite",
                "evaluation aggregate score must be finite",
                evaluation.evaluation_id,
            )
        if any(
            not math.isfinite(float(value))
            for value in evaluation.objective_metrics.values()
        ):
            finding(
                "evaluation_metric_nonfinite",
                "evaluation objective metrics must be finite",
                evaluation.evaluation_id,
            )
        if evaluation.threshold is not None and not math.isfinite(
            float(evaluation.threshold)
        ):
            finding(
                "evaluation_threshold_nonfinite",
                "evaluation threshold must be finite",
                evaluation.evaluation_id,
            )
        if str(evaluation.state) == RecordState.SUPERSEDED:
            finding(
                "evaluation_state_inconsistent",
                "evaluation cannot be marked superseded without a supersession link",
                evaluation.evaluation_id,
            )

        declared_source = str(
            evaluation.metadata.get("aggregate_score_source")
            or evaluation.metadata.get("score_source")
            or ""
        )
        expected_score: float | None = None
        aggregate_checkable = False
        if declared_source:
            source = linked_inputs.get(declared_source)
            if source is None:
                finding(
                    "evaluation_aggregate_source_unlinked",
                    "evaluation aggregate source is not one of its declared inputs",
                    evaluation.evaluation_id,
                )
            else:
                expected_score = score_value(source)
                aggregate_checkable = True
        else:
            input_values = tuple(
                value
                for value in (score_value(record) for record in linked_inputs.values())
                if value is not None
            )
            calculation = str(
                evaluation.metadata.get("aggregate_calculation") or ""
            ).strip().lower()
            if calculation and input_values:
                if calculation in {"mean", "arithmetic_mean"}:
                    expected_score = sum(input_values) / len(input_values)
                elif calculation == "sum":
                    expected_score = sum(input_values)
                elif calculation == "min":
                    expected_score = min(input_values)
                elif calculation == "max":
                    expected_score = max(input_values)
                else:
                    finding(
                        "evaluation_aggregate_calculation_invalid",
                        "evaluation declares an unsupported aggregate calculation",
                        evaluation.evaluation_id,
                    )
                aggregate_checkable = calculation in {
                    "mean",
                    "arithmetic_mean",
                    "sum",
                    "min",
                    "max",
                }
            elif input_values and evaluation.aggregate_score is not None:
                matching = tuple(
                    value
                    for value in input_values
                    if same_number(value, evaluation.aggregate_score)
                )
                if len(matching) == 1 or len(matching) == len(input_values):
                    expected_score = float(evaluation.aggregate_score)
                    aggregate_checkable = True
                else:
                    finding(
                        "evaluation_aggregate_calculation_missing",
                        "evaluation aggregate cannot be derived unambiguously from its inputs",
                        evaluation.evaluation_id,
                    )
            elif "native_score" in evaluation.objective_metrics:
                expected_score = float(evaluation.objective_metrics["native_score"])
                aggregate_checkable = True
        if aggregate_checkable and not same_number(
            evaluation.aggregate_score,
            expected_score,
        ):
            finding(
                "evaluation_aggregate_mismatch",
                "evaluation aggregate score disagrees with its declared inputs",
                evaluation.evaluation_id,
            )

    gating_criteria = {
        criterion.criterion_id
        for rubric in bundle.rubrics
        for criterion in rubric.criteria
        if str(criterion.role) == "gating"
    }
    superseded_verifier_ids = {
        result.supersedes_id
        for result in bundle.verifier_results
        if result.supersedes_id
    }
    for verdict in bundle.benchmark_verdicts:
        decision = str(verdict.decision).strip().lower()
        pass_decisions = {"pass", "passed", "accept", "accepted", "promote", "success"}
        fail_decisions = {"fail", "failed", "reject", "rejected", "do_not_promote"}
        if decision not in pass_decisions | fail_decisions:
            finding(
                "verdict_decision_invalid",
                "benchmark verdict decision is outside the standard pass/fail taxonomy",
                verdict.verdict_id,
            )
        if verdict.threshold is not None and not math.isfinite(
            float(verdict.threshold)
        ):
            finding(
                "verdict_threshold_nonfinite",
                "benchmark verdict threshold must be finite",
                verdict.verdict_id,
            )
        if str(verdict.state) == RecordState.SUPERSEDED:
            finding(
                "verdict_state_inconsistent",
                "benchmark verdict cannot be superseded without a supersession link",
                verdict.verdict_id,
            )
        required_evaluations_completed = True
        required_verifier_ids: set[str] = set()
        required_reward_ids: set[str] = set()
        for evaluation_id in verdict.required_evaluation_ids:
            evaluation = evaluation_results.get(evaluation_id)
            if evaluation is None:
                finding(
                    "verdict_evaluation_input_missing",
                    "benchmark verdict cites an evaluation absent from the bundle",
                    verdict.verdict_id,
                )
                required_evaluations_completed = False
            else:
                required_verifier_ids.update(evaluation.verifier_result_ids)
                required_reward_ids.update(
                    evaluation.environment_reward_record_ids
                )
                if str(evaluation.execution_status) != ExecutionStatus.COMPLETED:
                    finding(
                        "verdict_evaluation_incomplete",
                        "benchmark verdict depends on an incomplete evaluation",
                        verdict.verdict_id,
                    )
                    required_evaluations_completed = False
        gates_passed = True
        for gate_id in verdict.required_gates:
            if gate_id not in gating_criteria:
                finding(
                    "verdict_gate_input_missing",
                    "benchmark verdict cites a gate absent from the bundle rubrics",
                    verdict.verdict_id,
                )
                gates_passed = False
                continue
            gate_outcomes: set[bool] = set()
            for verifier_result in bundle.verifier_results:
                if (
                    (
                        required_verifier_ids
                        and verifier_result.verifier_result_id
                        not in required_verifier_ids
                    )
                    or verifier_result.verifier_result_id in superseded_verifier_ids
                    or str(verifier_result.state)
                    in {
                        RecordState.STALE,
                        RecordState.INVALIDATED,
                        RecordState.SUPERSEDED,
                    }
                    or str(verifier_result.execution_status) != ExecutionStatus.COMPLETED
                    or str(verifier_result.verification_status)
                    != VerificationStatus.VALID
                ):
                    continue
                rubric = rubrics.get(verifier_result.rubric_id)
                criterion = rubric.criterion(gate_id) if rubric is not None else None
                criterion_result = next(
                    (
                        item
                        for item in verifier_result.criterion_results
                        if item.criterion_id == gate_id
                    ),
                    None,
                )
                if criterion is None or criterion_result is None:
                    continue
                if criterion_result.passed is not None:
                    gate_outcomes.add(bool(criterion_result.passed))
                elif criterion_result.score is not None:
                    gate_outcomes.add(
                        criterion_result.score >= criterion.pass_threshold
                        if criterion.higher_is_better
                        else criterion_result.score <= criterion.pass_threshold
                    )
            if not gate_outcomes:
                finding(
                    "verdict_gate_result_missing",
                    "benchmark verdict gate has no current valid criterion result",
                    verdict.verdict_id,
                )
                gates_passed = False
            elif len(gate_outcomes) > 1:
                finding(
                    "verdict_gate_result_ambiguous",
                    "benchmark verdict gate has contradictory current results",
                    verdict.verdict_id,
                )
                gates_passed = False
            elif False in gate_outcomes:
                gates_passed = False

        threshold_passed = True
        source_score: float | None = None
        if verdict.score_source:
            source = score_sources.get(verdict.score_source)
            if source is None:
                finding(
                    "verdict_score_source_missing",
                    "benchmark verdict score source is absent from the bundle",
                    verdict.verdict_id,
                )
                threshold_passed = False
            else:
                source_score = score_value(source)
                source_linked = (
                    not verdict.required_evaluation_ids
                    or verdict.score_source in verdict.required_evaluation_ids
                    or verdict.score_source in required_verifier_ids
                    or verdict.score_source in required_reward_ids
                    or (
                        verdict.score_source in reward_aggregations
                        and set(
                            reward_aggregations[
                                verdict.score_source
                            ].input_reward_record_ids
                        )
                        <= required_reward_ids
                    )
                )
                if not source_linked:
                    finding(
                        "verdict_score_source_unlinked",
                        "benchmark verdict score source is not linked by a required evaluation",
                        verdict.verdict_id,
                    )
                    threshold_passed = False
        if verdict.threshold is not None:
            if source_score is None:
                finding(
                    "verdict_threshold_score_missing",
                    "benchmark verdict threshold has no numeric score source",
                    verdict.verdict_id,
                )
                threshold_passed = False
            else:
                threshold_passed = source_score >= verdict.threshold
        expected_pass = (
            required_evaluations_completed and gates_passed and threshold_passed
        )
        decision_checkable = (
            bool(verdict.required_gates)
            or verdict.threshold is not None
            or not required_evaluations_completed
        )
        if decision_checkable and decision in pass_decisions | fail_decisions and (
            (decision in pass_decisions) != expected_pass
        ):
            finding(
                "verdict_decision_mismatch",
                "benchmark verdict decision disagrees with its evaluations, gates, or threshold",
                verdict.verdict_id,
            )

    def check_supersession_chain(
        records: tuple[Any, ...],
        *,
        id_field: str,
        kind: str,
        logical_key: Any,
        revision_field: str | None = None,
    ) -> None:
        indexed = {str(getattr(record, id_field)): record for record in records}
        children: dict[str, list[str]] = {}
        for record in records:
            record_id_value = str(getattr(record, id_field))
            supersedes = str(getattr(record, "supersedes_id", "") or "")
            if not supersedes:
                if revision_field is not None and getattr(record, revision_field) != 1:
                    finding(
                        f"{kind}_root_revision_invalid",
                        f"{kind} root revision must be one",
                        record_id_value,
                    )
                continue
            parent = indexed.get(supersedes)
            if parent is None:
                finding(
                    f"{kind}_supersedes_missing",
                    f"{kind} supersedes a record absent from the bundle",
                    record_id_value,
                )
                continue
            if supersedes == record_id_value:
                finding(
                    f"{kind}_supersedes_cycle",
                    f"{kind} cannot supersede itself",
                    record_id_value,
                )
                continue
            children.setdefault(supersedes, []).append(record_id_value)
            if logical_key(record) != logical_key(parent):
                finding(
                    f"{kind}_supersedes_subject_mismatch",
                    f"{kind} supersession crosses definition or subject identity",
                    record_id_value,
                )
            if revision_field is not None and getattr(record, revision_field) != (
                getattr(parent, revision_field) + 1
            ):
                finding(
                    f"{kind}_revision_sequence_invalid",
                    f"{kind} revision does not increment its parent by one",
                    record_id_value,
                )
        for parent_id, child_ids in children.items():
            if len(child_ids) > 1:
                finding(
                    f"{kind}_supersedes_fork",
                    f"{kind} record has more than one direct successor",
                    parent_id,
                )
        for record in records:
            record_id_value = str(getattr(record, id_field))
            seen: set[str] = set()
            cursor = record
            while getattr(cursor, "supersedes_id", None):
                cursor_id = str(getattr(cursor, id_field))
                if cursor_id in seen:
                    finding(
                        f"{kind}_supersedes_cycle",
                        f"{kind} supersession chain contains a cycle",
                        record_id_value,
                    )
                    break
                seen.add(cursor_id)
                parent = indexed.get(str(cursor.supersedes_id))
                if parent is None:
                    break
                cursor = parent
            state = str(getattr(record, "state", ""))
            if state == RecordState.SUPERSEDED and record_id_value not in children:
                finding(
                    f"{kind}_state_inconsistent",
                    f"{kind} is marked superseded but has no successor",
                    record_id_value,
                )

    check_supersession_chain(
        bundle.verifier_results,
        id_field="verifier_result_id",
        kind="verifier_result",
        logical_key=lambda item: (
            item.verifier_id,
            item.rubric_id,
            item.subject,
        ),
    )
    check_supersession_chain(
        bundle.reward_records,
        id_field="reward_record_id",
        kind="reward_record",
        logical_key=lambda item: (
            item.reward_id,
            item.subject,
            item.actor_id,
            item.session_id,
        ),
    )
    check_supersession_chain(
        bundle.annotations,
        id_field="annotation_id",
        kind="annotation",
        logical_key=lambda item: (
            item.annotator_id,
            item.target,
            item.annotation_type,
        ),
        revision_field="revision",
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
    "validate_alias_integrity",
    "validate_evidence",
    "validate_trace",
]
