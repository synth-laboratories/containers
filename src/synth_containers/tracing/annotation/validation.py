"""Turn an annotator's proposal into sealed evidence records, or refuse.

This is the boundary between "what a model said" and "what the system records".
Every target and evidence selector is resolved against the sealed source trace;
anything that does not resolve is converted to a typed abstention or fails the
job, according to the annotator definition. Nothing here trusts the proposal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..canonical import record_id, utc_now
from ..models.document import TraceDocumentV5
from ..models.selectors import (
    GroundingStatus,
    SelectorKind,
    SelectorResolutionV1,
    TraceSelectorV1,
)
from ..models.standards import (
    JUDGMENT_SCHEMA_VERSION,
    AnnotationDerivationKind,
    AnnotationDerivationV1,
    AnnotationEvidenceGapsV1,
    AnnotationInspectionSource,
    AnnotationInspectionV1,
    AnnotationReviewState,
    AnnotationStatus,
    AnnotationV1,
    AnnotationValueKind,
    ConfidenceSemantics,
    ExecutionStatus,
    JudgmentStatus,
    JudgmentV1,
    ProducerKind,
    ProducerRefV1,
    RecordState,
    RubricDefinitionV2,
    TraceAnnotatorDefinitionV1,
    UnavailableAnnotationEvidenceV1,
    UnavailableEvidenceBehavior,
    VerificationStatus,
    VerifierDefinitionV1,
    VerifierKind,
    VerifierResultV2,
    aggregate_rubric_score,
)
from .jobs import AnnotationJobErrorCode, AnnotationJobErrorV1, AnnotationJobMode
from .proposal import check_proposal_shape
from .tools import ToolArgumentError, build_selector
from .trace_index import SealedTraceIndex


@dataclass(frozen=True, slots=True)
class RejectedFindingV1:
    index: int
    kind: str
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProposalValidationResult:
    annotations: tuple[AnnotationV1, ...] = ()
    verifier_result: VerifierResultV2 | None = None
    verifier_definition: VerifierDefinitionV1 | None = None
    rejected: tuple[RejectedFindingV1, ...] = ()
    applied_count: int = 0
    abstained_count: int = 0
    fatal: AnnotationJobErrorV1 | None = None
    global_abstentions: tuple[dict[str, Any], ...] = ()

    @property
    def ok(self) -> bool:
        return self.fatal is None


def _payload_value_matches(value: Any, value_kind: str) -> bool:
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


def _resolve(index: SealedTraceIndex, raw: Any) -> tuple[TraceSelectorV1 | None, SelectorResolutionV1 | None, str]:
    """Build and resolve one proposal selector; resolutions are memoized per sealed trace."""

    try:
        selector = build_selector(index.view, raw)
    except (ToolArgumentError, ValueError, KeyError, TypeError) as error:
        return None, None, f"selector_invalid: {error}"
    resolution = index.resolve(selector)
    if not resolution.resolved:
        return selector, resolution, resolution.reason
    return selector, resolution, ""


class ProposalValidator:
    def __init__(
        self,
        document: TraceDocumentV5,
        *,
        definition: TraceAnnotatorDefinitionV1,
        producer: ProducerRefV1,
        job_id: str,
        mode: AnnotationJobMode | str = AnnotationJobMode.ANNOTATE,
        rubric: RubricDefinitionV2 | None = None,
        program_digest: str | None = None,
        execution_trace_id: str | None = None,
        execution_trace_digest: str | None = None,
        existing_annotations: Sequence[AnnotationV1] = (),
        allowed_source_annotation_ids: Sequence[str] = (),
        index: SealedTraceIndex | None = None,
    ) -> None:
        if not document.content_digest:
            raise ValueError("proposal validation requires a sealed document")
        self.document = document
        # A shared index (from the service cache) makes citations repeated across
        # findings, or across jobs on this trace, resolve once. A foreign index is
        # never accepted: the memo is only valid for this exact sealed digest.
        if index is not None and index.trace_digest == document.content_digest and index.trace_id == document.trace_id:
            self.index = index
        else:
            self.index = SealedTraceIndex(document)
        self.definition = definition
        self.producer = producer
        self.job_id = job_id
        self.mode = AnnotationJobMode(str(mode))
        self.rubric = rubric
        self.program_digest = program_digest
        self.execution_trace_id = execution_trace_id
        self.execution_trace_digest = execution_trace_digest
        self.existing = {item.annotation_id: item for item in existing_annotations}
        self.allowed_sources = set(allowed_source_annotation_ids)
        self.created_at = utc_now()
        self._counter = 0

    # -- helpers ----------------------------------------------------------------------

    def _annotation_id(self) -> str:
        self._counter += 1
        return record_id(
            "ann",
            kind="annotation",
            scope=(self.document.trace_id,),
            key={"job_id": self.job_id, "index": self._counter},
        )

    def _trace_selector(self) -> TraceSelectorV1:
        return TraceSelectorV1(
            trace_id=self.document.trace_id,
            trace_digest=self.document.content_digest,
            kind=SelectorKind.TRACE,
        )

    def _inspection(self) -> AnnotationInspectionV1:
        return AnnotationInspectionV1(
            source=AnnotationInspectionSource.TRACE_AUTHORITY,
            trace_body_read=True,
        )

    def _behavior(self) -> str:
        return str(self.definition.unavailable_evidence_behavior)

    def _confidence(self, value: Any) -> tuple[float | None, str]:
        semantics = str(self.definition.confidence_semantics)
        if value is None:
            if semantics == ConfidenceSemantics.DETERMINISTIC:
                return 1.0, ""
            return None, ""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, "confidence_not_numeric"
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            return None, "confidence_out_of_range"
        if semantics == ConfidenceSemantics.NONE:
            return None, ""
        if semantics == ConfidenceSemantics.DETERMINISTIC and number != 1.0:
            return None, "deterministic_confidence_must_be_one"
        return number, ""

    def _base(
        self,
        *,
        target: TraceSelectorV1,
        annotation_type: str,
        status: AnnotationStatus,
        labels: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
        confidence: float | None = None,
        rationale: str = "",
        evidence: tuple[TraceSelectorV1, ...] = (),
        grounding: GroundingStatus = GroundingStatus.GROUNDED,
        abstention_reason: str | None = None,
        gaps: AnnotationEvidenceGapsV1 | None = None,
        derivation: AnnotationDerivationV1 | None = None,
    ) -> AnnotationV1:
        return AnnotationV1(
            annotation_id=self._annotation_id(),
            annotator_id=self.definition.annotator_id,
            annotator_version=self.definition.version,
            annotator_digest=self.definition.content_digest,
            target=target,
            annotation_type=annotation_type,
            labels=labels,
            author_kind=str(self.producer.kind),
            producer=self.producer,
            created_at=self.created_at,
            grounding=grounding,
            payload=dict(payload or {}),
            confidence=confidence,
            rationale=rationale,
            evidence=evidence,
            visibility=self.document.visibility,
            status=status,
            review_state=AnnotationReviewState.UNREVIEWED,
            abstention_reason=abstention_reason,
            unavailable_evidence=gaps,
            inspection=self._inspection(),
            derivation=derivation,
            annotator_execution_trace_id=self.execution_trace_id,
            annotator_execution_trace_digest=self.execution_trace_digest,
        ).sealed()

    def _abstain(
        self,
        *,
        annotation_type: str,
        reason: str,
        requirement: str,
        target: TraceSelectorV1 | None,
        attempted: TraceSelectorV1 | None,
        attempted_reason: str,
        rationale: str = "",
    ) -> AnnotationV1 | None:
        """An abstention needs a resolvable target of the declared scope."""

        scope = str(self.definition.required_subject_scope).strip().lower()
        if target is None or str(target.kind).lower() != scope:
            if scope == SelectorKind.TRACE:
                target = self._trace_selector()
            else:
                return None
        gap_selector = attempted
        gap_reason = attempted_reason or reason
        if gap_selector is not None and self.index.resolve(gap_selector).resolved:
            gap_selector = None
        emit_unavailable = self._behavior() == UnavailableEvidenceBehavior.EMIT_UNAVAILABLE
        return self._base(
            target=target,
            annotation_type=annotation_type,
            status=(
                AnnotationStatus.SOURCE_UNAVAILABLE if emit_unavailable else AnnotationStatus.ABSTAINED
            ),
            confidence=None,
            rationale=rationale,
            grounding=(
                GroundingStatus.SOURCE_UNAVAILABLE if emit_unavailable else GroundingStatus.UNINSPECTED
            ),
            abstention_reason=None if emit_unavailable else reason,
            gaps=AnnotationEvidenceGapsV1(
                gaps=(
                    UnavailableAnnotationEvidenceV1(
                        requirement=requirement or reason,
                        reason=gap_reason,
                        attempted_selector=gap_selector,
                    ),
                )
            ),
        )

    # -- main -------------------------------------------------------------------------

    def validate(self, proposal: Any) -> ProposalValidationResult:
        problems = check_proposal_shape(proposal)
        if problems:
            return ProposalValidationResult(
                fatal=AnnotationJobErrorV1(
                    code=AnnotationJobErrorCode.MALFORMED_OUTPUT,
                    message="annotator output is not a valid proposal",
                    detail={"problems": problems[:50]},
                )
            )
        if proposal["source_trace_id"] != self.document.trace_id or (
            proposal["source_trace_digest"] != self.document.content_digest
        ):
            return ProposalValidationResult(
                fatal=AnnotationJobErrorV1(
                    code=AnnotationJobErrorCode.SOURCE_DIGEST_MISMATCH,
                    message="proposal names a different trace than the job inspected",
                    detail={
                        "expected_trace_id": self.document.trace_id,
                        "expected_trace_digest": self.document.content_digest,
                        "proposal_trace_id": proposal["source_trace_id"],
                        "proposal_trace_digest": proposal["source_trace_digest"],
                    },
                )
            )
        annotations: list[AnnotationV1] = []
        rejected: list[RejectedFindingV1] = []
        global_abstentions: list[dict[str, Any]] = []
        fail_closed = self._behavior() == UnavailableEvidenceBehavior.FAIL

        def unsupported(index: int, kind: str, reason: str, detail: dict[str, Any]) -> AnnotationJobErrorV1 | None:
            rejected.append(RejectedFindingV1(index=index, kind=kind, reason=reason, detail=detail))
            if fail_closed:
                return AnnotationJobErrorV1(
                    code=AnnotationJobErrorCode.UNSUPPORTED_FINDING,
                    message=f"{kind}[{index}] {reason}",
                    detail=detail,
                )
            return None

        for index, finding in enumerate(proposal["findings"]):
            outcome = self._finding(index, finding)
            if isinstance(outcome, AnnotationV1):
                annotations.append(outcome)
                continue
            reason, detail, fallback = outcome
            fatal = unsupported(index, "finding", reason, detail)
            if fatal is not None:
                return ProposalValidationResult(rejected=tuple(rejected), fatal=fatal)
            if fallback is not None:
                annotations.append(fallback)

        if proposal["abstentions"] and fail_closed:
            return ProposalValidationResult(
                rejected=tuple(rejected),
                fatal=AnnotationJobErrorV1(
                    code=AnnotationJobErrorCode.UNSUPPORTED_FINDING,
                    message="annotator abstained but its definition requires failure on missing evidence",
                    detail={"abstentions": len(proposal["abstentions"])},
                ),
            )
        for index, abstention in enumerate(proposal["abstentions"]):
            target, _, _ = (
                _resolve(self.index, abstention["target"])
                if abstention.get("target") is not None
                else (None, None, "")
            )
            attempted, _, attempted_reason = (
                _resolve(self.index, abstention["attempted_selector"])
                if abstention.get("attempted_selector") is not None
                else (None, None, "")
            )
            record = self._abstain(
                annotation_type=str(abstention["annotation_type"]),
                reason=str(abstention["reason"]),
                requirement=str(abstention.get("requirement") or abstention["reason"]),
                target=target,
                attempted=attempted,
                attempted_reason=attempted_reason,
            )
            if record is None:
                if target is None and abstention.get("attempted_selector") is None:
                    # A whole-trace abstention from a narrower-scoped annotator ("there are no
                    # tool calls here") is a legitimate outcome, recorded on the job, not a rejection.
                    global_abstentions.append({"index": index, "annotation_type": str(abstention["annotation_type"]), "reason": str(abstention["reason"]), "requirement": str(abstention.get("requirement") or "")})
                    continue
                rejected.append(
                    RejectedFindingV1(
                        index=index,
                        kind="abstention",
                        reason="abstention target does not resolve to the annotator subject scope",
                    )
                )
                continue
            annotations.append(record)

        verifier_result: VerifierResultV2 | None = None
        verifier_definition: VerifierDefinitionV1 | None = None
        if self.mode == AnnotationJobMode.VERIFY:
            if self.rubric is None:
                return ProposalValidationResult(
                    rejected=tuple(rejected),
                    fatal=AnnotationJobErrorV1(
                        code=AnnotationJobErrorCode.RUBRIC_REQUIRED,
                        message="verification jobs need a sealed rubric",
                    ),
                )
            judgments = proposal.get("judgments") or []
            if not judgments and (proposal.get("findings") or []):
                return ProposalValidationResult(
                    rejected=tuple(rejected),
                    fatal=AnnotationJobErrorV1(
                        code=AnnotationJobErrorCode.MALFORMED_OUTPUT,
                        message="verification jobs must return judgments, not findings",
                    ),
                )
            outcome = self._judgments(judgments, rejected)
            if isinstance(outcome, AnnotationJobErrorV1):
                return ProposalValidationResult(rejected=tuple(rejected), fatal=outcome)
            verifier_definition, verifier_result = outcome

        applied = sum(
            1 for item in annotations if str(item.status) == AnnotationStatus.APPLIED
        )
        return ProposalValidationResult(
            annotations=tuple(annotations),
            verifier_result=verifier_result,
            verifier_definition=verifier_definition,
            rejected=tuple(rejected),
            applied_count=applied,
            abstained_count=len(annotations) - applied + len(global_abstentions),
            global_abstentions=tuple(global_abstentions),
        )

    # -- findings ---------------------------------------------------------------------

    def _finding(
        self, index: int, finding: dict[str, Any]
    ) -> AnnotationV1 | tuple[str, dict[str, Any], AnnotationV1 | None]:
        definition = self.definition
        contract = definition.output_contract
        annotation_type = str(finding["annotation_type"])
        labels = tuple(dict.fromkeys(str(item) for item in finding["labels"]))
        payload = dict(finding.get("payload") or {})
        rationale = str(finding.get("rationale") or "")

        def reject(reason: str, detail: dict[str, Any], *, target: TraceSelectorV1 | None, attempted: TraceSelectorV1 | None = None, attempted_reason: str = "") -> tuple[str, dict[str, Any], AnnotationV1 | None]:
            fallback = None
            if self._behavior() != UnavailableEvidenceBehavior.FAIL:
                fallback = self._abstain(
                    annotation_type=annotation_type,
                    reason=reason,
                    requirement=str(detail.get("requirement") or reason),
                    target=target,
                    attempted=attempted,
                    attempted_reason=attempted_reason,
                    rationale=rationale[:500],
                )
            return reason, detail, fallback

        if contract is not None and annotation_type not in contract.annotation_types:
            return reject(
                "annotation_type_unsupported",
                {"annotation_type": annotation_type, "allowed": list(contract.annotation_types)},
                target=None,
            )
        unknown = sorted(set(labels) - set(definition.taxonomy))
        if unknown:
            return reject(
                "label_not_in_taxonomy",
                {"unknown_labels": unknown},
                target=None,
            )
        if not labels:
            return reject("labels_empty", {}, target=None)

        target, target_resolution, target_reason = _resolve(self.index, finding["target"])
        if target is None or target_resolution is None or not target_resolution.resolved:
            return reject(
                "target_unresolved",
                {"reason": target_reason, "target": finding["target"], "requirement": "target selector"},
                target=None,
                attempted=target,
                attempted_reason=target_reason if target is not None else "",
            )
        scope = str(definition.required_subject_scope).strip().lower()
        if scope and str(target.kind).lower() != scope:
            return reject(
                "target_scope_mismatch",
                {"expected": scope, "actual": str(target.kind)},
                target=None,
            )

        evidence: dict[str, TraceSelectorV1] = {}
        for eindex, raw in enumerate(finding["evidence"]):
            selector, resolution, reason = _resolve(self.index, raw)
            if selector is None or resolution is None or not resolution.resolved:
                return reject(
                    "evidence_unresolved",
                    {"evidence_index": eindex, "reason": reason, "selector": raw, "requirement": "evidence selector"},
                    target=target,
                    attempted=selector,
                    attempted_reason=reason if selector is not None else "",
                )
            evidence.setdefault(self.index.selector_digest(selector), selector)
        if len(evidence) < definition.minimum_evidence:
            return reject(
                "minimum_evidence_unmet",
                {"cited": len(evidence), "required": definition.minimum_evidence, "requirement": "minimum evidence"},
                target=target,
            )

        confidence, confidence_problem = self._confidence(finding.get("confidence"))
        if confidence_problem:
            return reject(confidence_problem, {"confidence": finding.get("confidence")}, target=target)

        payload_schema = contract.payload_schema if contract is not None else None
        if payload_schema is not None:
            fields_by_name = {item.field_name: item for item in payload_schema.fields}
            missing = sorted(
                name for name, item in fields_by_name.items() if item.required and name not in payload
            )
            if missing:
                return reject("payload_required_field_missing", {"missing": missing}, target=target)
            unknown_fields = sorted(set(payload) - set(fields_by_name))
            if unknown_fields and not payload_schema.additional_fields_allowed:
                return reject("payload_unknown_field", {"unknown": unknown_fields}, target=target)
            for name, value in list(payload.items()):
                spec = fields_by_name.get(name)
                if spec is None:
                    continue
                if value is None and not spec.required:
                    payload.pop(name)  # a null optional is an absent optional
                    continue
                if not _payload_value_matches(value, str(spec.value_kind).lower()):
                    return reject(
                        "payload_field_type_mismatch",
                        {"field": name, "expected": str(spec.value_kind)},
                        target=target,
                    )
                if spec.allowed_values and not any(value == allowed for allowed in spec.allowed_values):
                    return reject(
                        "payload_field_value_disallowed",
                        {"field": name, "value": value, "allowed": list(spec.allowed_values)},
                        target=target,
                    )

        derivation: AnnotationDerivationV1 | None = None
        source_ids = tuple(str(item) for item in finding.get("source_annotation_ids") or ())
        if self.mode == AnnotationJobMode.ADJUDICATE:
            if not source_ids:
                return reject("adjudication_sources_missing", {}, target=target)
            unknown_sources = sorted(set(source_ids) - set(self.existing) - self.allowed_sources)
            if unknown_sources:
                return reject(
                    "adjudication_source_unknown",
                    {"unknown": unknown_sources},
                    target=target,
                )
            dissent = tuple(
                source_id
                for source_id in source_ids
                if source_id in self.existing and set(self.existing[source_id].labels) != set(labels)
            )
            derivation = AnnotationDerivationV1(
                kind=AnnotationDerivationKind.ADJUDICATION,
                source_annotation_ids=source_ids,
                method="arbiter",
                dissenting_annotation_ids=dissent,
            )
        elif source_ids:
            return reject("source_annotation_ids_unexpected", {"ids": list(source_ids)}, target=target)

        return self._base(
            target=target,
            annotation_type=annotation_type,
            status=AnnotationStatus.APPLIED,
            labels=labels,
            payload=payload,
            confidence=confidence,
            rationale=rationale,
            evidence=tuple(evidence.values()),
            grounding=GroundingStatus.GROUNDED,
            derivation=derivation,
        )

    # -- judgments (verify mode) -------------------------------------------------------

    def _judgments(
        self,
        judgments: list[dict[str, Any]],
        rejected: list[RejectedFindingV1],
    ) -> tuple[VerifierDefinitionV1, VerifierResultV2] | AnnotationJobErrorV1:
        rubric = self.rubric
        assert rubric is not None
        fail_closed = self._behavior() == UnavailableEvidenceBehavior.FAIL
        producer_kind = str(self.producer.kind)
        if producer_kind not in {item.value for item in VerifierKind}:
            producer_kind = VerifierKind.AGENTIC.value
        producer = ProducerRefV1(
            kind=producer_kind,
            name=self.producer.name,
            version=self.producer.version,
            model=self.producer.model,
            config_digest=self.producer.config_digest,
            credential_profile=self.producer.credential_profile,
        )
        verifier_definition = VerifierDefinitionV1(
            verifier_id=f"{self.definition.annotator_id}.verifier",
            name=f"{self.definition.name} verifier",
            kind=producer_kind,
            rubric_id=rubric.rubric_id,
            rubric_version=rubric.version,
            rubric_digest=rubric.content_digest,
            version=self.definition.version,
            program_ref=self.program_digest,
            model=self.definition.model,
            required_selectors=("trace",),
            requires_citation=True,
            deterministic=producer_kind == VerifierKind.DETERMINISTIC,
            metadata={"annotator_id": self.definition.annotator_id, "annotator_digest": self.definition.content_digest},
        ).sealed()
        subject = self._trace_selector()
        by_criterion = {item.criterion_id: item for item in rubric.criteria}
        seen: set[str] = set()
        records: list[JudgmentV1] = []
        for index, judgment in enumerate(judgments):
            criterion_id = str(judgment["criterion_id"])
            criterion = by_criterion.get(criterion_id)
            if criterion is None:
                rejected.append(RejectedFindingV1(index=index, kind="judgment", reason="unknown_criterion", detail={"criterion_id": criterion_id}))
                if fail_closed:
                    return AnnotationJobErrorV1(code=AnnotationJobErrorCode.UNSUPPORTED_FINDING, message=f"judgment[{index}] unknown criterion {criterion_id}")
                continue
            if criterion_id in seen:
                rejected.append(RejectedFindingV1(index=index, kind="judgment", reason="duplicate_criterion", detail={"criterion_id": criterion_id}))
                if fail_closed:
                    return AnnotationJobErrorV1(code=AnnotationJobErrorCode.UNSUPPORTED_FINDING, message=f"judgment[{index}] duplicate criterion {criterion_id}")
                continue
            seen.add(criterion_id)
            evidence: dict[str, TraceSelectorV1] = {}
            unresolved: str | None = None
            for raw in judgment.get("evidence") or []:
                selector, resolution, reason = _resolve(self.index, raw)
                if selector is None or resolution is None or not resolution.resolved:
                    unresolved = reason
                    break
                evidence.setdefault(self.index.selector_digest(selector), selector)
            status = str(judgment["status"])
            score = judgment.get("score")
            confidence, confidence_problem = self._confidence(judgment.get("confidence"))
            if confidence_problem:
                confidence = None
            rationale = str(judgment.get("rationale") or "")
            failure_modes = tuple(str(item) for item in judgment.get("failure_modes") or ())
            needs_evidence = criterion.requires_citation
            decisive = status == "decisive"
            if decisive and (
                score is None
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not criterion.min_score <= float(score) <= criterion.max_score
            ):
                reason = "score_out_of_range" if score is not None else "score_missing"
                rejected.append(RejectedFindingV1(index=index, kind="judgment", reason=reason, detail={"criterion_id": criterion_id, "score": score}))
                if fail_closed:
                    return AnnotationJobErrorV1(code=AnnotationJobErrorCode.UNSUPPORTED_FINDING, message=f"judgment[{index}] {reason}")
                decisive = False
                status = "abstained"
            if decisive and (unresolved is not None or (needs_evidence and not evidence)):
                reason = "evidence_unresolved" if unresolved is not None else "evidence_missing"
                rejected.append(RejectedFindingV1(index=index, kind="judgment", reason=reason, detail={"criterion_id": criterion_id, "reason": unresolved}))
                if fail_closed:
                    return AnnotationJobErrorV1(code=AnnotationJobErrorCode.UNSUPPORTED_FINDING, message=f"judgment[{index}] {reason}")
                decisive = False
                status = "abstained"
                evidence = {}
            if not decisive:
                if status == "not_applicable" and not criterion.allows_not_applicable:
                    rejected.append(RejectedFindingV1(index=index, kind="judgment", reason="not_applicable_disallowed", detail={"criterion_id": criterion_id}))
                    continue
                if status in {"abstained", "inconclusive"} and not criterion.allows_abstention:
                    rejected.append(RejectedFindingV1(index=index, kind="judgment", reason="abstention_disallowed", detail={"criterion_id": criterion_id}))
                    continue
                if needs_evidence and not evidence:
                    # Non-decisive judgments still need a citation when the criterion demands one.
                    evidence = {self.index.selector_digest(subject): subject}
                judgment_status = {
                    "abstained": JudgmentStatus.ABSTAINED,
                    "not_applicable": JudgmentStatus.NOT_APPLICABLE,
                    "inconclusive": JudgmentStatus.INCONCLUSIVE,
                }[status]
                verdict = {
                    JudgmentStatus.ABSTAINED: "abstained",
                    JudgmentStatus.NOT_APPLICABLE: "not_applicable",
                    JudgmentStatus.INCONCLUSIVE: "inconclusive",
                }[judgment_status]
                records.append(
                    self._judgment(
                        criterion=criterion,
                        subject=subject,
                        status=judgment_status,
                        score=None,
                        passed=None,
                        verdict=verdict,
                        rationale=rationale,
                        failure_modes=failure_modes,
                        evidence=tuple(evidence.values()),
                        confidence=confidence,
                        producer=producer,
                    )
                )
                continue
            numeric = float(score)
            passed = numeric >= criterion.pass_threshold if criterion.higher_is_better else numeric <= criterion.pass_threshold
            records.append(
                self._judgment(
                    criterion=criterion,
                    subject=subject,
                    status=JudgmentStatus.DECISIVE,
                    score=numeric,
                    passed=passed,
                    verdict="pass" if passed else "fail",
                    rationale=rationale,
                    failure_modes=failure_modes,
                    evidence=tuple(evidence.values()),
                    confidence=confidence,
                    producer=producer,
                    metadata={"reported_verdict": str(judgment.get("verdict") or "")},
                )
            )
        decisive_count = sum(1 for item in records if str(item.status) == JudgmentStatus.DECISIVE)
        verification = VerificationStatus.VALID if decisive_count else VerificationStatus.INCONCLUSIVE
        score_value: float | None = None
        passed_value: bool | None = None
        failures: tuple[str, ...] = ()
        if verification == VerificationStatus.VALID:
            score_value, passed_value, failures = aggregate_rubric_score(rubric, tuple(records))
        result = VerifierResultV2(
            verifier_result_id=record_id(
                "vres",
                kind="verifier_result",
                scope=(self.document.trace_id,),
                key={"job_id": self.job_id},
            ),
            verifier_id=verifier_definition.verifier_id,
            verifier_version=verifier_definition.version,
            rubric_id=rubric.rubric_id,
            rubric_digest=rubric.content_digest,
            subject=subject,
            execution_status=ExecutionStatus.COMPLETED,
            verification_status=verification,
            grounding=GroundingStatus.GROUNDED if decisive_count else GroundingStatus.UNINSPECTED,
            produced_at=self.created_at,
            producer=producer,
            score=score_value,
            pass_threshold=rubric.aggregation.pass_threshold,
            passed=passed_value,
            verdict=("pass" if passed_value else "fail") if passed_value is not None else "inconclusive",
            criterion_results=tuple(records),
            failure_modes=failures,
            verifier_execution_trace_id=self.execution_trace_id,
            metadata={
                "job_id": self.job_id,
                "annotator_id": self.definition.annotator_id,
                "execution_trace_digest": self.execution_trace_digest,
                "rejected_judgments": len([item for item in rejected if item.kind == "judgment"]),
            },
        ).sealed()
        return verifier_definition, result

    def _judgment(
        self,
        *,
        criterion: Any,
        subject: TraceSelectorV1,
        status: JudgmentStatus,
        score: float | None,
        passed: bool | None,
        verdict: str,
        rationale: str,
        failure_modes: tuple[str, ...],
        evidence: tuple[TraceSelectorV1, ...],
        confidence: float | None,
        producer: ProducerRefV1,
        metadata: dict[str, Any] | None = None,
    ) -> JudgmentV1:
        return JudgmentV1(
            criterion_id=criterion.criterion_id,
            score=score,
            verdict=verdict,
            passed=passed,
            rationale=rationale,
            failure_modes=failure_modes,
            evidence=evidence,
            grounding=GroundingStatus.GROUNDED if evidence else GroundingStatus.PARTIALLY_GROUNDED,
            confidence=confidence,
            metadata=dict(metadata or {}),
            judgment_id=record_id(
                "jdg",
                kind="judgment",
                scope=(self.document.trace_id,),
                key={"job_id": self.job_id, "criterion_id": criterion.criterion_id},
            ),
            criterion_version=criterion.version,
            criterion_digest=criterion.content_digest,
            subject=subject,
            status=status,
            producer=producer,
            produced_at=self.created_at,
            revision=1,
            state=RecordState.CURRENT,
            schema_version=JUDGMENT_SCHEMA_VERSION,
        ).sealed()


def producer_for(
    definition: TraceAnnotatorDefinitionV1,
    *,
    kind: ProducerKind | str,
    name: str,
    version: str,
    model: str | None = None,
    config_digest: str | None = None,
) -> ProducerRefV1:
    return ProducerRefV1(
        kind=str(kind),
        name=name,
        version=version,
        model=model if model is not None else definition.model,
        config_digest=config_digest,
    )


__all__ = [
    "ProposalValidationResult",
    "ProposalValidator",
    "RejectedFindingV1",
    "producer_for",
]
