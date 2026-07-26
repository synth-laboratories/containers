"""Companion evaluation standards: criterion, rubric, verifier, annotator, reward.

Definitions are immutable and content-addressed. Results are append-only facts about
one subject under one definition version. Re-running a definition creates another
result; it never mutates the previous one, and it never mutates the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
from typing import Any, Optional

from synth_containers.serde import JsonDataclassMixin, JsonValue

from ..canonical import seal_record
from .actors import Visibility
from .artifacts import ArtifactRefV5
from .projection import ProjectionLossV1
from .selectors import GroundingStatus, TraceSelectorV1


CRITERION_SCHEMA_VERSION = "synth.criterion.v1"
RUBRIC_SCHEMA_VERSION = "synth.rubric.v2"
VERIFIER_DEFINITION_SCHEMA_VERSION = "synth.verifier.v1"
VERIFIER_RESULT_SCHEMA_VERSION = "synth.verifier-result.v2"
ANNOTATOR_SCHEMA_VERSION = "synth.trace-annotator.v1"
ANNOTATION_SCHEMA_VERSION = "synth.annotation.v1"
REWARD_DEFINITION_SCHEMA_VERSION = "synth.reward.v1"
REWARD_RECORD_SCHEMA_VERSION = "synth.reward-record.v1"
REWARD_AGGREGATION_SCHEMA_VERSION = "synth.reward-aggregation.v1"
EVALUATION_RESULT_SCHEMA_VERSION = "synth.evaluation-result.v1"
BENCHMARK_VERDICT_SCHEMA_VERSION = "synth.benchmark-verdict.v1"
RECEIPT_SCHEMA_VERSION = "synth.receipt.v1"


class CriterionRole(StrEnum):
    GATING = "gating"
    REQUIRED = "required"
    OPTIONAL = "optional"
    INFORMATIONAL = "informational"


class ExecutionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


class VerificationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INCONCLUSIVE = "inconclusive"


class VerifierKind(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    AGENTIC = "agentic"
    HUMAN = "human"
    COMPOSITE = "composite"


class ProducerKind(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL = "model"
    AGENTIC = "agentic"
    HUMAN = "human"
    COMPOSITE = "composite"


class AnnotationTaskKind(StrEnum):
    CLASSIFY = "classify"
    EXTRACT = "extract"
    DESCRIBE = "describe"
    LABEL_SPAN = "label_span"
    RELATE_ENTITIES = "relate_entities"


class AnnotationValueKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class AnnotationStatus(StrEnum):
    APPLIED = "applied"
    ABSTAINED = "abstained"
    SOURCE_UNAVAILABLE = "source_unavailable"


class AnnotationReviewState(StrEnum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    DISPUTED = "disputed"


class AnnotationDerivationKind(StrEnum):
    CONSENSUS = "consensus"
    ADJUDICATION = "adjudication"


class AnnotationInspectionSource(StrEnum):
    TRACE_AUTHORITY = "trace_authority"
    PROJECTION = "projection"


class AnnotatorGroundingRequirement(StrEnum):
    EXACT_SELECTOR = "exact_selector"
    SELECTOR = "selector"
    SUMMARY_ALLOWED = "summary_allowed"
    NONE = "none"


class UnavailableEvidenceBehavior(StrEnum):
    ABSTAIN = "abstain"
    EMIT_UNAVAILABLE = "emit_unavailable"
    FAIL = "fail"


class ConfidenceSemantics(StrEnum):
    NONE = "none"
    SELF_REPORTED = "self_reported"
    CALIBRATED_PROBABILITY = "calibrated_probability"
    INTER_ANNOTATOR_AGREEMENT = "inter_annotator_agreement"
    DETERMINISTIC = "deterministic"


class RewardSourceKind(StrEnum):
    ENVIRONMENT = "environment"
    DETERMINISTIC_METRIC = "deterministic_metric"
    VERIFIER = "verifier"
    HUMAN = "human"
    MODEL = "model"
    LEARNED_REWARD_MODEL = "learned_reward_model"
    COMPOSITE = "composite"


class RewardEmission(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    TERMINAL = "terminal"
    POST_HOC = "post_hoc"


class RecordState(StrEnum):
    CURRENT = "current"
    STALE = "stale"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class ProducerRefV1(JsonDataclassMixin):
    """Who or what produced a result; credentials appear only as a profile name."""

    kind: ProducerKind | str
    name: str
    version: str = ""
    model: str | None = None
    config_digest: str | None = None
    credential_profile: str | None = None


@dataclass(frozen=True, slots=True)
class CriterionDefinitionV1(JsonDataclassMixin):
    criterion_id: str
    name: str
    requirement: str
    version: str = "v1"
    intent: str = ""
    role: CriterionRole | str = CriterionRole.REQUIRED
    weight: float = 1.0
    aggregation_group: str = "default"
    min_score: float = 0.0
    max_score: float = 1.0
    pass_threshold: float = 0.5
    higher_is_better: bool = True
    subject_scope: str = "trace"
    expected_evidence: tuple[str, ...] = ()
    deterministic_check_ref: str | None = None
    evaluator_instructions: str = ""
    allows_abstention: bool = False
    allows_not_applicable: bool = False
    failure_modes: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    requires_citation: bool = True
    schema_version: str = CRITERION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "CriterionDefinitionV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class RubricAggregationV1(JsonDataclassMixin):
    """How criterion results become one score; every edge case is declared."""

    strategy: str = "weighted_mean"
    gates_override_score: bool = True
    missing_criterion: str = "treat_as_missing"
    invalid_criterion: str = "treat_as_missing"
    not_applicable_criterion: str = "exclude"
    inconclusive_criterion: str = "treat_as_missing"
    pass_threshold: float = 0.5
    rounding: str = "none"
    tie_break: str = "fail_closed"
    human_precedence: bool = True


@dataclass(frozen=True, slots=True)
class RubricDefinitionV2(JsonDataclassMixin):
    rubric_id: str
    name: str
    task_family: str
    criteria: tuple[CriterionDefinitionV1, ...]
    version: str = "v1"
    benchmark: str | None = None
    aggregation: RubricAggregationV1 = field(default_factory=RubricAggregationV1)
    scoring_instructions: str = ""
    evidence_expectations: tuple[str, ...] = ()
    allowed_visibility: Visibility | str = Visibility.PRIVATE
    schema_version: str = RUBRIC_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "RubricDefinitionV2":
        return seal_record(self)

    def criterion(self, criterion_id: str) -> CriterionDefinitionV1 | None:
        return next((item for item in self.criteria if item.criterion_id == criterion_id), None)


@dataclass(frozen=True, slots=True)
class VerifierDefinitionV1(JsonDataclassMixin):
    verifier_id: str
    name: str
    kind: VerifierKind | str
    rubric_id: str
    rubric_version: str
    rubric_digest: str
    version: str = "v1"
    supported_subject_types: tuple[str, ...] = ("synth.trace.v5",)
    program_ref: str | None = None
    model: str | None = None
    required_selectors: tuple[str, ...] = ()
    allowed_network: bool = False
    timeout_seconds: float | None = None
    criterion_dispatch: str = "all"
    requires_citation: bool = True
    output_schema: str = VERIFIER_RESULT_SCHEMA_VERSION
    aggregation_owner: str = "verifier"
    abstention_behavior: str = "inconclusive"
    deterministic: bool = True
    schema_version: str = VERIFIER_DEFINITION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "VerifierDefinitionV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class CriterionResultV1(JsonDataclassMixin):
    criterion_id: str
    score: float | None
    verdict: str
    passed: bool | None = None
    rationale: str = ""
    failure_modes: tuple[str, ...] = ()
    evidence: tuple[TraceSelectorV1, ...] = ()
    grounding: GroundingStatus | str = GroundingStatus.UNINSPECTED
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerifierResultV2(JsonDataclassMixin):
    """Execution status and verification validity are separate on purpose."""

    verifier_result_id: str
    verifier_id: str
    verifier_version: str
    rubric_id: str
    rubric_digest: str
    subject: TraceSelectorV1
    execution_status: ExecutionStatus | str
    verification_status: VerificationStatus | str
    grounding: GroundingStatus | str
    produced_at: str
    producer: ProducerRefV1
    score: float | None = None
    pass_threshold: float | None = None
    passed: bool | None = None
    verdict: str = ""
    criterion_results: tuple[CriterionResultV1, ...] = ()
    failure_modes: tuple[str, ...] = ()
    evidence: tuple[TraceSelectorV1, ...] = ()
    artifacts: tuple[ArtifactRefV5, ...] = ()
    verifier_execution_trace_id: str | None = None
    state: RecordState | str = RecordState.CURRENT
    supersedes_id: str | None = None
    invalidation_reason: str | None = None
    schema_version: str = VERIFIER_RESULT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "VerifierResultV2":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class AnnotationTaxonV1(JsonDataclassMixin):
    """One canonical annotation label and its optional hierarchy."""

    label: str
    description: str = ""
    parent_label: Optional[str] = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationPayloadFieldV1(JsonDataclassMixin):
    """One typed top-level field in an annotator's structured output."""

    field_name: str
    value_kind: AnnotationValueKind | str
    required: bool = False
    description: str = ""
    allowed_values: tuple[JsonValue, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationPayloadSchemaV1(JsonDataclassMixin):
    """A compact, portable schema for the structured annotation payload."""

    schema_id: str
    version: str
    fields: tuple[AnnotationPayloadFieldV1, ...]
    additional_fields_allowed: bool = False


@dataclass(frozen=True, slots=True)
class AnnotationOutputContractV1(JsonDataclassMixin):
    """Typed output vocabulary layered over the legacy flat taxonomy."""

    task_kind: AnnotationTaskKind | str
    annotation_types: tuple[str, ...]
    taxonomy: tuple[AnnotationTaxonV1, ...] = ()
    payload_schema: Optional[AnnotationPayloadSchemaV1] = None
    allowed_producer_kinds: tuple[ProducerKind | str, ...] = ()


@dataclass(frozen=True, slots=True)
class UnavailableAnnotationEvidenceV1(JsonDataclassMixin):
    """One required source that could not be inspected."""

    requirement: str
    reason: str
    attempted_selector: Optional[TraceSelectorV1] = None
    source_projection: Optional[str] = None


@dataclass(frozen=True, slots=True)
class AnnotationEvidenceGapsV1(JsonDataclassMixin):
    """Typed, intentionally unresolved evidence; not a successful citation."""

    gaps: tuple[UnavailableAnnotationEvidenceV1, ...]


@dataclass(frozen=True, slots=True)
class AnnotationInspectionV1(JsonDataclassMixin):
    """What authority or lossy projection the annotator actually inspected."""

    source: AnnotationInspectionSource | str
    trace_body_read: bool
    projection_id: Optional[str] = None
    projection_digest: Optional[str] = None
    projection_manifest_digest: Optional[str] = None
    losses: tuple[ProjectionLossV1, ...] = ()


@dataclass(frozen=True, slots=True)
class AnnotationDerivationV1(JsonDataclassMixin):
    """Consensus or adjudication lineage over immutable source annotations."""

    kind: AnnotationDerivationKind | str
    source_annotation_ids: tuple[str, ...]
    method: str
    agreement: Optional[float] = None
    dissenting_annotation_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceAnnotatorDefinitionV1(JsonDataclassMixin):
    """Versioned descriptive/extractive contract, never a criterion judgment."""

    annotator_id: str
    name: str
    purpose: str
    taxonomy: tuple[str, ...]
    version: str = "v1"
    supported_trace_schemas: tuple[str, ...] = ("synth.trace.v5",)
    required_subject_scope: str = "trace"
    reasoning_policy: str = "not_captured"
    grounding_requirement: AnnotatorGroundingRequirement | str = (
        AnnotatorGroundingRequirement.EXACT_SELECTOR
    )
    minimum_evidence: int = 1
    program_ref: str | None = None
    model: str | None = None
    unavailable_evidence_behavior: UnavailableEvidenceBehavior | str = (
        UnavailableEvidenceBehavior.ABSTAIN
    )
    confidence_semantics: ConfidenceSemantics | str = ConfidenceSemantics.SELF_REPORTED
    output_contract: Optional[AnnotationOutputContractV1] = None
    confidence_calibration_ref: Optional[str] = None
    schema_version: str = ANNOTATOR_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "TraceAnnotatorDefinitionV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class AnnotationV1(JsonDataclassMixin):
    """A descriptive, extractive, or classificatory claim about exact trace evidence."""

    annotation_id: str
    annotator_id: str
    annotator_version: str
    annotator_digest: str
    target: TraceSelectorV1
    annotation_type: str
    labels: tuple[str, ...]
    author_kind: ProducerKind | str
    producer: ProducerRefV1
    created_at: str
    grounding: GroundingStatus | str = GroundingStatus.UNINSPECTED
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    rationale: str = ""
    evidence: tuple[TraceSelectorV1, ...] = ()
    visibility: Visibility | str = Visibility.PRIVATE
    inspected_projection: str | None = None
    revision: int = 1
    supersedes_id: str | None = None
    status: Optional[AnnotationStatus | str] = None
    review_state: Optional[AnnotationReviewState | str] = None
    abstention_reason: Optional[str] = None
    unavailable_evidence: Optional[AnnotationEvidenceGapsV1] = None
    inspection: Optional[AnnotationInspectionV1] = None
    derivation: Optional[AnnotationDerivationV1] = None
    annotator_execution_trace_id: Optional[str] = None
    annotator_execution_trace_digest: Optional[str] = None
    schema_version: str = ANNOTATION_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "AnnotationV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class RewardDefinitionV1(JsonDataclassMixin):
    reward_id: str
    name: str
    intent: str
    source_kind: RewardSourceKind | str
    emission: RewardEmission | str
    subject_scope: str
    version: str = "v1"
    producer_authority: str = ""
    vector_components: tuple[str, ...] = ()
    units: str = "scalar"
    lower_bound: float | None = None
    upper_bound: float | None = None
    higher_is_better: bool = True
    normalization: str = "none"
    clipping: str = "none"
    discounting: str = "none"
    aggregation_expression: str = ""
    missing_behavior: str = "omit"
    trainable: bool = False
    leakage_class: str = "visible"
    required_evidence: tuple[str, ...] = ()
    schema_version: str = REWARD_DEFINITION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "RewardDefinitionV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class RewardRecordV1(JsonDataclassMixin):
    reward_record_id: str
    reward_id: str
    reward_version: str
    reward_digest: str
    subject: TraceSelectorV1
    value: float | None
    provenance: str
    produced_at: str
    components: dict[str, float] = field(default_factory=dict)
    raw_value: float | None = None
    normalized_value: float | None = None
    actor_id: str | None = None
    session_id: str | None = None
    position: str | None = None
    source_result_ids: tuple[str, ...] = ()
    evidence: tuple[TraceSelectorV1, ...] = ()
    confidence: float | None = None
    validity: str = "valid"
    grounding: GroundingStatus | str = GroundingStatus.GROUNDED
    state: RecordState | str = RecordState.CURRENT
    supersedes_id: str | None = None
    schema_version: str = REWARD_RECORD_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "RewardRecordV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class RewardAggregationV1(JsonDataclassMixin):
    """A derived reward that names every input it consumed."""

    aggregation_id: str
    reward_id: str
    input_reward_record_ids: tuple[str, ...]
    input_digests: tuple[str, ...]
    definition_digest: str
    value: float | None
    produced_at: str
    components: dict[str, float] = field(default_factory=dict)
    grouping: str = "episode"
    window: str | None = None
    discount: float | None = None
    missing_component_handling: str = "omit"
    calculation: str = ""
    schema_version: str = REWARD_AGGREGATION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "RewardAggregationV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class EvaluationResultV1(JsonDataclassMixin):
    evaluation_id: str
    subject: TraceSelectorV1
    evaluator_kind: str
    execution_status: ExecutionStatus | str
    produced_at: str
    producer: ProducerRefV1
    suite: str | None = None
    benchmark: str | None = None
    task_id: str | None = None
    instance_id: str | None = None
    split: str | None = None
    seed: int | None = None
    config_digest: str | None = None
    objective_metrics: dict[str, float] = field(default_factory=dict)
    environment_reward_record_ids: tuple[str, ...] = ()
    verifier_result_ids: tuple[str, ...] = ()
    rubric_ids: tuple[str, ...] = ()
    aggregate_score: float | None = None
    threshold: float | None = None
    artifacts: tuple[ArtifactRefV5, ...] = ()
    trace_completeness: str = ""
    state: RecordState | str = RecordState.CURRENT
    error: str | None = None
    schema_version: str = EVALUATION_RESULT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "EvaluationResultV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class BenchmarkVerdictV1(JsonDataclassMixin):
    verdict_id: str
    benchmark_authority: str
    decision: str
    produced_at: str
    score_source: str = ""
    required_evaluation_ids: tuple[str, ...] = ()
    required_gates: tuple[str, ...] = ()
    threshold: float | None = None
    failure_reasons: tuple[str, ...] = ()
    scorer_digest: str | None = None
    artifacts: tuple[ArtifactRefV5, ...] = ()
    state: RecordState | str = RecordState.CURRENT
    schema_version: str = BENCHMARK_VERDICT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "BenchmarkVerdictV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class ReceiptV1(JsonDataclassMixin):
    """Evidence about an operation. Credentials appear only as a profile name."""

    receipt_id: str
    operation: str
    status: str
    started_at: str
    ended_at: str | None = None
    target_ids: tuple[str, ...] = ()
    producer: ProducerRefV1 | None = None
    return_code: int | None = None
    wall_time_seconds: float | None = None
    input_digests: tuple[str, ...] = ()
    output_digests: tuple[str, ...] = ()
    previous_state: str | None = None
    new_state: str | None = None
    completeness: str = "complete"
    errors: tuple[str, ...] = ()
    next_safe_action: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    schema_version: str = RECEIPT_SCHEMA_VERSION
    content_digest: str = ""

    def sealed(self) -> "ReceiptV1":
        return seal_record(self)


def aggregate_rubric_score(
    rubric: RubricDefinitionV2,
    results: tuple[CriterionResultV1, ...],
) -> tuple[float | None, bool, tuple[str, ...]]:
    """Aggregate criterion results under the rubric's declared policy.

    Returns ``(score, passed, failure_reasons)``. Gating criteria that fail force
    ``passed`` false regardless of the weighted score.
    """

    policy = rubric.aggregation
    ids = [item.criterion_id for item in results]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate criterion results are not allowed")
    known = {item.criterion_id for item in rubric.criteria}
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(f"unknown criterion results: {unknown}")
    by_id = {item.criterion_id: item for item in results}
    scored: list[tuple[float, float]] = []
    failures: list[str] = []

    def add_missing(
        criterion: CriterionDefinitionV1,
        *,
        behavior: str,
        reason: str,
    ) -> None:
        normalized_behavior = str(behavior).strip().lower()
        if normalized_behavior not in {
            "exclude",
            "omit",
            "treat_as_missing",
            "zero",
            "treat_as_zero",
            "fail",
        }:
            raise ValueError(
                f"unsupported rubric {reason} policy: {behavior!r}"
            )
        if normalized_behavior in {"zero", "treat_as_zero", "fail"}:
            scored.append((0.0, max(0.0, float(criterion.weight))))
        if normalized_behavior == "fail":
            failures.append(f"{reason}:{criterion.criterion_id}")

    for criterion in rubric.criteria:
        if criterion.max_score <= criterion.min_score:
            raise ValueError(f"invalid criterion score range: {criterion.criterion_id}")
        if criterion.weight < 0.0:
            raise ValueError(f"negative criterion weight: {criterion.criterion_id}")
        result = by_id.get(criterion.criterion_id)
        if result is None:
            if str(criterion.role) in {CriterionRole.GATING, CriterionRole.REQUIRED}:
                failures.append(f"missing:{criterion.criterion_id}")
            add_missing(
                criterion,
                behavior=policy.missing_criterion,
                reason="missing",
            )
            continue
        verdict = str(result.verdict).lower()
        if verdict == "not_applicable":
            if policy.not_applicable_criterion == "exclude":
                continue
            add_missing(
                criterion,
                behavior=policy.not_applicable_criterion,
                reason="not_applicable",
            )
            continue
        if verdict in {"invalid", "inconclusive", "abstain", "abstained"}:
            status = "inconclusive" if verdict in {"abstain", "abstained"} else verdict
            behavior = (
                policy.invalid_criterion
                if status == "invalid"
                else policy.inconclusive_criterion
            )
            if str(criterion.role) in {CriterionRole.GATING, CriterionRole.REQUIRED}:
                failures.append(f"{status}:{criterion.criterion_id}")
            add_missing(criterion, behavior=behavior, reason=status)
            continue
        if result.score is None:
            if str(criterion.role) in {CriterionRole.GATING, CriterionRole.REQUIRED}:
                failures.append(f"unscored:{criterion.criterion_id}")
            add_missing(
                criterion,
                behavior=policy.missing_criterion,
                reason="unscored",
            )
            continue
        weight = max(0.0, float(criterion.weight))
        bounded = min(criterion.max_score, max(criterion.min_score, float(result.score)))
        normalized = (bounded - criterion.min_score) / (
            criterion.max_score - criterion.min_score
        )
        if not criterion.higher_is_better:
            normalized = 1.0 - normalized
        scored.append((normalized, weight))
        threshold_failed = (
            bounded < criterion.pass_threshold
            if criterion.higher_is_better
            else bounded > criterion.pass_threshold
        )
        criterion_failed = (
            threshold_failed
            or result.passed is False
            or verdict in {"fail", "failed", "failure"}
        )
        if str(criterion.role) == CriterionRole.GATING and criterion_failed:
            failures.append(f"gate_failed:{criterion.criterion_id}")
        if str(criterion.role) == CriterionRole.REQUIRED and criterion_failed:
            failures.append(f"required_failed:{criterion.criterion_id}")

    strategy = str(policy.strategy).strip().lower()
    if not scored:
        score = None
    elif strategy in {"weighted_mean", "gates_only"}:
        denominator = sum(weight for _, weight in scored)
        score = (
            sum(value * weight for value, weight in scored) / denominator
            if denominator > 0.0
            else None
        )
    elif strategy == "weighted_sum":
        score = sum(value * weight for value, weight in scored)
    elif strategy in {"mean", "arithmetic_mean"}:
        score = sum(value for value, _ in scored) / len(scored)
    elif strategy == "sum":
        score = sum(value for value, _ in scored)
    elif strategy == "min":
        score = min(value for value, _ in scored)
    elif strategy == "max":
        score = max(value for value, _ in scored)
    else:
        raise ValueError(f"unsupported rubric aggregation strategy: {policy.strategy!r}")

    score = _round_aggregate(score, policy.rounding)
    tie_break = str(policy.tie_break).strip().lower()
    if tie_break not in {"fail_closed", "pass_closed"}:
        raise ValueError(f"unsupported rubric tie-break policy: {policy.tie_break!r}")
    passed = score is not None and (
        score > policy.pass_threshold
        if tie_break == "fail_closed" and strategy != "gates_only"
        else score >= policy.pass_threshold
    )
    if any(
        item.startswith(
            (
                "missing:",
                "unscored:",
                "invalid:",
                "inconclusive:",
                "required_failed:",
            )
        )
        for item in failures
    ):
        passed = False
    if policy.gates_override_score and any(item.startswith("gate_failed:") for item in failures):
        passed = False
    return score, passed, tuple(failures)


def aggregate_reward_values(
    definition: RewardDefinitionV1,
    records: tuple[RewardRecordV1, ...],
    aggregation: RewardAggregationV1,
) -> tuple[float | None, dict[str, float]]:
    """Recompute a reward aggregation from its declared ordered inputs.

    The calculation language is intentionally finite. Arbitrary expressions would
    make a sealed calculation receipt dependent on an unstated interpreter.
    """

    calculation = (
        str(aggregation.calculation).strip().lower()
        or str(definition.aggregation_expression).strip().lower()
        or ("identity" if len(records) == 1 else "sum")
    )
    missing = (
        str(aggregation.missing_component_handling).strip().lower()
        or str(definition.missing_behavior).strip().lower()
        or "omit"
    )
    values = _reward_inputs(
        tuple(
            record.value if str(record.validity).lower() == "valid" else None
            for record in records
        ),
        missing=missing,
    )
    value = _aggregate_reward_sequence(
        values,
        calculation=calculation,
        discount=aggregation.discount,
    )

    component_names = {
        *definition.vector_components,
        *aggregation.components,
        *(key for record in records for key in record.components),
    }
    components: dict[str, float] = {}
    for component in sorted(component_names):
        component_values = _reward_inputs(
            tuple(
                (
                    record.components.get(component)
                    if str(record.validity).lower() == "valid"
                    else None
                )
                for record in records
            ),
            missing=missing,
        )
        component_value = _aggregate_reward_sequence(
            component_values,
            calculation=calculation,
            discount=aggregation.discount,
        )
        if component_value is not None:
            components[component] = component_value
    return value, components


def _round_aggregate(value: float | None, policy: str) -> float | None:
    if value is None:
        return None
    normalized = str(policy).strip().lower()
    if normalized in {"", "none"}:
        return value
    if normalized in {"integer", "nearest_integer"}:
        return float(round(value))
    if normalized == "floor":
        return float(math.floor(value))
    if normalized == "ceil":
        return float(math.ceil(value))
    if normalized.startswith("decimal:"):
        digits = normalized.removeprefix("decimal:")
        if digits.isdigit():
            return round(value, int(digits))
    raise ValueError(f"unsupported rubric rounding policy: {policy!r}")


def _reward_inputs(
    values: tuple[float | None, ...],
    *,
    missing: str,
) -> tuple[float, ...]:
    if missing not in {"omit", "zero", "treat_as_zero", "fail"}:
        raise ValueError(f"unsupported reward missing-component policy: {missing!r}")
    if missing == "fail" and any(value is None for value in values):
        raise ValueError("reward aggregation has a missing or invalid input")
    return tuple(
        0.0 if value is None else float(value)
        for value in values
        if value is not None or missing in {"zero", "treat_as_zero"}
    )


def _aggregate_reward_sequence(
    values: tuple[float, ...],
    *,
    calculation: str,
    discount: float | None,
) -> float | None:
    aliases = {
        "average": "mean",
        "arithmetic_mean": "mean",
        "total": "sum",
    }
    normalized = aliases.get(calculation, calculation)
    if not values:
        return None
    if discount is not None and not 0.0 <= float(discount) <= 1.0:
        raise ValueError("reward aggregation discount must be between zero and one")
    if normalized == "identity":
        if len(values) != 1:
            raise ValueError("identity reward aggregation requires exactly one input")
        return values[0]
    if normalized in {"sum", "discounted_sum"}:
        if normalized == "discounted_sum" and discount is None:
            raise ValueError("discounted_sum reward aggregation requires a discount")
        if discount is None:
            return sum(values)
        return sum(value * (float(discount) ** index) for index, value in enumerate(values))
    if discount is not None:
        raise ValueError(
            f"reward discount is unsupported for {normalized!r} aggregation"
        )
    if normalized == "mean":
        return sum(values) / len(values)
    if normalized == "min":
        return min(values)
    if normalized == "max":
        return max(values)
    raise ValueError(f"unsupported reward aggregation calculation: {calculation!r}")


__all__ = [
    "ANNOTATION_SCHEMA_VERSION",
    "ANNOTATOR_SCHEMA_VERSION",
    "BENCHMARK_VERDICT_SCHEMA_VERSION",
    "CRITERION_SCHEMA_VERSION",
    "EVALUATION_RESULT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REWARD_AGGREGATION_SCHEMA_VERSION",
    "REWARD_DEFINITION_SCHEMA_VERSION",
    "REWARD_RECORD_SCHEMA_VERSION",
    "RUBRIC_SCHEMA_VERSION",
    "VERIFIER_DEFINITION_SCHEMA_VERSION",
    "VERIFIER_RESULT_SCHEMA_VERSION",
    "AnnotationDerivationKind",
    "AnnotationDerivationV1",
    "AnnotationEvidenceGapsV1",
    "AnnotationInspectionSource",
    "AnnotationInspectionV1",
    "AnnotationOutputContractV1",
    "AnnotationPayloadFieldV1",
    "AnnotationPayloadSchemaV1",
    "AnnotationReviewState",
    "AnnotationStatus",
    "AnnotationTaskKind",
    "AnnotationTaxonV1",
    "AnnotationValueKind",
    "AnnotationV1",
    "AnnotatorGroundingRequirement",
    "BenchmarkVerdictV1",
    "ConfidenceSemantics",
    "CriterionDefinitionV1",
    "CriterionResultV1",
    "CriterionRole",
    "EvaluationResultV1",
    "ExecutionStatus",
    "ProducerKind",
    "ProducerRefV1",
    "ReceiptV1",
    "RecordState",
    "RewardAggregationV1",
    "RewardDefinitionV1",
    "RewardEmission",
    "RewardRecordV1",
    "RewardSourceKind",
    "RubricAggregationV1",
    "RubricDefinitionV2",
    "TraceAnnotatorDefinitionV1",
    "UnavailableAnnotationEvidenceV1",
    "UnavailableEvidenceBehavior",
    "VerificationStatus",
    "VerifierDefinitionV1",
    "VerifierKind",
    "VerifierResultV2",
    "aggregate_reward_values",
    "aggregate_rubric_score",
]
