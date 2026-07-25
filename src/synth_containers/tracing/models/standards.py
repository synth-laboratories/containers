"""Companion evaluation standards: criterion, rubric, verifier, annotator, reward.

Definitions are immutable and content-addressed. Results are append-only facts about
one subject under one definition version. Re-running a definition creates another
result; it never mutates the previous one, and it never mutates the trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin

from ..canonical import seal_record
from .actors import Visibility
from .artifacts import ArtifactRefV5
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

    kind: str
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
class TraceAnnotatorDefinitionV1(JsonDataclassMixin):
    annotator_id: str
    name: str
    purpose: str
    taxonomy: tuple[str, ...]
    version: str = "v1"
    supported_trace_schemas: tuple[str, ...] = ("synth.trace.v5",)
    required_subject_scope: str = "trace"
    reasoning_policy: str = "not_captured"
    grounding_requirement: str = "exact_selector"
    minimum_evidence: int = 1
    program_ref: str | None = None
    model: str | None = None
    unavailable_evidence_behavior: str = "abstain"
    confidence_semantics: str = "self_reported"
    schema_version: str = ANNOTATOR_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    content_digest: str = ""

    def sealed(self) -> "TraceAnnotatorDefinitionV1":
        return seal_record(self)


@dataclass(frozen=True, slots=True)
class AnnotationV1(JsonDataclassMixin):
    annotation_id: str
    annotator_id: str
    annotator_version: str
    annotator_digest: str
    target: TraceSelectorV1
    annotation_type: str
    labels: tuple[str, ...]
    author_kind: str
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
    by_id = {item.criterion_id: item for item in results}
    numerator = 0.0
    denominator = 0.0
    failures: list[str] = []
    for criterion in rubric.criteria:
        result = by_id.get(criterion.criterion_id)
        if result is None:
            if str(criterion.role) in {CriterionRole.GATING, CriterionRole.REQUIRED}:
                failures.append(f"missing:{criterion.criterion_id}")
            continue
        if result.verdict == "not_applicable" and policy.not_applicable_criterion == "exclude":
            continue
        if result.score is None:
            if str(criterion.role) in {CriterionRole.GATING, CriterionRole.REQUIRED}:
                failures.append(f"unscored:{criterion.criterion_id}")
            continue
        weight = max(0.0, float(criterion.weight))
        numerator += float(result.score) * weight
        denominator += weight
        if str(criterion.role) == CriterionRole.GATING and result.passed is False:
            failures.append(f"gate_failed:{criterion.criterion_id}")
    score = (numerator / denominator) if denominator > 0.0 else None
    passed = score is not None and score >= policy.pass_threshold
    if failures and policy.gates_override_score:
        passed = False
    return score, passed, tuple(failures)


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
    "AnnotationV1",
    "BenchmarkVerdictV1",
    "CriterionDefinitionV1",
    "CriterionResultV1",
    "CriterionRole",
    "EvaluationResultV1",
    "ExecutionStatus",
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
    "VerificationStatus",
    "VerifierDefinitionV1",
    "VerifierKind",
    "VerifierResultV2",
    "aggregate_rubric_score",
]
