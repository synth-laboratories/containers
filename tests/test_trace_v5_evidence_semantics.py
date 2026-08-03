from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from synth_containers.tracing.adapters.atif import import_atif
from synth_containers.tracing.capture.binding import (
    BindingCaptureV1,
    BindingWorkloadV1,
    WorkloadKind,
    mint_binding,
)
from synth_containers.tracing.evidence_ops import attach, new_evidence_bundle
from synth_containers.tracing.models.evidence import (
    TraceEvidenceBundleV5,
    TraceRefV5,
)
from synth_containers.tracing.models.selectors import (
    GroundingStatus,
    SelectorKind,
    selector_for,
)
from synth_containers.tracing.models.standards import (
    AnnotationV1,
    BenchmarkVerdictV1,
    CriterionDefinitionV1,
    CriterionResultV1,
    CriterionRole,
    EvaluationResultV1,
    ExecutionStatus,
    ProducerRefV1,
    RewardAggregationV1,
    RewardDefinitionV1,
    RewardEmission,
    RewardRecordV1,
    RewardSourceKind,
    RubricAggregationV1,
    RubricDefinitionV2,
    TraceAnnotatorDefinitionV1,
    VerificationStatus,
    VerifierDefinitionV1,
    VerifierKind,
    VerifierResultV2,
)
from synth_containers.tracing.native_evaluation import attach_native_evaluation
from synth_containers.tracing.projections.inspector import (
    load_bundle,
    select_evidence_head,
    summarize,
)
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.validation.validator import validate_evidence


NOW = "2026-07-25T00:00:00Z"


def _trace(correlation: str = "semantic-correlation"):
    return import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": correlation,
            "agent": {"name": "semantic-agent", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": "inspect"}],
        }
    )


def _evidence(trace) -> TraceEvidenceBundleV5:
    subject = selector_for(trace, kind=SelectorKind.TRACE)
    producer = ProducerRefV1(kind="test", name="semantic-test")
    criterion = CriterionDefinitionV1(
        criterion_id="semantic-gate",
        name="semantic gate",
        requirement="must pass",
        role=CriterionRole.GATING,
        pass_threshold=0.5,
    ).sealed()
    rubric = RubricDefinitionV2(
        rubric_id="semantic-rubric",
        name="semantic rubric",
        task_family="semantic",
        criteria=(criterion,),
        aggregation=RubricAggregationV1(
            pass_threshold=0.5,
            tie_break="pass_closed",
        ),
    ).sealed()
    verifier_definition = VerifierDefinitionV1(
        verifier_id="semantic-verifier",
        name="semantic verifier",
        kind=VerifierKind.DETERMINISTIC,
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.version,
        rubric_digest=rubric.content_digest,
        requires_citation=False,
    ).sealed()
    verifier_result = VerifierResultV2(
        verifier_result_id="semantic-verifier-result",
        verifier_id=verifier_definition.verifier_id,
        verifier_version=verifier_definition.version,
        rubric_id=rubric.rubric_id,
        rubric_digest=rubric.content_digest,
        subject=subject,
        execution_status=ExecutionStatus.COMPLETED,
        verification_status=VerificationStatus.VALID,
        grounding=GroundingStatus.GROUNDED,
        produced_at=NOW,
        producer=producer,
        score=1.0,
        pass_threshold=0.5,
        passed=True,
        criterion_results=(
            CriterionResultV1(
                criterion_id=criterion.criterion_id,
                score=1.0,
                verdict="pass",
                passed=True,
            ),
        ),
    ).sealed()
    source_reward = RewardDefinitionV1(
        reward_id="semantic-source-reward",
        name="source reward",
        intent="environment metric",
        source_kind=RewardSourceKind.ENVIRONMENT,
        emission=RewardEmission.TERMINAL,
        subject_scope="trace",
        lower_bound=0.0,
        upper_bound=1.0,
    ).sealed()
    aggregate_reward = RewardDefinitionV1(
        reward_id="semantic-aggregate-reward",
        name="aggregate reward",
        intent="mean environment metric",
        source_kind=RewardSourceKind.COMPOSITE,
        emission=RewardEmission.POST_HOC,
        subject_scope="trace",
        lower_bound=0.0,
        upper_bound=1.0,
        aggregation_expression="mean",
    ).sealed()
    first_reward = RewardRecordV1(
        reward_record_id="semantic-reward-1",
        reward_id=source_reward.reward_id,
        reward_version=source_reward.version,
        reward_digest=source_reward.content_digest,
        subject=subject,
        value=0.2,
        provenance="observed",
        produced_at=NOW,
    ).sealed()
    second_reward = RewardRecordV1(
        reward_record_id="semantic-reward-2",
        reward_id=source_reward.reward_id,
        reward_version=source_reward.version,
        reward_digest=source_reward.content_digest,
        subject=subject,
        value=0.4,
        provenance="observed",
        produced_at=NOW,
    ).sealed()
    aggregation = RewardAggregationV1(
        aggregation_id="semantic-reward-aggregation",
        reward_id=aggregate_reward.reward_id,
        input_reward_record_ids=(
            first_reward.reward_record_id,
            second_reward.reward_record_id,
        ),
        input_digests=(
            first_reward.content_digest,
            second_reward.content_digest,
        ),
        definition_digest=aggregate_reward.content_digest,
        value=0.3,
        produced_at=NOW,
        calculation="mean",
    ).sealed()
    annotator = TraceAnnotatorDefinitionV1(
        annotator_id="semantic-annotator",
        name="semantic annotator",
        purpose="grounded label",
        taxonomy=("supported",),
        minimum_evidence=1,
        grounding_requirement="exact_selector",
    ).sealed()
    annotation = AnnotationV1(
        annotation_id="semantic-annotation",
        annotator_id=annotator.annotator_id,
        annotator_version=annotator.version,
        annotator_digest=annotator.content_digest,
        target=subject,
        annotation_type="semantic",
        labels=("supported",),
        author_kind="deterministic",
        producer=producer,
        created_at=NOW,
        grounding=GroundingStatus.GROUNDED,
        evidence=(subject,),
    ).sealed()
    evaluation = EvaluationResultV1(
        evaluation_id="semantic-evaluation",
        subject=subject,
        evaluator_kind="deterministic",
        execution_status=ExecutionStatus.COMPLETED,
        produced_at=NOW,
        producer=producer,
        environment_reward_record_ids=(
            first_reward.reward_record_id,
            second_reward.reward_record_id,
        ),
        verifier_result_ids=(verifier_result.verifier_result_id,),
        rubric_ids=(rubric.rubric_id,),
        aggregate_score=1.0,
        threshold=0.5,
        metadata={
            "aggregate_score_source": verifier_result.verifier_result_id,
        },
    ).sealed()
    verdict = BenchmarkVerdictV1(
        verdict_id="semantic-verdict",
        benchmark_authority="semantic",
        decision="pass",
        produced_at=NOW,
        score_source=evaluation.evaluation_id,
        required_evaluation_ids=(evaluation.evaluation_id,),
        required_gates=(criterion.criterion_id,),
        threshold=0.5,
    ).sealed()
    return TraceEvidenceBundleV5(
        bundle_id="semantic-evidence",
        trace_ref=TraceRefV5(trace.trace_id, trace.content_digest),
        created_at=NOW,
        criteria=(criterion,),
        rubrics=(rubric,),
        verifier_definitions=(verifier_definition,),
        annotator_definitions=(annotator,),
        reward_definitions=(source_reward, aggregate_reward),
        annotations=(annotation,),
        verifier_results=(verifier_result,),
        reward_records=(first_reward, second_reward),
        reward_aggregations=(aggregation,),
        evaluation_results=(evaluation,),
        benchmark_verdicts=(verdict,),
    ).sealed()


def _codes(trace, evidence) -> set[str]:
    findings, _, _ = validate_evidence(trace, evidence)
    return {item.code for item in findings}


def test_validator_recomputes_scores_rewards_evaluations_and_verdicts() -> None:
    trace = _trace()
    evidence = _evidence(trace)
    assert _codes(trace, evidence) == set()

    verifier_result = replace(
        evidence.verifier_results[0],
        score=0.25,
        passed=False,
        content_digest="",
    ).sealed()
    recompute_failure = replace(
        evidence,
        verifier_results=(verifier_result,),
        content_digest="",
    ).sealed()
    assert {
        "verifier_score_mismatch",
        "verifier_pass_mismatch",
    } <= _codes(trace, recompute_failure)

    status_result = replace(
        verifier_result,
        execution_status=ExecutionStatus.FAILED,
        verification_status=VerificationStatus.VALID,
        passed=True,
        content_digest="",
    ).sealed()
    status_failure = replace(
        evidence,
        verifier_results=(status_result,),
        content_digest="",
    ).sealed()
    assert {
        "verifier_status_inconsistent",
        "verifier_execution_pass_inconsistent",
    } <= _codes(trace, status_failure)

    first_reward = replace(
        evidence.reward_records[0],
        value=1.5,
        source_result_ids=("missing-result",),
        content_digest="",
    ).sealed()
    aggregation = replace(
        evidence.reward_aggregations[0],
        input_digests=(
            first_reward.content_digest,
            evidence.reward_records[1].content_digest,
        ),
        value=0.9,
        content_digest="",
    ).sealed()
    evaluation = replace(
        evidence.evaluation_results[0],
        aggregate_score=1.0,
        metadata={"aggregate_score_source": first_reward.reward_record_id},
        content_digest="",
    ).sealed()
    verdict = replace(
        evidence.benchmark_verdicts[0],
        threshold=1.1,
        content_digest="",
    ).sealed()
    reward_failure = replace(
        evidence,
        reward_records=(first_reward, evidence.reward_records[1]),
        reward_aggregations=(aggregation,),
        evaluation_results=(evaluation,),
        benchmark_verdicts=(verdict,),
        content_digest="",
    ).sealed()
    assert {
        "reward_value_out_of_bounds",
        "reward_source_result_missing",
        "reward_aggregation_value_mismatch",
        "evaluation_aggregate_mismatch",
        "verdict_decision_mismatch",
    } <= _codes(trace, reward_failure)

    cyclic_reward = replace(
        evidence.reward_records[0],
        source_result_ids=(evidence.reward_aggregations[0].aggregation_id,),
        content_digest="",
    ).sealed()
    cyclic_aggregation = replace(
        evidence.reward_aggregations[0],
        input_digests=(
            cyclic_reward.content_digest,
            evidence.reward_records[1].content_digest,
        ),
        content_digest="",
    ).sealed()
    cyclic = replace(
        evidence,
        reward_records=(cyclic_reward, evidence.reward_records[1]),
        reward_aggregations=(cyclic_aggregation,),
        content_digest="",
    ).sealed()
    assert "reward_source_result_cycle" in _codes(trace, cyclic)


def test_validator_enforces_annotator_taxonomy_evidence_and_grounding() -> None:
    trace = _trace()
    evidence = _evidence(trace)
    annotation = replace(
        evidence.annotations[0],
        labels=("unknown",),
        grounding=GroundingStatus.UNINSPECTED,
        evidence=(),
        content_digest="",
    ).sealed()
    damaged = replace(
        evidence,
        annotations=(annotation,),
        content_digest="",
    ).sealed()

    assert {
        "annotation_taxonomy_mismatch",
        "annotation_minimum_evidence_unmet",
        "annotation_grounding_requirement_unmet",
    } <= _codes(trace, damaged)


def test_validator_enforces_supersession_revision_state_and_forks() -> None:
    trace = _trace()
    evidence = _evidence(trace)
    original = evidence.annotations[0]
    successor = replace(
        original,
        annotation_id="semantic-annotation-2",
        revision=2,
        supersedes_id=original.annotation_id,
        content_digest="",
    ).sealed()
    linear = replace(
        evidence,
        annotations=(original, successor),
        content_digest="",
    ).sealed()
    assert not {
        code for code in _codes(trace, linear) if code.startswith("annotation_")
    }

    fork = replace(
        successor,
        annotation_id="semantic-annotation-fork",
        content_digest="",
    ).sealed()
    forked = replace(
        evidence,
        annotations=(original, successor, fork),
        content_digest="",
    ).sealed()
    assert "annotation_supersedes_fork" in _codes(trace, forked)

    bad_state = replace(
        evidence.reward_records[0],
        state="superseded",
        content_digest="",
    ).sealed()
    state_bundle = replace(
        evidence,
        reward_records=(bad_state, evidence.reward_records[1]),
        content_digest="",
    ).sealed()
    assert "reward_record_state_inconsistent" in _codes(trace, state_bundle)


def test_evidence_head_selection_is_order_independent_and_fails_closed() -> None:
    trace = _trace()
    root = new_evidence_bundle(trace)
    first = CriterionDefinitionV1(
        criterion_id="head-first",
        name="first",
        requirement="first",
    ).sealed()
    second = CriterionDefinitionV1(
        criterion_id="head-second",
        name="second",
        requirement="second",
    ).sealed()
    child = attach(root, kind="criterion", record=first)
    head = attach(child, kind="criterion", record=second)

    assert select_evidence_head((head, root, child)) == head
    with pytest.raises(ValueError, match="missing parent"):
        select_evidence_head((head,))

    fork_record = CriterionDefinitionV1(
        criterion_id="head-fork",
        name="fork",
        requirement="fork",
    ).sealed()
    fork = attach(root, kind="criterion", record=fork_record)
    with pytest.raises(ValueError, match="forks"):
        select_evidence_head((root, child, fork))

    other_root = new_evidence_bundle(trace)
    with pytest.raises(ValueError, match="roots"):
        select_evidence_head((root, other_root))


def test_native_evaluation_writes_reachable_typed_revisions_twice(
    tmp_path: Path,
) -> None:
    trace = _trace("native-correlation")
    _write_trace_bundle(tmp_path, trace)
    with pytest.raises(ValueError, match="selected 0 traces"):
        attach_native_evaluation(
            tmp_path,
            payload={
                "trace_correlation_id": "wrong-correlation",
                "verifier": {"score": 1.0},
            },
            source_name="wrong.json",
        )

    first = attach_native_evaluation(
        tmp_path,
        payload=_native_payload(attempt=1, reward=0.2, score=1.0),
        source_name="harbor-first.json",
    )
    assert first["validation_valid"] is True
    first_inspected = load_bundle(tmp_path)[0]
    assert first_inspected.evidence is not None
    assert len(first_inspected.evidence.criteria) == 1
    assert len(first_inspected.evidence.rubrics) == 1
    assert len(first_inspected.evidence.verifier_definitions) == 1
    assert len(first_inspected.evidence.verifier_results) == 1
    assert len(first_inspected.evidence.reward_definitions) == 1
    assert len(first_inspected.evidence.reward_records) == 1
    assert _codes(trace, first_inspected.evidence) == set()

    second = attach_native_evaluation(
        tmp_path,
        payload=_native_payload(attempt=2, reward=0.4, score=0.8),
        source_name="harbor-second.json",
    )
    assert second["validation_valid"] is True
    second_inspected = load_bundle(tmp_path)[0]
    assert second_inspected.evidence is not None
    assert len(second_inspected.evidence.evaluation_results) == 2
    assert len(second_inspected.evidence.verifier_results) == 2
    assert len(second_inspected.evidence.reward_records) == 2
    assert _codes(trace, second_inspected.evidence) == set()
    assert len(
        summarize(second_inspected)["evidence"]["evaluation_results"]
    ) == 2

    manifest = LocalTraceBundle(tmp_path).read_manifest()
    assert len(manifest["evidence"]) == 3
    assert manifest["evidence"][-1]["bundle_digest"] == second[
        "evidence_bundle_digest"
    ]


def test_native_evaluation_infers_bounds_that_include_unbounded_native_score(
    tmp_path: Path,
) -> None:
    trace = _trace("native-correlation")
    _write_trace_bundle(tmp_path, trace)

    attached = attach_native_evaluation(
        tmp_path,
        payload={
            "schema_version": "harbor.native-evaluation.v1",
            "authority": "harbor",
            "trace_correlation_id": "native-correlation",
            "status": "completed",
            "verifier": {
                "returncode": 0,
                "score": 1.01,
            },
        },
        source_name="harbor-above-unit-range.json",
    )
    inspected = load_bundle(tmp_path)[0]
    assert inspected.evidence is not None
    criterion = inspected.evidence.criteria[-1]
    verifier_result = inspected.evidence.verifier_results[-1]

    assert attached["validation_valid"] is True
    assert attached["aggregate_score"] == 1.0
    assert criterion.min_score == 0.0
    assert criterion.max_score == 1.01
    assert verifier_result.score == 1.0
    assert verifier_result.criterion_results[0].score == 1.01
    assert verifier_result.pass_threshold == pytest.approx(0.5 / 1.01)
    assert _codes(trace, inspected.evidence) == set()


def test_failed_native_evaluation_does_not_publish_placeholder_score_or_verdict(
    tmp_path: Path,
) -> None:
    trace = _trace("failed-native-correlation")
    _write_trace_bundle(tmp_path, trace)
    attached = attach_native_evaluation(
        tmp_path,
        payload={
            "schema_version": "harbor.native-evaluation.v1",
            "authority": "harbor",
            "trace_correlation_id": "failed-native-correlation",
            "status": "failed",
            "error": "local verifier could not resolve its workspace",
            "aggregate_score": 0.0,
            "decision": "fail",
            "metrics": {
                "wall_time_seconds": 1.5,
            },
            "verifier": {
                "id": "failed-verifier",
                "returncode": 1,
                "score": 0.0,
                "passed": False,
                "verdict": "fail",
                "criteria": [
                    {
                        "id": "placeholder-gate",
                        "name": "Placeholder gate",
                        "role": "gating",
                        "score": 0.0,
                        "passed": False,
                    }
                ],
            },
        },
        source_name="failed-harbor.json",
    )
    inspected = load_bundle(tmp_path)[0]
    assert inspected.evidence is not None
    evaluation = inspected.evidence.evaluation_results[-1]
    verifier_result = inspected.evidence.verifier_results[-1]

    assert attached["validation_valid"] is True
    assert attached["aggregate_score"] is None
    assert attached["verdict_id"] is None
    assert evaluation.execution_status == ExecutionStatus.FAILED
    assert evaluation.aggregate_score is None
    assert evaluation.threshold is None
    assert evaluation.objective_metrics == {"wall_time_seconds": 1.5}
    assert verifier_result.execution_status == ExecutionStatus.FAILED
    assert verifier_result.verification_status == VerificationStatus.INVALID
    assert verifier_result.score is None
    assert verifier_result.passed is None
    assert verifier_result.verdict == ""
    assert verifier_result.criterion_results == ()
    assert inspected.evidence.benchmark_verdicts == ()


def _write_trace_bundle(root: Path, trace) -> None:
    binding = mint_binding(
        trace_id=trace.trace_id,
        capture_id=trace.capture.capture_id,
        workload=BindingWorkloadV1(
            kind=WorkloadKind.OTHER,
            root_actor_id=trace.actors[0].actor_id,
            actor_session_id=trace.sessions[0].session_id,
        ),
        capture=BindingCaptureV1(output_artifact_root=str(root)),
        trace_kind=trace.trace_kind,
    )
    bundle = LocalTraceBundle(root)
    bundle.write_binding(binding)
    bundle.write_trace(trace, binding=binding, segments=())
    bundle.write_manifest()


def _native_payload(*, attempt: int, reward: float, score: float) -> dict:
    return {
        "schema_version": "harbor.native-evaluation.v1",
        "authority": "harbor",
        "trace_correlation_id": "native-correlation",
        "task_id": "gamebench/craftax",
        "benchmark_family": "gamebench",
        "attempt": attempt,
        "rubric": {
            "id": "craftax-rubric",
            "name": "Craftax native rubric",
            "pass_threshold": 0.5,
            "criteria": [
                {
                    "id": "quality",
                    "name": "Quality",
                    "description": "Native evaluator quality score.",
                    "role": "gating",
                    "pass_threshold": 0.5,
                }
            ],
        },
        "verifier": {
            "id": "craftax-verifier",
            "score": score,
            "pass_threshold": 0.5,
            "criteria": [
                {
                    "id": "quality",
                    "score": score,
                    "passed": score >= 0.5,
                    "verdict": "pass" if score >= 0.5 else "fail",
                }
            ],
        },
        "reward": {
            "primary_metric": "craftax_achievement_score",
            "value": reward,
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
        "metrics": {"wall_time_seconds": float(attempt)},
    }
