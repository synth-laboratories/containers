from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from synth_containers.tracing.adapters.atif import import_atif
from synth_containers.tracing.canonical import content_digest, text_digest
from synth_containers.tracing.evidence_ops import attach, new_evidence_bundle
from synth_containers.tracing.models.identity import AliasV1
from synth_containers.tracing.models.messages import BranchV5
from synth_containers.tracing.models.events import EventOrderV1, EventV5
from synth_containers.tracing.models.selectors import (
    SelectorKind,
    TextRangeV1,
    TokenSequence,
    TraceSelectorV1,
    resolve_selector,
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
    ReceiptV1,
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
    aggregate_rubric_score,
)
from synth_containers.tracing.models.spans import SpanKind, SpanV5
from synth_containers.tracing.models.tokens import (
    TokenCaptureProvenance,
    TokenCaptureV5,
    TokenSequenceRefV1,
)
from synth_containers.tracing.models.evidence import (
    TraceEvidenceBundleV5,
    TraceRefV5,
)
from synth_containers.tracing.store.projection import catalog_projection
from synth_containers.tracing.store.sqlite_catalog import SqliteCatalogStore
from synth_containers.tracing.validation.schema import json_schema
from synth_containers.tracing.validation.validator import validate_evidence


def _trace():
    trace = import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "integrity-trace",
            "agent": {"name": "agent", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": "hello world"}],
        }
    )
    actor = replace(
        trace.actors[0],
        aliases=(AliasV1("nested_actor", "actor-alias", trace.actors[0].actor_id, "actor"),),
    ).sealed()
    session = replace(
        trace.sessions[0],
        aliases=(
            AliasV1(
                "nested_session",
                "session-alias",
                trace.sessions[0].session_id,
                "session",
            ),
        ),
    ).sealed()
    message = replace(
        trace.messages[0],
        aliases=(
            AliasV1(
                "nested_message",
                "message-alias",
                trace.messages[0].message_id,
                "message",
            ),
        ),
    ).sealed()
    branch = BranchV5(
        branch_id="branch-main",
        head_message_id=message.message_id,
        actor_id=actor.actor_id,
        session_id=session.session_id,
    )
    return replace(
        trace,
        actors=(actor, *trace.actors[1:]),
        sessions=(session, *trace.sessions[1:]),
        messages=(message, *trace.messages[1:]),
        branches=(branch,),
        aliases=(
            AliasV1("root", "trace-alias", trace.trace_id, "trace"),
        ),
        content_digest="",
    ).sealed()


def _selector(trace) -> TraceSelectorV1:
    return TraceSelectorV1(
        trace_id=trace.trace_id,
        trace_digest=trace.content_digest,
        kind=SelectorKind.TRACE,
        entity_digest=trace.content_digest,
    )


def _valid_evidence(trace) -> TraceEvidenceBundleV5:
    selector = _selector(trace)
    criterion = CriterionDefinitionV1(
        criterion_id="gate",
        name="gate",
        requirement="must pass",
        role=CriterionRole.GATING,
        pass_threshold=0.5,
    ).sealed()
    rubric = RubricDefinitionV2(
        rubric_id="rubric",
        name="rubric",
        task_family="task",
        criteria=(criterion,),
    ).sealed()
    verifier = VerifierDefinitionV1(
        verifier_id="verifier",
        name="verifier",
        kind=VerifierKind.DETERMINISTIC,
        rubric_id=rubric.rubric_id,
        rubric_version=rubric.version,
        rubric_digest=rubric.content_digest,
    ).sealed()
    producer = ProducerRefV1(kind="test", name="test")
    verifier_result = VerifierResultV2(
        verifier_result_id="verifier-result",
        verifier_id=verifier.verifier_id,
        verifier_version=verifier.version,
        rubric_id=rubric.rubric_id,
        rubric_digest=rubric.content_digest,
        subject=selector,
        execution_status=ExecutionStatus.COMPLETED,
        verification_status=VerificationStatus.VALID,
        grounding="grounded",
        produced_at="2026-07-25T00:00:00Z",
        producer=producer,
        score=1.0,
        passed=True,
        criterion_results=(
            CriterionResultV1("gate", 1.0, "pass", passed=True),
        ),
    ).sealed()
    source_reward = RewardDefinitionV1(
        reward_id="source-reward",
        name="source",
        intent="source",
        source_kind=RewardSourceKind.ENVIRONMENT,
        emission=RewardEmission.TERMINAL,
        subject_scope="trace",
    ).sealed()
    aggregate_reward = RewardDefinitionV1(
        reward_id="aggregate-reward",
        name="aggregate",
        intent="aggregate",
        source_kind=RewardSourceKind.COMPOSITE,
        emission=RewardEmission.POST_HOC,
        subject_scope="trace",
    ).sealed()
    reward_record = RewardRecordV1(
        reward_record_id="reward-record",
        reward_id=source_reward.reward_id,
        reward_version=source_reward.version,
        reward_digest=source_reward.content_digest,
        subject=selector,
        value=1.0,
        provenance="observed",
        produced_at="2026-07-25T00:00:00Z",
    ).sealed()
    aggregation = RewardAggregationV1(
        aggregation_id="reward-aggregation",
        reward_id=aggregate_reward.reward_id,
        input_reward_record_ids=(reward_record.reward_record_id,),
        input_digests=(reward_record.content_digest,),
        definition_digest=aggregate_reward.content_digest,
        value=1.0,
        produced_at="2026-07-25T00:00:00Z",
    ).sealed()
    annotator = TraceAnnotatorDefinitionV1(
        annotator_id="annotator",
        name="annotator",
        purpose="test",
        taxonomy=("ok",),
    ).sealed()
    annotation = AnnotationV1(
        annotation_id="annotation",
        annotator_id=annotator.annotator_id,
        annotator_version=annotator.version,
        annotator_digest=annotator.content_digest,
        target=selector,
        annotation_type="test",
        labels=("ok",),
        author_kind="test",
        producer=producer,
        created_at="2026-07-25T00:00:00Z",
        grounding="grounded",
        evidence=(selector,),
    ).sealed()
    evaluation = EvaluationResultV1(
        evaluation_id="evaluation",
        subject=selector,
        evaluator_kind="test",
        execution_status=ExecutionStatus.COMPLETED,
        produced_at="2026-07-25T00:00:00Z",
        producer=producer,
        environment_reward_record_ids=(reward_record.reward_record_id,),
        verifier_result_ids=(verifier_result.verifier_result_id,),
        rubric_ids=(rubric.rubric_id,),
        aggregate_score=1.0,
    ).sealed()
    verdict = BenchmarkVerdictV1(
        verdict_id="verdict",
        benchmark_authority="test",
        decision="pass",
        produced_at="2026-07-25T00:00:00Z",
        score_source=evaluation.evaluation_id,
        required_evaluation_ids=(evaluation.evaluation_id,),
        required_gates=(criterion.criterion_id,),
    ).sealed()
    receipt = ReceiptV1(
        receipt_id="receipt",
        operation="evaluate",
        status="completed",
        started_at="2026-07-25T00:00:00Z",
    ).sealed()
    return TraceEvidenceBundleV5(
        bundle_id="evidence",
        trace_ref=TraceRefV5(trace.trace_id, trace.content_digest),
        created_at="2026-07-25T00:00:00Z",
        criteria=(criterion,),
        rubrics=(rubric,),
        verifier_definitions=(verifier,),
        annotator_definitions=(annotator,),
        reward_definitions=(source_reward, aggregate_reward),
        annotations=(annotation,),
        verifier_results=(verifier_result,),
        reward_records=(reward_record,),
        reward_aggregations=(aggregation,),
        evaluation_results=(evaluation,),
        benchmark_verdicts=(verdict,),
        receipts=(receipt,),
    ).sealed()


def _codes(trace, evidence) -> set[str]:
    findings, _, _ = validate_evidence(trace, evidence)
    return {finding.code for finding in findings}


def _sorted_rows(rows):
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(str(row.get(key)) for key in sorted(row)),
    )


def test_attach_is_typed_sealed_trace_bound_and_append_only() -> None:
    trace = _trace()
    original = new_evidence_bundle(trace)
    criterion = CriterionDefinitionV1(
        criterion_id="criterion",
        name="criterion",
        requirement="requirement",
    ).sealed()

    revised = attach(original, kind="criterion", record=criterion)

    assert original.criteria == ()
    assert revised.criteria == (criterion,)
    assert revised.metadata["supersedes_bundle_id"] == original.bundle_id
    assert revised.metadata["supersedes_bundle_digest"] == original.content_digest
    second_criterion = replace(
        criterion,
        criterion_id="criterion-2",
        content_digest="",
    ).sealed()
    second_revision = attach(
        revised,
        kind="criterion",
        record=second_criterion,
    )
    assert second_revision.metadata["supersedes_bundle_id"] == revised.bundle_id
    assert second_revision.metadata["supersedes_bundle_digest"] == revised.content_digest
    assert revised.metadata["supersedes_bundle_digest"] == original.content_digest
    with pytest.raises(ValueError, match="duplicate"):
        attach(revised, kind="criterion", record=criterion)
    with pytest.raises(TypeError, match="requires CriterionDefinitionV1"):
        attach(
            original,
            kind="criterion",
            record=RewardDefinitionV1(
                reward_id="reward",
                name="reward",
                intent="reward",
                source_kind=RewardSourceKind.ENVIRONMENT,
                emission=RewardEmission.TERMINAL,
                subject_scope="trace",
            ).sealed(),
        )
    with pytest.raises(ValueError, match="must be sealed"):
        attach(
            original,
            kind="criterion",
            record=replace(criterion, content_digest=""),
        )
    with pytest.raises(ValueError, match="does not match its content"):
        attach(
            original,
            kind="criterion",
            record=replace(criterion, name="tampered"),
        )
    with pytest.raises(ValueError, match="bundle content digest"):
        attach(
            replace(original, created_at="tampered"),
            kind="criterion",
            record=criterion,
        )

    wrong_selector = replace(_selector(trace), trace_id="another-trace")
    annotator = TraceAnnotatorDefinitionV1(
        annotator_id="annotator",
        name="annotator",
        purpose="test",
        taxonomy=("test",),
    ).sealed()
    annotation = AnnotationV1(
        annotation_id="annotation",
        annotator_id=annotator.annotator_id,
        annotator_version=annotator.version,
        annotator_digest=annotator.content_digest,
        target=wrong_selector,
        annotation_type="test",
        labels=("test",),
        author_kind="test",
        producer=ProducerRefV1(kind="test", name="test"),
        created_at="2026-07-25T00:00:00Z",
    ).sealed()
    with pytest.raises(ValueError, match="different trace id"):
        attach(original, kind="annotation", record=annotation)
    with pytest.raises(ValueError, match="different trace digest"):
        attach(
            original,
            kind="annotation",
            record=replace(
                annotation,
                target=replace(
                    annotation.target,
                    trace_id=trace.trace_id,
                    trace_digest="sha256:wrong",
                ),
                content_digest="",
            ).sealed(),
        )


def test_validator_redigests_nested_records_and_checks_definition_links() -> None:
    trace = _trace()
    evidence = _valid_evidence(trace)
    assert _codes(trace, evidence) == set()

    criterion = evidence.criteria[0]
    tampered_nested = replace(
        criterion,
        name="different sealed version",
        content_digest="",
    ).sealed()
    rubric = replace(
        evidence.rubrics[0],
        criteria=(tampered_nested,),
        content_digest="",
    ).sealed()
    verifier = replace(
        evidence.verifier_definitions[0],
        rubric_digest="sha256:wrong",
        content_digest="",
    ).sealed()
    verifier_result = replace(
        evidence.verifier_results[0],
        verifier_version="wrong-version",
        content_digest="",
    ).sealed()
    tampered_receipt = replace(evidence.receipts[0], status="tampered")
    damaged = replace(
        evidence,
        rubrics=(rubric,),
        verifier_definitions=(verifier,),
        verifier_results=(verifier_result,),
        receipts=(tampered_receipt,),
        content_digest="",
    ).sealed()

    codes = _codes(trace, damaged)
    assert "evidence_record_digest_mismatch" in codes
    assert "rubric_criterion_digest_mismatch" in codes
    assert "verifier_definition_rubric_digest_mismatch" in codes
    assert "verifier_definition_version_mismatch" in codes


def test_validator_checks_reward_aggregation_evaluation_and_verdict_inputs() -> None:
    trace = _trace()
    evidence = _valid_evidence(trace)
    reward_record = replace(
        evidence.reward_records[0],
        reward_digest="sha256:wrong",
        content_digest="",
    ).sealed()
    aggregation = replace(
        evidence.reward_aggregations[0],
        definition_digest="sha256:wrong",
        input_reward_record_ids=("reward-record", "missing-reward-input"),
        input_digests=("sha256:wrong", "sha256:missing"),
        content_digest="",
    ).sealed()
    evaluation = replace(
        evidence.evaluation_results[0],
        environment_reward_record_ids=("missing-reward",),
        verifier_result_ids=("missing-verifier",),
        rubric_ids=("missing-rubric",),
        content_digest="",
    ).sealed()
    verdict = replace(
        evidence.benchmark_verdicts[0],
        score_source="missing-source",
        required_evaluation_ids=("missing-evaluation",),
        required_gates=("missing-gate",),
        content_digest="",
    ).sealed()
    damaged = replace(
        evidence,
        reward_records=(reward_record,),
        reward_aggregations=(aggregation,),
        evaluation_results=(evaluation,),
        benchmark_verdicts=(verdict,),
        content_digest="",
    ).sealed()

    assert {
        "reward_definition_digest_mismatch",
        "reward_aggregation_definition_digest_mismatch",
        "reward_aggregation_input_digest_mismatch",
        "reward_aggregation_input_missing",
        "evaluation_reward_input_missing",
        "evaluation_verifier_input_missing",
        "evaluation_rubric_input_missing",
        "verdict_evaluation_input_missing",
        "verdict_gate_input_missing",
        "verdict_score_source_missing",
    } <= _codes(trace, damaged)


def test_gating_score_below_threshold_fails_when_passed_is_unspecified() -> None:
    criterion = CriterionDefinitionV1(
        criterion_id="gate",
        name="gate",
        requirement="gate",
        role=CriterionRole.GATING,
        pass_threshold=0.75,
    )
    rubric = RubricDefinitionV2(
        rubric_id="rubric",
        name="rubric",
        task_family="task",
        criteria=(criterion,),
        aggregation=RubricAggregationV1(pass_threshold=0.4),
    )

    score, passed, failures = aggregate_rubric_score(
        rubric,
        (CriterionResultV1("gate", 0.5, "valid", passed=None),),
    )

    assert score == 0.5
    assert passed is False
    assert failures == ("gate_failed:gate",)


def test_selectors_enforce_pointer_range_quote_and_every_entity_digest() -> None:
    trace = _trace()
    message = trace.messages[0]
    part = message.parts[0]
    branch = trace.branches[0]

    pointer = TraceSelectorV1(
        trace_id=trace.trace_id,
        trace_digest=trace.content_digest,
        kind=SelectorKind.MESSAGE,
        entity_id=message.message_id,
        entity_digest=message.content_digest,
        json_pointer="/parts/0/text",
        quote="hello world",
        quote_digest=text_digest("hello world"),
    )
    assert resolve_selector(trace, pointer).resolved_text == "hello world"
    assert resolve_selector(
        trace, replace(pointer, json_pointer="/parts/1/text")
    ).reason == "json_pointer_not_found"
    assert resolve_selector(
        trace, replace(pointer, json_pointer="parts/0/text")
    ).reason == "json_pointer_invalid"
    assert resolve_selector(
        trace, replace(pointer, quote_digest=text_digest("wrong"))
    ).reason == "quote_digest_mismatch"

    part_selector = TraceSelectorV1(
        trace_id=trace.trace_id,
        trace_digest=trace.content_digest,
        kind=SelectorKind.PART,
        entity_id=message.message_id,
        part_id=part.part_id,
        entity_digest=content_digest(part),
        range=TextRangeV1(0, 5),
        quote="hello",
        quote_digest=text_digest("hello"),
    )
    part_resolution = resolve_selector(trace, part_selector)
    assert part_resolution.resolved is True
    assert part_resolution.entity_digest == content_digest(part)
    assert part_resolution.resolved_text == "hello"
    assert resolve_selector(
        trace, replace(part_selector, range=TextRangeV1(-1, 5))
    ).reason == "range_invalid"
    assert resolve_selector(
        trace, replace(part_selector, entity_digest=message.content_digest)
    ).reason == "entity_digest_mismatch"

    branch_selector = TraceSelectorV1(
        trace_id=trace.trace_id,
        trace_digest=trace.content_digest,
        kind=SelectorKind.BRANCH,
        entity_id=branch.branch_id,
        entity_digest=content_digest(branch),
    )
    assert resolve_selector(trace, branch_selector).entity_digest == content_digest(branch)
    assert resolve_selector(
        trace, replace(branch_selector, entity_digest="sha256:wrong")
    ).reason == "entity_digest_mismatch"
    assert resolve_selector(
        trace, replace(_selector(trace), entity_digest="sha256:wrong")
    ).reason == "entity_digest_mismatch"


def test_model_call_token_range_selectors_resolve_inline_and_fail_closed_for_artifacts() -> None:
    trace = _trace()
    span = SpanV5(
        span_id="span-token-call",
        span_kind=SpanKind.MODEL_CALL,
        actor_id=trace.actors[0].actor_id,
        session_id=trace.sessions[0].session_id,
        started_at="2026-01-01T00:00:00Z",
        token_capture=TokenCaptureV5(
            provenance=TokenCaptureProvenance.IMPORTED,
            level="full_training",
            prompt=TokenSequenceRefV1(token_ids=(7, 11, 12, 19), count=4),
            completion=TokenSequenceRefV1(token_ids=(23, 29), count=2),
        ),
    ).sealed()
    trace = replace(trace, spans=(span,), content_digest="").sealed()
    selector = TraceSelectorV1(
        trace_id=trace.trace_id,
        trace_digest=trace.content_digest,
        kind=SelectorKind.SPAN,
        entity_id=span.span_id,
        entity_digest=span.content_digest,
        token_sequence=TokenSequence.PROMPT,
        range=TextRangeV1(1, 3, unit="token"),
    )

    resolution = resolve_selector(trace, selector)
    assert resolution.resolved is True
    assert resolution.detail == {
        "token_sequence": "prompt",
        "token_ids": [11, 12],
        "start": 1,
        "end": 3,
    }
    assert resolve_selector(
        trace,
        replace(
            selector,
            token_sequence=TokenSequence.COMPLETION,
            range=TextRangeV1(1, 2, unit="token"),
        ),
    ).detail["token_ids"] == [29]
    assert resolve_selector(
        trace,
        replace(selector, range=TextRangeV1(3, 5, unit="token")),
    ).reason == "range_invalid"
    assert resolve_selector(
        trace,
        replace(selector, token_sequence=None),
    ).reason == "token_sequence_required"

    artifact_span = replace(
        span,
        token_capture=replace(
            span.token_capture,
            prompt=TokenSequenceRefV1(
                artifact_id="artifact-prompt-tokens",
                count=4,
                digest=(
                    "sha256:"
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                ),
            ),
        ),
        content_digest="",
    ).sealed()
    artifact_trace = replace(trace, spans=(artifact_span,), content_digest="").sealed()
    artifact_selector = replace(
        selector,
        trace_digest=artifact_trace.content_digest,
        entity_digest=artifact_span.content_digest,
    )
    artifact_resolution = resolve_selector(artifact_trace, artifact_selector)
    assert artifact_resolution.resolved is False
    assert artifact_resolution.reason == "token_sequence_artifact_backed"
    assert artifact_resolution.detail == {"artifact_id": "artifact-prompt-tokens"}


def test_schema_does_not_require_omitted_no_default_nullable_fields() -> None:
    branch_schema = json_schema(BranchV5)
    criterion_result_schema = json_schema(CriterionResultV1)
    reward_record_schema = json_schema(RewardRecordV1)

    assert "head_message_id" not in branch_schema["required"]
    assert {"branch_id", "actor_id", "session_id"} <= set(branch_schema["required"])
    assert "score" not in criterion_result_schema["required"]
    assert "value" not in reward_record_schema["required"]


def test_sqlite_rows_exactly_match_public_projection_with_nested_aliases(
    tmp_path: Path,
) -> None:
    trace = _trace()
    evidence = _valid_evidence(trace)
    trace_projection = catalog_projection(trace)
    evidence_projection = catalog_projection(evidence)
    catalog = SqliteCatalogStore(tmp_path / "catalog.sqlite3")
    try:
        catalog.index_trace(trace)
        catalog.index_evidence(evidence)

        assert _sorted_rows(catalog.traces()) == _sorted_rows(
            trace_projection["documents"]
        )
        assert _sorted_rows(catalog.entities(limit=1000)) == _sorted_rows(
            trace_projection["entities"]
        )
        assert _sorted_rows(catalog.relationships()) == _sorted_rows(
            trace_projection["relationships"]
        )
        assert _sorted_rows(catalog.aliases()) == _sorted_rows(
            trace_projection["aliases"]
        )
        assert _sorted_rows(catalog.evidence()) == _sorted_rows(
            evidence_projection["evidence"]
        )
        assert {"nested_actor", "nested_session", "nested_message"} <= {
            row["namespace"] for row in catalog.aliases()
        }
    finally:
        catalog.close()


def test_structured_catalog_search_compounds_trace_entity_and_evidence_filters(
    tmp_path: Path,
) -> None:
    original = _trace()
    actor = replace(
        original.actors[0],
        provider="openai",
        model="gpt-5.4",
        task_id="task-search",
        workflow_id="workflow-search",
        content_digest="",
    ).sealed()
    session = original.sessions[0]
    span = SpanV5(
        span_id="span-search",
        span_kind=SpanKind.MODEL_CALL,
        actor_id=actor.actor_id,
        session_id=session.session_id,
        started_at="1970-01-01T00:00:00Z",
        workflow_address="workflow-search/map/0",
        detail={"model": "gpt-5.4", "provider": "openai"},
    ).sealed()
    event = EventV5(
        event_id="event-search",
        event_type="environment.reward",
        actor_id=actor.actor_id,
        session_id=session.session_id,
        occurred_at="1970-01-01T00:00:00Z",
        order=EventOrderV1(chronological_sequence=1),
        payload={"reward": 1.0},
    ).sealed()
    trace = replace(
        original,
        identity=replace(
            original.identity,
            task_id="task-search",
            run_id="run-search",
            correlation_id="correlation-search",
        ),
        actors=(actor, *original.actors[1:]),
        spans=(span,),
        events=(event,),
        content_digest="",
    ).sealed()
    evidence = _valid_evidence(trace)
    split_reward_evidence = replace(
        evidence,
        reward_records=(
            replace(
                evidence.reward_records[0],
                value=0.0,
                content_digest="",
            ).sealed(),
        ),
        reward_aggregations=(
            replace(
                evidence.reward_aggregations[0],
                value=2.0,
                content_digest="",
            ).sealed(),
        ),
        content_digest="",
    ).sealed()
    catalog = SqliteCatalogStore(tmp_path / "catalog.sqlite3")
    try:
        catalog.index_trace(trace)
        catalog.index_evidence(split_reward_evidence)

        rows = list(
            catalog.query_traces(
                query="hello",
                trace_id=trace.trace_id,
                trace_digest=trace.content_digest,
                task_id="task-search",
                run_id="run-search",
                correlation_id="correlation-search",
                actor_id=actor.actor_id,
                session_id=session.session_id,
                provider="openai",
                model="gpt-5.4",
                event_kind="environment.reward",
                span_kind="model_call",
                criterion_id="gate",
                annotation_id="annotation",
                reward_id="source-reward",
                reward_min=0.0,
                reward_max=0.0,
                workflow_address="workflow-search/map/0",
                started_after="1969-01-01T00:00:00Z",
                started_before="1971-01-01T00:00:00Z",
                completeness=str(trace.completeness.capture_status),
                visibility=str(trace.visibility),
                digest=span.content_digest,
            )
        )
        assert [row["trace_digest"] for row in rows] == [trace.content_digest]
        assert list(catalog.query_traces(reward_min=1.0, reward_max=1.0)) == []
        assert list(
            catalog.query_traces(
                reward_id="source-reward",
                reward_min=2.0,
            )
        ) == []
        with pytest.raises(ValueError, match="reward_min"):
            list(catalog.query_traces(reward_min=2.0, reward_max=1.0))
    finally:
        catalog.close()
