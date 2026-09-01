"""Deterministic vertical slice: jobs, idempotency, validation, persistence, review."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from synth_containers.tracing.annotation import (
    AnnotationJobErrorCode,
    AnnotationJobLimitsV1,
    AnnotationJobMode,
    AnnotationJobState,
    AnnotationJobV1,
    AnnotationService,
    AnnotationServiceError,
    AnnotationStore,
    AnnotatorProgramV1,
    DefinitionRegistry,
    LocalReservationBroker,
    ReservationBindingV1,
    RevisionConflict,
    RunnerKind,
    StoreCorruption,
    build_craftax_smoke_trace,
    idempotency_key,
    register_builtin_annotators,
)
from synth_containers.tracing.annotation.builtin import ENVIRONMENT_STEP_STATUS_ID
from synth_containers.tracing.annotation.consensus import agreement, consensus_annotation
from synth_containers.tracing.annotation.jobs import AnnotationJobRequestV1
from synth_containers.tracing.annotation.proposal import empty_proposal
from synth_containers.tracing.canonical import content_digest
from synth_containers.tracing.models.standards import (
    AnnotationOutputContractV1,
    AnnotationStatus,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    CriterionDefinitionV1,
    CriterionRole,
    RubricAggregationV1,
    RubricDefinitionV2,
    TraceAnnotatorDefinitionV1,
    UnavailableEvidenceBehavior,
)
from synth_containers.tracing.validation.validator import (
    Severity,
    validate_evidence,
    validate_trace,
)


def _errors(findings):
    return [item.code for item in findings if str(item.severity) == Severity.ERROR]


def _definition(annotator_id: str, *, behavior=UnavailableEvidenceBehavior.ABSTAIN, scope="message", taxonomy=("belief.contradicted", "belief.correct")) -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id=annotator_id,
        name=annotator_id,
        purpose="test annotator",
        taxonomy=taxonomy,
        required_subject_scope=scope,
        minimum_evidence=1,
        unavailable_evidence_behavior=behavior,
        confidence_semantics="deterministic",
        output_contract=AnnotationOutputContractV1(
            task_kind=AnnotationTaskKind.CLASSIFY,
            annotation_types=("belief",),
            taxonomy=tuple(AnnotationTaxonV1(label=label) for label in taxonomy),
        ),
    ).sealed()


def _program(program_id: str) -> AnnotatorProgramV1:
    return AnnotatorProgramV1(program_id=program_id, runner_kind=RunnerKind.DETERMINISTIC, program_ref=program_id).sealed()


def _service(tmp_path: Path, *, extra=()) -> tuple[AnnotationService, Any]:
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    for definition, program, fn in extra:
        registry.register(definition, program, deterministic_program=fn, domain="test")
    store = AnnotationStore(tmp_path / "store")
    service = AnnotationService(store=store, registry=registry)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return service, trace


def test_smoke_trace_is_sealed_and_valid() -> None:
    trace = build_craftax_smoke_trace()
    assert trace.content_digest.startswith("sha256:")
    assert content_digest(trace) == trace.content_digest
    assert build_craftax_smoke_trace().content_digest == trace.content_digest
    assert not _errors(validate_trace(trace))
    assert len(trace.spans_of_kind("environment_step")) == 11
    assert len(trace.spans_of_kind("model_call")) == 5


def test_deterministic_vertical_slice_seals_indexes_and_preserves_trace(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    request = service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID)
    job = service.submit_and_run(request)
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.applied_count == 11 and job.abstained_count == 0
    assert job.bundle_digest and job.prior_bundle_digest is None
    head = service.evidence_head(trace.trace_id)
    assert head is not None and head.content_digest == job.bundle_digest
    assert head.trace_ref.content_digest == trace.content_digest
    assert not _errors(validate_evidence(trace, head)[0])
    # engine authority untouched
    assert service.store.get_source(trace.trace_id, trace.content_digest).content_digest == trace.content_digest
    blocked = service.annotations(trace.trace_id, label="environment_step.blocked")
    assert len(blocked) == 1 and blocked[0].payload["reason"] == "blocked:tree"
    rows = list(service.store.catalog.evidence(trace_digest=trace.content_digest, record_kind="annotation"))
    assert len(rows) == 11
    facets = list(service.store.catalog.annotation_facets("label", trace_digest=trace.content_digest))
    assert {row["value"] for row in facets} >= {"environment_step.ok", "environment_step.no_effect", "environment_step.blocked"}
    receipts = service.store.receipts(job.job_id)
    assert [item.operation for item in receipts] == ["annotation.prepare", "annotation.start", "annotation.run"]
    assert receipts[-1].status == "sealed" and job.request.source_trace_digest in receipts[-1].input_digests
    history = service.store.job_history(job.job_id)
    assert [str(item.state) for item in history] == ["prepared", "running", "validating", "sealed"]


@pytest.mark.parametrize("crash_after_evidence_write", [False, True])
def test_validated_commit_recovers_on_both_sides_of_evidence_write(
    tmp_path: Path, crash_after_evidence_write: bool
) -> None:
    service, trace = _service(tmp_path)
    job = service.submit(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    original = service.store.ensure_evidence_committed

    def crash(bundle, *, expected_prior_digest, job_id):
        if crash_after_evidence_write:
            original(
                bundle,
                expected_prior_digest=expected_prior_digest,
                job_id=job_id,
            )
        raise RuntimeError("simulated crash at validated commit boundary")

    service.store.ensure_evidence_committed = crash  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="validated commit boundary"):
        service.run(job.job_id)
    service.store.ensure_evidence_committed = original  # type: ignore[assignment]

    interrupted = service.get(job.job_id)
    assert interrupted is not None and str(interrupted.state) == AnnotationJobState.VALIDATING
    if crash_after_evidence_write:
        assert service.evidence_head(trace.trace_id) is not None
    else:
        assert service.evidence_head(trace.trace_id) is None

    recovered = service.recover_interrupted()
    assert len(recovered) == 1
    terminal = recovered[0]
    assert terminal.job_id == job.job_id and str(terminal.state) == AnnotationJobState.SEALED
    assert terminal.applied_count == 11 and terminal.bundle_digest
    assert service.evidence_head(trace.trace_id).content_digest == terminal.bundle_digest
    assert service.store.annotation_job(terminal.annotation_ids[0]) == terminal.job_id
    assert [str(item.state) for item in service.store.job_history(job.job_id)] == [
        "prepared",
        "running",
        "validating",
        "sealed",
    ]


def test_identical_request_is_served_from_cache_without_new_job(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    request = service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID, metadata={"display_name": "first"})
    first = service.submit_and_run(request)
    again = service.submit_and_run(replace(request, metadata={"display_name": "second", "path": "/tmp/x"}))
    assert again.job_id == first.job_id
    assert [item.status for item in service.store.receipts(first.job_id)][-1] == "cached"
    assert len(service.store.list_jobs(trace_id=trace.trace_id)) == 1
    # a fresh process over the same store sees the same key and result
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    reopened = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry)
    estimate = reopened.estimate(request)
    assert estimate.cached and estimate.cached_job_id == first.job_id and not estimate.paid
    # a different repeat index is a different job
    repeat = service.submit_and_run(replace(request, repeat_index=1))
    assert repeat.job_id != first.job_id and str(repeat.state) == AnnotationJobState.SEALED


def test_idempotency_key_ignores_presentation_but_tracks_execution_identity() -> None:
    base = AnnotationJobRequestV1(source_trace_id="t", source_trace_digest="sha256:a", annotator_id="x", annotator_digest="sha256:b")
    key = idempotency_key(base, program_digest="p", tool_contract_digest="c")
    assert key == idempotency_key(replace(base, metadata={"x": 1}, limits=AnnotationJobLimitsV1(timeout_seconds=1)), program_digest="p", tool_contract_digest="c")
    assert key != idempotency_key(replace(base, model="m"), program_digest="p", tool_contract_digest="c")
    assert key != idempotency_key(base, program_digest="p2", tool_contract_digest="c")
    assert key != idempotency_key(replace(base, repeat_index=1), program_digest="p", tool_contract_digest="c")
    assert key != idempotency_key(replace(base, source_trace_digest="sha256:z"), program_digest="p", tool_contract_digest="c")


def _finding(target: dict[str, Any], labels: list[str], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return {"target": target, "annotation_type": "belief", "labels": labels, "payload": {}, "confidence": 1.0, "rationale": "test", "evidence": evidence}


def test_invented_selector_becomes_typed_abstention(tmp_path: Path) -> None:
    def program(document, context):
        proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
        reply = next(m for m in document.messages if str(m.role) == "assistant")
        proposal["findings"].append(_finding({"kind": "message", "entity_id": reply.message_id}, ["belief.contradicted"], [{"kind": "message", "entity_id": "msg_does_not_exist"}]))
        proposal["findings"].append(_finding({"kind": "message", "entity_id": reply.message_id}, ["belief.contradicted"], [{"kind": "message", "entity_id": reply.message_id, "quote": "not in the text"}]))
        proposal["findings"].append(_finding({"kind": "message", "entity_id": reply.message_id}, ["belief.correct"], [{"kind": "message", "entity_id": reply.message_id, "quote": "tree directly in front"}]))
        proposal["abstentions"].append({"annotation_type": "belief", "reason": "evidence_unavailable", "requirement": "engine snapshot after step 4", "target": {"kind": "message", "entity_id": reply.message_id}})
        return proposal

    service, trace = _service(tmp_path, extra=[(_definition("test.belief"), _program("test.belief.program"), program)])
    job = service.submit_and_run(service.request_for(trace, "test.belief"))
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.applied_count == 1 and job.abstained_count == 3
    head = service.evidence_head(trace.trace_id)
    assert not _errors(validate_evidence(trace, head)[0])
    abstained = [a for a in head.annotations if str(a.status) == AnnotationStatus.ABSTAINED]
    reasons = sorted(a.abstention_reason for a in abstained)
    assert reasons == ["evidence_unavailable", "evidence_unresolved", "evidence_unresolved"]
    gaps = {g.reason for a in abstained for g in a.unavailable_evidence.gaps}
    assert {"entity_not_found", "quote_mismatch"} <= gaps
    assert all(not a.labels and not a.payload for a in abstained)
    applied = [a for a in head.annotations if str(a.status) == AnnotationStatus.APPLIED][0]
    assert applied.evidence[0].quote == "tree directly in front" and applied.confidence == 1.0


def test_fail_closed_definition_rejects_unsupported_finding(tmp_path: Path) -> None:
    def program(document, context):
        proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
        proposal["findings"].append(_finding({"kind": "message", "entity_id": "nope"}, ["belief.correct"], []))
        return proposal

    service, trace = _service(tmp_path, extra=[(_definition("test.strict", behavior=UnavailableEvidenceBehavior.FAIL), _program("test.strict.program"), program)])
    job = service.submit_and_run(service.request_for(trace, "test.strict"))
    assert str(job.state) == AnnotationJobState.FAILED
    assert job.error is not None and job.error.code == AnnotationJobErrorCode.UNSUPPORTED_FINDING
    assert service.evidence_head(trace.trace_id) is None
    assert service.store.get_proposal(job.job_id) is not None


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda p: {"not": "a proposal"}, AnnotationJobErrorCode.MALFORMED_OUTPUT),
        (lambda p: {**p, "source_trace_digest": "sha256:" + "f" * 64}, AnnotationJobErrorCode.SOURCE_DIGEST_MISMATCH),
        (lambda p: {**p, "findings": [{"target": {"kind": "message"}, "annotation_type": "belief", "labels": ["x"], "evidence": "nope"}]}, AnnotationJobErrorCode.MALFORMED_OUTPUT),
    ],
)
def test_malformed_or_mismatched_output_fails_closed(tmp_path: Path, mutate, code) -> None:
    def program(document, context):
        return mutate(empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest))

    service, trace = _service(tmp_path, extra=[(_definition("test.bad"), _program("test.bad.program"), program)])
    job = service.submit_and_run(service.request_for(trace, "test.bad"))
    assert str(job.state) == AnnotationJobState.FAILED and job.error.code == code
    # a failed job never blocks a corrected re-run under a new program digest
    assert service.store.find_cached_job(job.idempotency_key) is None


def test_review_appends_revision_and_refuses_forks(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    job = service.submit_and_run(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    original_id = job.annotation_ids[0]
    revised = service.review(original_id, decision="accepted", reviewer="josh", rationale="looks right")
    assert revised.supersedes_id == original_id and revised.revision == 2 and str(revised.review_state) == "accepted"
    head = service.evidence_head(trace.trace_id)
    assert not _errors(validate_evidence(trace, head)[0])
    assert len(service.annotations(trace.trace_id)) == 11  # superseded hidden by default
    assert len(service.annotations(trace.trace_id, include_superseded=True)) == 12
    with pytest.raises(AnnotationServiceError) as error:
        service.review(original_id, decision="rejected", reviewer="someone")
    assert error.value.code == AnnotationJobErrorCode.REVISION_CONFLICT
    assert len(service.store.evidence_bundles(trace.trace_id)) == 2


def test_parallel_submits_create_exactly_one_job(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    request = service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID)
    results: list[AnnotationJobV1] = []
    threads = [threading.Thread(target=lambda: results.append(service.submit(request))) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len({job.job_id for job in results}) == 1
    assert len(service.store.list_jobs(trace_id=trace.trace_id)) == 1


def test_evidence_head_cas_rejects_stale_writer(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    service.submit_and_run(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    head = service.evidence_head(trace.trace_id)
    with pytest.raises(RevisionConflict):
        service.store.put_evidence(head, expected_prior_digest=None)


def test_store_corruption_fails_closed_and_rebuilds_from_sealed_authority(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    job = service.submit_and_run(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    path = service.store.evidence_path(trace.trace_id, job.bundle_digest)
    original = path.read_text()
    path.chmod(0o644)
    path.write_text(original.replace("environment_step.blocked", "environment_step.ok", 1))
    with pytest.raises(StoreCorruption):
        service.evidence_head(trace.trace_id)
    assert not service.store.verify()["ok"]
    path.write_text(original)
    (tmp_path / "store" / "jobs.sqlite").unlink()
    (tmp_path / "store" / "catalog.sqlite").unlink()
    store = AnnotationStore(tmp_path / "store")
    counts = store.rebuild_index()
    assert counts == {"jobs": 1, "evidence": 1}
    assert store.find_cached_job(job.idempotency_key).job_id == job.job_id
    assert store.get_annotation(job.annotation_ids[0]) is not None


def test_crash_recovery_fails_interrupted_jobs_without_inventing_results(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    job = service.submit(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    service.store.save_job(job.transition(AnnotationJobState.RUNNING))
    recovered = service.recover_interrupted()
    assert [str(item.state) for item in recovered] == ["failed"]
    assert recovered[0].error.code == AnnotationJobErrorCode.TRANSPORT_DISCONNECTED
    assert service.evidence_head(trace.trace_id) is None
    rerun = service.submit_and_run(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    assert rerun.job_id != job.job_id and str(rerun.state) == AnnotationJobState.SEALED


def test_cancel_prepared_job(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    job = service.submit(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    cancelled = service.cancel(job.job_id)
    assert str(cancelled.state) == AnnotationJobState.CANCELLED
    assert service.store.receipts(job.job_id)[-1].next_safe_action == "resubmit_with_new_repeat_index"


def _paid_service(tmp_path: Path, broker=None):
    registry = DefinitionRegistry()
    definition = _definition("test.paid")
    program = AnnotatorProgramV1(program_id="test.paid.program", runner_kind=RunnerKind.CODEX_APP_SERVER, prompt="Find beliefs.", paid=True).sealed()
    registry.register(definition, program, domain="test")
    from synth_containers.tracing.annotation import CodexAppServerRunner

    runner = CodexAppServerRunner(lambda cwd: None, default_model="gpt-5.6-luna", default_effort="medium", proxy_enforces_reservation=True)
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    return service, trace


def test_paid_annotator_is_impossible_without_a_broker(tmp_path: Path) -> None:
    service, trace = _paid_service(tmp_path)
    request = service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000))
    estimate = service.estimate(request)
    assert estimate.paid and estimate.requires_reservation and estimate.resolved_model == "gpt-5.6-luna"
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request)
    assert error.value.code == AnnotationJobErrorCode.RESERVATION_REQUIRED
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request, reservation_id="rsv_anything", session_id="sess-1")
    assert error.value.code == AnnotationJobErrorCode.RESERVATION_REJECTED and error.value.detail["reason"] == "reservation_broker_unavailable"
    assert service.store.list_jobs(trace_id=trace.trace_id) == ()
    assert service.store.ledger.entries()[0].stage == "abandoned"  # the refused intent is recorded, not lost


def test_reservations_are_single_use_bound_and_capped(tmp_path: Path) -> None:
    broker = LocalReservationBroker(tmp_path / "broker")
    service, trace = _paid_service(tmp_path, broker)
    request = service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000))
    binding = ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model="gpt-5.6-luna", session_id="sess-1")
    # forged id
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request, reservation_id="rsv_forged", session_id="sess-1")
    assert error.value.detail["reason"] == "reservation_unknown"
    # wildcard bindings are refused at issue time unless explicitly allowed
    from synth_containers.tracing.annotation import ReservationError

    with pytest.raises(ReservationError) as refused:
        broker.issue(cap_usd_micros=500_000, binding=ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model=None, session_id="sess-1"))
    assert refused.value.code == "reservation_binding_incomplete"
    # wrong binding: different trace
    other = broker.issue(cap_usd_micros=500_000, binding=ReservationBindingV1(trace_digest="sha256:" + "1" * 64, annotator_id="test.paid", model="gpt-5.6-luna", session_id="sess-1"))
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request, reservation_id=other.reservation_id, session_id="sess-1")
    assert error.value.detail["reason"] == "reservation_binding_mismatch"
    # a paid submit without a session is refused before any claim
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request, reservation_id=other.reservation_id)
    assert error.value.detail["reason"] == "session_required"
    # wrong session
    bound = broker.issue(cap_usd_micros=500_000, binding=binding)
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request, reservation_id=bound.reservation_id, session_id="sess-2")
    assert error.value.detail["reason"] == "reservation_binding_mismatch"
    # good claim: cap becomes the job ceiling, request without its own ceiling
    job = service.submit(request, reservation_id=bound.reservation_id, session_id="sess-1")
    assert str(job.state) == AnnotationJobState.PREPARED and job.reservation_id == bound.reservation_id
    assert job.request.limits.max_cost_usd == 0.5 and job.metadata["reservation"]["cap_usd_micros"] == 500_000
    assert broker.get(bound.reservation_id).claimed_by_job_id == job.job_id
    assert service.store.ledger.get(job.job_id).stage == "prepared"
    # replay under a new identity (repeat index) is refused: the reservation is consumed
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(replace(request, repeat_index=1), reservation_id=bound.reservation_id, session_id="sess-1")
    assert error.value.detail["reason"] == "reservation_consumed"
    # a request declaring a higher ceiling than the cap is clamped to the cap
    second = broker.issue(cap_usd_micros=200_000, binding=binding)
    clamped = service.submit(replace(request, repeat_index=2, limits=AnnotationJobLimitsV1(max_total_tokens=50_000, max_cost_usd=5.0)), reservation_id=second.reservation_id, session_id="sess-1")
    assert clamped.request.limits.max_cost_usd == 0.2
    # paid jobs without a token ceiling are refused before any claim
    third = broker.issue(cap_usd_micros=200_000, binding=binding)
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(replace(request, repeat_index=3, limits=AnnotationJobLimitsV1(max_total_tokens=None)), reservation_id=third.reservation_id, session_id="sess-1")
    assert error.value.code == AnnotationJobErrorCode.RESERVATION_REJECTED
    assert broker.get(third.reservation_id).claimed_by_job_id is None


def test_cache_key_pins_resolved_model_so_default_changes_never_alias(tmp_path: Path) -> None:
    service, trace = _paid_service(tmp_path)
    request = service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000))
    assert request.model == "gpt-5.6-luna"  # resolved from the definition/runner default at request time
    blank = replace(request, model=None, reasoning_effort=None)
    key_blank = service.estimate(blank).idempotency_key
    key_explicit = service.estimate(replace(request, model="gpt-5.6-luna", reasoning_effort="medium")).idempotency_key
    assert key_blank == key_explicit
    service.runners["codex_app_server"].default_effort = "high"
    assert service.estimate(blank).idempotency_key != key_blank
    from synth_containers.tracing.annotation import CodexAppServerRunner

    service.runners["codex_app_server"] = CodexAppServerRunner(lambda cwd: None, default_model="other-model", default_effort="medium")
    assert service.estimate(blank).idempotency_key != key_blank
    assert service.estimate(blank).resolved_model == "other-model"


def test_worker_runs_enqueued_jobs_and_claims_are_exclusive(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation import AnnotationWorker

    service, trace = _service(tmp_path)
    job = service.submit(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID))
    assert str(job.state) == AnnotationJobState.PREPARED
    worker = AnnotationWorker(service, poll_seconds=0.01)
    assert worker.run_once() == 1
    done = service.get(job.job_id)
    assert str(done.state) == AnnotationJobState.SEALED and done.applied_count == 11
    assert [str(item.state) for item in service.store.job_history(job.job_id)] == ["prepared", "running", "validating", "sealed"]
    with pytest.raises(AnnotationServiceError) as error:
        service.run(job.job_id) if not done.terminal else (_ for _ in ()).throw(AnnotationServiceError("revision_conflict", "terminal"))
    assert error.value.code == "revision_conflict"
    second = service.submit(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID, repeat_index=1))
    worker.start()
    assert worker.wait_for(second.job_id, timeout=10)
    worker.stop()
    assert str(service.get(second.job_id).state) == AnnotationJobState.SEALED


def test_verify_mode_seals_rubric_result_and_recomputes_score(tmp_path: Path) -> None:
    criteria = (
        CriterionDefinitionV1(criterion_id="grounding", name="State grounding", requirement="decisions match visible state", min_score=0.0, max_score=4.0, pass_threshold=2.0, allows_abstention=True).sealed(),
        CriterionDefinitionV1(criterion_id="tooling", name="Tool reliability", requirement="calls are well formed", min_score=0.0, max_score=4.0, pass_threshold=2.0, role=CriterionRole.GATING).sealed(),
    )
    rubric = RubricDefinitionV2(rubric_id="test.rubric", name="test rubric", task_family="craftax", criteria=criteria, aggregation=RubricAggregationV1(pass_threshold=0.5)).sealed()

    def program(document, context):
        proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
        reply = next(m for m in document.messages if str(m.role) == "assistant")
        proposal["judgments"] = [
            {"criterion_id": "grounding", "status": "decisive", "score": 1.0, "verdict": "weak", "rationale": "claimed a tree in front of grass", "failure_modes": ["false_belief"], "evidence": [{"kind": "message", "entity_id": reply.message_id, "quote": "tree directly in front"}]},
            {"criterion_id": "tooling", "status": "decisive", "score": 4.0, "verdict": "pass", "rationale": "always parsed", "failure_modes": [], "evidence": [{"kind": "message", "entity_id": reply.message_id}]},
            {"criterion_id": "unknown", "status": "decisive", "score": 4.0, "verdict": "pass", "rationale": "", "failure_modes": [], "evidence": []},
        ]
        return proposal

    definition = _definition("test.verifier", scope="trace")
    registry = DefinitionRegistry()
    registry.register(definition, _program("test.verifier.program"), rubric=rubric, deterministic_program=program, domain="test", requires_rubric=True)
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    job = service.submit_and_run(service.request_for(trace, "test.verifier", mode=AnnotationJobMode.VERIFY))
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    inferred = service.request_for(trace, "test.verifier")
    assert str(inferred.mode) == AnnotationJobMode.VERIFY
    assert job.verifier_result_ids and job.rejected_count == 1
    head = service.evidence_head(trace.trace_id)
    assert not _errors(validate_evidence(trace, head)[0])
    result = head.verifier_results[0]
    from synth_containers.tracing.models.standards import aggregate_rubric_score

    expected_score, expected_passed, _ = aggregate_rubric_score(rubric, result.judgments)
    assert result.score == pytest.approx(expected_score) == pytest.approx(0.625)
    assert result.passed is expected_passed
    assert result.pass_threshold == 0.5 and result.verdict == ("pass" if expected_passed else "fail")
    judgments = {item.criterion_id: item for item in result.judgments}
    assert judgments["grounding"].passed is False and judgments["grounding"].verdict == "fail"
    assert judgments["tooling"].passed is True
    listed = service.evidence_bundles(trace.trace_id)
    assert listed and listed[0]["verifier_result_count"] == 1
    summary = listed[0]["verifier_results"][0]
    assert summary["score"] == pytest.approx(0.625) and len(summary["criterion_results"]) == 2


def test_verify_mode_refuses_findings_in_place_of_judgments(tmp_path: Path) -> None:
    criteria = (
        CriterionDefinitionV1(criterion_id="grounding", name="State grounding", requirement="decisions match visible state", min_score=0.0, max_score=4.0, pass_threshold=2.0, allows_abstention=True).sealed(),
    )
    rubric = RubricDefinitionV2(rubric_id="test.rubric", name="test rubric", task_family="craftax", criteria=criteria, aggregation=RubricAggregationV1(pass_threshold=0.5)).sealed()

    def program(document, context):
        del context
        proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
        reply = next(m for m in document.messages if str(m.role) == "assistant")
        proposal["findings"].append(_finding({"kind": "message", "entity_id": reply.message_id}, ["belief.correct"], [{"kind": "message", "entity_id": reply.message_id}]))
        return proposal

    definition = _definition("test.verifier.findings", scope="trace")
    registry = DefinitionRegistry()
    registry.register(definition, _program("test.verifier.findings.program"), rubric=rubric, deterministic_program=program, domain="test", requires_rubric=True)
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    job = service.submit_and_run(service.request_for(trace, "test.verifier.findings"))
    assert str(job.state) == AnnotationJobState.FAILED
    assert job.error is not None and "judgments" in job.error.message
    assert not job.verifier_result_ids


def test_consensus_over_repeats_appends_derived_record(tmp_path: Path) -> None:
    service, trace = _service(tmp_path)
    for repeat in range(3):
        job = service.submit_and_run(service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID, repeat_index=repeat))
        assert str(job.state) == AnnotationJobState.SEALED
    report = service.agreement(trace.trace_id, ENVIRONMENT_STEP_STATUS_ID)
    assert report["annotation_count"] == 33 and report["mean_label_agreement"] == 1.0
    derived = service.consensus(trace.trace_id, ENVIRONMENT_STEP_STATUS_ID)
    assert len(derived) == 11
    assert all(item.derivation is not None and str(item.derivation.kind) == "consensus" for item in derived)
    head = service.evidence_head(trace.trace_id)
    assert not _errors(validate_evidence(trace, head)[0])
    assert service.consensus(trace.trace_id, ENVIRONMENT_STEP_STATUS_ID) == ()


def test_agreement_reports_dissent() -> None:
    service_trace = build_craftax_smoke_trace()
    definition = _definition("test.agree")
    from synth_containers.tracing.annotation.validation import ProposalValidator
    from synth_containers.tracing.models.standards import ProducerRefV1

    reply = next(m for m in service_trace.messages if str(m.role) == "assistant")
    records = []
    for index, labels in enumerate((["belief.contradicted"], ["belief.contradicted"], ["belief.correct"])):
        validator = ProposalValidator(service_trace, definition=definition, producer=ProducerRefV1(kind="deterministic", name="t"), job_id=f"job-{index}")
        proposal = empty_proposal(trace_id=service_trace.trace_id, trace_digest=service_trace.content_digest)
        proposal["findings"].append(_finding({"kind": "message", "entity_id": reply.message_id}, labels, [{"kind": "message", "entity_id": reply.message_id}]))
        records.extend(validator.validate(proposal).annotations)
    report = agreement(records)
    assert report.group_count == 1 and report.groups[0].majority_labels == ("belief.contradicted",)
    assert len(report.groups[0].dissenting_annotation_ids) == 1
    assert report.mean_label_agreement == pytest.approx(1 / 3)
    consensus = consensus_annotation(records, definition=definition)
    assert consensus is not None and consensus.labels == ("belief.contradicted",)
    assert consensus.derivation.dissenting_annotation_ids == (records[2].annotation_id,)


def test_http_router_serves_definitions_jobs_and_evidence(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from synth_containers.tracing.annotation.api import build_annotation_router

    service, trace = _service(tmp_path)
    app = fastapi.FastAPI()
    app.include_router(build_annotation_router(service))
    client = TestClient(app)
    definitions = client.get(f"/traces/{trace.trace_id}/annotation-definitions").json()
    ids = {item["annotator_id"] for item in definitions["annotators"]}
    assert ENVIRONMENT_STEP_STATUS_ID in ids
    request = service.request_for(trace, ENVIRONMENT_STEP_STATUS_ID).to_dict()
    estimate = client.post(f"/traces/{trace.trace_id}/annotation-estimates", json={"request": request}).json()
    assert estimate["cached"] is False and estimate["paid"] is False
    response = client.post(f"/traces/{trace.trace_id}/annotation-jobs", json={"request": request})
    assert response.status_code == 202
    started = response.json()
    assert started["job"]["state"] == "prepared" and started["accepted"] is True and started["poll"] == "annotation_get"
    job_id = started["job"]["job_id"]
    assert client.get(f"/annotation-jobs/{job_id}").json()["terminal"] is False
    from synth_containers.tracing.annotation import AnnotationWorker

    assert AnnotationWorker(service).run_once() == 1
    assert client.get(f"/annotation-jobs/{job_id}").json()["terminal"] is True
    forged = client.post(f"/traces/{trace.trace_id}/annotation-jobs", json={"request": request, "reservation_id": {"cap": 99}})
    assert forged.status_code == 400
    listed = client.get(f"/traces/{trace.trace_id}/annotations", params={"label": "environment_step.blocked"}).json()
    assert listed["count"] == 1
    annotation_id = listed["annotations"][0]["annotation_id"]
    evidence = client.get(f"/annotations/{annotation_id}").json()
    assert evidence["target"]["resolved"] is True and evidence["evidence"][0]["resolved"] is True
    bundles = client.get(f"/traces/{trace.trace_id}/evidence-bundles").json()["bundles"]
    assert len(bundles) == 1 and bundles[0]["is_head"] is True
    assert bundles[0]["verifier_result_count"] == 0 and bundles[0]["verifier_results"] == []
    reviewed = client.post(f"/annotations/{annotation_id}/reviews", json={"decision": "accepted", "reviewer": "josh"}).json()
    assert reviewed["supersedes_id"] == annotation_id
    assert client.get("/annotation/operations").json()["operations"][0]["name"] == "annotation_list_definitions"
    cached = client.post(f"/traces/{trace.trace_id}/annotation-jobs", json={"request": request})
    assert cached.status_code == 202 and cached.json()["job"]["job_id"] == job_id and cached.json()["cached"] is True


def test_crash_between_claim_and_job_save_is_recoverable(tmp_path: Path) -> None:
    broker = LocalReservationBroker(tmp_path / "broker")
    service, trace = _paid_service(tmp_path, broker)
    request = service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000))
    binding = ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model="gpt-5.6-luna", session_id="sess-1")
    reservation = broker.issue(cap_usd_micros=300_000, binding=binding)
    # Simulate the crash: the service wrote the intent and the broker claimed, but save_job never ran.
    original_save = service.store.save_job

    def crash(job):
        raise RuntimeError("simulated crash after claim")

    service.store.save_job = crash  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        service.submit(request, reservation_id=reservation.reservation_id, session_id="sess-1")
    service.store.save_job = original_save  # type: ignore[assignment]
    entry = service.store.ledger.entries()[0]
    assert entry.stage == "claimed" and broker.get(reservation.reservation_id).claimed_by_job_id == entry.job_id
    assert service.get(entry.job_id) is None
    # Restart: the intent is resumed with the same job id; the broker claim is idempotent for it.
    recovered = service.recover_interrupted()
    assert [job.job_id for job in recovered] == [entry.job_id]
    job = service.get(entry.job_id)
    assert job is not None and str(job.state) == AnnotationJobState.PREPARED and job.request.limits.max_cost_usd == 0.3
    assert service.store.ledger.get(entry.job_id).stage == "prepared"
    # A different job can still not take that reservation.
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(replace(request, repeat_index=5), reservation_id=reservation.reservation_id, session_id="sess-1")
    assert error.value.detail["reason"] == "reservation_consumed"


def test_reconciliation_survives_broker_outage_and_is_retried(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation import FlakyBroker

    inner = LocalReservationBroker(tmp_path / "broker")
    broker = FlakyBroker(inner, fail_reconciles=2)
    service, trace = _paid_service(tmp_path, broker)
    request = service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000))
    binding = ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model="gpt-5.6-luna", session_id="sess-1")
    reservation = inner.issue(cap_usd_micros=300_000, binding=binding)
    job = service.submit(request, reservation_id=reservation.reservation_id, session_id="sess-1")
    cancelled = service.cancel(job.job_id)  # terminal without running a task
    assert str(cancelled.state) == AnnotationJobState.CANCELLED
    entry = service.store.ledger.get(job.job_id)
    assert entry.stage == "terminal" and entry.outcome == "cancelled" and entry.reconcile_attempts == 1
    assert "broker_unreachable" in entry.last_error and inner.get(reservation.reservation_id).reconciled_at is None
    assert service.retry_reconciliations() == {"acknowledged": 0, "pending": 1}
    assert service.retry_reconciliations() == {"acknowledged": 1, "pending": 0}
    assert service.store.ledger.get(job.job_id).stage == "acknowledged"
    assert inner.get(reservation.reservation_id).outcome == "cancelled"
    assert service.retry_reconciliations() == {"acknowledged": 0, "pending": 0}
    # a fresh worker on restart drains the outbox before serving
    from synth_containers.tracing.annotation import AnnotationWorker

    AnnotationWorker(service).start().stop()
    assert broker.reconcile_calls == 3


def test_paid_ledger_terminal_outcome_is_compare_and_set(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation.ledger import PaidLedger, PaidLedgerEntryV1

    request = AnnotationJobRequestV1(
        source_trace_id="trace-1",
        source_trace_digest="sha256:trace",
        annotator_id="test.paid",
        annotator_digest="sha256:annotator",
        model="gpt-5.6-luna",
    )
    ledger = PaidLedger(tmp_path / "ledger")
    ledger.write(
        PaidLedgerEntryV1(
            job_id="job-1",
            reservation_id="rsv-1",
            idempotency_key="key-1",
            request=request,
            program_digest="sha256:program",
            tool_contract_digest="sha256:tools",
            created_at="2026-08-31T00:00:00Z",
            session_id="session-1",
            stage="prepared",
        )
    )
    sealed = ledger.mark_terminal(
        "job-1", outcome="sealed", actual_cost_usd_micros=12_000
    )
    assert ledger.mark_terminal(
        "job-1", outcome="sealed", actual_cost_usd_micros=12_000
    ) == sealed
    with pytest.raises(ValueError, match="terminal outcome.*immutable"):
        ledger.mark_terminal(
            "job-1", outcome="failed", actual_cost_usd_micros=12_000
        )
    assert ledger.get("job-1").outcome == "sealed"


def test_paid_execution_fails_closed_without_dollar_enforcement(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation import CodexAppServerRunner

    broker = LocalReservationBroker(tmp_path / "broker")
    service, trace = _paid_service(tmp_path, broker)
    service.runners["codex_app_server"] = CodexAppServerRunner(lambda cwd: None, default_model="gpt-5.6-luna", default_effort="medium")
    request = service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000))
    binding = ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model="gpt-5.6-luna", session_id="sess-1")
    reservation = broker.issue(cap_usd_micros=300_000, binding=binding)
    with pytest.raises(AnnotationServiceError) as error:
        service.submit(request, reservation_id=reservation.reservation_id, session_id="sess-1")
    assert error.value.detail["reason"] == "cost_enforcement_unavailable"
    assert broker.get(reservation.reservation_id).claimed_by_job_id is None  # refused before any claim
    service.runners["codex_app_server"] = CodexAppServerRunner(lambda cwd: None, default_model="gpt-5.6-luna", default_effort="medium", usd_per_million_tokens=5.0)
    job = service.submit(request, reservation_id=reservation.reservation_id, session_id="sess-1")
    assert service.store.ledger.get(job.job_id).metadata["cost_enforcement"] == "pinned_price"


def test_bundle_trace_loader_resolves_sealed_and_promoted_digests(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation.sources import bundle_trace_loader, bundle_trace_refs, chain_loaders
    from synth_containers.tracing.capture.binding import BindingCaptureV1, BindingWorkloadV1, WorkloadKind, mint_binding
    from synth_containers.tracing.store.bundle import LocalTraceBundle

    trace = build_craftax_smoke_trace()
    binding = mint_binding(trace_id=trace.trace_id, capture_id=trace.capture.capture_id, workload=BindingWorkloadV1(kind=WorkloadKind.OTHER, root_actor_id=trace.actors[0].actor_id, actor_session_id=trace.sessions[0].session_id), capture=BindingCaptureV1(output_artifact_root=str(tmp_path / "bundle")), trace_kind=trace.trace_kind)
    bundle = LocalTraceBundle(tmp_path / "bundle")
    bundle.write_binding(binding)
    bundle.write_trace(trace, binding=binding, segments=())
    bundle.write_manifest()

    def promote(document, sealed_digest):
        return replace(document, extensions={**document.extensions, "promoted_from": sealed_digest}, content_digest="").sealed()

    loader = bundle_trace_loader(tmp_path / "bundle", promote=promote)
    assert loader(trace.trace_id, trace.content_digest).content_digest == trace.content_digest
    refs = bundle_trace_refs(tmp_path / "bundle", promote=promote)
    assert refs[0]["sealed_digest"] == trace.content_digest and refs[0]["digest"] != trace.content_digest
    assert loader(trace.trace_id, refs[0]["digest"]).extensions["promoted_from"] == trace.content_digest
    assert loader(trace.trace_id, "sha256:" + "0" * 64) is None
    assert loader("nope", trace.content_digest) is None
    chained = chain_loaders([lambda t, d: None, loader])
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=DefinitionRegistry(), trace_loader=chained)
    assert service.resolve_trace(trace.trace_id, trace.content_digest).content_digest == trace.content_digest
    assert service.store.has_source(trace.trace_id, trace.content_digest)  # materialized on first resolve


def test_http_broker_maps_wire_errors_and_never_swallows(tmp_path: Path) -> None:
    import fastapi
    import httpx
    from fastapi.testclient import TestClient

    from synth_containers.tracing.annotation import ReservationError
    from synth_containers.tracing.annotation.http_broker import HttpReservationBroker

    inner = LocalReservationBroker(tmp_path / "broker")
    host = fastapi.FastAPI()

    @host.post("/reservations/{rid}/claim")
    def claim(rid: str, body: dict):
        try:
            return inner.claim(rid, binding=ReservationBindingV1(**body["binding"]), job_id=body["job_id"]).to_dict()
        except ReservationError as error:
            status = 404 if error.code == "reservation_unknown" else 409
            return fastapi.responses.JSONResponse(status_code=status, content={"code": error.code, "message": str(error)})

    @host.post("/reservations/{rid}/reconcile")
    def reconcile(rid: str, body: dict):
        if body["outcome"] == "boom":
            return fastapi.responses.JSONResponse(status_code=503, content={"code": "ledger_down"})
        inner.reconcile(rid, job_id=body["job_id"], outcome=body["outcome"], actual_cost_usd_micros=body.get("actual_cost_usd_micros"))
        return {"ok": True}

    transport = httpx.WSGITransport if False else None
    client = TestClient(host)
    broker = HttpReservationBroker("http://broker", token="t", client=client)  # TestClient is an httpx.Client
    binding = ReservationBindingV1(trace_digest="sha256:" + "a" * 64, annotator_id="x", model="m", session_id="s")
    with pytest.raises(ReservationError) as error:
        broker.claim("rsv_forged", binding=binding, job_id="job-1")
    assert error.value.code == "reservation_unknown"
    issued = inner.issue(cap_usd_micros=100_000, binding=binding)
    claimed = broker.claim(issued.reservation_id, binding=binding, job_id="job-1")
    assert claimed.claimed_by_job_id == "job-1" and claimed.cap_usd_micros == 100_000
    assert broker.claim(issued.reservation_id, binding=binding, job_id="job-1").reservation_id == issued.reservation_id  # idempotent resume
    with pytest.raises(ReservationError) as error:
        broker.claim(issued.reservation_id, binding=binding, job_id="job-2")
    assert error.value.code == "reservation_consumed"
    with pytest.raises(ReservationError) as error:
        broker.reconcile(issued.reservation_id, job_id="job-1", outcome="boom", actual_cost_usd_micros=None)
    assert error.value.code == "ledger_down"
    broker.reconcile(issued.reservation_id, job_id="job-1", outcome="sealed", actual_cost_usd_micros=1234)
    assert inner.get(issued.reservation_id).actual_cost_usd_micros == 1234
    assert client.headers.get("Authorization") is None  # token only travels per request


def test_signed_reservations_verify_bind_and_are_single_use(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation import ReservationError
    from synth_containers.tracing.annotation.signed_broker import SignedReservationBroker, decode_signed_reservation, issue_signed_reservation

    secret = b"launch-time-secret"
    broker = SignedReservationBroker(tmp_path / "rsv", secret=secret)
    binding = ReservationBindingV1(trace_digest="sha256:" + "a" * 64, annotator_id="x", model="m", session_id="s")
    token = issue_signed_reservation(secret=secret, cap_usd_micros=250_000, binding=binding, approver="josh")
    payload = decode_signed_reservation(token)
    with pytest.raises(ReservationError) as error:
        broker.claim(token[:-4] + "AAAA", binding=binding, job_id="j1")  # tampered signature/body
    assert error.value.code in {"reservation_signature_invalid", "reservation_unknown"}
    forged = issue_signed_reservation(secret=b"other", cap_usd_micros=250_000, binding=binding)
    with pytest.raises(ReservationError) as error:
        broker.claim(forged, binding=binding, job_id="j1")
    assert error.value.code == "reservation_signature_invalid"
    with pytest.raises(ReservationError) as error:
        broker.claim(token, binding=replace(binding, session_id="other"), job_id="j1")
    assert error.value.code == "reservation_binding_mismatch"
    claimed = broker.claim(token, binding=binding, job_id="j1")
    assert claimed.reservation_id == payload["reservation_id"] and claimed.cap_usd_micros == 250_000 and claimed.metadata["signed"]
    assert broker.claim(token, binding=binding, job_id="j1").claimed_at == claimed.claimed_at  # idempotent resume
    with pytest.raises(ReservationError) as error:
        broker.claim(token, binding=binding, job_id="j2")
    assert error.value.code == "reservation_consumed"
    broker.reconcile(claimed.reservation_id, job_id="j1", outcome="sealed", actual_cost_usd_micros=4000)
    assert broker.reconciled()[0]["actual_cost_usd_micros"] == 4000 and broker.reconciled()[0]["metadata"]["reconciled_via"] == "local"
    with pytest.raises(ReservationError):
        broker.reconcile("rsv_never", job_id="j1", outcome="sealed", actual_cost_usd_micros=None)
    # expiry is enforced at claim time
    stale = issue_signed_reservation(secret=secret, cap_usd_micros=1, binding=binding, expires_at="2000-01-01T00:00:00Z")
    with pytest.raises(ReservationError) as error:
        broker.claim(stale, binding=binding, job_id="j3")
    assert error.value.code == "reservation_expired"
    # end to end through the service: the token is the reservation id
    service, trace = _paid_service(tmp_path, broker)
    live_binding = ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model="gpt-5.6-luna", session_id="sess-1")
    live = issue_signed_reservation(secret=secret, cap_usd_micros=300_000, binding=live_binding)
    job = service.submit(service.request_for(trace, "test.paid", limits=AnnotationJobLimitsV1(max_total_tokens=50_000)), reservation_id=live, session_id="sess-1")
    assert str(job.state) == AnnotationJobState.PREPARED and job.request.limits.max_cost_usd == 0.3
    assert service.store.ledger.get(job.job_id).stage == "prepared"


def test_whole_trace_abstention_from_a_part_scoped_annotator_is_not_a_rejection(tmp_path: Path) -> None:
    def program(document, context):
        proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
        proposal["abstentions"].append({"annotation_type": "belief", "reason": "no_tool_calls", "requirement": "tool_call parts"})
        return proposal

    definition = _definition("test.partscoped", scope="part")
    service, trace = _service(tmp_path, extra=[(definition, _program("test.partscoped.program"), program)])
    job = service.submit_and_run(service.request_for(trace, "test.partscoped"))
    assert str(job.state) == AnnotationJobState.ABSTAINED and job.rejected_count == 0 and job.abstained_count == 1
    receipt = service.store.receipts(job.job_id)[-1]
    assert receipt.detail["global_abstentions"][0]["reason"] == "no_tool_calls"


def test_trace_get_event_exposes_outcome_payloads() -> None:
    from synth_containers.tracing.annotation import TraceInspectionTools

    trace = build_craftax_smoke_trace()
    tools = TraceInspectionTools(trace, limits=AnnotationJobLimitsV1())
    listing = tools.call("trace_list_entities", {"kind": "event", "event_type": "craftax.eval.run.terminal"})
    terminal = listing["items"][0]
    assert terminal["preview"]["reward"] == 2.0 and terminal["preview"]["stopped_on"] == "max_steps"
    event = tools.call("trace_get_event", {"event_id": terminal["event_id"]})
    assert event["payload"]["reward"] == 2.0 and event["payload"]["usage"]["calls"] == 5
    unlocked = tools.call("trace_list_entities", {"kind": "event", "event_type": "craftax.transcript"})
    assert any(item.get("preview", {}).get("payload.achievement") == "collect_wood" for item in unlocked["items"])


def test_trace_selector_is_canonical_regardless_of_echoed_entity_id() -> None:
    from synth_containers.tracing.annotation.tools import ToolArgumentError, build_selector

    trace = build_craftax_smoke_trace()
    bare = build_selector(trace, {"kind": "trace"})
    echoed = build_selector(trace, {"kind": "trace", "entity_id": trace.trace_id})
    assert bare.entity_id is None and echoed.entity_id is None
    assert content_digest(bare) == content_digest(echoed)
    with pytest.raises(ToolArgumentError):
        build_selector(trace, {"kind": "trace", "entity_id": "span_somewhere"})
    with pytest.raises(ToolArgumentError):
        build_selector(trace, {"kind": "trace", "part_id": "msg_x:content"})


def test_consensus_payload_satisfies_required_typed_fields() -> None:
    from synth_containers.tracing.annotation.validation import ProposalValidator
    from synth_containers.tracing.models.standards import (
        AnnotationPayloadFieldV1,
        AnnotationPayloadSchemaV1,
        ProducerRefV1,
    )
    from synth_containers.tracing.models.evidence import TraceEvidenceBundleV5, TraceRefV5

    trace = build_craftax_smoke_trace()
    base = _definition("test.payload", scope="trace")
    contract = replace(
        base.output_contract,
        payload_schema=AnnotationPayloadSchemaV1(
            schema_id="test.payload",
            version="1",
            fields=(
                AnnotationPayloadFieldV1(field_name="quality_0_4", value_kind="integer", required=True),
                AnnotationPayloadFieldV1(field_name="note", value_kind="string", required=False),
            ),
        ),
    )
    definition = replace(base, output_contract=contract).sealed()
    records = []
    for index, (labels, quality, note, entity) in enumerate(
        (
            (["belief.contradicted"], 2, "a", None),
            (["belief.contradicted"], 4, "b", trace.trace_id),
            (["belief.correct"], 3, "b", None),
        )
    ):
        validator = ProposalValidator(trace, definition=definition, producer=ProducerRefV1(kind="deterministic", name="t"), job_id=f"job-{index}")
        proposal = empty_proposal(trace_id=trace.trace_id, trace_digest=trace.content_digest)
        target = {"kind": "trace"} if entity is None else {"kind": "trace", "entity_id": entity}
        finding = _finding(target, labels, [{"kind": "trace"}])
        finding["payload"] = {"quality_0_4": quality, "note": note}
        proposal["findings"].append(finding)
        result = validator.validate(proposal)
        assert result.annotations and not result.rejected, result.rejected
        records.extend(result.annotations)
    report = agreement(records)
    assert report.group_count == 1, [r.target.to_dict() for r in records]
    consensus = consensus_annotation(records, definition=definition)
    assert consensus is not None and consensus.labels == ("belief.contradicted",)
    assert consensus.payload == {"quality_0_4": 3, "note": "b"}
    bundle = TraceEvidenceBundleV5(
        bundle_id="bundle_test",
        trace_ref=TraceRefV5(trace_id=trace.trace_id, content_digest=trace.content_digest),
        created_at="2026-08-31T00:00:00Z",
        annotator_definitions=(definition,),
        annotations=tuple(records) + (consensus,),
    ).sealed()
    assert not _errors(validate_evidence(trace, bundle)[0])
