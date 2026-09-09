"""Scheduler, campaign fan-out, and the model-API runner class."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from synth_containers.tracing.annotation import (
    AnnotationCampaign,
    AnnotationJobLimitsV1,
    AnnotationJobState,
    AnnotationScheduler,
    AnnotationService,
    AnnotationStore,
    AnnotatorPlan,
    AnnotatorProgramV1,
    CampaignPlan,
    CompletionResult,
    DefinitionRegistry,
    LocalReservationBroker,
    ModelApiRunner,
    ReservationBindingV1,
    RunnerKind,
    ThroughputLimits,
    build_craftax_compaction_trace,
    build_craftax_smoke_trace,
    plan_from_refs,
    register_builtin_annotators,
)
from synth_containers.tracing.annotation.builtin import ENVIRONMENT_STEP_STATUS_ID, TOOL_CALL_INTEGRITY_ID
from synth_containers.tracing.annotation.proposal import PROPOSAL_SCHEMA_VERSION, empty_proposal
from synth_containers.tracing.models.standards import (
    AnnotationOutputContractV1,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    TraceAnnotatorDefinitionV1,
)


class CountingRunner:
    """Deterministic-class runner that records peak concurrency and sleeps a little."""

    kind = RunnerKind.DETERMINISTIC.value
    version = "counting@1"

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()
        self.order: list[str] = []

    def run(self, context):
        from synth_containers.tracing.annotation.service import DeterministicRunner

        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.order.append(context.job.job_id)
        try:
            time.sleep(self.delay)
            return DeterministicRunner().run(context)
        finally:
            with self.lock:
                self.active -= 1


def _slow_definition(annotator_id: str) -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id=annotator_id,
        name=annotator_id,
        purpose="test",
        taxonomy=("x",),
        required_subject_scope="trace",
        minimum_evidence=1,
        confidence_semantics="deterministic",
        output_contract=AnnotationOutputContractV1(task_kind=AnnotationTaskKind.CLASSIFY, annotation_types=("t",), taxonomy=(AnnotationTaxonV1(label="x"),)),
    ).sealed()


def _trace_finding(document, context):
    proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
    proposal["findings"].append({"target": {"kind": "trace"}, "annotation_type": "t", "labels": ["x"], "payload": {}, "confidence": 1.0, "rationale": "", "evidence": [{"kind": "trace"}]})
    return proposal


def _service(tmp_path: Path, runner: CountingRunner | None = None):
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    for name in ("a", "b", "c"):
        registry.register(_slow_definition(f"test.slow.{name}"), AnnotatorProgramV1(program_id=f"test.slow.{name}.p", runner_kind=RunnerKind.DETERMINISTIC, program_ref="t").sealed(), deterministic_program=_trace_finding, domain="test")
    runners = {RunnerKind.DETERMINISTIC.value: runner} if runner else None
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners=runners)
    traces = [build_craftax_smoke_trace(), build_craftax_compaction_trace()]
    for trace in traces:
        service.register_trace(trace)
    return service, traces


def test_scheduler_honours_per_class_and_global_limits(tmp_path: Path) -> None:
    runner = CountingRunner(delay=0.08)
    service, traces = _service(tmp_path, runner)
    scheduler = AnnotationScheduler(service, limits=ThroughputLimits(max_concurrent_total=3, per_class={RunnerKind.DETERMINISTIC.value: 2}, poll_seconds=0.01))
    jobs = []
    for trace in traces:
        for name in ("a", "b", "c"):
            for repeat in range(2):
                jobs.append(service.submit(service.request_for(trace, f"test.slow.{name}", repeat_index=repeat)))
    positions = [scheduler.enqueue(job.job_id) for job in jobs]
    assert positions == list(range(1, 13))
    assert scheduler.drain(timeout=30)
    assert runner.peak == 2  # class limit binds before the global limit
    assert scheduler.snapshot()["peak_by_class"][RunnerKind.DETERMINISTIC.value] == 2
    assert all(str(service.get(job.job_id).state) == AnnotationJobState.SEALED for job in jobs)
    assert scheduler.snapshot()["completed"] == 12 and scheduler.snapshot()["queued"] == 0
    # FIFO: the first two started are the first two enqueued
    assert runner.order[:2] == [jobs[0].job_id, jobs[1].job_id]


def test_scheduler_background_start_recovers_and_dedupes(tmp_path: Path) -> None:
    runner = CountingRunner(delay=0.02)
    service, traces = _service(tmp_path, runner)
    job = service.submit(service.request_for(traces[0], "test.slow.a"))
    scheduler = AnnotationScheduler(service, limits=ThroughputLimits(poll_seconds=0.01))
    assert scheduler.enqueue(job.job_id) == 1 and scheduler.enqueue(job.job_id) == 0  # already queued
    scheduler.start()  # also picks up prepared jobs left by other processes
    assert scheduler.wait_for([job.job_id], timeout=10)
    scheduler.stop()
    assert str(service.get(job.job_id).state) == AnnotationJobState.SEALED
    assert scheduler.enqueue(job.job_id) == -1  # terminal jobs are never re-queued


def test_paid_inflight_cap_serialises_expensive_jobs(tmp_path: Path) -> None:
    from synth_containers.tracing.annotation import CodexAppServerRunner, ScriptedAppServer

    def agent():
        manifest = yield ("trace_get_manifest", {})
        time.sleep(0.05)
        return json.dumps({"schema_version": PROPOSAL_SCHEMA_VERSION, "source_trace_id": manifest["trace_id"], "source_trace_digest": manifest["trace_digest"], "findings": [], "abstentions": []})

    registry = DefinitionRegistry()
    definition = TraceAnnotatorDefinitionV1(annotator_id="test.paid", name="p", purpose="p", taxonomy=("x",), required_subject_scope="message", model="m", output_contract=AnnotationOutputContractV1(task_kind=AnnotationTaskKind.CLASSIFY, annotation_types=("t",), taxonomy=(AnnotationTaxonV1(label="x"),))).sealed()
    registry.register(definition, AnnotatorProgramV1(program_id="test.paid.p", runner_kind=RunnerKind.CODEX_APP_SERVER, prompt="go", paid=True).sealed(), domain="test")
    active = {"n": 0, "peak": 0}
    lock = threading.Lock()

    class Fake(ScriptedAppServer):
        def start(self):
            with lock:
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])

        def close(self):
            with lock:
                active["n"] -= 1
            super().close()

    runner = CodexAppServerRunner(lambda cwd: Fake(agent), poll_seconds=0.01, default_effort="low", proxy_enforces_reservation=True)
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    scheduler = AnnotationScheduler(service, limits=ThroughputLimits(max_concurrent_total=8, per_class={RunnerKind.CODEX_APP_SERVER.value: 4}, max_inflight_paid_usd_micros=250_000, poll_seconds=0.01))
    jobs = []
    for repeat in range(4):
        request = service.request_for(trace, "test.paid", repeat_index=repeat, limits=AnnotationJobLimitsV1(max_total_tokens=10_000, max_cost_usd=0.2))
        reservation = broker.issue(cap_usd_micros=200_000, binding=ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.paid", model="m", session_id="s"))
        jobs.append(service.submit(request, reservation_id=reservation.reservation_id, session_id="s"))
        scheduler.enqueue(jobs[-1].job_id)
    assert scheduler.drain(timeout=30)
    assert active["peak"] == 1  # $0.20 each under a $0.25 in-flight cap ⇒ strictly serial
    assert all(str(service.get(j.job_id).state) == AnnotationJobState.SEALED for j in jobs)
    assert scheduler.snapshot()["inflight_paid_usd_micros"] == 0


def test_campaign_fans_out_and_counts_cache_hits(tmp_path: Path) -> None:
    runner = CountingRunner(delay=0.01)
    service, traces = _service(tmp_path, runner)
    scheduler = AnnotationScheduler(service, limits=ThroughputLimits(poll_seconds=0.01))
    campaign = AnnotationCampaign(service, scheduler)
    refs = [{"kind": "trace_v5", "id": t.trace_id, "digest": t.content_digest} for t in traces] + [{"kind": "trace_v5_partial", "id": "x", "digest": "y"}]
    plan = plan_from_refs(refs, [AnnotatorPlan(ENVIRONMENT_STEP_STATUS_ID), AnnotatorPlan(TOOL_CALL_INTEGRITY_ID), AnnotatorPlan("test.slow.a", repeats=2)], session_id="sess", label="unit")
    assert len(plan.traces) == 2 and plan.job_count == 8
    estimate = campaign.estimate(plan)
    assert estimate.job_count == 8 and estimate.cached == 0 and estimate.paid_new == 0 and estimate.free_new == 8
    run = campaign.submit(plan)
    assert run.enqueued == 8 and run.cache_hits == 0 and not run.refused
    assert campaign.wait(run, timeout=30)
    summary = run.summary(service)
    assert summary["terminal"] and summary["states"] == {"sealed": 8}
    again = campaign.submit(plan)
    assert again.cache_hits == 8 and again.enqueued == 0
    assert campaign.estimate(plan).cached == 8
    # a paid annotator without a reservation provider is refused per job, not per campaign
    registry = service.registry
    registry.register(TraceAnnotatorDefinitionV1(annotator_id="test.paid2", name="p", purpose="p", taxonomy=("x",), required_subject_scope="message", model="m", output_contract=AnnotationOutputContractV1(task_kind=AnnotationTaskKind.CLASSIFY, annotation_types=("t",), taxonomy=(AnnotationTaxonV1(label="x"),))).sealed(), AnnotatorProgramV1(program_id="test.paid2.p", runner_kind=RunnerKind.CODEX_APP_SERVER, prompt="go", paid=True).sealed(), domain="test")
    from synth_containers.tracing.annotation import CodexAppServerRunner

    service.runners[RunnerKind.CODEX_APP_SERVER.value] = CodexAppServerRunner(lambda cwd: None, default_effort="low", proxy_enforces_reservation=True)
    mixed = campaign.submit(CampaignPlan(traces=plan.traces, annotators=(AnnotatorPlan("test.paid2", limits=AnnotationJobLimitsV1(max_total_tokens=1000)), AnnotatorPlan(ENVIRONMENT_STEP_STATUS_ID)), session_id="sess"))
    assert len(mixed.refused) == 2 and all(r["reason"] == "reservation_required" for r in mixed.refused)
    assert mixed.cache_hits == 2
    paid_estimate = campaign.estimate(CampaignPlan(traces=plan.traces, annotators=(AnnotatorPlan("test.paid2", limits=AnnotationJobLimitsV1(max_total_tokens=1000, max_cost_usd=0.3)),), session_id="sess"))
    assert paid_estimate.paid_new == 2 and paid_estimate.max_cost_usd == pytest.approx(0.6) and paid_estimate.requires_reservations == 2
    assert [job["repeat_index"] for job in paid_estimate.paid_jobs] == [0, 0] and paid_estimate.paid_jobs[0]["model"] == "m"


def test_model_api_runner_is_a_first_class_paid_class(tmp_path: Path) -> None:
    calls: list[dict] = []

    def complete(*, model, instructions, context, schema, max_output_tokens):
        calls.append({"model": model, "context": context})
        assert "trace_digest:" in context and "Non-negotiable rules" in instructions and schema["properties"]["schema_version"]["enum"] == [PROPOSAL_SCHEMA_VERSION]
        trace_id = context.split("trace_id: ")[1].splitlines()[0]
        digest = context.split("trace_digest: ")[1].splitlines()[0]
        reply_id = next(line for line in context.splitlines() if "[assistant]" in line).split("entity_id=")[1].split()[0]
        return CompletionResult(text=json.dumps({"schema_version": PROPOSAL_SCHEMA_VERSION, "source_trace_id": trace_id, "source_trace_digest": digest, "findings": [{"target": {"kind": "message", "entity_id": reply_id}, "annotation_type": "t", "labels": ["x"], "payload": {}, "confidence": 0.7, "rationale": "single shot", "evidence": [{"kind": "message", "entity_id": reply_id}]}], "abstentions": []}), input_tokens=3000, output_tokens=200, total_tokens=3200, cost_usd=0.004)

    registry = DefinitionRegistry()
    definition = TraceAnnotatorDefinitionV1(annotator_id="test.model", name="m", purpose="m", taxonomy=("x",), required_subject_scope="message", minimum_evidence=1, output_contract=AnnotationOutputContractV1(task_kind=AnnotationTaskKind.CLASSIFY, annotation_types=("t",), taxonomy=(AnnotationTaxonV1(label="x"),))).sealed()
    registry.register(definition, AnnotatorProgramV1(program_id="test.model.p", runner_kind=RunnerKind.MODEL_API, prompt="Label the first reply.", paid=True).sealed(), domain="test")
    runner = ModelApiRunner(complete, default_model="cheap-model", usd_per_million_tokens=1.0)
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners={runner.kind: runner}, broker=broker)
    trace = build_craftax_smoke_trace()
    service.register_trace(trace)
    request = service.request_for(trace, "test.model", limits=AnnotationJobLimitsV1(max_total_tokens=10_000, max_cost_usd=0.01))
    assert request.model == "cheap-model" and service.estimate(request).runner_kind == "model_api"
    reservation = broker.issue(cap_usd_micros=10_000, binding=ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.model", model="cheap-model", session_id="s"))
    job = service.submit_and_run(request, reservation_id=reservation.reservation_id, session_id="s")
    assert str(job.state) == AnnotationJobState.SEALED, job.error
    assert job.applied_count == 1 and job.usage.cost_usd == 0.004 and job.usage.cost_status == "reported"
    assert broker.get(reservation.reservation_id).actual_cost_usd_micros == 4000
    head = service.evidence_head(trace.trace_id)
    assert str(head.annotations[0].author_kind) == "model" and head.annotations[0].producer.model == "cheap-model"
    assert service.store.get_execution_trace(job.job_id) is not None
    # token ceiling from the cost cap: $0.001 at $1/M tokens = 1000 tokens < 3200 used
    tight = service.request_for(trace, "test.model", repeat_index=1, limits=AnnotationJobLimitsV1(max_total_tokens=10_000, max_cost_usd=0.001))
    reservation2 = broker.issue(cap_usd_micros=10_000, binding=ReservationBindingV1(trace_digest=trace.content_digest, annotator_id="test.model", model="cheap-model", session_id="s"))
    capped = service.submit_and_run(tight, reservation_id=reservation2.reservation_id, session_id="s")
    assert capped.error.code == "token_limit_exceeded"
