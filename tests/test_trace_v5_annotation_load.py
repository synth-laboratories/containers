"""Throughput load test: the scheduler saturates every cap and never exceeds one.

Slow by design (a few hundred jobs at 200 ms each; ~20 s). Runners are fakes
with deterministic sleeps and no network, one per runner class. Each phase is
arranged so that exactly one limit binds, which makes "saturated" checkable:
the observed peak must *equal* the binding cap, the other limits must never be
exceeded, and the wall time must be close to ``jobs x duration / parallelism``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from synth_containers.tracing.annotation import (
    AnnotationCampaign,
    AnnotationJobLimitsV1,
    AnnotationJobState,
    AnnotationJobUsageV1,
    AnnotationScheduler,
    AnnotationService,
    AnnotationStore,
    AnnotationWorker,
    AnnotatorPlan,
    AnnotatorProgramV1,
    CampaignPlan,
    DefinitionRegistry,
    ExecutionCapture,
    LocalReservationBroker,
    ReservationBindingV1,
    RunOutcome,
    RunnerKind,
    ThroughputLimits,
    build_craftax_smoke_trace,
    usd_to_micros,
)
from synth_containers.tracing.annotation.proposal import empty_proposal
from synth_containers.tracing.annotation.validation import producer_for
from synth_containers.tracing.canonical import utc_now
from synth_containers.tracing.models.standards import (
    AnnotationOutputContractV1,
    AnnotationTaskKind,
    AnnotationTaxonV1,
    ProducerKind,
    TraceAnnotatorDefinitionV1,
)

JOB_SECONDS = 0.2
PAID_CAP_USD = 0.10
DET, MODEL, CODEX = RunnerKind.DETERMINISTIC.value, RunnerKind.MODEL_API.value, RunnerKind.CODEX_APP_SERVER.value
ANNOTATOR = {DET: "load.det", MODEL: "load.model", CODEX: "load.codex"}


@dataclass
class Tracker:
    """Observed concurrency, measured inside the runners (the only ground truth)."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    active: dict[str, int] = field(default_factory=dict)
    peak: dict[str, int] = field(default_factory=dict)
    active_total: int = 0
    peak_total: int = 0
    active_paid_micros: int = 0
    peak_paid_micros: int = 0
    started: int = 0
    busy_seconds: float = 0.0  # integral of active_total over time
    _last: float = 0.0

    def _tick(self) -> None:
        now = time.monotonic()
        if self._last:
            self.busy_seconds += self.active_total * (now - self._last)
        self._last = now

    def enter(self, kind: str, paid_micros: int) -> None:
        with self.lock:
            self._tick()
            self.started += 1
            self.active[kind] = self.active.get(kind, 0) + 1
            self.peak[kind] = max(self.peak.get(kind, 0), self.active[kind])
            self.active_total += 1
            self.peak_total = max(self.peak_total, self.active_total)
            self.active_paid_micros += paid_micros
            self.peak_paid_micros = max(self.peak_paid_micros, self.active_paid_micros)

    def exit(self, kind: str, paid_micros: int) -> None:
        with self.lock:
            self._tick()
            self.active[kind] -= 1
            self.active_total -= 1
            self.active_paid_micros -= paid_micros


class FakeRunner:
    """One runner class, no network: sleep, then return a trace-level finding."""

    version = "fake@1"

    def __init__(self, kind: str, tracker: Tracker, *, delay: float = JOB_SECONDS, fail_every: int = 0) -> None:
        self.kind = kind
        self.tracker = tracker
        self.delay = delay
        self.fail_every = fail_every
        self.calls = 0
        self.lock = threading.Lock()

    def resolve_model(self, requested: str | None, definition_model: str | None) -> str | None:
        return requested or definition_model or "fake-model"

    def resolve_effort(self, requested: str | None, program_default: str | None) -> str | None:
        return requested or program_default

    def cost_enforcement(self, model: str | None = None) -> str | None:
        return "provider_proxy"

    def run(self, context: Any) -> RunOutcome:
        job = context.job
        cap = job.request.limits.max_cost_usd
        paid_micros = usd_to_micros(cap) if job.reservation_id and cap is not None else 0
        self.tracker.enter(self.kind, paid_micros)
        started = utc_now()
        clock = time.monotonic()
        try:
            with self.lock:
                self.calls += 1
                call = self.calls
            time.sleep(self.delay)
            if self.fail_every and call % self.fail_every == 0:
                raise RuntimeError(f"scripted crash on call {call}")
            document = context.document
            proposal = empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest)
            proposal["findings"].append({"target": {"kind": "trace"}, "annotation_type": "t", "labels": ["x"], "payload": {}, "confidence": 1.0, "rationale": "", "evidence": [{"kind": "trace"}]})
            usage = AnnotationJobUsageV1(cost_usd=0.001 if paid_micros else 0.0, cost_status="reported" if paid_micros else "free", wall_time_seconds=time.monotonic() - clock)
            capture = ExecutionCapture(started_at=started, ended_at=utc_now(), instructions_digest=context.instructions_digest, tool_calls=(), final_output=proposal, usage=usage, runner_kind=self.kind, model=job.request.model)
            producer_kind = {DET: ProducerKind.DETERMINISTIC, MODEL: ProducerKind.MODEL, CODEX: ProducerKind.AGENTIC}[self.kind]
            producer = producer_for(context.entry.definition, kind=producer_kind, name=f"fake_{self.kind}", version="1", model=job.request.model, config_digest=context.entry.program.content_digest)
            return RunOutcome(proposal=proposal, capture=capture, producer=producer)
        finally:
            self.tracker.exit(self.kind, paid_micros)


def _definition(annotator_id: str, *, model: str | None) -> TraceAnnotatorDefinitionV1:
    return TraceAnnotatorDefinitionV1(
        annotator_id=annotator_id,
        name=annotator_id,
        purpose="load",
        taxonomy=("x",),
        required_subject_scope="trace",
        minimum_evidence=1,
        confidence_semantics="deterministic",
        model=model,
        output_contract=AnnotationOutputContractV1(task_kind=AnnotationTaskKind.CLASSIFY, annotation_types=("t",), taxonomy=(AnnotationTaxonV1(label="x"),)),
    ).sealed()


@dataclass
class Harness:
    service: AnnotationService
    broker: LocalReservationBroker
    tracker: Tracker
    runners: dict[str, FakeRunner]
    traces: list[Any]
    floor_traces: list[Any]  # separate traces, so floor jobs are never cache hits for a phase
    cpu_per_job: dict[str, float] = field(default_factory=dict)

    def measure_cpu_floor(self) -> dict[str, float]:
        """Serial bookkeeping cost per job and class (run + validate + seal + persist), with no sleep.

        The seal path is pure Python, so under the GIL it never parallelizes: a
        process cannot finish jobs faster than one per this many seconds even
        with every slot busy. Saturation is judged against that floor.
        """

        settings = {kind: (runner.delay, runner.fail_every) for kind, runner in self.runners.items()}
        for runner in self.runners.values():
            runner.delay, runner.fail_every = 0.0, 0
        try:
            for kind in (DET, MODEL, CODEX):
                plan = self.plan({kind: 2}, label=f"cpu-floor-{kind}", traces=self.floor_traces)
                run = AnnotationCampaign(self.service, AnnotationScheduler(self.service)).submit(plan, reservation_for=self.reservation_for)
                assert not run.refused, run.refused
                worker = AnnotationWorker(self.service, poll_seconds=0.01)
                started = time.monotonic()
                assert worker.run_once() == len(run.jobs)
                self.cpu_per_job[kind] = (time.monotonic() - started) / len(run.jobs)
        finally:
            for kind, runner in self.runners.items():
                runner.delay, runner.fail_every = settings[kind]
                runner.calls = 0
        self.tracker.__init__()  # the floor measurement must not pollute the phase's peaks or counts
        return dict(self.cpu_per_job)

    def plan(self, repeats: dict[str, int], *, label: str, traces: list[Any] | None = None) -> CampaignPlan:
        paid_limits = AnnotationJobLimitsV1(max_total_tokens=10_000, max_cost_usd=PAID_CAP_USD)
        annotators = tuple(
            AnnotatorPlan(ANNOTATOR[kind], repeats=count, limits=paid_limits if kind != DET else None)
            for kind, count in repeats.items()
            if count > 0
        )
        return CampaignPlan(traces=tuple((t.trace_id, t.content_digest) for t in (traces if traces is not None else self.traces)), annotators=annotators, session_id="load", label=label)

    def reservation_for(self, request, session_id):  # noqa: ANN001
        binding = ReservationBindingV1(trace_digest=request.source_trace_digest, annotator_id=request.annotator_id, model=request.model, session_id=session_id)
        return self.broker.issue(cap_usd_micros=usd_to_micros(PAID_CAP_USD), binding=binding).reservation_id


def _harness(tmp_path: Path, *, traces: int, fail_every: int = 0) -> Harness:
    tracker = Tracker()
    registry = DefinitionRegistry()
    # The registry insists on a callable for deterministic programs; the fake runner never calls it.
    registry.register(_definition(ANNOTATOR[DET], model=None), AnnotatorProgramV1(program_id="load.det.p", runner_kind=RunnerKind.DETERMINISTIC, program_ref="fake").sealed(), deterministic_program=lambda document, context: empty_proposal(trace_id=document.trace_id, trace_digest=document.content_digest), domain="load")
    registry.register(_definition(ANNOTATOR[MODEL], model="fake-model"), AnnotatorProgramV1(program_id="load.model.p", runner_kind=RunnerKind.MODEL_API, prompt="go", paid=True).sealed(), domain="load")
    registry.register(_definition(ANNOTATOR[CODEX], model="fake-model"), AnnotatorProgramV1(program_id="load.codex.p", runner_kind=RunnerKind.CODEX_APP_SERVER, prompt="go", paid=True).sealed(), domain="load")
    runners = {kind: FakeRunner(kind, tracker, fail_every=fail_every) for kind in (DET, MODEL, CODEX)}
    broker = LocalReservationBroker(tmp_path / "broker")
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry, runners=runners, broker=broker)
    documents = [build_craftax_smoke_trace(lane=f"load/{index}") for index in range(traces)]
    floor_documents = [build_craftax_smoke_trace(lane=f"cpu-floor/{index}") for index in range(2)]
    for document in documents + floor_documents:
        service.register_trace(document)
    return Harness(service=service, broker=broker, tracker=tracker, runners=runners, traces=documents, floor_traces=floor_documents)


@dataclass
class Measured:
    jobs: int
    wall: float
    ideal: float  # cap-limited: jobs x duration / binding parallelism
    cpu_floor: float  # sum of serial per-job bookkeeping
    tracker: Tracker
    snapshot: dict[str, Any]
    states: dict[str, int]
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def parallelism(self) -> float:
        return self.jobs * JOB_SECONDS / self.wall

    @property
    def expected(self) -> float:
        """No-overlap pipeline bound: sleeps in parallel across the slots, plus every job's serialized bookkeeping."""

        return self.ideal + self.cpu_floor

    def line(self, label: str) -> str:
        return (
            f"{label}: jobs={self.jobs} wall={self.wall:.2f}s cap_ideal={self.ideal:.2f}s cpu_floor={self.cpu_floor:.2f}s bound={self.expected:.2f}s "
            f"parallelism={self.parallelism:.2f} mean_active={self.tracker.busy_seconds / self.wall:.2f} "
            f"peak_by_class={self.tracker.peak} peak_total={self.tracker.peak_total} "
            f"peak_paid_usd_micros={self.tracker.peak_paid_micros} states={self.states} errors={self.errors}"
        )


def _run_phase(harness: Harness, limits: ThroughputLimits, repeats: dict[str, int], *, label: str, ideal: float, timeout: float = 120.0) -> Measured:
    cpu = harness.measure_cpu_floor()
    cpu_floor = sum(len(harness.traces) * count * cpu[kind] for kind, count in repeats.items())
    scheduler = AnnotationScheduler(harness.service, limits=limits)
    campaign = AnnotationCampaign(harness.service, scheduler)
    plan = harness.plan(repeats, label=label)
    run = campaign.submit(plan, reservation_for=harness.reservation_for)
    assert not run.refused, run.refused
    assert run.enqueued == plan.job_count == len(run.jobs)
    started = time.monotonic()
    scheduler.start()
    try:
        assert scheduler.drain(timeout=timeout), scheduler.snapshot()
    finally:
        wall = time.monotonic() - started
        scheduler.stop()
    states: dict[str, int] = {}
    errors: dict[str, int] = {}
    for job_id in run.job_ids:
        job = harness.service.get(job_id)
        assert job is not None and job.terminal, job_id
        states[str(job.state)] = states.get(str(job.state), 0) + 1
        if job.error is not None:
            errors[str(job.error.code)] = errors.get(str(job.error.code), 0) + 1
    snapshot = scheduler.snapshot()
    assert snapshot["running"] == 0 and snapshot["queued"] == 0 and snapshot["inflight_paid_usd_micros"] == 0
    assert harness.tracker.active_total == 0 and harness.tracker.active_paid_micros == 0
    return Measured(jobs=len(run.jobs), wall=wall, ideal=ideal, cpu_floor=cpu_floor, tracker=harness.tracker, snapshot=snapshot, states=states, errors=errors)


def _assert_saturated(measured: Measured, *, slack: float = 0.25) -> None:
    """Wall within 25% (+0.5 s) of the no-overlap bound.

    A scheduler that serialized would take ``jobs x (duration + bookkeeping)``,
    several times the bound; one that saturates its slots lands between
    ``max(cap_ideal, cpu_floor)`` and their sum.
    """

    assert measured.wall <= measured.expected * (1.0 + slack) + 0.5, measured.line("not saturated")
    assert measured.wall >= max(measured.ideal, measured.cpu_floor) * 0.9, measured.line("faster than physically possible; the measurement is wrong")


def test_per_class_caps_bind_and_are_reached(tmp_path: Path, capsys) -> None:
    harness = _harness(tmp_path, traces=10)
    per_class = {DET: 4, MODEL: 6, CODEX: 2}
    repeats = {DET: 4, MODEL: 6, CODEX: 2}  # each class alone: 10 x repeats x 0.2 / cap = 2.0 s
    ideal = max(10 * repeats[k] * JOB_SECONDS / per_class[k] for k in per_class)
    measured = _run_phase(harness, ThroughputLimits(max_concurrent_total=12, per_class=per_class, poll_seconds=0.01), repeats, label="class-caps", ideal=ideal)
    with capsys.disabled():
        print("\n" + measured.line("class caps"))
    assert measured.states == {AnnotationJobState.SEALED.value: measured.jobs}
    assert measured.tracker.peak == per_class  # (a) each class reaches, and never exceeds, its cap
    assert measured.snapshot["peak_by_class"] == per_class
    assert measured.tracker.peak_total <= 12 and measured.snapshot["peak_running"] <= 12  # (b)
    _assert_saturated(measured)  # (d)


def test_global_cap_binds_and_is_never_exceeded(tmp_path: Path, capsys) -> None:
    harness = _harness(tmp_path, traces=10)
    per_class = {DET: 8, MODEL: 8, CODEX: 8}
    repeats = {DET: 3, MODEL: 3, CODEX: 2}  # 80 jobs / 6 slots = 2.67 s
    measured = _run_phase(harness, ThroughputLimits(max_concurrent_total=6, per_class=per_class, poll_seconds=0.01), repeats, label="global-cap", ideal=80 * JOB_SECONDS / 6)
    with capsys.disabled():
        print("\n" + measured.line("global cap"))
    assert measured.states == {AnnotationJobState.SEALED.value: measured.jobs}
    assert measured.tracker.peak_total == 6 and measured.snapshot["peak_running"] == 6  # (b) reached, never exceeded
    assert all(peak <= 8 for peak in measured.tracker.peak.values())
    _assert_saturated(measured)


def test_paid_usd_cap_binds_and_is_never_exceeded(tmp_path: Path, capsys) -> None:
    harness = _harness(tmp_path, traces=10)
    cap_micros = 5 * usd_to_micros(PAID_CAP_USD)  # five $0.10 jobs in flight at once
    per_class = {MODEL: 8, CODEX: 8}
    repeats = {MODEL: 5, CODEX: 5}  # 100 paid jobs / 5 slots = 4.0 s
    measured = _run_phase(harness, ThroughputLimits(max_concurrent_total=16, per_class=per_class, max_inflight_paid_usd_micros=cap_micros, poll_seconds=0.01), repeats, label="usd-cap", ideal=100 * JOB_SECONDS / 5)
    with capsys.disabled():
        print("\n" + measured.line("paid usd cap"))
    assert measured.states == {AnnotationJobState.SEALED.value: measured.jobs}
    assert measured.tracker.peak_paid_micros == cap_micros  # (c) reached, never exceeded
    assert measured.tracker.peak_total == 5
    assert all(peak <= 8 for peak in measured.tracker.peak.values())
    _assert_saturated(measured)
    reconciled = [harness.broker.get(job.reservation_id) for job in harness.service.store.list_jobs() if job.reservation_id]
    assert reconciled and all(item.actual_cost_usd_micros == 1_000 for item in reconciled)


def test_failed_runs_release_their_slots(tmp_path: Path, capsys) -> None:
    harness = _harness(tmp_path, traces=8, fail_every=3)
    per_class = {DET: 4, MODEL: 3, CODEX: 2}
    repeats = {DET: 3, MODEL: 3, CODEX: 2}  # 64 jobs, a third of them crash inside the runner
    measured = _run_phase(harness, ThroughputLimits(max_concurrent_total=9, per_class=per_class, poll_seconds=0.01), repeats, label="failures", ideal=max(8 * repeats[k] * JOB_SECONDS / per_class[k] for k in per_class))
    with capsys.disabled():
        print("\n" + measured.line("failures"))
    assert measured.states[AnnotationJobState.FAILED.value] == sum(runner.calls // 3 for runner in harness.runners.values())
    assert measured.states[AnnotationJobState.SEALED.value] == measured.jobs - measured.states[AnnotationJobState.FAILED.value]
    assert measured.tracker.peak == per_class  # a crashed job frees its class slot; caps still reached
    assert measured.snapshot["failed"] == measured.states[AnnotationJobState.FAILED.value]
    _assert_saturated(measured)


def test_worker_is_the_serial_control(tmp_path: Path, capsys) -> None:
    harness = _harness(tmp_path, traces=3)
    plan = harness.plan({DET: 2}, label="serial")
    jobs = [harness.service.submit(request) for _, request in AnnotationCampaign(harness.service, AnnotationScheduler(harness.service))._requests(plan)]
    worker = AnnotationWorker(harness.service, poll_seconds=0.01)
    started = time.monotonic()
    assert worker.run_once() == len(jobs)
    wall = time.monotonic() - started
    with capsys.disabled():
        print(f"\nworker (serial control): jobs={len(jobs)} wall={wall:.2f}s parallelism={len(jobs) * JOB_SECONDS / wall:.2f} peak={harness.tracker.peak_total}")
    assert harness.tracker.peak_total == 1
    assert wall >= len(jobs) * JOB_SECONDS


if __name__ == "__main__":  # pragma: no cover - manual profiling entry point
    pytest.main([__file__, "-q", "-s"])
