"""The annotation job lifecycle: prepare, run, validate, seal, persist, receipt.

``AnnotationService`` is the one place that turns a request into evidence. It is
runner-agnostic: deterministic programs and Codex app-server tasks both produce a
proposal, and everything after that (selector validation, evidence append,
indexing, receipts) is shared and identical.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

from ..canonical import content_digest, text_digest, utc_now
from ..evidence_ops import attach_many, new_evidence_bundle
from ..models.document import TraceDocumentV5
from ..models.evidence import TraceEvidenceBundleV5
from ..models.projection import ProjectionManifestV1
from ..models.selectors import TraceSelectorV1
from ..models.standards import (
    AnnotationReviewState,
    AnnotationV1,
    ProducerKind,
    ProducerRefV1,
    RubricDefinitionV2,
)
from ..validation.validator import Severity
from .broker import DenyAllBroker, PaidComputeBroker, ReservationBindingV1, ReservationError, usd_to_micros
from .consensus import agreement, consensus_annotation
from .definitions import (
    DefinitionRegistry,
    ProgramContext,
    RegisteredAnnotator,
    RunnerKind,
)
from .evidence_check import validate_appended_evidence
from .execution_trace import ExecutionCapture, build_execution_trace
from .ledger import PaidLedgerEntryV1
from .jobs import (
    RUNNER_VERSION,
    AnnotationEstimateV1,
    AnnotationJobErrorCode,
    AnnotationJobErrorV1,
    AnnotationJobLimitsV1,
    AnnotationJobMode,
    AnnotationJobRequestV1,
    AnnotationJobState,
    AnnotationJobUsageV1,
    AnnotationJobV1,
    idempotency_key,
    new_job,
)
from .persistence import AnnotationStore, RevisionConflict, StoreCorruption
from .receipts import job_receipt
from .streams import AnnotationEventStreamer
from .tools import TraceInspectionTools, tool_contract_digest
from .trace_index import SealedTraceCache, SealedTraceIndex
from .validation import ProposalValidator, producer_for
from .workspace import (
    build_workspace_manifest,
    materialize_workspace,
    render_instructions,
    unlock_workspace,
)


class AnnotationServiceError(RuntimeError):
    def __init__(self, code: AnnotationJobErrorCode | str, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = str(code)
        self.detail = detail or {}

    def as_error(self) -> AnnotationJobErrorV1:
        return AnnotationJobErrorV1(code=self.code, message=str(self), detail=self.detail)


@dataclass(frozen=True, slots=True)
class RunContext:
    job: AnnotationJobV1
    document: TraceDocumentV5
    entry: RegisteredAnnotator
    rubric: RubricDefinitionV2 | None
    tools: TraceInspectionTools
    workspace_dir: Path
    instructions_text: str
    instructions_digest: str
    source_annotations: tuple[AnnotationV1, ...] = ()
    cancel_requested: Callable[[], bool] = lambda: False


@dataclass(frozen=True, slots=True)
class RunOutcome:
    proposal: Any
    capture: ExecutionCapture
    error: AnnotationJobErrorV1 | None = None
    producer: ProducerRefV1 | None = None


class AnnotatorRunner(Protocol):
    kind: str

    def run(self, context: RunContext) -> RunOutcome: ...


class DeterministicRunner:
    kind = RunnerKind.DETERMINISTIC.value

    def run(self, context: RunContext) -> RunOutcome:
        started = utc_now()
        clock = time.monotonic()
        program = context.entry.deterministic_program
        assert program is not None
        try:
            proposal = program(
                context.document,
                ProgramContext(
                    definition=context.entry.definition,
                    rubric=context.rubric,
                    parameters=dict(context.entry.program.parameters),
                    source_annotations=context.source_annotations,
                ),
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - a program crash is a typed job failure
            proposal = None
            error = AnnotationJobErrorV1(
                code=AnnotationJobErrorCode.INTERNAL,
                message=f"deterministic program raised {type(exc).__name__}: {exc}",
            )
        ended = utc_now()
        capture = ExecutionCapture(
            started_at=started,
            ended_at=ended,
            instructions_digest=context.instructions_digest,
            tool_calls=tuple(context.tools.calls),
            final_output=proposal if isinstance(proposal, dict) else None,
            usage=AnnotationJobUsageV1(
                tool_calls=len(context.tools.calls),
                tool_bytes=context.tools.total_bytes,
                cost_usd=0.0,
                cost_status="free",
                wall_time_seconds=time.monotonic() - clock,
            ),
            runner_kind=self.kind,
            error=error.message if error else None,
        )
        producer = producer_for(
            context.entry.definition,
            kind=ProducerKind.DETERMINISTIC,
            name=context.entry.program.program_id,
            version=context.entry.program.version,
            config_digest=context.entry.program.content_digest,
        )
        return RunOutcome(proposal=proposal, capture=capture, error=error, producer=producer)


TraceLoader = Callable[[str, str], Optional[TraceDocumentV5]]


def cost_enforcement_for(runner: Any, model: str | None) -> str | None:
    """Ask a runner how it bounds dollars for ``model``; runners predating the price table take no argument."""

    probe = getattr(runner, "cost_enforcement", None)
    if probe is None:
        return None
    try:
        takes_model = bool(inspect.signature(probe).parameters)
    except (TypeError, ValueError):
        takes_model = False
    return probe(model) if takes_model else probe()


class AnnotationService:
    def __init__(
        self,
        *,
        store: AnnotationStore,
        registry: DefinitionRegistry,
        runners: dict[str, AnnotatorRunner] | None = None,
        trace_loader: TraceLoader | None = None,
        projections: Callable[[str, str], tuple[ProjectionManifestV1, ...]] | None = None,
        broker: PaidComputeBroker | None = None,
        trace_cache_size: int = 8,
    ) -> None:
        self.store = store
        self.registry = registry
        # Paid execution is impossible until the host hands us a broker.
        self.broker: PaidComputeBroker = broker or DenyAllBroker()
        self.runners: dict[str, AnnotatorRunner] = {RunnerKind.DETERMINISTIC.value: DeterministicRunner()}
        if runners:
            self.runners.update(runners)
        self.trace_loader = trace_loader
        self.projections = projections
        # Selector resolutions and verified evidence revisions, per sealed trace
        # digest. Sealed traces are immutable, so nothing here ever goes stale;
        # the LRU only bounds memory.
        self.trace_index = SealedTraceCache(max_traces=trace_cache_size)
        self.events = AnnotationEventStreamer(store)

    def index_for(self, document: TraceDocumentV5) -> SealedTraceIndex:
        return self.trace_index.get(document)

    # -- traces ---------------------------------------------------------------------

    def register_trace(self, document: TraceDocumentV5) -> str:
        """Materialize a sealed trace as local authority; returns its digest."""

        self.store.put_source(document)
        return document.content_digest

    def resolve_trace(self, trace_id: str, digest: str) -> TraceDocumentV5:
        document = self.store.get_source(trace_id, digest)
        if document is None and self.trace_loader is not None:
            document = self.trace_loader(trace_id, digest)
            if document is not None:
                if document.content_digest != digest or content_digest(document) != digest:
                    raise AnnotationServiceError(
                        AnnotationJobErrorCode.SOURCE_DIGEST_MISMATCH,
                        f"loaded trace {trace_id} has digest {document.content_digest}, expected {digest}",
                    )
                self.store.put_source(document)
        if document is None:
            raise AnnotationServiceError(
                AnnotationJobErrorCode.SOURCE_TRACE_UNAVAILABLE,
                f"sealed trace {trace_id}@{digest} is not available locally",
            )
        return document

    # -- definitions ----------------------------------------------------------------

    def list_definitions(
        self,
        *,
        trace_schema: str | None = "synth.trace.v5",
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        return [self.registry.describe(entry) for entry in self.registry.list(trace_schema=trace_schema, domain=domain)]

    def request_for(
        self,
        document: TraceDocumentV5,
        annotator_id: str,
        *,
        mode: AnnotationJobMode | str = AnnotationJobMode.ANNOTATE,
        model: str | None = None,
        reasoning_effort: str | None = None,
        runner_kind: str | None = None,
        rubric_id: str | None = None,
        repeat_index: int = 0,
        parent_job_id: str | None = None,
        source_annotation_ids: Sequence[str] = (),
        limits: AnnotationJobLimitsV1 | None = None,
        metadata: dict[str, Any] | None = None,
        scope_session_ids: Sequence[str] = (),
    ) -> AnnotationJobRequestV1:
        """Build a request whose digests are filled from the registry and the sealed trace.

        ``scope_session_ids`` restricts the annotator to those sessions (lanes) of a
        multi-lane trace; it is part of the idempotency key via ``target_selector_ids``.
        """

        entry = self._entry(annotator_id)
        rubric = self._rubric_for(entry, rubric_id)
        resolved_limits = limits or AnnotationJobLimitsV1()
        declared_tools = entry.program.parameters.get("max_tool_calls")
        if (
            limits is None
            and isinstance(declared_tools, int)
            and declared_tools > resolved_limits.max_tool_calls
        ):
            resolved_limits = replace(resolved_limits, max_tool_calls=declared_tools)
        request = AnnotationJobRequestV1(
            source_trace_id=document.trace_id,
            source_trace_digest=document.content_digest,
            annotator_id=entry.annotator_id,
            annotator_digest=entry.definition.content_digest,
            model=model if model is not None else entry.definition.model,
            reasoning_effort=reasoning_effort,
            runner_kind=runner_kind,
            mode=_inferred_mode(entry, mode),
            rubric_id=rubric.rubric_id if rubric else None,
            rubric_digest=rubric.content_digest if rubric else None,
            repeat_index=repeat_index,
            parent_job_id=parent_job_id,
            source_annotation_ids=tuple(source_annotation_ids),
            target_selector_ids=tuple(f"session:{item}" for item in scope_session_ids),
            limits=resolved_limits,
            runner_version=RUNNER_VERSION,
            metadata=dict(metadata or {}),
        )
        return self._resolve(request, entry)

    def _entry(self, annotator_id: str, digest: str | None = None) -> RegisteredAnnotator:
        try:
            return self.registry.require(annotator_id, digest=digest)
        except KeyError:
            raise AnnotationServiceError(
                AnnotationJobErrorCode.DEFINITION_UNKNOWN, f"unknown annotator {annotator_id}"
            ) from None
        except ValueError as error:
            raise AnnotationServiceError(
                AnnotationJobErrorCode.DEFINITION_DIGEST_MISMATCH, str(error)
            ) from None

    def _rubric_for(self, entry: RegisteredAnnotator, rubric_id: str | None, digest: str | None = None) -> RubricDefinitionV2 | None:
        if rubric_id is None:
            return entry.rubric
        rubric = self.registry.rubric(rubric_id, digest=digest)
        if rubric is None:
            raise AnnotationServiceError(AnnotationJobErrorCode.RUBRIC_REQUIRED, f"unknown rubric {rubric_id}")
        return rubric

    def _resolved_runner_kind(self, request: AnnotationJobRequestV1, entry: RegisteredAnnotator) -> str:
        """Pin the runner on the request. Agentic programs may switch Codex / model-api / jesterky."""

        default = str(entry.program.runner_kind)
        requested = str(request.runner_kind) if request.runner_kind else default
        if default == RunnerKind.DETERMINISTIC and requested != default:
            raise AnnotationServiceError(
                AnnotationJobErrorCode.RUNNER_UNAVAILABLE,
                f"annotator {entry.annotator_id} is deterministic; cannot run as {requested}",
            )
        if requested == RunnerKind.DETERMINISTIC and default != RunnerKind.DETERMINISTIC:
            raise AnnotationServiceError(
                AnnotationJobErrorCode.RUNNER_UNAVAILABLE,
                f"annotator {entry.annotator_id} is paid; cannot run as deterministic",
            )
        allowed = {
            RunnerKind.DETERMINISTIC.value,
            RunnerKind.MODEL_API.value,
            RunnerKind.CODEX_APP_SERVER.value,
            RunnerKind.JESTERKY.value,
        }
        if requested not in allowed:
            raise AnnotationServiceError(AnnotationJobErrorCode.RUNNER_UNAVAILABLE, f"unknown runner_kind {requested}")
        return requested

    def _resolve(self, request: AnnotationJobRequestV1, entry: RegisteredAnnotator) -> AnnotationJobRequestV1:
        """Pin model, effort, and runner_kind *before* anything is keyed or stored.

        A request that leaves them blank inherits the definition/program/runner
        defaults now, so a later change of default can never serve this job's
        cached output to a different model or runner.
        """

        runner_kind = self._resolved_runner_kind(request, entry)
        runner = self.runners.get(runner_kind)
        if runner_kind == RunnerKind.DETERMINISTIC:
            return replace(request, model=None, reasoning_effort=None, runner_kind=runner_kind)
        model = request.model or entry.definition.model
        effort = request.reasoning_effort or entry.program.parameters.get("default_effort")
        if runner is not None:
            model = runner.resolve_model(request.model, entry.definition.model) if hasattr(runner, "resolve_model") else model
            effort = runner.resolve_effort(request.reasoning_effort, entry.program.parameters.get("default_effort")) if hasattr(runner, "resolve_effort") else effort
        if not model:
            raise AnnotationServiceError(AnnotationJobErrorCode.RUNNER_UNAVAILABLE, f"annotator {entry.annotator_id} has no model: pass one or configure a runner default")
        return replace(request, model=model, reasoning_effort=effort, runner_kind=runner_kind)

    def _key(self, request: AnnotationJobRequestV1, entry: RegisteredAnnotator) -> tuple[str, str, str]:
        tool_names = entry.program.tool_names or None
        contract_digest = tool_contract_digest(tool_names)
        runner = self.runners.get(str(request.runner_kind or entry.program.runner_kind))
        key = idempotency_key(
            request,
            program_digest=entry.program.content_digest,
            tool_contract_digest=contract_digest,
            runner_version=getattr(runner, "version", None),
        )
        return key, entry.program.content_digest, contract_digest

    # -- estimate / submit ------------------------------------------------------------

    def estimate(self, request: AnnotationJobRequestV1) -> AnnotationEstimateV1:
        entry = self._entry(request.annotator_id, request.annotator_digest)
        request = self._resolve(request, entry)
        key, _, _ = self._key(request, entry)
        cached = self.store.find_cached_job(key)
        notes: list[str] = []
        if cached is not None:
            notes.append("identical sealed request exists; no new provider call will be made")
        if entry.paid and cached is None:
            kind = str(request.runner_kind)
            if kind == RunnerKind.JESTERKY:
                notes.append("starts one paid jesterky swarm; needs a broker reservation id bound to this trace/annotator/model")
            elif kind == RunnerKind.MODEL_API:
                notes.append("starts one paid model-api completion; needs a broker reservation id bound to this trace/annotator/model")
            else:
                notes.append("starts one paid Codex app-server task; needs a broker reservation id bound to this trace/annotator/model")
            if request.limits.max_total_tokens is None:
                notes.append("paid jobs must declare max_total_tokens; the runner enforces cost as a token ceiling")
            if kind not in self.runners:
                notes.append(f"runner {kind} is not mounted on this container")
        if str(request.mode) == AnnotationJobMode.VERIFY and self._rubric_for(entry, request.rubric_id) is None:
            notes.append("verification requires a rubric")
        return AnnotationEstimateV1(
            idempotency_key=key,
            cached=cached is not None,
            cached_job_id=cached.job_id if cached else None,
            paid=entry.paid and cached is None,
            runner_kind=str(request.runner_kind or entry.program.runner_kind),
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            max_tool_calls=request.limits.max_tool_calls,
            max_total_tokens=request.limits.max_total_tokens,
            max_cost_usd=request.limits.max_cost_usd,
            repeat_index=request.repeat_index,
            requires_reservation=entry.paid and cached is None,
            resolved_model=request.model,
            resolved_reasoning_effort=request.reasoning_effort,
            notes=tuple(notes),
        )

    def submit(
        self,
        request: AnnotationJobRequestV1,
        *,
        reservation_id: str | None = None,
        session_id: str | None = None,
    ) -> AnnotationJobV1:
        """Return a cached terminal job, a job already in flight, or a new prepared job.

        Paid annotators need a ``reservation_id`` issued by the host broker; it is
        claimed atomically (single use) and bound to this job before the job exists.
        """

        entry = self._entry(request.annotator_id, request.annotator_digest)
        request = self._resolve(request, entry)
        if str(request.mode) == AnnotationJobMode.VERIFY and self._rubric_for(entry, request.rubric_id, request.rubric_digest) is None:
            raise AnnotationServiceError(AnnotationJobErrorCode.RUBRIC_REQUIRED, "verification jobs need a rubric")
        if str(request.mode) == AnnotationJobMode.ADJUDICATE and not request.source_annotation_ids:
            raise AnnotationServiceError(
                AnnotationJobErrorCode.UNSUPPORTED_FINDING,
                "adjudication jobs must name the source annotation ids they resolve",
            )
        key, program_digest, contract_digest = self._key(request, entry)
        with self.store.lock():
            cached = self.store.find_cached_job(key)
            if cached is not None:
                started = utc_now()
                self.store.save_receipt(
                    cached.job_id,
                    job_receipt(
                        cached,
                        operation="annotation.cache_lookup",
                        status="cached",
                        started_at=started,
                        detail={"cache_hit": True, "request_metadata": dict(request.metadata)},
                        output_digests=tuple(d for d in (cached.bundle_digest,) if d),
                    ),
                )
                return cached
            active = self.store.find_active_job(key)
            if active is not None:
                return active
            if entry.paid:
                if reservation_id is None:
                    raise AnnotationServiceError(
                        AnnotationJobErrorCode.RESERVATION_REQUIRED,
                        f"annotator {entry.annotator_id} starts paid compute; a broker reservation id is required",
                        {"estimate": self.estimate(request).to_dict()},
                    )
                if request.limits.max_total_tokens is None:
                    raise AnnotationServiceError(
                        AnnotationJobErrorCode.RESERVATION_REJECTED,
                        "paid jobs must declare limits.max_total_tokens; cost is enforced as a token ceiling",
                    )
            job = new_job(
                request,
                key=key,
                program_digest=program_digest,
                tool_contract_digest=contract_digest,
                reservation_id=reservation_id if entry.paid else None,
            )
            if entry.paid:
                assert reservation_id is not None
                if session_id is None:
                    raise AnnotationServiceError(
                        AnnotationJobErrorCode.RESERVATION_REJECTED,
                        "paid jobs must be bound to a session_id",
                        {"reason": "session_required"},
                    )
                runner = self.runners.get(str(request.runner_kind or entry.program.runner_kind))
                enforcement = cost_enforcement_for(runner, request.model)
                if enforcement is None:
                    raise AnnotationServiceError(
                        AnnotationJobErrorCode.RESERVATION_REJECTED,
                        "no dollar enforcement is configured for paid execution: price the resolved "
                        f"model ({request.model!r}) in the runner's price table, pin a flat price "
                        "(usd_per_million_tokens), or assert provider-proxy enforcement on the runner",
                        {"reason": "cost_enforcement_unavailable", "model": request.model},
                    )
                # Durable intent first: a crash after the claim can be resumed with this job id.
                self.store.ledger.write(
                    PaidLedgerEntryV1(
                        job_id=job.job_id,
                        reservation_id=reservation_id,
                        idempotency_key=key,
                        request=request,
                        program_digest=program_digest,
                        tool_contract_digest=contract_digest,
                        created_at=job.created_at,
                        session_id=session_id,
                        stage="intent",
                        metadata={"cost_enforcement": enforcement},
                    )
                )
                job = self._claim_and_prepare(job, request, entry, reservation_id=reservation_id, session_id=session_id)
            else:
                self.store.save_job(job)
            self.store.save_receipt(
                job.job_id,
                job_receipt(job, operation="annotation.prepare", status="prepared", started_at=job.created_at, new_state=str(job.state)),
            )
            self.events.prepared(job)
            return job

    def _claim_and_prepare(self, job: AnnotationJobV1, request: AnnotationJobRequestV1, entry: RegisteredAnnotator, *, reservation_id: str, session_id: str | None) -> AnnotationJobV1:
        """Claim (idempotent for this job id), then persist the prepared job. Ledger tracks each step."""

        try:
            reservation = self.broker.claim(
                reservation_id,
                binding=ReservationBindingV1(
                    trace_digest=request.source_trace_digest,
                    annotator_id=request.annotator_id,
                    model=request.model,
                    session_id=session_id,
                ),
                job_id=job.job_id,
            )
        except ReservationError as error:
            self.store.ledger.update(job.job_id, stage="abandoned", last_error=f"{error.code}: {error}", last_attempt_at=utc_now())
            raise AnnotationServiceError(AnnotationJobErrorCode.RESERVATION_REJECTED, str(error), {"reason": error.code}) from error
        # A signed reservation arrives as a token; the job keeps the broker's canonical id.
        self.store.ledger.update(job.job_id, stage="claimed", cap_usd_micros=reservation.cap_usd_micros, reservation_id=reservation.reservation_id)
        cap_usd = reservation.cap_usd
        declared = request.limits.max_cost_usd
        ceiling = cap_usd if declared is None else min(declared, cap_usd)
        job = replace(
            job,
            reservation_id=reservation.reservation_id,
            request=replace(request, limits=replace(request.limits, max_cost_usd=ceiling)),
            metadata={**job.metadata, "reservation": {"reservation_id": reservation.reservation_id, "cap_usd_micros": reservation.cap_usd_micros, "session_id": session_id}},
            content_digest="",
        ).sealed()
        self.store.save_job(job)
        self.store.ledger.update(job.job_id, stage="prepared")
        return job

    def recover_paid_intents(self) -> tuple[AnnotationJobV1, ...]:
        """Resume preparations that crashed between the ledger intent and the saved job.

        The broker claim is idempotent for the same job id, so a claim that landed
        before the crash is returned again; one that never landed is made now.
        """

        recovered: list[AnnotationJobV1] = []
        with self.store.lock():
            for entry_record in self.store.ledger.pending_recovery():
                if self.store.get_job(entry_record.job_id) is not None:
                    self.store.ledger.update(entry_record.job_id, stage="prepared")
                    continue
                try:
                    entry = self._entry(entry_record.request.annotator_id, entry_record.request.annotator_digest)
                except AnnotationServiceError as error:
                    self.store.ledger.update(entry_record.job_id, stage="abandoned", last_error=str(error))
                    continue
                job = AnnotationJobV1(
                    job_id=entry_record.job_id,
                    request=entry_record.request,
                    idempotency_key=entry_record.idempotency_key,
                    state=AnnotationJobState.PREPARED,
                    created_at=entry_record.created_at,
                    updated_at=utc_now(),
                    program_digest=entry_record.program_digest,
                    tool_contract_digest=entry_record.tool_contract_digest,
                    reservation_id=entry_record.reservation_id,
                    metadata={"recovered_from_ledger": True},
                ).sealed()
                try:
                    recovered.append(self._claim_and_prepare(job, entry_record.request, entry, reservation_id=entry_record.reservation_id, session_id=entry_record.session_id))
                except AnnotationServiceError:
                    continue
        return tuple(recovered)

    def retry_reconciliations(self) -> dict[str, int]:
        """Deliver every pending terminal outcome to the broker; failures stay pending and are counted."""

        acknowledged = 0
        failed = 0
        for entry in self.store.ledger.pending_reconciliation():
            if self._deliver_reconciliation(entry.job_id):
                acknowledged += 1
            else:
                failed += 1
        return {"acknowledged": acknowledged, "pending": failed}

    def _deliver_reconciliation(self, job_id: str) -> bool:
        entry = self.store.ledger.get(job_id)
        if entry is None or entry.stage != "terminal":
            return False
        try:
            self.broker.reconcile(
                entry.reservation_id,
                job_id=job_id,
                outcome=entry.outcome or "unknown",
                actual_cost_usd_micros=entry.actual_cost_usd_micros,
            )
        except ReservationError as error:
            self.store.ledger.record_attempt(job_id, f"{error.code}: {error}")
            return False
        except Exception as error:  # noqa: BLE001 - broker transport errors are recorded, never lost
            self.store.ledger.record_attempt(job_id, f"{type(error).__name__}: {error}")
            return False
        self.store.ledger.mark_acknowledged(job_id)
        return True

    def get(self, job_id: str) -> AnnotationJobV1 | None:
        return self.store.get_job(job_id)

    def cancel(self, job_id: str) -> AnnotationJobV1:
        job = self.store.get_job(job_id)
        if job is None:
            raise AnnotationServiceError(AnnotationJobErrorCode.INTERNAL, f"unknown job {job_id}")
        if job.terminal:
            return job
        state = AnnotationJobState(str(job.state))
        if state == AnnotationJobState.PREPARED:
            return self._fail(job, AnnotationJobErrorV1(code=AnnotationJobErrorCode.CANCELLED, message="cancelled before start"), state=AnnotationJobState.CANCELLED)
        flag = self.store.job_dir(job_id) / "cancel"
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(utc_now(), encoding="utf-8")
        return job

    def recover_interrupted(self) -> tuple[AnnotationJobV1, ...]:
        """Fail closed any job left non-terminal by a crash; results are never invented."""

        recovered: list[AnnotationJobV1] = list(self.recover_paid_intents())
        for state in (AnnotationJobState.RUNNING, AnnotationJobState.VALIDATING):
            for job in self.store.list_jobs(state=state.value):
                if state == AnnotationJobState.VALIDATING:
                    committed = self._recover_validated_commit(job)
                    if committed is not None:
                        recovered.append(committed)
                        continue
                recovered.append(
                    self._fail(
                        job,
                        AnnotationJobErrorV1(
                            code=AnnotationJobErrorCode.TRANSPORT_DISCONNECTED,
                            message="job was interrupted before it reached a terminal state",
                        ),
                    )
                )
        self.retry_reconciliations()
        return tuple(recovered)

    def _recover_validated_commit(self, job: AnnotationJobV1) -> AnnotationJobV1 | None:
        """Finish a validated evidence commit interrupted at any persistence boundary."""

        intent = self.store.get_commit_intent(job.job_id)
        if intent is None:
            return None
        bundle, terminal_job, receipt, expected_prior_digest = intent
        if terminal_job.job_id != job.job_id or terminal_job.revision != job.revision + 1:
            raise StoreCorruption(f"commit intent for {job.job_id} does not follow its validating revision")
        self.store.ensure_evidence_committed(
            bundle,
            expected_prior_digest=expected_prior_digest,
            job_id=job.job_id,
        )
        self.store.save_receipt(job.job_id, receipt)
        self._mark_terminal(terminal_job)
        self.store.save_job(terminal_job)
        self._reconcile(terminal_job)
        return terminal_job

    # -- run ----------------------------------------------------------------------------

    def submit_and_run(
        self,
        request: AnnotationJobRequestV1,
        *,
        reservation_id: str | None = None,
        session_id: str | None = None,
    ) -> AnnotationJobV1:
        """In-process convenience: submit then run to a terminal state on this thread."""

        job = self.submit(request, reservation_id=reservation_id, session_id=session_id)
        if job.terminal or str(job.state) != AnnotationJobState.PREPARED:
            return job
        return self.run(job.job_id)

    def run(self, job_id: str) -> AnnotationJobV1:
        with self.store.lock():
            job = self.store.get_job(job_id)
            if job is None:
                raise AnnotationServiceError(AnnotationJobErrorCode.INTERNAL, f"unknown job {job_id}")
            if job.terminal:
                return job
            if str(job.state) != AnnotationJobState.PREPARED:
                raise AnnotationServiceError(AnnotationJobErrorCode.REVISION_CONFLICT, f"job {job_id} is already {job.state}")
            # Claiming is the transition: two workers cannot both run one job.
            job = job.transition(AnnotationJobState.RUNNING)
            self.store.save_job(job)
        started = utc_now()
        clock = time.monotonic()
        self.events.running(job)
        self.store.save_receipt(
            job.job_id,
            job_receipt(job, operation="annotation.start", status="running", started_at=started, previous_state="prepared", new_state="running"),
        )
        try:
            document = self.resolve_trace(job.request.source_trace_id, job.request.source_trace_digest)
            entry = self._entry(job.request.annotator_id, job.request.annotator_digest)
            rubric = self._rubric_for(entry, job.request.rubric_id, job.request.rubric_digest)
            if entry.requires_rubric and rubric is None:
                raise AnnotationServiceError(AnnotationJobErrorCode.RUBRIC_REQUIRED, "annotator requires a rubric")
            runner = self.runners.get(str(job.request.runner_kind or entry.program.runner_kind))
            if runner is None:
                raise AnnotationServiceError(
                    AnnotationJobErrorCode.RUNNER_UNAVAILABLE,
                    f"no runner registered for {job.request.runner_kind or entry.program.runner_kind}",
                )
        except AnnotationServiceError as error:
            return self._fail(job, error.as_error())
        except StoreCorruption as error:
            return self._fail(job, AnnotationJobErrorV1(code=AnnotationJobErrorCode.STORE_CORRUPT, message=str(error)))

        head = self.store.evidence_head(document.trace_id)
        existing = head.annotations if head is not None else ()
        source_annotations = tuple(
            item for item in existing if item.annotation_id in set(job.request.source_annotation_ids)
        )
        if str(job.request.mode) == AnnotationJobMode.ADJUDICATE and len(source_annotations) != len(job.request.source_annotation_ids):
            missing = sorted(set(job.request.source_annotation_ids) - {item.annotation_id for item in source_annotations})
            return self._fail(job, AnnotationJobErrorV1(code=AnnotationJobErrorCode.UNSUPPORTED_FINDING, message="adjudication sources are absent from the evidence head", detail={"missing": missing}))

        projections = self.projections(document.trace_id, document.content_digest) if self.projections else ()
        tool_names = entry.program.tool_names or None
        tools = TraceInspectionTools(
            document,
            limits=job.request.limits,
            tool_names=tool_names,
            projections=projections,
            on_call=lambda record, current=job: self.events.tool(current, record),
        )
        manifest = build_workspace_manifest(
            job_id=job.job_id,
            request=job.request,
            trace_schema=document.schema_version,
            definition=entry.definition,
            program=entry.program,
            tool_contract_digest=tools.contract_digest(),
            tool_names=tools.tool_names,
            rubric=rubric,
            projections=projections,
        )
        instructions = render_instructions(
            manifest,
            program=entry.program,
            definition=entry.definition,
            rubric=rubric,
            source_annotations=tuple(item.to_dict() for item in source_annotations),
        )
        instructions_digest = text_digest(instructions)
        workspace_dir = self.store.workspace_dir(job.job_id)
        unlock_workspace(workspace_dir)
        materialize_workspace(
            workspace_dir,
            manifest,
            instructions=instructions,
            definition=entry.definition,
            rubric=rubric,
            tool_specs=tools.specs(),
        )
        cancel_flag = self.store.job_dir(job.job_id) / "cancel"
        context = RunContext(
            job=job,
            document=document,
            entry=entry,
            rubric=rubric,
            tools=tools,
            workspace_dir=workspace_dir,
            instructions_text=instructions,
            instructions_digest=instructions_digest,
            source_annotations=source_annotations,
            cancel_requested=cancel_flag.exists,
        )
        try:
            outcome = runner.run(context)
        except Exception as exc:  # noqa: BLE001 - runner crash is a typed failure
            outcome = RunOutcome(
                proposal=None,
                capture=ExecutionCapture(
                    started_at=started,
                    ended_at=utc_now(),
                    instructions_digest=instructions_digest,
                    tool_calls=tuple(tools.calls),
                    runner_kind=runner.kind,
                    error=f"{type(exc).__name__}: {exc}",
                ),
                error=AnnotationJobErrorV1(code=AnnotationJobErrorCode.INTERNAL, message=f"runner raised {type(exc).__name__}: {exc}"),
            )
        finally:
            unlock_workspace(workspace_dir)
        usage = replace(
            outcome.capture.usage,
            tool_calls=len(tools.calls),
            tool_bytes=tools.total_bytes,
            wall_time_seconds=outcome.capture.usage.wall_time_seconds or (time.monotonic() - clock),
        )
        agentic = runner.kind != RunnerKind.DETERMINISTIC.value
        execution_trace: TraceDocumentV5 | None = None
        if agentic:
            execution_trace = build_execution_trace(job, replace(outcome.capture, usage=usage), instructions_text=instructions)
            self.store.save_execution_trace(job.job_id, execution_trace)
        if outcome.proposal is not None:
            self.store.save_proposal(job.job_id, outcome.proposal)
        if cancel_flag.exists() and outcome.error is None:
            outcome = replace(outcome, error=AnnotationJobErrorV1(code=AnnotationJobErrorCode.CANCELLED, message="cancelled while running"))
        if outcome.error is not None:
            return self._fail(
                job,
                outcome.error,
                usage=usage,
                execution_trace=execution_trace,
                state=AnnotationJobState.CANCELLED if str(outcome.error.code) == AnnotationJobErrorCode.CANCELLED else AnnotationJobState.FAILED,
            )
        job = job.transition(
            AnnotationJobState.VALIDATING,
            usage=usage,
            workspace_manifest_digest=manifest.content_digest,
            execution_trace_id=execution_trace.trace_id if execution_trace else None,
            execution_trace_digest=execution_trace.content_digest if execution_trace else None,
        )
        self.store.save_job(job)
        self.events.validating(job)
        producer = outcome.producer or producer_for(
            entry.definition,
            kind=ProducerKind.AGENTIC if agentic else ProducerKind.DETERMINISTIC,
            name=runner.kind,
            version=RUNNER_VERSION,
            model=job.request.model,
            config_digest=entry.program.content_digest,
        )
        index = self.index_for(document)
        validator = ProposalValidator(
            document,
            definition=entry.definition,
            producer=producer,
            job_id=job.job_id,
            mode=job.request.mode,
            rubric=rubric,
            program_digest=entry.program.content_digest,
            execution_trace_id=job.execution_trace_id,
            execution_trace_digest=job.execution_trace_digest,
            existing_annotations=existing,
            allowed_source_annotation_ids=job.request.source_annotation_ids,
            index=index,
        )
        result = validator.validate(outcome.proposal)
        if result.fatal is not None:
            return self._fail(job, result.fatal, usage=usage, execution_trace=execution_trace, rejected=len(result.rejected))
        return self._seal(job, document, entry, rubric, result, started=started, usage=usage, index=index)

    # -- sealing ----------------------------------------------------------------------

    def _seal(self, job: AnnotationJobV1, document: TraceDocumentV5, entry: RegisteredAnnotator, rubric: RubricDefinitionV2 | None, result: Any, *, started: str, usage: AnnotationJobUsageV1, index: SealedTraceIndex | None = None) -> AnnotationJobV1:
        """Append the validated result to the evidence head, optimistically first.

        Attaching and validating run against a snapshot of the head *without* the
        store lock, so one job's validation never stalls every other job's commit.
        The lock is held only to confirm the head is still that snapshot, journal
        the intent, and commit. If another job appended meanwhile, the append is
        rebuilt once more on the current head *under* the lock, so the second
        attempt cannot lose; validation is incremental, so it costs only this
        job's own records.
        """

        index = index if index is not None else self.index_for(document)
        head = self.store.evidence_head(document.trace_id)
        built = self._build_revision(job, document, entry, rubric, result, head, started=started, usage=usage, index=index)
        if isinstance(built, AnnotationJobV1):
            return built
        candidate, receipt, terminal, prior_digest = built
        with self.store.lock():
            current = self.store.evidence_head(document.trace_id)
            if (current.content_digest if current is not None else None) != prior_digest:
                built = self._build_revision(job, document, entry, rubric, result, current, started=started, usage=usage, index=index)
                if isinstance(built, AnnotationJobV1):
                    return built
                candidate, receipt, terminal, prior_digest = built
            # Journal the complete validated result before the first evidence write.
            # Recovery can now finish either side of the evidence/terminal-job boundary.
            self.store.save_commit_intent(
                job.job_id,
                bundle=candidate,
                terminal_job=terminal,
                receipt=receipt,
                expected_prior_digest=prior_digest,
            )
            try:
                self.store.ensure_evidence_committed(candidate, expected_prior_digest=prior_digest, job_id=job.job_id)
            except RevisionConflict as error:
                return self._fail(job, AnnotationJobErrorV1(code=AnnotationJobErrorCode.REVISION_CONFLICT, message=str(error)), usage=usage)
        # Per-job records need no store lock; their order (receipt, outbox, terminal job) is what recovery expects.
        self.store.save_receipt(job.job_id, receipt)
        self._mark_terminal(terminal)
        self.store.save_job(terminal)
        for annotation in result.annotations:
            self.events.finding(terminal, annotation)
        self.events.terminal(terminal)
        self._reconcile(terminal)
        return terminal

    def _build_revision(
        self,
        job: AnnotationJobV1,
        document: TraceDocumentV5,
        entry: RegisteredAnnotator,
        rubric: RubricDefinitionV2 | None,
        result: Any,
        head: TraceEvidenceBundleV5 | None,
        *,
        started: str,
        usage: AnnotationJobUsageV1,
        index: SealedTraceIndex,
    ) -> AnnotationJobV1 | tuple[TraceEvidenceBundleV5, Any, AnnotationJobV1, str | None]:
        """Attach this job's records to ``head`` and validate; a failed job means refusal."""

        prior_digest = head.content_digest if head is not None else None
        source_revision_changed = (
            head is not None
            and head.trace_ref.content_digest != document.content_digest
        )
        # A trace id can acquire a new immutable source revision (for example,
        # when recovered/out-of-band events are promoted after an initial
        # annotation pass). Evidence cannot be carried across that boundary:
        # every selector is pinned to one exact trace digest. Keep the old
        # evidence revision immutable, but start the candidate from a fresh
        # bundle for the requested source digest. The final candidate still
        # supersedes the current store head so the pointer update remains one
        # atomic compare-and-set operation.
        bundle = (
            new_evidence_bundle(document)
            if head is None or source_revision_changed
            else head
        )
        records: list[tuple[str, Any]] = []
        if not any(item.annotator_id == entry.definition.annotator_id for item in bundle.annotator_definitions):
            records.append(("annotator_definition", entry.definition))
        if rubric is not None and result.verifier_result is not None:
            existing_criteria = {item.criterion_id for item in bundle.criteria}
            for criterion in rubric.criteria:
                if criterion.criterion_id not in existing_criteria:
                    records.append(("criterion", criterion))
            if not any(item.rubric_id == rubric.rubric_id for item in bundle.rubrics):
                records.append(("rubric", rubric))
            if result.verifier_definition is not None and not any(
                item.verifier_id == result.verifier_definition.verifier_id for item in bundle.verifier_definitions
            ):
                records.append(("verifier_definition", result.verifier_definition))
        records.extend(("annotation", item) for item in result.annotations)
        if result.verifier_result is not None:
            records.append(("verifier_result", result.verifier_result))
        output_digests = tuple(item.content_digest for item in result.annotations) + (
            (result.verifier_result.content_digest,) if result.verifier_result else ()
        )
        terminal_state = (
            AnnotationJobState.ABSTAINED
            if result.applied_count == 0 and result.abstained_count > 0 and result.verifier_result is None
            else AnnotationJobState.SEALED
        )
        receipt = job_receipt(
            job,
            operation="annotation.run",
            status=terminal_state.value,
            started_at=started,
            usage=usage,
            previous_state="validating",
            new_state=terminal_state.value,
            output_digests=output_digests,
            detail={
                "applied_count": result.applied_count,
                "abstained_count": result.abstained_count,
                "rejected_count": len(result.rejected),
                "global_abstentions": list(getattr(result, "global_abstentions", ())),
                "rejected": [
                    {"index": item.index, "kind": item.kind, "reason": item.reason, "detail": item.detail}
                    for item in result.rejected[:50]
                ],
                "verifier_result_id": result.verifier_result.verifier_result_id if result.verifier_result else None,
            },
        )
        records.append(("receipt", receipt))
        try:
            candidate = attach_many(bundle, records=tuple(records))
        except (ValueError, TypeError) as error:
            return self._fail(job, AnnotationJobErrorV1(code=AnnotationJobErrorCode.EVIDENCE_INVALID, message=f"evidence append refused: {error}"), usage=usage)
        if source_revision_changed:
            assert head is not None
            candidate = replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "supersedes_bundle_id": head.bundle_id,
                    "supersedes_bundle_digest": head.content_digest,
                    "source_trace_revision_from_digest": head.trace_ref.content_digest,
                },
                content_digest="",
            ).sealed()
        findings, _ = validate_appended_evidence(
            index,
            candidate,
            prior=None if source_revision_changed else head,
        )
        errors = [item for item in findings if str(item.severity) == Severity.ERROR]
        if errors:
            return self._fail(
                job,
                AnnotationJobErrorV1(
                    code=AnnotationJobErrorCode.EVIDENCE_INVALID,
                    message="sealed evidence failed validation; nothing was persisted",
                    detail={"findings": [item.to_dict() for item in errors[:25]]},
                ),
                usage=usage,
            )
        terminal = job.transition(
            terminal_state,
            bundle_id=candidate.bundle_id,
            bundle_digest=candidate.content_digest,
            prior_bundle_digest=prior_digest,
            annotation_ids=tuple(item.annotation_id for item in result.annotations),
            verifier_result_ids=(result.verifier_result.verifier_result_id,) if result.verifier_result else (),
            applied_count=result.applied_count,
            abstained_count=result.abstained_count,
            rejected_count=len(result.rejected),
            receipt_ids=tuple(job.receipt_ids) + (receipt.receipt_id,),
            usage=usage,
        )
        return candidate, receipt, terminal, prior_digest

    def _mark_terminal(self, job: AnnotationJobV1) -> None:
        """Outbox first: the outcome is durable before the terminal job revision exists."""

        if not job.reservation_id or self.store.ledger.get(job.job_id) is None:
            return
        cost = job.usage.cost_usd
        self.store.ledger.mark_terminal(
            job.job_id,
            outcome=str(job.state),
            actual_cost_usd_micros=usd_to_micros(cost) if cost is not None else None,
        )

    def _reconcile(self, job: AnnotationJobV1) -> None:
        if not job.reservation_id:
            return
        self._deliver_reconciliation(job.job_id)

    def _fail(
        self,
        job: AnnotationJobV1,
        error: AnnotationJobErrorV1,
        *,
        usage: AnnotationJobUsageV1 | None = None,
        execution_trace: TraceDocumentV5 | None = None,
        state: AnnotationJobState = AnnotationJobState.FAILED,
        rejected: int = 0,
    ) -> AnnotationJobV1:
        current = self.store.get_job(job.job_id) or job
        if current.terminal:
            return current
        started = current.updated_at
        changes: dict[str, Any] = {"error": error, "rejected_count": rejected}
        if usage is not None:
            changes["usage"] = usage
        if execution_trace is not None:
            changes["execution_trace_id"] = execution_trace.trace_id
            changes["execution_trace_digest"] = execution_trace.content_digest
        receipt = job_receipt(
            current,
            operation="annotation.run",
            status=state.value,
            started_at=started,
            usage=usage,
            previous_state=str(current.state),
            new_state=state.value,
            errors=(f"{error.code}: {error.message}",),
            detail={"error": error.to_dict()},
        )
        self.store.save_receipt(current.job_id, receipt)
        next_job = current.transition(state, receipt_ids=tuple(current.receipt_ids) + (receipt.receipt_id,), **changes)
        self._mark_terminal(next_job)
        self.store.save_job(next_job)
        self.events.terminal(next_job)
        self._reconcile(next_job)
        return next_job

    # -- reads ------------------------------------------------------------------------

    def annotations(self, trace_id: str, **filters: Any) -> tuple[AnnotationV1, ...]:
        return self.store.annotations(trace_id, **filters)

    def evidence_bundles(self, trace_id: str) -> tuple[dict[str, Any], ...]:
        return self.store.evidence_bundles(trace_id)

    def evidence_head(self, trace_id: str) -> TraceEvidenceBundleV5 | None:
        return self.store.evidence_head(trace_id)

    def get_annotation(self, annotation_id: str) -> tuple[AnnotationV1, str] | None:
        return self.store.get_annotation(annotation_id)

    def annotation_evidence(self, annotation_id: str) -> dict[str, Any] | None:
        found = self.store.get_annotation(annotation_id)
        if found is None:
            return None
        annotation, trace_id = found
        document = self.resolve_trace(trace_id, annotation.target.trace_digest)
        index = self.index_for(document)

        def describe(selector: TraceSelectorV1) -> dict[str, Any]:
            resolution = index.resolve(selector)
            return {
                "selector": selector.to_dict(),
                "resolved": resolution.resolved,
                "reason": resolution.reason,
                "entity_kind": resolution.entity_kind,
                "text": (resolution.resolved_text or "")[:4000] if resolution.resolved else None,
            }

        return {
            "annotation": annotation.to_dict(),
            "trace_id": trace_id,
            "target": describe(annotation.target),
            "evidence": [describe(item) for item in annotation.evidence],
            "job_id": self.store.annotation_job(annotation_id),
        }

    # -- review -----------------------------------------------------------------------

    def review(
        self,
        annotation_id: str,
        *,
        decision: AnnotationReviewState | str,
        reviewer: str,
        rationale: str = "",
        evidence: Sequence[dict[str, Any]] = (),
    ) -> AnnotationV1:
        """Append a reviewed revision; the original record is never modified."""

        decision_value = AnnotationReviewState(str(decision))
        if decision_value == AnnotationReviewState.UNREVIEWED:
            raise AnnotationServiceError(AnnotationJobErrorCode.UNSUPPORTED_FINDING, "a review must decide something")
        found = self.store.get_annotation(annotation_id)
        if found is None:
            raise AnnotationServiceError(AnnotationJobErrorCode.INTERNAL, f"unknown annotation {annotation_id}")
        original, trace_id = found
        document = self.resolve_trace(trace_id, original.target.trace_digest)
        index = self.index_for(document)
        from .tools import build_selector

        extra: list[TraceSelectorV1] = []
        for raw in evidence:
            selector = build_selector(document, raw)
            if not index.resolve(selector).resolved:
                raise AnnotationServiceError(AnnotationJobErrorCode.EVIDENCE_INVALID, "review evidence does not resolve")
            extra.append(selector)
        with self.store.lock():
            head = self.store.evidence_head(trace_id)
            if head is None:
                raise AnnotationServiceError(AnnotationJobErrorCode.INTERNAL, "no evidence head")
            if any(item.supersedes_id == annotation_id for item in head.annotations):
                raise AnnotationServiceError(
                    AnnotationJobErrorCode.REVISION_CONFLICT,
                    f"annotation {annotation_id} already has a successor revision",
                )
            merged = {content_digest(item): item for item in (*original.evidence, *extra)}
            revised = replace(
                original,
                annotation_id=f"{original.annotation_id}.r{original.revision + 1}",
                revision=original.revision + 1,
                supersedes_id=original.annotation_id,
                review_state=decision_value,
                evidence=tuple(merged.values()),
                created_at=utc_now(),
                content_digest="",
                # Review provenance rides in the payload namespace the validator leaves open,
                # and in the receipt appended alongside.
            ).sealed()
            receipt = job_receipt(
                self.store.get_job(self.store.annotation_job(annotation_id) or "") or _placeholder_job(original, trace_id),
                operation="annotation.review",
                status="reviewed",
                started_at=revised.created_at,
                producer=ProducerRefV1(kind=ProducerKind.HUMAN, name=reviewer),
                detail={
                    "annotation_id": annotation_id,
                    "revised_annotation_id": revised.annotation_id,
                    "decision": decision_value.value,
                    "rationale": rationale[:2000],
                    "reviewer": reviewer,
                },
                output_digests=(revised.content_digest,),
            )
            candidate = attach_many(head, records=(("annotation", revised), ("receipt", receipt)))
            findings, _ = validate_appended_evidence(index, candidate, prior=head)
            errors = [item for item in findings if str(item.severity) == Severity.ERROR]
            if errors:
                raise AnnotationServiceError(
                    AnnotationJobErrorCode.EVIDENCE_INVALID,
                    "review revision failed validation",
                    {"findings": [item.to_dict() for item in errors[:25]]},
                )
            self.store.put_evidence(candidate, expected_prior_digest=head.content_digest, job_id=None)
        return revised

    # -- consensus --------------------------------------------------------------------

    def agreement(self, trace_id: str, annotator_id: str) -> dict[str, Any] | None:
        found = self.store.annotations(trace_id, annotator_id=annotator_id)
        candidates = tuple(item for item in found if item.derivation is None)
        if not candidates:
            return None
        return agreement(candidates).to_dict()

    def consensus(self, trace_id: str, annotator_id: str, *, majority_threshold: float = 0.5) -> tuple[AnnotationV1, ...]:
        """Append consensus records for every target with >=2 applied repeats."""

        entry = self._entry(annotator_id)
        found = tuple(
            item for item in self.store.annotations(trace_id, annotator_id=annotator_id) if item.derivation is None
        )
        if not found:
            return ()
        groups: dict[tuple[str, str], list[AnnotationV1]] = {}
        for item in found:
            groups.setdefault((content_digest(item.target), item.annotation_type), []).append(item)
        derived: list[AnnotationV1] = []
        for members in groups.values():
            record = consensus_annotation(members, definition=entry.definition, majority_threshold=majority_threshold)
            if record is not None:
                derived.append(record)
        if not derived:
            return ()
        document = self.resolve_trace(trace_id, found[0].target.trace_digest)
        index = self.index_for(document)
        with self.store.lock():
            head = self.store.evidence_head(trace_id)
            assert head is not None
            existing_ids = {item.annotation_id for item in head.annotations}
            fresh = tuple(item for item in derived if item.annotation_id not in existing_ids)
            if not fresh:
                return ()
            candidate = attach_many(head, records=tuple(("annotation", item) for item in fresh))
            findings, _ = validate_appended_evidence(index, candidate, prior=head)
            errors = [item for item in findings if str(item.severity) == Severity.ERROR]
            if errors:
                raise AnnotationServiceError(
                    AnnotationJobErrorCode.EVIDENCE_INVALID,
                    "consensus records failed validation",
                    {"findings": [item.to_dict() for item in errors[:25]]},
                )
            self.store.put_evidence(candidate, expected_prior_digest=head.content_digest, job_id=None)
        return fresh


def _inferred_mode(entry: RegisteredAnnotator, requested: AnnotationJobMode | str) -> AnnotationJobMode:
    """Honor an explicit mode; otherwise take ``metadata.mode`` or ``requires_rubric``."""

    mode = AnnotationJobMode(str(requested))
    if mode != AnnotationJobMode.ANNOTATE:
        return mode
    declared = str((entry.definition.metadata or {}).get("mode") or "").strip().lower()
    if declared == "verify" or entry.requires_rubric:
        return AnnotationJobMode.VERIFY
    if declared == "adjudicate":
        return AnnotationJobMode.ADJUDICATE
    return mode


def _placeholder_job(annotation: AnnotationV1, trace_id: str) -> AnnotationJobV1:
    """Receipts need a job-shaped identity even for annotations imported without one."""

    request = AnnotationJobRequestV1(
        source_trace_id=trace_id,
        source_trace_digest=annotation.target.trace_digest,
        annotator_id=annotation.annotator_id,
        annotator_digest=annotation.annotator_digest,
    )
    now = utc_now()
    return AnnotationJobV1(
        job_id=f"review:{annotation.annotation_id}",
        request=request,
        idempotency_key="",
        state=AnnotationJobState.SEALED,
        created_at=now,
        updated_at=now,
    ).sealed()


__all__ = [
    "AnnotationService",
    "AnnotationServiceError",
    "AnnotatorRunner",
    "DeterministicRunner",
    "RunContext",
    "RunOutcome",
]
