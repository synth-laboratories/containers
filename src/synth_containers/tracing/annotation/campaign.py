"""Campaigns: fan one annotation plan out over many traces, annotators, and repeats.

A campaign is the unit an eval run's post-rollout stage submits: the sealed
trace refs of the run × the configured annotators × repeats. Cache hits resolve
immediately without a slot; everything else is queued into the scheduler. Paid
jobs need one reservation each (the broker contract is per job): the host hands
the campaign a ``reservation_for(request, session_id) -> reservation_id``
callback backed by a single approval whose total is the campaign estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..canonical import record_id, utc_now
from ..models.document import TraceDocumentV5
from .jobs import AnnotationEstimateV1, AnnotationJobLimitsV1, AnnotationJobMode, AnnotationJobRequestV1, AnnotationJobState, AnnotationJobV1
from .scheduler import AnnotationScheduler
from .service import AnnotationService, AnnotationServiceError

ReservationProvider = Callable[[AnnotationJobRequestV1, str], str]


@dataclass(frozen=True, slots=True)
class AnnotatorPlan:
    annotator_id: str
    mode: AnnotationJobMode | str = AnnotationJobMode.ANNOTATE
    repeats: int = 1
    model: str | None = None
    reasoning_effort: str | None = None
    runner_kind: str | None = None
    rubric_id: str | None = None
    limits: AnnotationJobLimitsV1 | None = None
    scope_session_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    traces: tuple[tuple[str, str], ...]  # (trace_id, trace_digest)
    annotators: tuple[AnnotatorPlan, ...]
    session_id: str | None = None
    label: str = ""

    @property
    def job_count(self) -> int:
        return len(self.traces) * sum(max(1, item.repeats) for item in self.annotators)


@dataclass(frozen=True, slots=True)
class CampaignEstimate:
    job_count: int
    cached: int
    paid_new: int
    free_new: int
    max_cost_usd: float
    max_total_tokens: int
    per_annotator: dict[str, dict[str, Any]]
    requires_reservations: int
    notes: tuple[str, ...] = ()
    paid_jobs: tuple[dict[str, Any], ...] = ()


@dataclass
class CampaignRun:
    campaign_id: str
    plan: CampaignPlan
    created_at: str
    jobs: list[AnnotationJobV1] = field(default_factory=list)
    refused: list[dict[str, Any]] = field(default_factory=list)
    cache_hits: int = 0
    enqueued: int = 0

    @property
    def job_ids(self) -> list[str]:
        return [job.job_id for job in self.jobs]

    def summary(self, service: AnnotationService) -> dict[str, Any]:
        states: dict[str, int] = {}
        applied = abstained = rejected = 0
        for job_id in self.job_ids:
            job = service.get(job_id)
            if job is None:
                continue
            states[str(job.state)] = states.get(str(job.state), 0) + 1
            applied += job.applied_count
            abstained += job.abstained_count
            rejected += job.rejected_count
        return {
            "campaign_id": self.campaign_id,
            "label": self.plan.label,
            "traces": len(self.plan.traces),
            "jobs": len(self.jobs),
            "cache_hits": self.cache_hits,
            "enqueued": self.enqueued,
            "refused": self.refused,
            "states": states,
            "applied": applied,
            "abstained": abstained,
            "rejected": rejected,
            "terminal": all((service.get(j) or job).terminal for j, job in zip(self.job_ids, self.jobs)),
        }


class AnnotationCampaign:
    def __init__(self, service: AnnotationService, scheduler: AnnotationScheduler) -> None:
        self.service = service
        self.scheduler = scheduler

    def _requests(self, plan: CampaignPlan, *, documents: dict[str, TraceDocumentV5] | None = None) -> list[tuple[AnnotatorPlan, AnnotationJobRequestV1]]:
        requests: list[tuple[AnnotatorPlan, AnnotationJobRequestV1]] = []
        for trace_id, digest in plan.traces:
            document = (documents or {}).get(trace_id) or self.service.resolve_trace(trace_id, digest)
            for item in plan.annotators:
                for repeat in range(max(1, item.repeats)):
                    requests.append(
                        (
                            item,
                            self.service.request_for(
                                document,
                                item.annotator_id,
                                mode=item.mode,
                                model=item.model,
                                reasoning_effort=item.reasoning_effort,
                                runner_kind=item.runner_kind,
                                rubric_id=item.rubric_id,
                                repeat_index=repeat,
                                limits=item.limits,
                                metadata={"campaign_label": plan.label},
                                scope_session_ids=item.scope_session_ids,
                            ),
                        )
                    )
        return requests

    def estimate(self, plan: CampaignPlan) -> CampaignEstimate:
        """One number for the approval prompt: cached jobs are free, everything else is bounded."""

        cached = paid_new = free_new = 0
        cost = 0.0
        tokens = 0
        per: dict[str, dict[str, Any]] = {}
        notes: list[str] = []
        paid_jobs: list[dict[str, Any]] = []
        for item, request in self._requests(plan):
            estimate: AnnotationEstimateV1 = self.service.estimate(request)
            row = per.setdefault(item.annotator_id, {"jobs": 0, "cached": 0, "paid_new": 0, "max_cost_usd": 0.0, "runner_kind": estimate.runner_kind, "model": estimate.resolved_model})
            row["jobs"] += 1
            if estimate.cached:
                cached += 1
                row["cached"] += 1
                continue
            if estimate.paid:
                paid_new += 1
                row["paid_new"] += 1
                # One reservation per paid job: the host issues them from this list.
                paid_jobs.append({"trace_id": request.source_trace_id, "trace_digest": request.source_trace_digest, "annotator_id": request.annotator_id, "repeat_index": request.repeat_index, "model": estimate.resolved_model, "reasoning_effort": estimate.resolved_reasoning_effort, "runner_kind": estimate.runner_kind, "max_cost_usd": estimate.max_cost_usd, "max_total_tokens": estimate.max_total_tokens, "idempotency_key": estimate.idempotency_key})
                if estimate.max_cost_usd is None:
                    notes.append(f"{item.annotator_id}: no max_cost_usd declared; the reservation cap will bound it")
                else:
                    cost += estimate.max_cost_usd
                    row["max_cost_usd"] += estimate.max_cost_usd
                tokens += estimate.max_total_tokens or 0
            else:
                free_new += 1
        return CampaignEstimate(
            job_count=len(plan.traces) * sum(max(1, a.repeats) for a in plan.annotators),
            cached=cached,
            paid_new=paid_new,
            free_new=free_new,
            max_cost_usd=cost,
            max_total_tokens=tokens,
            per_annotator=per,
            requires_reservations=paid_new,
            notes=tuple(dict.fromkeys(notes)),
            paid_jobs=tuple(paid_jobs),
        )

    def submit(self, plan: CampaignPlan, *, reservation_for: ReservationProvider | None = None) -> CampaignRun:
        """Submit every job (cache hits resolve immediately) and enqueue the rest, in plan order."""

        run = CampaignRun(campaign_id=record_id("acmp", kind="annotation_campaign", key={"label": plan.label, "traces": [d for _, d in plan.traces], "at": utc_now()}), plan=plan, created_at=utc_now())
        for item, request in self._requests(plan):
            entry = self.service.registry.get(item.annotator_id)
            reservation_id: str | None = None
            if entry is not None and entry.paid and not self.service.estimate(request).cached:
                if reservation_for is None:
                    run.refused.append({"annotator_id": item.annotator_id, "trace_digest": request.source_trace_digest, "repeat_index": request.repeat_index, "reason": "reservation_required"})
                    continue
                try:
                    reservation_id = reservation_for(request, plan.session_id or "")
                except Exception as error:  # noqa: BLE001 - the host refused; record, do not stop the campaign
                    run.refused.append({"annotator_id": item.annotator_id, "trace_digest": request.source_trace_digest, "repeat_index": request.repeat_index, "reason": f"reservation_provider: {error}"})
                    continue
            try:
                job = self.service.submit(request, reservation_id=reservation_id, session_id=plan.session_id)
            except AnnotationServiceError as error:
                run.refused.append({"annotator_id": item.annotator_id, "trace_digest": request.source_trace_digest, "repeat_index": request.repeat_index, "reason": error.code, "detail": error.detail})
                continue
            run.jobs.append(job)
            if job.terminal:
                run.cache_hits += 1
            elif str(job.state) == AnnotationJobState.PREPARED:
                if self.scheduler.enqueue(job.job_id) > 0:
                    run.enqueued += 1
        return run

    def wait(self, run: CampaignRun, *, timeout: float) -> bool:
        return self.scheduler.wait_for(run.job_ids, timeout=timeout)


def plan_from_refs(refs: Sequence[dict[str, Any]], annotators: Sequence[AnnotatorPlan], *, session_id: str | None = None, label: str = "") -> CampaignPlan:
    """Build a plan from ``{kind: trace_v5, id, digest}`` evidence refs (Workshop's shape)."""

    traces = tuple((str(ref["id"]), str(ref["digest"])) for ref in refs if ref.get("kind") == "trace_v5" and ref.get("id") and ref.get("digest"))
    return CampaignPlan(traces=traces, annotators=tuple(annotators), session_id=session_id, label=label)


__all__ = ["AnnotationCampaign", "AnnotatorPlan", "CampaignEstimate", "CampaignPlan", "CampaignRun", "plan_from_refs"]
