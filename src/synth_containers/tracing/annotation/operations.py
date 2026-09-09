"""Agent-facing operations: the exact surface Workshop tools proxy to.

Each operation is a plain function over JSON-able arguments so a thin MCP shim
(``synth_annotations_mcp``) or the HTTP router can expose it 1:1. The
descriptors carry the read-only / paid / immutable-id facts an agent must know
before calling; they are the source of truth for tool descriptions.
"""

from __future__ import annotations

from typing import Any

from synth_containers.serde import jsonable

from .jobs import AnnotationJobMode, AnnotationJobRequestV1
from .service import AnnotationService, AnnotationServiceError
from .endpoints import annotation_api_catalog, annotation_stream_descriptor
from ..validation.rehydrate import build


GUIDANCE = (
    "Inspect before launching. Reuse cached sealed evidence when the idempotency key matches. "
    "Start with deterministic facts. Use the smallest annotator set that answers the question. "
    "Cite trace evidence rather than repeating hidden reasoning. Ask once for bounded paid approval. "
    "Wait through completion and surface typed failures. Never claim that an annotation changed the underlying run."
)

OPERATION_DESCRIPTORS: tuple[dict[str, Any], ...] = (
    {
        "name": "annotation_list_definitions",
        "read_only": True,
        "paid": False,
        "description": (
            "List annotators and rubrics compatible with a trace schema and domain: ids, immutable "
            "definition/program digests, taxonomy, runner kind, and whether running one incurs paid compute."
        ),
        "arguments": {"trace_id": "optional", "domain": "optional", "trace_schema": "default synth.trace.v5"},
    },
    {
        "name": "annotation_estimate",
        "read_only": True,
        "paid": False,
        "description": (
            "Compute the idempotency key for a request and report whether a sealed result already "
            "exists (cached: no provider call), the resolved model/effort, the limits, and whether a broker "
            "reservation is needed."
        ),
        "arguments": {"request": "AnnotationJobRequestV1"},
    },
    {
        "name": "annotation_start",
        "read_only": False,
        "paid": True,
        "description": (
            "Enqueue one annotation job (accepted, not finished: poll annotation_get or "
            "subscribe to the job stream). Returns the existing sealed job when the "
            "idempotency key matches (no paid compute). Paid annotators require a "
            "reservation_id issued by the host's paid-compute broker; it is claimed once and bound to this job. "
            "Request must bear model, reasoning_effort, and runner_kind. "
            "Job, annotation, bundle, and execution-trace ids are immutable."
        ),
        "arguments": {"request": "AnnotationJobRequestV1", "reservation_id": "opaque broker reservation id (paid annotators only)", "session_id": "optional session binding"},
    },
    {
        "name": "annotation_get",
        "read_only": True,
        "paid": False,
        "description": "Fetch a job with its state, typed error, receipts, usage, sealed output ids, and stream URLs.",
        "arguments": {"job_id": "immutable job id"},
    },
    {
        "name": "annotation_events",
        "read_only": True,
        "paid": False,
        "description": (
            "Poll the annotation job event log (sequence cursor). Events cover prepared → running → "
            "tool → validating → sealed/abstained/failed/cancelled. Hidden chain-of-thought is never included."
        ),
        "arguments": {"job_id": "immutable job id", "after": "sequence cursor, default 0", "limit": "page size, default 1000"},
    },
    {
        "name": "annotation_cancel",
        "read_only": False,
        "paid": False,
        "description": "Cancel a prepared or running job. Sealed results are never removed.",
        "arguments": {"job_id": "immutable job id"},
    },
    {
        "name": "annotation_list",
        "read_only": True,
        "paid": False,
        "description": (
            "List current annotations on a trace from the local sealed evidence head with filters "
            "(annotator, type, label, status, review state). Qualitative labels are not reward."
        ),
        "arguments": {"trace_id": "trace id", "filters": "optional"},
    },
    {
        "name": "annotation_get_evidence",
        "read_only": True,
        "paid": False,
        "description": "Resolve an annotation's target and evidence selectors against the sealed trace and return the cited text.",
        "arguments": {"annotation_id": "immutable annotation id"},
    },
    {
        "name": "verification_start",
        "read_only": False,
        "paid": True,
        "description": (
            "Enqueue a rubric verification over a trace; seals a VerifierResultV2 when the worker finishes "
            "(poll verification_get). Scores never modify environment reward; `verified` milestones require "
            "engine evidence. Cached when the idempotency key matches."
        ),
        "arguments": {"request": "AnnotationJobRequestV1 with mode=verify", "reservation_id": "opaque broker reservation id (paid verifiers only)", "session_id": "optional session binding"},
    },
    {
        "name": "verification_get",
        "read_only": True,
        "paid": False,
        "description": "Fetch a verification job and its sealed verifier result id.",
        "arguments": {"job_id": "immutable job id"},
    },
    {
        "name": "annotation_review",
        "read_only": False,
        "paid": False,
        "description": (
            "Accept, reject, dispute, or flag an annotation for review. Appends a new revision that supersedes "
            "the original; historical records are never mutated."
        ),
        "arguments": {"annotation_id": "immutable annotation id", "decision": "accepted|rejected|disputed|needs_review", "reviewer": "name", "rationale": "text", "evidence": "optional selectors"},
    },
    {
        "name": "annotation_consensus",
        "read_only": False,
        "paid": False,
        "description": "Compute inter-annotator agreement for repeated jobs and append majority consensus records.",
        "arguments": {"trace_id": "trace id", "annotator_id": "annotator id", "majority_threshold": "default 0.5"},
    },
)


def _request_from(payload: dict[str, Any]) -> AnnotationJobRequestV1:
    return build(AnnotationJobRequestV1, dict(payload))


class AnnotationOperations:
    def __init__(self, service: AnnotationService) -> None:
        self.service = service

    @staticmethod
    def descriptors() -> list[dict[str, Any]]:
        return [dict(item) for item in OPERATION_DESCRIPTORS]

    @staticmethod
    def guidance() -> str:
        return GUIDANCE

    def annotation_list_definitions(self, *, trace_id: str | None = None, domain: str | None = None, trace_schema: str = "synth.trace.v5") -> dict[str, Any]:
        definitions = self.service.list_definitions(trace_schema=trace_schema, domain=domain)
        rubrics = [
            {
                "rubric_id": rubric.rubric_id,
                "version": rubric.version,
                "digest": rubric.content_digest,
                "task_family": rubric.task_family,
                "criteria": [criterion.criterion_id for criterion in rubric.criteria],
            }
            for rubric in self.service.registry.rubrics()
        ]
        return {"trace_id": trace_id, "annotators": definitions, "rubrics": rubrics, "guidance": GUIDANCE}

    def annotation_estimate(self, *, request: dict[str, Any]) -> dict[str, Any]:
        return self.service.estimate(_request_from(request)).to_dict()

    def annotation_start(self, *, request: dict[str, Any], reservation_id: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        """Enqueue only. Execution belongs to a worker; callers poll ``annotation_get``."""

        built = _request_from(request)
        job = self.service.submit(built, reservation_id=reservation_id, session_id=session_id)
        return self._job_payload(job)

    def verification_start(self, *, request: dict[str, Any], reservation_id: str | None = None, session_id: str | None = None) -> dict[str, Any]:
        payload = dict(request)
        payload["mode"] = AnnotationJobMode.VERIFY.value
        return self.annotation_start(request=payload, reservation_id=reservation_id, session_id=session_id)

    def annotation_get(self, *, job_id: str) -> dict[str, Any] | None:
        job = self.service.get(job_id)
        return self._job_payload(job) if job is not None else None

    def annotation_events(self, *, job_id: str, after: int = 0, limit: int = 1000) -> dict[str, Any] | None:
        job = self.service.get(job_id)
        if job is None:
            return None
        return self.service.events.payload(job, after=int(after or 0), limit=int(limit or 1000))

    verification_get = annotation_get

    def annotation_cancel(self, *, job_id: str) -> dict[str, Any]:
        return self._job_payload(self.service.cancel(job_id))

    def annotation_list(self, *, trace_id: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        allowed = {"annotator_id", "annotation_type", "label", "status", "review_state", "include_superseded", "target_entity_id"}
        clean = {key: value for key, value in (filters or {}).items() if key in allowed}
        found = self.service.annotations(trace_id, **clean)
        head = self.service.evidence_head(trace_id)
        return {
            "trace_id": trace_id,
            "bundle_digest": head.content_digest if head else None,
            "count": len(found),
            "annotations": [jsonable(item) for item in found],
            "note": "labels and scores are diagnostic annotations, not environment reward",
        }

    def annotation_get_evidence(self, *, annotation_id: str) -> dict[str, Any] | None:
        return self.service.annotation_evidence(annotation_id)

    def annotation_review(self, *, annotation_id: str, decision: str, reviewer: str, rationale: str = "", evidence: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        revised = self.service.review(annotation_id, decision=decision, reviewer=reviewer, rationale=rationale, evidence=tuple(evidence or ()))
        return {"annotation": jsonable(revised), "supersedes_id": revised.supersedes_id}

    def annotation_consensus(self, *, trace_id: str, annotator_id: str, majority_threshold: float = 0.5) -> dict[str, Any]:
        report = self.service.agreement(trace_id, annotator_id)
        derived = self.service.consensus(trace_id, annotator_id, majority_threshold=majority_threshold)
        return {"agreement": report, "consensus_annotation_ids": [item.annotation_id for item in derived]}

    def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Name-based dispatch for shims; errors are returned as typed payloads."""

        handler = getattr(self, name, None)
        if handler is None or name not in {item["name"] for item in OPERATION_DESCRIPTORS}:
            return {"ok": False, "error": {"code": "unknown_operation", "message": name}}
        try:
            result = handler(**(arguments or {}))
        except AnnotationServiceError as error:
            return {"ok": False, "error": error.as_error().to_dict()}
        except (TypeError, ValueError, KeyError) as error:
            return {"ok": False, "error": {"code": "invalid_arguments", "message": str(error)}}
        return {"ok": True, "result": result}

    def _job_payload(self, job: Any) -> dict[str, Any]:
        receipts = self.service.store.receipts(job.job_id)
        return {
            "job": jsonable(job),
            "terminal": job.terminal,
            "accepted": not job.terminal,
            "poll": None if job.terminal else "annotation_get",
            "stream": annotation_stream_descriptor(job.job_id),
            "receipts": [jsonable(item) for item in receipts],
            "cached": job.cached_from_job_id is not None or any(item.status == "cached" for item in receipts),
        }

    def catalog(self) -> dict[str, Any]:
        return annotation_api_catalog(operations=self.descriptors(), guidance=GUIDANCE)


__all__ = ["GUIDANCE", "OPERATION_DESCRIPTORS", "AnnotationOperations"]
