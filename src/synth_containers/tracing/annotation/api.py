"""HTTP surface for annotation jobs and sealed annotation evidence.

Deliberately separate from ``GET /rollouts/{rollout_id}/annotations``, which
serves lightweight runtime rollout annotations.

Writes are asynchronous: ``POST .../annotation-jobs`` enqueues and answers
``202 Accepted``; an ``AnnotationWorker`` runs the job; clients poll
``GET /annotation-jobs/{job_id}``. Nothing on the request path executes an
app-server task, and no authorization object is ever read from a body — only an
opaque ``reservation_id`` the host broker issued.

Mount with ``app.include_router(build_annotation_router(service))`` and start a
worker (``create_annotation_app`` does both).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException

from .operations import AnnotationOperations
from .service import AnnotationService, AnnotationServiceError


def build_annotation_router(service: AnnotationService, *, prefix: str = "", scheduler: Any = None) -> APIRouter:
    operations = AnnotationOperations(service)
    router = APIRouter(prefix=prefix, tags=["trace-annotation"])

    def enqueue(payload: dict[str, Any]) -> dict[str, Any]:
        if scheduler is not None and payload.get("accepted"):
            payload["queue_position"] = scheduler.enqueue(payload["job"]["job_id"])
        return payload

    @router.get("/annotation/scheduler")
    async def scheduler_status() -> dict[str, Any]:
        return scheduler.snapshot() if scheduler is not None else {"scheduler": None}

    def guard(callable_: Any, **kwargs: Any) -> Any:
        try:
            return callable_(**kwargs)
        except AnnotationServiceError as error:
            status = 402 if error.code == "reservation_required" else 403 if error.code == "reservation_rejected" else 409 if error.code == "revision_conflict" else 400
            raise HTTPException(status_code=status, detail=error.as_error().to_dict()) from error

    @router.get("/annotation/operations")
    async def list_operations() -> dict[str, Any]:
        return {"operations": operations.descriptors(), "guidance": operations.guidance()}

    @router.get("/traces/{trace_id}/annotation-definitions")
    async def annotation_definitions(trace_id: str, domain: str | None = None) -> dict[str, Any]:
        return operations.annotation_list_definitions(trace_id=trace_id, domain=domain)

    @router.post("/traces/{trace_id}/annotation-estimates")
    async def annotation_estimate(trace_id: str, body: dict[str, Any]) -> dict[str, Any]:
        request = dict(body.get("request") or body)
        request.setdefault("source_trace_id", trace_id)
        return guard(operations.annotation_estimate, request=request)

    @router.post("/traces/{trace_id}/annotation-jobs", status_code=202)
    async def start_annotation(trace_id: str, body: dict[str, Any]) -> dict[str, Any]:
        request = dict(body.get("request") or {})
        request.setdefault("source_trace_id", trace_id)
        return enqueue(
            guard(
                operations.annotation_start,
                request=request,
                reservation_id=_opaque(body.get("reservation_id")),
                session_id=_opaque(body.get("session_id")),
            )
        )

    @router.get("/annotation-jobs/{job_id}")
    async def get_annotation_job(job_id: str) -> dict[str, Any]:
        payload = operations.annotation_get(job_id=job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "job_id": job_id})
        return payload

    @router.post("/annotation-jobs/{job_id}/cancel")
    async def cancel_annotation_job(job_id: str) -> dict[str, Any]:
        return guard(operations.annotation_cancel, job_id=job_id)

    @router.get("/traces/{trace_id}/evidence-bundles")
    async def evidence_bundles(trace_id: str) -> dict[str, Any]:
        return {"trace_id": trace_id, "bundles": list(service.evidence_bundles(trace_id))}

    @router.get("/traces/{trace_id}/annotations")
    async def list_annotations(
        trace_id: str,
        annotator_id: str | None = None,
        annotation_type: str | None = None,
        label: str | None = None,
        status: str | None = None,
        review_state: str | None = None,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        filters = {
            "annotator_id": annotator_id,
            "annotation_type": annotation_type,
            "label": label,
            "status": status,
            "review_state": review_state,
            "include_superseded": include_superseded,
        }
        return operations.annotation_list(trace_id=trace_id, filters={k: v for k, v in filters.items() if v is not None})

    @router.get("/annotations/{annotation_id}")
    async def get_annotation(annotation_id: str) -> dict[str, Any]:
        payload = operations.annotation_get_evidence(annotation_id=annotation_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"code": "annotation_not_found", "annotation_id": annotation_id})
        return payload

    @router.post("/traces/{trace_id}/verification-jobs", status_code=202)
    async def start_verification(trace_id: str, body: dict[str, Any]) -> dict[str, Any]:
        request = dict(body.get("request") or {})
        request.setdefault("source_trace_id", trace_id)
        return enqueue(
            guard(
                operations.verification_start,
                request=request,
                reservation_id=_opaque(body.get("reservation_id")),
                session_id=_opaque(body.get("session_id")),
            )
        )

    @router.get("/verification-jobs/{job_id}")
    async def get_verification_job(job_id: str) -> dict[str, Any]:
        payload = operations.verification_get(job_id=job_id)
        if payload is None:
            raise HTTPException(status_code=404, detail={"code": "job_not_found", "job_id": job_id})
        return payload

    @router.post("/annotations/{annotation_id}/reviews")
    async def review_annotation(annotation_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return guard(
            operations.annotation_review,
            annotation_id=annotation_id,
            decision=str(body.get("decision") or ""),
            reviewer=str(body.get("reviewer") or "unknown"),
            rationale=str(body.get("rationale") or ""),
            evidence=list(body.get("evidence") or ()),
        )

    @router.post("/traces/{trace_id}/annotation-consensus")
    async def consensus(trace_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return guard(
            operations.annotation_consensus,
            trace_id=trace_id,
            annotator_id=str(body.get("annotator_id") or ""),
            majority_threshold=float(body.get("majority_threshold", 0.5)),
        )

    return router


def _opaque(value: Any) -> str | None:
    """Reservation and session ids are opaque strings; anything else is refused."""

    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise HTTPException(status_code=400, detail={"code": "invalid_argument", "message": "ids must be short strings"})
    return value


def create_annotation_app(service: AnnotationService, *, start_worker: bool = True, limits: Any = None) -> FastAPI:
    """A standalone app: router plus a throughput-bounded scheduler owned by the app lifecycle."""

    from .scheduler import AnnotationScheduler, ThroughputLimits

    app = FastAPI(title="synth trace annotation")
    scheduler = AnnotationScheduler(service, limits=limits or ThroughputLimits())
    app.include_router(build_annotation_router(service, scheduler=scheduler))
    app.state.annotation_scheduler = scheduler

    @app.on_event("startup")
    async def _start() -> None:
        if start_worker:
            scheduler.start()  # recovers interrupted jobs and drains the reconciliation outbox first
        else:
            service.recover_interrupted()

    @app.on_event("shutdown")
    async def _stop() -> None:
        scheduler.stop()

    return app


__all__ = ["build_annotation_router", "create_annotation_app"]
