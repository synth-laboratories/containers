"""Durable receipts for annotation jobs.

A receipt records what ran, under which sealed inputs, what it consumed, and what
it produced. It never carries credentials, prompts, or hidden reasoning.
"""

from __future__ import annotations

from typing import Any

from ..canonical import record_id, utc_now
from ..models.standards import ProducerRefV1, ReceiptV1
from .jobs import AnnotationJobUsageV1, AnnotationJobV1


def job_receipt(
    job: AnnotationJobV1,
    *,
    operation: str,
    status: str,
    started_at: str,
    ended_at: str | None = None,
    producer: ProducerRefV1 | None = None,
    usage: AnnotationJobUsageV1 | None = None,
    previous_state: str | None = None,
    new_state: str | None = None,
    errors: tuple[str, ...] = (),
    detail: dict[str, Any] | None = None,
    output_digests: tuple[str, ...] = (),
) -> ReceiptV1:
    ended = ended_at or utc_now()
    inputs = tuple(
        digest
        for digest in (
            job.request.source_trace_digest,
            job.request.annotator_digest,
            job.request.rubric_digest,
            job.program_digest,
            job.tool_contract_digest,
            job.workspace_manifest_digest,
        )
        if digest
    )
    usage_payload = (usage or job.usage).to_dict()
    receipt_detail = {
        "job_id": job.job_id,
        "idempotency_key": job.idempotency_key,
        "source_trace_id": job.request.source_trace_id,
        "annotator_id": job.request.annotator_id,
        "annotator_digest": job.request.annotator_digest,
        "rubric_id": job.request.rubric_id,
        "rubric_digest": job.request.rubric_digest,
        "program_digest": job.program_digest,
        "tool_contract_digest": job.tool_contract_digest,
        "model": job.request.model,
        "reasoning_effort": job.request.reasoning_effort,
        "mode": str(job.request.mode),
        "repeat_index": job.request.repeat_index,
        "parent_job_id": job.request.parent_job_id,
        "runner_version": job.request.runner_version,
        "limits": job.request.limits.to_dict(),
        "usage": usage_payload,
        "cached_from_job_id": job.cached_from_job_id,
        "reservation_id": job.reservation_id,
        "execution_trace_id": job.execution_trace_id,
        "execution_trace_digest": job.execution_trace_digest,
        **(detail or {}),
    }
    return ReceiptV1(
        receipt_id=record_id(
            "rcpt",
            kind="annotation_job_receipt",
            scope=(job.job_id,),
            key={"operation": operation, "status": status, "started_at": started_at, "ended_at": ended},
        ),
        operation=operation,
        status=status,
        started_at=started_at,
        ended_at=ended,
        target_ids=(job.request.source_trace_id, job.job_id),
        producer=producer,
        wall_time_seconds=(usage or job.usage).wall_time_seconds,
        input_digests=inputs,
        output_digests=output_digests,
        previous_state=previous_state,
        new_state=new_state,
        completeness="complete" if status in {"sealed", "abstained", "cached"} else "partial",
        errors=errors,
        next_safe_action=_next_safe_action(status),
        detail=receipt_detail,
    ).sealed()


def _next_safe_action(status: str) -> str | None:
    if status in {"sealed", "abstained", "cached"}:
        return "reuse_sealed_result"
    if status == "cancelled":
        return "resubmit_with_new_repeat_index"
    if status == "failed":
        return "inspect_error_then_resubmit"
    return None


__all__ = ["job_receipt"]
