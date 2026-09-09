"""Live event log for one annotation (or verification) job.

Persisted beside the job as ``events.jsonl``. Clients poll
``GET /annotation-jobs/{id}/events`` or subscribe to SSE. Payloads name job
state, runner, model, inspection tools, and sealed finding ids — not hidden
chain-of-thought or execution-trace message text.
"""

from __future__ import annotations

import threading
from typing import Any

from synth_containers.event_log import CONTROL_SUBSCRIBED, RolloutEventLog, poll_payload

from .endpoints import ANNOTATION_EVENT_KINDS, annotation_stream_descriptor
from .jobs import AnnotationJobState, AnnotationJobV1
from .persistence import AnnotationStore
from .tools import ToolCallRecordV1


_TERMINAL_KIND = {
    AnnotationJobState.SEALED.value: "annotation.sealed",
    AnnotationJobState.ABSTAINED.value: "annotation.abstained",
    AnnotationJobState.FAILED.value: "annotation.failed",
    AnnotationJobState.CANCELLED.value: "annotation.cancelled",
}


class AnnotationEventStreamer:
    def __init__(self, store: AnnotationStore) -> None:
        self.store = store
        self._logs: dict[str, RolloutEventLog] = {}
        self._lock = threading.RLock()

    def log_for(self, job_id: str) -> RolloutEventLog:
        with self._lock:
            existing = self._logs.get(job_id)
            if existing is not None:
                return existing
            path = self.store.job_dir(job_id) / "events.jsonl"
            stream_id = f"annotation:{job_id}"
            if path.exists():
                log = RolloutEventLog.recover(rollout_id=job_id, stream_id=stream_id, journal_path=path)
            else:
                log = RolloutEventLog(rollout_id=job_id, stream_id=stream_id, journal_path=path)
            self._logs[job_id] = log
            return log

    def ensure_open(self, job_id: str) -> RolloutEventLog:
        log = self.log_for(job_id)
        if log.closed:
            return log
        if any(item.control and item.kind == CONTROL_SUBSCRIBED for item in log.after(0)):
            return log
        log.append_control(
            CONTROL_SUBSCRIBED,
            {
                "type": CONTROL_SUBSCRIBED,
                "stream.id": log.stream_id,
                "job_id": job_id,
                "kind": "annotation",
                "next_sequence": log.high_water + 1,
                "ready": True,
            },
        )
        return log

    def prepared(self, job: AnnotationJobV1) -> None:
        self._emit(job, "annotation.prepared")

    def running(self, job: AnnotationJobV1) -> None:
        self._emit(job, "annotation.running")

    def tool(self, job: AnnotationJobV1, record: ToolCallRecordV1) -> None:
        self._emit(
            job,
            "annotation.tool",
            extra={
                "tool": record.tool,
                "index": record.index,
                "ok": record.ok,
                "arguments": dict(record.arguments),
                "response_bytes": record.response_bytes,
                "truncated": record.truncated,
                "error": record.error,
            },
        )

    def validating(self, job: AnnotationJobV1) -> None:
        self._emit(job, "annotation.validating")

    def finding(self, job: AnnotationJobV1, annotation: Any) -> None:
        target = annotation.target.to_dict() if hasattr(annotation.target, "to_dict") else dict(annotation.target)
        evidence = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in (annotation.evidence or ())
        ]
        self._emit(
            job,
            "annotation.finding",
            extra={
                "annotation_id": annotation.annotation_id,
                "annotation_type": annotation.annotation_type,
                "labels": list(annotation.labels),
                "status": str(annotation.status) if annotation.status else None,
                "target": target,
                "evidence": evidence,
                "abstention_reason": annotation.abstention_reason,
            },
        )

    def terminal(self, job: AnnotationJobV1) -> None:
        kind = _TERMINAL_KIND.get(str(job.state), "annotation.failed")
        extra: dict[str, Any] = {
            "annotation_ids": list(job.annotation_ids),
            "bundle_digest": job.bundle_digest,
            "applied_count": job.applied_count,
            "abstained_count": job.abstained_count,
            "rejected_count": job.rejected_count,
        }
        if job.error is not None:
            extra["error"] = job.error.to_dict() if hasattr(job.error, "to_dict") else {"message": str(job.error)}
        self._emit(job, kind, extra=extra, close=True)

    def hydrate(self, job: AnnotationJobV1) -> RolloutEventLog:
        """Open a log for reads. Reconstruct one terminal snapshot for pre-stream jobs."""

        log = self.log_for(job.job_id)
        if log.high_water > 0 or log.closed:
            return log
        if job.terminal:
            self._emit(job, _TERMINAL_KIND.get(str(job.state), "annotation.failed"), close=True)
            return self.log_for(job.job_id)
        self.ensure_open(job.job_id)
        return log

    def payload(self, job: AnnotationJobV1, *, after: int = 0, limit: int = 1000) -> dict[str, Any]:
        log = self.hydrate(job)
        page = poll_payload(
            log,
            after=after,
            limit=limit,
            subject_id_field="job_id",
            subject_id=job.job_id,
        )
        page["stream"] = annotation_stream_descriptor(job.job_id)
        page["event_kinds"] = list(ANNOTATION_EVENT_KINDS)
        page["terminal"] = job.terminal
        return page

    def _emit(
        self,
        job: AnnotationJobV1,
        kind: str,
        *,
        extra: dict[str, Any] | None = None,
        close: bool = False,
    ) -> None:
        with self._lock:
            log = self.ensure_open(job.job_id)
            if log.closed:
                return
            payload: dict[str, Any] = {
                "job_id": job.job_id,
                "state": str(job.state),
                "annotator_id": job.request.annotator_id,
                "mode": str(job.request.mode),
                "runner_kind": job.request.runner_kind,
                "model": job.request.model,
                "reasoning_effort": job.request.reasoning_effort,
                "source_trace_id": job.request.source_trace_id,
                "source_trace_digest": job.request.source_trace_digest,
            }
            if extra:
                payload.update(extra)
            log.append(kind, payload)
            if close:
                log.mark_closed()
