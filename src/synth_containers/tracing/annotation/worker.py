"""Background execution of prepared annotation jobs.

The HTTP/MCP surface only *enqueues* (``202 Accepted``). A worker claims prepared
jobs — the claim is the ``prepared → running`` transition under the store lock,
so several workers can share one store without running a job twice — and drives
each to a terminal state. Callers poll ``annotation_get``.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .jobs import AnnotationJobState
from .service import AnnotationService, AnnotationServiceError


class AnnotationWorker:
    def __init__(
        self,
        service: AnnotationService,
        *,
        poll_seconds: float = 0.5,
        on_error: Callable[[str, BaseException], None] | None = None,
        reconcile_retry_seconds: float = 30.0,
    ) -> None:
        self.service = service
        self.poll_seconds = poll_seconds
        self.reconcile_retry_seconds = reconcile_retry_seconds
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed = 0

    def run_once(self) -> int:
        """Run every currently prepared job; returns how many reached a terminal state."""

        count = 0
        for job in self.service.store.list_jobs(state=AnnotationJobState.PREPARED.value):
            try:
                result = self.service.run(job.job_id)
            except AnnotationServiceError as error:
                # Someone else claimed it first, or it was cancelled meanwhile.
                if error.code == "revision_conflict":
                    continue
                if self.on_error:
                    self.on_error(job.job_id, error)
                continue
            except BaseException as error:  # noqa: BLE001 - keep the loop alive; job is failed closed by the service
                if self.on_error:
                    self.on_error(job.job_id, error)
                continue
            if result.terminal:
                count += 1
        self.processed += count
        return count

    def _loop(self) -> None:
        last_retry = 0.0
        while not self._stop.is_set():
            ran = self.run_once()
            now = time.monotonic()
            if now - last_retry >= self.reconcile_retry_seconds:
                last_retry = now
                try:
                    self.service.retry_reconciliations()
                except Exception as error:  # noqa: BLE001
                    if self.on_error:
                        self.on_error("reconciliation", error)
            if not ran:
                self._stop.wait(self.poll_seconds)

    def start(self) -> "AnnotationWorker":
        # Startup: resume crashed paid preparations and pending reconciliations first.
        self.service.recover_interrupted()
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="annotation-worker", daemon=True)
            self._thread.start()
        return self

    def stop(self, *, timeout: float = 30.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def wait_for(self, job_id: str, *, timeout: float) -> bool:
        """Test/CLI helper: block until the job is terminal or the timeout passes."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.service.get(job_id)
            if job is not None and job.terminal:
                return True
            time.sleep(min(0.05, self.poll_seconds))
        return False


__all__ = ["AnnotationWorker"]
