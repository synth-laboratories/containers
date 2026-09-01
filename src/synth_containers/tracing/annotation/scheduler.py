"""Throughput-bounded scheduling of prepared annotation jobs.

Jobs are queued FIFO (campaign order preserved) and dispatched to a thread
pool subject to three limits that are enforced together:

* a global cap on running jobs;
* a per-runner-class cap (deterministic work is CPU-bound and cheap; Codex
  app-server tasks are subprocesses bounded by the provider; model-API calls
  are bounded by the provider's rate limit);
* an in-flight paid cap in USD micros (the sum of the reservation caps of
  running paid jobs), so a burst of paid jobs can never exceed what the host
  approved at once.

Claiming a job is still the service's atomic ``prepared → running`` transition,
so several schedulers on one store never double-run a job. Reconciliation retry
and crash recovery run on start and periodically, as in ``AnnotationWorker``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .broker import usd_to_micros
from .definitions import RunnerKind
from .jobs import AnnotationJobState, AnnotationJobV1
from .service import AnnotationService, AnnotationServiceError


@dataclass(frozen=True, slots=True)
class ThroughputLimits:
    max_concurrent_total: int = 8
    per_class: dict[str, int] = field(
        default_factory=lambda: {
            RunnerKind.DETERMINISTIC.value: 4,
            RunnerKind.MODEL_API.value: 8,
            RunnerKind.CODEX_APP_SERVER.value: 2,
        }
    )
    max_inflight_paid_usd_micros: int | None = None
    reconcile_retry_seconds: float = 30.0
    poll_seconds: float = 0.25

    def class_limit(self, runner_class: str) -> int:
        return max(1, int(self.per_class.get(runner_class, 1)))


@dataclass
class SchedulerStats:
    queued: int = 0
    running: int = 0
    running_by_class: dict[str, int] = field(default_factory=dict)
    inflight_paid_usd_micros: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    peak_running: int = 0
    peak_by_class: dict[str, int] = field(default_factory=dict)


class AnnotationScheduler:
    def __init__(
        self,
        service: AnnotationService,
        *,
        limits: ThroughputLimits | None = None,
        on_terminal: Callable[[AnnotationJobV1], None] | None = None,
        on_error: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        self.service = service
        self.limits = limits or ThroughputLimits()
        self.on_terminal = on_terminal
        self.on_error = on_error
        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        # Runner class and paid cap never change for a job, so they are read once
        # at enqueue; a blocked class then costs the dispatcher no store reads.
        self._queued_meta: dict[str, tuple[str, int]] = {}  # job_id -> (runner class, cap micros)
        self._running: dict[str, str] = {}  # job_id -> runner class
        self._paid_inflight: dict[str, int] = {}  # job_id -> cap micros
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._threads: set[threading.Thread] = set()
        self.stats = SchedulerStats()
        self._last_retry = 0.0

    # -- queue ----------------------------------------------------------------------

    def runner_class(self, job: AnnotationJobV1) -> str:
        entry = self.service.registry.get(job.request.annotator_id)
        return str(entry.program.runner_kind) if entry else RunnerKind.DETERMINISTIC.value

    def enqueue(self, job_id: str) -> int:
        """Queue a prepared job; returns its position. Terminal or unknown jobs are ignored."""

        job = self.service.get(job_id)
        if job is None or job.terminal or str(job.state) != AnnotationJobState.PREPARED:
            return -1
        runner_class = self.runner_class(job)
        with self._lock:
            if job_id in self._queued or job_id in self._running:
                return list(self._queue).index(job_id) if job_id in self._queued else 0
            self._queue.append(job_id)
            self._queued.add(job_id)
            self._queued_meta[job_id] = (runner_class, self._paid_cap(job))
            self.stats.queued = len(self._queue)
            self._lock.notify_all()
            return len(self._queue)

    def enqueue_prepared(self) -> int:
        """Pick up every prepared job in the store (startup, or jobs submitted by other processes)."""

        count = 0
        for job in self.service.store.list_jobs(state=AnnotationJobState.PREPARED.value):
            if self.enqueue(job.job_id) > 0:
                count += 1
        return count

    # -- limits ---------------------------------------------------------------------

    def _paid_cap(self, job: AnnotationJobV1) -> int:
        if not job.reservation_id:
            return 0
        cap = job.request.limits.max_cost_usd
        return usd_to_micros(cap) if cap is not None else 0

    def _slots_free(self, runner_class: str, cap: int) -> bool:
        if len(self._running) >= self.limits.max_concurrent_total:
            return False
        if sum(1 for cls in self._running.values() if cls == runner_class) >= self.limits.class_limit(runner_class):
            return False
        limit = self.limits.max_inflight_paid_usd_micros
        if cap and limit is not None and sum(self._paid_inflight.values()) + cap > limit:
            return False
        return True

    def _can_start(self, job: AnnotationJobV1, runner_class: str) -> bool:
        return self._slots_free(runner_class, self._paid_cap(job))

    def _drop(self, job_id: str) -> None:
        self._queue.remove(job_id)
        self._queued.discard(job_id)
        self._queued_meta.pop(job_id, None)

    def _next_startable(self) -> tuple[str, AnnotationJobV1, str] | None:
        """First queued job whose class/paid slots are free; later jobs may overtake a blocked class.

        Capacity is decided from the class and cap cached at enqueue; only a job
        that fits is re-read from the store to confirm it is still prepared.
        """

        if len(self._running) >= self.limits.max_concurrent_total:
            return None
        for job_id in list(self._queue):
            meta = self._queued_meta.get(job_id)
            if meta is None:
                job = self.service.get(job_id)
                if job is None:
                    self._drop(job_id)
                    self.stats.skipped += 1
                    continue
                meta = (self.runner_class(job), self._paid_cap(job))
                self._queued_meta[job_id] = meta
            runner_class, cap = meta
            if not self._slots_free(runner_class, cap):
                continue
            job = self.service.get(job_id)
            if job is None or job.terminal or str(job.state) != AnnotationJobState.PREPARED:
                self._drop(job_id)
                self.stats.skipped += 1
                continue
            return job_id, job, runner_class
        return None

    # -- execution ------------------------------------------------------------------

    def _start(self, job_id: str, job: AnnotationJobV1, runner_class: str) -> None:
        self._drop(job_id)
        self._running[job_id] = runner_class
        cap = self._paid_cap(job)
        if cap:
            self._paid_inflight[job_id] = cap
        self.stats.queued = len(self._queue)
        self.stats.running = len(self._running)
        self.stats.running_by_class[runner_class] = self.stats.running_by_class.get(runner_class, 0) + 1
        self.stats.inflight_paid_usd_micros = sum(self._paid_inflight.values())
        self.stats.peak_running = max(self.stats.peak_running, self.stats.running)
        self.stats.peak_by_class[runner_class] = max(self.stats.peak_by_class.get(runner_class, 0), self.stats.running_by_class[runner_class])
        thread = threading.Thread(target=self._execute, args=(job_id, runner_class), name=f"annotation-{runner_class}-{job_id[-8:]}", daemon=True)
        self._threads.add(thread)
        thread.start()

    def _execute(self, job_id: str, runner_class: str) -> None:
        result: AnnotationJobV1 | None = None
        try:
            result = self.service.run(job_id)
        except AnnotationServiceError as error:
            if error.code != "revision_conflict" and self.on_error:
                self.on_error(job_id, error)
        except BaseException as error:  # noqa: BLE001 - the service fails the job closed; keep the pool alive
            if self.on_error:
                self.on_error(job_id, error)
        finally:
            with self._lock:
                self._running.pop(job_id, None)
                self._paid_inflight.pop(job_id, None)
                self.stats.running = len(self._running)
                self.stats.running_by_class[runner_class] = max(0, self.stats.running_by_class.get(runner_class, 0) - 1)
                self.stats.inflight_paid_usd_micros = sum(self._paid_inflight.values())
                if result is not None and result.terminal:
                    if str(result.state) in {AnnotationJobState.SEALED, AnnotationJobState.ABSTAINED}:
                        self.stats.completed += 1
                    else:
                        self.stats.failed += 1
                self._threads.discard(threading.current_thread())
                self._lock.notify_all()
        if result is not None and self.on_terminal:
            try:
                self.on_terminal(result)
            except Exception as error:  # noqa: BLE001
                if self.on_error:
                    self.on_error(job_id, error)

    def dispatch_once(self) -> int:
        """Start every startable queued job right now (non-blocking); returns how many started."""

        started = 0
        with self._lock:
            while True:
                candidate = self._next_startable()
                if candidate is None:
                    break
                self._start(*candidate)
                started += 1
        return started

    def _loop(self) -> None:
        self.service.recover_interrupted()
        self.enqueue_prepared()
        while not self._stop.is_set():
            self.dispatch_once()
            now = time.monotonic()
            if now - self._last_retry >= self.limits.reconcile_retry_seconds:
                self._last_retry = now
                try:
                    self.service.retry_reconciliations()
                except Exception as error:  # noqa: BLE001
                    if self.on_error:
                        self.on_error("reconciliation", error)
            with self._lock:
                self._lock.wait(self.limits.poll_seconds)

    def start(self) -> "AnnotationScheduler":
        if self._dispatcher is None:
            self._stop.clear()
            self._dispatcher = threading.Thread(target=self._loop, name="annotation-scheduler", daemon=True)
            self._dispatcher.start()
        return self

    def stop(self, *, timeout: float = 60.0) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=timeout)
            self._dispatcher = None
        deadline = time.monotonic() + timeout
        for thread in list(self._threads):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def drain(self, *, timeout: float) -> bool:
        """Run in the calling thread until the queue and pool are empty or the timeout passes."""

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.dispatch_once()
            with self._lock:
                if not self._queue and not self._running:
                    return True
                self._lock.wait(min(self.limits.poll_seconds, max(0.0, deadline - time.monotonic())))
        return False

    def wait_for(self, job_ids: list[str], *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all((job := self.service.get(job_id)) is not None and job.terminal for job_id in job_ids):
                return True
            if self._dispatcher is None:
                # No background dispatcher: the waiting thread drives the pool itself.
                self.dispatch_once()
            time.sleep(min(0.05, self.limits.poll_seconds))
        return False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued": len(self._queue),
                "running": len(self._running),
                "running_by_class": dict(self.stats.running_by_class),
                "inflight_paid_usd_micros": sum(self._paid_inflight.values()),
                "completed": self.stats.completed,
                "failed": self.stats.failed,
                "skipped": self.stats.skipped,
                "peak_running": self.stats.peak_running,
                "peak_by_class": dict(self.stats.peak_by_class),
                "limits": {"max_concurrent_total": self.limits.max_concurrent_total, "per_class": dict(self.limits.per_class), "max_inflight_paid_usd_micros": self.limits.max_inflight_paid_usd_micros},
            }


__all__ = ["AnnotationScheduler", "SchedulerStats", "ThroughputLimits"]
