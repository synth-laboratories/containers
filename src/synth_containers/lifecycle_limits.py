"""Durable phase admission and stop custody for a single execution owner.

Deadlines survive owner restart; stopping the substrate remains an explicit
adapter operation. asyncio cancellation cannot fence a remote provider by itself.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .operator_journal import OperatorJournal, read_operator_events


@dataclass(frozen=True)
class LifecycleLimits:
    overall_seconds: float
    setup_seconds: float = 300
    work_seconds: float = 300
    verifier_seconds: float = 120
    publication_seconds: float = 120
    cleanup_seconds: float = 90

    def __post_init__(self):
        for name, value in asdict(self).items():
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value <= 86400
            ):
                raise ValueError(f"{name} must be finite positive seconds at most 86400")
        if self.cleanup_seconds >= self.overall_seconds:
            raise ValueError("Overall lifetime must reserve time for cleanup")


class RolloutStopped(RuntimeError):
    pass


class DurableRolloutSupervisor:
    """Exclusive owner; reopen with identical limits to recover, never replenish.

    Cleanup has a reserved window inside the overall deadline. A worker returning
    after expiry still gets one bounded emergency cleanup call, explicitly marked
    overdue. This is not a watchdog surviving worker loss.
    """

    def __init__(self, output: Path, run_id: str, limits: LifecycleLimits):
        if not isinstance(run_id, str) or not 1 <= len(run_id) <= 255:
            raise ValueError("Invalid run identity")
        self.output, self.run_id, self.limits = output, run_id, limits
        output.mkdir(parents=True, exist_ok=True)
        self._lock = (output / "limit-owner.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claim_path = output / "limit-claim.json"
            if claim_path.exists():
                raw = claim_path.read_bytes()
                if len(raw) > 16384:
                    raise ValueError("Limit claim exceeds bound")
                claim = json.loads(raw)
                if claim["run_id"] != run_id or claim["limits"] != asdict(limits):
                    raise ValueError("Recovered execution cannot reset identity or allowances")
            else:
                now = datetime.now(UTC)
                claim = {
                    "run_id": run_id,
                    "limits": asdict(limits),
                    "created_at": now.isoformat(),
                    "deadline": (now + timedelta(seconds=limits.overall_seconds)).isoformat(),
                }
                with claim_path.open("x") as handle:
                    json.dump(claim, handle, allow_nan=False, sort_keys=True)
                    handle.flush()
                    os.fsync(handle.fileno())
                descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            self.deadline = datetime.fromisoformat(claim["deadline"])
            created = datetime.fromisoformat(claim["created_at"])
            if (
                self.deadline.tzinfo is None
                or created.tzinfo is None
                or abs((self.deadline - created).total_seconds() - limits.overall_seconds)
                > 0.000001
            ):
                raise ValueError("Invalid persisted deadline")
            # Wall time supports recovery; monotonic time prevents allowance growth
            # if the system clock moves backwards while this owner is running.
            self._loop = asyncio.get_running_loop()
            self._monotonic_deadline = self._loop.time() + self.remaining_seconds()
            self.journal = OperatorJournal(output / "limit-events.jsonl", run_id=run_id)
            self.causes: list[str] = []
            self._phase_deadlines: dict[str, datetime] = {}
            self._completed: set[str] = set()
            if self.journal.path.exists():
                page = read_operator_events(self.journal.path, run_id=run_id, limit=1000)
                if page["has_more"]:
                    raise ValueError("Limit history exceeds recovery bound")
                for row in page["events"]:
                    if row["event"] == "limit.stop_decided":
                        self.causes.append(row["cause"])
                    elif row["event"] == "limit.phase_started":
                        self._phase_deadlines[row["phase"]] = datetime.fromisoformat(
                            row["deadline"]
                        )
                    elif row["event"] == "limit.phase_completed":
                        self._completed.add(row["phase"])
            self.receipt_errors: list[str] = []
            self._stop_lock = asyncio.Lock()
        except BaseException:
            self._lock.close()
            raise

    def remaining_seconds(self) -> float:
        wall = (self.deadline - datetime.now(UTC)).total_seconds()
        monotonic = getattr(self, "_monotonic_deadline", None)
        return wall if monotonic is None else min(wall, monotonic - self._loop.time())

    def event(self, event: str, **fields):
        return self.journal.append(
            lambda seq: {
                "schema_version": "synth.rollout-limit-event.v1",
                "run_id": self.run_id,
                "seq": seq,
                "event": event,
                "occurred_at": datetime.now(UTC).isoformat(),
                **fields,
            }
        )

    def decide_stop(self, cause: str):
        if cause not in self.causes:
            self.causes.append(cause)
            try:
                self.event(
                    "limit.stop_decided",
                    cause=cause,
                    primary_cause=self.causes[0],
                    remaining_seconds=self.remaining_seconds(),
                    enforcement="coordinator_deadline",
                    termination_confirmed=False,
                )
            except Exception as error:  # noqa: BLE001 - stopping must survive any custody failure
                # Persistence failures must not disable safety actuators.
                self.receipt_errors.append(type(error).__name__)

    async def run_phase(
        self,
        phase: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        preserve_evidence: bool = False,
    ):
        if phase not in {"setup", "work", "verifier", "publication", "cleanup"}:
            raise ValueError("Unknown execution phase")
        if phase in self._completed:
            raise RolloutStopped("Completed phase cannot be repeated under the same identity")
        if preserve_evidence and phase != "publication":
            raise ValueError("Preservation exception applies only to publication")
        if self.causes and phase != "cleanup" and not preserve_evidence:
            raise RolloutStopped(self.causes[0])
        allowance = getattr(self.limits, phase + "_seconds")
        remaining = self.remaining_seconds()
        if phase != "cleanup":
            remaining -= self.limits.cleanup_seconds
        seconds = min(allowance, remaining)
        if phase in self._phase_deadlines:
            seconds = min(
                seconds, (self._phase_deadlines[phase] - datetime.now(UTC)).total_seconds()
            )
        if seconds <= 0 and phase != "cleanup":
            self.decide_stop("overall_deadline")
            raise RolloutStopped("Execution deadline exhausted")
        if phase == "cleanup" and seconds <= 0:
            seconds = allowance
            self.event("limit.cleanup_overdue", original_deadline=self.deadline.isoformat())
        if phase not in self._phase_deadlines:
            deadline = datetime.now(UTC) + timedelta(seconds=seconds)
            self._phase_deadlines[phase] = deadline
            self.event("limit.phase_started", phase=phase, deadline=deadline.isoformat())
        try:
            async with asyncio.timeout(seconds):
                result = await operation()
        except TimeoutError:
            self.decide_stop(phase + "_deadline")
            raise
        self.event("limit.phase_completed", phase=phase)
        self._completed.add(phase)
        return result

    async def stop(self, cause: str, terminate: Callable[[], Awaitable[dict]]):
        """Terminate must confirm absence or raise; returns adapter custody receipt."""
        async with self._stop_lock:
            self.decide_stop(cause)
            try:
                async with asyncio.timeout(self.limits.cleanup_seconds):
                    receipt = await terminate()
                if not isinstance(receipt, dict) or receipt.get("cleanup_status") != "confirmed":
                    raise RolloutStopped("Termination did not return confirmed cleanup custody")
            except BaseException as error:
                try:
                    self.event(
                        "limit.cleanup_blocked",
                        error_type=type(error).__name__,
                        receipt_gaps=self.receipt_errors,
                    )
                except Exception as receipt_error:  # noqa: BLE001 - preserve stop failure
                    self.receipt_errors.append(type(receipt_error).__name__)
                raise
            self.event(
                "limit.termination_confirmed",
                receipt=receipt,
                receipt_gaps=self.receipt_errors,
                settlement_status="requires_budget_authority",
            )
            return receipt

    def close(self):
        self._lock.close()
