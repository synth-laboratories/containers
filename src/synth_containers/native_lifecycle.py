"""Shared native coordinator with bounded phase callbacks and cleanup custody."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lifecycle_limits import DurableRolloutSupervisor, LifecycleLimits


@dataclass
class NativeLifecycleOperations:
    setup: Callable[[], Awaitable[Any]]
    work: Callable[[], Awaitable[Any]]
    verifier: Callable[[], Awaitable[Any]]
    publication: Callable[[], Awaitable[Any]]
    cleanup: Callable[[], Awaitable[dict]]


async def execute_native_lifecycle(
    output: Path, *, run_id: str, limits: LifecycleLimits, operations: NativeLifecycleOperations
) -> dict:
    """Callbacks share caller state; cleanup must return independent absence proof.

    Persist evidence before releasing its last copy. Interrupted phases are not
    replayed: recovery owns reconciliation, never duplicate agent execution.
    A remote provider needs its own TTL/fencing in addition to these deadlines.
    """
    supervisor = DurableRolloutSupervisor(output, run_id, limits)
    results: dict[str, Any] = {}
    error: BaseException | None = None
    try:
        if supervisor._phase_deadlines:
            supervisor.decide_stop("recovery_requires_reconciliation")
        try:
            for phase in ("setup", "work", "verifier", "publication"):
                results[phase] = await supervisor.run_phase(phase, getattr(operations, phase))
        except BaseException as caught:  # noqa: BLE001 - cleanup must follow cancellation too
            error = caught
            supervisor.decide_stop(type(caught).__name__)
        try:
            # stop() calls the independent provider deletion/absence actuator
            # even after failures in the journal or phase cancellation.
            results["cleanup"] = await supervisor.stop(
                "execution_failed" if error else "execution_complete", operations.cleanup
            )
        except BaseException as cleanup_error:
            if error is None:
                raise
            error.add_note("Cleanup remains unconfirmed: " + type(cleanup_error).__name__)
        if error is not None:
            raise error
        return results
    finally:
        supervisor.close()
