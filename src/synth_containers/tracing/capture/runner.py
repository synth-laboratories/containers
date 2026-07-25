"""Run an unchanged child command under a local Trace V5 capture supervisor."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Sequence

from ..canonical import bytes_digest, canonical_bytes, utc_now
from ..models.completeness import TerminationV5, TraceStatus
from .supervisor import CaptureSupervisor, SupervisorConfig


TRACE_RUN_RECEIPT_SCHEMA_VERSION = "synth.trace-run-receipt.v1"


@dataclass(frozen=True, slots=True)
class CapturedCommandResult:
    exit_code: int
    receipt: dict[str, Any]


def run_captured_command(
    config: SupervisorConfig,
    command: Sequence[str],
    *,
    timeout_seconds: float | None = None,
    projections: tuple[str, ...] = ("v4",),
    environ: dict[str, str] | None = None,
) -> CapturedCommandResult:
    """Capture one child process, seal on every terminal path, and preserve its RC."""

    argv = tuple(str(item) for item in command)
    if not argv:
        raise ValueError("captured command must not be empty")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    supervisor = CaptureSupervisor(config)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    child_exit_code = 127
    status: TraceStatus | str = TraceStatus.FAILED
    termination: TerminationV5 | None = None

    startup_ok = False
    try:
        supervisor.start_capture()
        startup_ok = True
    except BaseException as exc:
        child_exit_code = 70
        status = TraceStatus.FAILED
        termination = TerminationV5(
            reason="capture_startup_error",
            exit_code=child_exit_code,
            detail=f"{type(exc).__name__}: {exc}",
        )

    if startup_ok:
        try:
            child_environment = dict(os.environ if environ is None else environ)
            child_environment.update(supervisor.environment())
            process = subprocess.Popen(argv, env=child_environment)
            try:
                child_exit_code = process.wait(timeout=timeout_seconds)
                if child_exit_code == 0:
                    status = TraceStatus.COMPLETED
                elif child_exit_code < 0:
                    status = TraceStatus.INTERRUPTED
                    termination = TerminationV5(
                        reason="child_signal",
                        exit_code=child_exit_code,
                        signal=_signal_name(-child_exit_code),
                    )
                else:
                    status = TraceStatus.FAILED
                    termination = TerminationV5(
                        reason="child_exit",
                        exit_code=child_exit_code,
                    )
            except subprocess.TimeoutExpired:
                _stop_child(process)
                child_exit_code = 124
                status = TraceStatus.INTERRUPTED
                termination = TerminationV5(
                    reason="timeout",
                    exit_code=child_exit_code,
                    detail=f"child exceeded {timeout_seconds} seconds",
                )
            except KeyboardInterrupt:
                _interrupt_child(process)
                child_exit_code = 130
                status = TraceStatus.INTERRUPTED
                termination = TerminationV5(
                    reason="operator_interrupt",
                    exit_code=child_exit_code,
                    signal="SIGINT",
                )
        except OSError as exc:
            child_exit_code = 127
            status = TraceStatus.FAILED
            termination = TerminationV5(
                reason="launch_error",
                exit_code=child_exit_code,
                detail=f"{type(exc).__name__}: {exc}",
            )

    sealed = supervisor.finalize(
        status=status,
        termination=termination,
        child_exit_code=child_exit_code,
    )
    projected = tuple(
        supervisor.materialize_projection(kind)
        for kind in dict.fromkeys(projections)
    )
    elapsed_seconds = time.monotonic() - started_monotonic
    receipt = {
        "schema_version": TRACE_RUN_RECEIPT_SCHEMA_VERSION,
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "bundle": str(Path(config.bundle_root)),
        "trace_id": sealed.document.trace_id,
        "trace_digest": sealed.document.content_digest,
        "capture_id": sealed.document.capture.capture_id,
        "binding_id": supervisor.binding.binding_id,
        "binding_digest": supervisor.binding.content_digest,
        "child_exit_code": child_exit_code,
        "termination": termination.to_dict() if termination is not None else None,
        "command": {
            "executable": Path(argv[0]).name,
            "argument_count": len(argv) - 1,
            "digest": bytes_digest(canonical_bytes(argv)),
        },
        "projections": [
            {
                "kind": item["kind"],
                "path": item["path"],
                "manifest_digest": item["manifest"]["content_digest"],
            }
            for item in projected
        ],
        "coverage_receipt_id": supervisor.sealed.coverage.receipt_id,
        "coverage_digest": supervisor.sealed.coverage.content_digest,
    }
    supervisor.bundle.write_receipt("supervisor-run", receipt)
    supervisor.bundle.write_manifest()
    return CapturedCommandResult(exit_code=child_exit_code, receipt=receipt)


def _stop_child(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _interrupt_child(process: subprocess.Popen[bytes]) -> None:
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            _stop_child(process)


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        return str(number)


__all__ = [
    "CapturedCommandResult",
    "TRACE_RUN_RECEIPT_SCHEMA_VERSION",
    "run_captured_command",
]
