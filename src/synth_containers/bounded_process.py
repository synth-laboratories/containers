"""POSIX process-group actuator with collection admission before each disk write.

The output bound covers merged stdout/stderr only. Files written directly by the
child require a separate filesystem quota. A killed CLI does not prove remote
sandbox cleanup; native provider custody remains an independent operation.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
from collections.abc import Mapping, Sequence
from pathlib import Path


class ProcessOutputLimit(RuntimeError):
    pass


async def run_bounded_process(
    argv: Sequence[str],
    *,
    output: Path,
    max_output_bytes: int,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    termination_grace_seconds: float = 5,
    redact: Sequence[str] = (),
) -> int:
    if not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
        raise ValueError("Process requires an argument vector")
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("Output allowance must be positive bytes")
    if (
        type(termination_grace_seconds) not in (int, float)
        or not 0 < termination_grace_seconds <= 30
    ):
        raise ValueError("Termination grace must be at most 30 seconds")
    if len(redact) > 128 or any(
        not isinstance(value, str) or not value or len(value.encode()) > 4096 for value in redact
    ):
        raise ValueError("Redaction requires at most 128 nonempty strings up to 4096 UTF-8 bytes")
    secrets = sorted({value.encode() for value in redact}, key=len, reverse=True)
    pattern = re.compile(b"|".join(re.escape(value) for value in secrets)) if secrets else None
    withheld = max((len(value) for value in secrets), default=1) - 1
    # Exclusive output also prevents a resumed attempt from resetting allowance.
    with output.open("xb") as handle:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            limit=65536,
        )

        async def terminate():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), termination_grace_seconds)
            except TimeoutError:
                pass
            # Kill descendants even if the leader exited during graceful stop.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            async def reap():
                # Discard bounded chunks after kill so a full PIPE cannot keep
                # asyncio's subprocess transport from completing wait().
                if process.stdout is not None:
                    while await process.stdout.read(65536):
                        pass
                await process.wait()

            await asyncio.wait_for(reap(), termination_grace_seconds)

        consumed = 0
        written = 0
        pending = b""

        def retain(chunk: bytes, *, final: bool = False):
            nonlocal pending, written
            pending += chunk
            cutoff = len(pending) if final else max(0, len(pending) - withheld)
            cursor = 0
            pieces = []
            if pattern is not None:
                for match in pattern.finditer(pending):
                    if match.start() >= cutoff:
                        break
                    pieces.extend((pending[cursor : match.start()], b"[REDACTED]"))
                    cursor = match.end()
            cutoff = max(cutoff, cursor)
            pieces.append(pending[cursor:cutoff])
            data = b"".join(pieces)
            pending = pending[cutoff:]
            allowance = max_output_bytes - written
            handle.write(data[:allowance])
            written += min(len(data), allowance)
            if len(data) > allowance:
                raise ProcessOutputLimit("Redacted process output exceeded admitted bytes")

        try:
            assert process.stdout is not None
            while True:
                chunk = await process.stdout.read(min(65536, max_output_bytes - consumed + 1))
                if not chunk:
                    break
                remaining = max_output_bytes - consumed
                retained = chunk[:remaining]
                consumed += len(retained)
                retain(retained)
                if len(chunk) > remaining:
                    raise ProcessOutputLimit("Merged process output exceeded admitted bytes")
            retain(b"", final=True)
            return await process.wait()
        finally:
            # Always fence the process group, including successful leaders which
            # leave a daemon behind. Shield cleanup from the triggering cancel.
            cleanup = asyncio.create_task(terminate())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                # On cancellation/overflow the withheld suffix is deliberately
                # omitted; flushing it could expose a partially emitted secret.
                handle.flush()
                os.fsync(handle.fileno())
