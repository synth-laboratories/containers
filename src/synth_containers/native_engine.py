"""Explicitly owned local native engine, no image build or provider credentials."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import selectors
import subprocess
import sys


@contextmanager
def owned_gold_engine(plan, *, owner: str, receipt: Path):
    process = subprocess.Popen([sys.executable, "-m", "synth_containers.engine_guard"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True)
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    allocation_observed = False
    try:
        process.stdin.write(json.dumps({"owner": owner, "receipt": str(receipt.absolute()),
            "image": plan.image_digest, "cpu": plan.cpu, "memory_mb": plan.memory_mb}) + "\n")
        process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=45):
                raise RuntimeError("native_engine_startup_timeout")
            line = process.stdout.readline()
        if not line:
            # Retain bounded diagnostics privately; do not mask startup failure
            # with a missing cleanup receipt for an allocation that never began.
            errors = process.stderr.read(65536)
            from .tracing.atomic import durable_write
            durable_write(receipt.with_suffix(".stderr"), errors.encode())
            raise RuntimeError("native_engine_startup_failed")
        url = json.loads(line)["url"]
        allocation_observed = True
        yield url
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            # Preserve guard custody: terminating it could strand the container.
            raise RuntimeError("native_engine_cleanup_unconfirmed") from None
        finally:
            process.stdout.close()
            process.stderr.close()
        if (receipt.exists() and json.loads(receipt.read_bytes())["cleanup"] != "stopped") or (
            not receipt.exists() and (allocation_observed or process.returncode == 0)
        ):
            raise RuntimeError("native_engine_cleanup_unconfirmed")
