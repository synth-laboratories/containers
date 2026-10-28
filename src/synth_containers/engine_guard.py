"""Foreground-owned Docker allocation. Parent pipe EOF stops the exact allocation.

The guard never kills a Docker CLI call it started: a killed ``docker create``
can still materialize a container in the daemon. Instead every phase is awaited
while the guard reports progress (state transitions plus heartbeats) on stdout,
and cleanup is confirmed only by observing the labelled allocation is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
from uuid import uuid4

import httpx
from synth_containers.tracing.atomic import durable_write
from synth_containers.package_models import content_bytes

OWNER_LABEL = "ai.synth.eval.owner"
ALLOCATION_LABEL = "ai.synth.eval.allocation"
DEFAULT_HEARTBEAT_SECONDS = 1.0
TERMINAL_CLEANUP_STATES = ("removed", "unconfirmed")


def engine_port(request: dict) -> int:
    port = request.get("port", 8098)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("engine port outside 1..65535")
    return port


def docker_create_argv(request: dict, name: str, docker: str = "docker") -> list[str]:
    """Owned allocation shape; isolation limits are fixed, only command/port vary."""
    port = engine_port(request)
    command = request.get("command")
    if command is not None and (not command or any(type(a) is not str or not a for a in command)):
        raise ValueError("engine command must be non-empty strings")
    return [docker, "create", "--pull", "never", "--name", name,
        "--label", f"{OWNER_LABEL}=" + request["owner"], "--label", f"{ALLOCATION_LABEL}={name}",
        "--read-only", "--tmpfs", "/tmp:rw,size=67108864", "--memory", str(request["memory_mb"]) + "m",
        "--cpus", str(request["cpu"]), "--pids-limit", "128",
        "-p", f"127.0.0.1::{port}", "-e", f"PORT={port}", request["image"], *(command or ())]


@dataclass
class GuardReporter:
    """Emits progress lines to the parent; a vanished parent never stops cleanup."""

    name: str
    heartbeat: float
    transitions: list[dict] = field(default_factory=list)
    parent_gone: bool = False
    cancelled: bool = False
    last_observed: str | None = None

    def emit(self, payload: dict) -> None:
        if self.parent_gone:
            return
        try:
            sys.stdout.write(json.dumps(payload) + "\n")
            sys.stdout.flush()
        except OSError:
            self.parent_gone = True

    def state(self, state: str, **extra) -> None:
        if "observed" in extra and extra["observed"] is not None:
            self.last_observed = extra["observed"]
        entry = {"state": state, "at": time.time(), "observed": self.last_observed, **extra}
        self.transitions.append(entry)
        self.emit({"event": "state", "name": self.name, **entry})

    def wait_stdin(self, timeout: float) -> None:
        """Wait on the parent pipe; EOF there is the cancel signal."""
        if self.cancelled:
            time.sleep(timeout)
            return
        with selectors.DefaultSelector() as selector:
            selector.register(sys.stdin.fileno(), selectors.EVENT_READ)
            if selector.select(timeout=timeout) and not os.read(sys.stdin.fileno(), 4096):
                self.cancelled = True

    def await_process(self, argv: list[str], phase: str) -> subprocess.CompletedProcess:
        """Run a Docker CLI call to completion, heartbeating; never kill it."""
        started = time.monotonic()
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while process.poll() is None:
            self.wait_stdin(self.heartbeat)
            self.emit({"event": "heartbeat", "name": self.name, "phase": phase, "at": time.time(),
                "elapsed": round(time.monotonic() - started, 3), "observed": self.last_observed})
        stdout, stderr = process.communicate()
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def observe(reporter: GuardReporter, docker: str, name: str) -> str | None:
    """Daemon-reported status for the exact allocation; ``absent`` when gone, None when unobservable."""
    listed = reporter.await_process([docker, "ps", "-a", "-q", "--filter", f"label={ALLOCATION_LABEL}={name}"], "observe")
    if listed.returncode != 0:
        return None
    if not listed.stdout.strip():
        return "absent"
    inspected = reporter.await_process([docker, "inspect", "-f", "{{.State.Status}}", name], "observe")
    return (inspected.stdout.strip() or None) if inspected.returncode == 0 else None


def await_ready(reporter: GuardReporter, docker: str, name: str, url: str) -> None:
    """Health polling continues while the container is running; exit is terminal."""
    with httpx.Client(timeout=1, trust_env=False) as client:
        polls = 0
        while not reporter.cancelled and not reporter.parent_gone:
            try:
                if client.get(url + "/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            polls += 1
            if polls % 10 == 0:
                status = observe(reporter, docker, name)
                if status != "running":
                    raise RuntimeError(f"native_engine_not_running: observed {status}")
                reporter.emit({"event": "heartbeat", "name": name, "phase": "readiness", "at": time.time(),
                    "observed": status})
            reporter.wait_stdin(.1)
    raise RuntimeError("native_engine_startup_cancelled")


def remove(reporter: GuardReporter, docker: str, name: str) -> tuple[str, int | None]:
    """Remove the labelled allocation and confirm by observation; retries while Docker answers."""
    reporter.state("stopping")
    returncode = None
    while True:
        returncode = reporter.await_process([docker, "rm", "-f", name], "stopping").returncode
        status = observe(reporter, docker, name)
        if status == "absent":
            reporter.state("removed", observed="absent")
            return "removed", returncode
        if status is None:
            reporter.state("unconfirmed", observed=reporter.last_observed, reason="docker_unobservable")
            return "unconfirmed", returncode
        reporter.state("stopping", observed=status)
        time.sleep(reporter.heartbeat)


def main():
    request = json.loads(sys.stdin.readline())
    docker = request.get("docker_cli", "docker")
    heartbeat = float(request.get("heartbeat_seconds", DEFAULT_HEARTBEAT_SECONDS))
    name = "synth-eval-" + uuid4().hex[:16]
    record = Path(request["receipt"])
    record.parent.mkdir(parents=True, exist_ok=True)
    reporter = GuardReporter(name=name, heartbeat=heartbeat)
    identifier = None
    started = time.monotonic()
    create_seconds = None
    error = None
    reporter.state("allocating", owner=request["owner"], label=f"{OWNER_LABEL}={request['owner']}")
    try:
        argv = docker_create_argv(request, name, docker)
        reporter.state("creating")
        created = reporter.await_process(argv, "creating")
        create_seconds = round(time.monotonic() - started, 3)
        if created.returncode != 0:
            raise RuntimeError("native_engine_create_failed: " + created.stderr.strip()[-512:])
        identifier = created.stdout.strip()
        reporter.state("created", observed=observe(reporter, docker, name), container=identifier,
            create_seconds=create_seconds)
        if reporter.cancelled or reporter.parent_gone:
            raise RuntimeError("native_engine_startup_cancelled")
        reporter.state("starting")
        start = reporter.await_process([docker, "start", name], "starting")
        if start.returncode != 0:
            raise RuntimeError("native_engine_start_failed: " + start.stderr.strip()[-512:])
        reporter.state("running", observed=observe(reporter, docker, name))
        mapped = reporter.await_process([docker, "port", name, f"{engine_port(request)}/tcp"], "port")
        if mapped.returncode != 0 or not mapped.stdout.strip():
            raise RuntimeError("native_engine_port_unmapped: " + mapped.stderr.strip()[-512:])
        url = "http://127.0.0.1:" + mapped.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        await_ready(reporter, docker, name, url)
        reporter.state("ready", url=url, container=identifier)
        # The controlling invocation alone holds this pipe's write end. It is
        # closed by normal finally cleanup, Ctrl-C and abrupt process death.
        while not reporter.cancelled and not reporter.parent_gone:
            reporter.wait_stdin(3600)
    except Exception as failure:  # reported to the parent; cleanup below still runs
        error = str(failure)
        reporter.state("failed", error=error)
    finally:
        cleanup, returncode = remove(reporter, docker, name)
        durable_write(record, content_bytes({"container": identifier, "name": name, "owner": request["owner"],
            "label": f"{OWNER_LABEL}={request['owner']}", "cleanup": cleanup, "returncode": returncode,
            "last_observed_state": reporter.last_observed, "transitions": reporter.transitions,
            "create_seconds": create_seconds, "error": error, "recorded_by": "guard"}), replace=True)
        reporter.emit({"event": "receipt", "name": name, "cleanup": cleanup})


if __name__ == "__main__":
    main()
