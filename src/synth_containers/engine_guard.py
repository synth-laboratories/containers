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
# The engine publishes a loopback port the parent connects to. Docker cannot
# publish a port from ``--network none`` or an ``--internal`` network, so the
# engine runs on the default bridge with unrestricted egress. This is a stated
# capability, recorded on every receipt, not an enforced restriction.
EGRESS_CAPABILITY = "unrestricted_bridge"
EGRESS_NETWORK_MODE = "bridge"
HOST_CONFIG_FORMAT = "{{json .HostConfig}}"


class LimitsMismatch(RuntimeError):
    """Docker did not apply the declared limits; never downgraded to a warning."""

    code = "native_engine_limits_mismatch"

    def __init__(self, declared: dict, observed: dict | None, cause: str):
        self.declared, self.observed, self.cause = declared, observed, cause
        super().__init__(f"{self.code}: stage=limits cause={cause}")


def engine_port(request: dict) -> int:
    port = request.get("port", 8098)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("engine port outside 1..65535")
    return port


def declared_limits(request: dict) -> dict:
    """Declared limits in Docker's own HostConfig units; refuses an undeclared field."""
    missing = [key for key in ("cpu", "memory_mb") if request.get(key) is None]
    if missing:
        raise ValueError("limits_undeclared: missing " + ",".join(missing))
    cpu, memory_mb = request["cpu"], request["memory_mb"]
    if type(cpu) not in (int, float) or not 0 < cpu <= 16:
        raise ValueError("limits_invalid: cpu outside (0, 16]")
    if type(memory_mb) is not int or not 128 <= memory_mb <= 8192:
        raise ValueError("limits_invalid: memory_mb outside 128..8192")
    memory = memory_mb * 1024 * 1024
    return {"cpu": cpu, "memory_mb": memory_mb, "nano_cpus": round(cpu * 1_000_000_000),
        "memory_bytes": memory, "memory_swap_bytes": memory, "network_mode": EGRESS_NETWORK_MODE,
        "egress": EGRESS_CAPABILITY}


def observed_limits(host_config: dict) -> dict:
    """The subset of ``docker inspect`` HostConfig that carries resource and network limits."""
    nano, memory = host_config.get("NanoCpus"), host_config.get("Memory")
    return {"cpu": nano / 1_000_000_000 if isinstance(nano, int) else None,
        "memory_mb": memory // (1024 * 1024) if isinstance(memory, int) else None,
        "nano_cpus": nano, "memory_bytes": memory, "memory_swap_bytes": host_config.get("MemorySwap"),
        "network_mode": host_config.get("NetworkMode"), "pids_limit": host_config.get("PidsLimit"),
        "readonly_rootfs": host_config.get("ReadonlyRootfs"),
        "egress": EGRESS_CAPABILITY if host_config.get("NetworkMode") in ("bridge", "default") else None}


def verify_limits(declared: dict, observed: dict) -> None:
    for key in ("nano_cpus", "memory_bytes", "memory_swap_bytes"):
        if observed.get(key) != declared[key]:
            raise LimitsMismatch(declared, observed, f"{key} declared={declared[key]} observed={observed.get(key)}")
    if observed.get("network_mode") not in ("bridge", "default"):
        raise LimitsMismatch(declared, observed, f"network_mode declared=bridge observed={observed.get('network_mode')}")


def inspect_limits(reporter: "GuardReporter", docker: str, name: str, declared: dict) -> dict:
    inspected = reporter.await_process([docker, "inspect", "-f", HOST_CONFIG_FORMAT, name], "limits")
    if inspected.returncode != 0:
        raise LimitsMismatch(declared, None, "inspect_failed " + inspected.stderr.strip()[-256:])
    try:
        host_config = json.loads(inspected.stdout)
    except ValueError:
        raise LimitsMismatch(declared, None, "inspect_unparseable") from None
    if not isinstance(host_config, dict):
        raise LimitsMismatch(declared, None, "inspect_unparseable")
    observed = observed_limits(host_config)
    verify_limits(declared, observed)
    return observed


def docker_create_argv(request: dict, name: str, docker: str = "docker") -> list[str]:
    """Owned allocation shape; isolation limits are fixed, only command/port vary."""
    port = engine_port(request)
    limits = declared_limits(request)
    command = request.get("command")
    if command is not None and (not command or any(type(a) is not str or not a for a in command)):
        raise ValueError("engine command must be non-empty strings")
    return [docker, "create", "--pull", "never", "--name", name,
        "--label", f"{OWNER_LABEL}=" + request["owner"], "--label", f"{ALLOCATION_LABEL}={name}",
        "--read-only", "--tmpfs", "/tmp:rw,size=67108864", "--memory", f"{limits['memory_mb']}m",
        "--memory-swap", f"{limits['memory_mb']}m", "--cpus", str(request["cpu"]), "--pids-limit", "128",
        "--network", limits["network_mode"],
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


def handshake(name: str) -> None:
    """First report, before any Docker work: the parent starts silence accounting here."""
    sys.stdout.write(json.dumps({"event": "hello", "name": name, "pid": os.getpid(), "at": time.time()}) + "\n")
    sys.stdout.flush()


def main():
    name = "synth-eval-" + uuid4().hex[:16]
    handshake(name)
    request = json.loads(sys.stdin.readline())
    docker = request.get("docker_cli", "docker")
    heartbeat = float(request.get("heartbeat_seconds", DEFAULT_HEARTBEAT_SECONDS))
    record = Path(request["receipt"])
    record.parent.mkdir(parents=True, exist_ok=True)
    reporter = GuardReporter(name=name, heartbeat=heartbeat)
    identifier = None
    started = time.monotonic()
    create_seconds = None
    error = None
    code = None
    limits = {"declared": None, "observed": None, "egress": EGRESS_CAPABILITY,
        "egress_reason": "published loopback port requires a bridge network; Docker cannot publish from none/internal"}
    reporter.state("allocating", owner=request["owner"], label=f"{OWNER_LABEL}={request['owner']}")
    try:
        limits["declared"] = declared_limits(request)
        argv = docker_create_argv(request, name, docker)
        reporter.state("creating")
        created = reporter.await_process(argv, "creating")
        create_seconds = round(time.monotonic() - started, 3)
        if created.returncode != 0:
            raise RuntimeError("native_engine_create_failed: " + created.stderr.strip()[-512:])
        identifier = created.stdout.strip()
        reporter.state("created", observed=observe(reporter, docker, name), container=identifier,
            create_seconds=create_seconds)
        limits["observed"] = inspect_limits(reporter, docker, name, limits["declared"])
        reporter.state("limits_verified", limits=limits)
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
        if isinstance(failure, LimitsMismatch):
            code, limits["observed"] = failure.code, failure.observed
        elif isinstance(failure, ValueError) and error.startswith(("limits_undeclared", "limits_invalid")):
            code = error.split(":", 1)[0]
        reporter.state("failed", error=error, code=code, limits=limits)
    finally:
        cleanup, returncode = remove(reporter, docker, name)
        durable_write(record, content_bytes({"container": identifier, "name": name, "owner": request["owner"],
            "label": f"{OWNER_LABEL}={request['owner']}", "cleanup": cleanup, "returncode": returncode,
            "last_observed_state": reporter.last_observed, "transitions": reporter.transitions,
            "create_seconds": create_seconds, "error": error, "code": code, "limits": limits, "recorded_by": "guard"}), replace=True)
        reporter.emit({"event": "receipt", "name": name, "cleanup": cleanup})


if __name__ == "__main__":
    main()
