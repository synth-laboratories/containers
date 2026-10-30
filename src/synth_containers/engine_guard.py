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
import threading
import time
from uuid import uuid4

import httpx
from synth_containers.tracing.atomic import durable_write
from synth_containers.package_models import content_bytes
from synth_containers.engine_egress import (BRIDGE_NETWORK, EGRESS_NONE, OBSERVED_NONE, SIDECAR_IMAGE,
    EgressMismatch, EgressNames, EgressUnsupported, engine_address, network_create_argv, network_names,
    parse_json, requested_egress, sidecar_create_argv, verify_isolation)

OWNER_LABEL = "ai.synth.eval.owner"
ALLOCATION_LABEL = "ai.synth.eval.allocation"
DEFAULT_HEARTBEAT_SECONDS = 1.0
TERMINAL_CLEANUP_STATES = ("removed", "unconfirmed")
# Declared egress ``unrestricted_bridge`` (the default): the engine publishes its
# loopback port from the default bridge and has unrestricted egress, a stated
# capability recorded on every receipt. Declared egress ``none``: the engine is
# alone on an owned internal network and reached through a forwarding sidecar
# (see engine_egress); the receipt records the observed placement.
EGRESS_CAPABILITY = "unrestricted_bridge"
EGRESS_NETWORK_MODE = BRIDGE_NETWORK
HOST_CONFIG_FORMAT = "{{json .HostConfig}}"
STATE_FORMAT = "{{json .State}}"


class LimitsMismatch(RuntimeError):
    """Docker did not apply the declared limits; never downgraded to a warning."""

    code = "native_engine_limits_mismatch"

    stage = "limits"

    def __init__(self, declared: dict, observed: dict | None, cause: str):
        self.declared, self.observed, self.cause = declared, observed, cause
        super().__init__(f"{self.code}: stage=limits cause={cause}")


def engine_port(request: dict) -> int:
    port = request.get("port", 8098)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("engine port outside 1..65535")
    return port


def declared_limits(request: dict, network: str | None = None) -> dict:
    """Declared limits in Docker's own HostConfig units; refuses an undeclared field.

    ``network`` is the owned internal network when the declared egress is ``none``.
    """
    missing = [key for key in ("cpu", "memory_mb") if request.get(key) is None]
    if missing:
        raise ValueError("limits_undeclared: missing " + ",".join(missing))
    cpu, memory_mb = request["cpu"], request["memory_mb"]
    if type(cpu) not in (int, float) or not 0 < cpu <= 16:
        raise ValueError("limits_invalid: cpu outside (0, 16]")
    if type(memory_mb) is not int or not 128 <= memory_mb <= 8192:
        raise ValueError("limits_invalid: memory_mb outside 128..8192")
    memory = memory_mb * 1024 * 1024
    egress = requested_egress(request)
    if egress == EGRESS_NONE and not network:
        raise ValueError("egress_network_missing: declared egress none needs an owned internal network")
    return {"cpu": cpu, "memory_mb": memory_mb, "nano_cpus": round(cpu * 1_000_000_000),
        "memory_bytes": memory, "memory_swap_bytes": memory,
        "network_mode": network if egress == EGRESS_NONE else EGRESS_NETWORK_MODE,
        "egress": egress}


def observed_limits(host_config: dict) -> dict:
    """The subset of ``docker inspect`` HostConfig that carries resource and network limits."""
    nano, memory = host_config.get("NanoCpus"), host_config.get("Memory")
    return {"cpu": nano / 1_000_000_000 if isinstance(nano, int) else None,
        "memory_mb": memory // (1024 * 1024) if isinstance(memory, int) else None,
        "nano_cpus": nano, "memory_bytes": memory, "memory_swap_bytes": host_config.get("MemorySwap"),
        "network_mode": host_config.get("NetworkMode"), "pids_limit": host_config.get("PidsLimit"),
        "readonly_rootfs": host_config.get("ReadonlyRootfs"),
        "port_bindings": host_config.get("PortBindings") or None,
        "egress": EGRESS_CAPABILITY if host_config.get("NetworkMode") in ("bridge", "default") else None}


def verify_limits(declared: dict, observed: dict) -> None:
    for key in ("nano_cpus", "memory_bytes", "memory_swap_bytes"):
        if observed.get(key) != declared[key]:
            raise LimitsMismatch(declared, observed, f"{key} declared={declared[key]} observed={observed.get(key)}")
    accepted = ("bridge", "default") if declared["network_mode"] == EGRESS_NETWORK_MODE else (declared["network_mode"],)
    if observed.get("network_mode") not in accepted:
        raise LimitsMismatch(declared, observed,
            f"network_mode declared={declared['network_mode']} observed={observed.get('network_mode')}")


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


def docker_create_argv(request: dict, name: str, docker: str = "docker", network: str | None = None) -> list[str]:
    """Owned allocation shape; isolation limits are fixed, only command/port/egress vary.

    With declared egress ``none`` the engine joins ``network`` and publishes nothing.
    """
    port = engine_port(request)
    limits = declared_limits(request, network)
    command = request.get("command")
    if command is not None and (not command or any(type(a) is not str or not a for a in command)):
        raise ValueError("engine command must be non-empty strings")
    return [docker, "create", "--pull", "never", "--name", name,
        "--label", f"{OWNER_LABEL}=" + request["owner"], "--label", f"{ALLOCATION_LABEL}={name}",
        "--read-only", "--tmpfs", "/tmp:rw,size=67108864", "--memory", f"{limits['memory_mb']}m",
        "--memory-swap", f"{limits['memory_mb']}m", "--cpus", str(request["cpu"]), "--pids-limit", "128",
        "--network", limits["network_mode"],
        *(() if limits["egress"] == EGRESS_NONE else ("-p", f"127.0.0.1::{port}")),
        "-e", f"PORT={port}", request["image"], *(command or ())]


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


def observe_network(reporter: GuardReporter, docker: str, network: str) -> str | None:
    """``present`` or ``absent`` for the exact owned network; None when unobservable."""
    listed = reporter.await_process([docker, "network", "ls", "-q", "--filter", f"label={ALLOCATION_LABEL}={network}"],
        "observe")
    if listed.returncode != 0:
        return None
    return "present" if listed.stdout.strip() else "absent"


def inspect_json(reporter: GuardReporter, docker: str, target: list[str], template: str, phase: str):
    inspected = reporter.await_process([docker, *target[:-1], "inspect", "-f", template, target[-1]], phase)
    return parse_json(inspected.stdout) if inspected.returncode == 0 else None


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


def engine_exit(reporter: GuardReporter, docker: str, name: str) -> dict | None:
    """How the engine itself ended, read from Docker before removal; None when unobservable.

    ``oom_killed`` is the kernel's verdict under the declared ``--memory``/``--memory-swap``
    limit. Without it an OOM-killed engine looks to the parent like a network fault
    ("gold unreachable") instead of a memory limit the plan declared too small.
    """
    state = inspect_json(reporter, docker, [name], STATE_FORMAT, "exit")
    if not isinstance(state, dict):
        return None
    status, exit_code = state.get("Status"), state.get("ExitCode")
    return {"status": status if isinstance(status, str) else None, "oom_killed": state.get("OOMKilled") is True,
        "exit_code": exit_code if type(exit_code) is int else None}


def remove_container(reporter: GuardReporter, docker: str, name: str) -> tuple[str, int | None]:
    """Remove one labelled container and confirm by observation; retries while Docker answers."""
    while True:
        returncode = reporter.await_process([docker, "rm", "-f", name], "stopping").returncode
        status = observe(reporter, docker, name)
        if status == "absent":
            return "absent", returncode
        if status is None:
            return "unobservable", returncode
        reporter.state("stopping", observed=status, resource=name)
        time.sleep(reporter.heartbeat)


def remove_network(reporter: GuardReporter, docker: str, network: str) -> tuple[str, int | None]:
    """Remove the owned network; a refused removal of a still-present network is reported, not retried."""
    while True:
        removed = reporter.await_process([docker, "network", "rm", network], "stopping")
        status = observe_network(reporter, docker, network)
        if status in ("absent", None):
            return status or "unobservable", removed.returncode
        if removed.returncode != 0:
            return "present", removed.returncode
        reporter.state("stopping", observed=reporter.last_observed, resource=network)
        time.sleep(reporter.heartbeat)


def remove(reporter: GuardReporter, docker: str, name: str, names: EgressNames | None = None
           ) -> tuple[str, int | None, list[str]]:
    """Remove sidecar, engine, then network; ``removed`` only when every owned resource is observed absent."""
    reporter.state("stopping")
    steps = [] if names is None else [(names.sidecar, remove_container)]
    steps.append((name, remove_container))
    if names is not None:
        steps.append((names.network, remove_network))
    remaining, returncode = [], None
    for resource, remover in steps:
        status, code = remover(reporter, docker, resource)
        if resource == name:
            returncode = code
        if status != "absent":
            remaining.append(f"{resource}:{status}")
    if not remaining:
        reporter.state("removed", observed="absent")
        return "removed", returncode, remaining
    reporter.state("unconfirmed", observed=reporter.last_observed,
        reason="docker_unobservable" if all(item.endswith(":unobservable") for item in remaining) else "resources_remain",
        remaining=remaining)
    return "unconfirmed", returncode, remaining


def handshake(name: str) -> None:
    """First report, before any Docker work: the parent starts silence accounting here."""
    sys.stdout.write(json.dumps({"event": "hello", "name": name, "pid": os.getpid(), "at": time.time()}) + "\n")
    sys.stdout.flush()


def persist_receipt(reporter: GuardReporter, record: Path, document: dict) -> None:
    """Keep the parent informed while durable receipt I/O is still in progress."""
    finished = threading.Event()
    errors: list[BaseException] = []

    def write() -> None:
        try:
            durable_write(record, content_bytes(document), replace=True)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    writer = threading.Thread(target=write, name="engine-receipt-writer")
    writer.start()
    started = time.monotonic()
    while not finished.wait(reporter.heartbeat):
        reporter.emit({"event": "heartbeat", "name": reporter.name, "phase": "receipt",
                       "at": time.time(), "elapsed": round(time.monotonic() - started, 3),
                       "observed": reporter.last_observed})
    writer.join()
    if errors:
        raise errors[0]


def require_sidecar_image(reporter: GuardReporter, docker: str, image: str) -> str:
    """Detect, never assume: the forwarder image must already be local (it is never pulled)."""
    inspected = reporter.await_process([docker, "image", "inspect", "-f", "{{.Id}}", image], "egress")
    identifier = inspected.stdout.strip()
    if inspected.returncode != 0 or not identifier.startswith("sha256:"):
        raise EgressUnsupported(EGRESS_NONE, f"sidecar image {image} not present locally")
    return identifier


def isolate(reporter: GuardReporter, docker: str, request: dict, names: EgressNames, owner_label: str,
            egress: dict) -> str:
    """Engine is created and started on the internal network; start the sidecar and verify placement.

    Returns the parent-reachable loopback URL published by the sidecar.
    """
    port = engine_port(request)
    declared = {"network": names.network, "engine_port": port, "sidecar_image": egress["sidecar"]["image"]}
    engine_networks = inspect_json(reporter, docker, [names.engine], "{{json .NetworkSettings.Networks}}", "egress")
    address = engine_address(engine_networks, names.network)
    if address is None:
        raise EgressMismatch(declared, {"engine_networks": network_names(engine_networks)}, "engine_address observed=none")
    reporter.state("sidecar_creating", resource=names.sidecar)
    created = reporter.await_process(sidecar_create_argv(docker, names, owner_label, ALLOCATION_LABEL, port, address,
        egress["sidecar"]["image_id"]), "sidecar")
    if created.returncode != 0:
        raise RuntimeError("native_engine_sidecar_create_failed: " + created.stderr.strip()[-512:])
    egress["sidecar"]["container"] = created.stdout.strip()
    connected = reporter.await_process([docker, "network", "connect", names.network, names.sidecar], "sidecar")
    if connected.returncode != 0:
        raise RuntimeError("native_engine_sidecar_connect_failed: " + connected.stderr.strip()[-512:])
    started = reporter.await_process([docker, "start", names.sidecar], "sidecar")
    if started.returncode != 0:
        raise RuntimeError("native_engine_sidecar_start_failed: " + started.stderr.strip()[-512:])
    engine_host = inspect_json(reporter, docker, [names.engine], "{{json .HostConfig.PortBindings}}", "egress")
    sidecar_networks = inspect_json(reporter, docker, [names.sidecar], "{{json .NetworkSettings.Networks}}", "egress")
    sidecar_bindings = inspect_json(reporter, docker, [names.sidecar], "{{json .HostConfig.PortBindings}}", "egress")
    internal = inspect_json(reporter, docker, ["network", names.network], "{{json .Internal}}", "egress")
    observed = {"network_internal": internal, "engine_networks": network_names(engine_networks),
        "engine_port_bindings": engine_host or None, "engine_address": address,
        "sidecar_networks": network_names(sidecar_networks), "sidecar_port_bindings": sidecar_bindings}
    egress["observed_placement"] = observed
    verify_isolation(declared, observed, port)
    mapped = reporter.await_process([docker, "port", names.sidecar, f"{port}/tcp"], "port")
    if mapped.returncode != 0 or not mapped.stdout.strip():
        raise RuntimeError("native_engine_port_unmapped: " + mapped.stderr.strip()[-512:])
    host_port = mapped.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
    egress["forwarding"] = {"host": f"127.0.0.1:{host_port}", "sidecar": f"{names.sidecar}:{port}",
        "engine": f"{address}:{port}"}
    egress["observed"] = OBSERVED_NONE
    return "http://127.0.0.1:" + host_port


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
    stage = None
    owner_label = f"{OWNER_LABEL}={request['owner']}"
    names = None
    egress = {"declared": request.get("egress") or EGRESS_CAPABILITY, "observed": None}
    limits = {"declared": None, "observed": None, "egress": EGRESS_CAPABILITY,
        "egress_reason": "published loopback port requires a bridge network; Docker cannot publish from none/internal"}
    reporter.state("allocating", owner=request["owner"], label=owner_label)
    try:
        declared_egress = requested_egress(request)
        if declared_egress == EGRESS_NONE:
            names = EgressNames(name)
            limits["egress"] = None
            limits["egress_reason"] = (f"engine alone on internal network {names.network}; parent reaches it "
                f"through sidecar {names.sidecar} on bridge and {names.network}")
            image = request.get("sidecar_image") or SIDECAR_IMAGE
            egress.update(network={"name": names.network, "id": None},
                sidecar={"name": names.sidecar, "image": image, "image_id": None, "container": None})
            egress["sidecar"]["image_id"] = require_sidecar_image(reporter, docker, image)
        limits["declared"] = declared_limits(request, names.network if names else None)
        argv = docker_create_argv(request, name, docker, names.network if names else None)
        if names is not None:
            reporter.state("network_creating", resource=names.network)
            network = reporter.await_process(network_create_argv(docker, names, owner_label, ALLOCATION_LABEL), "network")
            if network.returncode != 0:
                raise RuntimeError("native_engine_network_create_failed: " + network.stderr.strip()[-512:])
            egress["network"]["id"] = network.stdout.strip()
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
        if names is not None:
            url = isolate(reporter, docker, request, names, owner_label, egress)
            limits["egress"] = OBSERVED_NONE
            reporter.state("egress_verified", egress=egress)
        else:
            mapped = reporter.await_process([docker, "port", name, f"{engine_port(request)}/tcp"], "port")
            if mapped.returncode != 0 or not mapped.stdout.strip():
                raise RuntimeError("native_engine_port_unmapped: " + mapped.stderr.strip()[-512:])
            url = "http://127.0.0.1:" + mapped.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
            egress["observed"] = EGRESS_CAPABILITY
        await_ready(reporter, docker, name, url)
        reporter.state("ready", url=url, container=identifier)
        # The controlling invocation alone holds this pipe's write end. It is
        # closed by normal finally cleanup, Ctrl-C and abrupt process death.
        while not reporter.cancelled and not reporter.parent_gone:
            reporter.wait_stdin(3600)
    except Exception as failure:  # reported to the parent; cleanup below still runs
        error = str(failure)
        if isinstance(failure, LimitsMismatch):
            code, stage, limits["observed"] = failure.code, failure.stage, failure.observed
        elif isinstance(failure, EgressMismatch):
            code, stage = failure.code, failure.stage
            egress["observed_placement"] = failure.observed
        elif isinstance(failure, EgressUnsupported):
            code, stage = failure.code, failure.stage
            egress["reason"] = failure.reason
        elif isinstance(failure, ValueError) and error.startswith(("limits_undeclared", "limits_invalid", "egress_")):
            code = error.split(":", 1)[0]
            stage = "egress" if code.startswith("egress_") else "limits"
        limits["egress_detail"] = egress
        reporter.state("failed", error=error, code=code, stage=stage, limits=limits)
    finally:
        exited = engine_exit(reporter, docker, name) if identifier is not None else None
        if exited is not None and exited["oom_killed"]:
            reporter.state("oom_killed", observed=exited["status"], exit_code=exited["exit_code"])
        cleanup, returncode, remaining = remove(reporter, docker, name, names)
        limits["egress_detail"] = egress
        persist_receipt(reporter, record, {"container": identifier, "name": name, "owner": request["owner"],
            "label": owner_label, "cleanup": cleanup, "remaining": remaining, "returncode": returncode,
            "containers": {"engine": identifier, "sidecar": (egress.get("sidecar") or {}).get("container")},
            "network": egress.get("network"), "egress": egress,
            "last_observed_state": reporter.last_observed, "transitions": reporter.transitions,
            "create_seconds": create_seconds, "error": error, "code": code, "stage": stage, "limits": limits,
            "engine_exit": exited, "recorded_by": "guard"})
        reporter.emit({"event": "receipt", "name": name, "cleanup": cleanup})


if __name__ == "__main__":
    main()
