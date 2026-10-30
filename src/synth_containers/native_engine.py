"""Explicitly owned local native engine, no image build or provider credentials.

The parent waits on the guard while the guard reports progress. It fails closed
only when the guard reports a terminal state, exits, or stops reporting; it
never terminates the guard, which holds custody of the Docker allocation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

from .engine_guard import DEFAULT_HEARTBEAT_SECONDS, OWNER_LABEL, TERMINAL_CLEANUP_STATES

# Liveness bound on guard *reporting*, not on Docker work: the guard heartbeats
# every DEFAULT_HEARTBEAT_SECONDS while any Docker call is in flight. It applies
# only after the guard's ``hello`` handshake; before it, interpreter start-up on a
# loaded host is not silence, and the parent waits for the hello or guard exit.
DEFAULT_SILENCE_SECONDS = 120.0
DEFAULT_GUARD_COMMAND = (sys.executable, "-m", "synth_containers.engine_guard")


class NativeEngineError(RuntimeError):
    """Typed engine failure; ``str()`` starts with the stable code."""

    def __init__(self, code: str, *, stage: str, name: str | None, last_state: str | None, cause: str | None = None,
                 limits: dict | None = None):
        self.code, self.stage, self.name, self.last_state, self.cause = code, stage, name, last_state, cause
        self.limits = limits
        super().__init__(f"{code}: stage={stage} name={name} last_state={last_state}" + (f" cause={cause}" if cause else ""))


def guard_request(plan, *, owner: str, receipt: Path, docker_cli: str = "docker",
                  heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS) -> dict:
    return {"owner": owner, "receipt": str(receipt.absolute()), "image": plan.image_digest,
        "cpu": plan.cpu, "memory_mb": plan.memory_mb, "port": plan.effective_engine_port,
        "command": list(plan.engine_command) if plan.engine_command is not None else None,
        "egress": plan.egress, "docker_cli": docker_cli, "heartbeat_seconds": heartbeat_seconds}


@dataclass
class GuardChannel:
    """Line reader over the guard's stdout with a reporting-liveness bound."""

    fd: int
    silence_seconds: float
    buffer: bytes = b""
    eof: bool = False
    name: str | None = None
    last_state: str | None = None
    handshaken: bool = False
    events: list[dict] = field(default_factory=list)

    def next_event(self) -> dict | None:
        """Next event; None at EOF. Raises TimeoutError when the guard stops reporting."""
        with selectors.DefaultSelector() as selector:
            selector.register(self.fd, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                if self.eof:
                    return None
                # Before the hello, block until the guard reports or exits (EOF):
                # a slow interpreter start is progress, not silence.
                if not selector.select(timeout=self.silence_seconds if self.handshaken else None):
                    raise TimeoutError("guard_silent")
                chunk = os.read(self.fd, 65536)
                if not chunk:
                    self.eof = True
                    continue
                self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        event = json.loads(line)
        self.name = event.get("name", self.name)
        if event.get("event") == "hello":
            self.handshaken = True
        if event.get("event") == "state":
            self.last_state = event["state"]
            self.events.append(event)
        return event

    def until(self, states: tuple[str, ...]) -> dict | None:
        while (event := self.next_event()) is not None:
            if event.get("event") == "state" and event["state"] in states:
                return event
        return None


def _parent_receipt(receipt: Path, channel: GuardChannel, owner: str, reason: str) -> None:
    """The guard did not record cleanup; retain what a later sweep needs to find the allocation.

    A guard that never sent its hello did no Docker work, so nothing was allocated.
    """
    from .tracing.atomic import durable_write
    from .package_models import content_bytes
    cleanup = "unconfirmed" if channel.handshaken else "not_allocated"
    durable_write(receipt, content_bytes({"container": None, "name": channel.name, "owner": owner,
        "label": f"{OWNER_LABEL}={owner}", "cleanup": cleanup, "reason": reason,
        "last_observed_state": channel.last_state, "transitions": channel.events, "recorded_by": "parent",
        "recorded_at": time.time()}), replace=True)


@contextmanager
def owned_gold_engine(plan, *, owner: str, receipt: Path, docker_cli: str = "docker",
                      heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
                      silence_seconds: float = DEFAULT_SILENCE_SECONDS,
                      guard_command: tuple[str, ...] = DEFAULT_GUARD_COMMAND):
    process = subprocess.Popen(list(guard_command),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True)
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    channel = GuardChannel(process.stdout.fileno(), silence_seconds)
    silent = False
    try:
        request = guard_request(plan, owner=owner, receipt=receipt, docker_cli=docker_cli,
            heartbeat_seconds=heartbeat_seconds)
        process.stdin.write((json.dumps(request) + "\n").encode())
        process.stdin.flush()
        try:
            ready = channel.until(("ready", "failed"))
        except TimeoutError:
            silent = True
            raise NativeEngineError("native_engine_guard_silent", stage="startup", name=channel.name,
                last_state=channel.last_state) from None
        if ready is None:
            # EOF: the guard closed its report stream by exiting.
            raise NativeEngineError("native_engine_startup_failed", stage="startup", name=channel.name,
                last_state=channel.last_state, cause=f"guard_exited_{process.wait()}"
                + ("" if channel.handshaken else "_before_hello"))
        if ready["state"] != "ready":
            code = ready.get("code")
            raise NativeEngineError(code or "native_engine_startup_failed",
                stage=ready.get("stage") or ("limits" if code else "startup"),
                name=channel.name, last_state=channel.last_state, cause=ready.get("error"), limits=ready.get("limits"))
        yield ready["url"]
    finally:
        process.stdin.close()
        terminal = None
        if not silent:
            try:
                terminal = channel.until(TERMINAL_CLEANUP_STATES)
            except TimeoutError:
                silent = True
        if silent:
            # Preserve guard custody: terminating it could strand the container.
            # The guard overwrites this receipt if it later confirms removal.
            _parent_receipt(receipt, channel, owner, "guard_silent")
            process.stdout.close()
            process.stderr.close()
            raise NativeEngineError("native_engine_cleanup_unconfirmed", stage="cleanup", name=channel.name,
                last_state=channel.last_state, cause="guard_silent")
        try:
            while channel.next_event() is not None:
                pass
        except TimeoutError:
            # A removal observation is not a durable receipt. Do not terminate
            # the guard, which may still be persisting its authoritative record.
            _parent_receipt(receipt, channel, owner, "guard_silent_after_cleanup")
            process.stdout.close()
            process.stderr.close()
            raise NativeEngineError("native_engine_cleanup_unconfirmed", stage="cleanup", name=channel.name,
                last_state=channel.last_state, cause="guard_silent_after_cleanup") from None
        process.wait()
        errors = process.stderr.read(65536)
        process.stdout.close()
        process.stderr.close()
        if terminal is None:
            from .tracing.atomic import durable_write
            durable_write(receipt.with_suffix(".stderr"), errors, replace=True)
        try:
            recorded = json.loads(receipt.read_bytes())
        except (OSError, ValueError):
            recorded = None
        if (not isinstance(recorded, dict) or recorded.get("recorded_by") != "guard"
                or recorded.get("owner") != owner or recorded.get("name") != channel.name
                or recorded.get("cleanup") not in ("removed", "not_allocated", "unconfirmed")):
            _parent_receipt(receipt, channel, owner, f"guard_exited_{process.returncode}")
            recorded = json.loads(receipt.read_bytes())
        if recorded["cleanup"] not in ("removed", "not_allocated"):
            raise NativeEngineError("native_engine_cleanup_unconfirmed", stage="cleanup", name=channel.name,
                last_state=recorded.get("last_observed_state", channel.last_state), cause=recorded.get("reason"))
        exited = recorded.get("engine_exit")
        if isinstance(exited, dict) and exited.get("oom_killed") is True:
            # Replaces the in-flight transport error (kept as __context__): the engine
            # did not become unreachable, the kernel killed it at the declared limit.
            raise NativeEngineError("native_engine_oom_killed", stage="running", name=channel.name,
                last_state="oom_killed", cause=f"memory_mb={plan.memory_mb} memory_swap_mb={plan.memory_mb}",
                limits=recorded.get("limits"))
