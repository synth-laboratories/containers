# SPDX-License-Identifier: Apache-2.0
# Extracted from Synth Laboratories synth-optimizers eval/executor.py.
# Modified 2026-09-10: remove optimizer imports; accept structural execution limits.
# See LICENSES/Apache-2.0.txt and NOTICE for provenance.
"""Launching one trial in the recipe-pinned target container.

The executor is the only place in `eval` that starts a process, and it will
only ever start the image a trusted recipe pinned by digest. It mounts the
candidate read-only, monitors the writable `/output` footprint, denies
the network by default, and inherits no credentials: a container gets a policy,
a trial description, and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import stat
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from synth_containers.rollout_limits import (
    RolloutLimitKind,
    RolloutLimits,
    RolloutLimitSupervisor,
)


class ExecutionContractError(ValueError):
    """An execution request cannot be represented by this provider."""


class TrialExecutionLimits(Protocol):
    """Runtime controls supplied by the calling evaluation contract.

    Parallelism and scientific budgets remain owned by the submitting supervisor.
    Output bytes are an observed threshold, not a filesystem quota.
    """

    timeout_seconds: int
    cpus: float
    memory_mb: int
    max_output_bytes: int


class ContainerRuntimeError(RuntimeError):
    """The OCI runtime, not the evaluated policy, is what went wrong."""


class ContainerEventStreamError(ContainerRuntimeError):
    """Target observation failed; do not present its evidence as complete."""


class StopFailureReason(StrEnum):
    TIMEOUT = "timeout"
    COMMAND_FAILED = "command_failed"


class ContainerStopUnconfirmed(ContainerRuntimeError):
    """The local actuator could not establish that its container stopped."""

    def __init__(self, container: str, runtime: str, reason: StopFailureReason) -> None:
        self.container = container
        self.runtime = runtime
        self.reason = reason
        detail = "timed out" if reason == StopFailureReason.TIMEOUT else "unconfirmed"
        super().__init__(f"container stop {detail} for {container}; reconcile before retry")

    def to_payload(self) -> dict[str, str | bool]:
        return {
            "container_id": self.container,
            "runtime": self.runtime,
            "stop_status": "unconfirmed",
            "reason": self.reason.value,
            "reconciliation_required": True,
        }


@dataclass(frozen=True, slots=True)
class TrialRunRequest:
    trial_id: str
    image_reference: str
    input_dir: Path
    policy_dir: Path
    output_dir: Path
    limits: TrialExecutionLimits
    network: str
    secrets: Mapping[str, str] = field(default_factory=dict)
    extra_hosts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TrialExecution:
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    started_at: float
    finished_at: float
    stderr_tail: str
    output_limit_exceeded: bool = False


class TrialExecutor(Protocol):
    def run(
        self,
        request: TrialRunRequest,
        *,
        on_event: Callable[[dict[str, Any]], None],
        should_cancel: Callable[[], bool],
        heartbeat: Callable[[], None],
    ) -> TrialExecution: ...


def validate_trial_request(request: TrialRunRequest) -> None:
    """Refuse unrepresentable controls before filesystem or provider work."""
    reference = request.image_reference
    if (
        not isinstance(reference, str)
        or reference.startswith("-")
        or re.fullmatch(r"(?:[^\s@]+@)?sha256:[0-9a-f]{64}", reference) is None
    ):
        raise ExecutionContractError("trial image must be pinned by sha256 digest")
    if request.network not in {"none", "bridge"}:
        raise ExecutionContractError("trial network must be none or bridge")
    cpu = request.limits.cpus
    if isinstance(cpu, bool) or not isinstance(cpu, (int, float)):
        raise ExecutionContractError("trial cpus must be finite and positive")
    try:
        valid_cpu = math.isfinite(cpu) and cpu > 0
    except OverflowError:
        valid_cpu = False
    if not valid_cpu:
        raise ExecutionContractError("trial cpus must be finite and positive")
    if type(request.limits.memory_mb) is not int or request.limits.memory_mb <= 0:
        raise ExecutionContractError("trial memory_mb must be a positive integer")
    try:
        RolloutLimits(request.limits.timeout_seconds, request.limits.max_output_bytes)
    except (ValueError, TypeError, OverflowError) as error:
        raise ExecutionContractError("invalid trial time/output limits") from error
    for path in (request.input_dir, request.policy_dir, request.output_dir):
        if not path.is_absolute() or "," in str(path) or path.is_symlink():
            raise ExecutionContractError(
                "trial mount paths must be absolute, non-symlink and comma-free"
            )
    for name, value in request.secrets.items():
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
            raise ExecutionContractError("invalid injected environment name")
        if not isinstance(value, str) or "\0" in value:
            raise ExecutionContractError("invalid injected environment value")


class OciTrialExecutor:
    """Runs pinned OCI images through `docker` or `podman`."""

    def __init__(self, runtime: str = "docker") -> None:
        if runtime not in {"docker", "podman"}:
            raise ExecutionContractError("container runtime must be docker or podman")
        self.runtime_name = runtime
        binary = shutil.which(runtime)
        if binary is None:
            raise ContainerRuntimeError(
                f"{runtime} is not on PATH; install it or change container_runtime "
                f"in the eval home's runtime.toml"
            )
        self.binary = binary

    def resource_identity(self, trial_id: str) -> dict[str, str]:
        fingerprint = hashlib.sha256(trial_id.encode("utf-8")).hexdigest()[:12]
        return {
            "runtime": self.runtime_name,
            "container_id": f"synth-eval-{trial_id[:40]}-{fingerprint}",
        }

    def _env(self) -> dict[str, str]:
        """A minimal env for the CLI itself. Nothing here reaches the container."""

        env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}
        for name in ("HOME", "DOCKER_HOST", "DOCKER_CONFIG", "XDG_RUNTIME_DIR"):
            value = os.environ.get(name)
            if value:
                env[name] = value
        return env

    def image_digests(self, image: str) -> tuple[str | None, tuple[str, ...]]:
        """Return the local image id and any repository digests it carries."""

        completed = subprocess.run(  # noqa: S603 - fixed binary, fixed argv
            [self.binary, "image", "inspect", "--format", "{{json .}}", image],
            capture_output=True,
            text=True,
            env=self._env(),
            check=False,
        )
        if completed.returncode != 0:
            return None, ()
        payload = json.loads(completed.stdout or "{}")
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        repo_digests = tuple(
            entry.split("@", 1)[1] for entry in payload.get("RepoDigests", []) or [] if "@" in entry
        )
        return payload.get("Id"), repo_digests

    def resolve_reference(self, image: str, digest: str) -> str:
        """Verify the pin and return the reference that runs exactly it.

        A published target is addressed by repository digest. A locally built
        one has no repository digest yet, so its image id is the pin. Either
        way the runner never launches a bare tag, which could be re-pointed
        between two trials of the same run.
        """

        image_id, repo_digests = self.image_digests(image)
        if image_id is None:
            raise ContainerRuntimeError(
                f"target image {image} is not present locally; pull or build it first"
            )
        if digest in repo_digests:
            return f"{image}@{digest}"
        if digest == image_id:
            return digest
        raise ContainerRuntimeError(
            f"target image {image} resolves to {image_id}, which does not match the "
            f"pinned digest {digest}"
        )

    def run(
        self,
        request: TrialRunRequest,
        *,
        on_event: Callable[[dict[str, Any]], None],
        should_cancel: Callable[[], bool],
        heartbeat: Callable[[], None],
    ) -> TrialExecution:
        validate_trial_request(request)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        # Docker must not create this nested mountpoint inside a read-only parent.
        policy_mount = request.input_dir / "policy"
        if policy_mount.is_symlink():
            raise ContainerRuntimeError("policy mountpoint must not be a symlink")
        policy_mount.mkdir(parents=True, exist_ok=True)
        # Truncating a trial id collides: two trials of the same candidate
        # differ only in their tail. Keep a readable prefix, then a digest of
        # the whole id so the name is unique as well as short.
        container = self.resource_identity(request.trial_id)["container_id"]
        argv = [
            self.binary,
            "run",
            "--rm",
            "--name",
            container,
            "--network",
            "none" if request.network == "none" else "bridge",
            "--cpus",
            str(request.limits.cpus),
            "--memory",
            f"{request.limits.memory_mb}m",
            "--pids-limit",
            "512",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=bind,source={request.input_dir},target=/input,readonly",
            "--mount",
            f"type=bind,source={request.policy_dir},target=/input/policy,readonly",
            "--mount",
            f"type=bind,source={request.output_dir},target=/output",
        ]
        for name, value in request.secrets.items():
            argv.extend(["--env", f"{name}={value}"])
        for mapping in request.extra_hosts:
            if mapping != "host.docker.internal:host-gateway":
                raise ContainerRuntimeError(f"unsupported eval container host mapping: {mapping}")
            argv.extend(["--add-host", mapping])
        argv.append(request.image_reference)

        stderr_path = request.output_dir / "container.stderr.log"
        started = time.time()
        if should_cancel():
            return TrialExecution(None, False, True, started, time.time(), "")
        supervisor = RolloutLimitSupervisor(
            RolloutLimits(request.limits.timeout_seconds, request.limits.max_output_bytes)
        )
        stderr_fd = os.open(
            stderr_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_NONBLOCK,
            0o600,
        )
        with os.fdopen(stderr_fd, "wb") as stderr_handle:
            if not stat.S_ISREG(os.fstat(stderr_handle.fileno()).st_mode):
                raise ExecutionContractError("stderr output must be a regular file")
            process = subprocess.Popen(  # noqa: S603 - fixed binary, recipe-pinned argv
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_handle,
                env=self._env(),
            )
            stop_tailing = threading.Event()
            tail_errors: queue.SimpleQueue[Exception] = queue.SimpleQueue()

            def follow_target_events() -> None:
                try:
                    _tail_events(request.output_dir / "events.jsonl", on_event, stop_tailing)
                except Exception as error:
                    # Relay across the thread boundary; never lose a target or
                    # observer failure as an unobserved background exception.
                    tail_errors.put(error)

            def check_target_events() -> None:
                try:
                    error = tail_errors.get_nowait()
                except queue.Empty:
                    return
                raise ContainerEventStreamError("target event observation failed") from error

            tail = threading.Thread(target=follow_target_events, daemon=True)
            tail.start()
            timed_out = False
            cancelled = False
            output_limit_exceeded = False
            try:
                while True:
                    check_target_events()
                    decision = supervisor.observe(output_bytes=_output_bytes(request.output_dir))
                    if decision is not None:
                        timed_out = decision.kind == RolloutLimitKind.WORK_TIME
                        output_limit_exceeded = decision.kind == RolloutLimitKind.OUTPUT_BYTES
                        # Observer failure must not prevent the stop actuator.
                        try:
                            on_event({"event": "eval.limit.stop", **decision.to_payload()})
                        finally:
                            self._kill(container, process)
                        break
                    if should_cancel():
                        cancelled = True
                        self._kill(container, process)
                        break
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        heartbeat()
                        continue
                    # A short-lived target can exceed the cap between polls.
                    decision = supervisor.observe(output_bytes=_output_bytes(request.output_dir))
                    if decision is not None:
                        timed_out = decision.kind == RolloutLimitKind.WORK_TIME
                        output_limit_exceeded = decision.kind == RolloutLimitKind.OUTPUT_BYTES
                        on_event({"event": "eval.limit.stop", **decision.to_payload()})
                    break
            finally:
                try:
                    if process.poll() is None:
                        self._kill(container, process)
                finally:
                    stop_tailing.set()
                    tail.join(timeout=2.0)
            if tail.is_alive():
                raise ContainerEventStreamError("target event drain exceeded two seconds")
            check_target_events()
        finished = time.time()
        return TrialExecution(
            exit_code=process.returncode,
            timed_out=timed_out,
            cancelled=cancelled,
            started_at=started,
            finished_at=finished,
            stderr_tail=_tail_text(stderr_path),
            output_limit_exceeded=output_limit_exceeded,
        )

    def _kill(self, container: str, process: subprocess.Popen[bytes]) -> None:
        # Removal handles both running and created-but-not-started resources.
        # A failed/ambiguous response still quarantines ownership; do not infer
        # absence from CLI exit or provider error text.
        try:
            stopped = subprocess.run(  # noqa: S603 - fixed binary, fixed argv
                [self.binary, "rm", "--force", container],
                capture_output=True,
                env=self._env(),
                check=False,
                timeout=10,
            )
            if stopped.returncode != 0:
                raise ContainerStopUnconfirmed(
                    container, self.runtime_name, StopFailureReason.COMMAND_FAILED
                )
        except subprocess.TimeoutExpired as error:
            raise ContainerStopUnconfirmed(
                container, self.runtime_name, StopFailureReason.TIMEOUT
            ) from error
        finally:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _output_bytes(root: Path, *, max_entries: int = 100_000) -> int:
    """Bounded sampled footprint, skipping symlink entries.

    This is an observed stop threshold, not a hard disk quota. Disappearing
    files are expected during a live sample; other inspection errors propagate.
    The live filesystem may change during inspection; this is not a containment
    boundary or a measurement of transient bytes created between samples.
    """
    pending = [root]
    total = 0
    entries = 0
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as children:
                for child in children:
                    entries += 1
                    if entries > max_entries:
                        raise ContainerRuntimeError("output inspection exceeded its entry bound")
                    try:
                        if child.is_dir(follow_symlinks=False):
                            pending.append(Path(child.path))
                        elif child.is_file(follow_symlinks=False):
                            total += child.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            continue
    return total


def _tail_events(
    path: Path,
    on_event: Callable[[dict[str, Any]], None],
    stop: threading.Event,
    *,
    max_record_bytes: int = 1024 * 1024,
) -> None:
    """Observe optional JSONL without silently dropping corrupt evidence.

    A file that never appears is optional. Once observed, replacement, truncation,
    malformed records, I/O errors and observer errors fail the observation. Each
    read is bounded; partial records wait for completion until the final drain.
    This is not a content-addressed evidence seal or durable custody receipt.
    """
    if type(max_record_bytes) is not int or max_record_bytes <= 0:
        raise ValueError("max_record_bytes must be a positive integer")
    offset = 0
    identity: tuple[int, int] | None = None

    def reject_constant(value: str) -> None:
        raise ValueError("nonfinite JSON value")

    while True:
        draining = stop.is_set()
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        except FileNotFoundError:
            if identity is not None:
                raise ContainerEventStreamError("target event file disappeared")
            if draining:
                return
            stop.wait(0.4)
            continue
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ContainerEventStreamError("target event path is not a regular file")
            observed_identity = (info.st_dev, info.st_ino)
            if identity is not None and observed_identity != identity:
                raise ContainerEventStreamError("target event file was replaced")
            identity = observed_identity
            if info.st_size < offset:
                raise ContainerEventStreamError("target event file was truncated")
            handle.seek(offset)
            for _ in range(256):
                line = handle.readline(max_record_bytes + 1)
                if not line:
                    if draining:
                        return
                    break
                if len(line) > max_record_bytes:
                    raise ContainerEventStreamError("target event exceeds record bound")
                if not line.endswith(b"\n"):
                    if draining:
                        raise ContainerEventStreamError("target event ends with a partial record")
                    break
                if line.strip():
                    try:
                        payload = json.loads(line.decode("utf-8"), parse_constant=reject_constant)
                    except (UnicodeDecodeError, ValueError) as error:
                        raise ContainerEventStreamError("malformed target event record") from error
                    if not isinstance(payload, dict):
                        raise ContainerEventStreamError("target event must be a JSON object")
                    on_event(payload)
                offset += len(line)
            else:
                # Bound each page, then immediately continue from its byte cursor.
                continue
        if not draining:
            stop.wait(0.4)


def _tail_text(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - limit))
        data = handle.read(limit)
    return data.decode("utf-8", errors="replace")
