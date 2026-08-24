"""Sibling trial containers on the **host** daemon (DooD, never Docker-in-Docker).

Harbor and Apex platforms start one or more short-lived containers per rollout.
Those are peers of the platform container on the same daemon, reached through a
bind-mounted socket or ``DOCKER_HOST``. There is no dockerd inside the platform.

Two invariants this module exists to hold:

**Labels.** Every sibling carries ``synth.parent=<platform id>`` and
``synth.rollout=<rollout id>``. ``synth-containers down`` reaps by that label
*before* stopping the platform. Skip it and verifier containers leak — the
Harbor analogue of leftover rust.

**Host paths.** The daemon resolves ``-v`` against the HOST filesystem. A path
that only exists in the platform container's mount namespace silently mounts an
empty directory. ``HostWorkspace`` translates: the platform writes to
``/work/rollouts/<rid>`` and hands the daemon
``<host root>/rollouts/<rid>`` for the same bytes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DEADLINE_LABEL",
    "HostWorkspace",
    "NestedError",
    "NestedTrial",
    "PARENT_LABEL",
    "ROLLOUT_LABEL",
    "TrialResult",
    "host_docker_available",
    "platform_id",
    "reap_expired",
    "reap_rollout",
]

PARENT_LABEL = "synth.parent"
ROLLOUT_LABEL = "synth.rollout"
# The wall-clock second after which this container is garbage, stamped by the
# process that started it. `timeout_seconds` below is enforced by the CLIENT: it
# kills `docker run`, not the container, so a client that is SIGKILLed (a killed
# tmux session, the OOM killer, an agent tearing down) leaves the container
# running on the daemon with nobody watching. A DEO verifier is a 45-minute
# grade that spawns its own sandbox containers, so a handful of orphans is
# enough to take a laptop down -- which is exactly what happened. The deadline
# is on the container itself so any later process can reap it without knowing
# anything about the one that started it.
DEADLINE_LABEL = "synth.deadline"

_PINNED = re.compile(r"^[A-Za-z0-9._/:-]+@sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class NestedError(RuntimeError):
    """Secret-free refusal while preparing or running a sibling trial container."""


def platform_id() -> str:
    """The launcher stamps this in. Without it we cannot label, so we refuse."""

    value = os.environ.get("SYNTH_PLATFORM_ID", "").strip()
    if not value:
        raise NestedError("nested_platform_id_missing")
    return value


def host_docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    if os.environ.get("DOCKER_HOST"):
        return True
    return any(
        Path(candidate).exists()
        for candidate in ("/var/run/docker.sock", str(Path.home() / ".docker/run/docker.sock"))
    )


@dataclass(frozen=True, slots=True)
class HostWorkspace:
    """The one directory the platform and the daemon both see, under two names.

    ``mount`` is the path inside the platform container; ``host`` is what the
    daemon must be given. The launcher sets ``SYNTH_WORKSPACE_ROOT`` and
    ``SYNTH_WORKSPACE_HOST_ROOT`` to exactly these.
    """

    mount: Path
    host: Path

    @classmethod
    def from_env(cls) -> "HostWorkspace":
        mount = os.environ.get("SYNTH_WORKSPACE_ROOT", "").strip()
        host = os.environ.get("SYNTH_WORKSPACE_HOST_ROOT", "").strip()
        if not mount or not host:
            raise NestedError("nested_workspace_not_mounted")
        return cls(mount=Path(mount), host=Path(host))

    def rollout_dir(self, rollout_id: str) -> Path:
        """Create and return the platform-side directory for one rollout."""

        if not _SAFE_NAME.fullmatch(rollout_id):
            raise NestedError(f"nested_rollout_id_invalid:{rollout_id[:40]}")
        path = self.mount / "rollouts" / rollout_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_host(self, path: Path | str) -> Path:
        """Translate a platform-side path to the host path the daemon needs."""

        resolved = Path(path)
        try:
            relative = resolved.relative_to(self.mount)
        except ValueError as exc:
            # A -v of a path outside the shared root would mount an empty dir on
            # the host. That silent-empty failure is worse than refusing.
            raise NestedError(f"nested_path_outside_workspace:{resolved}") from exc
        return self.host / relative


@dataclass(frozen=True, slots=True)
class TrialResult:
    name: str
    exit_code: int
    stdout: str
    stderr: str
    image: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True, slots=True)
class NestedTrial:
    """One ``docker run`` of a digest-pinned trial image, labelled and reapable."""

    image: str
    rollout_id: str
    command: Sequence[str] = ()
    mounts: Mapping[str, str] = field(default_factory=dict)  # host path -> container path
    environment: Mapping[str, str] = field(default_factory=dict)
    network: str | None = "none"
    workdir: str | None = None
    detach: bool = False
    timeout_seconds: float = 900.0
    allow_nonzero: bool = False
    binary: str = "docker"
    require_pinned: bool = True

    def __post_init__(self) -> None:
        if self.require_pinned and not _PINNED.fullmatch(self.image):
            # Same rule harbor_docker already enforces: a moving tag makes a
            # trial unreproducible and the receipt worthless.
            raise NestedError(f"nested_image_not_pinned:{self.image[:80]}")
        for key in self.environment:
            if not _ENV_NAME.fullmatch(key):
                raise NestedError("nested_environment_name_invalid")

    def argv(self, *, parent: str, name: str) -> list[str]:
        argv = [
            self.binary,
            "run",
            "-d" if self.detach else "--rm",
            "--name",
            name,
            "--label",
            f"{PARENT_LABEL}={parent}",
            "--label",
            f"{ROLLOUT_LABEL}={self.rollout_id}",
            "--label",
            f"{DEADLINE_LABEL}={int(time.time() + self.timeout_seconds)}",
        ]
        if self.network:
            argv.extend(["--network", self.network])
        if self.workdir:
            argv.extend(["-w", self.workdir])
        for host_path, container_path in sorted(self.mounts.items()):
            if not str(host_path).startswith("/"):
                raise NestedError(f"nested_mount_not_host_absolute:{host_path}")
            argv.extend(["-v", f"{host_path}:{container_path}"])
        for key, value in sorted(self.environment.items()):
            argv.extend(["-e", f"{key}={value}"])
        argv.append(self.image)
        argv.extend(str(item) for item in self.command)
        return argv

    def run(self, *, parent: str | None = None, name: str | None = None) -> TrialResult:
        owner = parent or platform_id()
        container = name or f"synth-trial-{owner}-{self.rollout_id}"[:60]
        argv = self.argv(parent=owner, name=container)
        try:
            completed = subprocess.run(  # noqa: S603
                argv, capture_output=True, text=True, timeout=self.timeout_seconds, check=False
            )
        except FileNotFoundError as exc:
            raise NestedError("nested_docker_missing") from exc
        except subprocess.TimeoutExpired as exc:
            subprocess.run(  # noqa: S603
                [self.binary, "rm", "-f", container], capture_output=True, check=False
            )
            raise NestedError(f"nested_trial_timeout:{container}") from exc
        result = TrialResult(
            name=container,
            exit_code=completed.returncode,
            stdout=(completed.stdout or "")[-8000:],
            stderr=(completed.stderr or "")[-4000:],
            image=self.image,
        )
        if not result.ok and not self.allow_nonzero:
            raise NestedError(f"nested_trial_failed:{container}:{result.exit_code}")
        return result


def reap_rollout(rollout_id: str, *, parent: str | None = None, binary: str = "docker") -> int:
    """Remove every sibling this rollout started. Called on rollout close."""

    owner = parent or platform_id()
    listed = subprocess.run(  # noqa: S603
        [
            binary,
            "ps",
            "-aq",
            "--filter",
            f"label={PARENT_LABEL}={owner}",
            "--filter",
            f"label={ROLLOUT_LABEL}={rollout_id}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if ids:
        subprocess.run([binary, "rm", "-f", *ids], capture_output=True, check=False)  # noqa: S603
    return len(ids)


def reap_expired(*, binary: str = "docker", now: float | None = None) -> list[str]:
    """Remove every trial container whose own deadline has passed.

    Deliberately blind to the parent: an orphan's parent is by definition gone,
    so a reaper keyed on it would never fire. A container past its deadline is
    garbage no matter who started it or whether that process is still alive.

    Safe to call when the daemon is missing or unreachable -- it returns an
    empty list rather than failing a run that has not started yet.
    """

    moment = time.time() if now is None else now
    listed = subprocess.run(  # noqa: S603
        [
            binary,
            "ps",
            "-a",
            "--filter",
            f"label={DEADLINE_LABEL}",
            "--format",
            "{{.ID}} {{.Label \"" + DEADLINE_LABEL + "\"}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return []
    expired: list[str] = []
    for line in listed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        container, raw = parts
        try:
            deadline = float(raw)
        except ValueError:
            # An unparseable deadline is not evidence of expiry. Leave it.
            continue
        if deadline <= moment:
            expired.append(container)
    if expired:
        subprocess.run(  # noqa: S603
            [binary, "rm", "-f", *expired], capture_output=True, check=False
        )
    return expired


def nested_health(*, parent: str | None = None) -> dict[str, Any]:
    """``TargetSpec.health_probe`` for a nested platform: socket + workspace.

    Fail closed. A Harbor platform without a daemon or without the host
    workspace cannot run a single trial, so it must not report ready.
    """

    payload: dict[str, Any] = {"nested": "host_docker"}
    if not host_docker_available():
        return {**payload, "status": "unhealthy", "reason": "host_docker_unavailable"}
    try:
        workspace = HostWorkspace.from_env()
    except NestedError as exc:
        return {**payload, "status": "unhealthy", "reason": str(exc)}
    try:
        owner = parent or platform_id()
    except NestedError as exc:
        return {**payload, "status": "unhealthy", "reason": str(exc)}
    probe = subprocess.run(  # noqa: S603
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return {**payload, "status": "unhealthy", "reason": "host_docker_unreachable"}
    return {
        **payload,
        "docker_ok": True,
        "daemon_version": probe.stdout.strip(),
        "platform_id": owner,
        "workspace_mount": str(workspace.mount),
        "workspace_host_root": str(workspace.host),
    }
