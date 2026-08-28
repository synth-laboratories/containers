"""Local image catalog + Docker lifecycle for eval platform containers.

Two nouns: **image** (a directory under the catalog root that fully defines a
container) and **container** (a running instance). One verb pair: **up / down**.

``up_image`` builds (if needed) and ``docker run -d``s the image, waits for
``/health``, and writes a run record. ``down_image`` reads that record, reaps
sibling trial containers labelled ``synth.parent=<id>``, then stops and removes
the platform container. Nothing else in the tree may start or stop these.

Nested platforms (Harbor, Apex) set ``nested = "host_docker"``: the launcher
bind-mounts the **host** docker socket and a **host** workspace root so the
platform can ``docker run`` siblings on the same daemon. Never Docker-in-Docker.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import tomllib

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .sdk import ContainerHandle

HTTP_TASK = "http_task"
HARBOR_ENVIRONMENT = "harbor_environment"
ROLLOUT_ENVIRONMENT = "rollout_environment"
CONTRACTS = frozenset({HTTP_TASK, HARBOR_ENVIRONMENT, ROLLOUT_ENVIRONMENT})

NESTED_HOST_DOCKER = "host_docker"

PARENT_LABEL = "synth.parent"
ROLLOUT_LABEL = "synth.rollout"
PLATFORM_LABEL = "synth.platform"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PINNED = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_PORT = 8080
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


class LaunchError(RuntimeError):
    """Secret-free refusal while resolving, building, running, or stopping an image."""


# --------------------------------------------------------------------------- spec


@dataclass(frozen=True, slots=True)
class ImageSpec:
    id: str
    contract: str
    root: Path
    dockerfile: Path
    context: Path
    image_name: str
    target_id: str | None
    required_env: tuple[str, ...]
    port: int
    pull: bool
    extra_env: dict[str, str]
    build_contexts: dict[str, Path] = field(default_factory=dict)
    build_args: dict[str, str] = field(default_factory=dict)
    nested: str | None = None
    mount_docker_socket: bool = False
    workspace_host_root: bool = False
    workspace_mount: str = "/work"
    volumes: tuple[tuple[str, str, bool], ...] = ()
    health_path: str = "/health"
    startup_timeout_seconds: float = 120.0

    @property
    def is_nested(self) -> bool:
        return self.nested == NESTED_HOST_DOCKER

    def pinned_name(self, digest: str) -> str:
        digest = digest.strip().lower()
        if not _DIGEST.fullmatch(digest):
            raise LaunchError(f"image_digest_invalid:{self.id}")
        return f"{self.image_name}@{digest}"


def default_catalog_root() -> Path | None:
    """Prefer an explicit env path, then a sibling evals checkout."""

    env = os.environ.get("SYNTH_CONTAINER_IMAGE_CATALOG", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    github_root = Path(__file__).resolve().parents[3]
    sibling = github_root / "evals" / "containers" / "images"
    if sibling.is_dir() and (sibling / "catalog.toml").is_file():
        return sibling
    cwd = Path.cwd() / "containers" / "images"
    if cwd.is_dir() and (cwd / "catalog.toml").is_file():
        return cwd.resolve()
    return None


def load_catalog(root: str | Path | None = None) -> dict[str, ImageSpec]:
    catalog_root = Path(root).expanduser().resolve() if root is not None else default_catalog_root()
    if catalog_root is None:
        raise LaunchError("container_image_catalog_missing")
    if catalog_root.is_file():
        catalog_root = catalog_root.parent
    catalog_path = catalog_root / "catalog.toml"
    if not catalog_path.is_file():
        raise LaunchError(f"container_image_catalog_missing:{catalog_root}")
    try:
        payload = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise LaunchError("container_image_catalog_invalid") from exc
    rows = payload.get("image")
    if not isinstance(rows, list) or not rows:
        raise LaunchError("container_image_catalog_empty")
    specs: dict[str, ImageSpec] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise LaunchError("container_image_catalog_invalid")
        spec = _spec_from_row(catalog_root, raw)
        if spec.id in specs:
            raise LaunchError(f"container_image_id_duplicate:{spec.id}")
        specs[spec.id] = spec
    return specs


def get_image_spec(image_id: str, *, catalog: str | Path | None = None) -> ImageSpec:
    image_id = image_id.strip()
    if not image_id:
        raise LaunchError("container_image_id_missing")
    specs = load_catalog(catalog)
    spec = specs.get(image_id)
    if spec is None:
        raise LaunchError(f"container_image_unknown:{image_id}")
    return spec


# ------------------------------------------------------------------------ backend


class DockerBackend(Protocol):
    def inspect_id(self, reference: str) -> str | None: ...

    def build(self, spec: ImageSpec, *, tag: str) -> str: ...

    def pull(self, reference: str) -> str: ...

    def run(self, *, reference: str, name: str, args: Sequence[str]) -> str: ...

    def stop(self, name: str) -> None: ...

    def logs(self, name: str, *, tail: int = 200) -> str: ...

    def container_ids(self, *, label: str) -> tuple[str, ...]: ...

    def remove(self, references: Sequence[str]) -> None: ...

    def exists(self, name: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class SubprocessDocker:
    binary: str = "docker"

    def _run(
        self, args: list[str], *, check: bool = True, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if shutil.which(self.binary) is None:
            raise LaunchError(f"docker_missing:{self.binary}")
        completed = subprocess.run(  # noqa: S603
            [self.binary, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if check and completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            tail = detail[-1] if detail else f"exit {completed.returncode}"
            raise LaunchError(f"docker_failed:{args[0]}:{tail[:400]}")
        return completed

    def inspect_id(self, reference: str) -> str | None:
        completed = self._run(["image", "inspect", "--format", "{{.Id}}", reference], check=False)
        image_id = completed.stdout.strip()
        if completed.returncode != 0 or not image_id:
            return None
        if image_id.startswith("sha256:") and _DIGEST.fullmatch(image_id):
            return image_id
        if _DIGEST.fullmatch(f"sha256:{image_id}"):
            return f"sha256:{image_id}"
        return None

    def build(self, spec: ImageSpec, *, tag: str) -> str:
        if spec.image_name.endswith(":latest") or tag.endswith(":latest"):
            raise LaunchError("container_image_latest_forbidden")
        args = ["build", "-f", str(spec.dockerfile), "-t", tag]
        for name, path in sorted(spec.build_contexts.items()):
            args.extend(["--build-context", f"{name}={path}"])
        for name, value in sorted(spec.build_args.items()):
            args.extend(["--build-arg", f"{name}={value}"])
        args.append(str(spec.context))
        env_backup = os.environ.get("DOCKER_BUILDKIT")
        if spec.build_contexts and not env_backup:
            os.environ["DOCKER_BUILDKIT"] = "1"
        try:
            self._run(args)
        finally:
            if spec.build_contexts and not env_backup:
                os.environ.pop("DOCKER_BUILDKIT", None)
        digest = self.inspect_id(tag)
        if digest is None:
            raise LaunchError(f"container_image_build_unresolved:{spec.id}")
        return digest

    def pull(self, reference: str) -> str:
        self._run(["pull", reference])
        digest = self.inspect_id(reference)
        if digest is None:
            raise LaunchError(f"container_image_pull_unresolved:{reference}")
        return digest

    def run(self, *, reference: str, name: str, args: Sequence[str]) -> str:
        completed = self._run(["run", "-d", "--name", name, *args, reference])
        container_id = completed.stdout.strip()
        if not container_id:
            raise LaunchError("container_image_run_missing_id")
        return container_id

    def stop(self, name: str) -> None:
        self._run(["stop", "-t", "5", name], check=False)
        self._run(["rm", "-f", name], check=False)

    def logs(self, name: str, *, tail: int = 200) -> str:
        logged = self._run(["logs", "--tail", str(tail), name], check=False)
        return (logged.stdout or "") + (logged.stderr or "")

    def container_ids(self, *, label: str) -> tuple[str, ...]:
        completed = self._run(["ps", "-aq", "--filter", f"label={label}"], check=False)
        return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())

    def remove(self, references: Sequence[str]) -> None:
        if not references:
            return
        self._run(["rm", "-f", *references], check=False)

    def exists(self, name: str) -> bool:
        completed = self._run(["inspect", "--format", "{{.Id}}", name], check=False)
        return completed.returncode == 0 and bool(completed.stdout.strip())


# --------------------------------------------------------------------- run record


def run_root() -> Path:
    override = os.environ.get("SYNTH_CONTAINERS_RUN_ROOT", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".synth-containers" / "run"
    return root


def _record_path(image_id: str, port: int) -> Path:
    return run_root() / f"{image_id}-{port}.json"


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    port: int
    host: str
    url: str
    container_id: str
    container_name: str
    image_name: str
    digest: str
    nested: str | None
    workspace_host_root: str | None
    started_at: float

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "port": self.port,
            "host": self.host,
            "url": self.url,
            "container_id": self.container_id,
            "container_name": self.container_name,
            "image_name": self.image_name,
            "digest": self.digest,
            "nested": self.nested,
            "workspace_host_root": self.workspace_host_root,
            "started_at": self.started_at,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "RunRecord":
        return cls(
            id=str(payload.get("id") or ""),
            port=int(payload.get("port") or 0),
            host=str(payload.get("host") or "127.0.0.1"),
            url=str(payload.get("url") or ""),
            container_id=str(payload.get("container_id") or ""),
            container_name=str(payload.get("container_name") or ""),
            image_name=str(payload.get("image_name") or ""),
            digest=str(payload.get("digest") or ""),
            nested=(str(payload["nested"]) if payload.get("nested") else None),
            workspace_host_root=(
                str(payload["workspace_host_root"]) if payload.get("workspace_host_root") else None
            ),
            started_at=float(payload.get("started_at") or 0.0),
        )


def write_run_record(record: RunRecord) -> Path:
    path = _record_path(record.id, record.port)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record.to_json(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def read_run_record(image_id: str, port: int) -> RunRecord | None:
    path = _record_path(image_id, port)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return RunRecord.from_json(payload)


def list_run_records(image_id: str | None = None) -> tuple[RunRecord, ...]:
    root = run_root()
    if not root.is_dir():
        return ()
    records: list[RunRecord] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        record = RunRecord.from_json(payload)
        if image_id and record.id != image_id:
            continue
        records.append(record)
    return tuple(records)


def delete_run_record(image_id: str, port: int) -> None:
    _record_path(image_id, port).unlink(missing_ok=True)


# ------------------------------------------------------------------------ resolve


def resolve_local_digest(
    spec: ImageSpec,
    *,
    image: str | None = None,
    build: bool = True,
    pull: bool = False,
    backend: DockerBackend | None = None,
) -> str:
    docker = backend or SubprocessDocker()
    explicit = (image or "").strip()
    reference = explicit or (
        spec.image_name if pull or spec.pull else f"{spec.image_name}:local"
    )
    if reference.endswith(":latest") or reference == "latest":
        raise LaunchError("container_image_latest_forbidden")
    if _PINNED.fullmatch(reference):
        existing = docker.inspect_id(reference)
        if existing:
            return existing
        existing = docker.inspect_id(reference.split("@", 1)[0])
        if existing and reference.endswith(existing):
            return existing
        if pull or spec.pull:
            return docker.pull(reference)
        raise LaunchError(f"container_image_not_local:{spec.id}")
    if pull or spec.pull:
        existing = docker.inspect_id(reference)
        return existing or docker.pull(reference)
    # A source-backed catalog launch must rebuild when build=True.  Reusing a
    # coincidental unqualified/:latest tag can attest one digest while running
    # another, and it disconnects the image from the validated build contexts.
    if build:
        return docker.build(spec, tag=f"{spec.image_name}:local")
    existing = docker.inspect_id(reference)
    if existing:
        return existing
    if not build:
        raise LaunchError(f"container_image_not_local:{spec.id}")
    raise LaunchError(f"container_image_not_local:{spec.id}")


def build_image(
    image_id: str,
    *,
    catalog: str | Path | None = None,
    backend: DockerBackend | None = None,
) -> str:
    spec = get_image_spec(image_id, catalog=catalog)
    docker = backend or SubprocessDocker()
    return docker.build(spec, tag=f"{spec.image_name}:local")


def _merged_env(spec: ImageSpec, env: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(spec.extra_env)
    if env:
        merged.update({key: value for key, value in env.items() if value is not None})
    missing = [
        name
        for name in spec.required_env
        if not str(merged.get(name) or os.environ.get(name) or "").strip()
    ]
    if missing:
        raise LaunchError(f"container_image_env_missing:{spec.id}:{','.join(missing)}")
    for name in spec.required_env:
        if name not in merged:
            merged[name] = os.environ[name]
    if spec.target_id:
        merged.setdefault("SYNTH_CONTAINER_TARGET", spec.target_id)
    merged.setdefault("SYNTH_CONTAINER_PORT", str(spec.port))
    merged.setdefault("SYNTH_PLATFORM_ID", spec.id)
    return merged


def _docker_host_socket() -> str:
    host = os.environ.get("DOCKER_HOST", "").strip()
    if host.startswith("unix://"):
        return host[len("unix://") :]
    if host:
        return ""
    for candidate in (
        Path.home() / ".orbstack" / "run" / "docker.sock",
        Path.home() / ".docker" / "run" / "docker.sock",
        Path("/var/run/docker.sock"),
    ):
        if candidate.exists():
            return str(candidate)
    return "/var/run/docker.sock"


def workspace_host_dir(image_id: str) -> Path:
    override = os.environ.get("SYNTH_CONTAINERS_WORKSPACE_ROOT", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".synth-containers" / "work"
    return root / image_id


def _port_is_free(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def _run_args(
    spec: ImageSpec,
    *,
    host: str,
    host_port: int,
    env: Mapping[str, str],
) -> list[str]:
    if host not in _LOOPBACK:
        raise LaunchError("container_image_bind_host_invalid")
    args = [
        "-p",
        f"{host}:{host_port}:{spec.port}",
        "--label",
        f"{PLATFORM_LABEL}={spec.id}",
        "--label",
        f"synth.image_name={spec.image_name}",
    ]
    payload = dict(env)
    if spec.is_nested:
        socket_path = _docker_host_socket()
        if socket_path:
            args.extend(["-v", f"{socket_path}:/var/run/docker.sock"])
        else:
            payload.setdefault("DOCKER_HOST", os.environ["DOCKER_HOST"])
        if spec.workspace_host_root:
            host_root = workspace_host_dir(spec.id)
            host_root.mkdir(parents=True, exist_ok=True)
            args.extend(["-v", f"{host_root}:{spec.workspace_mount}"])
            # Nested ``-v`` paths are interpreted by the HOST daemon. The platform
            # must translate its own mount path back to the host path.
            payload["SYNTH_WORKSPACE_HOST_ROOT"] = str(host_root)
            payload["SYNTH_WORKSPACE_ROOT"] = spec.workspace_mount
        payload["SYNTH_NESTED"] = NESTED_HOST_DOCKER
    for source, target, read_only in spec.volumes:
        resolved = str(Path(os.path.expandvars(source)).expanduser())
        if "$" in resolved:
            # An unexpanded variable would otherwise be taken as a relative path
            # and created as an empty directory by the daemon.
            raise LaunchError(f"container_image_volume_source_unset:{spec.id}:{source}")
        # Docker creates a missing bind source as an empty directory, which is
        # how a credential mount turns into a silently unauthenticated image.
        # A declared volume must already exist.
        if not Path(resolved).exists():
            raise LaunchError(f"container_image_volume_source_missing:{spec.id}:{resolved}")
        args.extend(["-v", f"{resolved}:{target}:ro" if read_only else f"{resolved}:{target}"])
    for key, value in sorted(payload.items()):
        args.extend(["-e", f"{key}={value}"])
    return args


# ----------------------------------------------------------------------- up/down


def _assert_serves_this_image(
    spec: ImageSpec,
    payload: object,
    *,
    url: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Refuse a URL that answers with a different platform than we just started.

    Publishing a container port does not guarantee the host address reaches it:
    a process that already holds the port keeps answering, and `docker run`
    reports success anyway. `up` would then hand back a URL serving somebody
    else's environment, and a rollout would be attributed to the wrong image.
    The platform states its target on `/health`, so compare it against the
    merged runtime env (CLI `--env` overlays catalog `extra_env`).
    """

    expected = str((env or spec.extra_env).get("SYNTH_CONTAINER_TARGET") or "").strip()
    if not expected:
        return
    served = ""
    if isinstance(payload, Mapping):
        served = str(payload.get("target") or "").strip()
    if not served or served == expected:
        return
    raise LaunchError(
        f"container_image_port_shadowed:{spec.id}:{url}:serves={served}:expected={expected}"
    )


def up_image(
    image_id: str,
    *,
    catalog: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    replace: bool = False,
    build: bool = True,
    pull: bool = False,
    image: str | None = None,
    startup_timeout_seconds: float | None = None,
    backend: DockerBackend | None = None,
) -> RunRecord:
    """Build (if needed) and ``docker run -d`` the platform image; wait for health.

    Fail closed: an unhealthy container is stopped, removed, and no record is
    written. The daemon holds the instance afterwards; this call returns.
    """

    spec = get_image_spec(image_id, catalog=catalog)
    if spec.contract == ROLLOUT_ENVIRONMENT:
        raise LaunchError(f"container_image_rollout_not_up:{spec.id}")
    if spec.contract != HTTP_TASK:
        raise LaunchError(f"container_image_not_http_task:{spec.id}:{spec.contract}")
    host_port = int(port or spec.port)
    docker = backend or SubprocessDocker()

    existing = read_run_record(spec.id, host_port)
    if existing is not None:
        if not replace:
            if docker.exists(existing.container_name):
                raise LaunchError(f"container_image_already_up:{spec.id}:{host_port}")
            delete_run_record(spec.id, host_port)
        else:
            down_image(spec.id, port=host_port, catalog=catalog, backend=docker)

    merged = _merged_env(spec, env)
    digest = resolve_local_digest(spec, image=image, build=build, pull=pull, backend=docker)
    # The daemon-resolved image config digest is authoritative. Never allow a
    # caller-supplied value to attest a different workload than the one below.
    merged["SYNTH_CONTAINER_IMAGE_DIGEST"] = digest
    reference = digest
    selected_digest = docker.inspect_id(reference)
    if selected_digest != digest:
        reference = f"{spec.image_name}:local"
        selected_digest = docker.inspect_id(reference)
    if selected_digest != digest:
        raise LaunchError(f"container_image_digest_mismatch:{spec.id}")

    name = f"synth-{spec.id}-{host_port}"
    if docker.exists(name):
        if not replace:
            raise LaunchError(f"container_image_name_taken:{name}")
        docker.stop(name)

    timeout = float(startup_timeout_seconds or spec.startup_timeout_seconds)
    args = _run_args(spec, host=host, host_port=host_port, env=merged)
    container_id = docker.run(reference=reference, name=name, args=args)

    url = f"http://{'127.0.0.1' if host in {'0.0.0.0', 'localhost'} else host}:{host_port}"
    handle = ContainerHandle(url=url, stop=lambda: docker.stop(name), log_reader=lambda: docker.logs(name))
    deadline = time.monotonic() + timeout
    last_error = "unhealthy"
    while time.monotonic() < deadline:
        if not docker.exists(name):
            raise LaunchError(f"container_image_exited:{spec.id}")
        try:
            payload = handle.health(timeout_seconds=2.0)
            _assert_serves_this_image(spec, payload, url=url, env=merged)
            record = RunRecord(
                id=spec.id,
                port=host_port,
                host=host,
                url=url,
                container_id=container_id,
                container_name=name,
                image_name=spec.image_name,
                digest=digest,
                nested=spec.nested,
                workspace_host_root=(
                    str(workspace_host_dir(spec.id))
                    if spec.is_nested and spec.workspace_host_root
                    else None
                ),
                started_at=time.time(),
            )
            write_run_record(record)
            return record
        except LaunchError:
            # A shadowed port never heals by waiting, and retrying would bury
            # the real reason under a generic `unhealthy` timeout.
            docker.stop(name)
            raise
        except Exception as exc:  # noqa: BLE001 — health wait is fail-closed at timeout
            last_error = type(exc).__name__
            time.sleep(0.25)
    tail = docker.logs(name, tail=40)
    docker.stop(name)
    raise LaunchError(
        f"container_image_unhealthy:{spec.id}:{last_error}:{tail.strip().splitlines()[-1][:200] if tail.strip() else 'no logs'}"
    )


def reap_children(
    image_id: str,
    *,
    backend: DockerBackend | None = None,
) -> tuple[str, ...]:
    """Remove sibling trial containers this platform started (``synth.parent``)."""

    docker = backend or SubprocessDocker()
    ids = docker.container_ids(label=f"{PARENT_LABEL}={image_id}")
    if ids:
        docker.remove(ids)
    return ids


def down_image(
    image_id: str,
    *,
    port: int | None = None,
    catalog: str | Path | None = None,
    backend: DockerBackend | None = None,
) -> bool:
    """Reap labelled siblings, stop + remove the platform, drop the run record.

    Returns True when something was stopped. Down with no record and no matching
    container is a no-op success.
    """

    docker = backend or SubprocessDocker()
    records: tuple[RunRecord, ...]
    if port is None:
        records = list_run_records(image_id)
    else:
        record = read_run_record(image_id, int(port))
        records = (record,) if record is not None else ()

    reaped = reap_children(image_id, backend=docker)
    stopped = bool(reaped)

    names: list[tuple[str, int]] = [(record.container_name, record.port) for record in records]
    if not names:
        try:
            spec = get_image_spec(image_id, catalog=catalog)
            default_name = f"synth-{spec.id}-{port or spec.port}"
        except LaunchError:
            default_name = f"synth-{image_id}-{port or _DEFAULT_PORT}"
        if docker.exists(default_name):
            names.append((default_name, int(port or _DEFAULT_PORT)))

    for name, record_port in names:
        if docker.exists(name):
            docker.stop(name)
            stopped = True
        if docker.exists(name):
            raise LaunchError(f"container_image_down_failed:{image_id}:{name}")
        delete_run_record(image_id, record_port)

    leftovers = docker.container_ids(label=f"{PARENT_LABEL}={image_id}")
    if leftovers:
        raise LaunchError(f"container_image_children_leaked:{image_id}:{len(leftovers)}")
    return stopped


def handle_for(record: RunRecord, *, backend: DockerBackend | None = None) -> ContainerHandle:
    docker = backend or SubprocessDocker()
    return ContainerHandle(
        url=record.url,
        stop=lambda: down_image(record.id, port=record.port, backend=docker),
        log_reader=lambda: docker.logs(record.container_name),
    )


def logs_image(
    image_id: str,
    *,
    port: int | None = None,
    tail: int = 200,
    backend: DockerBackend | None = None,
) -> str:
    docker = backend or SubprocessDocker()
    records = list_run_records(image_id) if port is None else ()
    if port is not None:
        record = read_run_record(image_id, int(port))
        records = (record,) if record is not None else ()
    if not records:
        raise LaunchError(f"container_image_not_running:{image_id}")
    return "\n".join(docker.logs(record.container_name, tail=tail) for record in records)


def serve_image(
    image_id: str | None = None,
    *,
    image: str | None = None,
    catalog: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    host: str = "127.0.0.1",
    port: int | None = None,
    startup_timeout_seconds: float | None = None,
    build: bool = True,
    pull: bool = False,
    replace: bool = True,
    backend: DockerBackend | None = None,
) -> ContainerHandle:
    """``up_image`` with a handle whose ``down()`` is ``down_image``.

    Kept for callers that want scoped ownership (tests, one-shot proofs).
    Operators use ``up`` / ``down``.
    """

    if not image_id and not image:
        raise LaunchError("container_image_ref_missing")
    if not image_id:
        image_id = _image_id_for_reference(image or "", catalog)
    record = up_image(
        image_id,
        catalog=catalog,
        env=env,
        host=host,
        port=port,
        replace=replace,
        build=build,
        pull=pull,
        image=image,
        startup_timeout_seconds=startup_timeout_seconds,
        backend=backend,
    )
    return handle_for(record, backend=backend)


def _image_id_for_reference(image: str, catalog: str | Path | None) -> str:
    image = image.strip()
    if not image:
        raise LaunchError("container_image_ref_missing")
    name = image.split("@", 1)[0]
    for spec in load_catalog(catalog).values():
        if spec.image_name == name or spec.id == name:
            return spec.id
    raise LaunchError(f"container_image_unknown:{name}")


def catalog_payload(root: str | Path | None = None) -> dict[str, Any]:
    specs = load_catalog(root)
    return {
        "catalog": str(next(iter(specs.values())).root.parent if specs else root or ""),
        "images": [
            {
                "id": spec.id,
                "contract": spec.contract,
                "image_name": spec.image_name,
                "target_id": spec.target_id,
                "port": spec.port,
                "pull": spec.pull,
                "nested": spec.nested,
                "required_env": list(spec.required_env),
            }
            for spec in specs.values()
        ],
    }


def status_payload(catalog: str | Path | None = None) -> dict[str, Any]:
    del catalog
    return {"running": [record.to_json() for record in list_run_records()]}


# -------------------------------------------------------------------- toml parse


def _resolve_declared_path(catalog_root: Path, directory: Path, raw: str, *, image_id: str) -> Path:
    candidate = Path(os.path.expandvars(raw)).expanduser()
    resolved = candidate if candidate.is_absolute() else (directory / candidate)
    resolved = resolved.resolve()
    if not resolved.exists():
        raise LaunchError(f"container_image_build_context_missing:{image_id}:{raw}")
    del catalog_root
    return resolved


def _spec_from_row(catalog_root: Path, raw: Mapping[str, Any]) -> ImageSpec:
    image_id = str(raw.get("id") or "").strip()
    relative = str(raw.get("path") or image_id).strip()
    if not image_id or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise LaunchError("container_image_id_invalid")
    directory = (catalog_root / relative).resolve()
    try:
        directory.relative_to(catalog_root)
    except ValueError as exc:
        raise LaunchError(f"container_image_path_invalid:{image_id}") from exc
    overlay: dict[str, Any] = dict(raw)
    image_toml = directory / "image.toml"
    if image_toml.is_file():
        try:
            overlay.update(tomllib.loads(image_toml.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise LaunchError(f"container_image_toml_invalid:{image_id}") from exc
    contract = str(overlay.get("contract") or HTTP_TASK).strip()
    if contract not in CONTRACTS:
        raise LaunchError(f"container_image_contract_invalid:{image_id}")
    if overlay.get("command"):
        raise LaunchError(f"container_image_command_forbidden:{image_id}")
    dockerfile_name = str(overlay.get("dockerfile") or "Dockerfile").strip()
    dockerfile = (directory / dockerfile_name).resolve()
    if not dockerfile.is_file():
        shared = (catalog_root / dockerfile_name).resolve()
        if shared.is_file():
            dockerfile = shared
        else:
            raise LaunchError(f"container_image_dockerfile_missing:{image_id}")
    try:
        dockerfile.relative_to(catalog_root)
    except ValueError as exc:
        raise LaunchError(f"container_image_dockerfile_invalid:{image_id}") from exc
    context_name = str(overlay.get("context") or ".").strip() or "."
    context = (directory / context_name).resolve()
    if not context.exists():
        raise LaunchError(f"container_image_context_missing:{image_id}")
    try:
        context.relative_to(catalog_root)
    except ValueError as exc:
        raise LaunchError(f"container_image_context_invalid:{image_id}") from exc
    image_name = str(overlay.get("image_name") or f"evals-{image_id}").strip()
    if not image_name or image_name.endswith(":latest"):
        raise LaunchError(f"container_image_name_invalid:{image_id}")
    extra = overlay.get("extra_env") or {}
    if not isinstance(extra, dict):
        raise LaunchError(f"container_image_extra_env_invalid:{image_id}")
    extra_env = {str(key): str(value) for key, value in extra.items()}
    target_id = str(overlay.get("target_id") or "").strip() or None
    if target_id is None:
        target_id = extra_env.get("SYNTH_CONTAINER_TARGET") or None
    required = overlay.get("required_env") or []
    if not isinstance(required, list) or any(
        not _ENV_NAME.fullmatch(str(item).strip()) for item in required
    ):
        raise LaunchError(f"container_image_required_env_invalid:{image_id}")
    port = int(overlay.get("port") or _DEFAULT_PORT)
    if port <= 0 or port > 65535:
        raise LaunchError(f"container_image_port_invalid:{image_id}")

    raw_contexts = overlay.get("build_contexts") or {}
    if not isinstance(raw_contexts, dict):
        raise LaunchError(f"container_image_build_contexts_invalid:{image_id}")
    build_contexts = {
        str(name): _resolve_declared_path(catalog_root, directory, str(value), image_id=image_id)
        for name, value in raw_contexts.items()
    }
    raw_build_args = overlay.get("build_args") or {}
    if not isinstance(raw_build_args, dict):
        raise LaunchError(f"container_image_build_args_invalid:{image_id}")

    nested = str(overlay.get("nested") or "").strip() or None
    if nested is not None and nested != NESTED_HOST_DOCKER:
        raise LaunchError(f"container_image_nested_invalid:{image_id}")
    docker_block = overlay.get("docker") or {}
    if not isinstance(docker_block, dict):
        raise LaunchError(f"container_image_docker_block_invalid:{image_id}")
    mount_socket = bool(docker_block.get("mount_docker_socket") or overlay.get("mount_docker_socket"))
    workspace_root = bool(docker_block.get("workspace_host_root") or overlay.get("workspace_host_root"))
    if nested == NESTED_HOST_DOCKER:
        mount_socket = True if docker_block.get("mount_docker_socket") is None else mount_socket
        workspace_root = True if docker_block.get("workspace_host_root") is None else workspace_root
    elif mount_socket or workspace_root:
        raise LaunchError(f"container_image_nested_required:{image_id}")

    raw_volumes = overlay.get("volumes") or []
    if not isinstance(raw_volumes, list):
        raise LaunchError(f"container_image_volumes_invalid:{image_id}")
    volumes: list[tuple[str, str, bool]] = []
    for item in raw_volumes:
        if not isinstance(item, dict) or not item.get("source") or not item.get("target"):
            raise LaunchError(f"container_image_volumes_invalid:{image_id}")
        # `$VAR` in a source lets a catalog name a location that differs per
        # operator (a corpus checkout, say) without writing one machine's path
        # into the repo. It is expanded and checked at launch, not here: reading
        # the catalog must not depend on the reader's environment.
        source = str(item["source"])
        if (
            not source.startswith("/")
            and not source.startswith("~")
            and not source.startswith("$")
            and "/" in source
        ):
            source = str((directory / source).resolve())
        volumes.append((source, str(item["target"]), bool(item.get("read_only", False))))

    return ImageSpec(
        id=image_id,
        contract=contract,
        root=directory,
        dockerfile=dockerfile,
        context=context,
        image_name=image_name,
        target_id=target_id,
        required_env=tuple(str(item).strip() for item in required),
        port=port,
        pull=bool(overlay.get("pull") or False),
        extra_env=extra_env,
        build_contexts=build_contexts,
        build_args={str(k): str(v) for k, v in raw_build_args.items()},
        nested=nested,
        mount_docker_socket=mount_socket,
        workspace_host_root=workspace_root,
        workspace_mount=str(overlay.get("workspace_mount") or docker_block.get("workspace_mount") or "/work"),
        volumes=tuple(volumes),
        health_path=str(overlay.get("health_path") or "/health"),
        startup_timeout_seconds=float(overlay.get("startup_timeout_seconds") or 120.0),
    )


def dumps_catalog(specs: Mapping[str, ImageSpec]) -> str:
    return json.dumps(
        [
            {
                "id": spec.id,
                "contract": spec.contract,
                "image_name": spec.image_name,
                "target_id": spec.target_id,
            }
            for spec in specs.values()
        ],
        indent=2,
        sort_keys=True,
    )
