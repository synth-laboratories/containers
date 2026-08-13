"""Harbor Docker fold. Real TB-shaped trial when the daemon is present.

`env:harbor_docker` must not fall through to the in-process fixture. Agent and
verifier are distinct `docker run --rm` executions. Native `reward.txt` is the
verifier file; `/reward` agrees. Missing file stays null — never coerce to 0.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...event_log import RolloutEventLog
from ..state import CompatPlatform, RolloutPin

DOCKER_ENVIRONMENT = "env:harbor_docker"
PUBLIC_INSTRUCTION = "Write the word ok to /workspace/answer.txt"
PUBLIC_TESTS = "tests/test.sh"
PUBLIC_IMAGE = "alpine:3.20"
BUNDLE_SCHEMA = "synth.harbor-pinned-bundle.v1"
RUNTIME_BUNDLE_KEY = "harbor_pinned_bundle"
_STDOUT_LIMIT = 4096
_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")
_IMAGE_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class DockerExecution:
    role: str
    exit_code: int
    stdout: str
    name: str


@dataclass(frozen=True)
class DockerVolume:
    """One explicit Docker bind. Task authority is always mounted read-only."""

    host_path: str
    read_only: bool = False


@dataclass(frozen=True)
class HarborPinnedBundle:
    """Validated, immutable-by-digest Harbor task bundle.

    The digest is supplied out of band by the launcher.  A bundle may not pin
    itself by editing its own manifest, and mutable image tags are refused.
    """

    root: Path
    bundle_id: str
    bundle_digest: str
    image: str
    instruction_path: str
    agent_command: tuple[str, ...]
    verifier_command: tuple[str, ...]
    task_tree_path: str
    task_tree_mount: str
    task_tree_digest: str
    required_paths: tuple[str, ...]
    reward_path: str
    tests: str

    @classmethod
    def from_runtime_config(cls, config: Mapping[str, Any]) -> "HarborPinnedBundle | None":
        raw = config.get(RUNTIME_BUNDLE_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise HarborBundleError("harbor_bundle_config_invalid")
        root_raw = str(raw.get("root") or "").strip()
        expected = str(raw.get("digest") or "").strip().lower()
        if not root_raw or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            raise HarborBundleError("harbor_bundle_pin_required")
        return cls.from_directory(Path(root_raw), expected_digest=expected)

    @classmethod
    def from_directory(
        cls,
        root: Path,
        *,
        expected_digest: str,
    ) -> "HarborPinnedBundle":
        root = root.expanduser().resolve()
        manifest_path = root / "bundle.json"
        if not root.is_dir() or not manifest_path.is_file():
            raise HarborBundleError("harbor_bundle_missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HarborBundleError("harbor_bundle_manifest_invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != BUNDLE_SCHEMA:
            raise HarborBundleError("harbor_bundle_manifest_invalid")

        actual_digest = compute_bundle_digest(root)
        if actual_digest != expected_digest:
            raise HarborBundleError("harbor_bundle_digest_mismatch")

        bundle_id = _required_text(manifest, "bundle_id", "harbor_bundle_manifest_invalid")
        image = _required_text(manifest, "image", "harbor_bundle_image_invalid")
        if not _IMAGE_DIGEST.fullmatch(image):
            raise HarborBundleError("harbor_bundle_image_unpinned")
        instruction_path = _required_relative_file(
            root, manifest, "instruction_path", "harbor_bundle_instruction_missing"
        )
        agent = manifest.get("agent")
        verifier = manifest.get("verifier")
        task_tree = manifest.get("task_tree")
        if not isinstance(agent, Mapping) or not isinstance(verifier, Mapping):
            raise HarborBundleError("harbor_bundle_roles_invalid")
        if not isinstance(task_tree, Mapping):
            raise HarborBundleError("harbor_bundle_task_tree_invalid")
        agent_command = _command(agent.get("command"), "harbor_bundle_agent_invalid")
        verifier_command = _command(verifier.get("command"), "harbor_bundle_verifier_invalid")
        reward_path = _safe_relative(
            str(verifier.get("reward_path") or "verifier/reward.txt"),
            "harbor_bundle_reward_path_invalid",
        )
        tests = str(verifier.get("tests") or "tests/test.sh").strip()
        if not tests or len(tests) > 256:
            raise HarborBundleError("harbor_bundle_verifier_invalid")

        task_tree_path = _safe_relative(
            str(task_tree.get("path") or ""), "harbor_bundle_task_tree_invalid"
        )
        task_root = _within(root, task_tree_path, "harbor_bundle_task_tree_invalid")
        if not task_root.is_dir():
            raise HarborBundleError("harbor_bundle_task_tree_missing")
        task_tree_mount = str(task_tree.get("mount") or "").strip()
        if not task_tree_mount.startswith("/workspace/gamebench/tasks/"):
            raise HarborBundleError("harbor_bundle_task_tree_mount_invalid")
        if ":" in task_tree_mount or ".." in Path(task_tree_mount).parts:
            raise HarborBundleError("harbor_bundle_task_tree_mount_invalid")
        expected_tree = str(task_tree.get("digest") or "").strip().lower()
        actual_tree = compute_tree_digest(task_root)
        if expected_tree != actual_tree:
            raise HarborBundleError("harbor_bundle_task_tree_digest_mismatch")
        raw_required = manifest.get("required_paths")
        if not isinstance(raw_required, list) or not raw_required:
            raise HarborBundleError("harbor_bundle_required_paths_missing")
        required_paths: list[str] = []
        for raw_path in raw_required:
            relative = _safe_relative(str(raw_path or ""), "harbor_bundle_required_path_invalid")
            required = _within(root, relative, "harbor_bundle_required_path_invalid")
            if not required.is_file() or required.is_symlink():
                # A missing GameBench runner is unavailable evidence, not a
                # verifier-authored zero. Refuse the bundle before Docker.
                raise HarborBundleError("harbor_bundle_required_path_missing")
            required_paths.append(relative)

        return cls(
            root=root,
            bundle_id=bundle_id,
            bundle_digest=actual_digest,
            image=image,
            instruction_path=instruction_path,
            agent_command=agent_command,
            verifier_command=verifier_command,
            task_tree_path=task_tree_path,
            task_tree_mount=task_tree_mount,
            task_tree_digest=actual_tree,
            required_paths=tuple(required_paths),
            reward_path=reward_path,
            tests=tests,
        )


class DockerRunError(RuntimeError):
    """Secret-free failure from a docker execution. Message is an error_type."""


class HarborBundleError(RuntimeError):
    """Secret-free bundle validation refusal."""


def docker_runtime_available() -> bool:
    if shutil.which("docker") is None:
        return False
    if os.environ.get("DOCKER_HOST"):
        return True
    socket_paths = (
        Path("/var/run/docker.sock"),
        Path.home() / ".docker" / "run" / "docker.sock",
    )
    return any(path.exists() for path in socket_paths)


def execute_docker_role(
    *,
    role: str,
    image: str,
    command: list[str],
    volumes: Mapping[str, str | DockerVolume],
    name: str,
    environment: Mapping[str, str] | None = None,
    allow_nonzero: bool = False,
    timeout_seconds: float = 120.0,
) -> DockerExecution:
    """One short-lived `docker run --rm`. Tests may replace this."""
    argv = ["docker", "run", "--rm", "--network", "none", "--name", name]
    for key, value in sorted((environment or {}).items()):
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise DockerRunError("harbor_docker_environment_invalid")
        argv.extend(["-e", f"{key}={value}"])
    for container_path, mount in volumes.items():
        if isinstance(mount, DockerVolume):
            host_path = mount.host_path
            suffix = ":ro" if mount.read_only else ""
        else:
            host_path = mount
            suffix = ""
        argv.extend(["-v", f"{host_path}:{container_path}{suffix}"])
    argv.append(image)
    argv.extend(command)
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DockerRunError("harbor_docker_run_failed") from exc
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        raise DockerRunError("harbor_docker_run_failed") from exc
    if completed.returncode != 0 and not allow_nonzero:
        raise DockerRunError("harbor_docker_run_failed")
    return DockerExecution(
        role=role,
        exit_code=completed.returncode,
        stdout=_clip_stdout(completed.stdout),
        name=name,
    )


def run_docker_trial(platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
    try:
        bundle = HarborPinnedBundle.from_runtime_config(platform.runtime_config)
    except HarborBundleError as exc:
        _fail(pin, log, str(exc) or "harbor_bundle_invalid")
        return
    if not docker_runtime_available():
        _fail(pin, log, "harbor_docker_unavailable")
        return
    workspace_root = tempfile.mkdtemp(prefix="harbor-docker-ws-")
    logs_root = tempfile.mkdtemp(prefix="harbor-docker-logs-")
    try:
        if bundle is None:
            _run_public_fixture(pin, log, workspace_root=workspace_root, logs_root=logs_root)
        else:
            _run_pinned_bundle(
                pin,
                log,
                bundle=bundle,
                workspace_root=workspace_root,
                logs_root=logs_root,
            )
    except DockerRunError as exc:
        _fail(
            pin, log, "harbor_docker_run_failed", error_type=str(exc) or "harbor_docker_run_failed"
        )
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(logs_root, ignore_errors=True)


def _run_pinned_bundle(
    pin: RolloutPin,
    log: RolloutEventLog,
    *,
    bundle: HarborPinnedBundle,
    workspace_root: str,
    logs_root: str,
) -> None:
    agent_name = _container_name("harbor-agent", pin.rollout_id)
    verifier_name = _container_name("harbor-verifier", pin.rollout_id)
    instruction = _clip_stdout((bundle.root / bundle.instruction_path).read_text(encoding="utf-8"))
    task_tree_host = str(bundle.root / bundle.task_tree_path)
    shared_volumes: dict[str, str | DockerVolume] = {
        "/workspace": workspace_root,
        "/harbor/bundle": DockerVolume(str(bundle.root), read_only=True),
        bundle.task_tree_mount: DockerVolume(task_tree_host, read_only=True),
    }

    log.append(
        "trial.planned",
        {
            "instruction": instruction,
            "tests": bundle.tests,
            "bundle_id": bundle.bundle_id,
            "bundle_digest": bundle.bundle_digest,
            "task_tree": {
                "mount": bundle.task_tree_mount,
                "digest": bundle.task_tree_digest,
            },
        },
    )
    log.append(
        "trial.launched",
        {
            "sandbox": DOCKER_ENVIRONMENT,
            "container_id": agent_name,
            "bundle_digest": bundle.bundle_digest,
        },
    )
    log.append("span.agent.opened", {"role": "agent", "execution": "distinct"})
    agent = execute_docker_role(
        role="agent",
        image=bundle.image,
        command=list(bundle.agent_command),
        volumes=shared_volumes,
        name=agent_name,
        environment={
            "SYNTH_POLICY_HARNESS": str(pin.policy_ref.get("harness") or ""),
            "SYNTH_POLICY_CONFIG": str(pin.policy_ref.get("config") or ""),
        },
    )
    log.append("tools", {"name": "bundle_agent", "stdout": agent.stdout, "execution": agent.name})
    log.append("stdout", {"text": agent.stdout})
    log.append("span.agent.closed", {"role": "agent"})

    log.append("span.verifier.opened", {"role": "verifier", "execution": "distinct"})
    verifier_volumes = dict(shared_volumes)
    verifier_volumes["/logs"] = logs_root
    verifier = execute_docker_role(
        role="verifier",
        image=bundle.image,
        command=list(bundle.verifier_command),
        volumes=verifier_volumes,
        name=verifier_name,
        allow_nonzero=True,
    )
    reward = _read_reward_txt(Path(logs_root) / bundle.reward_path)
    if reward is None:
        log.append("span.verifier.closed", {"role": "verifier", "status": "failed"})
        _fail(pin, log, "harbor_docker_reward_missing", error_type="harbor_docker_reward_missing")
        return
    log.append(
        "verifier",
        {
            "script": bundle.tests,
            "reward.txt": reward,
            "bundle_digest": bundle.bundle_digest,
            "task_tree_digest": bundle.task_tree_digest,
            "exit_code": verifier.exit_code,
        },
    )
    log.append("span.verifier.closed", {"role": "verifier"})
    pin.native_script_reward = reward
    pin.status = "completed"
    pin.terminal = True
    pin.usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    log.append("status", {"status": "completed"})
    log.mark_closed()


def _run_public_fixture(
    pin: RolloutPin,
    log: RolloutEventLog,
    *,
    workspace_root: str,
    logs_root: str,
) -> None:
    agent_name = _container_name("harbor-agent", pin.rollout_id)
    verifier_name = _container_name("harbor-verifier", pin.rollout_id)
    volumes = {"/workspace": workspace_root, "/logs": logs_root}

    log.append(
        "trial.planned",
        {"instruction": PUBLIC_INSTRUCTION, "tests": PUBLIC_TESTS},
    )
    log.append(
        "trial.launched",
        {"sandbox": DOCKER_ENVIRONMENT, "container_id": agent_name},
    )

    log.append("span.agent.opened", {"role": "agent", "execution": "distinct"})
    agent = execute_docker_role(
        role="agent",
        image=PUBLIC_IMAGE,
        command=[
            "sh",
            "-c",
            "echo ok > /workspace/answer.txt && cat /workspace/answer.txt",
        ],
        volumes={"/workspace": workspace_root},
        name=agent_name,
    )
    log.append("tools", {"name": "sh", "stdout": agent.stdout, "execution": agent.name})
    log.append("stdout", {"text": agent.stdout})
    log.append("span.agent.closed", {"role": "agent"})

    log.append("span.verifier.opened", {"role": "verifier", "execution": "distinct"})
    execute_docker_role(
        role="verifier",
        image=PUBLIC_IMAGE,
        command=[
            "sh",
            "-c",
            "mkdir -p /logs/verifier; "
            "if grep -qx ok /workspace/answer.txt; then echo 1.0 > /logs/verifier/reward.txt; "
            "else echo 0.0 > /logs/verifier/reward.txt; fi",
        ],
        volumes=volumes,
        name=verifier_name,
    )
    reward = _read_reward_txt(Path(logs_root) / "verifier" / "reward.txt")
    if reward is None:
        log.append("span.verifier.closed", {"role": "verifier", "status": "failed"})
        _fail(pin, log, "harbor_docker_reward_missing", error_type="harbor_docker_reward_missing")
        return
    log.append("verifier", {"script": PUBLIC_TESTS, "reward.txt": reward})
    log.append("span.verifier.closed", {"role": "verifier"})

    pin.native_script_reward = reward
    pin.status = "completed"
    pin.terminal = True
    pin.usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    log.append("status", {"status": "completed"})
    log.mark_closed()


def _read_reward_txt(path: Path) -> float | None:
    """Parse verifier-authored reward.txt. Missing/unparseable stays null, never 0."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    token = text.split()[0]
    try:
        value = float(token)
    except ValueError:
        return None
    if value != value:  # NaN
        return None
    return value


def compute_bundle_digest(root: Path) -> str:
    """Content pin for every regular file in a Harbor bundle."""
    return compute_tree_digest(root)


def compute_tree_digest(root: Path) -> str:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise HarborBundleError("harbor_bundle_tree_missing")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise HarborBundleError("harbor_bundle_symlink_refused")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        try:
            size = path.stat().st_size
            digest.update(size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise HarborBundleError("harbor_bundle_unreadable") from exc
    return f"sha256:{digest.hexdigest()}"


def _required_text(manifest: Mapping[str, Any], field: str, code: str) -> str:
    value = str(manifest.get(field) or "").strip()
    if not value or len(value) > 512:
        raise HarborBundleError(code)
    return value


def _safe_relative(value: str, code: str) -> str:
    raw = str(value or "").strip()
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or ":" in raw:
        raise HarborBundleError(code)
    return path.as_posix()


def _within(root: Path, relative: str, code: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HarborBundleError(code) from exc
    return target


def _required_relative_file(
    root: Path,
    manifest: Mapping[str, Any],
    field: str,
    code: str,
) -> str:
    relative = _safe_relative(str(manifest.get(field) or ""), code)
    target = _within(root, relative, code)
    if not target.is_file() or target.is_symlink():
        raise HarborBundleError(code)
    return relative


def _command(value: Any, code: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 64:
        raise HarborBundleError(code)
    command = tuple(str(item) for item in value)
    if any(not item or "\x00" in item or len(item) > 4096 for item in command):
        raise HarborBundleError(code)
    return command


def _clip_stdout(raw: str | None) -> str:
    text = (raw or "").replace("\x00", "")
    if len(text) > _STDOUT_LIMIT:
        return text[:_STDOUT_LIMIT]
    return text


def _container_name(prefix: str, rollout_id: str) -> str:
    slug = _NAME_SAFE.sub("-", rollout_id).strip("-.")[:24] or "trial"
    return f"{prefix}-{slug}-{uuid.uuid4().hex[:8]}"


def _fail(
    pin: RolloutPin,
    log: RolloutEventLog,
    reason: str,
    *,
    error_type: str | None = None,
) -> None:
    pin.status = "failed"
    pin.terminal = True
    pin.native_script_reward = None
    pin.usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    payload: dict[str, str] = {"status": "failed", "reason": reason}
    if error_type is not None:
        payload["error_type"] = error_type
    if not log.closed:
        log.append("status", payload)
        log.mark_closed()
