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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ...event_log import RolloutEventLog
from ..state import CompatPlatform, RolloutPin

DOCKER_ENVIRONMENT = "env:harbor_docker"
PUBLIC_INSTRUCTION = "Write the word ok to /workspace/answer.txt"
PUBLIC_TESTS = "tests/test.sh"
PUBLIC_IMAGE = "alpine:3.20"
_STDOUT_LIMIT = 4096
_NAME_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclass(frozen=True)
class DockerExecution:
    role: str
    exit_code: int
    stdout: str
    name: str


class DockerRunError(RuntimeError):
    """Secret-free failure from a docker execution. Message is an error_type."""


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
    volumes: Mapping[str, str],
    name: str,
    timeout_seconds: float = 120.0,
) -> DockerExecution:
    """One short-lived `docker run --rm`. Tests may replace this."""
    argv = ["docker", "run", "--rm", "--network", "none", "--name", name]
    for container_path, host_path in volumes.items():
        argv.extend(["-v", f"{host_path}:{container_path}"])
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
    if completed.returncode != 0:
        raise DockerRunError("harbor_docker_run_failed")
    return DockerExecution(
        role=role,
        exit_code=completed.returncode,
        stdout=_clip_stdout(completed.stdout),
        name=name,
    )


def run_docker_trial(platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
    del platform
    if not docker_runtime_available():
        _fail(pin, log, "harbor_docker_unavailable")
        return
    workspace_root = tempfile.mkdtemp(prefix="harbor-docker-ws-")
    logs_root = tempfile.mkdtemp(prefix="harbor-docker-logs-")
    try:
        _run_public_fixture(pin, log, workspace_root=workspace_root, logs_root=logs_root)
    except DockerRunError as exc:
        _fail(pin, log, "harbor_docker_run_failed", error_type=str(exc) or "harbor_docker_run_failed")
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)
        shutil.rmtree(logs_root, ignore_errors=True)


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
