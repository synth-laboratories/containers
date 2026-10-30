"""CardCodeBench producer for Rhodes' native Harbor phases (SYN-4843).

Consumes a sanitized public task and a separately assembled portable verifier.
Private verifier bytes travel only in the post-agent phase payload, never in
an image build context. Grading and compilation remain inside Rhodes sandboxes.
"""

from __future__ import annotations

import base64
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rhodes_harbor_daytona import _tar_b64


@dataclass(frozen=True)
class CardCodeTask:
    task_id: str
    image: str
    instruction: bytes
    public_files: dict[str, bytes]
    verifier_files: dict[str, bytes]
    agent_user: str
    agent_timeout_s: int
    verifier_timeout_s: int
    cpu_cores: int
    memory_mb: int


def _files(root: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("cardcode_bundle_symlink")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def load_task(
    public_task: Path, verifier: Path, *, image: str, verifier_sha256: str, agent_user: str
) -> CardCodeTask:
    """Verifier digest is the canonical file manifest, independent of tar metadata."""
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("cardcode_worker_image_unpinned")
    if not re.fullmatch(r"[a-z_][a-z0-9_-]*", agent_user) or agent_user == "root":
        raise ValueError("cardcode_agent_user_must_be_unprivileged")
    config = tomllib.loads((public_task / "task.toml").read_text())
    public = _files(public_task / "environment" / "workspace")
    if not public or any(
        set(Path(name).parts) & {".private", "tests", "solution", "reference", "cases.json"}
        for name in public
    ):
        raise ValueError("cardcode_public_context_contains_private_paths")
    private = _files(verifier)
    if "test.sh" not in private:
        raise ValueError("cardcode_portable_verifier_missing")
    digest = bundle_digest(private)
    if digest != verifier_sha256:
        raise ValueError("cardcode_verifier_digest_mismatch")
    environment = config.get("environment", {})
    verifier_environment = config.get("verifier", {}).get("environment", {})
    return CardCodeTask(
        task_id=str(config["task"]["name"]),
        image=image,
        instruction=(public_task / "instruction.md").read_bytes(),
        public_files=public,
        verifier_files=private,
        agent_user=agent_user,
        agent_timeout_s=int(config["agent"]["timeout_sec"]),
        verifier_timeout_s=int(config["verifier"]["timeout_sec"]),
        # Agent and verifier share one declared deployment envelope. Reserve
        # the larger requirement so isolated grading never receives less than
        # the task's verifier environment requests.
        cpu_cores=max(int(environment["cpus"]), int(verifier_environment.get("cpus", 0))),
        memory_mb=max(
            int(environment["memory_mb"]), int(verifier_environment.get("memory_mb", 0))
        ),
    )


def bundle_digest(files: dict[str, bytes]) -> str:
    import json

    manifest = {name: hashlib.sha256(body).hexdigest() for name, body in sorted(files.items())}
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_payload(task: CardCodeTask, *, task_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    """Public build and private grading payloads remain separate at every step."""
    if agent.get("name") not in {"codex", "fixed_solution"}:
        raise ValueError("cardcode_agent_unsupported")
    if min(task.agent_timeout_s, task.verifier_timeout_s, task.cpu_cores, task.memory_mb) <= 0:
        raise ValueError("cardcode_limits_invalid")
    # Native Harbor's trusted supervisor needs root to drop the candidate to
    # the named unprivileged account and run the separate verifier. Bases may
    # default to an unnamed numeric UID; create the explicit account and make
    # only the public workspace writable before any candidate code executes.
    dockerfile = (
        f"FROM {task.image}\nUSER root\n"
        f"RUN (id -u {task.agent_user} >/dev/null 2>&1 || "
        f"useradd --create-home --shell /bin/bash {task.agent_user}) "
        f"&& test \"$(id -u {task.agent_user})\" -ne 0\n"
        "COPY workspace /work\n"
        f"RUN chown -R {task.agent_user} /work\n"
        "WORKDIR /work\n"
    )
    public = {
        "Dockerfile": dockerfile.encode(),
        **{"workspace/" + name: body for name, body in task.public_files.items()},
    }
    phases = {
        "instruction_b64": base64.b64encode(task.instruction).decode(),
        "tests_archive_b64": _tar_b64(task.verifier_files),
        "agent_timeout_s": task.agent_timeout_s,
        "verifier_timeout_s": task.verifier_timeout_s,
        "allow_internet": False,
        "workdir": "/work",
        "agent_user": task.agent_user,
        "verifier_user": "",
        "verifier_environment_mode": "isolated",
        "verifier_workspace_transfer": "archive",
        "collect_phase_logs": True,
    }
    return {
        "task_id": task_id,
        "lifecycle_profile": "cardcodebench",
        "context_tar_base64": _tar_b64(public),
        "entrypoint": "native_harbor_phases",
        "entrypoint_mode": "native_harbor",
        "harbor_agent": agent,
        "harbor_native_phases": phases,
        "limits": {
            "cpu_cores": task.cpu_cores,
            "memory_mb": task.memory_mb,
            "gpu_count": 0,
            "pids_limit": 128,
            "timeout_s": task.agent_timeout_s + task.verifier_timeout_s + 300,
        },
        "metadata": {
            "lifecycle_profile": "cardcodebench",
            "cardcode_task": task.task_id,
            "public_bundle_sha256": bundle_digest(task.public_files),
            "verifier_bundle_sha256": bundle_digest(task.verifier_files),
        },
    }
