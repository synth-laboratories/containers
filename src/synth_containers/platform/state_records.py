"""Value records and durable serialization helpers for the compat platform."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _checkpoint_digest(payload: Any) -> str:
    """Return the full content address used for durable checkpoint evidence."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _seed_from_task_instance_id(task_instance_id: str | None) -> int:
    """Parse seed from `seed:N` or a trailing `:N`. Absent → 0. No integer suffix → 0.

    A suffix that contains digits but is not an integer raises; do not coerce to 0.
    """
    if task_instance_id is None or task_instance_id == "":
        return 0
    if ":" not in task_instance_id:
        return 0
    suffix = task_instance_id.rsplit(":", 1)[-1]
    if suffix == "" or not any(ch.isdigit() for ch in suffix):
        return 0
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"task_instance_id seed suffix is not an integer: {task_instance_id!r}"
        ) from exc


@dataclass
class PolicyConfig:
    config_id: str
    harness: str
    config: dict[str, Any]
    code: bytes | None = None
    revision: int = 1


@dataclass
class PolicyRevision:
    revision_id: str
    digest: str
    harness: str
    config_id: str | None
    code: bytes | None
    isolation_receipt: dict[str, Any]
    namespace: str
    name: str
    configuration_digest: str
    model_digest: str
    source_revision: str | None
    installed_at: str


@dataclass
class RewardNode:
    node_id: str
    kind: str  # gate | aggregate | env_reward | script
    authority: str
    status: str
    value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "authority": self.authority,
            "status": self.status,
            "value": self.value,
        }


@dataclass
class RolloutPin:
    rollout_id: str
    world_ref: str
    environment_ref: str
    policy_ref: dict[str, Any]
    evaluation_plan_ref: str
    task_instance_id: str
    stream_id: str
    engine_generation: int
    policy_revision_id: str | None
    seed: int | None
    environment_version: str | None = None
    max_steps: int | None = None
    max_calls: int | None = None
    child_rollout_id: str | None = None
    child_resource_ref: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    terminal: bool = False
    status: str = "prepared"
    terminal_reason: str | None = None
    started: bool = False
    reward_signals: list[float | None] = field(default_factory=list)
    native_script_reward: float | None = None
    hillclimb_nodes: tuple[RewardNode, ...] | None = None
    env_generation: int = 1
    omit_reward: bool = False
    outcome: str | None = None
    session_dropped: bool = False
    reward_kind: str = "env_sum"
    checkpoint_schedule: dict[str, Any] | None = None
    resume_from_checkpoint_id: str | None = None
    scheduled_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    # Observe-only live annotation protocol bound to this rollout, if any.
    annotation_protocol_revision_id: str | None = None


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
