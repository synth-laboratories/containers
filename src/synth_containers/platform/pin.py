"""Rollout pin status, admission identity, and the one status writer."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def _canonical_sha256(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class PinStatus(str, Enum):
    """The one status vocabulary a rollout pin can be in.

    Wire values are unchanged (``pin.status == "failed"`` still holds); every
    move goes through :func:`transition`, which is the only writer.
    """

    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    CANCELLED = "cancelled"
    CRASHED = "crashed"
    GAME_OVER = "game_over"

    def __str__(self) -> str:
        return self.value

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_PIN_STATUSES


_TERMINAL_PIN_STATUSES = frozenset(
    {
        PinStatus.COMPLETED,
        PinStatus.FAILED,
        PinStatus.TRUNCATED,
        PinStatus.CANCELLED,
        PinStatus.CRASHED,
        PinStatus.GAME_OVER,
    }
)

_PIN_TRANSITIONS: dict[PinStatus, frozenset[PinStatus]] = {
    PinStatus.PREPARED: frozenset({PinStatus.RUNNING}),
    PinStatus.RUNNING: frozenset({PinStatus.RUNNING, *_TERMINAL_PIN_STATUSES}),
}


class IllegalPinTransition(ValueError):
    """A status move the transition table does not name, or one without evidence."""


def transition(pin: "RolloutPin", to: "PinStatus | str", evidence: Mapping[str, Any]) -> PinStatus:
    """Move ``pin`` to ``to`` with a named reason; the sole writer of ``pin.status``."""
    target = PinStatus(to)
    current = PinStatus(str(pin.status))
    if target not in _PIN_TRANSITIONS.get(current, frozenset()):
        raise IllegalPinTransition(
            f"pin_transition_illegal:{pin.rollout_id}:{current.value}->{target.value}"
        )
    reason = evidence.get("reason") if isinstance(evidence, Mapping) else None
    if not isinstance(reason, str) or not reason.strip():
        raise IllegalPinTransition(
            f"pin_transition_evidence_required:{pin.rollout_id}:{target.value}"
        )
    pin.status = target
    pin.terminal = target.is_terminal
    if target.is_terminal:
        pin.completed_at = pin.completed_at or _utc_now()
    pin.transitions.append(
        {"from": current.value, "to": target.value, "at": _utc_now(), **dict(evidence)}
    )
    return target


def admission_identity_payload(
    *,
    harness: str,
    config: str | None,
    code: bytes | None,
    task_instance_id: str,
    world_ref: str,
    evaluation_plan_ref: str,
    omit_reward: bool,
    transport: str,
    retention: str,
    resume_from_checkpoint_id: str | None,
    execution: str | None,
) -> dict[str, Any]:
    """Canonical replay identity. Equality is digest equality, not a 10-clause boolean."""
    return {
        "harness": harness,
        "config": config,
        "code_digest": _canonical_sha256(code) if code else None,
        "task_instance_id": task_instance_id,
        "world_ref": world_ref,
        "evaluation_plan_ref": evaluation_plan_ref,
        "omit_reward": omit_reward,
        "transport": transport,
        "retention": retention,
        "resume_from_checkpoint_id": resume_from_checkpoint_id,
        "execution": execution,
    }


def admission_identity_digest(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(dict(payload))


LEASE_IDENTITY_KEYS = (
    "policy_ref",
    "task_instance_id",
    "seed",
    "config_digest",
    "capability_digest",
)


def require_lease_identity(record: Mapping[str, Any], *, source: str) -> None:
    """Fail closed when an orphaned lease is missing identity-bearing fields."""
    missing: list[str] = []
    for key in LEASE_IDENTITY_KEYS:
        if key not in record:
            missing.append(key)
            continue
        value = record[key]
        if value is None:
            missing.append(key)
        elif key == "task_instance_id" and str(value).strip() == "":
            missing.append(key)
        elif key in {"config_digest", "capability_digest"} and str(value).strip() == "":
            missing.append(key)
        elif key == "policy_ref" and not isinstance(value, dict):
            missing.append(key)
    if missing:
        raise ValueError(f"orphaned_lease_identity_missing:{source}:{','.join(missing)}")


def admission_from_raw(raw: Any) -> AdmissionReceipt | None:
    if not isinstance(raw, dict) or not raw.get("identity_digest"):
        return None
    return AdmissionReceipt(
        identity_digest=str(raw["identity_digest"]),
        rollout_id=str(raw.get("rollout_id") or ""),
        task_instance_id=str(raw.get("task_instance_id") or ""),
        seed=raw.get("seed"),
        world_ref=str(raw.get("world_ref") or ""),
        evaluation_plan_ref=str(raw.get("evaluation_plan_ref") or ""),
        omit_reward=bool(raw.get("omit_reward")),
        transport=str(raw.get("transport") or ""),
        retention=str(raw.get("retention") or ""),
        resume_from_checkpoint_id=raw.get("resume_from_checkpoint_id"),
        execution=raw.get("execution"),
        config_digest=str(raw.get("config_digest") or ""),
        capability_digest=str(raw.get("capability_digest") or ""),
        policy_harness=str(raw.get("policy_harness") or ""),
        policy_config=raw.get("policy_config"),
        accepted_at=str(raw.get("accepted_at") or ""),
    )
    """Fail closed when an orphaned lease is missing identity-bearing fields."""
    missing: list[str] = []
    for key in LEASE_IDENTITY_KEYS:
        if key not in record:
            missing.append(key)
            continue
        value = record[key]
        if value is None:
            missing.append(key)
        elif key == "task_instance_id" and str(value).strip() == "":
            missing.append(key)
        elif key in {"config_digest", "capability_digest"} and str(value).strip() == "":
            missing.append(key)
        elif key == "policy_ref" and not isinstance(value, dict):
            missing.append(key)
    if missing:
        raise ValueError(f"orphaned_lease_identity_missing:{source}:{','.join(missing)}")


@dataclass(frozen=True)
class AdmissionReceipt:
    """Write-once identity of an admitted rollout. Replay is digest equality."""

    identity_digest: str
    rollout_id: str
    task_instance_id: str
    seed: int | None
    world_ref: str
    evaluation_plan_ref: str
    omit_reward: bool
    transport: str
    retention: str
    resume_from_checkpoint_id: str | None
    execution: str | None
    config_digest: str
    capability_digest: str
    policy_harness: str
    policy_config: str | None
    accepted_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_digest": self.identity_digest,
            "rollout_id": self.rollout_id,
            "task_instance_id": self.task_instance_id,
            "seed": self.seed,
            "world_ref": self.world_ref,
            "evaluation_plan_ref": self.evaluation_plan_ref,
            "omit_reward": self.omit_reward,
            "transport": self.transport,
            "retention": self.retention,
            "resume_from_checkpoint_id": self.resume_from_checkpoint_id,
            "execution": self.execution,
            "config_digest": self.config_digest,
            "capability_digest": self.capability_digest,
            "policy_harness": self.policy_harness,
            "policy_config": self.policy_config,
            "accepted_at": self.accepted_at,
        }


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
    child_rollout_id: str | None = None
    child_resource_ref: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    terminal: bool = False
    status: PinStatus | str = PinStatus.PREPARED
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
    config_digest: str | None = None
    capability_digest: str | None = None
    execution: str | None = None
    idempotency_key: str | None = None
    accepted_at: str | None = None
    completed_at: str | None = None
    simulating: bool = False
    owner_id: str | None = None
    owner_kind: str | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    identity_digest: str | None = None
    admission: AdmissionReceipt | None = None


@dataclass(frozen=True)
class RolloutCredentialLease:
    """Per-rollout sampler credential. Never stored in ``policy_configs``."""

    rollout_id: str
    endpoint: str
    bearer: str = field(repr=False)

