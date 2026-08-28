"""In-process containers-compat façade: pins, leases, policies, logs, /reward."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..event_log import (
    CONTROL_SUBSCRIBED,
    RolloutEventLog,
    stream_descriptor,
    validate_rollout_id,
)
from ..metadata import (
    LIVE_EVAL_PROTOCOL,
    RuntimeReadiness,
    compose_metadata_payload,
    runtime_provenance_from_environment,
)
from .affordances import bind_recipe
from .http_requests import CreateRolloutRequest, ISOLATED_POLICY_HARNESS, NANOHORIZON_HARNESS
from .policy_process import DEFAULT_HEURISTIC, IsolatedPolicyProcess
from .reward_plan import PlanOutcome, classify_plan_outcome
from .runtime import runtime_for
from .seal import seal_rollout_log, validate_rollout_seal
from .terminal_projection import terminal_journal_facts
from .targets import (
    PolicyInstallStatus,
    RewardKind,
    TargetRuntimeKind,
    TargetSpec,
    TaskInstanceStatus,
)
from .trace_bundle import (
    HarborTraceBundleRef,
    inspect_harbor_trace_bundle,
    materialize_harbor_trace_bundle,
)


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
    child_rollout_id: str | None = None
    child_resource_ref: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    terminal: bool = False
    status: str = "prepared"
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


class CompatPlatform:
    def __init__(
        self,
        spec: TargetSpec,
        *,
        storage_root: str | Path | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        # Process-local runtime extensions (for example a private, pinned
        # Harbor/Dock task bundle) are deliberately not part of target
        # metadata or the trace envelope.  Runtimes may read this immutable
        # construction input, but callers cannot mutate it through HTTP.
        self.runtime_config = dict(runtime_config or {})
        self.storage_root = (
            Path(storage_root)
            if storage_root is not None
            else Path(tempfile.mkdtemp(prefix="synth-containers-events-"))
        )
        self.engine_generation = 1
        self.policy_generation = 1
        self.active_leases = 0
        self.logs: dict[str, RolloutEventLog] = {}
        self.stream_bindings: dict[str, tuple[str, str]] = {}
        self.pins: dict[str, RolloutPin] = {}
        self.materialized_task_instances: dict[str, dict[str, Any]] = (
            self._load_materialized_task_instances()
        )
        self.policy_configs: dict[str, PolicyConfig] = {}
        self.policy_revisions: dict[str, PolicyRevision] = {}
        self.current_policy_revision_id: str | None = None
        self.reward_executions: dict[str, dict[str, Any]] = {}
        self.reward_by_execution_id: dict[str, dict[str, Any]] = {}
        self.evaluations: dict[str, dict[str, Any]] = {}
        self.evaluation_logs: dict[str, list[dict[str, Any]]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.seals: dict[str, dict[str, Any]] = {}
        self.trace_bundles: dict[str, HarborTraceBundleRef] = {}
        self.trace_bundle_errors: dict[str, str] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.stopped_worlds: set[str] = set()
        self.step_calls = 0
        self.start_session_calls = 0
        self.policy_code: bytes | None = None
        self.policy_process: IsolatedPolicyProcess | None = None
        self._state_lock = threading.RLock()
        self._seed_default_policies()
        self._recover_checkpoints()
        self._recover_completed_rollouts()

    def _task_instances_path(self) -> Path:
        return self.storage_root / "task_instances.json"

    def _load_materialized_task_instances(self) -> dict[str, dict[str, Any]]:
        path = self._task_instances_path()
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("task_instances_unrecoverable: expected an array")
        rows: dict[str, dict[str, Any]] = {}
        for row in payload:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise RuntimeError("task_instances_unrecoverable: invalid instance")
            rows[row["id"]] = row
        return rows

    def _persist_materialized_task_instances_locked(self) -> None:
        path = self._task_instances_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(list(self.materialized_task_instances.values()), sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _durable_key(self, rollout_id: str) -> str:
        return hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()

    def _manifest_path(self, rollout_id: str) -> Path:
        return self.storage_root / "run_manifests" / f"{self._durable_key(rollout_id)}.json"

    def _reward_path(self, rollout_id: str) -> Path:
        return self.storage_root / "reward_receipts" / f"{self._durable_key(rollout_id)}.json"

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.storage_root / "checkpoints" / f"{self._durable_key(checkpoint_id)}.json"

    def _recover_checkpoints(self) -> None:
        root = self.storage_root / "checkpoints"
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            checkpoint_id = str(record.get("checkpoint_id") or "")
            if record.get("schema_version") != "synth.containers.checkpoint.v1":
                raise ValueError(f"checkpoint_schema:{path.name}")
            if path.name != f"{self._durable_key(checkpoint_id)}.json":
                raise ValueError(f"checkpoint_identity:{path.name}")
            if record.get("target_id") != self.spec.target_id:
                continue
            expected_digest = _checkpoint_digest(
                {key: value for key, value in record.items() if key != "content_digest"}
            )
            if record.get("content_digest") != expected_digest:
                raise ValueError(f"checkpoint_digest:{checkpoint_id}")
            self.checkpoints[checkpoint_id] = record

    def record_checkpoint(self, record: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(record.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            raise ValueError("checkpoint_id_required")
        durable = {
            "schema_version": "synth.containers.checkpoint.v1",
            "target_id": self.spec.target_id,
            **record,
        }
        durable["content_digest"] = _checkpoint_digest(durable)
        existing = self.checkpoints.get(checkpoint_id)
        if existing is not None:
            if existing != durable:
                raise ValueError(f"checkpoint_identity_conflict:{checkpoint_id}")
            return existing
        self._atomic_json(self._checkpoint_path(checkpoint_id), durable)
        self.checkpoints[checkpoint_id] = durable
        return durable

    @staticmethod
    def _receipt_digest(value: dict[str, Any]) -> str:
        blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
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

    def _persist_completed_rollout(self, pin: RolloutPin) -> None:
        manifest = {
            "schema": "synth.containers.completed-rollout.v1",
            "target_id": self.spec.target_id,
            "rollout_id": pin.rollout_id,
            "stream_id": pin.stream_id,
            "stream_binding": list(self.stream_bindings[pin.rollout_id]),
            "pin": {
                "world_ref": pin.world_ref,
                "environment_ref": pin.environment_ref,
                "policy_ref": {key: value for key, value in pin.policy_ref.items() if key != "code"},
                "evaluation_plan_ref": pin.evaluation_plan_ref,
                "task_instance_id": pin.task_instance_id,
                "engine_generation": pin.engine_generation,
                "policy_revision_id": pin.policy_revision_id,
                "seed": pin.seed,
                "child_rollout_id": pin.child_rollout_id,
                "child_resource_ref": pin.child_resource_ref,
                "usage": pin.usage,
                "status": pin.status,
                "reward_signals": pin.reward_signals,
                "native_script_reward": pin.native_script_reward,
                "hillclimb_nodes": [node.to_dict() for node in (pin.hillclimb_nodes or ())],
                "env_generation": pin.env_generation,
                "omit_reward": pin.omit_reward,
                "outcome": pin.outcome,
                "session_dropped": pin.session_dropped,
                "reward_kind": pin.reward_kind,
                "checkpoint_schedule": pin.checkpoint_schedule,
                "resume_from_checkpoint_id": pin.resume_from_checkpoint_id,
                "scheduled_checkpoints": pin.scheduled_checkpoints,
            },
        }
        self._atomic_json(self._manifest_path(pin.rollout_id), manifest)

    def _recover_completed_rollouts(self) -> None:
        root = self.storage_root / "run_manifests"
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "synth.containers.completed-rollout.v1":
                raise ValueError(f"completed_rollout_manifest_schema:{path.name}")
            if manifest.get("target_id") != self.spec.target_id:
                continue
            rollout_id = str(manifest.get("rollout_id") or "")
            validate_rollout_id(rollout_id)
            expected_name = f"{self._durable_key(rollout_id)}.json"
            if path.name != expected_name:
                raise ValueError(f"completed_rollout_manifest_identity:{path.name}")
            stream_id = str(manifest.get("stream_id") or "")
            if stream_id != f"stream:{rollout_id}":
                raise ValueError(f"completed_rollout_stream_identity:{rollout_id}")
            journal = self.storage_root / "event_logs" / expected_name.replace(".json", ".jsonl")
            log = RolloutEventLog.recover(
                rollout_id=rollout_id,
                stream_id=stream_id,
                journal_path=journal,
            )
            seal_path = self.storage_root / "seals" / f"{rollout_id}.trace-v5.json"
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            validate_rollout_seal(seal)
            if seal.get("rollout_id") != rollout_id or seal.get("high_water") != log.high_water:
                raise ValueError(f"completed_rollout_seal_identity:{rollout_id}")
            expected_seal = seal_rollout_log(
                log,
                pin=seal.get("pin") if isinstance(seal.get("pin"), dict) else None,
            )
            if expected_seal["content_digest"] != seal["content_digest"]:
                raise ValueError(f"completed_rollout_seal_log_mismatch:{rollout_id}")
            binding = manifest.get("stream_binding")
            if not isinstance(binding, list) or len(binding) != 2:
                raise ValueError(f"completed_rollout_stream_binding:{rollout_id}")
            raw_pin = manifest.get("pin")
            if not isinstance(raw_pin, dict):
                raise ValueError(f"completed_rollout_pin:{rollout_id}")
            nodes = tuple(
                RewardNode(
                    node_id=str(row["node_id"]),
                    kind=str(row["kind"]),
                    authority=str(row["authority"]),
                    status=str(row["status"]),
                    value=row.get("value"),
                )
                for row in (raw_pin.get("hillclimb_nodes") or [])
            )
            pin = RolloutPin(
                rollout_id=rollout_id,
                world_ref=str(raw_pin["world_ref"]),
                environment_ref=str(raw_pin["environment_ref"]),
                policy_ref=dict(raw_pin["policy_ref"]),
                evaluation_plan_ref=str(raw_pin["evaluation_plan_ref"]),
                task_instance_id=str(raw_pin["task_instance_id"]),
                stream_id=stream_id,
                engine_generation=int(raw_pin["engine_generation"]),
                policy_revision_id=raw_pin.get("policy_revision_id"),
                seed=raw_pin.get("seed"),
                child_rollout_id=raw_pin.get("child_rollout_id"),
                child_resource_ref=raw_pin.get("child_resource_ref"),
                usage=raw_pin.get("usage"),
                terminal=True,
                status=str(raw_pin["status"]),
                started=True,
                reward_signals=list(raw_pin.get("reward_signals") or []),
                native_script_reward=raw_pin.get("native_script_reward"),
                hillclimb_nodes=nodes or None,
                env_generation=int(raw_pin.get("env_generation") or 1),
                omit_reward=bool(raw_pin.get("omit_reward")),
                outcome=raw_pin.get("outcome"),
                session_dropped=bool(raw_pin.get("session_dropped")),
                reward_kind=str(raw_pin.get("reward_kind") or self.spec.reward_kind),
                checkpoint_schedule=raw_pin.get("checkpoint_schedule"),
                resume_from_checkpoint_id=raw_pin.get("resume_from_checkpoint_id"),
                scheduled_checkpoints=list(raw_pin.get("scheduled_checkpoints") or []),
            )
            self.logs[rollout_id] = log
            self.stream_bindings[rollout_id] = (str(binding[0]), str(binding[1]))
            self.pins[rollout_id] = pin
            self.seals[rollout_id] = seal
            self._ensure_harbor_trace_bundle(pin=pin, log=log, seal=seal)
            reward_path = self._reward_path(rollout_id)
            if reward_path.is_file():
                wrapper = json.loads(reward_path.read_text(encoding="utf-8"))
                reward = wrapper.get("receipt") if isinstance(wrapper, dict) else None
                if (
                    wrapper.get("schema") != "synth.containers.reward-receipt.v1"
                    or not isinstance(reward, dict)
                    or wrapper.get("content_digest") != self._receipt_digest(reward)
                ):
                    raise ValueError(f"reward_receipt_digest:{rollout_id}")
                if reward.get("rollout_id") != rollout_id:
                    raise ValueError(f"reward_receipt_identity:{rollout_id}")
                self.reward_executions[rollout_id] = reward
                execution_id = reward.get("execution_id")
                if execution_id:
                    self.reward_by_execution_id[str(execution_id)] = reward

    def _seed_default_policies(self) -> None:
        for seed in self.spec.policy_seeds:
            self.policy_configs[seed.config_id] = PolicyConfig(
                config_id=seed.config_id,
                harness=seed.harness,
                config=dict(seed.config),
            )
        if self.spec.default_policy_harness in {ISOLATED_POLICY_HARNESS, NANOHORIZON_HARNESS}:
            return
        self.policy_configs.setdefault(
            "luna_med",
            PolicyConfig(
                config_id="luna_med",
                harness=self.spec.default_policy_harness,
                config={"model": "gpt-5.6-luna", "effort": "medium", "compact_every": 16},
            ),
        )
        self.policy_configs.setdefault(
            "sol_med",
            PolicyConfig(
                config_id="sol_med",
                harness=self.spec.default_policy_harness,
                config={"model": "gpt-5.6-sol", "effort": "medium"},
            ),
        )

    def _advertised_policy_refs(self) -> list[dict[str, Any]]:
        """Pins Desktop can bind before start. Harbor = two configs; dig.bench = two harnesses."""
        if self.spec.policy_seeds:
            refs = []
            for seed in self.spec.policy_seeds:
                ref = {"harness": seed.harness, "config": seed.config_id, "code": None}
                # Policy identity is useful to admission and the UI, but the
                # sampler configuration may also contain credential selectors
                # and transport details. Project only the public identity.
                for key in ("model", "provider", "api"):
                    value = seed.config.get(key)
                    if isinstance(value, str) and value.strip():
                        ref[key] = value.strip()
                refs.append(ref)
            return refs
        if self.spec.default_policy_harness == ISOLATED_POLICY_HARNESS:
            return [{"harness": ISOLATED_POLICY_HARNESS, "config": None, "code": None}]
        if self.spec.default_policy_harness == NANOHORIZON_HARNESS:
            return [{"harness": NANOHORIZON_HARNESS, "config": None, "code": None}]
        return [
            {
                "harness": self.spec.default_policy_harness,
                "config": "luna_med",
                "code": None,
            },
            {
                "harness": self.spec.default_policy_harness,
                "config": "sol_med",
                "code": None,
            },
        ]

    def _task_identity(self) -> tuple[str, str]:
        if self.spec.task_id and self.spec.task_family:
            return self.spec.task_id, self.spec.task_family
        identity_haystack = " ".join(
            (
                self.spec.target_id,
                self.spec.world_ref,
                self.spec.environment_ref,
                self.spec.evaluation_plan_ref,
            )
        ).lower()
        inferred_family = "craftax" if "craftax" in identity_haystack else self.spec.target_id
        return self.spec.task_id or inferred_family, self.spec.task_family or inferred_family

    def _task_definition_locked(self) -> dict[str, Any]:
        task_id, family = self._task_identity()
        return {
            "id": task_id,
            "family": family,
            "world_ref": self.spec.world_ref,
            "environment_ref": self.spec.environment_ref,
            "evaluation_plan_ref": self.spec.evaluation_plan_ref,
            "max_episode_steps": self.spec.max_episode_steps,
        }

    def _catalog_policy_ref_locked(self, pin: RolloutPin) -> dict[str, Any]:
        """Project policy identity without executable code or secret-bearing config."""
        harness = pin.policy_ref.get("harness")
        config_id = pin.policy_ref.get("config")
        result: dict[str, Any] = {"harness": harness, "config": config_id}
        config = self.policy_configs.get(str(config_id or ""))
        if config is not None:
            for key in ("model", "provider"):
                value = config.config.get(key)
                if isinstance(value, str) and value.strip():
                    result[key] = value.strip()
        return result

    @staticmethod
    def _catalog_status(pin: RolloutPin) -> TaskInstanceStatus:
        if not pin.started:
            return TaskInstanceStatus.PREPARED
        if not pin.terminal:
            return TaskInstanceStatus.RUNNING
        try:
            return TaskInstanceStatus(pin.status)
        except ValueError:
            return TaskInstanceStatus.TERMINAL

    def _task_instance_locked(self, pin: RolloutPin, task_id: str) -> dict[str, Any]:
        reward = self.reward_executions.get(pin.rollout_id)
        return {
            "id": pin.rollout_id,
            "task_id": task_id,
            "task_instance_id": pin.task_instance_id,
            "rollout_id": pin.rollout_id,
            "seed": pin.seed,
            "status": self._catalog_status(pin),
            "terminal": pin.terminal,
            "started": pin.started,
            "reward": reward.get("reward") if reward is not None else None,
            "policy_ref": self._catalog_policy_ref_locked(pin),
            "policy_revision_id": pin.policy_revision_id,
        }

    def materialize_task_instances(self, task_id: str, seeds: list[int]) -> list[dict[str, Any]]:
        """Idempotently declare deterministic, non-spending task instances."""
        with self._state_lock:
            declared_task = str(self._task_definition_locked()["id"])
            if task_id != declared_task:
                raise ValueError(f"unknown_task:{task_id}")
            rows: list[dict[str, Any]] = []
            for seed in seeds:
                instance_id = f"{task_id}:seed:{seed}"
                row = self.materialized_task_instances.setdefault(
                    instance_id,
                    {
                        "id": instance_id,
                        "task_id": task_id,
                        "task_instance_id": instance_id,
                        "seed": seed,
                        "status": "planned",
                        "terminal": False,
                        "started": False,
                        "rollout_id": None,
                        "reward": None,
                        "policy_ref": None,
                        "policy_revision_id": None,
                    },
                )
                rows.append(dict(row))
            self._persist_materialized_task_instances_locked()
            return rows

    def task_catalog_payload(self) -> dict[str, Any]:
        """Snapshot definitions and RolloutPin-backed instances atomically."""
        with self._state_lock:
            task = self._task_definition_locked()
            by_instance = {
                key: dict(value) for key, value in self.materialized_task_instances.items()
            }
            for pin in self.pins.values():
                row = self._task_instance_locked(pin, str(task["id"]))
                row["id"] = pin.task_instance_id
                by_instance[pin.task_instance_id] = row
            instances = list(by_instance.values())
            return {
                "schema_version": "synth.container.task-catalog.v1",
                "tasks": [task],
                "instances": instances,
            }

    def task_info_payload(self) -> dict[str, Any]:
        with self._state_lock:
            return self._task_definition_locked()

    def metadata_payload(self) -> dict[str, Any]:
        services = {
            "world": self.spec.world_ref,
            "environment": self.spec.environment_ref,
            "policy": f"policy:{self.spec.default_policy_harness}",
            "evaluator": self.spec.evaluation_plan_ref,
            "relay": "relay:event_log",
        }
        advertised = self.spec.affordances.advertised()
        booleans = {
            role: {name: level != "unsupported" for name, level in items.items()}
            for role, items in advertised.items()
        }
        policy_refs = self._advertised_policy_refs()
        payload = {
            "world_ref": self.spec.world_ref,
            "environment_ref": self.spec.environment_ref,
            "policy_ref": policy_refs[0],
            "policy_refs": policy_refs,
            "evaluation_plan_ref": self.spec.evaluation_plan_ref,
            "task_instance_id": None,
            "adapter_chain": list(self.spec.adapter_chain),
            "affordances": advertised,
            "affordance_booleans": booleans,
            "scale_leases": self.spec.scale_leases,
            "active_leases": self.active_leases,
            "retention": self.spec.retention,
            "logical_service_ids": services,
            "reward_authority": (
                "trusted_scorer" if self.spec.reward_kind == "script" else "environment"
            ),
            "live_reward": self.spec.live_reward,
            "live_frames": self.spec.live_frames,
            "true_checkpoint": self.spec.true_checkpoint,
            "blocking_trial": self.spec.blocking_trial,
            "mcp_bind": self.spec.mcp_bind,
            "reconnect": self.spec.reconnect,
            "event_kinds": list(self.spec.event_kinds),
            "target_id": self.spec.target_id,
            "runtime_family": self.spec.runtime_family.value,
            "max_episode_steps": self.spec.max_episode_steps,
            # This facade owns the complete prepare → start → reconcile →
            # reward workflow and seals its durable event log on terminal
            # rollout.  Advertise that contract explicitly so Workshop can
            # register it as a first-class live environment rather than
            # inferring compatibility from a healthy HTTP endpoint or SSE.
            "capabilities": {
                "protocol": "synth.container.live-eval.v1",
                "operations": {
                    "rollouts.prepare": True,
                    "rollouts.start_prepared": True,
                    "rollouts.get": True,
                    "rollouts.poll": True,
                    "reward.get": True,
                    "trace_v5.capture": True,
                },
                "policy_refs": policy_refs,
            },
        }
        optimizer_contracts = None
        if contract := self._gepa_v2_contract():
            optimizer_contracts = {"gepa": contract}
        return compose_metadata_payload(
            base=payload,
            protocol=LIVE_EVAL_PROTOCOL,
            live_frames=self.spec.live_frames,
            readiness=RuntimeReadiness(
                policy_ready=any(ref.get("config") for ref in policy_refs),
                program_ready=False,
            ),
            optimizer_contracts=optimizer_contracts,
            scale_leases=self.spec.scale_leases,
        )

    def _gepa_v2_contract(self) -> dict[str, Any] | None:
        contract = self.spec.optimizer_contracts
        return dict(contract) if isinstance(contract, dict) else None

    def bind(self, recipe: dict[str, Any] | None) -> dict[str, Any] | None:
        return bind_recipe(self.spec.affordances, recipe)

    def _refuse_transport(self, transport: str) -> dict[str, Any] | None:
        if transport == "auto":
            return {
                "error": "transport_refused",
                "status_code": 422,
                "detail": "telemetry.transport=auto is refused on authoritative / visual-attached runs",
            }
        if transport == "sse" and self.spec.affordances.level("sse") == "unsupported":
            return {
                "error": "transport_refused",
                "status_code": 422,
                "detail": "sse not advertised",
            }
        if transport == "websocket" and self.spec.affordances.level("websocket") == "unsupported":
            return {
                "error": "transport_refused",
                "status_code": 422,
                "detail": "websocket not advertised",
            }
        return None

    def prepare(
        self,
        rollout_id: str,
        transport: str,
        retention: str,
        request: CreateRolloutRequest | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            return self._prepare_locked(rollout_id, transport, retention, request=request)

    def _prepare_locked(
        self,
        rollout_id: str,
        transport: str,
        retention: str,
        *,
        request: CreateRolloutRequest | None = None,
    ) -> dict[str, Any]:
        stream_id = f"stream:{rollout_id}"
        # rollout_id is caller-controlled; never use it as a path component.
        journal_name = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()
        journal_path = self.storage_root / "event_logs" / f"{journal_name}.jsonl"
        try:
            log = RolloutEventLog.recover(
                rollout_id=rollout_id,
                stream_id=stream_id,
                journal_path=journal_path,
            )
        except ValueError as exc:
            raise RuntimeError(f"event_log_unrecoverable:{exc}") from exc
        if log.closed:
            raise RuntimeError(f"event_log_sealed:{rollout_id}")
        log.append_control(CONTROL_SUBSCRIBED, log.subscribed_payload())
        self.logs[rollout_id] = log
        self.stream_bindings[rollout_id] = (transport, retention or self.spec.retention)
        if rollout_id not in self.pins:
            task_instance_id = request.task_instance_id if request is not None else None
            seed = _seed_from_task_instance_id(task_instance_id)
            policy_ref = request.policy_ref if request is not None else None
            self.pins[rollout_id] = RolloutPin(
                rollout_id=rollout_id,
                world_ref=str(
                    (request.world_ref if request is not None else None) or self.spec.world_ref
                ),
                environment_ref=self.spec.environment_ref,
                policy_ref={
                    "harness": policy_ref.harness if policy_ref is not None else None,
                    "config": policy_ref.config if policy_ref is not None else None,
                    "code": policy_ref.code if policy_ref is not None else None,
                },
                evaluation_plan_ref=str(
                    (request.evaluation_plan_ref if request is not None else None)
                    or self.spec.evaluation_plan_ref
                ),
                task_instance_id=task_instance_id or f"seed:{seed}",
                stream_id=stream_id,
                engine_generation=self.engine_generation,
                policy_revision_id=self.current_policy_revision_id,
                seed=seed,
            )
        return stream_descriptor(
            rollout_id=rollout_id,
            stream_id=stream_id,
            bound_transport=transport,
            retention=retention or self.spec.retention,
        )

    def occupy_or_busy(self) -> dict[str, Any] | None:
        if self.active_leases >= self.spec.scale_leases:
            return {
                "status": "busy",
                "affordance": "scale_leases",
                "scale_leases": self.spec.scale_leases,
                "active_leases": self.active_leases,
            }
        return None

    def start_rollout(self, request: CreateRolloutRequest) -> dict[str, Any]:
        with self._state_lock:
            result = self._start_rollout_locked(request, defer_sync=True)
            rollout_id = str(result.get("rollout_id") or request.rollout_id or "")
            pin = self.pins.get(rollout_id)
            log = self.logs.get(rollout_id)
            should_run = (
                request.submission_mode != "async"
                and pin is not None
                and log is not None
                and pin.status == "running"
                and not bool(result.get("replayed"))
            )
        if not should_run:
            return result
        try:
            # Policy and evaluator calls can take minutes. They are isolated by
            # rollout identity and must not hold the platform-wide admission
            # lock, otherwise one synchronous rollout makes advertised leases
            # fictitious and blocks prepare/start for every other rollout.
            # Keep the same fail-closed terminalization contract as the
            # locked synchronous path. Moving execution outside the admission
            # lock must not turn runtime exceptions into permanently running
            # pins that block later policy registration or replay.
            self._simulate_or_fail(pin, log)
        finally:
            with self._state_lock:
                self.active_leases = max(0, self.active_leases - 1)
                result = self._rollout_response(
                    pin, self.stream_descriptor_for(rollout_id)
                )
        return result

    def _start_rollout_locked(
        self, request: CreateRolloutRequest, *, defer_sync: bool = False
    ) -> dict[str, Any]:
        slot = request.slot
        if slot in {"live", "jobs"}:
            return {
                "error": "slot_refused",
                "status_code": 400,
                "detail": f"slot {slot!r} is not bindable; use declared stream.id",
            }
        refusal = self.bind(request.recipe)
        if refusal:
            return {"error": "bind_refused", "status_code": 403, **refusal}

        telemetry = request.telemetry
        transport = telemetry.transport
        if telemetry.enabled:
            bad = self._refuse_transport(transport)
            if bad:
                return bad

        rollout_id = str(request.rollout_id or f"roll_{uuid.uuid4().hex[:12]}")
        existing_pin = self.pins.get(rollout_id)
        if existing_pin is not None and existing_pin.started:
            requested_harness = (request.policy_ref.harness or "").strip()
            requested_config = request.policy_ref.config
            requested_seed = _seed_from_task_instance_id(request.task_instance_id)
            requested_task = request.task_instance_id or f"seed:{requested_seed}"
            requested_retention = (
                telemetry.retention
                if telemetry.retention is not None
                else self.spec.retention
            )
            bound_transport, bound_retention = self.stream_bindings.get(
                rollout_id,
                (transport, requested_retention),
            )
            same_identity = (
                existing_pin.policy_ref.get("harness") == requested_harness
                and existing_pin.policy_ref.get("config") == requested_config
                and existing_pin.policy_ref.get("code") == request.policy_ref.code
                and existing_pin.task_instance_id == requested_task
                and existing_pin.world_ref == str(request.world_ref or self.spec.world_ref)
                and existing_pin.evaluation_plan_ref
                == str(request.evaluation_plan_ref or self.spec.evaluation_plan_ref)
                and existing_pin.omit_reward == request.omit_reward
                and bound_transport == transport
                and bound_retention == requested_retention
                and existing_pin.resume_from_checkpoint_id
                == request.resume_from_checkpoint_id
            )
            if not same_identity:
                return {
                    "error": "rollout_identity_conflict",
                    "status_code": 409,
                    "rollout_id": rollout_id,
                }
            replay = self._rollout_response(
                existing_pin, self.stream_descriptor_for(rollout_id)
            )
            replay["replayed"] = True
            return replay

        busy = self.occupy_or_busy()
        if busy:
            return {"error": "occupancy", "status_code": 429, **busy}
        retention = telemetry.retention if telemetry.retention is not None else self.spec.retention
        if rollout_id not in self.logs:
            descriptor = self.prepare(rollout_id, transport, retention)
        else:
            bound_transport, bound_retention = self.stream_bindings.get(
                rollout_id,
                (transport, retention),
            )
            if telemetry.enabled and (
                transport != bound_transport or retention != bound_retention
            ):
                return {
                    "error": "stream_binding_mismatch",
                    "status_code": 409,
                    "detail": "start must use the transport and retention declared by prepare",
                    "prepared": {
                        "transport": bound_transport,
                        "retention": bound_retention,
                    },
                    "requested": {"transport": transport, "retention": retention},
                }
            descriptor = self.stream_descriptor_for(rollout_id)

        policy_ref = request.policy_ref
        harness = (policy_ref.harness or "").strip()
        config_id = policy_ref.config
        if not harness:
            return {
                "error": "policy_ref_required",
                "status_code": 422,
                "detail": "POST /rollouts requires policy_ref.harness; the platform does not pick a recipe",
            }
        if harness == ISOLATED_POLICY_HARNESS and config_id:
            return {
                "error": "bind_refused",
                "status_code": 403,
                "affordance": "bind_policy_config",
                "advertised": "unsupported",
            }
        if harness != ISOLATED_POLICY_HARNESS and not (config_id or "").strip():
            return {
                "error": "policy_ref_config_required",
                "status_code": 422,
                "detail": "POST /rollouts requires policy_ref.config; the platform does not default luna_med",
            }
        if config_id and config_id not in self.policy_configs and harness != ISOLATED_POLICY_HARNESS:
            return {"error": "unknown_policy_config", "status_code": 404, "config_id": config_id}
        registered_config = self.policy_configs.get(str(config_id or ""))
        if registered_config is not None and registered_config.harness != harness:
            return {
                "error": "policy_configuration_mismatch",
                "status_code": 409,
                "requested_policy_ref": {"harness": harness, "config": config_id},
                "registered_policy_config": {
                    "harness": registered_config.harness,
                    "config": registered_config.config_id,
                },
            }
        if harness == NANOHORIZON_HARNESS:
            installed_revision = self.policy_revisions.get(
                str(self.current_policy_revision_id or "")
            )
            if installed_revision is None:
                return {
                    "error": "policy_not_installed",
                    "status_code": 409,
                    "detail": "harness nanohorizon requires PUT /policy before POST /rollouts",
                }
            requested_revision = request.policy_revision_id
            if not requested_revision:
                return {
                    "error": "policy_revision_required",
                    "status_code": 422,
                    "detail": "POST /rollouts requires policy_revision_id for harness nanohorizon",
                }
            revision = self.policy_revisions.get(requested_revision)
            if revision is None or not revision.code:
                return {
                    "error": "policy_revision_unknown",
                    "status_code": 404,
                    "policy_revision_id": requested_revision,
                    "detail": "the requested immutable policy revision is not installed",
                }
            if requested_revision != self.current_policy_revision_id:
                return {
                    "error": "policy_revision_mismatch",
                    "status_code": 409,
                    "requested_policy_revision_id": requested_revision,
                    "installed_policy_revision_id": self.current_policy_revision_id,
                }
            # Only the harness is an identity claim about the revision. The
            # revision deliberately does not own a config id -- PUT stores
            # config_id=None -- because a policy is installed once while sampler
            # configs are bound per episode: nanohorizon's binder uses
            # nh-<session_id> so the sampler base_url can carry the session. The
            # hot_swap flow happens to pass the same string for both, which is
            # what made comparing revision.name to config_id look correct; it is
            # a category error, and it 409s every rollout for any caller that
            # binds config per episode. config_id is already validated above
            # against self.policy_configs (404 unknown_policy_config).
            if revision.harness != harness:
                return {
                    "error": "policy_configuration_mismatch",
                    "status_code": 409,
                    "policy_revision_id": requested_revision,
                    "requested_policy_ref": {"namespace": harness, "name": config_id},
                    "installed_policy_ref": {
                        "namespace": revision.namespace,
                        "name": revision.name,
                    },
                }

        if self.spec.admission is not None:
            refusal = self.spec.admission(self, request)
            if isinstance(refusal, dict) and refusal:
                return refusal

        if request.resume_from_checkpoint_id:
            if self.spec.affordances.level("restore") != "native" or self.spec.affordances.level(
                "fork"
            ) != "native":
                return {
                    "error": "checkpoint_resume_unsupported",
                    "status_code": 409,
                    "checkpoint_id": request.resume_from_checkpoint_id,
                }
            checkpoint = self.checkpoints.get(request.resume_from_checkpoint_id)
            if checkpoint is None:
                return {
                    "error": "unknown_checkpoint",
                    "status_code": 404,
                    "checkpoint_id": request.resume_from_checkpoint_id,
                }
            if checkpoint.get("environment_ref") != self.spec.environment_ref:
                return {
                    "error": "checkpoint_environment_mismatch",
                    "status_code": 409,
                    "checkpoint_id": request.resume_from_checkpoint_id,
                }

        seed_i = _seed_from_task_instance_id(request.task_instance_id)
        task_instance_id = request.task_instance_id or f"seed:{seed_i}"
        async_mode = request.submission_mode == "async"
        pin = RolloutPin(
            rollout_id=rollout_id,
            world_ref=str(request.world_ref or self.spec.world_ref),
            environment_ref=self.spec.environment_ref,
            policy_ref={"harness": harness, "config": config_id, "code": policy_ref.code},
            evaluation_plan_ref=str(request.evaluation_plan_ref or self.spec.evaluation_plan_ref),
            task_instance_id=task_instance_id,
            stream_id=descriptor["id"],
            engine_generation=self.engine_generation,
            policy_revision_id=request.policy_revision_id,
            seed=seed_i,
            usage=None,
            omit_reward=request.omit_reward,
            outcome=request.outcome,
            reward_kind=self.spec.reward_kind,
            checkpoint_schedule=request.checkpoint_schedule,
            resume_from_checkpoint_id=request.resume_from_checkpoint_id,
        )
        self.pins[rollout_id] = pin
        self.active_leases += 1
        log = self.logs[rollout_id]
        if not any(item.kind == "trace.opened" for item in log.after(0)):
            # Immutable, secret-free lane identity belongs in the durable trace.
            # In particular, Workshop must be able to distinguish two harnesses
            # without guessing from rollout ids or from MCP-shaped events.
            log.append(
                "trace.opened",
                {
                    "rollout_id": rollout_id,
                    "stream.id": log.stream_id,
                    "world_ref": pin.world_ref,
                    "environment_ref": pin.environment_ref,
                    "task_instance_id": pin.task_instance_id,
                    "policy_ref": {
                        "harness": harness,
                        "config": config_id,
                    },
                },
            )
        pin.started = True
        pin.status = "running"
        if async_mode or defer_sync:
            return self._rollout_response(pin, descriptor)
        try:
            self._simulate_or_fail(pin, log)
        finally:
            self.active_leases = max(0, self.active_leases - 1)
        return self._rollout_response(pin, descriptor)

    def complete_rollout(self, rollout_id: str) -> dict[str, Any]:
        with self._state_lock:
            pin = self.pins.get(rollout_id)
            log = self.logs.get(rollout_id)
            if pin is None or log is None:
                return {"error": "unknown_rollout", "status_code": 404}
            if pin.terminal:
                return self._rollout_response(pin, self.stream_descriptor_for(rollout_id))
        try:
            # Async submissions are admitted and pinned by POST /rollouts, then
            # executed here. Model waits must not hold the platform-wide state
            # lock or distinct leases silently serialize behind one another.
            self._simulate_or_fail(pin, log)
        finally:
            with self._state_lock:
                self.active_leases = max(0, self.active_leases - 1)
                result = self._rollout_response(
                    pin, self.stream_descriptor_for(rollout_id)
                )
        return result

    def _complete_rollout_locked(self, rollout_id: str) -> dict[str, Any]:
        pin = self.pins.get(rollout_id)
        log = self.logs.get(rollout_id)
        if pin is None or log is None:
            return {"error": "unknown_rollout", "status_code": 404}
        if pin.terminal:
            return self._rollout_response(pin, self.stream_descriptor_for(rollout_id))
        try:
            self._simulate_or_fail(pin, log)
        finally:
            self.active_leases = max(0, self.active_leases - 1)
        return self._rollout_response(pin, self.stream_descriptor_for(rollout_id))

    def rollout_status(self, rollout_id: str) -> dict[str, Any]:
        pin = self.pins.get(rollout_id)
        if pin is None:
            if rollout_id in self.logs:
                return {
                    "rollout_id": rollout_id,
                    "status": "prepared",
                    "started": False,
                    "terminated": False,
                    "stream": self.stream_descriptor_for(rollout_id),
                }
            return {"error": "unknown_rollout", "status_code": 404}
        return {
            **self._rollout_response(pin, self.stream_descriptor_for(rollout_id)),
            "started": pin.started,
        }

    def stream_descriptor_for(self, rollout_id: str) -> dict[str, Any]:
        log = self.logs[rollout_id]
        transport, retention = self.stream_bindings.get(
            rollout_id,
            ("poll", self.spec.retention),
        )
        return stream_descriptor(
            rollout_id=rollout_id,
            stream_id=log.stream_id,
            bound_transport=transport,
            retention=retention,
        )

    def transport_is_bound(self, rollout_id: str, transport: str) -> bool:
        binding = self.stream_bindings.get(rollout_id)
        if binding is None:
            return False
        bound, _ = binding
        if transport == "poll":
            return True
        if transport == "sse":
            return bound in {"sse", "websocket"}
        if transport == "websocket":
            return bound == "websocket"
        return False

    def _rollout_response(self, pin: RolloutPin, descriptor: dict[str, Any]) -> dict[str, Any]:
        reward = self.reward_executions.get(pin.rollout_id)
        terminal = terminal_journal_facts(self.logs.get(pin.rollout_id))
        failed_terminal = pin.terminal and pin.status in {
            "failed",
            "truncated",
            "cancelled",
            "terminated",
            "stopped",
        }
        reason = terminal.get("reason")
        detail = terminal.get("detail") or terminal.get("error")
        if failed_terminal:
            reason = reason or "producer_failure"
            detail = detail or f"producer terminated rollout with status {pin.status}"
        return {
            "rollout_id": pin.rollout_id,
            "status": pin.status,
            "world_ref": pin.world_ref,
            "environment_ref": pin.environment_ref,
            "policy_ref": pin.policy_ref,
            "evaluation_plan_ref": pin.evaluation_plan_ref,
            "task_instance_id": pin.task_instance_id,
            "stream": descriptor,
            "usage": pin.usage,
            # A zero is a scored outcome, not an absent reward. Harbor
            # finalizes its verifier result as part of its terminal runtime,
            # so expose the materialized receipt on the authoritative rollout
            # record as well as on the dedicated reward route.
            "reward": reward.get("reward") if reward is not None else None,
            "reward_status": reward.get("status") if reward is not None else None,
            "steps": terminal.get("steps"),
            "reason": reason,
            "detail": detail,
            "child_rollout_id": pin.child_rollout_id,
            "child_resource_ref": pin.child_resource_ref,
            "engine_generation": pin.engine_generation,
            "policy_revision_id": pin.policy_revision_id,
            "terminated": pin.terminal,
            "truncated": pin.status == "truncated",
            "resume_from_checkpoint_id": pin.resume_from_checkpoint_id,
            "scheduled_checkpoints": pin.scheduled_checkpoints,
            "trace": self._sealed_trace_reference(pin.rollout_id),
        }

    def _sealed_trace_reference(self, rollout_id: str) -> dict[str, Any] | None:
        """Announce the sealed trace on the authoritative terminal record.

        A seal that only exists at ``/rollouts/{id}/trace`` is a seal the
        consumer has to already know about. Workshop read terminal rollout
        records, saw no trace at all, and its own index stayed empty while this
        process held a complete sealed trace on disk — split authority with no
        edge between the halves. The terminal record now carries the identity
        and where to fetch it; the seal itself stays behind its own route.
        """
        seal = self.seals.get(rollout_id)
        if seal is None:
            return None
        reference = {
            "schema_version": seal.get("schema_version"),
            "trace_id": seal.get("trace_id"),
            "content_digest": seal.get("content_digest"),
            "event_count": len(seal.get("events") or []),
            "high_water": seal.get("high_water"),
            "closed": seal.get("closed"),
            "url": f"/rollouts/{rollout_id}/trace",
        }
        bundle = self.trace_bundles.get(rollout_id)
        if bundle is not None:
            reference.update(
                {
                    "bundle_url": f"/rollouts/{rollout_id}/trace/bundle",
                    "kind": "trace_v5_bundle",
                    "inspectable": True,
                    "bundle_digest": bundle.bundle_digest,
                    "bundle_archive_digest": bundle.archive_digest,
                    "bundle_trace_id": bundle.trace_id,
                    "bundle_trace_digest": bundle.trace_digest,
                    "bundle_byte_size": bundle.byte_size,
                }
            )
        return reference

    def trace_bundle_archive(self, rollout_id: str) -> Path | None:
        """Return only a verified self-contained Harbor archive for HTTP serving."""
        bundle = self.trace_bundles.get(rollout_id)
        if bundle is None or not bundle.archive_path.is_file():
            return None
        return bundle.archive_path

    def _trace_bundle_path(self, rollout_id: str) -> Path:
        return self.storage_root / "trace_bundles" / f"{self._durable_key(rollout_id)}.zip"

    def _ensure_harbor_trace_bundle(
        self,
        *,
        pin: RolloutPin,
        log: RolloutEventLog,
        seal: dict[str, Any],
    ) -> None:
        """Derive an inspectable archive without making raw evidence disposable.

        The durable compatibility-layer journal exists for every runtime
        family.  In particular, NanoHorizon targets use the external runtime,
        so restricting promotion to Harbor silently left their sealed traces
        without a downloadable Trace V5 bundle.
        """
        archive_path = self._trace_bundle_path(pin.rollout_id)
        try:
            if archive_path.is_file():
                try:
                    self.trace_bundles[pin.rollout_id] = inspect_harbor_trace_bundle(archive_path)
                    self.trace_bundle_errors.pop(pin.rollout_id, None)
                    return
                except Exception:
                    # The journal and lite seal are the durable authorities.
                    # A derived archive may be regenerated without discarding
                    # either of them, including after a partial host write.
                    pass
            runtime_provenance = runtime_provenance_from_environment()
            self.trace_bundles[pin.rollout_id] = materialize_harbor_trace_bundle(
                output_path=archive_path,
                spec=self.spec,
                log=log,
                seal=seal,
                pin={
                    "world_ref": pin.world_ref,
                    "environment_ref": pin.environment_ref,
                    "policy_ref": pin.policy_ref,
                    "evaluation_plan_ref": pin.evaluation_plan_ref,
                    "task_instance_id": pin.task_instance_id,
                },
                status=pin.status,
                producer_commit=runtime_provenance.producer_source_revision,
                container_image_digest=runtime_provenance.image_digest,
            )
            self.trace_bundle_errors.pop(pin.rollout_id, None)
        except Exception as exc:  # Raw journal + lite seal remain the fallback evidence.
            self.trace_bundles.pop(pin.rollout_id, None)
            self.trace_bundle_errors[pin.rollout_id] = str(exc)

    def _simulate_or_fail(self, pin: RolloutPin, log: RolloutEventLog) -> None:
        """Run the rollout, terminalizing the pin even when it raises.

        `register_policy_config` refuses while any pin is started and not
        terminal. A rollout that raised before its runtime could record an
        outcome — a malformed policy config, say — left its pin pinned forever,
        so every later bind returned 409 and the container had to be restarted
        to accept work again. The error still propagates; it just no longer
        takes the container down with it.
        """
        try:
            self._simulate(pin, log)
        except BaseException:
            pin.status = "failed"
            pin.terminal = True
            raise

    def _simulate(self, pin: RolloutPin, log: RolloutEventLog) -> None:
        pin.env_generation += 1
        runtime_for(self.spec).simulate(self, pin, log)
        # Persist a terminal reward receipt as soon as its authority has
        # finished, instead of requiring a separate caller-authored POST after
        # the run closes. Harbor supplies a native verifier; ENV_SUM targets
        # already hold authoritative environment signals. This preserves null
        # for missing evidence while making an authoritative 0.0 discoverable.
        if pin.terminal and (
            self.spec.runtime_family == TargetRuntimeKind.HARBOR
            or self.spec.reward_kind == RewardKind.ENV_SUM
        ):
            self.compute_reward(
                rollout_id=pin.rollout_id,
                evidence=None,
                mode="terminal",
                rescore=False,
                plan_ref=pin.evaluation_plan_ref,
            )
        self.seals[pin.rollout_id] = seal_rollout_log(
            log,
            pin={
                "world_ref": pin.world_ref,
                "environment_ref": pin.environment_ref,
                "policy_ref": pin.policy_ref,
                "evaluation_plan_ref": pin.evaluation_plan_ref,
                "task_instance_id": pin.task_instance_id,
            },
        )
        seal_path = self.storage_root / "seals" / f"{pin.rollout_id}.trace-v5.json"
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = seal_path.with_name(
            f".{seal_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        encoded = json.dumps(
            self.seals[pin.rollout_id],
            sort_keys=True,
            separators=(",", ":"),
        )
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, seal_path)
        directory_fd = os.open(seal_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._ensure_harbor_trace_bundle(
            pin=pin,
            log=log,
            seal=self.seals[pin.rollout_id],
        )
        self._persist_completed_rollout(pin)

    def _ensure_policy_process(self) -> IsolatedPolicyProcess:
        if self.policy_process is not None and self.policy_process._proc.poll() is None:
            return self.policy_process
        code = self.policy_code or DEFAULT_HEURISTIC.encode("utf-8")
        self.policy_process = IsolatedPolicyProcess(code)
        return self.policy_process

    def _close_policy_process(self) -> None:
        process = self.policy_process
        self.policy_process = None
        if process is not None:
            process.close()

    def put_policy(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.spec.affordances.level("update_policy_code") == "unsupported":
            return {"error": "bind_refused", "status_code": 403, "affordance": "update_policy_code"}
        code = body.get("code")
        harness = str(body.get("harness") or self.spec.default_policy_harness)
        namespace = str(body.get("namespace") or harness).strip()
        name = str(body.get("name") or "").strip()
        configuration = body.get("configuration") or {}
        model = body.get("model") or {}
        source_revision = body.get("source_revision")
        if not namespace or not name:
            return {
                "error": "policy_identity_required",
                "status_code": 422,
                "detail": "PUT /policy requires namespace and name",
            }
        secret_suffixes = ("apikey", "secret", "token", "password")

        def secret_key(key: Any) -> bool:
            normalized = str(key).replace("_", "").replace("-", "").lower()
            return normalized == "credential" or normalized.endswith(secret_suffixes)

        def contains_secret(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    secret_key(key) or contains_secret(child)
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return any(contains_secret(child) for child in value)
            return False

        if contains_secret(configuration) or contains_secret(model):
            return {
                "error": "policy_credential_forbidden",
                "status_code": 422,
                "detail": "PUT /policy accepts identity and configuration, never credentials",
            }
        raw = code if isinstance(code, (bytes, bytearray)) else str(code or "").encode("utf-8")
        if harness == NANOHORIZON_HARNESS and not raw:
            return {
                "error": "policy_source_required",
                "status_code": 422,
                "detail": "PUT /policy requires non-empty source for harness nanohorizon",
            }
        canonical = json.dumps(
            {
                "code_sha256": hashlib.sha256(raw).hexdigest(),
                "harness": harness,
                "namespace": namespace,
                "name": name,
                "configuration": configuration,
                "model": model,
                "source_revision": source_revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        revision_id = f"polrev_{digest[:16]}"
        existing = self.policy_revisions.get(revision_id)
        if existing is not None:
            self.current_policy_revision_id = revision_id
            return {**self.policy_state_payload(), "idempotent": True}
        configuration_digest = "sha256:" + hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        model_digest = "sha256:" + hashlib.sha256(
            json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        installed_at = datetime.now(timezone.utc).isoformat()
        revision = PolicyRevision(
            revision_id=revision_id,
            digest=digest,
            harness=harness,
            config_id=None,
            code=bytes(raw),
            isolation_receipt={"sandbox": harness, "digest": digest},
            namespace=namespace,
            name=name,
            configuration_digest=configuration_digest,
            model_digest=model_digest,
            source_revision=str(source_revision) if source_revision is not None else None,
            installed_at=installed_at,
        )
        self.policy_revisions[revision_id] = revision
        self.current_policy_revision_id = revision_id
        self.policy_generation += 1
        self.policy_code = bytes(raw)
        self._close_policy_process()
        receipt = dict(revision.isolation_receipt)
        spawn = harness == ISOLATED_POLICY_HARNESS or (
            self.spec.default_policy_harness == ISOLATED_POLICY_HARNESS
            and harness != NANOHORIZON_HARNESS
        )
        if spawn:
            try:
                process = self._ensure_policy_process()
                receipt.update(process.isolation_receipt)
                revision.isolation_receipt = receipt
            except Exception as exc:
                receipt["spawn_error"] = str(exc)
        else:
            receipt["spawned"] = False
        return {**self.policy_state_payload(), "idempotent": False, "isolation_receipt": receipt}

    def policy_state_payload(self) -> dict[str, Any]:
        """Return the installed policy identity without source or credentials."""
        with self._state_lock:
            revision = self.policy_revisions.get(str(self.current_policy_revision_id or ""))
            if revision is None:
                return {
                    "schema_version": "synth.container-policy.v1",
                    "status": PolicyInstallStatus.NOT_INSTALLED,
                    "policy_ref": None,
                    "policy_revision_id": None,
                    "source_revision": None,
                    "configuration_digest": None,
                    "model_digest": None,
                    "installed_at": None,
                    "compatible_operations": ["rollout"],
                    "credential_state": "not_exposed",
                }
            return {
                "schema_version": "synth.container-policy.v1",
                "status": PolicyInstallStatus.INSTALLED,
                "policy_ref": {"namespace": revision.namespace, "name": revision.name},
                "policy_revision_id": revision.revision_id,
                "source_revision": revision.source_revision,
                "configuration_digest": revision.configuration_digest,
                "model_digest": revision.model_digest,
                "installed_at": revision.installed_at,
                "compatible_operations": ["rollout"],
                "credential_state": "not_exposed",
            }

    def restart_policy(self) -> dict[str, Any]:
        self._close_policy_process()
        if self.spec.default_policy_harness == "isolated_policy_process":
            self._ensure_policy_process()
        self.policy_generation += 1
        return {
            "restarted": True,
            "engine_generation": self.engine_generation,
            "policy_generation": self.policy_generation,
            "durable_logs": list(self.logs),
            "active_leases": self.active_leases,
            "isolation_receipt": (
                self.policy_process.isolation_receipt if self.policy_process else None
            ),
        }

    def register_policy_config(self, config_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if self.spec.affordances.level("bind_policy_config") == "unsupported":
            return {"error": "bind_refused", "status_code": 403, "affordance": "bind_policy_config"}
        # Policy configs are immutable, named inputs. Registering a new config
        # does not mutate the config already pinned by an in-flight rollout, so
        # concurrent optimizer runs may safely add checkpoint configs while
        # other rollouts are active. The rollout's policy_ref remains the
        # authority for selecting its config.
        cfg = PolicyConfig(
            config_id=config_id,
            harness=str(body.get("harness") or self.spec.default_policy_harness),
            config=dict(body.get("config") or body),
        )
        self.policy_configs[config_id] = cfg
        return {"config_id": config_id, "harness": cfg.harness, "engine_generation": self.engine_generation}

    def world_stop(self) -> dict[str, Any]:
        self.stopped_worlds.add(self.spec.world_ref)
        return {
            "stopped": True,
            "world_ref": self.spec.world_ref,
            "retention": self.spec.retention,
            "environment_restart": self.spec.true_checkpoint,
        }

    def restart_world(self) -> dict[str, Any]:
        """Restart only when the target advertises a proven environment checkpoint."""
        if self.spec.true_checkpoint == "unsupported":
            return {
                "error": "environment_restart_unsupported",
                "status_code": 409,
                "affordance": "true_checkpoint",
                "world_ref": self.spec.world_ref,
                "policy_generation": self.policy_generation,
            }
        return {
            "error": "environment_restart_not_implemented",
            "status_code": 501,
            "affordance": "true_checkpoint",
            "world_ref": self.spec.world_ref,
            "policy_generation": self.policy_generation,
        }

    def artifact(self, artifact_id: str) -> dict[str, Any] | None:
        row = self.artifacts.get(artifact_id)
        if row is None:
            return None
        copied = self.spec.world_ref in self.stopped_worlds and self.spec.retention == "run"
        return {**row, "available": True, "retention": self.spec.retention, "copied": copied}

    def drop_session(self, rollout_id: str) -> dict[str, Any]:
        pin = self.pins.get(rollout_id)
        if pin is None:
            return {"error": "unknown_rollout", "status_code": 404}
        pin.session_dropped = True
        return {"dropped": True, "rollout_id": rollout_id}

    def compute_reward(
        self,
        *,
        rollout_id: str | None,
        evidence: dict[str, Any] | None,
        mode: str,
        rescore: bool,
        plan_ref: str | None,
        after_sequence: int | None = None,
    ) -> dict[str, Any]:
        if (rollout_id is None) == (evidence is None):
            return {"error": "xor", "status_code": 422, "detail": "rollout_id XOR evidence"}
        if evidence is not None:
            value = evidence.get("reward")
            if value is None and isinstance(evidence.get("reward.txt"), (int, float)):
                value = float(evidence["reward.txt"])
            record = {
                "execution_id": f"eval_evidence_{_digest(evidence)}",
                "rollout_id": None,
                "status": "scored" if value is not None else "absent",
                "reward": value,
                "evaluation_plan_ref": plan_ref or self.spec.evaluation_plan_ref,
                "node_results": [
                    {
                        "node_id": "provided",
                        "kind": "script" if self.spec.reward_kind == "script" else "env_reward",
                        "authority": (
                            "trusted_scorer" if self.spec.reward_kind == "script" else "environment"
                        ),
                        "status": "scored" if value is not None else "skipped",
                        "value": value,
                    }
                ],
            }
            self.reward_by_execution_id[record["execution_id"]] = record
            return record
        pin = self.pins.get(str(rollout_id))
        if pin is None:
            return {"error": "unknown_rollout", "status_code": 404}
        env_generation_before = pin.env_generation
        step_before = self.step_calls
        start_before = self.start_session_calls
        if mode == "provisional" and not self.spec.live_reward:
            return {
                "error": "live_reward_unsupported",
                "status_code": 409,
                "status": "refused",
                "reward": None,
                "reasons": ["live_reward_unsupported"],
            }
        if mode == "terminal" and not pin.terminal:
            return {
                "error": "incomplete",
                "status_code": 409,
                "status": "incomplete",
                "missing_evidence": ["terminal_status"],
                "reward": None,
            }
        plan = plan_ref or pin.evaluation_plan_ref
        plan_outcome = classify_plan_outcome(plan)
        if plan_outcome is PlanOutcome.GATED:
            record = {
                "execution_id": f"eval_{pin.rollout_id}_gated",
                "rollout_id": pin.rollout_id,
                "status": "gated",
                "reward": None,
                "evaluation_plan_ref": plan,
                "reasons": ["gate_failed"],
                "node_results": [
                    {
                        "node_id": "gate",
                        "kind": "gate",
                        "authority": "trusted_scorer",
                        "status": "gated",
                        "value": None,
                    }
                ],
            }
            self.reward_by_execution_id[record["execution_id"]] = record
            self.reward_executions[pin.rollout_id] = record
            return record
        if plan_outcome is PlanOutcome.REFUSED:
            record = {
                "execution_id": f"eval_{pin.rollout_id}_refused",
                "rollout_id": pin.rollout_id,
                "status": "refused",
                "reward": None,
                "evaluation_plan_ref": plan,
                "reasons": ["scorer_refused"],
                "node_results": [],
            }
            self.reward_by_execution_id[record["execution_id"]] = record
            self.reward_executions[pin.rollout_id] = record
            return record

        if mode == "provisional":
            signals = self._signals_up_to(pin, after_sequence)
            if any(item is None for item in signals):
                reward = None
                status = "absent"
            else:
                reward = float(sum(float(item) for item in signals if item is not None))
                status = "scored"
            record = {
                "execution_id": f"eval_{pin.rollout_id}_prov_{after_sequence or pin.rollout_id}",
                "rollout_id": pin.rollout_id,
                "status": status,
                "reward": reward,
                "evaluation_plan_ref": plan,
                "mode": "provisional",
                "node_results": [
                    {
                        "node_id": "env_sum",
                        "kind": "env_reward",
                        "authority": "environment",
                        "status": status,
                        "value": reward,
                    }
                ],
            }
            return record

        evidence_digest = _digest(
            {
                "rollout": pin.rollout_id,
                "signals": pin.reward_signals,
                "script": pin.native_script_reward,
                "hillclimb": [node.to_dict() for node in (pin.hillclimb_nodes or ())],
            }
        )
        if not rescore:
            existing = self.reward_executions.get(pin.rollout_id)
            if (
                existing
                and existing.get("evidence_digest") == evidence_digest
                and existing.get("evaluation_plan_ref") == plan
            ):
                return existing
        reward, nodes, status, reason = self._reward_nodes(pin)
        record = {
            "execution_id": f"eval_{pin.rollout_id}_{_digest((plan, evidence_digest, rescore, uuid.uuid4().hex if rescore else 'once'))}",
            "rollout_id": pin.rollout_id,
            "status": status,
            "reward": reward,
            "evaluation_plan_ref": plan,
            "evidence_digest": evidence_digest,
            "reasons": [reason] if reason else [],
            "node_results": nodes,
            "child_rollout_id": pin.child_rollout_id,
            "env_generation": pin.env_generation,
            "step_calls_delta": self.step_calls - step_before,
            "start_session_delta": self.start_session_calls - start_before,
            "env_mutated": pin.env_generation != env_generation_before,
        }
        if pin.reward_kind == "script" and self.spec.blocking_trial == "native":
            evaluation_id = record["execution_id"]
            record["evaluation_id"] = evaluation_id
            record["http_status"] = 202
            self.evaluations[evaluation_id] = record
            self.evaluation_logs[evaluation_id] = [
                {"kind": "evaluation.started", "evaluation_id": evaluation_id},
                {"kind": "evaluation.completed", "evaluation_id": evaluation_id, "status": status},
            ]
        self.reward_by_execution_id[record["execution_id"]] = record
        self.reward_executions[pin.rollout_id] = record
        self._atomic_json(
            self._reward_path(pin.rollout_id),
            {
                "schema": "synth.containers.reward-receipt.v1",
                "receipt": record,
                "content_digest": self._receipt_digest(record),
            },
        )
        return record

    def _signals_up_to(self, pin: RolloutPin, after_sequence: int | None) -> list[float | None]:
        log = self.logs.get(pin.rollout_id)
        if log is None:
            return list(pin.reward_signals)
        values: list[float | None] = []
        for item in log.after(0):
            if item.control or item.kind != "reward_signal":
                continue
            if after_sequence is not None and item.sequence is not None and item.sequence > after_sequence:
                break
            values.append(item.payload.get("value"))
        return values

    def _reward_nodes(self, pin: RolloutPin) -> tuple[float | None, list[dict[str, Any]], str, str | None]:
        if pin.hillclimb_nodes:
            nodes = [node.to_dict() for node in pin.hillclimb_nodes]
            gates = [node for node in pin.hillclimb_nodes if node.kind == "gate"]
            if any(node.status == "gated" or node.value is None for node in gates):
                return None, nodes, "gated", "heldout_gate"
            gate_value = gates[0].value if gates else None
            return float(gate_value) if gate_value is not None else None, nodes, "scored", None
        kind = pin.reward_kind
        if kind == "env_sum":
            if any(item is None for item in pin.reward_signals):
                return None, [
                    {
                        "node_id": "env_sum",
                        "kind": "env_reward",
                        "authority": "environment",
                        "status": "skipped",
                        "value": None,
                    }
                ], "absent", "missing_reward_signal"
            total = float(
                sum(float(item) for item in pin.reward_signals if item is not None)
            )
            return total, [
                {
                    "node_id": "env_sum",
                    "kind": "env_reward",
                    "authority": "environment",
                    "status": "scored",
                    "value": total,
                }
            ], "scored", None
        if kind == "script":
            node_id = str(self.spec.script_node)
            value = pin.native_script_reward
            return value, [
                {
                    "node_id": node_id,
                    "kind": "script" if node_id == "reward.txt" else "gate",
                    "authority": "trusted_scorer",
                    "status": "scored" if value is not None else "skipped",
                    "value": value,
                }
            ], "scored" if value is not None else "absent", None
        if kind == "env_status":
            value = pin.native_script_reward
            return value, [
                {
                    "node_id": "env_status",
                    "kind": "env_reward",
                    "authority": "environment",
                    "status": "scored" if value is not None else "skipped",
                    "value": value,
                }
            ], "scored" if value is not None else "absent", None
        return None, [], "absent", "unknown_plan"

    def get_reward(self, rollout_id: str) -> dict[str, Any]:
        existing = self.reward_executions.get(rollout_id)
        if existing is None:
            return {"status": "absent", "reward": None, "rollout_id": rollout_id}
        return existing

    def product_combiner(self, bases: dict[str, float | None], required: list[str]) -> dict[str, Any]:
        values = []
        for name in required:
            item = bases.get(name)
            if item is None:
                return {"status": "absent", "reward": None, "reasons": [f"missing_basis:{name}"]}
            values.append(float(item))
        product = 1.0
        for item in values:
            product *= item
        return {"status": "scored", "reward": product}

    def events_payload(
        self,
        rollout_id: str,
        after: int,
        limit: int = 1000,
        *,
        ack: int | None = None,
    ) -> dict[str, Any]:
        log = self.logs.get(rollout_id)
        if log is None:
            return {"error": "unknown_rollout", "status_code": 404}
        if isinstance(limit, bool) or limit < 1 or limit > 10_000:
            return {"error": "invalid_page_limit", "status_code": 422}
        if ack is not None:
            try:
                log.record_ack(ack)
            except ValueError:
                return {"error": "invalid_ack", "status_code": 422}
        available = log.after(after)
        controls = [item for item in available if item.sequence is None]
        evidence = [item for item in available if item.sequence is not None]
        page = [*controls, *evidence[:limit]]
        envelopes = [item.to_dict() for item in page]
        for row in envelopes:
            row["rollout_id"] = rollout_id
            if "Authorization" in json.dumps(row) or "DIGBENCH_API_TOKEN" in json.dumps(row):
                raise RuntimeError("token_leaked_into_log")
        return {
            "rollout_id": rollout_id,
            "stream_id": log.stream_id,
            "cursor": {
                "kind": "sequence",
                "after": after,
                "high_water": log.high_water,
                "closed": log.closed,
                "next": max(
                    [after, *(item.sequence for item in page if item.sequence is not None)]
                ),
                "has_more": len(evidence) > limit,
                "chain_head": log.chain_head,
                "acked": log.last_acked,
            },
            "retention": self._journal_retention(log),
            "events": envelopes,
        }

    def _retention_ttl_seconds(self) -> int:
        raw = (self.runtime_config or {}).get("journal_retention_ttl_seconds")
        try:
            ttl = int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            ttl = 0
        return ttl if ttl > 0 else 604_800

    def _journal_retention(
        self, log: RolloutEventLog, *, now: float | None = None
    ) -> dict[str, Any]:
        ttl = self._retention_ttl_seconds()
        moment = time.time() if now is None else now
        expires_at = None
        released = False
        reason = None
        if log.closed:
            if log.last_acked >= log.high_water:
                released = True
                reason = "acked"
            if log.closed_at:
                closed_epoch = datetime.fromisoformat(
                    log.closed_at.replace("Z", "+00:00")
                ).timestamp()
                expires_epoch = closed_epoch + ttl
                expires_at = (
                    datetime.fromtimestamp(expires_epoch, tz=timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z")
                )
                if not released and moment >= expires_epoch:
                    released = True
                    reason = "ttl_expired"
        return {
            "policy": "until-acked-or-ttl",
            "ttl_seconds": ttl,
            "acked": log.last_acked,
            "high_water": log.high_water,
            "closed": log.closed,
            "released": released,
            "released_reason": reason,
            "expires_at": expires_at,
        }

    def release_retained_journals(
        self,
        *,
        now: float | None = None,
        remove: Any = None,
    ) -> list[dict[str, Any]]:
        released_rows: list[dict[str, Any]] = []
        for rollout_id, log in list(self.logs.items()):
            state = self._journal_retention(log, now=now)
            if not state["released"]:
                continue
            paths: list[Path] = []
            if log.journal_path is not None:
                paths.append(log.journal_path)
                rollout_key = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()
                paths.append(self.storage_root / "frame_assets" / rollout_key)
            row = {
                "rollout_id": rollout_id,
                "reason": state["released_reason"],
                "paths": [str(item) for item in paths],
            }
            released_rows.append(row)
            if callable(remove):
                remove(rollout_id, [str(item) for item in paths])
        return released_rows
