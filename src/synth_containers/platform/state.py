"""In-process containers-compat façade: pins, leases, policies, logs, /reward."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..event_log import (
    CONTROL_SUBSCRIBED,
    MAX_STREAMS_PER_ROLLOUT,
    STREAM_RETRY_AFTER_S,
    RolloutEventLog,
    stream_descriptor,
)
from .react import CRAFTAX_REACT_SYSTEM_PROMPT
from .affordances import bind_recipe
from .http_requests import CreateRolloutRequest, ISOLATED_POLICY_HARNESS
from .manifests import CompletedRolloutMixin
from .pin import (
    AdmissionReceipt,
    PinStatus,
    RolloutCredentialLease,
    RolloutPin,
    admission_identity_digest,
    admission_identity_payload,
    require_lease_identity,
    transition,
)
from .policy_process import DEFAULT_HEURISTIC, IsolatedPolicyProcess
from .reward_plan import PlanOutcome, classify_plan_outcome
from .runtime import runtime_for
from .seal import seal_rollout_log
from .targets import TargetSpec


def _materialize_trace_bundle(seal_path: Path, archive_path: Path, rollout_id: str) -> None:
    """Promote the platform event seal into a portable self-contained Trace V5 bundle.

    The HTTP platform is the capture authority for its own rollout.  Leaving the
    canonical bundle to an optional sidecar meant the advertised bundle endpoint
    returned 404 for every ordinary local run.  Native import supplies the
    missing canonical actor/session identity without changing the retained lite
    seal, then emits the deterministic archive served by the endpoint.
    """
    from ..tracing.adapters.native import import_native_to_bundle
    from ..tracing.store.bundle import LocalTraceBundle

    scratch = archive_path.parent / f".{rollout_id}.trace-bundle.{os.getpid()}.{threading.get_ident()}"
    source = scratch / "source.json"
    bundle_root = scratch / "bundle"
    try:
        scratch.mkdir(parents=True, exist_ok=False)
        payload = json.loads(seal_path.read_text(encoding="utf-8"))
        payload["run_id"] = rollout_id
        payload["trace_correlation_id"] = rollout_id
        source.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        bundle = LocalTraceBundle(bundle_root, bundle_id=f"bundle-{rollout_id}")
        import_native_to_bundle(source, source_format="craftax_react", bundle=bundle)
        body = bundle.archive_bytes()
        temporary = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, archive_path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _canonical_sha256(payload: Any) -> str:
    """Full sha256 of canonical JSON. Used for config, capability, and manifest digests."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _checkpoint_digest(payload: Any) -> str:
    """Return the full content address used for durable checkpoint evidence."""
    return _canonical_sha256(payload)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


TASK_INSTANCE_ID_PATTERN = r"^.*:(-?\d+)$"


def _owner_from_metadata(metadata: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not isinstance(metadata, dict):
        return None, None
    owner_id = (
        metadata.get("owner_id")
        or metadata.get("workshop_instance_id")
        or metadata.get("service_instance_id")
    )
    if owner_id is None or str(owner_id).strip() == "":
        return None, None
    kind = str(metadata.get("owner_kind") or "workshop_instance").strip() or "workshop_instance"
    return str(owner_id).strip(), kind


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


class CompatPlatform(CompletedRolloutMixin):
    def __init__(
        self,
        spec: TargetSpec,
        *,
        storage_root: str | Path,
        runtime_config: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        # Process-local runtime extensions (for example a private, pinned
        # Harbor/Dock task bundle) are deliberately not part of target
        # metadata or the trace envelope.  Runtimes may read this immutable
        # construction input, but callers cannot mutate it through HTTP.
        self.runtime_config = dict(runtime_config or {})
        # Durable state is the point of this façade: leases, seals, receipts
        # and manifests must survive the process. A silent temporary root made
        # every recovery guarantee conditional on nobody having forgotten the
        # argument, so the root is required and named by the caller.
        if storage_root is None or str(storage_root).strip() == "":
            raise ValueError("storage_root_required")
        self.storage_root = Path(storage_root)
        self.engine_generation = 1
        self.policy_generation = 1
        self.active_leases = 0
        self.logs: dict[str, RolloutEventLog] = {}
        self.stream_bindings: dict[str, tuple[str, str]] = {}
        self.pins: dict[str, RolloutPin] = {}
        self.policy_configs: dict[str, PolicyConfig] = {}
        self.policy_revisions: dict[str, PolicyRevision] = {}
        self.current_policy_revision_id: str | None = None
        self.reward_executions: dict[str, dict[str, Any]] = {}
        self.reward_by_execution_id: dict[str, dict[str, Any]] = {}
        self.evaluations: dict[str, dict[str, Any]] = {}
        self.evaluation_logs: dict[str, list[dict[str, Any]]] = {}
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.seals: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.stopped_worlds: set[str] = set()
        self.step_calls = 0
        self.start_session_calls = 0
        self.policy_code: bytes | None = None
        self.policy_process: IsolatedPolicyProcess | None = None
        self._state_lock = threading.RLock()
        self._background_done: dict[str, threading.Event] = {}
        self._background_threads: dict[str, threading.Thread] = {}
        self.execution_manifests: dict[str, dict[str, Any]] = {}
        self._stream_occupancy: dict[str, int] = {}
        self.credential_leases: dict[str, RolloutCredentialLease] = {}
        self.instance_id = str(
            (self.runtime_config.get("instance_id") if self.runtime_config else None)
            or uuid.uuid4()
        )
        self._seed_default_policies()
        self._recover_checkpoints()
        self._recover_completed_rollouts()
        self._recover_orphaned_leases()

    def _durable_key(self, rollout_id: str) -> str:
        return hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()

    def _manifest_path(self, rollout_id: str) -> Path:
        return self.storage_root / "run_manifests" / f"{self._durable_key(rollout_id)}.json"

    def _lease_path(self, rollout_id: str) -> Path:
        return self.storage_root / "leases" / f"{self._durable_key(rollout_id)}.json"

    def _admission_path(self, rollout_id: str) -> Path:
        return self.storage_root / "admissions" / f"{self._durable_key(rollout_id)}.json"

    def _trace_bundle_path(self, rollout_id: str) -> Path:
        configured = self.runtime_config.get("trace_bundle_root")
        root = Path(configured) if configured else (self.storage_root / "seals")
        return root / f"{rollout_id}.trace-bundle.zip"

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

    def _persist_lease(self, pin: RolloutPin) -> None:
        self._atomic_json(
            self._lease_path(pin.rollout_id),
            {
                "schema": "synth.containers.lease.v1",
                "target_id": self.spec.target_id,
                "instance_id": self.instance_id,
                "rollout_id": pin.rollout_id,
                "owner_id": pin.owner_id,
                "owner_kind": pin.owner_kind,
                "status": str(pin.status),
                "accepted_at": pin.accepted_at,
                "policy_ref": {
                    key: value for key, value in pin.policy_ref.items() if key != "code"
                },
                "task_instance_id": pin.task_instance_id,
                "seed": pin.seed,
                "config_digest": pin.config_digest,
                "capability_digest": pin.capability_digest,
                "identity_digest": pin.identity_digest,
                "stream_id": pin.stream_id,
            },
        )

    def _write_admission_receipt(self, pin: RolloutPin) -> None:
        receipt = pin.admission
        if receipt is None:
            raise ValueError(f"admission_receipt_missing:{pin.rollout_id}")
        path = self._admission_path(pin.rollout_id)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("identity_digest") != receipt.identity_digest:
                raise ValueError(f"admission_receipt_conflict:{pin.rollout_id}")
            return
        self._atomic_json(
            path,
            {"schema": "synth.containers.admission.v1", **receipt.to_dict()},
        )

    def _drop_lease(self, rollout_id: str) -> None:
        path = self._lease_path(rollout_id)
        if path.is_file():
            path.unlink()

    def _recover_orphaned_leases(self) -> None:
        root = self.storage_root / "leases"
        if not root.is_dir():
            return
        for path in sorted(root.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("schema") != "synth.containers.lease.v1":
                continue
            if record.get("target_id") != self.spec.target_id:
                continue
            rollout_id = str(record.get("rollout_id") or "")
            if not rollout_id:
                continue
            existing = self.pins.get(rollout_id)
            if existing is not None and existing.terminal:
                path.unlink()
                continue
            if existing is not None and not existing.terminal:
                continue
            require_lease_identity(record, source=path.name)
            pin = RolloutPin(
                rollout_id=rollout_id,
                world_ref=str(record.get("world_ref") or self.spec.world_ref),
                environment_ref=self.spec.environment_ref,
                policy_ref=dict(record["policy_ref"]),
                evaluation_plan_ref=str(
                    record.get("evaluation_plan_ref") or self.spec.evaluation_plan_ref
                ),
                task_instance_id=str(record["task_instance_id"]),
                stream_id=str(record.get("stream_id") or f"stream:{rollout_id}"),
                engine_generation=self.engine_generation,
                policy_revision_id=None,
                seed=record["seed"],
                status=PinStatus.RUNNING,
                started=True,
                owner_id=record.get("owner_id"),
                owner_kind=record.get("owner_kind"),
                accepted_at=record.get("accepted_at"),
                config_digest=str(record["config_digest"]),
                capability_digest=str(record["capability_digest"]),
                identity_digest=record.get("identity_digest"),
            )
            transition(
                pin,
                PinStatus.CRASHED,
                {"reason": "orphaned_lease", "lease_instance_id": record.get("instance_id")},
            )
            journal = self.storage_root / "event_logs" / f"{self._durable_key(rollout_id)}.jsonl"
            if journal.is_file():
                log = RolloutEventLog.recover(
                    rollout_id=rollout_id,
                    stream_id=pin.stream_id,
                    journal_path=journal,
                )
                self.logs[rollout_id] = log
                self.stream_bindings[rollout_id] = ("sse", self.spec.retention)
            self.pins[rollout_id] = pin
            path.unlink()

    def _seed_default_policies(self) -> None:
        if self.spec.default_policy_harness == "isolated_policy_process":
            return
        # Seeded configs must name every policy-identity field. A partial seed
        # is worse than none: the harness would fill the gaps from its own
        # defaults, and a result reported for "luna_med" would describe a
        # policy nobody chose. `sol_med` previously omitted compact_every and
        # both omitted max_tokens.
        self.policy_configs["luna_med"] = PolicyConfig(
            config_id="luna_med",
            harness=self.spec.default_policy_harness,
            config={
                "model": "gpt-5.6-luna",
                "effort": "medium",
                "max_tokens": 1024,
                "context_token_budget": 16000,
                "compact_at": 0.7,
                "keep_recent_messages": 8,
                "keep_recent_frames": 2,
                "observation_mode": "text",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "parse_retries": 0,
                "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT,
            },
        )
        self.policy_configs["sol_med"] = PolicyConfig(
            config_id="sol_med",
            harness=self.spec.default_policy_harness,
            config={
                "model": "gpt-5.6-sol",
                "effort": "medium",
                "max_tokens": 1024,
                "context_token_budget": 16000,
                "compact_at": 0.7,
                "keep_recent_messages": 8,
                "keep_recent_frames": 2,
                "observation_mode": "text",
                "provider": "openrouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "parse_retries": 0,
                "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT,
            },
        )
        for seed in self.spec.policy_seeds:
            self.policy_configs[seed.config_id] = PolicyConfig(
                config_id=seed.config_id,
                harness=seed.harness,
                config=dict(seed.config),
            )

    def policy_for(self, pin: RolloutPin) -> PolicyConfig | None:
        """Policy used for this pin, with a per-rollout credential overlay.

        The sampler bearer lives on ``credential_leases``, never in
        ``policy_configs``. Runtimes that need it receive a copy here.
        """
        config_id = str((pin.policy_ref or {}).get("config") or "").strip()
        registered = self.policy_configs.get(config_id) if config_id else None
        if registered is None:
            return None
        overlay = dict(registered.config)
        candidate_instruction = (pin.policy_ref or {}).get("code")
        if isinstance(candidate_instruction, str) and candidate_instruction.strip():
            base_prompt = str(overlay.get("system_prompt") or "").strip()
            overlay["system_prompt"] = (
                f"{base_prompt}\n\nCandidate instruction: {candidate_instruction.strip()}"
            )
        lease = self.credential_leases.get(pin.rollout_id)
        if lease is not None:
            target = dict(overlay.get("inference_target") or {})
            target["auth_bearer"] = lease.bearer
            overlay["inference_target"] = target
        return PolicyConfig(
            config_id=registered.config_id,
            harness=registered.harness,
            config=overlay,
            code=registered.code,
            revision=registered.revision,
        )

    def _advertised_policy_refs(self) -> list[dict[str, Any]]:
        """Pins Desktop can bind before start. Harbor = two configs; dig.bench = two harnesses."""
        if self.spec.policy_seeds:
            return [
                {"harness": seed.harness, "config": seed.config_id, "code": None}
                for seed in self.spec.policy_seeds
            ]
        if self.spec.default_policy_harness == ISOLATED_POLICY_HARNESS:
            return [{"harness": ISOLATED_POLICY_HARNESS, "config": None, "code": None}]
        # Policy configs are a live, versioned endpoint contract. A caller that
        # registers a local-MLX or standard-API config must be able to discover
        # and bind it; accepting the config while advertising only baked-in
        # defaults makes the authoritative Workshop preflight refuse it.
        return [
            {"harness": config.harness, "config": config.config_id, "code": None}
            for config in sorted(self.policy_configs.values(), key=lambda item: item.config_id)
            if config.harness == self.spec.default_policy_harness
        ]

    def capability_metadata(self) -> dict[str, Any]:
        """Stable capability advertisement hashed into contract.capability_digest."""
        payload = {
            "target_id": self.spec.target_id,
            "runtime_family": self.spec.runtime_family.value,
            "environment_ref": self.spec.environment_ref,
            "event_kinds": list(self.spec.event_kinds),
            "max_episode_steps": self.spec.max_episode_steps,
            "policy_refs": self._advertised_policy_refs(),
            "affordances": self.spec.affordances.advertised(),
            "input_schema": self._input_schema(),
        }
        dataset = self._dataset_manifest()
        if dataset is not None:
            # The rows a score was measured on are part of what the container
            # is, so the pin (revision, split digests, order) is hashed into
            # the capability digest rather than reported beside it.
            payload["dataset"] = dataset
        return payload

    def _dataset_manifest(self) -> dict[str, Any] | None:
        if self.spec.runtime_family.value == "gsm8k":
            from .gsm8k_world import dataset_manifest

            return dataset_manifest()
        return None

    def capabilities_digest(self) -> str:
        return _canonical_sha256(self.capability_metadata())

    def _action_vocabulary(self) -> list[str]:
        family = self.spec.runtime_family.value
        if family == "craftax":
            from .craftax_world import ACTIONS

            return list(ACTIONS)
        if family == "banking77":
            from .banking77_world import HELDOUT_SPLIT, TRAIN_SPLIT, load_row, split_size

            labels: set[str] = set()
            for split in (TRAIN_SPLIT, HELDOUT_SPLIT):
                for seed in range(split_size(split)):
                    row = load_row(split, seed)
                    if row is not None:
                        labels.add(row.label)
            return sorted(labels)
        return []

    def _seed_range(self) -> dict[str, int]:
        family = self.spec.runtime_family.value
        if family == "banking77":
            from .banking77_world import HELDOUT_SPLIT, TRAIN_SPLIT, split_size

            return {
                "minimum": 0,
                "maximum": max(split_size(TRAIN_SPLIT), split_size(HELDOUT_SPLIT)) - 1,
            }
        if family == "gsm8k":
            from .gsm8k_world import HELDOUT_SPLIT, TRAIN_SPLIT, split_size

            return {
                "minimum": 0,
                "maximum": max(split_size(TRAIN_SPLIT), split_size(HELDOUT_SPLIT)) - 1,
            }
        return {"minimum": -2_147_483_648, "maximum": 2_147_483_647}

    def _input_schema(self) -> dict[str, Any]:
        seed_range = self._seed_range()
        vocabulary = self._action_vocabulary()
        properties: dict[str, Any] = {
            "task_instance_id": {
                "type": "string",
                "pattern": TASK_INSTANCE_ID_PATTERN,
                "examples": ["seed:0"],
            },
            "seed": {
                "type": "integer",
                "minimum": seed_range["minimum"],
                "maximum": seed_range["maximum"],
            },
        }
        if vocabulary:
            properties["action_vocabulary"] = {
                "type": "array",
                "items": {"type": "string", "enum": vocabulary},
            }
        return {
            "type": "object",
            "required": ["task_instance_id"],
            "properties": properties,
        }

    def _policy_config_digest(
        self,
        config_id: str | None,
        harness: str,
        code: Any = None,
    ) -> str:
        cfg = self.policy_configs.get(config_id) if config_id else None
        payload: dict[str, Any] = {
            "config_id": config_id,
            "harness": cfg.harness if cfg is not None else harness,
            "config": None if cfg is None else cfg.config,
        }
        if code:
            raw = code if isinstance(code, (bytes, bytearray)) else str(code).encode("utf-8")
            payload["code_sha256"] = hashlib.sha256(bytes(raw)).hexdigest()
        return _canonical_sha256(payload)

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
        schema = self._input_schema()
        digest = self.capabilities_digest()
        policy_refs = []
        for policy_ref in self._advertised_policy_refs():
            row = dict(policy_ref)
            registered = self.policy_configs.get(str(row.get("config") or ""))
            if registered is not None:
                config = registered.config
                if config.get("model") is not None:
                    row["model"] = config["model"]
                effort = config.get("effort") or config.get("reasoning_effort")
                if effort is not None:
                    row["reasoning_effort"] = effort
            policy_refs.append(row)
        payload = {
            "world_ref": self.spec.world_ref,
            "environment_ref": self.spec.environment_ref,
            "policy_ref": {
                "harness": self.spec.default_policy_harness,
                "config": None,
                "code": None,
            },
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
            "input_schema": schema,
            "capabilities_digest": digest,
            "capabilities": {
                "protocol": "synth.container.live-eval.v1",
                "operations": {
                    "rollouts.prepare": True,
                    "rollouts.start_prepared": True,
                    "rollouts.get": True,
                    "rollouts.poll": True,
                    "reward.get": True,
                    # Every accepted start opens the durable rollout event log;
                    # terminalization validates and atomically persists its
                    # Trace V5 seal before the terminal manifest is exposed.
                    "trace_v5.capture": True,
                },
                "policy_refs": policy_refs,
                self.spec.target_id: {
                    "runtime_family": self.spec.runtime_family.value,
                    "input_schema": schema,
                },
            },
            "dataset": self._dataset_manifest(),
            "identity": {
                "target_id": self.spec.target_id,
                "instance_id": self.instance_id,
            },
            "capacity": self._capacity_snapshot(),
        }
        if contract := self._gepa_v2_contract():
            payload["optimizer_contracts"] = {"gepa": contract}
        return payload

    def _capacity_snapshot(self) -> dict[str, Any]:
        active = sum(1 for pin in self.pins.values() if pin.started and not pin.terminal and pin.simulating)
        reserved = int(self.active_leases)
        by_owner: dict[str, dict[str, int]] = {}
        for pin in self.pins.values():
            if not pin.started or pin.terminal:
                continue
            key = pin.owner_id or "unowned"
            row = by_owner.setdefault(key, {"active": 0, "reserved": 0})
            row["reserved"] += 1
            if pin.simulating:
                row["active"] += 1
        return {
            "declared": self.spec.scale_leases,
            "active": active,
            "reserved": reserved,
            "by_owner": by_owner,
        }

    def health_payload(self) -> dict[str, Any]:
        crashed = [
            {
                "rollout_id": pin.rollout_id,
                "owner_id": pin.owner_id,
                "status": str(pin.status),
            }
            for pin in self.pins.values()
            if pin.status == PinStatus.CRASHED
        ]
        return {
            "status": "ok",
            "target": self.spec.target_id,
            "runtime_family": self.spec.runtime_family.value,
            "environment_ref": self.spec.environment_ref,
            "identity": {
                "target_id": self.spec.target_id,
                "instance_id": self.instance_id,
            },
            "capacity": self._capacity_snapshot(),
            "crash_signals": crashed,
        }
    def _gepa_v2_contract(self) -> dict[str, Any] | None:
        family = self.spec.runtime_family.value
        if family == "healthbench":
            return {
                "version": "synth_optimizers.gepa.v2",
                "program_route": "/program",
                "taskset_route": "/taskset",
                "taskset_tasks_route": "/taskset/tasks",
                "rollout_route": "/rollout",
                "trace_route": "/rollouts/{rollout_id}/events",
            }
        if family == "banking77":
            return {
                "version": "synth_optimizers.gepa.v2",
                "program_route": "/program",
                "taskset_route": "/taskset",
                "rollout_route": "/rollouts",
                "prepare_route": "/rollouts/prepare",
                "trace_route": "/rollouts/{rollout_id}/events",
            }
        return None

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

    def prepare(self, rollout_id: str, transport: str, retention: str) -> dict[str, Any]:
        with self._state_lock:
            return self._prepare_locked(rollout_id, transport, retention)

    def _prepare_locked(self, rollout_id: str, transport: str, retention: str) -> dict[str, Any]:
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
        spawn_id: str | None = None
        should_run = False
        pin: RolloutPin | None = None
        log: RolloutEventLog | None = None
        rollout_id = ""
        with self._state_lock:
            result = self._start_rollout_locked(request, defer_sync=True)
            if isinstance(result, dict) and result.pop("_spawn_background", False):
                spawn_id = str(result.get("rollout_id") or "")
            else:
                rollout_id = str(result.get("rollout_id") or request.rollout_id or "")
                pin = self.pins.get(rollout_id)
                log = self.logs.get(rollout_id)
                should_run = (
                    request.submission_mode != "async"
                    and pin is not None
                    and log is not None
                    and pin.status == PinStatus.RUNNING
                    and not bool(result.get("replayed"))
                )
        if spawn_id:
            self._spawn_background(spawn_id)
            return result
        if not should_run:
            return result
        assert pin is not None and log is not None
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
            requested_identity = admission_identity_payload(
                harness=requested_harness,
                config=requested_config,
                code=request.policy_ref.code,
                task_instance_id=requested_task,
                world_ref=str(request.world_ref or self.spec.world_ref),
                evaluation_plan_ref=str(
                    request.evaluation_plan_ref or self.spec.evaluation_plan_ref
                ),
                omit_reward=request.omit_reward,
                transport=transport,
                retention=requested_retention,
                resume_from_checkpoint_id=request.resume_from_checkpoint_id,
                execution=(
                    request.execution
                    if request.submission_mode == "async"
                    else existing_pin.execution
                ),
            )
            if existing_pin.identity_digest != admission_identity_digest(requested_identity):
                return {
                    "error": "rollout_identity_conflict",
                    "status_code": 409,
                    "rollout_id": rollout_id,
                }
            replay = self._rollout_response(
                existing_pin, self.stream_descriptor_for(rollout_id)
            )
            replay["replayed"] = True
            if request.submission_mode == "async":
                return self._acceptance_payload(
                    existing_pin,
                    self.stream_descriptor_for(rollout_id),
                    replayed=True,
                )
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
        execution = request.execution if async_mode else None
        config_digest = self._policy_config_digest(config_id, harness, policy_ref.code)
        capability_digest = self.capabilities_digest()
        idempotency_key = request.idempotency_key or rollout_id
        owner_id, owner_kind = _owner_from_metadata(request.metadata)
        identity_payload = admission_identity_payload(
            harness=harness,
            config=config_id,
            code=policy_ref.code,
            task_instance_id=task_instance_id,
            world_ref=str(request.world_ref or self.spec.world_ref),
            evaluation_plan_ref=str(request.evaluation_plan_ref or self.spec.evaluation_plan_ref),
            omit_reward=request.omit_reward,
            transport=transport,
            retention=retention,
            resume_from_checkpoint_id=request.resume_from_checkpoint_id,
            execution=execution,
        )
        identity_digest = admission_identity_digest(identity_payload)
        accepted_at = _utc_now()
        admission = AdmissionReceipt(
            identity_digest=identity_digest,
            rollout_id=rollout_id,
            task_instance_id=task_instance_id,
            seed=seed_i,
            world_ref=str(request.world_ref or self.spec.world_ref),
            evaluation_plan_ref=str(request.evaluation_plan_ref or self.spec.evaluation_plan_ref),
            omit_reward=request.omit_reward,
            transport=transport,
            retention=retention,
            resume_from_checkpoint_id=request.resume_from_checkpoint_id,
            execution=execution,
            config_digest=config_digest,
            capability_digest=capability_digest,
            policy_harness=harness,
            policy_config=config_id,
            accepted_at=accepted_at,
        )
        pin = RolloutPin(
            rollout_id=rollout_id,
            world_ref=str(request.world_ref or self.spec.world_ref),
            environment_ref=self.spec.environment_ref,
            policy_ref={"harness": harness, "config": config_id, "code": policy_ref.code},
            evaluation_plan_ref=str(request.evaluation_plan_ref or self.spec.evaluation_plan_ref),
            task_instance_id=task_instance_id,
            stream_id=descriptor["id"],
            engine_generation=self.engine_generation,
            policy_revision_id=self.current_policy_revision_id,
            seed=seed_i,
            usage=None,
            omit_reward=request.omit_reward,
            outcome=request.outcome,
            reward_kind=self.spec.reward_kind,
            checkpoint_schedule=request.checkpoint_schedule,
            resume_from_checkpoint_id=request.resume_from_checkpoint_id,
            config_digest=config_digest,
            capability_digest=capability_digest,
            execution=execution,
            idempotency_key=idempotency_key,
            accepted_at=accepted_at,
            owner_id=owner_id,
            owner_kind=owner_kind,
            identity_digest=identity_digest,
            admission=admission,
        )
        self.pins[rollout_id] = pin
        self.active_leases += 1
        self._write_admission_receipt(pin)
        self._persist_lease(pin)
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
                    "config_digest": config_digest,
                    "capability_digest": capability_digest,
                    "owner_id": owner_id,
                    "owner_kind": owner_kind,
                    "instance_id": self.instance_id,
                },
            )
        pin.started = True
        transition(pin, PinStatus.RUNNING, {"reason": "admitted"})
        if async_mode:
            accepted = self._acceptance_payload(pin, descriptor)
            if execution == "background":
                accepted["_spawn_background"] = True
            return accepted
        if defer_sync:
            return self._rollout_response(pin, descriptor)
        try:
            self._simulate_or_fail(pin, log)
        finally:
            self.active_leases = max(0, self.active_leases - 1)
        return self._rollout_response(pin, descriptor)

    def _acceptance_payload(
        self,
        pin: RolloutPin,
        descriptor: dict[str, Any],
        *,
        replayed: bool = False,
    ) -> dict[str, Any]:
        """202 async receipt. Same-ID replay returns this record and spawns nothing."""
        payload = {
            "accepted": True,
            "rollout_id": pin.rollout_id,
            "status": str(pin.status),
            # Admission freezes these identities. Async callers must not need
            # a second GET to learn which exact policy and runtime generation
            # accepted their work, especially while configs continue to be
            # registered for other optimizer candidates.
            "policy_ref": dict(pin.policy_ref),
            "engine_generation": pin.engine_generation,
            "policy_revision_id": pin.policy_revision_id,
            "stream": descriptor,
            "lease": {
                "held": not pin.terminal,
                "active_leases": self.active_leases,
                "scale_leases": self.spec.scale_leases,
            },
            "contract": {
                "config_digest": pin.config_digest,
                "capability_digest": pin.capability_digest,
            },
            "idempotency_key": pin.idempotency_key,
            "execution": pin.execution,
            "owner_id": pin.owner_id,
            "owner_kind": pin.owner_kind,
        }
        if replayed:
            payload["replayed"] = True
        return payload

    def complete_rollout(self, rollout_id: str) -> dict[str, Any]:
        wait_for: threading.Event | None = None
        with self._state_lock:
            pin = self.pins.get(rollout_id)
            log = self.logs.get(rollout_id)
            if pin is None or log is None:
                return {"error": "unknown_rollout", "status_code": 404}
            if pin.terminal:
                return self._rollout_response(pin, self.stream_descriptor_for(rollout_id))
            if pin.execution == "background":
                wait_for = self._background_done.setdefault(rollout_id, threading.Event())
            else:
                try:
                    self._simulate_or_fail(pin, log)
                finally:
                    self.active_leases = max(0, self.active_leases - 1)
                return self._rollout_response(pin, self.stream_descriptor_for(rollout_id))
        wait_for.wait(timeout=120)
        with self._state_lock:
            pin = self.pins[rollout_id]
            return self._rollout_response(pin, self.stream_descriptor_for(rollout_id))

    def _spawn_background(self, rollout_id: str) -> None:
        with self._state_lock:
            if rollout_id in self._background_threads:
                return
            pin = self.pins.get(rollout_id)
            if pin is None or pin.terminal or pin.execution != "background":
                return
            self._background_done.setdefault(rollout_id, threading.Event())
            thread = threading.Thread(
                target=self._run_background_rollout,
                args=(rollout_id,),
                name=f"rollout-bg-{rollout_id}",
                daemon=True,
            )
            self._background_threads[rollout_id] = thread
        thread.start()

    def _run_background_rollout(self, rollout_id: str) -> None:
        done = self._background_done.setdefault(rollout_id, threading.Event())
        pin: RolloutPin | None = None
        log: RolloutEventLog | None = None
        try:
            with self._state_lock:
                pin = self.pins.get(rollout_id)
                log = self.logs.get(rollout_id)
                if pin is None or log is None or pin.terminal or pin.simulating:
                    return
                pin.simulating = True
            try:
                self._simulate_or_fail(pin, log)
            finally:
                with self._state_lock:
                    pin.simulating = False
                    self.active_leases = max(0, self.active_leases - 1)
        finally:
            done.set()

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

    def acquire_stream(self, rollout_id: str) -> dict[str, Any] | None:
        with self._state_lock:
            current = self._stream_occupancy.get(rollout_id, 0)
            if current >= MAX_STREAMS_PER_ROLLOUT:
                return {
                    "error": "stream_backpressure",
                    "status_code": 429,
                    "retry_after": STREAM_RETRY_AFTER_S,
                    "max_streams": MAX_STREAMS_PER_ROLLOUT,
                    "active_streams": current,
                    "rollout_id": rollout_id,
                }
            self._stream_occupancy[rollout_id] = current + 1
            return None

    def release_stream(self, rollout_id: str) -> None:
        with self._state_lock:
            current = self._stream_occupancy.get(rollout_id, 0)
            if current <= 1:
                self._stream_occupancy.pop(rollout_id, None)
            else:
                self._stream_occupancy[rollout_id] = current - 1

    def _rollout_response(self, pin: RolloutPin, descriptor: dict[str, Any]) -> dict[str, Any]:
        return {
            "rollout_id": pin.rollout_id,
            "status": str(pin.status),
            "world_ref": pin.world_ref,
            "environment_ref": pin.environment_ref,
            "policy_ref": pin.policy_ref,
            "evaluation_plan_ref": pin.evaluation_plan_ref,
            "task_instance_id": pin.task_instance_id,
            "stream": descriptor,
            "usage": pin.usage,
            "child_rollout_id": pin.child_rollout_id,
            "child_resource_ref": pin.child_resource_ref,
            "engine_generation": pin.engine_generation,
            "policy_revision_id": pin.policy_revision_id,
            "terminated": pin.terminal,
            "truncated": pin.status == PinStatus.TRUNCATED,
            "resume_from_checkpoint_id": pin.resume_from_checkpoint_id,
            "scheduled_checkpoints": pin.scheduled_checkpoints,
            "trace": self._sealed_trace_reference(pin.rollout_id),
            "config_digest": pin.config_digest,
            "capability_digest": pin.capability_digest,
            "owner_id": pin.owner_id,
            "owner_kind": pin.owner_kind,
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
        bundle_path = self._trace_bundle_path(rollout_id)
        has_bundle = bundle_path.is_file()
        return {
            "schema_version": seal.get("schema_version"),
            "trace_id": seal.get("trace_id"),
            "content_digest": seal.get("content_digest"),
            "event_count": len(seal.get("events") or []),
            "high_water": seal.get("high_water"),
            "closed": seal.get("closed"),
            "url": f"/rollouts/{rollout_id}/trace",
            "bundle_url": f"/rollouts/{rollout_id}/trace/bundle",
            "kind": "trace_v5_bundle" if has_bundle else "lite_seal",
            "inspectable": has_bundle,
        }

    def _simulate_or_fail(self, pin: RolloutPin, log: RolloutEventLog) -> None:
        """Run the rollout, terminalizing the pin even when it raises.

        `register_policy_config` refuses while any pin is started and not
        terminal. A rollout that raised before its runtime could record an
        outcome — a malformed policy config, say — left its pin pinned forever,
        so every later bind returned 409 and the container had to be restarted
        to accept work again. The error still propagates; it just no longer
        takes the container down with it. Terminalization is the same path
        runtime ``_fail`` uses: status + closed events, seal, completed
        manifest, drop lease.
        """
        try:
            self._simulate(pin, log)
        except BaseException as exc:
            self._fail(
                pin,
                log,
                reason="runtime_exception",
                error_type=type(exc).__name__,
            )
            raise

    def _fail(
        self,
        pin: RolloutPin,
        log: RolloutEventLog,
        *,
        reason: str,
        error_type: str | None = None,
    ) -> None:
        """One terminalization path: status events, seal, completed manifest, drop lease."""
        if not pin.terminal:
            evidence: dict[str, Any] = {"reason": reason}
            if error_type:
                evidence["error_type"] = error_type
            transition(pin, PinStatus.FAILED, evidence)
        if log is not None and not log.closed:
            payload: dict[str, Any] = {"status": str(pin.status), "reason": reason}
            if error_type:
                payload["error_type"] = error_type
            log.append("env.episode.closed", payload)
            log.append("status", payload)
            high_water = log.high_water
            log.append("capture.high_water", {"high_water": high_water})
            log.append("capture.closed", {"high_water": high_water})
            log.mark_closed()
        self._seal_and_persist(pin, log)

    def _hold_if_requested(self, pin: RolloutPin) -> None:
        raw = (self.runtime_config or {}).get("simulate_hold_path")
        if not raw:
            return
        marker = Path(str(raw))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(pin.rollout_id, encoding="utf-8")
        while marker.exists():
            time.sleep(0.05)

    def _seal_and_persist(self, pin: RolloutPin, log: RolloutEventLog) -> None:
        if pin.rollout_id in self.seals:
            self._write_execution_manifest(pin, log)
            self._persist_completed_rollout(pin)
            self._drop_lease(pin.rollout_id)
            return
        if log is None:
            self._drop_lease(pin.rollout_id)
            return
        pin.completed_at = pin.completed_at or _utc_now()
        self.seals[pin.rollout_id] = seal_rollout_log(
            log,
            pin={
                "world_ref": pin.world_ref,
                "environment_ref": pin.environment_ref,
                "policy_ref": pin.policy_ref,
                "evaluation_plan_ref": pin.evaluation_plan_ref,
                "task_instance_id": pin.task_instance_id,
                "config_digest": pin.config_digest,
                "capability_digest": pin.capability_digest,
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
        archive_path = self._trace_bundle_path(pin.rollout_id)
        if not archive_path.is_file():
            _materialize_trace_bundle(seal_path, archive_path, pin.rollout_id)
        self._write_execution_manifest(pin, log)
        self._persist_completed_rollout(pin)
        self._drop_lease(pin.rollout_id)

    def _simulate(self, pin: RolloutPin, log: RolloutEventLog) -> None:
        pin.env_generation += 1
        self._hold_if_requested(pin)
        runtime_for(self.spec).simulate(self, pin, log)
        if not pin.terminal:
            transition(pin, PinStatus.COMPLETED, {"reason": "runtime_returned"})
        self._seal_and_persist(pin, log)

    def _execution_manifest_path(self, rollout_id: str) -> Path:
        return self.storage_root / "seals" / f"{rollout_id}.manifest.json"

    def _taxonomy_counters(self, log: RolloutEventLog) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in log.after(0):
            if item.control:
                continue
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    def _write_execution_manifest(self, pin: RolloutPin, log: RolloutEventLog) -> dict[str, Any]:
        """Write-once terminal execution manifest next to the sealed trace."""
        path = self._execution_manifest_path(pin.rollout_id)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            self.execution_manifests[pin.rollout_id] = existing
            return existing
        seal = self.seals.get(pin.rollout_id) or {}
        body = {
            "schema": "synth.containers.execution-manifest.v1",
            "rollout_id": pin.rollout_id,
            "status": str(pin.status),
            "taxonomy": {"event_kind_counts": self._taxonomy_counters(log)},
            "usage": pin.usage,
            "reward": {
                "signals": list(pin.reward_signals),
                "native_script_reward": pin.native_script_reward,
            },
            "digests": {
                "config_digest": pin.config_digest,
                "capability_digest": pin.capability_digest,
            },
            "trace_digest": seal.get("content_digest"),
            "timestamps": {
                "accepted_at": pin.accepted_at,
                "completed_at": pin.completed_at,
            },
        }
        manifest = {**body, "content_digest": _canonical_sha256(body)}
        self._atomic_json(path, manifest)
        self.execution_manifests[pin.rollout_id] = manifest
        return manifest

    def get_execution_manifest(self, rollout_id: str) -> dict[str, Any]:
        row = self.execution_manifests.get(rollout_id)
        if row is not None:
            return row
        path = self._execution_manifest_path(rollout_id)
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            self.execution_manifests[rollout_id] = row
            return row
        return {"error": "manifest_not_sealed", "status_code": 404, "rollout_id": rollout_id}

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
        raw = code if isinstance(code, (bytes, bytearray)) else str(code or "").encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()[:16]
        revision_id = f"polrev_{digest}"
        revision = PolicyRevision(
            revision_id=revision_id,
            digest=digest,
            harness=harness,
            config_id=None,
            code=bytes(raw),
            isolation_receipt={"sandbox": "isolated_policy_process", "digest": digest},
        )
        self.policy_revisions[revision_id] = revision
        self.current_policy_revision_id = revision_id
        self.policy_generation += 1
        self.policy_code = bytes(raw)
        self._close_policy_process()
        receipt = dict(revision.isolation_receipt)
        try:
            process = self._ensure_policy_process()
            receipt.update(process.isolation_receipt)
            revision.isolation_receipt = receipt
        except Exception as exc:
            receipt["spawn_error"] = str(exc)
        return {
            "policy_revision_id": revision_id,
            "digest": digest,
            "engine_generation": self.engine_generation,
            "isolation_receipt": receipt,
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
        existing = self.policy_configs.get(config_id)
        if existing is not None:
            same = existing.harness == cfg.harness and existing.config == cfg.config
            if not same:
                return {
                    "error": "policy_config_conflict",
                    "status_code": 409,
                    "config_id": config_id,
                    "detail": "policy configs are immutable; a different body for the same id is refused",
                }
            return {
                "config_id": config_id,
                "harness": existing.harness,
                "engine_generation": self.engine_generation,
                "replayed": True,
            }
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

    def trace_for(self, rollout_id: str) -> dict[str, Any]:
        """Return the lite seal, or refuse an unsealed crashed log."""
        seal = self.seals.get(rollout_id)
        if seal is not None:
            return seal
        pin = self.pins.get(rollout_id)
        if pin is not None and pin.status == PinStatus.CRASHED:
            return {
                "error": "unsealed_log",
                "status_code": 409,
                "rollout_id": rollout_id,
                "task_instance_id": pin.task_instance_id,
            }
        return {"error": "trace_not_sealed", "status_code": 404, "rollout_id": rollout_id}

    def get_trace_bundle(self, rollout_id: str) -> dict[str, Any]:
        """Serve a capture-supervisor Trace V5 bundle archive when one exists.

        Lite seals are not bundles. Workshop tries this route first and treats
        404 as fallback to GET /rollouts/{id}/trace. Lying with a zip of the
        lite seal would make the inspector claim inspectability it does not have.
        """
        if rollout_id not in self.pins and rollout_id not in self.seals:
            return {"error": "unknown_rollout", "status_code": 404, "rollout_id": rollout_id}
        path = self._trace_bundle_path(rollout_id)
        if not path.is_file():
            return {
                "error": "trace_bundle_absent",
                "status_code": 404,
                "rollout_id": rollout_id,
                "kind": "lite_seal",
                "inspectable": False,
                "fallback": f"/rollouts/{rollout_id}/trace",
            }
        return {"path": str(path), "media_type": "application/zip"}

    def cancel_rollout(self, rollout_id: str, *, owner_id: str | None = None) -> dict[str, Any]:
        pin = self.pins.get(rollout_id)
        log = self.logs.get(rollout_id)
        if pin is None:
            return {"error": "unknown_rollout", "status_code": 404, "rollout_id": rollout_id}
        if owner_id is not None and pin.owner_id and pin.owner_id != owner_id:
            return {
                "error": "owner_mismatch",
                "status_code": 403,
                "rollout_id": rollout_id,
                "owner_id": pin.owner_id,
            }
        if pin.terminal:
            return self._rollout_response(pin, self.stream_descriptor_for(rollout_id) if log else {"id": pin.stream_id})
        held = not pin.terminal
        transition(
            pin,
            PinStatus.CANCELLED,
            {"reason": "cancel_requested", "owner_id": owner_id},
        )
        if log is not None and not log.closed:
            log.append(
                "status",
                {
                    "status": "cancelled",
                    "completion": "infra_complete",
                    "owner_id": pin.owner_id,
                },
            )
            log.mark_closed()
        if held:
            self.active_leases = max(0, self.active_leases - 1)
        self._drop_lease(rollout_id)
        descriptor = self.stream_descriptor_for(rollout_id) if log else {"id": pin.stream_id}
        return self._rollout_response(pin, descriptor)

    def cleanup_owned(self, owner_id: str) -> dict[str, Any]:
        if not str(owner_id or "").strip():
            return {"error": "owner_id_required", "status_code": 422}
        cancelled: list[str] = []
        skipped: list[str] = []
        for pin in list(self.pins.values()):
            if pin.owner_id != owner_id:
                skipped.append(pin.rollout_id)
                continue
            if pin.terminal:
                continue
            result = self.cancel_rollout(pin.rollout_id, owner_id=owner_id)
            if result.get("error"):
                skipped.append(pin.rollout_id)
            else:
                cancelled.append(pin.rollout_id)
        return {
            "owner_id": owner_id,
            "cancelled": cancelled,
            "skipped": [rid for rid in skipped if self.pins.get(rid) and self.pins[rid].owner_id != owner_id],
            "instance_id": self.instance_id,
        }

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

    def events_payload(self, rollout_id: str, after: int, limit: int = 1000) -> dict[str, Any]:
        log = self.logs.get(rollout_id)
        if log is None:
            return {"error": "unknown_rollout", "status_code": 404}
        if isinstance(limit, bool) or limit < 1 or limit > 10_000:
            return {"error": "invalid_page_limit", "status_code": 422}
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
            },
            "events": envelopes,
        }
