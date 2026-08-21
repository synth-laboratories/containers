"""Write-once completed-rollout manifests and reopen recovery."""

from __future__ import annotations

import json

from ..event_log import RolloutEventLog, validate_rollout_id
from .pin import PinStatus, RewardNode, RolloutPin, admission_from_raw
from .seal import seal_rollout_log, validate_rollout_seal


class CompletedRolloutMixin:
    def _persist_completed_rollout(self, pin: RolloutPin) -> None:
        binding = self.stream_bindings.get(pin.rollout_id)
        if binding is None:
            binding = ("poll", self.spec.retention)
            self.stream_bindings[pin.rollout_id] = binding
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
                "status": str(pin.status),
                "transitions": list(pin.transitions),
                "identity_digest": pin.identity_digest,
                "admission": pin.admission.to_dict() if pin.admission is not None else None,
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
                "config_digest": pin.config_digest,
                "capability_digest": pin.capability_digest,
                "execution": pin.execution,
                "idempotency_key": pin.idempotency_key,
                "accepted_at": pin.accepted_at,
                "completed_at": pin.completed_at,
                "owner_id": pin.owner_id,
                "owner_kind": pin.owner_kind,
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
                status=PinStatus(str(raw_pin["status"])),
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
                config_digest=raw_pin.get("config_digest"),
                capability_digest=raw_pin.get("capability_digest"),
                execution=raw_pin.get("execution"),
                idempotency_key=raw_pin.get("idempotency_key"),
                accepted_at=raw_pin.get("accepted_at"),
                completed_at=raw_pin.get("completed_at"),
                owner_id=raw_pin.get("owner_id"),
                owner_kind=raw_pin.get("owner_kind"),
                transitions=list(raw_pin.get("transitions") or []),
                identity_digest=raw_pin.get("identity_digest"),
                admission=admission_from_raw(raw_pin.get("admission")),
            )
            self.logs[rollout_id] = log
            self.stream_bindings[rollout_id] = (str(binding[0]), str(binding[1]))
            self.pins[rollout_id] = pin
            self.seals[rollout_id] = seal
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
