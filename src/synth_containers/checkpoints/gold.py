"""Native Gold engine + scripted-policy joint cuts using portable checkpoint v1.

The foreground local adapter owns its engine. This cannot manufacture Rhodes
admission or restore arbitrary live/shared environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

from ..eval_models import EvaluationRunDeclarationV1, ScriptedCandidateV1
from ..package_models import content_digest
from ..tracing.atomic import durable_write
from ..package_models import content_bytes as canonical_bytes
from .contracts import ApplicationCheckpoint, ParticipantSnapshot, ParticipantCut, ParticipantIdentity, SessionFence
from .portable_models import (
    CheckpointExecutionV1, CheckpointComponentV1, CheckpointCursorV1, CheckpointLiabilityV1,
    PortableCheckpointManifestV1, CheckpointCommitAnchorV1, CheckpointRestoreRequestV1,
    CheckpointRuntimeSupportV1,
    RestoreMode,
)
from .portable import export_checkpoint_bundle, load_checkpoint_bundle, CheckpointBundleStore, check_checkpoint_restore
from .storage import FileArtifactStore

GOLD_CODEC = content_digest({"codec": "gold-http-base64-checkpoint-v1", "scope": "native-world-rng-events"})
SCRIPT_CODEC = content_digest({"codec": "scripted-actions-v1", "scope": "history-pending-plan-cursor"})


def execution_identity(run: EvaluationRunDeclarationV1, candidate: ScriptedCandidateV1,
                       store: FileArtifactStore) -> CheckpointExecutionV1:
    assert run.plan.image_digest is not None
    # The full runtime configuration includes limits and checkpoint cadence.
    return CheckpointExecutionV1(source=run.plan.source_digest, image=run.plan.image_digest,
        dependencies=run.plan.dependencies_digest, task=content_digest(run.plan.task),
        world=content_digest(run.plan.task.get("world", {})), scorer=run.plan.scorer_digest,
        protocol=content_digest({"adapter": run.plan.adapter, "capture": run.plan.capture}),
        configuration=store.put(canonical_bytes(run.plan.model_dump(mode="json"))),
        policies=store.put(canonical_bytes(candidate.model_dump(mode="json"))))


def commit_gold_checkpoint(
    root: Path, run: EvaluationRunDeclarationV1, candidate: ScriptedCandidateV1,
    *, world, rows: list[dict], chain_head: str, calls: int, active_ns: int, terminal: bool,
) -> Path:
    """Publish complete blobs then portable manifest; append the owning commit last."""
    store = FileArtifactStore(root / "objects", namespace="local-gold-checkpoint")
    environment = world.checkpoint()
    position = len(rows)
    policy = {"schema_version": "synth.scripted-policy-state.v1", "candidate": candidate.model_dump(mode="json"),
        "history": [{"action": row["action"], "reward": row["reward"]} for row in rows],
        "position": position, "calls": calls, "active_ns": active_ns,
        "pending_plan": list(candidate.actions[position % len(candidate.actions):]) if candidate.repeat else list(candidate.actions[position:]),
        "chain_head": chain_head}
    operation, checkpoint = uuid4(), uuid4()
    snapshots = []
    for name, payload in (("environment", environment), ("policy", policy)):
        snapshots.append(ParticipantSnapshot(cut=ParticipantCut(operation=operation, receipt=uuid4(), pending_operations=0, unresolved_operations=0,
            participant=ParticipantIdentity(name=name, service_identity="local-gold-" + name,
                origin_instance=uuid4(), semantics="true_environment_snapshot")),
            artifact=store.put(canonical_bytes(payload))))
    application = ApplicationCheckpoint(checkpoint=checkpoint, operation=operation,
        fence=SessionFence(connector=UUID(run.lineage_id), session=UUID(run.lineage_id),
            source_generation=position + 1, generation=position + 2, permit_barrier_ms=0),
        semantics="true_environment_snapshot", participants=tuple(snapshots))
    components = (
        CheckpointComponentV1(participant="environment", codec=GOLD_CODEC, coverage=("state", "rng", "native_events")),
        CheckpointComponentV1(participant="policy", codec=SCRIPT_CODEC, coverage=("history", "pending_plan", "cursor", "usage")),
    )
    manifest = PortableCheckpointManifestV1(application=store.put(application.canonical_bytes()),
        checkpoint=checkpoint, run_id=run.run_id, lineage_id=UUID(run.lineage_id), sequence=position,
        chain_head=chain_head, execution=execution_identity(run, candidate, store), components=components,
        cursors=tuple(CheckpointCursorV1(name=name, position=position, chain_head=chain_head)
                      for name in ("event", "action", "observation", "reward")),
        liabilities=(CheckpointLiabilityV1(provider="scripted", resource="calls", unit="count", consumed=str(calls), outstanding="0"),
            CheckpointLiabilityV1(provider="local", resource="active_time", unit="nanoseconds", consumed=str(active_ns), outstanding="0"),
            CheckpointLiabilityV1(provider="scripted", resource="cost", unit="usd", consumed="0", outstanding="0")),
        clock_policy=run.plan.limits.clock_policy, terminal=terminal,
        modes=("inspect", "replay", "strict_continue", "same_policy_fork", "edited_policy_fork"))
    destination = root / "bundles" / (f"{position:08d}-" + str(checkpoint))
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_checkpoint_bundle(manifest, store, destination)
    durable_write(root / "commits" / f"{position:08d}-{manifest.digest()[7:]}.json",
        canonical_bytes({"manifest": manifest.digest(), "sequence": position, "chain_head": chain_head,
                         "bundle": str(destination.relative_to(root))}))
    return destination


def restore_gold_checkpoint(
    bundle: Path, run: EvaluationRunDeclarationV1, candidate: ScriptedCandidateV1, *,
    world, mode: RestoreMode = "strict_continue", child_budget: dict | None = None,
    expected_commit: CheckpointCommitAnchorV1,
):
    inspection = load_checkpoint_bundle(bundle)
    manifest = inspection.manifest
    reader = CheckpointBundleStore(bundle, namespace=manifest.application.namespace)
    target_store = FileArtifactStore(bundle.parent.parent / "restore-inputs", namespace=manifest.application.namespace)
    target = execution_identity(run, candidate, target_store)
    request = CheckpointRestoreRequestV1(mode=mode, source_run_id=manifest.run_id,
        source_lineage_id=manifest.lineage_id, target_run_id=run.run_id, target_lineage_id=UUID(run.lineage_id),
        execution=target, commit=expected_commit,
        child_budget=target_store.put(canonical_bytes(child_budget)) if child_budget is not None else None)
    support = CheckpointRuntimeSupportV1(modes=manifest.modes, components=(
        CheckpointComponentV1(participant="environment", codec=GOLD_CODEC, coverage=("state", "rng", "native_events")),
        CheckpointComponentV1(participant="policy", codec=SCRIPT_CODEC, coverage=("history", "pending_plan", "cursor", "usage"))),
        required_cursors=("event", "action", "observation", "reward"), trusted_decoders=(GOLD_CODEC, SCRIPT_CODEC),
        clock_policy=run.plan.limits.clock_policy)
    report = check_checkpoint_restore(manifest, reader, request, support)
    if not report.declaration_supported:
        raise ValueError(";".join(report.blockers))
    application = ApplicationCheckpoint.model_validate_json(reader.get(manifest.application))
    values = {item.cut.participant.name: json.loads(reader.get(item.artifact)) for item in application.participants}
    policy = values["policy"]
    if policy["position"] != manifest.sequence or policy["chain_head"] != manifest.chain_head:
        raise ValueError("native_checkpoint_cursor_inconsistent")
    result = world.restore(values["environment"]["blob"])
    # Prime the native event prefix so replayed pre-cut events are not emitted as
    # new effects; the native world owns the snapshot/RNG codec.
    world.drain_native_events()
    return manifest, policy, result
