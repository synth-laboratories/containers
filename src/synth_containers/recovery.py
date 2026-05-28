from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .contracts import CheckpointResumeContract
from .serde import JsonDataclassMixin


class RecoveryActionKind(StrEnum):
    RESUME_FROM_CHECKPOINT = "resume_from_checkpoint"
    REPLAY_FROM_REQUEST_SNAPSHOT = "replay_from_request_snapshot"
    RESTART_FRESH = "restart_fresh"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"
    NONE = "none"


class ArtifactTrustLevel(StrEnum):
    TRUSTED = "trusted"
    PARTIAL = "partial"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class TransformFailure(JsonDataclassMixin):
    transform_id: str = ""
    retriable: bool = False
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RecoveryPoint(JsonDataclassMixin):
    checkpoint_id: str = ""
    rollout_id: str = ""
    reward: float | None = None
    restore_semantics: str = ""
    resume_eligible: bool = False
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunRecoveryProjection(JsonDataclassMixin):
    run_id: str
    profile: str
    recovery_action: RecoveryActionKind = RecoveryActionKind.NONE
    artifact_trust_level: ArtifactTrustLevel = ArtifactTrustLevel.PARTIAL
    supports_true_resume: bool = False
    supports_request_snapshot_replay: bool = False
    supports_fresh_restart: bool = True
    resumable: bool = False
    replayable: bool = False
    highest_quality_recovery_point: RecoveryPoint | None = None
    resume_blockers: list[str] = field(default_factory=list)
    transform_failures: list[TransformFailure] = field(default_factory=list)
    operator_next_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def derive_run_recovery_projection(
    *,
    run_id: str,
    profile: str,
    checkpoint_resume: CheckpointResumeContract,
    run_outcome: str,
    run_phase: str,
    checkpoints: list[dict[str, Any]] | None = None,
    transform_failures: list[TransformFailure] | None = None,
) -> RunRecoveryProjection:
    checkpoint_rows = [dict(item) for item in list(checkpoints or []) if isinstance(item, dict)]
    failures = list(transform_failures or [])
    eligible_checkpoints = [row for row in checkpoint_rows if bool(row.get("resume_eligible") or row.get("restore_eligible"))]
    eligible_checkpoints.sort(
        key=lambda row: (
            float(row.get("reward") or 0.0),
            float(row.get("frontier_score") or 0.0),
            str(row.get("checkpoint_id") or ""),
        ),
        reverse=True,
    )
    best_checkpoint = eligible_checkpoints[0] if eligible_checkpoints else None
    best_checkpoint_restore_semantics = str(
        (best_checkpoint or {}).get("restore_semantics")
        or (best_checkpoint or {}).get("checkpoint_semantics")
        or ""
    ).strip()
    supports_true_resume = bool(
        checkpoint_resume.resume_support
        and checkpoint_resume.checkpoint_support
        and (
            checkpoint_resume.true_environment_snapshot
            or checkpoint_resume.checkpoint_semantics == "true_environment_snapshot"
            or checkpoint_resume.restore_semantics == "true_environment_snapshot"
            or checkpoint_resume.checkpoint_semantics == "codex_session_workspace_snapshot"
            or checkpoint_resume.restore_semantics == "codex_session_workspace_snapshot"
            or best_checkpoint_restore_semantics == "true_environment_snapshot"
            or best_checkpoint_restore_semantics == "codex_session_workspace_snapshot"
        )
    )
    supports_request_snapshot_replay = bool(
        checkpoint_resume.resume_support
        and (
            checkpoint_resume.resume_semantics == "request_snapshot_replay"
            or checkpoint_resume.restore_semantics == "request_snapshot_replay"
            or checkpoint_resume.resume_mode == "request_snapshot_replay"
        )
    )
    resume_blockers = sorted(
        {
            str(blocker).strip()
            for checkpoint in checkpoint_rows
            for blocker in list(checkpoint.get("resume_blockers") or [])
            if str(blocker).strip()
        }
    )
    outcome = str(run_outcome or "").strip().lower()
    phase = str(run_phase or "").strip().lower()
    retriable_failure = any(bool(failure.retriable) for failure in failures)
    has_failure = outcome == "failed" or phase == "failed" or bool(failures)
    if outcome == "succeeded":
        action = RecoveryActionKind.NONE
        trust = ArtifactTrustLevel.TRUSTED
        next_action = "No recovery action needed; artifacts are trusted."
    elif best_checkpoint is not None and supports_true_resume:
        action = RecoveryActionKind.RESUME_FROM_CHECKPOINT
        trust = ArtifactTrustLevel.PARTIAL
        next_action = f"Resume from checkpoint {best_checkpoint.get('checkpoint_id') or '<unknown>'}."
    elif supports_request_snapshot_replay:
        action = RecoveryActionKind.REPLAY_FROM_REQUEST_SNAPSHOT
        trust = ArtifactTrustLevel.PARTIAL
        next_action = "Replay the run from the latest request-boundary snapshot."
    elif retriable_failure or has_failure:
        action = RecoveryActionKind.RESTART_FRESH
        trust = ArtifactTrustLevel.INVALID if failures else ArtifactTrustLevel.PARTIAL
        next_action = "Restart the run fresh; no authoritative resumable checkpoint is available."
    else:
        action = RecoveryActionKind.MANUAL_INTERVENTION_REQUIRED
        trust = ArtifactTrustLevel.INVALID
        next_action = "Manual investigation required before resuming or restarting."
    recovery_point = (
        RecoveryPoint(
            checkpoint_id=str(best_checkpoint.get("checkpoint_id") or ""),
            rollout_id=str(best_checkpoint.get("rollout_id") or ""),
            reward=float(best_checkpoint.get("reward") or 0.0) if best_checkpoint is not None else None,
            restore_semantics=str(
                best_checkpoint.get("restore_semantics")
                or best_checkpoint.get("checkpoint_semantics")
                or checkpoint_resume.restore_semantics
                or ""
            ).strip(),
            resume_eligible=bool(best_checkpoint.get("resume_eligible") or best_checkpoint.get("restore_eligible")),
            source="checkpoint",
            metadata={
                "task_instance_id": str(best_checkpoint.get("task_instance_id") or ""),
                "seed": best_checkpoint.get("seed"),
            },
        )
        if best_checkpoint is not None
        else None
    )
    return RunRecoveryProjection(
        run_id=str(run_id),
        profile=str(profile),
        recovery_action=action,
        artifact_trust_level=trust,
        supports_true_resume=supports_true_resume,
        supports_request_snapshot_replay=supports_request_snapshot_replay,
        supports_fresh_restart=True,
        resumable=best_checkpoint is not None and supports_true_resume,
        replayable=supports_request_snapshot_replay,
        highest_quality_recovery_point=recovery_point,
        resume_blockers=resume_blockers,
        transform_failures=failures,
        operator_next_action=next_action,
        metadata={
            "checkpoint_count": len(checkpoint_rows),
            "eligible_checkpoint_count": len(eligible_checkpoints),
            "run_outcome": outcome,
            "run_phase": phase,
        },
    )
