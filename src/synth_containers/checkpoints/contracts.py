"""Application cuts are distinct from tunnel authority and live transport.

See docs/specs/tunnel-v2-connector.md and synth-tunnel/docs/SESSION_CHECKPOINTS.md.
"""

from __future__ import annotations
from enum import StrEnum
from hashlib import sha256
from uuid import UUID
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CheckpointFailure(RuntimeError):
    """Stable error code; no participant payload or credentials in messages."""


class CheckpointSemantics(StrEnum):
    APPLICATION_COLD_RESTORE = "application_cold_restore"
    TRUE_ENVIRONMENT_SNAPSHOT = "true_environment_snapshot"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def non_nil_identity(cls, value):
        if isinstance(value, UUID) and value.int == 0:
            raise ValueError("nil_checkpoint_identity")
        return value


class ArtifactReference(Contract):
    namespace: str = Field(min_length=1, max_length=128)
    object_key: str = Field(pattern=r"^sha256/[a-f0-9]{64}$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size_bytes: int = Field(ge=0, le=64 * 1024 * 1024)

    @model_validator(mode="after")
    def canonical_reference(self):
        if len(self.namespace.encode()) > 128 or self.object_key != "sha256/" + self.sha256:
            raise ValueError("invalid_checkpoint_artifact_reference")
        return self


class ParticipantIdentity(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    service_identity: str = Field(min_length=1, max_length=128)
    origin_instance: UUID
    semantics: CheckpointSemantics

    @field_validator("service_identity")
    @classmethod
    def canonical_service(cls, value: str) -> str:
        if value != value.strip() or len(value.encode()) > 128:
            raise ValueError("invalid_participant_service")
        return value


class SessionFence(Contract):
    connector: UUID
    session: UUID
    source_generation: int = Field(ge=1, le=9_007_199_254_740_991)
    generation: int = Field(ge=1, le=9_007_199_254_740_991)
    permit_barrier_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def advanced_generation(self):
        if self.generation <= self.source_generation:
            raise ValueError("checkpoint_generation_not_advanced")
        return self


class ParticipantCut(Contract):
    operation: UUID
    receipt: UUID
    participant: ParticipantIdentity
    pending_operations: int = Field(ge=0)
    unresolved_operations: int = Field(ge=0)


class ParticipantSnapshot(Contract):
    cut: ParticipantCut
    artifact: ArtifactReference


class ApplicationCheckpoint(Contract):
    schema_version: Literal["synth.application-checkpoint.v1"] = "synth.application-checkpoint.v1"
    checkpoint: UUID
    operation: UUID
    fence: SessionFence
    semantics: CheckpointSemantics
    participants: tuple[ParticipantSnapshot, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def consistent_joint_cut(self):
        if len({item.cut.participant.name for item in self.participants}) != len(self.participants):
            raise ValueError("duplicate_checkpoint_participant")
        for item in self.participants:
            if (
                item.cut.operation != self.operation
                or item.cut.pending_operations
                or item.cut.unresolved_operations
            ):
                raise ValueError("inconsistent_checkpoint_cut")
            if (
                self.semantics == CheckpointSemantics.TRUE_ENVIRONMENT_SNAPSHOT
                and item.cut.participant.semantics != self.semantics
            ):
                raise ValueError("checkpoint_semantics_unsupported")
        return self

    def canonical_bytes(self) -> bytes:
        import json

        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()

    def digest(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


class ParticipantRestore(Contract):
    operation: UUID
    checkpoint: UUID
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    participant_name: str
    service_identity: str
    source_origin_instance: UUID
    restored_origin_instance: UUID
    receipt: UUID

    @model_validator(mode="after")
    def fresh_origin(self):
        if self.restored_origin_instance == self.source_origin_instance:
            raise ValueError("checkpoint_restore_requires_fresh_origin")
        return self
