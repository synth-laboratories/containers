"""Portable checkpoint metadata; opaque participant codecs remain benchmark-owned."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ..package_models import Digest, content_digest
from .contracts import ArtifactReference

Count = Annotated[int, Field(strict=True, ge=0, le=9_007_199_254_740_991)]
Amount = Annotated[str, StringConstraints(pattern=r"^(0|[1-9][0-9]{0,17})(\.[0-9]{1,12})?$")]
Name = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$", max_length=128)]
RestoreMode = Literal["inspect", "replay", "strict_continue", "same_policy_fork", "edited_policy_fork"]


class PortableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    @model_validator(mode="after")
    def non_nil_identity(self):
        if any(isinstance(value, UUID) and value.int == 0 for value in self.__dict__.values()):
            raise ValueError("nil_checkpoint_identity")
        return self


class CheckpointExecutionV1(PortableModel):
    """Configuration covers cadence/limits; policies covers every actor's policy settings."""

    source: Digest
    image: Digest
    dependencies: Digest
    task: Digest
    world: Digest
    scorer: Digest
    protocol: Digest
    configuration: ArtifactReference
    policies: ArtifactReference


class CheckpointCursorV1(PortableModel):
    name: Name
    position: Count
    chain_head: Digest


class CheckpointComponentV1(PortableModel):
    participant: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    codec: Digest
    coverage: Annotated[tuple[Name, ...], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def unique_coverage(self):
        if len(set(self.coverage)) != len(self.coverage):
            raise ValueError("duplicate_checkpoint_coverage")
        return self


class CheckpointLiabilityV1(PortableModel):
    """Exact decimal strings; null is unknown, including outstanding reservations."""

    resource: Name
    provider: Name
    unit: Name
    consumed: Amount | None
    outstanding: Amount | None


class PortableCheckpointManifestV1(PortableModel):
    schema_version: Literal["synth.portable-checkpoint.v1"] = "synth.portable-checkpoint.v1"
    application: ArtifactReference
    checkpoint: UUID
    run_id: Name
    lineage_id: UUID
    sequence: Count
    chain_head: Digest
    execution: CheckpointExecutionV1
    components: Annotated[tuple[CheckpointComponentV1, ...], Field(min_length=1, max_length=16)]
    cursors: Annotated[tuple[CheckpointCursorV1, ...], Field(min_length=1, max_length=64)]
    liabilities: Annotated[tuple[CheckpointLiabilityV1, ...], Field(max_length=256)]
    clock_policy: Literal["active_runtime", "absolute_deadline"]
    terminal: bool = Field(strict=True)
    unresolved_effects: Count = 0
    modes: Annotated[tuple[RestoreMode, ...], Field(min_length=1, max_length=5)]

    @model_validator(mode="after")
    def unique_entries(self):
        groups = (
            [item.participant for item in self.components],
            [item.name for item in self.cursors],
            [(item.provider, item.resource, item.unit) for item in self.liabilities],
            list(self.modes),
        )
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("duplicate_checkpoint_entry")
        event = next((item for item in self.cursors if item.name == "event"), None)
        if event is None or event.position != self.sequence or event.chain_head != self.chain_head:
            raise ValueError("checkpoint_event_cursor_inconsistent")
        return self

    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class CheckpointCommitAnchorV1(PortableModel):
    """An owning journal's supplied commit, never authority obtained from bundle contents."""

    manifest: Digest
    sequence: Count
    chain_head: Digest


class CheckpointRestoreRequestV1(PortableModel):
    schema_version: Literal["synth.checkpoint-restore-request.v1"] = "synth.checkpoint-restore-request.v1"
    mode: RestoreMode
    source_run_id: Name
    source_lineage_id: UUID
    target_run_id: Name
    target_lineage_id: UUID
    execution: CheckpointExecutionV1
    commit: CheckpointCommitAnchorV1
    child_budget: ArtifactReference | None = None

    @model_validator(mode="after")
    def explicit_child(self):
        fork = self.mode in ("same_policy_fork", "edited_policy_fork")
        if fork:
            if (
                self.target_run_id == self.source_run_id
                or self.target_lineage_id == self.source_lineage_id
                or self.child_budget is None
            ):
                raise ValueError("checkpoint_fork_requires_child_identity_and_budget")
        elif (
            self.source_run_id != self.target_run_id
            or self.source_lineage_id != self.target_lineage_id
            or self.child_budget is not None
        ):
            raise ValueError("checkpoint_continuation_identity_changed")
        return self


class CheckpointRuntimeSupportV1(PortableModel):
    schema_version: Literal["synth.checkpoint-runtime-support.v1"] = "synth.checkpoint-runtime-support.v1"
    modes: Annotated[tuple[RestoreMode, ...], Field(min_length=1, max_length=5)]
    components: Annotated[tuple[CheckpointComponentV1, ...], Field(min_length=1, max_length=16)]
    required_cursors: Annotated[tuple[Name, ...], Field(min_length=1, max_length=64)]
    trusted_decoders: Annotated[tuple[Digest, ...], Field(max_length=64)]
    clock_policy: Literal["active_runtime", "absolute_deadline"]

    @model_validator(mode="after")
    def unique_entries(self):
        groups = (
            list(self.modes), [row.participant for row in self.components],
            list(self.required_cursors), list(self.trusted_decoders),
        )
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("duplicate_checkpoint_runtime_entry")
        return self


class CheckpointRestoreReportV1(PortableModel):
    schema_version: Literal["synth.checkpoint-restore-report.v1"] = "synth.checkpoint-restore-report.v1"
    manifest: Digest
    request: Digest
    runtime_support: Digest
    selection_scope: Digest
    mode: RestoreMode
    checkpoint: UUID
    sequence: Count
    chain_head: Digest
    declaration_supported: bool = Field(strict=True)
    blockers: tuple[str, ...]
    verified_blobs: Count
    verified_bytes: Count
    retained_liabilities: tuple[CheckpointLiabilityV1, ...]
    # Compatibility is deliberately separate from current hosted authority/effects.
    admission: Literal["not_checked"] = "not_checked"
    suffix_reconciliation: Literal["required"] = "required"
    liability_reconciliation: Literal["required"] = "required"
    decoder_execution: Literal["not_run"] = "not_run"
    target_inputs: Literal["not_checked"] = "not_checked"

    @model_validator(mode="after")
    def consistent_result(self):
        if self.declaration_supported != (not self.blockers):
            raise ValueError("inconsistent_checkpoint_restore_report")
        return self


class CheckpointInspectionV1(PortableModel):
    schema_version: Literal["synth.checkpoint-inspection.v1"] = "synth.checkpoint-inspection.v1"
    manifest: PortableCheckpointManifestV1
    manifest_digest: Digest
    verified_blobs: Count
    verified_bytes: Count
    publication: Literal["not_checked"] = "not_checked"
    decoder_execution: Literal["not_run"] = "not_run"
