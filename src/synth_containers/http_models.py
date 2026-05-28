from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

HttpObject = dict[str, object]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class CheckpointRefModel(StrictModel):
    checkpoint_id: str | None = None
    checkpoint_uri: str | None = None
    checkpoint_version: str | None = None


class RolloutActorSpecModel(StrictModel):
    actor_id: str
    role: str | None = None
    config: HttpObject = Field(default_factory=dict)


class RolloutRequestModel(StrictModel):
    rollout_id: str | None = None
    trace_correlation_id: str | None = None
    trial_id: str | None = None
    submission_mode: str | None = None
    run_id: str | None = None
    mode: str | None = None
    task_id: str | None = None
    task_instance_id: str | None = None
    task_metadata: HttpObject = Field(default_factory=dict)
    env: HttpObject = Field(default_factory=dict)
    policy: HttpObject = Field(default_factory=dict)
    candidate: HttpObject = Field(default_factory=dict)
    candidate_overlay: HttpObject = Field(default_factory=dict)
    dataset_row: HttpObject = Field(default_factory=dict)
    metadata: HttpObject = Field(default_factory=dict)
    checkpoint: HttpObject | str | None = None
    checkpoint_id: str | None = None
    checkpoint_data_base64: str | None = None
    target_rollout_id: str | None = None
    long_horizon: HttpObject = Field(default_factory=dict)
    terminator: HttpObject = Field(default_factory=dict)
    task_payload: HttpObject = Field(default_factory=dict)
    dataset: HttpObject = Field(default_factory=dict)
    actors: list[RolloutActorSpecModel] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    actor_overrides: HttpObject = Field(default_factory=dict)


class PauseRequestModel(StrictModel):
    reason: str | None = None
    metadata: HttpObject = Field(default_factory=dict)


class TerminateRequestModel(StrictModel):
    reason: str | None = None


class CreateCheckpointRequestModel(StrictModel):
    checkpoint_id: str | None = None
    checkpoint_uri: str | None = None
    checkpoint_version: str | None = None
    label: str | None = None
    labels: list[str] = Field(default_factory=list)
    source: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    metadata: HttpObject = Field(default_factory=dict)
    annotations: HttpObject = Field(default_factory=dict)
    artifact_refs: list[HttpObject] = Field(default_factory=list)
    restore_eligible: bool | None = None


class ResumeRequestModel(StrictModel):
    checkpoint_id: str | None = None
    target_rollout_id: str | None = None
    mode: str | None = None
    submission_mode: str | None = None
    overrides: HttpObject = Field(default_factory=dict)
    branch_metadata: HttpObject = Field(default_factory=dict)
    checkpoint_data_base64: str | None = None


class CheckpointLabelsRequestModel(StrictModel):
    labels: list[str] = Field(default_factory=list)
    annotations: HttpObject = Field(default_factory=dict)
    metadata: HttpObject = Field(default_factory=dict)
