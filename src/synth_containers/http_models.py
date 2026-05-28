from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class CheckpointRefModel(FlexibleModel):
    checkpoint_id: str | None = None
    checkpoint_uri: str | None = None
    checkpoint_version: str | None = None


class RolloutRequestModel(FlexibleModel):
    trace_correlation_id: str | None = None
    trial_id: str | None = None
    submission_mode: str | None = None
    task_instance_id: str | None = None
    env: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    checkpoint: dict[str, Any] | str | None = None
    long_horizon: dict[str, Any] = Field(default_factory=dict)
    terminator: dict[str, Any] = Field(default_factory=dict)
    task_payload: dict[str, Any] = Field(default_factory=dict)
    dataset: dict[str, Any] = Field(default_factory=dict)


class PauseRequestModel(FlexibleModel):
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TerminateRequestModel(FlexibleModel):
    reason: str | None = None


class CreateCheckpointRequestModel(FlexibleModel):
    checkpoint_id: str | None = None
    checkpoint_uri: str | None = None
    checkpoint_version: str | None = None
    label: str | None = None
    labels: list[str] = Field(default_factory=list)
    source: str | None = None
    actor_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    restore_eligible: bool | None = None


class ResumeRequestModel(FlexibleModel):
    checkpoint_id: str | None = None
    target_rollout_id: str | None = None
    mode: str | None = None
    submission_mode: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    branch_metadata: dict[str, Any] = Field(default_factory=dict)
    checkpoint_data_base64: str | None = None


class CheckpointLabelsRequestModel(FlexibleModel):
    labels: list[str] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
