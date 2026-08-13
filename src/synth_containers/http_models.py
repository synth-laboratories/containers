from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .event_log import validate_rollout_id

from .proxying import (
    CredentialMode,
    InferenceApiFamily,
    InferenceTarget,
    PolicyDisableReasoning,
    ProxyMode,
    ToolCallStyle,
)

HttpObject = dict[str, object]
_RAW_CREDENTIAL_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_key",
}


def _find_raw_credential_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for raw_key, raw_value in value.items():
            normalized_key = str(raw_key).strip().lower().replace("-", "_")
            if normalized_key in _RAW_CREDENTIAL_KEYS or normalized_key.endswith("_api_key"):
                return str(raw_key)
            nested = _find_raw_credential_key(raw_value)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _find_raw_credential_key(item)
            if nested is not None:
                return nested
    return None


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


class RolloutPolicySpecModel(StrictModel):
    provider: str
    model: str
    api_family: InferenceApiFamily = InferenceApiFamily.CHAT_COMPLETIONS
    base_url: str | None = None
    inference_url: str | None = None
    max_tokens: int | None = None
    disable_reasoning: PolicyDisableReasoning = PolicyDisableReasoning.AUTO
    tool_call_style: ToolCallStyle = ToolCallStyle.NONE
    proxy_mode: ProxyMode = ProxyMode.ALLOW_DIRECT
    credential_mode: CredentialMode = CredentialMode.BYOK
    config: HttpObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_raw_credentials(self) -> "RolloutPolicySpecModel":
        raw_key = _find_raw_credential_key(self.config)
        if raw_key is not None:
            raise ValueError(
                f"policy.config must not carry raw credential field {raw_key!r}; "
                "set credential_mode and resolve credentials inside the container or proxy"
            )
        return self

    def to_inference_target(self) -> InferenceTarget:
        return InferenceTarget.from_policy_spec(self)


class RolloutTelemetryFrameModel(StrictModel):
    enabled: bool = False
    format: str = "png"
    every_n_steps: int = Field(default=1, ge=1, le=10_000)


class RolloutTelemetryModel(StrictModel):
    enabled: bool = False
    transport: str = "sse"
    detail: str = "standard"
    poll_interval_ms: int = Field(default=500, ge=100, le=30_000)
    events: list[str] = Field(default_factory=list)
    frame: RolloutTelemetryFrameModel = Field(default_factory=RolloutTelemetryFrameModel)
    retention: str = "run"

    @model_validator(mode="after")
    def validate_transport(self) -> "RolloutTelemetryModel":
        allowed = {"poll", "sse", "websocket"}
        if self.enabled:
            if self.transport == "auto":
                raise ValueError(
                    "telemetry.transport=auto is refused on authoritative / visual-attached runs"
                )
            if self.transport not in allowed:
                raise ValueError("telemetry.transport must be poll, sse, or websocket")
        elif self.transport not in allowed | {"auto"}:
            raise ValueError("telemetry.transport must be poll, sse, websocket, or auto")
        return self


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
    policy: RolloutPolicySpecModel | None = None
    candidate: HttpObject = Field(default_factory=dict)
    candidate_overlay: HttpObject = Field(default_factory=dict)
    task: HttpObject = Field(default_factory=dict)
    metadata: HttpObject = Field(default_factory=dict)
    checkpoint: HttpObject | str | None = None
    checkpoint_id: str | None = None
    checkpoint_data_base64: str | None = None
    target_rollout_id: str | None = None
    long_horizon: HttpObject = Field(default_factory=dict)
    terminator: HttpObject = Field(default_factory=dict)
    task_payload: HttpObject = Field(default_factory=dict)
    taskset: HttpObject = Field(default_factory=dict)
    actors: list[RolloutActorSpecModel] = Field(default_factory=list)
    actor_ids: list[str] = Field(default_factory=list)
    actor_overrides: HttpObject = Field(default_factory=dict)
    telemetry: RolloutTelemetryModel | None = None

    @field_validator("rollout_id")
    @classmethod
    def validate_public_rollout_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_rollout_id(value)


class RewardRequestModel(StrictModel):
    rollout_id: str | None = None
    evidence: HttpObject | None = None
    evaluation_plan_ref: str | None = None
    mode: str = "terminal"
    node_ids: list[str] = Field(default_factory=list)
    idempotency_key: str | None = None
    rescore: bool = False

    @model_validator(mode="after")
    def xor_rollout_or_evidence(self) -> "RewardRequestModel":
        has_rollout = bool(self.rollout_id)
        has_evidence = self.evidence is not None
        if has_rollout == has_evidence:
            raise ValueError("POST /reward requires exactly one of rollout_id or evidence")
        if self.mode not in {"terminal", "provisional"}:
            raise ValueError("mode must be terminal or provisional")
        return self


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
