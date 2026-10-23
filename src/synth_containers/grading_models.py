"""Immutable grading requests; computation identity is separate from an attempt."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .package_models import Digest, Name, PackageModel, content_digest
from .tracing.models.selectors import TraceSelectorV1

Count = Annotated[int, Field(strict=True, ge=0)]
Positive = Annotated[int, Field(strict=True, gt=0)]


class GradingInputV1(PackageModel):
    name: Name
    digest: Digest
    media_type: str
    size_bytes: Count
    visibility: Literal["public", "private"]
    required: bool = True
    retention_root: str


class GraderLimitsV1(PackageModel):
    wall_seconds: Annotated[float, Field(gt=0, le=3600, allow_inf_nan=False)] = 60
    cpu_seconds: Positive = 60
    memory_bytes: Annotated[int, Field(strict=True, ge=64 * 1024 * 1024, le=8 * 1024**3)] = 512 * 1024**2
    output_bytes: Annotated[int, Field(strict=True, ge=1024, le=64 * 1024**2)] = 1024 * 1024
    input_bytes: Annotated[int, Field(strict=True, ge=1, le=256 * 1024**2)] = 64 * 1024**2
    network: Literal["none"] = "none"
    cost_ceiling_usd: Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")] = "0"


class GradingPlanV1(PackageModel):
    schema_version: Literal["synth.grading-plan.v1"] = "synth.grading-plan.v1"
    operation_id: Name
    trace_id: str
    trace_digest: Digest
    submission_digest: Digest | None = None
    task_digest: Digest
    world_digest: Digest
    dataset_digest: Digest
    split_digest: Digest
    scorer: Literal["json-exact-v1", "craftax-trajectory-metrics-v1", "cardcodebench-captured-v1", "cardcodebench-captured-v2"]
    scorer_program_digest: Digest
    scorer_image_digest: Digest | None = None
    dependencies_digest: Digest
    rubric_digest: Digest
    aggregation: Literal["rubric-v5-and-coverage-v1"] = "rubric-v5-and-coverage-v1"
    seed: Count
    judge_configuration_digest: Digest
    execution_profile: Literal["trusted-offline-rubric-v1", "admitted-external-grader-v1"]
    permissions_digest: Digest
    limits: GraderLimitsV1
    case_ids: Annotated[tuple[Name, ...], Field(min_length=1)]
    inputs: tuple[GradingInputV1, ...]
    required_selectors: tuple[TraceSelectorV1, ...] = ()
    required_facts: tuple[str, ...] = ()
    supersedes_result_id: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> GradingPlanV1:
        if len(set(self.case_ids)) != len(self.case_ids) or len({i.name for i in self.inputs}) != len(self.inputs):
            raise ValueError("duplicate grading case/input")
        if sum(item.size_bytes for item in self.inputs) > self.limits.input_bytes:
            raise ValueError("grading input allowance exceeded")
        if any(item.trace_id != self.trace_id or item.trace_digest != self.trace_digest for item in self.required_selectors):
            raise ValueError("grading selector has a foreign subject")
        if self.scorer.startswith("cardcodebench") and self.submission_digest is None:
            raise ValueError("CardCodeBench grading requires a frozen submission identity")
        return self

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class GradingPreflightV1(PackageModel):
    schema_version: Literal["synth.grading-preflight.v1"] = "synth.grading-preflight.v1"
    plan_digest: Digest
    sufficient: bool
    blockers: tuple[str, ...]
    verified_inputs: tuple[Digest, ...]
    execution_profile: str
    cost_ceiling_usd: str
    compute_invoked: Literal[False] = False


class GradingResultV1(PackageModel):
    schema_version: Literal["synth.grading-result.v1"] = "synth.grading-result.v1"
    operation_id: Name
    plan_digest: Digest
    attempt_id: str
    result_id: str
    evidence_digest: Digest
    execution_status: Literal["completed", "failed", "timed_out", "output_limit", "memory_limit", "interrupted"]
    planned_cases: Count
    completed_cases: Count
    scored_cases: Count
    mean_score: float | None
    case_scores: dict[str, float | None]
    elapsed_ns: Count | None
    stdout_digest: Digest
    stderr_digest: Digest | None
    usage: Literal["no-provider-calls"] = "no-provider-calls"
    cleanup: Literal["owned-worker-stopped", "worker-exit-not-observed"] = "owned-worker-stopped"


class GradingAuthorityV1(PackageModel):
    name: Name
    result_ids: tuple[str, ...]

