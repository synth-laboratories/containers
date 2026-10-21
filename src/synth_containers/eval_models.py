"""Portable evaluation declaration shared by CLI, local adapters and services."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .package_models import Digest, Name, PackageModel, content_digest


class ScriptedCandidateV1(PackageModel):
    kind: Literal["scripted-actions-v1"]
    actions: Annotated[tuple[str, ...], Field(min_length=1, max_length=10000)]
    repeat: bool = False

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class EvaluationLimitsV1(PackageModel):
    steps: Annotated[int, Field(strict=True, ge=1, le=10000)]
    calls: Annotated[int, Field(strict=True, ge=1, le=10000)]
    active_seconds: Annotated[float, Field(gt=0, le=3600, allow_inf_nan=False)]
    clock_policy: Literal["active_runtime", "absolute_deadline"] = "active_runtime"
    deadline_at: str | None = None
    provider_cost_ceiling_usd: Literal["0"] = "0"

    @model_validator(mode="after")
    def deadline(self):
        from datetime import datetime
        if self.clock_policy == "absolute_deadline":
            if self.deadline_at is None or datetime.fromisoformat(self.deadline_at.replace("Z", "+00:00")).utcoffset() is None:
                raise ValueError("absolute deadline requires an exact timezone-qualified timestamp")
        elif self.deadline_at is not None:
            raise ValueError("active-runtime limit cannot silently carry an absolute deadline")
        return self


class EvaluationPlanV1(PackageModel):
    schema_version: Literal["synth.evaluation-plan.v1"] = "synth.evaluation-plan.v1"
    benchmark: Literal["craftax", "retained-grading"]
    adapter: Literal["local-gold-scripted-v1", "retained-grading-v1", "rhodes-v1"]
    source_digest: Digest
    dependencies_digest: Digest
    scorer_digest: Digest
    image_digest: Digest | None = None
    task: dict
    seeds: Annotated[tuple[Annotated[int, Field(strict=True, ge=0)], ...], Field(min_length=1, max_length=1000)] = (0,)
    replicates: Annotated[int, Field(strict=True, ge=1, le=100)] = 1
    parallelism: Annotated[int, Field(strict=True, ge=1, le=64)] = 1
    checkpoint_every_steps: Annotated[int, Field(strict=True, ge=1, le=1000)] = 3
    limits: EvaluationLimitsV1
    cpu: Annotated[float, Field(gt=0, le=16, allow_inf_nan=False)] = 1
    memory_mb: Annotated[int, Field(strict=True, ge=128, le=8192)] = 512
    capture: Literal["native-state-actions-rewards-v1"] = "native-state-actions-rewards-v1"
    annotations: Literal["disabled"] = "disabled"
    credentials: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    grading_plan: str | None = None
    grading_bundle: str | None = None
    trusted_scorer_source_digests: tuple[Digest, ...] = ()

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.seeds)) != len(self.seeds) or any(type(seed) is not int or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be unique nonnegative integers")
        if self.credentials:
            raise ValueError("provider-free profiles consume no credentials; select an admitted provider adapter")
        if self.benchmark == "craftax" and (self.image_digest is None or self.adapter == "retained-grading-v1"):
            raise ValueError("Craftax requires pinned native image and gold adapter")
        if self.benchmark == "retained-grading" and (self.adapter != "retained-grading-v1" or not self.grading_plan or not self.grading_bundle or len(self.seeds) != 1 or self.replicates != 1):
            raise ValueError("retained grading requires one explicit grading plan/bundle operation")
        return self

    @property
    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


class CandidateBindingV1(PackageModel):
    label: Name
    content: ScriptedCandidateV1


class EvaluationRunDeclarationV1(PackageModel):
    schema_version: Literal["synth.evaluation-run.v1"] = "synth.evaluation-run.v1"
    run_id: Name
    lineage_id: str
    plan: EvaluationPlanV1
    candidates: tuple[CandidateBindingV1, ...]
    parent_run_id: Name | None = None
    parent_checkpoint_digest: Digest | None = None
    restore_mode: Literal["strict_continue", "same_policy_fork", "edited_policy_fork"] = "strict_continue"
    child_budget: dict | None = None

    @property
    def digest(self):
        return content_digest(self.model_dump(mode="json"))
