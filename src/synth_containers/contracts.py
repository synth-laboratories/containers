from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capabilities import RuntimeCapabilitySurface, RuntimeMetadata, TaskInfo
from .ontology import RuntimeFamily
from .proxying import InferenceTarget, TraceIdentity
from .serde import JsonDataclassMixin, jsonable
from .tool_runtime import ToolRuntimeCapabilities


@dataclass(frozen=True, slots=True)
class ArtifactPaths(JsonDataclassMixin):
    contract_dump: str = "artifacts/contract_dump.json"
    run_recovery_projection: str = "artifacts/run_recovery_projection.json"
    best_candidate: str = "artifacts/best_candidate.json"
    heldout_eval: str = "artifacts/heldout_eval.json"
    run_summary: str = "artifacts/go_ex_run_summary.json"
    result_manifest: str = "artifacts/result_manifest.json"
    reportbench_output: str = "artifacts/reportbench_output.json"


@dataclass(frozen=True, slots=True)
class ArtifactContract(JsonDataclassMixin):
    run_id: str
    result_filename: str = "go_explore_mvp_result.json"
    paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LeverBundle(JsonDataclassMixin):
    bundle_hash: str
    values: dict[str, Any]
    manifest: dict[str, Any] = field(default_factory=dict)
    prompt_text: str = ""
    mutated_lever_ids: list[str] = field(default_factory=list)
    parent_candidate_id: str | None = None
    merged_parent_candidate_ids: list[str] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TaskContract(JsonDataclassMixin):
    task_id: str
    task_name: str
    task_family: str
    container_profile: str
    version: str = "v1"
    dataset: dict[str, Any] = field(default_factory=dict)
    seed_semantics: dict[str, Any] = field(default_factory=dict)
    evaluator_bundle: dict[str, Any] = field(default_factory=dict)
    capability_declaration: dict[str, Any] = field(default_factory=dict)
    route_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task_info(cls, task_info: TaskInfo) -> "TaskContract":
        profiles = task_info.capabilities.normalized_profiles()
        profile = profiles[0].value if profiles else "unspecified"
        task = task_info.task
        return cls(
            task_id=task.task_id,
            task_name=task.task_name,
            task_family=task.task_family,
            container_profile=profile,
            version=task.version,
            dataset=task_info.dataset.to_dict(),
            capability_declaration=task_info.capabilities.to_dict(),
            route_hints=task_info.capabilities.route_hints.to_dict(),
            metadata={**dict(task.metadata), **dict(task_info.task_metadata), **dict(task_info.metadata)},
        )


@dataclass(frozen=True, slots=True)
class CheckpointResumeContract(JsonDataclassMixin):
    checkpoint_semantics: str = ""
    restore_semantics: str = ""
    resume_semantics: str = ""
    checkpoint_support: bool = False
    pause_support: bool = False
    resume_support: bool = False
    terminate_support: bool = False
    state_support: bool = False
    trace_support: bool = False
    fork_support: bool = False
    supports_branching: bool = False
    true_environment_snapshot: bool = False
    control_boundary: str = ""
    resume_mode: str = ""
    fork_mode: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_capabilities(cls, capabilities: RuntimeCapabilitySurface) -> "CheckpointResumeContract":
        return cls(
            checkpoint_semantics=str(capabilities.checkpoint_semantics),
            restore_semantics=str(capabilities.restore_semantics or capabilities.checkpoint_semantics),
            resume_semantics=str(capabilities.resume_semantics),
            checkpoint_support=capabilities.checkpoint_support,
            pause_support=capabilities.pause_support,
            resume_support=capabilities.resume_support,
            terminate_support=capabilities.terminate_support,
            state_support=capabilities.state_support,
            trace_support=capabilities.trace_support,
            fork_support=capabilities.supports_branching,
            supports_branching=capabilities.supports_branching,
            true_environment_snapshot=capabilities.true_environment_snapshot,
            control_boundary="runtime",
            resume_mode=str(capabilities.resume_semantics),
            fork_mode="checkpoint" if capabilities.supports_branching else "unsupported",
            metadata=capabilities.metadata,
        )


@dataclass(frozen=True, slots=True)
class TransformPreferencePolicy(JsonDataclassMixin):
    preferred_candidate_transform: str = ""
    preferred_candidate_transform_group: str = ""
    candidate_transform_defaults: dict[str, Any] = field(default_factory=dict)
    candidate_transform_presets: dict[str, Any] = field(default_factory=dict)
    candidate_transform_group_defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunPolicyContract(JsonDataclassMixin):
    lane_execution_mode: str = "disabled"
    role_lane_execution_modes: dict[str, str] = field(default_factory=dict)
    required_real_roles: dict[str, bool] = field(default_factory=dict)
    submission_mode: str = "sync"
    live_control_required: bool = False
    verifier_source_policy: str = "container"
    strict_performance_mode: bool = False
    allow_smoke_components: bool = False
    allow_request_snapshot_resume: bool = True
    accepted_live_resume_semantics: list[str] = field(default_factory=list)
    holdout_top_k: int = 1
    holdout_seed_count: int | None = None
    min_non_baseline_candidate_fresh_rollouts: int = 0
    reward_source: str = ""
    terminator_control_mode: str = "boundary"
    transform_preference: TransformPreferencePolicy = field(default_factory=TransformPreferencePolicy)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mode_for_role(self, role: str) -> str:
        return str(self.role_lane_execution_modes.get(role) or self.lane_execution_mode or "disabled").strip() or "disabled"

    def role_required(self, role: str) -> bool:
        return bool(self.required_real_roles.get(role, False))


@dataclass(frozen=True, slots=True)
class ContainerExecutionContract(JsonDataclassMixin):
    runtime_family: RuntimeFamily | str
    task: TaskContract
    checkpoint_resume: CheckpointResumeContract
    tool_runtime: ToolRuntimeCapabilities
    reward_source: str = ""
    verifier_contract: dict[str, Any] = field(default_factory=dict)
    route_hints: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_runtime_family(self) -> RuntimeFamily:
        if isinstance(self.runtime_family, RuntimeFamily):
            return self.runtime_family
        text = str(self.runtime_family or "").strip().lower().replace("-", "_")
        if text in {"codex_session", "codex"}:
            return RuntimeFamily.CODEX_SESSION
        if text == "mcp_world":
            return RuntimeFamily.MCP_WORLD
        return RuntimeFamily.REQUEST_RESPONSE

    @classmethod
    def from_runtime_metadata(
        cls,
        *,
        metadata: RuntimeMetadata,
        task_info: TaskInfo,
        reward_source: str = "",
        verifier_contract: dict[str, Any] | None = None,
    ) -> "ContainerExecutionContract":
        capabilities = metadata.capabilities
        profiles = capabilities.normalized_profiles()
        runtime_family = RuntimeFamily.MCP_WORLD if any(item.value == "sandboxed_mcp_world" for item in profiles) else RuntimeFamily.REQUEST_RESPONSE
        return cls(
            runtime_family=runtime_family,
            task=TaskContract.from_task_info(task_info),
            checkpoint_resume=CheckpointResumeContract.from_capabilities(capabilities),
            tool_runtime=capabilities.tool_runtime,
            reward_source=reward_source,
            verifier_contract=dict(verifier_contract or {}),
            route_hints=capabilities.route_hints.to_dict(),
            capabilities=capabilities.to_dict(),
            metadata=dict(metadata.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_family": self.normalized_runtime_family().value,
            "task": self.task.to_dict(),
            "checkpoint_resume": self.checkpoint_resume.to_dict(),
            "tool_runtime": self.tool_runtime.to_dict(),
            "reward_source": self.reward_source,
            "verifier_contract": dict(self.verifier_contract),
            "route_hints": dict(self.route_hints),
            "capabilities": dict(self.capabilities),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class GoExContractDump(JsonDataclassMixin):
    run_id: str
    profile: str
    task: TaskContract
    checkpoint_resume: CheckpointResumeContract
    artifact: ArtifactContract
    baseline_lever_bundle: LeverBundle
    run_policy: RunPolicyContract = field(default_factory=RunPolicyContract)
    container_execution: dict[str, Any] = field(default_factory=dict)
    lane_inference_targets: dict[str, InferenceTarget] = field(default_factory=dict)
    lane_tool_runtime: dict[str, ToolRuntimeCapabilities] = field(default_factory=dict)
    container_tool_runtime: ToolRuntimeCapabilities = field(default_factory=ToolRuntimeCapabilities)
    trace_identity: TraceIdentity | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = jsonable(self)
        if not isinstance(payload, dict):
            raise TypeError("GoExContractDump.to_dict() expected a dict payload")
        return payload
