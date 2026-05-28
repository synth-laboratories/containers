from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .nouns import TaskDefinition, TaskInstance
from .ontology import (
    CONTRACT_VERSION,
    CapabilityLevel,
    CheckpointSemantics,
    CoreNoun,
    ExecutionProfile,
    PrimitiveProtocol,
    ResumeSemantics,
    RolloutMode,
    RuntimeKind,
    StatefulnessTier,
)
from .profiles import infer_profiles
from .serde import JsonDataclassMixin
from .tool_runtime import ToolRuntimeCapabilities


@dataclass(slots=True)
class DatasetDescriptor(JsonDataclassMixin):
    dataset_id: str = ""
    split: str | None = None
    visible_splits: list[str] = field(default_factory=list)
    default_split: str | None = None
    row_count: int | None = None
    source: str = ""
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenEmissionCapabilities(JsonDataclassMixin):
    token_ids: bool = False
    tokens: bool = False
    logprobs: bool = False
    logits: bool = False
    top_logprobs: bool = False
    old_logprobs: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def any_supported(self) -> bool:
        return any((self.token_ids, self.tokens, self.logprobs, self.logits, self.top_logprobs, self.old_logprobs))


@dataclass(slots=True)
class RouteHints(JsonDataclassMixin):
    metadata_routes: list[str] = field(default_factory=lambda: ["/metadata", "/info"])
    task_info_routes: list[str] = field(default_factory=lambda: ["/task_info"])
    task_catalog_routes: list[str] = field(default_factory=lambda: ["/task_catalog"])
    rollout_routes: list[str] = field(default_factory=lambda: ["/rollout", "/rollouts"])
    state_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/state"])
    pause_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/pause"])
    terminate_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/terminate"])
    checkpoint_routes: list[str] = field(
        default_factory=lambda: [
            "/rollouts/{rollout_id}/checkpoints",
            "/rollouts/{rollout_id}/checkpoints/{checkpoint_id}",
            "/checkpoints",
            "/checkpoints/{checkpoint_id}",
            "/checkpoints/{checkpoint_id}/labels",
        ]
    )
    resume_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/resume"])
    summary_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/summary"])
    usage_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/usage"])
    event_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/events"])
    trace_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/trace"])
    artifact_routes: list[str] = field(default_factory=lambda: ["/rollouts/{rollout_id}/artifacts"])


@dataclass(slots=True)
class RuntimeCapabilitySurface(JsonDataclassMixin):
    contract_version: str = CONTRACT_VERSION
    runtime_kind: RuntimeKind | str = RuntimeKind.ENVIRONMENT
    profiles: list[ExecutionProfile | str] = field(default_factory=list)
    rollout_modes: list[RolloutMode | str] = field(default_factory=lambda: [RolloutMode.BLOCKING])
    statefulness_tier: StatefulnessTier | str = StatefulnessTier.EPISODIC
    noun_fidelity: dict[CoreNoun | str, CapabilityLevel | str] = field(default_factory=dict)
    protocol_fidelity: dict[PrimitiveProtocol | str, CapabilityLevel | str] = field(default_factory=dict)
    profile_fidelity: dict[ExecutionProfile | str, CapabilityLevel | str] = field(default_factory=dict)
    checkpoint_semantics: CheckpointSemantics | str = CheckpointSemantics.NONE
    restore_semantics: str = ""
    resume_semantics: ResumeSemantics | str = ResumeSemantics.UNSUPPORTED
    checkpoint_support: bool = False
    pause_support: bool = False
    resume_support: bool = False
    terminate_support: bool = False
    state_support: bool = False
    trace_support: bool = False
    reward_support: bool = False
    verifier_support: bool = False
    artifact_support: bool = False
    tool_runtime: ToolRuntimeCapabilities = field(default_factory=ToolRuntimeCapabilities)
    token_emission: TokenEmissionCapabilities = field(default_factory=TokenEmissionCapabilities)
    multi_actor: bool = False
    proxied_inference: bool = False
    supports_branching: bool = False
    true_environment_snapshot: bool = False
    route_hints: RouteHints = field(default_factory=RouteHints)
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_profiles(self) -> list[ExecutionProfile]:
        explicit = [ExecutionProfile(str(item)) for item in self.profiles]
        if explicit:
            return explicit
        inferred = infer_profiles(self.normalized_protocol_fidelity())
        return inferred

    def normalized_rollout_modes(self) -> list[RolloutMode]:
        return [RolloutMode(str(item)) for item in self.rollout_modes]

    def normalized_protocol_fidelity(self) -> dict[PrimitiveProtocol, CapabilityLevel]:
        return {
            PrimitiveProtocol(str(key)): CapabilityLevel.parse(value)
            for key, value in self.protocol_fidelity.items()
        }

    def normalized_noun_fidelity(self) -> dict[CoreNoun, CapabilityLevel]:
        return {
            CoreNoun(str(key)): CapabilityLevel.parse(value)
            for key, value in self.noun_fidelity.items()
        }

    def normalized_profile_fidelity(self) -> dict[ExecutionProfile, CapabilityLevel]:
        return {
            ExecutionProfile(str(key)): CapabilityLevel.parse(value)
            for key, value in self.profile_fidelity.items()
        }

    def protocol_level(self, protocol: PrimitiveProtocol | str) -> CapabilityLevel:
        return self.normalized_protocol_fidelity().get(PrimitiveProtocol(str(protocol)), CapabilityLevel.UNSUPPORTED)

    def profile_level(self, profile: ExecutionProfile | str) -> CapabilityLevel:
        explicit = self.normalized_profile_fidelity().get(ExecutionProfile(str(profile)))
        if explicit is not None:
            return explicit
        return CapabilityLevel.NATIVE if ExecutionProfile(str(profile)) in self.normalized_profiles() else CapabilityLevel.UNSUPPORTED

    def noun_level(self, noun: CoreNoun | str) -> CapabilityLevel:
        return self.normalized_noun_fidelity().get(CoreNoun(str(noun)), CapabilityLevel.UNSUPPORTED)

    def supports_protocol(
        self,
        protocol: PrimitiveProtocol | str,
        *,
        minimum_level: CapabilityLevel = CapabilityLevel.DERIVED,
    ) -> bool:
        return self.protocol_level(protocol).rank >= minimum_level.rank

    def validate(self) -> None:
        if self.true_environment_snapshot and not self.checkpoint_support:
            raise ValueError("true_environment_snapshot requires checkpoint_support=True")
        if self.true_environment_snapshot and not self.resume_support:
            raise ValueError("true_environment_snapshot requires resume_support=True")
        if self.supports_branching and not self.resume_support:
            raise ValueError("supports_branching=True requires resume_support=True")
        if self.token_emission.any_supported() and not self.trace_support:
            raise ValueError("token emission requires trace_support=True")
        if self.multi_actor and not self.supports_protocol(PrimitiveProtocol.MULTI_ACTOR, minimum_level=CapabilityLevel.APPROXIMATE):
            raise ValueError("multi_actor=True requires protocol_fidelity for multi_actor")
        self.tool_runtime.validate()

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "contract_version": self.contract_version,
            "runtime_kind": str(self.runtime_kind),
            "profiles": [item.value for item in self.normalized_profiles()],
            "rollout_modes": [item.value for item in self.normalized_rollout_modes()],
            "statefulness_tier": str(self.statefulness_tier),
            "noun_fidelity": {key.value: value.value for key, value in self.normalized_noun_fidelity().items()},
            "protocol_fidelity": {key.value: value.value for key, value in self.normalized_protocol_fidelity().items()},
            "profile_fidelity": {key.value: value.value for key, value in self.normalized_profile_fidelity().items()},
            "checkpoint_semantics": str(self.checkpoint_semantics),
            "restore_semantics": self.restore_semantics,
            "resume_semantics": str(self.resume_semantics),
            "checkpoint_support": bool(self.checkpoint_support),
            "pause_support": bool(self.pause_support),
            "resume_support": bool(self.resume_support),
            "terminate_support": bool(self.terminate_support),
            "state_support": bool(self.state_support),
            "trace_support": bool(self.trace_support),
            "reward_support": bool(self.reward_support),
            "verifier_support": bool(self.verifier_support),
            "artifact_support": bool(self.artifact_support),
            "tool_runtime": self.tool_runtime.to_dict(),
            "token_emission": self.token_emission.to_dict(),
            "multi_actor": bool(self.multi_actor),
            "proxied_inference": bool(self.proxied_inference),
            "supports_branching": bool(self.supports_branching),
            "true_environment_snapshot": bool(self.true_environment_snapshot),
            "route_hints": self.route_hints.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RuntimeMetadata(JsonDataclassMixin):
    runtime_id: str
    name: str
    description: str = ""
    capabilities: RuntimeCapabilitySurface = field(default_factory=RuntimeCapabilitySurface)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskInfo(JsonDataclassMixin):
    task: TaskDefinition
    dataset: DatasetDescriptor = field(default_factory=DatasetDescriptor)
    capabilities: RuntimeCapabilitySurface = field(default_factory=RuntimeCapabilitySurface)
    limits: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    environment: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskCatalog(JsonDataclassMixin):
    catalog_id: str
    tasks: list[TaskDefinition] = field(default_factory=list)
    instances: list[TaskInstance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def task_ids(self) -> list[str]:
        return [task.task_id for task in self.tasks]

    def get_task(self, task_id: str) -> TaskDefinition | None:
        target = str(task_id)
        for task in self.tasks:
            if task.task_id == target:
                return task
        return None

    def get_instance(self, task_instance_id: str) -> TaskInstance | None:
        target = str(task_instance_id)
        for item in self.instances:
            if item.task_instance_id == target:
                return item
        return None

    def instances_for_split(self, split: str) -> list[TaskInstance]:
        target = str(split)
        return [item for item in self.instances if item.split == target]

    def query(
        self,
        *,
        split: str | None = None,
        tags: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[TaskInstance]:
        required_tags = {str(tag) for tag in (tags or []) if str(tag)}
        rows = list(self.instances)
        if split is not None:
            rows = [item for item in rows if item.split == split]
        if required_tags:
            rows = [item for item in rows if required_tags.issubset(set(item.tags))]
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        return rows
