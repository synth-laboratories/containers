from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .capabilities import RuntimeCapabilitySurface, RuntimeMetadata
from .ontology import CapabilityLevel, ExecutionProfile, PrimitiveProtocol, RolloutMode, StatefulnessTier
from .tool_runtime import ToolRuntimeKind


class ConsumerTarget(StrEnum):
    GO_EX = "go_ex"
    MIPROV2 = "miprov2"
    STANDARD_EVALS = "standard_evals"
    PIPELINE_RL = "pipeline_rl"
    PIPELINE_RL_TOKEN_IDS = "pipeline_rl_token_ids"
    PIPELINE_RL_LOGPROBS = "pipeline_rl_logprobs"
    PIPELINE_RL_LOGITS = "pipeline_rl_logits"
    HARBOR_PROXY = "harbor_proxy"
    OPENENV_PROXY = "openenv_proxy"
    ARCHIPELAGO_PROXY = "archipelago_proxy"


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    code: str
    message: str
    fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fields": list(self.fields),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    target: ConsumerTarget
    supported: bool
    missing_profiles: tuple[str, ...] = ()
    missing_protocols: tuple[str, ...] = ()
    missing_protocol_groups: tuple[str, ...] = ()
    missing_features: tuple[str, ...] = ()
    issues: tuple[CompatibilityIssue, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.value,
            "supported": self.supported,
            "missing_profiles": list(self.missing_profiles),
            "missing_protocols": list(self.missing_protocols),
            "missing_protocol_groups": list(self.missing_protocol_groups),
            "missing_features": list(self.missing_features),
            "issues": [issue.to_dict() for issue in self.issues],
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ConsumerRequirement:
    target: ConsumerTarget
    summary: str
    required_profiles: tuple[ExecutionProfile, ...] = ()
    required_protocols: tuple[PrimitiveProtocol, ...] = ()
    required_any_protocol_groups: tuple[tuple[PrimitiveProtocol, ...], ...] = ()
    required_rollout_modes: tuple[RolloutMode, ...] = ()
    minimum_statefulness: StatefulnessTier | None = None
    minimum_level: CapabilityLevel = CapabilityLevel.DERIVED
    requires_checkpoint_support: bool = False
    requires_resume_support: bool = False
    requires_branching: bool = False
    requires_state_support: bool = False
    requires_trace: bool = False
    requires_reward: bool = False
    requires_verifier: bool = False
    requires_tool_runtime: bool = False
    requires_token_ids: bool = False
    requires_logprobs: bool = False
    requires_logits: bool = False
    requires_proxied_inference: bool = False


_STATEFULNESS_RANK: dict[StatefulnessTier, int] = {
    StatefulnessTier.STATELESS: 0,
    StatefulnessTier.EPISODIC: 1,
    StatefulnessTier.STATEFUL: 2,
    StatefulnessTier.LONG_HORIZON: 3,
}


CONSUMER_REQUIREMENTS: dict[ConsumerTarget, ConsumerRequirement] = {
    ConsumerTarget.GO_EX: ConsumerRequirement(
        target=ConsumerTarget.GO_EX,
        summary="Checkpointable long-horizon search with branchable recovery and trace visibility.",
        required_profiles=(ExecutionProfile.CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.CHECKPOINTABLE,
            PrimitiveProtocol.RESTORABLE,
            PrimitiveProtocol.TRACE_EMITTING,
        ),
        required_any_protocol_groups=(
            (
                PrimitiveProtocol.STEPPABLE,
                PrimitiveProtocol.ROLLOUT_RUNNABLE,
                PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            ),
            (
                PrimitiveProtocol.REWARD_EMITTING,
                PrimitiveProtocol.VERIFIER_BACKED,
            ),
        ),
        required_rollout_modes=(RolloutMode.ASYNC,),
        minimum_statefulness=StatefulnessTier.LONG_HORIZON,
        requires_checkpoint_support=True,
        requires_resume_support=True,
        requires_branching=True,
        requires_state_support=True,
        requires_trace=True,
    ),
    ConsumerTarget.MIPROV2: ConsumerRequirement(
        target=ConsumerTarget.MIPROV2,
        summary="Prompt/demo optimizer surface with rollout evaluation and scalar feedback.",
        required_any_protocol_groups=(
            (
                PrimitiveProtocol.ROLLOUT_RUNNABLE,
                PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            ),
            (
                PrimitiveProtocol.REWARD_EMITTING,
                PrimitiveProtocol.VERIFIER_BACKED,
            ),
        ),
        minimum_statefulness=StatefulnessTier.STATELESS,
    ),
    ConsumerTarget.STANDARD_EVALS: ConsumerRequirement(
        target=ConsumerTarget.STANDARD_EVALS,
        summary="Eval harness shape with task catalog and verifier/summary support.",
        required_profiles=(ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.CATALOG_BACKED,
            PrimitiveProtocol.VERIFIER_BACKED,
        ),
        minimum_statefulness=StatefulnessTier.STATELESS,
        requires_verifier=True,
    ),
    ConsumerTarget.PIPELINE_RL: ConsumerRequirement(
        target=ConsumerTarget.PIPELINE_RL,
        summary="Pipeline RL trajectory emitter without token-level traces.",
        required_profiles=(ExecutionProfile.RL_TRAJECTORY_EMITTER,),
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.TRACE_EMITTING,
        ),
        minimum_statefulness=StatefulnessTier.EPISODIC,
        requires_trace=True,
        requires_reward=True,
    ),
    ConsumerTarget.PIPELINE_RL_TOKEN_IDS: ConsumerRequirement(
        target=ConsumerTarget.PIPELINE_RL_TOKEN_IDS,
        summary="Pipeline RL with token IDs.",
        required_profiles=(ExecutionProfile.TOKEN_LEVEL_RL_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.TRACE_EMITTING,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED,
        ),
        minimum_statefulness=StatefulnessTier.EPISODIC,
        requires_trace=True,
        requires_reward=True,
        requires_token_ids=True,
        requires_proxied_inference=True,
    ),
    ConsumerTarget.PIPELINE_RL_LOGPROBS: ConsumerRequirement(
        target=ConsumerTarget.PIPELINE_RL_LOGPROBS,
        summary="Pipeline RL with token IDs + logprobs.",
        required_profiles=(ExecutionProfile.TOKEN_LEVEL_RL_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.TRACE_EMITTING,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED,
        ),
        minimum_statefulness=StatefulnessTier.EPISODIC,
        requires_trace=True,
        requires_reward=True,
        requires_token_ids=True,
        requires_logprobs=True,
        requires_proxied_inference=True,
    ),
    ConsumerTarget.PIPELINE_RL_LOGITS: ConsumerRequirement(
        target=ConsumerTarget.PIPELINE_RL_LOGITS,
        summary="Pipeline RL with token IDs + logits.",
        required_profiles=(ExecutionProfile.TOKEN_LEVEL_RL_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.TRACE_EMITTING,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED,
        ),
        minimum_statefulness=StatefulnessTier.EPISODIC,
        requires_trace=True,
        requires_reward=True,
        requires_token_ids=True,
        requires_logits=True,
        requires_proxied_inference=True,
    ),
    ConsumerTarget.HARBOR_PROXY: ConsumerRequirement(
        target=ConsumerTarget.HARBOR_PROXY,
        summary="Harbor proxying through async, catalog-backed, verifier-backed control plane.",
        required_profiles=(ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.CATALOG_BACKED,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            PrimitiveProtocol.VERIFIER_BACKED,
        ),
        required_rollout_modes=(RolloutMode.ASYNC,),
        minimum_statefulness=StatefulnessTier.STATELESS,
        requires_verifier=True,
    ),
    ConsumerTarget.OPENENV_PROXY: ConsumerRequirement(
        target=ConsumerTarget.OPENENV_PROXY,
        summary="OpenEnv-style gym-compatible reset/step/observe/state flow.",
        required_profiles=(ExecutionProfile.GYM_STYLE_ENVIRONMENT,),
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
        ),
        minimum_statefulness=StatefulnessTier.EPISODIC,
        requires_state_support=True,
    ),
    ConsumerTarget.ARCHIPELAGO_PROXY: ConsumerRequirement(
        target=ConsumerTarget.ARCHIPELAGO_PROXY,
        summary="Archipelago-style sandbox world with tools, traces, and async execution.",
        required_profiles=(ExecutionProfile.SANDBOXED_MCP_WORLD,),
        required_protocols=(
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            PrimitiveProtocol.TOOL_CALLABLE,
            PrimitiveProtocol.TRACE_EMITTING,
        ),
        required_rollout_modes=(RolloutMode.ASYNC,),
        minimum_statefulness=StatefulnessTier.EPISODIC,
        requires_trace=True,
        requires_tool_runtime=True,
    ),
}


def consumer_requirement(target: ConsumerTarget | str) -> ConsumerRequirement:
    normalized = target if isinstance(target, ConsumerTarget) else ConsumerTarget(str(target))
    return CONSUMER_REQUIREMENTS[normalized]


def _capabilities_from(value: RuntimeMetadata | RuntimeCapabilitySurface) -> RuntimeCapabilitySurface:
    if isinstance(value, RuntimeMetadata):
        return value.capabilities
    return value


def _statefulness_at_least(actual: StatefulnessTier | str, minimum: StatefulnessTier) -> bool:
    actual_tier = actual if isinstance(actual, StatefulnessTier) else StatefulnessTier(str(actual))
    return _STATEFULNESS_RANK[actual_tier] >= _STATEFULNESS_RANK[minimum]


def evaluate_runtime_requirement(
    metadata: RuntimeMetadata | RuntimeCapabilitySurface,
    requirement: ConsumerRequirement,
) -> CompatibilityReport:
    capabilities = _capabilities_from(metadata)
    minimum_level = requirement.minimum_level

    missing_profiles = tuple(
        profile.value
        for profile in requirement.required_profiles
        if capabilities.profile_level(profile).rank < minimum_level.rank
    )
    missing_protocols = tuple(
        protocol.value
        for protocol in requirement.required_protocols
        if capabilities.protocol_level(protocol).rank < minimum_level.rank
    )
    missing_protocol_groups = tuple(
        "|".join(protocol.value for protocol in group)
        for group in requirement.required_any_protocol_groups
        if not any(
            capabilities.protocol_level(protocol).rank >= minimum_level.rank
            for protocol in group
        )
    )

    missing_features: list[str] = []
    if requirement.minimum_statefulness is not None and not _statefulness_at_least(
        capabilities.statefulness_tier, requirement.minimum_statefulness
    ):
        missing_features.append(f"minimum_statefulness:{requirement.minimum_statefulness.value}")
    for rollout_mode in requirement.required_rollout_modes:
        if rollout_mode not in capabilities.normalized_rollout_modes():
            missing_features.append(f"rollout_mode:{rollout_mode.value}")
    if requirement.requires_checkpoint_support and not capabilities.checkpoint_support:
        missing_features.append("checkpoint_support")
    if requirement.requires_resume_support and not capabilities.resume_support:
        missing_features.append("resume_support")
    if requirement.requires_branching and not capabilities.supports_branching:
        missing_features.append("supports_branching")
    if requirement.requires_state_support and not capabilities.state_support:
        missing_features.append("state_support")
    if requirement.requires_trace and not capabilities.trace_support:
        missing_features.append("trace_support")
    if requirement.requires_reward and not capabilities.reward_support:
        missing_features.append("reward_support")
    if requirement.requires_verifier and not capabilities.verifier_support:
        missing_features.append("verifier_support")
    if requirement.requires_tool_runtime and capabilities.tool_runtime.normalized_runtime_kind() is ToolRuntimeKind.NONE:
        missing_features.append("tool_runtime")
    if requirement.requires_token_ids and not capabilities.token_emission.token_ids:
        missing_features.append("token_ids")
    if requirement.requires_logprobs and not capabilities.token_emission.logprobs:
        missing_features.append("logprobs")
    if requirement.requires_logits and not capabilities.token_emission.logits:
        missing_features.append("logits")
    if requirement.requires_proxied_inference and not capabilities.proxied_inference:
        missing_features.append("proxied_inference")

    issues: list[CompatibilityIssue] = []
    for profile in missing_profiles:
        issues.append(
            CompatibilityIssue(
                code="missing_profile",
                message=f"missing profile: {profile}",
                fields=("profile_fidelity", "profiles"),
            )
        )
    for protocol in missing_protocols:
        issues.append(
            CompatibilityIssue(
                code="missing_protocol",
                message=f"missing protocol: {protocol}",
                fields=("protocol_fidelity",),
            )
        )
    for group in missing_protocol_groups:
        issues.append(
            CompatibilityIssue(
                code="missing_protocol_group",
                message=f"missing one of protocol group: {group}",
                fields=("protocol_fidelity",),
            )
        )
    for feature in missing_features:
        issues.append(
            CompatibilityIssue(
                code="missing_feature",
                message=f"missing feature: {feature}",
                fields=("capabilities",),
            )
        )

    supported = (
        not missing_profiles
        and not missing_protocols
        and not missing_protocol_groups
        and not missing_features
    )
    notes = tuple(sorted({str(note).strip() for note in capabilities.metadata.get("compatibility_notes", []) if str(note).strip()}))
    return CompatibilityReport(
        target=requirement.target,
        supported=supported,
        missing_profiles=missing_profiles,
        missing_protocols=missing_protocols,
        missing_protocol_groups=missing_protocol_groups,
        missing_features=tuple(missing_features),
        issues=tuple(issues),
        notes=notes,
    )


def evaluate_consumer_support(
    metadata: RuntimeMetadata | RuntimeCapabilitySurface,
    target: ConsumerTarget | str,
) -> CompatibilityReport:
    return evaluate_runtime_requirement(metadata, consumer_requirement(target))


def assert_consumer_support(metadata: RuntimeMetadata | RuntimeCapabilitySurface, target: ConsumerTarget | str) -> None:
    result = evaluate_consumer_support(metadata, target)
    if result.supported:
        return
    chunks: list[str] = []
    if result.missing_profiles:
        chunks.append(f"missing_profiles={','.join(result.missing_profiles)}")
    if result.missing_protocols:
        chunks.append(f"missing_protocols={','.join(result.missing_protocols)}")
    if result.missing_protocol_groups:
        chunks.append(f"missing_protocol_groups={','.join(result.missing_protocol_groups)}")
    if result.missing_features:
        chunks.append(f"missing_features={','.join(result.missing_features)}")
    raise ValueError(f"consumer {result.target.value} is unsupported ({'; '.join(chunks)})")


def compatibility_matrix(
    metadata: RuntimeMetadata | RuntimeCapabilitySurface,
    targets: tuple[ConsumerTarget, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    selected_targets = targets or tuple(ConsumerTarget)
    return {
        target.value: evaluate_consumer_support(metadata, target).to_dict()
        for target in selected_targets
    }
