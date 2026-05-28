from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .ontology import CapabilityLevel, ExecutionProfile, PrimitiveProtocol


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    profile: ExecutionProfile
    required_protocols: tuple[PrimitiveProtocol, ...]
    description: str = ""


_PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        profile=ExecutionProfile.STATELESS_EVALUATOR,
        required_protocols=(
            PrimitiveProtocol.CATALOG_BACKED,
            PrimitiveProtocol.ROLLOUT_RUNNABLE,
            PrimitiveProtocol.VERIFIER_BACKED,
        ),
        description="Blocking benchmark/evaluator shape with catalog and verifier outputs.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.GYM_STYLE_ENVIRONMENT,
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
        ),
        description="Classic reset/step/state environment contract.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.CHECKPOINTABLE_STATEFUL_ENVIRONMENT,
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
            PrimitiveProtocol.CHECKPOINTABLE,
            PrimitiveProtocol.RESTORABLE,
        ),
        description="Stateful env with real checkpoint and restore semantics.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT,
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
            PrimitiveProtocol.CHECKPOINTABLE,
            PrimitiveProtocol.RESTORABLE,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            PrimitiveProtocol.TRACE_EMITTING,
        ),
        description="Long-horizon env with async control and checkpoint semantics.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.MULTI_AGENT_LONG_HORIZON_ENVIRONMENT,
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
            PrimitiveProtocol.CHECKPOINTABLE,
            PrimitiveProtocol.RESTORABLE,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            PrimitiveProtocol.TRACE_EMITTING,
            PrimitiveProtocol.MULTI_ACTOR,
        ),
        description="Checkpointable long-horizon environment with multiple actors.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.SANDBOXED_MCP_WORLD,
        required_protocols=(
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            PrimitiveProtocol.TOOL_CALLABLE,
            PrimitiveProtocol.TRACE_EMITTING,
        ),
        description="Sandbox/tool-runtime world with async execution and traces.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.RL_TRAJECTORY_EMITTER,
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.TRACE_EMITTING,
        ),
        description="Trajectory- and reward-emitting environment for RL collection.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.TOKEN_LEVEL_RL_ENVIRONMENT,
        required_protocols=(
            PrimitiveProtocol.RESETTABLE,
            PrimitiveProtocol.STEPPABLE,
            PrimitiveProtocol.OBSERVABLE,
            PrimitiveProtocol.STATE_READABLE,
            PrimitiveProtocol.REWARD_EMITTING,
            PrimitiveProtocol.TRACE_EMITTING,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED,
        ),
        description="RL environment that emits token traces and depends on model inference runtime.",
    ),
    ProfileSpec(
        profile=ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT,
        required_protocols=(
            PrimitiveProtocol.CATALOG_BACKED,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE,
            PrimitiveProtocol.VERIFIER_BACKED,
        ),
        description="Harness-managed benchmark or eval service.",
    ),
)


def all_profile_specs() -> tuple[ProfileSpec, ...]:
    return _PROFILE_SPECS


def spec_for_profile(profile: ExecutionProfile | str) -> ProfileSpec:
    normalized = ExecutionProfile(str(profile))
    for spec in _PROFILE_SPECS:
        if spec.profile == normalized:
            return spec
    raise KeyError(f"Unknown profile: {profile!r}")


def infer_profiles(
    protocol_support: dict[PrimitiveProtocol | str, CapabilityLevel | str],
    *,
    minimum_level: CapabilityLevel = CapabilityLevel.DERIVED,
) -> list[ExecutionProfile]:
    normalized = {
        PrimitiveProtocol(str(key)): CapabilityLevel.parse(value)
        for key, value in protocol_support.items()
    }
    matches: list[ExecutionProfile] = []
    for spec in _PROFILE_SPECS:
        if all(normalized.get(proto, CapabilityLevel.UNSUPPORTED).rank >= minimum_level.rank for proto in spec.required_protocols):
            matches.append(spec.profile)
    return matches


def missing_protocols(
    profile: ExecutionProfile | str,
    protocol_support: dict[PrimitiveProtocol | str, CapabilityLevel | str],
    *,
    minimum_level: CapabilityLevel = CapabilityLevel.DERIVED,
) -> list[PrimitiveProtocol]:
    spec = spec_for_profile(ExecutionProfile(str(profile)))
    normalized = {
        PrimitiveProtocol(str(key)): CapabilityLevel.parse(value)
        for key, value in protocol_support.items()
    }
    return [
        proto
        for proto in spec.required_protocols
        if normalized.get(proto, CapabilityLevel.UNSUPPORTED).rank < minimum_level.rank
    ]


def describe_profiles(profiles: Iterable[ExecutionProfile | str]) -> dict[str, str]:
    return {spec_for_profile(ExecutionProfile(str(profile))).profile.value: spec_for_profile(ExecutionProfile(str(profile))).description for profile in profiles}
