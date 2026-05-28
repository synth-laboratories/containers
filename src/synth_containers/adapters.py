from __future__ import annotations

from dataclasses import dataclass, field

from .ontology import CapabilityLevel, CoreNoun, ExecutionProfile, PrimitiveProtocol, RuntimeKind
from .serde import JsonDataclassMixin


@dataclass(frozen=True, slots=True)
class FrameworkAdapterDescriptor(JsonDataclassMixin):
    framework: str
    runtime_kind: RuntimeKind
    noun_fidelity: dict[CoreNoun, CapabilityLevel]
    protocol_fidelity: dict[PrimitiveProtocol, CapabilityLevel]
    profile_fidelity: dict[ExecutionProfile, CapabilityLevel]
    notes: list[str] = field(default_factory=list)



def environments_old_descriptor() -> FrameworkAdapterDescriptor:
    return FrameworkAdapterDescriptor(
        framework="environments_old",
        runtime_kind=RuntimeKind.ENVIRONMENT,
        noun_fidelity={
            CoreNoun.RUNTIME: CapabilityLevel.NATIVE,
            CoreNoun.ACTOR: CapabilityLevel.NATIVE,
            CoreNoun.ACTION: CapabilityLevel.NATIVE,
            CoreNoun.OBSERVATION: CapabilityLevel.NATIVE,
            CoreNoun.STATE: CapabilityLevel.NATIVE,
            CoreNoun.EXECUTION: CapabilityLevel.NATIVE,
            CoreNoun.OUTCOME: CapabilityLevel.DERIVED,
            CoreNoun.TASK_INSTANCE: CapabilityLevel.NATIVE,
            CoreNoun.CHECKPOINT: CapabilityLevel.NATIVE,
            CoreNoun.TASK_CATALOG: CapabilityLevel.NATIVE,
        },
        protocol_fidelity={
            PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.NATIVE,
            PrimitiveProtocol.RESETTABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.STEPPABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.OBSERVABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.STATE_READABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.CHECKPOINTABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.RESTORABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.FORKABLE: CapabilityLevel.DERIVED,
            PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE: CapabilityLevel.DERIVED,
            PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.DERIVED,
            PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.VERIFIER_BACKED: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.TOOL_CALLABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING: CapabilityLevel.UNSUPPORTED,
            PrimitiveProtocol.MULTI_ACTOR: CapabilityLevel.UNSUPPORTED,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED: CapabilityLevel.UNSUPPORTED,
        },
        profile_fidelity={
            ExecutionProfile.GYM_STYLE_ENVIRONMENT: CapabilityLevel.NATIVE,
            ExecutionProfile.CHECKPOINTABLE_STATEFUL_ENVIRONMENT: CapabilityLevel.NATIVE,
            ExecutionProfile.CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT: CapabilityLevel.DERIVED,
            ExecutionProfile.RL_TRAJECTORY_EMITTER: CapabilityLevel.APPROXIMATE,
        },
        notes=[
            "Preserves engine-vs-environment split and checkpointing semantics.",
            "Async rollout control usually arrives through a wrapper service rather than the core framework.",
        ],
    )



def openenv_descriptor() -> FrameworkAdapterDescriptor:
    return FrameworkAdapterDescriptor(
        framework="openenv",
        runtime_kind=RuntimeKind.ENVIRONMENT,
        noun_fidelity={
            CoreNoun.RUNTIME: CapabilityLevel.NATIVE,
            CoreNoun.ACTOR: CapabilityLevel.NATIVE,
            CoreNoun.ACTION: CapabilityLevel.NATIVE,
            CoreNoun.OBSERVATION: CapabilityLevel.NATIVE,
            CoreNoun.STATE: CapabilityLevel.NATIVE,
            CoreNoun.EXECUTION: CapabilityLevel.NATIVE,
            CoreNoun.OUTCOME: CapabilityLevel.NATIVE,
            CoreNoun.TASK_INSTANCE: CapabilityLevel.APPROXIMATE,
            CoreNoun.TASK_CATALOG: CapabilityLevel.UNSUPPORTED,
        },
        protocol_fidelity={
            PrimitiveProtocol.RESETTABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.STEPPABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.OBSERVABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.STATE_READABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.CHECKPOINTABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.RESTORABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.DERIVED,
            PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.NATIVE,
            PrimitiveProtocol.VERIFIER_BACKED: CapabilityLevel.UNSUPPORTED,
            PrimitiveProtocol.TOOL_CALLABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.MULTI_ACTOR: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.UNSUPPORTED,
        },
        profile_fidelity={
            ExecutionProfile.GYM_STYLE_ENVIRONMENT: CapabilityLevel.NATIVE,
            ExecutionProfile.RL_TRAJECTORY_EMITTER: CapabilityLevel.NATIVE,
            ExecutionProfile.CHECKPOINTABLE_STATEFUL_ENVIRONMENT: CapabilityLevel.APPROXIMATE,
            ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT: CapabilityLevel.UNSUPPORTED,
        },
        notes=[
            "Best external reference for clean reset/step/state client-server interaction.",
            "Task catalog and benchmark orchestration are weaker than in the Synth substrate.",
        ],
    )



def archipelago_descriptor() -> FrameworkAdapterDescriptor:
    return FrameworkAdapterDescriptor(
        framework="archipelago",
        runtime_kind=RuntimeKind.SANDBOX,
        noun_fidelity={
            CoreNoun.RUNTIME: CapabilityLevel.NATIVE,
            CoreNoun.ACTOR: CapabilityLevel.NATIVE,
            CoreNoun.ACTION: CapabilityLevel.NATIVE,
            CoreNoun.OBSERVATION: CapabilityLevel.APPROXIMATE,
            CoreNoun.STATE: CapabilityLevel.APPROXIMATE,
            CoreNoun.EXECUTION: CapabilityLevel.NATIVE,
            CoreNoun.OUTCOME: CapabilityLevel.NATIVE,
            CoreNoun.TASK_INSTANCE: CapabilityLevel.NATIVE,
            CoreNoun.ARTIFACT: CapabilityLevel.NATIVE,
            CoreNoun.TOOL: CapabilityLevel.NATIVE,
            CoreNoun.CHECKPOINT: CapabilityLevel.APPROXIMATE,
        },
        protocol_fidelity={
            PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.RESETTABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.STEPPABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.OBSERVABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.STATE_READABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.CHECKPOINTABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.RESTORABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.FORKABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.DERIVED,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.NATIVE,
            PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.VERIFIER_BACKED: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TOOL_CALLABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING: CapabilityLevel.UNSUPPORTED,
            PrimitiveProtocol.MULTI_ACTOR: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED: CapabilityLevel.APPROXIMATE,
        },
        profile_fidelity={
            ExecutionProfile.SANDBOXED_MCP_WORLD: CapabilityLevel.NATIVE,
            ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT: CapabilityLevel.NATIVE,
            ExecutionProfile.CHECKPOINTABLE_STATEFUL_ENVIRONMENT: CapabilityLevel.APPROXIMATE,
        },
        notes=[
            "Sandbox and MCP gateway patterns are first-class.",
            "Snapshots are often grading-oriented rather than branchable true environment checkpoints.",
        ],
    )



def harbor_descriptor() -> FrameworkAdapterDescriptor:
    return FrameworkAdapterDescriptor(
        framework="harbor",
        runtime_kind=RuntimeKind.HARNESS,
        noun_fidelity={
            CoreNoun.RUNTIME: CapabilityLevel.NATIVE,
            CoreNoun.ACTOR: CapabilityLevel.NATIVE,
            CoreNoun.ACTION: CapabilityLevel.NATIVE,
            CoreNoun.OBSERVATION: CapabilityLevel.APPROXIMATE,
            CoreNoun.STATE: CapabilityLevel.UNSUPPORTED,
            CoreNoun.EXECUTION: CapabilityLevel.NATIVE,
            CoreNoun.OUTCOME: CapabilityLevel.NATIVE,
            CoreNoun.TASK_INSTANCE: CapabilityLevel.NATIVE,
            CoreNoun.ARTIFACT: CapabilityLevel.NATIVE,
            CoreNoun.TRACE: CapabilityLevel.NATIVE,
            CoreNoun.VERIFIER_RESULT: CapabilityLevel.NATIVE,
            CoreNoun.TASK_CATALOG: CapabilityLevel.NATIVE,
        },
        protocol_fidelity={
            PrimitiveProtocol.CATALOG_BACKED: CapabilityLevel.NATIVE,
            PrimitiveProtocol.RESETTABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.STEPPABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.OBSERVABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.STATE_READABLE: CapabilityLevel.UNSUPPORTED,
            PrimitiveProtocol.CHECKPOINTABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.RESTORABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.FORKABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.ASYNC_ROLLOUT_RUNNABLE: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TRACE_EMITTING: CapabilityLevel.NATIVE,
            PrimitiveProtocol.REWARD_EMITTING: CapabilityLevel.NATIVE,
            PrimitiveProtocol.VERIFIER_BACKED: CapabilityLevel.NATIVE,
            PrimitiveProtocol.TOOL_CALLABLE: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.TOKEN_TRACE_EMITTING: CapabilityLevel.APPROXIMATE,
            PrimitiveProtocol.MULTI_ACTOR: CapabilityLevel.DERIVED,
            PrimitiveProtocol.PROXIED_INFERENCE_BACKED: CapabilityLevel.NATIVE,
        },
        profile_fidelity={
            ExecutionProfile.HARNESS_MANAGED_BENCHMARK_ENVIRONMENT: CapabilityLevel.NATIVE,
            ExecutionProfile.RL_TRAJECTORY_EMITTER: CapabilityLevel.APPROXIMATE,
            ExecutionProfile.CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT: CapabilityLevel.APPROXIMATE,
        },
        notes=[
            "Strong benchmark/eval/RL orchestration reference.",
            "Harness-centric design means engine state and native checkpoint semantics are weaker than substrate-first systems.",
        ],
    )



def all_framework_descriptors() -> list[FrameworkAdapterDescriptor]:
    return [
        environments_old_descriptor(),
        openenv_descriptor(),
        archipelago_descriptor(),
        harbor_descriptor(),
    ]
