from __future__ import annotations

from enum import StrEnum


CONTRACT_VERSION = "2026-04-23"


class CoreNoun(StrEnum):
    RUNTIME = "runtime"
    ACTOR = "actor"
    ACTION = "action"
    OBSERVATION = "observation"
    STATE = "state"
    EXECUTION = "execution"
    OUTCOME = "outcome"
    TASK_INSTANCE = "task_instance"
    TASK = "task"
    ARTIFACT = "artifact"
    TRACE = "trace"
    CHECKPOINT = "checkpoint"
    TRAJECTORY = "trajectory"
    REWARD = "reward"
    VERIFIER_RESULT = "verifier_result"
    TOOL = "tool"
    AGENT_SESSION = "agent_session"
    TASK_CATALOG = "task_catalog"


class RuntimeKind(StrEnum):
    ENVIRONMENT = "environment"
    SANDBOX = "sandbox"
    SESSION = "session"
    HARNESS = "harness"
    EVALUATOR = "evaluator"
    PROXY = "proxy"


class ExecutionKind(StrEnum):
    ROLLOUT = "rollout"
    SESSION = "session"
    EPISODE = "episode"
    EVAL_RUN = "eval_run"


class OutcomeKind(StrEnum):
    REWARD = "reward"
    SCORE = "score"
    GRADE = "grade"
    VERIFIER_RESULT = "verifier_result"
    PASS_FAIL = "pass_fail"


class PrimitiveProtocol(StrEnum):
    CATALOG_BACKED = "catalog_backed"
    RESETTABLE = "resettable"
    STEPPABLE = "steppable"
    OBSERVABLE = "observable"
    STATE_READABLE = "state_readable"
    CHECKPOINTABLE = "checkpointable"
    RESTORABLE = "restorable"
    FORKABLE = "forkable"
    ROLLOUT_RUNNABLE = "rollout_runnable"
    ASYNC_ROLLOUT_RUNNABLE = "async_rollout_runnable"
    TRACE_EMITTING = "trace_emitting"
    REWARD_EMITTING = "reward_emitting"
    VERIFIER_BACKED = "verifier_backed"
    TOOL_CALLABLE = "tool_callable"
    TOKEN_TRACE_EMITTING = "token_trace_emitting"
    MULTI_ACTOR = "multi_actor"
    PROXIED_INFERENCE_BACKED = "proxied_inference_backed"


class ExecutionProfile(StrEnum):
    STATELESS_EVALUATOR = "stateless_evaluator"
    GYM_STYLE_ENVIRONMENT = "gym_style_environment"
    CHECKPOINTABLE_STATEFUL_ENVIRONMENT = "checkpointable_stateful_environment"
    CHECKPOINTABLE_LONG_HORIZON_ENVIRONMENT = "checkpointable_long_horizon_environment"
    MULTI_AGENT_LONG_HORIZON_ENVIRONMENT = "multi_agent_long_horizon_environment"
    SANDBOXED_MCP_WORLD = "sandboxed_mcp_world"
    RL_TRAJECTORY_EMITTER = "rl_trajectory_emitter"
    TOKEN_LEVEL_RL_ENVIRONMENT = "token_level_rl_environment"
    HARNESS_MANAGED_BENCHMARK_ENVIRONMENT = "harness_managed_benchmark_environment"


class CapabilityLevel(StrEnum):
    NATIVE = "native"
    DERIVED = "derived"
    APPROXIMATE = "approximate"
    UNSUPPORTED = "unsupported"

    @property
    def rank(self) -> int:
        return {
            CapabilityLevel.UNSUPPORTED: 0,
            CapabilityLevel.APPROXIMATE: 1,
            CapabilityLevel.DERIVED: 2,
            CapabilityLevel.NATIVE: 3,
        }[self]

    @classmethod
    def parse(cls, value: "CapabilityLevel | str") -> "CapabilityLevel":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower().replace("-", "_")
        if not text:
            return cls.UNSUPPORTED
        return cls(text)


class RolloutMode(StrEnum):
    BLOCKING = "blocking"
    ASYNC = "async"


class StatefulnessTier(StrEnum):
    STATELESS = "stateless"
    EPISODIC = "episodic"
    STATEFUL = "stateful"
    LONG_HORIZON = "long_horizon"


class CheckpointSemantics(StrEnum):
    NONE = "none"
    TRUE_ENVIRONMENT_SNAPSHOT = "true_environment_snapshot"
    REQUEST_SNAPSHOT_REPLAY = "request_snapshot_replay"
    AUDIT_SNAPSHOT = "audit_snapshot"
    GRADING_SNAPSHOT = "grading_snapshot"
    PARTIAL_STATE = "partial_state"
    CODEX_SESSION_WORKSPACE_SNAPSHOT = "codex_session_workspace_snapshot"


class ResumeSemantics(StrEnum):
    UNSUPPORTED = "unsupported"
    TRUE_ENVIRONMENT_SNAPSHOT = "true_environment_snapshot"
    REQUEST_SNAPSHOT_REPLAY = "request_snapshot_replay"
    MANUAL_REPLAY = "manual_replay"
    CODEX_SESSION_WORKSPACE_SNAPSHOT = "codex_session_workspace_snapshot"


class RewardSource(StrEnum):
    UNKNOWN = "unknown"
    ENVIRONMENT = "environment"
    VERIFIER = "verifier"
    WRAPPER_PROXY = "wrapper_proxy"
    MIXED = "mixed"


class RuntimeFamily(StrEnum):
    REQUEST_RESPONSE = "request_response"
    CODEX_SESSION = "codex_session"
    MCP_WORLD = "mcp_world"
