"""Target specs for the containers-compat façade (in-process fixtures for PR CI)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from .affordances import AffordanceMap


class TargetRuntimeKind(StrEnum):
    """Child of TargetRuntime. Not contracts.RuntimeFamily (codex/mcp/http)."""

    CRAFTAX = "craftax"
    HARBOR = "harbor"
    DIGBENCH = "digbench"
    OPENENV = "openenv"
    BANKING77 = "banking77"
    GSM8K = "gsm8k"
    HEALTHBENCH = "healthbench"


class RewardKind(StrEnum):
    ENV_SUM = "env_sum"
    SCRIPT = "script"
    ENV_STATUS = "env_status"


class ScriptNode(StrEnum):
    REWARD_TXT = "reward.txt"
    HELDOUT_GATE = "heldout_gate"


@dataclass(frozen=True)
class PolicySeed:
    config_id: str
    harness: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    runtime_family: TargetRuntimeKind
    adapter_chain: tuple[str, ...]
    world_ref: str
    environment_ref: str
    evaluation_plan_ref: str
    default_policy_harness: str
    scale_leases: int
    retention: str
    reward_kind: RewardKind
    live_reward: bool
    live_frames: str
    true_checkpoint: str
    blocking_trial: str
    mcp_bind: str
    reconnect: str
    event_kinds: tuple[str, ...]
    affordances: AffordanceMap = field(default_factory=AffordanceMap)
    policy_seeds: tuple[PolicySeed, ...] = ()
    script_node: ScriptNode = ScriptNode.REWARD_TXT
    max_episode_steps: int | None = None
    gold_base_url: str | None = None


def _env(items: dict[str, str]) -> AffordanceMap:
    return AffordanceMap(by_role={"environment": dict(items), "policy": {}, "evaluator": {}})


CRAFTAX_ENGINE = TargetSpec(
    target_id="craftax_engine",
    runtime_family=TargetRuntimeKind.CRAFTAX,
    adapter_chain=(),
    world_ref="world:craftax_default@symbolic_survival",
    environment_ref="env:craftax_fixture",
    evaluation_plan_ref="eval:craftax.env_sum",
    default_policy_harness="react",
    scale_leases=10,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=True,
    live_frames="native",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "env.episode.opened",
        "observation",
        "action",
        "reward_signal",
        "frame",
        "span.policy.opened",
        "span.step.opened",
        "capture.closed",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "live_frames": "native",
            "partial_trace": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "native",
            "scale_leases": "native",
            "bind_policy_config": "native",
            "restart_policy": "native",
        }
    ),
    max_episode_steps=8,
)

CRAFTAX_REACT = TargetSpec(
    target_id="craftax_react",
    runtime_family=TargetRuntimeKind.CRAFTAX,
    adapter_chain=(),
    world_ref="world:craftax_default@symbolic_survival",
    environment_ref="env:craftax_gold",
    evaluation_plan_ref="eval:craftax.env_sum",
    default_policy_harness="react",
    scale_leases=10,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=True,
    live_frames="native",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "env.episode.opened",
        "observation",
        "action",
        "reward_signal",
        "frame",
        "task_resolved",
        "action_applied",
        "state_transition",
        "entity_transition",
        "resource_delta",
        "achievement_unlocked",
        "reward_delta",
        "checkpoint_cadence",
        "episode_truncated",
        "terminal",
        "span.policy.opened",
        "span.step.opened",
        "capture.closed",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "live_frames": "native",
            "partial_trace": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "native",
            "scale_leases": "native",
            "bind_policy_config": "native",
            "restart_policy": "native",
            "token_trace": "derived",
        }
    ),
    policy_seeds=(
        PolicySeed(
            "muse_spark_medium",
            "react",
            {
                "provider": "openrouter",
                "model": "meta/muse-spark-1.1",
                "effort": "medium",
                "api_key_env": "OPENROUTER_API_KEY",
                # Muse Spark medium needs room for hidden reasoning before its
                # short structured action batch. 384 consistently ended at
                # the reasoning budget with no model-authored content.
                "max_tokens": 2048,
                "parse_retries": 2,
                "context_token_budget": 16000,
                "compact_at": 0.7,
                "keep_recent_messages": 8,
                "keep_recent_frames": 2,
                "observation_mode": "text",
            },
        ),
    ),
    max_episode_steps=120,
    gold_base_url="http://127.0.0.1:8098",
)

# Dedicated GoEx target: same gold Rust environment and paid ReAct harness, but
# only this surface advertises the paired environment+policy checkpoint contract
# implemented by the Containers adapter. The ordinary Craftax viewer remains
# conservatively unsupported.
CRAFTAX_GOEX = replace(
    CRAFTAX_REACT,
    target_id="craftax_goex",
    true_checkpoint="native",
    event_kinds=CRAFTAX_REACT.event_kinds
    + ("rollout.checkpoint", "rollout.restored"),
    affordances=_env(
        {
            **CRAFTAX_REACT.affordances.by_role["environment"],
            "true_checkpoint": "native",
            "restore": "native",
            "fork": "native",
        }
    ),
)

CRAFTAX_CODE_POLICY = TargetSpec(
    target_id="craftax_code_policy",
    runtime_family=TargetRuntimeKind.CRAFTAX,
    adapter_chain=(),
    world_ref="world:craftax_default@symbolic_survival",
    environment_ref="env:craftax_fixture",
    evaluation_plan_ref="eval:craftax.env_sum",
    default_policy_harness="isolated_policy_process",
    scale_leases=4,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=True,
    live_frames="native",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=("trace.opened", "observation", "action", "reward_signal", "frame", "status"),
    affordances=_env(
        {
            "step": "native",
            "live_frames": "native",
            "update_policy_code": "native",
            "restart_policy": "native",
            "bind_policy_config": "unsupported",
            "true_checkpoint": "unsupported",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_reward": "native",
        }
    ),
    max_episode_steps=8,
)

HARBOR_PUBLIC = TargetSpec(
    target_id="harbor_public",
    runtime_family=TargetRuntimeKind.HARBOR,
    adapter_chain=("harbor",),
    world_ref="world:harbor_tb_public",
    environment_ref="env:harbor_sandbox",
    evaluation_plan_ref="eval:harbor.script_reward_txt",
    default_policy_harness="harbor_fused",
    scale_leases=2,
    retention="run",
    reward_kind=RewardKind.SCRIPT,
    live_reward=False,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="native",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "trial.planned",
        "trial.launched",
        "tools",
        "stdout",
        "verifier",
        "status",
    ),
    affordances=_env(
        {
            "blocking_trial": "native",
            "step": "unsupported",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "live_reward": "unsupported",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "bind_policy_config": "native",
            "grading_snapshot": "derived",
        }
    ),
    policy_seeds=(
        PolicySeed("luna_med", "harbor_fused", {"model": "gpt-5.6-luna", "effort": "medium"}),
        PolicySeed("sol_med", "harbor_fused", {"model": "gpt-5.6-sol", "effort": "medium"}),
    ),
)

HARBOR_DOCKER = TargetSpec(
    target_id="harbor_docker",
    runtime_family=TargetRuntimeKind.HARBOR,
    adapter_chain=("harbor",),
    world_ref="world:harbor_tb_public",
    environment_ref="env:harbor_docker",
    evaluation_plan_ref="eval:harbor.script_reward_txt",
    default_policy_harness="harbor_fused",
    scale_leases=2,
    retention="run",
    reward_kind=RewardKind.SCRIPT,
    live_reward=False,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="native",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "trial.planned",
        "trial.launched",
        "tools",
        "stdout",
        "verifier",
        "status",
    ),
    affordances=_env(
        {
            "blocking_trial": "native",
            "step": "unsupported",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "live_reward": "unsupported",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "bind_policy_config": "native",
            "grading_snapshot": "derived",
        }
    ),
    policy_seeds=(
        PolicySeed("luna_med", "harbor_fused", {"model": "gpt-5.6-luna", "effort": "medium"}),
        PolicySeed("sol_med", "harbor_fused", {"model": "gpt-5.6-sol", "effort": "medium"}),
    ),
)

DEO_NESTED = TargetSpec(
    target_id="deo_nested",
    runtime_family=TargetRuntimeKind.HARBOR,
    adapter_chain=("harbor",),
    world_ref="world:harbor_gamebench_deo",
    environment_ref="env:harbor_sandbox",
    evaluation_plan_ref="eval:deo.heldout_gate",
    default_policy_harness="harbor_fused",
    scale_leases=2,
    retention="run",
    reward_kind=RewardKind.SCRIPT,
    live_reward=False,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="native",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=("trace.opened", "trial.planned", "trial.launched", "verifier", "status"),
    script_node=ScriptNode.HELDOUT_GATE,
    affordances=_env(
        {
            "blocking_trial": "native",
            "live_frames": "unsupported",
            "step": "unsupported",
            "true_checkpoint": "unsupported",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
        }
    ),
)

OPENENV_ECHO = TargetSpec(
    target_id="openenv_echo",
    runtime_family=TargetRuntimeKind.OPENENV,
    adapter_chain=("openenv",),
    world_ref="world:openenv_echo",
    environment_ref="env:echo",
    evaluation_plan_ref="eval:echo.env_reward",
    default_policy_harness="gym_loop",
    scale_leases=4,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=False,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="unsupported",
    event_kinds=("trace.opened", "observation", "action", "reward_signal", "status"),
    affordances=_env(
        {
            "step": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "live_frames": "unsupported",
            "live_reward": "unsupported",
        }
    ),
)

DIGBENCH_PUBLIC = TargetSpec(
    target_id="digbench_public",
    runtime_family=TargetRuntimeKind.DIGBENCH,
    adapter_chain=(),
    world_ref="world:digbench:P-1",
    environment_ref="env:digbench_relay",
    evaluation_plan_ref="eval:digbench.env_status",
    default_policy_harness="react_legal_actions",
    scale_leases=2,
    retention="run",
    reward_kind=RewardKind.ENV_STATUS,
    live_reward=False,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="native",
    reconnect="native",
    event_kinds=(
        "trace.opened",
        "session.opened",
        "observation",
        "legal_actions",
        "stats",
        "action",
        "invalid_action",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "unsupported",
            "reconnect": "native",
            "mcp_bind": "native",
            "bind_policy_config": "native",
            "credential_isolation": "native",
        }
    ),
    policy_seeds=(
        PolicySeed("react_legal_actions", "react_legal_actions", {"mcp_bind": "unused"}),
        PolicySeed(
            "agentic_codex", "codex", {"model": "gpt-5.6-luna", "mcp_bind": "digbench-mcp"}
        ),
    ),
)

DIGBENCH_MOCK = TargetSpec(
    target_id="digbench_mock",
    runtime_family=TargetRuntimeKind.DIGBENCH,
    adapter_chain=(),
    world_ref="world:digbench:P-1",
    environment_ref="env:digbench_mock",
    evaluation_plan_ref="eval:digbench.env_status",
    default_policy_harness="react_legal_actions",
    scale_leases=2,
    retention="run",
    reward_kind=RewardKind.ENV_STATUS,
    live_reward=False,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="native",
    reconnect="native",
    event_kinds=(
        "trace.opened",
        "session.opened",
        "observation",
        "legal_actions",
        "stats",
        "action",
        "invalid_action",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "unsupported",
            "reconnect": "native",
            "mcp_bind": "native",
            "bind_policy_config": "native",
            "credential_isolation": "native",
        }
    ),
    policy_seeds=(
        PolicySeed("react_legal_actions", "react_legal_actions", {"mcp_bind": "unused"}),
        PolicySeed(
            "agentic_codex", "codex", {"model": "gpt-5.6-luna", "mcp_bind": "digbench-mcp"}
        ),
    ),
)

BANKING77_CLASSIFY = TargetSpec(
    target_id="banking77_classify",
    runtime_family=TargetRuntimeKind.BANKING77,
    adapter_chain=(),
    world_ref="world:banking77@heldout",
    environment_ref="env:banking77_dataset",
    evaluation_plan_ref="banking77_eval.v1",
    default_policy_harness="classify",
    scale_leases=8,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=True,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "env.episode.opened",
        "observation",
        "action",
        "reward_signal",
        "span.policy.opened",
        "span.policy.closed",
        "env.episode.closed",
        "capture.closed",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "native",
            "scale_leases": "native",
            "bind_policy_config": "native",
            "restart_policy": "native",
        }
    ),
    policy_seeds=(
        PolicySeed("dataset_gold", "dataset_gold", {"source": "environment"}),
        PolicySeed("classify", "classify", {"source": "policy"}),
    ),
)

GSM8K_SOLVE = TargetSpec(
    target_id="gsm8k_solve",
    runtime_family=TargetRuntimeKind.GSM8K,
    adapter_chain=(),
    world_ref="world:gsm8k@heldout",
    environment_ref="env:gsm8k_dataset",
    evaluation_plan_ref="gsm8k_eval.v1",
    default_policy_harness="solve",
    scale_leases=8,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=True,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "env.episode.opened",
        "observation",
        "action",
        "token_capture",
        "reward_signal",
        "span.policy.opened",
        "span.policy.closed",
        "env.episode.closed",
        "capture.closed",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "native",
            "scale_leases": "native",
            "bind_policy_config": "native",
            "restart_policy": "native",
        }
    ),
    policy_seeds=(
        PolicySeed("dataset_gold", "dataset_gold", {"source": "environment"}),
        PolicySeed("solve", "solve", {"source": "policy"}),
    ),
)

HEALTHBENCH_CHAT = TargetSpec(
    target_id="healthbench_chat",
    runtime_family=TargetRuntimeKind.HEALTHBENCH,
    adapter_chain=(),
    world_ref="world:healthbench@eval",
    environment_ref="env:healthbench_physician_rubrics",
    evaluation_plan_ref="healthbench_eval.v1",
    default_policy_harness="chat_completion",
    scale_leases=30,
    retention="run",
    reward_kind=RewardKind.ENV_SUM,
    live_reward=True,
    live_frames="unsupported",
    true_checkpoint="unsupported",
    blocking_trial="unsupported",
    mcp_bind="unused",
    reconnect="derived",
    event_kinds=(
        "trace.opened",
        "env.episode.opened",
        "observation",
        "span.policy.opened",
        "span.policy.closed",
        "action",
        "span.evaluator.opened",
        "rubric.grade",
        "span.evaluator.closed",
        "reward_signal",
        "env.episode.closed",
        "capture.closed",
        "status",
    ),
    affordances=_env(
        {
            "step": "native",
            "poll": "native",
            "sse": "derived",
            "websocket": "derived",
            "live_frames": "unsupported",
            "true_checkpoint": "unsupported",
            "restore": "unsupported",
            "fork": "unsupported",
            "live_reward": "native",
            "scale_leases": "native",
            "bind_policy_config": "native",
            "restart_policy": "native",
            "physician_rubric_grading": "native",
        }
    ),
    policy_seeds=(
        PolicySeed(
            "groq_llama31_8b",
            "chat_completion",
            {
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "api_key_env": "GROQ_API_KEY",
                "base_url": "https://api.groq.com/openai/v1",
                "max_tokens": 1536,
            },
        ),
        PolicySeed(
            "openai_gpt41_mini",
            "chat_completion",
            {
                "provider": "openai",
                "model": "gpt-4.1-mini-2025-04-14",
                "api_key_env": "OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "max_tokens": 1536,
            },
        ),
    ),
)

TARGETS: dict[str, TargetSpec] = {
    spec.target_id: spec
    for spec in (
        CRAFTAX_ENGINE,
        CRAFTAX_REACT,
        CRAFTAX_GOEX,
        CRAFTAX_CODE_POLICY,
        HARBOR_PUBLIC,
        HARBOR_DOCKER,
        DEO_NESTED,
        OPENENV_ECHO,
        DIGBENCH_MOCK,
        DIGBENCH_PUBLIC,
        BANKING77_CLASSIFY,
        GSM8K_SOLVE,
        HEALTHBENCH_CHAT,
    )
}

PR_TARGETS = (
    "craftax_engine",
    "craftax_code_policy",
    "harbor_public",
    "deo_nested",
    "openenv_echo",
    "digbench_mock",
    "banking77_classify",
    "gsm8k_solve",
)

PAID_TARGETS = (
    "craftax_react",
    "craftax_goex",
    "digbench_public",
    "healthbench_chat",
)
