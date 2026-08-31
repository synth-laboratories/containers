"""Target specs for the containers-compat façade (in-process fixtures for PR CI)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .affordances import AffordanceMap


class TargetRuntimeKind(StrEnum):
    """Child of TargetRuntime. Not contracts.RuntimeFamily (codex/mcp/http)."""

    EXTERNAL = "external"
    HARBOR = "harbor"
    DIGBENCH = "digbench"
    OPENENV = "openenv"


class RewardKind(StrEnum):
    ENV_SUM = "env_sum"
    SCRIPT = "script"
    ENV_STATUS = "env_status"


class TaskInstanceStatus(StrEnum):
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    GAME_OVER = "game_over"
    CANCELLED = "cancelled"
    TERMINAL = "terminal"


class PolicyInstallStatus(StrEnum):
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    FAILED = "failed"


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
    runtime: Any | None = None
    health_probe: Any | None = None
    optimizer_contracts: dict[str, Any] | None = None
    metadata_extra: Any | None = None
    mount_routes: Any | None = None
    compat_rollout: Any | None = None
    # Optional task-specific admission. It runs after a caller has bound an
    # advertised policy but before any runtime/child-container work begins.
    # Return a normal platform error payload to refuse without synthesizing an
    # attempt from an environment that cannot honestly start.
    admission: Any | None = None
    # Stable catalog identity. Older image packages may omit these fields; the
    # platform then derives a conservative identity from the target refs.
    task_id: str | None = None
    task_family: str | None = None


def _env(items: dict[str, str]) -> AffordanceMap:
    return AffordanceMap(by_role={"environment": dict(items), "policy": {}, "evaluator": {}})


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
    policy_seeds=(PolicySeed("echo", "gym_loop", {}),),
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

TARGETS: dict[str, TargetSpec] = {
    spec.target_id: spec
    for spec in (
        HARBOR_PUBLIC,
        HARBOR_DOCKER,
        DEO_NESTED,
        OPENENV_ECHO,
        DIGBENCH_MOCK,
        DIGBENCH_PUBLIC,
    )
}

PR_TARGETS = (
    "harbor_public",
    "deo_nested",
    "openenv_echo",
    "digbench_mock",
)

PAID_TARGETS = (
    "digbench_public",
)
