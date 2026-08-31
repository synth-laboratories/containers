"""Catalog targets for the Craftax gold image. Not registered in public TARGETS."""

from __future__ import annotations

import os

from dataclasses import replace

from synth_containers.gold_http import facade_health_for
from synth_containers.gold_runtime import GoldRuntime
from synth_containers.platform.affordances import AffordanceMap
from synth_containers.platform.targets import (
    PolicySeed,
    RewardKind,
    TargetRuntimeKind,
    TargetSpec,
)

from .gepa import GEPA, compat_rollout, metadata_extra, mount_routes
from .world import (
    ENGINE,
    ENVIRONMENT_REF,
    MAX_STEPS_ENV,
    URL_ENV,
    task_payload,
    task_payload_default,
)

OBJECTIVE = "Make measurable Craftax achievement progress while staying alive."


def _nanohorizon_policy_seed() -> PolicySeed:
    """Public policy identity configured by the image deployment.

    Runtime secrets and sampler URLs are deliberately not part of metadata;
    they continue to arrive through PUT /policy and the secrets proxy.
    """
    return PolicySeed(
        os.environ.get("SYNTH_NANOHORIZON_POLICY_CONFIG", "glm-5.3-flash"),
        "nanohorizon",
        {
            "provider": os.environ.get("SYNTH_NANOHORIZON_POLICY_PROVIDER", "openrouter"),
            "model": os.environ.get("SYNTH_NANOHORIZON_POLICY_MODEL", "z-ai/glm-5.3-flash"),
            "api": os.environ.get("SYNTH_NANOHORIZON_POLICY_API", "chat_completions"),
            "base_url": os.environ.get(
                "SYNTH_NANOHORIZON_POLICY_BASE_URL", "https://openrouter.ai/api/v1"
            ),
            "api_key_env": os.environ.get(
                "SYNTH_NANOHORIZON_POLICY_API_KEY_ENV", "OPENROUTER_API_KEY"
            ),
            "effort": os.environ.get("SYNTH_NANOHORIZON_POLICY_EFFORT", "medium"),
            "timeout_seconds": float(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_TIMEOUT_SECONDS", "180")
            ),
            "thinking_budget": int(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_THINKING_BUDGET", "640")
            ),
            "answer_max_tokens": int(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_ANSWER_MAX_TOKENS", "256")
            ),
            "context_token_budget": int(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_CONTEXT_TOKEN_BUDGET", "32000")
            ),
            "compact_after_tokens": int(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_COMPACT_AFTER_TOKENS", "10000")
            ),
            "max_compactions": int(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_MAX_COMPACTIONS", "20")
            ),
            "min_request_interval": float(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_MIN_REQUEST_INTERVAL", "1")
            ),
            "sampler_retries": int(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_SAMPLER_RETRIES", "12")
            ),
            "retry_max_wait": float(
                os.environ.get("SYNTH_NANOHORIZON_POLICY_RETRY_MAX_WAIT", "60")
            ),
        },
    )


_RUNTIME = GoldRuntime(
    environment_ref=ENVIRONMENT_REF,
    task_payload=task_payload,
    url_env=URL_ENV,
    engine=ENGINE,
    max_steps_env=MAX_STEPS_ENV,
)

facade_health = facade_health_for(URL_ENV, engine=ENGINE)


def _env(items: dict[str, str]) -> AffordanceMap:
    return AffordanceMap(by_role={"environment": dict(items), "policy": {}, "evaluator": {}})


# The policy ladder. One image, four rungs; the rollout picks by policy_ref.
POLICY_SEEDS = (
    # Free transport proof. Never a graded eval: no model is involved.
    PolicySeed(
        "engine_acceptance",
        "scripted_react",
        {"env_name": ENGINE},
    ),
    PolicySeed(
        "uniform_baseline",
        "valid_action_uniform",
        {"seed": 0, "env_name": ENGINE},
    ),
    PolicySeed(
        "single_call_muse",
        "single_call",
        {
            "provider": "openrouter",
            "model": "meta/muse-spark-1.1",
            "api": "chat_completions",
            "api_key_env": "OPENROUTER_API_KEY",
            "effort": "low",
            "max_tokens": 8192,
            "horizon": 32,
            "env_name": ENGINE,
            "objective": OBJECTIVE,
        },
    ),
    PolicySeed(
        "muse_spark_medium",
        "react",
        {
            "provider": "openrouter",
            "model": "meta/muse-spark-1.1",
            "effort": "medium",
            "api_key_env": "OPENROUTER_API_KEY",
            "max_tokens": 2048,
            "parse_retries": 2,
            "compact_every": 16,
            "env_name": ENGINE,
            "objective": OBJECTIVE,
        },
    ),
    PolicySeed(
        "responses_react_gpt",
        "responses_react",
        {
            "provider": "openai_responses",
            "model": "gpt-5.6",
            "effort": "medium",
            "api_key_env": "OPENAI_API_KEY",
            "max_tokens": 4096,
            "plan_min": 1,
            "plan_max": 5,
            "env_name": ENGINE,
            "objective": OBJECTIVE,
        },
    ),
    PolicySeed(
        "agentic_codex",
        "codex_agentic",
        {
            "model": "gpt-5.6-luna",
            "sandbox": "workspace-write",
            "plan_max": 5,
            "timeout_seconds": 300,
            "env_name": ENGINE,
            "objective": OBJECTIVE,
        },
    ),
)

CRAFTAX_REACT = TargetSpec(
    target_id="craftax_react",
    runtime_family=TargetRuntimeKind.EXTERNAL,
    adapter_chain=(),
    world_ref="world:craftax_default@symbolic_survival",
    environment_ref=ENVIRONMENT_REF,
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
    policy_seeds=POLICY_SEEDS,
    max_episode_steps=120,
    runtime=_RUNTIME,
    health_probe=facade_health,
)

CRAFTAX_GOEX = replace(
    CRAFTAX_REACT,
    target_id="craftax_goex",
    true_checkpoint="native",
    event_kinds=CRAFTAX_REACT.event_kinds + ("rollout.checkpoint", "rollout.restored"),
    affordances=_env(
        {
            **CRAFTAX_REACT.affordances.by_role["environment"],
            "true_checkpoint": "native",
            "restore": "native",
            "fork": "native",
        }
    ),
)

CRAFTAX_CODE_POLICY = replace(
    CRAFTAX_REACT,
    target_id="craftax_code_policy",
    default_policy_harness="isolated_policy_process",
    policy_seeds=(PolicySeed("heuristic", "isolated_policy_process", {}),),
    affordances=_env(
        {
            **CRAFTAX_REACT.affordances.by_role["environment"],
            "bind_policy_config": "unsupported",
            "update_policy_code": "native",
            "restart_policy": "native",
        }
    ),
)

CRAFTAX_NANOHORIZON = replace(
    CRAFTAX_REACT,
    target_id="craftax_nanohorizon",
    default_policy_harness="nanohorizon",
    policy_seeds=(_nanohorizon_policy_seed(),),
    max_episode_steps=2000,
    # Inherited 10 from craftax_react, where it was the honest limit: every step
    # in the engine process serialised on one mutex, so an eleventh rollout only
    # made the other ten slower. That lock is now per-session, and a rollout is
    # idle almost all of its life -- ~7s per call waiting on the model against
    # sub-millisecond rust steps. Admission should bound what one process can
    # usefully hold, not force callers to run more processes to get concurrency.
    scale_leases=int(os.environ.get("SYNTH_CRAFTAX_SCALE_LEASES", "128")),
    affordances=_env(
        {
            **CRAFTAX_REACT.affordances.by_role["environment"],
            "bind_policy_config": "native",
            "update_policy_code": "native",
            "restart_policy": "native",
        }
    ),
    runtime=GoldRuntime(
        environment_ref=ENVIRONMENT_REF,
        task_payload=task_payload_default,
        url_env=URL_ENV,
        engine=ENGINE,
        max_steps_env=MAX_STEPS_ENV,
    ),
    optimizer_contracts=GEPA,
    metadata_extra=metadata_extra,
    mount_routes=mount_routes,
    compat_rollout=compat_rollout,
)

TARGETS = {
    spec.target_id: spec
    for spec in (CRAFTAX_REACT, CRAFTAX_GOEX, CRAFTAX_CODE_POLICY, CRAFTAX_NANOHORIZON)
}
