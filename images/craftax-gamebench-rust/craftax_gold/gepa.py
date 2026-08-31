"""Public GEPA v2 extras for ``craftax_nanohorizon``.

Public ``synth-optimizers`` GEPA talks to these routes. Contest ``POST /rollouts``
stays the nanohorizon PUT-policy path. GEPA uses singular ``POST /rollout`` so
the overlay (candidate ``system_prompt`` + student sampler) does not collide
with the kit harness.

Do not use the cookbook ``crafter_container`` (JAX). This is rust GameBench gold.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from synth_containers.platform.http_requests import parse_create_rollout
from synth_containers.platform.state import CompatPlatform, PolicyConfig
from synth_containers.platform.targets import TargetSpec

CONTEST_EVAL_SEEDS = frozenset(range(91001, 91011))
TRAIN_SEEDS = list(range(93001, 93009))
HELDOUT_SEEDS = list(range(94001, 94005))

GEPA = {
    "version": "synth_optimizers.gepa.v2",
    "program_route": "/program",
    "taskset_route": "/taskset",
    "taskset_tasks_route": "/taskset/tasks",
    "rollout_route": "/rollout",
    "trace_route": "/rollouts/{rollout_id}/events",
}

# Keep in lockstep with nanohorizon ``submissions/gepa/policy.py`` SYSTEM_PROMPT.
SEED_PROMPT = (
    "You control the GameBench Craftax Rust environment. Maximise newly "
    "unlocked achievements while staying alive. "
    "The complete action vocabulary is: {action_names}. It never changes and is "
    "not repeated each turn. It is syntactic only: an action the schema accepts "
    "can still do nothing when its spatial, material, or tool prerequisite is "
    "missing. "
    "{state_guide} "
    "{mechanics_guide} "
    "Think about the current state, then call craftax_interact exactly once "
    "with {min_actions} to {max_actions} action names in the tool's actions "
    "argument. Actions execute in order without replanning, so choose a "
    "coherent short plan and avoid crafts whose prerequisites are not currently "
    "satisfied. Assistant content is thinking, not the action list — never "
    "write the actions as JSON or prose instead of the tool call. The tool "
    "result reports what each action did, including why one had no effect, and "
    "then the resulting state — read both and re-plan from them, not from a "
    "target that disappeared."
)


def metadata_extra(payload: dict[str, Any]) -> dict[str, Any]:
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    capabilities.setdefault("contract_version", "container_contract.v1")
    capabilities.setdefault("rollout_modes", ["blocking"])
    capabilities.setdefault(
        "operations",
        {"prepare": True, "start": True, "get": True, "poll": True, "reward": True},
    )
    metadata_blob = capabilities.setdefault("metadata", {})
    if isinstance(metadata_blob, dict):
        metadata_blob["policy_ready"] = True
        metadata_blob["program_ready"] = True
    capabilities["optimizer_contracts"] = {"gepa": GEPA}
    payload["capabilities"] = capabilities
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["optimizer_contracts"] = {"gepa": GEPA}
    payload["metadata"] = metadata
    return payload


def mount_routes(app: FastAPI, platform: CompatPlatform, spec: TargetSpec) -> None:
    del platform, spec

    @app.get("/task_info")
    async def task_info() -> Any:
        return {
            "task_id": "craftax.nanohorizon",
            "name": "NanoHorizon Craftax (rust GameBench gold)",
            "objective": "maximize episode env reward while staying alive",
            "output_kind": "tool_calls",
            "literal_training_targets": "forbidden",
            "env": "craftax_gold rust, world craftax_default@symbolic_survival",
            "proposer_hints": {
                "do_not_memorize": True,
                "do_not_use_eval_seeds": sorted(CONTEST_EVAL_SEEDS),
                "focus": (
                    "Keep the glyph legend and action vocabulary placeholders. "
                    "Improve how a 2B student gathers, crafts, and stays alive."
                ),
            },
        }

    @app.get("/program")
    async def program() -> Any:
        return {
            "version": "prompt_program.v1",
            "program_id": "nanohorizon.craftax.system_prompt.v1",
            "modules": [
                {
                    "module_id": "system_prompt",
                    "role": "system",
                    "mutable": True,
                    "candidate_field": "system_prompt",
                    "content": SEED_PROMPT,
                    "template_variables": [
                        "action_names",
                        "state_guide",
                        "mechanics_guide",
                        "min_actions",
                        "max_actions",
                    ],
                }
            ],
            "target_modules": [
                {
                    "module_id": "system_prompt",
                    "candidate_field": "system_prompt",
                    "objective": "outcome_reward",
                }
            ],
            "seed_candidate": {"system_prompt": SEED_PROMPT},
            "rollout_overlay_schema": {"system_prompt": "policy.system_prompt"},
        }

    @app.get("/taskset")
    async def taskset() -> Any:
        return {
            "taskset_id": "nanohorizon.craftax.pools",
            "splits": {"train": len(TRAIN_SEEDS), "heldout": len(HELDOUT_SEEDS)},
            "metadata": {
                "train_seeds": TRAIN_SEEDS,
                "heldout_seeds": HELDOUT_SEEDS,
                "contest_eval_seeds_forbidden": sorted(CONTEST_EVAL_SEEDS),
            },
        }

    @app.post("/taskset/tasks")
    async def taskset_tasks(request: Request) -> Any:
        body = await request.json()
        tasks = []
        for task_id in body.get("task_ids") or []:
            seed = _seed_from_task_id(str(task_id))
            if seed in CONTEST_EVAL_SEEDS:
                raise HTTPException(status_code=422, detail=f"contest_eval_seed_forbidden:{seed}")
            split = "heldout" if seed in HELDOUT_SEEDS else "train"
            tasks.append(
                {
                    "task_id": str(task_id),
                    "task_instance_id": f"seed:{seed}",
                    "seed": seed,
                    "split": str(body.get("split") or split),
                    "objective": "episode env reward",
                }
            )
        return {"tasks": tasks, "metadata": {"requested": len(tasks)}}


async def compat_rollout(request: Request, platform: CompatPlatform, spec: TargetSpec) -> Any:
    body = await request.json()
    task = body.get("task") if isinstance(body.get("task"), dict) else {}
    seed = int(task.get("seed") or _seed_from_task_id(str(task.get("task_id") or "0")))
    if seed in CONTEST_EVAL_SEEDS:
        return JSONResponse(
            status_code=422,
            content={"error": "contest_eval_seed_forbidden", "seed": seed},
        )
    if not platform.policy_code:
        return JSONResponse(
            status_code=400,
            content={
                "error": "nanohorizon_missing_policy_code",
                "detail": "PUT submissions/gepa/policy.py before a GEPA rollout",
            },
        )
    candidate = body.get("candidate") if isinstance(body.get("candidate"), dict) else {}
    policy = body.get("policy") if isinstance(body.get("policy"), dict) else {}
    extra = policy.get("config") if isinstance(policy.get("config"), dict) else {}
    policy_config = {
        "model": policy.get("model") or extra.get("model") or "Qwen/Qwen3.5-2B",
        "api": "chat_completions",
        "base_url": policy.get("base_url") or extra.get("base_url") or "http://127.0.0.1:8787/v1",
        "openai_compatible_local": True,
        "enable_thinking": True,
        "thinking_budget": int(extra.get("thinking_budget") or 640),
        "answer_max_tokens": int(extra.get("answer_max_tokens") or 128),
        "max_calls": int(extra.get("max_calls") or 10),
        "max_steps": int(extra.get("max_steps") or 2000),
        "min_actions": int(extra.get("min_actions") or 2),
        "max_actions": int(extra.get("max_actions") or 10),
        "temperature": float(extra.get("temperature") or 1.0),
        "top_p": float(extra.get("top_p") or 0.95),
        "top_k": int(extra.get("top_k") or 20),
        "timeout_seconds": float(extra.get("timeout_seconds") or 180.0),
        "compact_after_tokens": int(extra.get("compact_after_tokens") or 2800),
        "context_token_budget": int(extra.get("context_token_budget") or 3600),
        "system_prompt": str(candidate.get("system_prompt") or SEED_PROMPT),
        "env_name": "craftax",
    }
    digest = hashlib.sha256(json.dumps(policy_config, sort_keys=True).encode("utf-8")).hexdigest()[
        :12
    ]
    config_id = f"gepa_nh_{digest}"
    platform.policy_configs.setdefault(
        config_id,
        PolicyConfig(config_id=config_id, harness="nanohorizon", config=policy_config),
    )
    rollout_id = str(
        body.get("rollout_id")
        or body.get("trace_correlation_id")
        or f"roll_{uuid.uuid4().hex[:12]}"
    )
    if rollout_id not in platform.logs:
        platform.prepare(rollout_id, "poll", "run")
    req = parse_create_rollout(
        {
            "rollout_id": rollout_id,
            "submission_mode": "sync",
            "slot": "stream",
            "telemetry": {"enabled": True, "transport": "poll", "retention": "run"},
            "task_instance_id": f"seed:{seed}",
            "world_ref": spec.world_ref,
            "evaluation_plan_ref": spec.evaluation_plan_ref,
            "policy_ref": {"harness": "nanohorizon", "config": config_id},
        }
    )
    import asyncio

    result = await asyncio.to_thread(platform.start_rollout, req)
    if "error" in result:
        status = int(result.get("status_code") or 400)
        return JSONResponse(status_code=status, content=result)
    reward_record = platform.compute_reward(
        rollout_id=rollout_id,
        evidence=None,
        mode="terminal",
        rescore=False,
        plan_ref=spec.evaluation_plan_ref,
        after_sequence=None,
    )
    reward = reward_record.get("reward")
    events = platform.events_payload(rollout_id, 0, 10_000).get("events", [])
    return {
        **result,
        "status": result.get("status"),
        "success_status": "success" if reward is not None else "failed",
        "reward": reward,
        "reward_info": {"outcome_reward": reward, "metrics": {"env_reward": reward}},
        "trace": {"events": events},
        "events": events,
        "summary": {"seed": seed, "outcome_reward": reward},
    }


def _seed_from_task_id(task_id: str) -> int:
    raw = task_id.rsplit(":", 1)[-1]
    try:
        return int(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid_task_id:{task_id}") from exc
