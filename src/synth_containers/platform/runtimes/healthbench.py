"""Normalized HealthBench rollout with real physician-rubric grading.

Unknown cost remains null. Known token-derived prices are explicitly marked as
estimates and cite the pinned public price table date; they are never presented
as provider-settled invoices.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Any

import httpx

from ...event_log import RolloutEventLog
from ..healthbench_world import load_row, public_observation
from ..local_provider import (
    RESPONSES,
    endpoint_suffix,
    is_local_provider,
    local_endpoint,
    normalize_api_family,
)
from ..state import CompatPlatform, RolloutPin


POLICY_MODEL = "llama-3.1-8b-instant"
GRADER_MODEL = "gpt-4.1-2025-04-14"
SCALED_GRADER_MODEL = "gpt-4.1-mini-2025-04-14"
PRICES = {
    ("groq", POLICY_MODEL): (0.05, 0.08, "groq_public_price_table_2026-08-13"),
    ("openai", "gpt-4.1"): (2.0, 8.0, "openai_public_price_table_2026-08-13"),
    ("openai", GRADER_MODEL): (2.0, 8.0, "openai_public_price_table_2026-08-13"),
    ("openai", "gpt-4.1-mini"): (0.4, 1.6, "openai_public_price_table_2026-08-13"),
    ("openai", SCALED_GRADER_MODEL): (0.4, 1.6, "openai_public_price_table_2026-08-13"),
}


def model_roles() -> dict[str, dict[str, Any]]:
    """Describe the two independent paid model roles without exposing credentials."""

    grader_model = os.environ.get("HEALTHBENCH_GRADER_MODEL", GRADER_MODEL)
    policy_key_env = os.environ.get("HEALTHBENCH_POLICY_API_KEY_ENV", "OPENAI_API_KEY")
    grader_key_env = os.environ.get("HEALTHBENCH_GRADER_API_KEY_ENV", "OPENAI_API_KEY")
    return {
        "policy": {
            "purpose": "generate_candidate_response",
            "configuration_authority": "policy_ref",
            "api_key_env": policy_key_env,
            "credential_present": bool(os.environ.get(policy_key_env, "").strip()),
            "usage_lane": "policy",
            "required": True,
        },
        "scorer": {
            "purpose": "score_response_against_physician_rubrics",
            "provider": "openai",
            "model": grader_model,
            "api_key_env": grader_key_env,
            "credential_present": bool(os.environ.get(grader_key_env, "").strip()),
            "base_url": os.environ.get("HEALTHBENCH_GRADER_BASE_URL", "https://api.openai.com/v1"),
            "evaluation_plan_ref": (
                "healthbench_eval.v1"
                if grader_model == GRADER_MODEL
                else "healthbench_scaled_grader.v1"
            ),
            "canonical": grader_model == GRADER_MODEL,
            "usage_lane": "grader",
            "call_pattern": "one_call_per_rubric_item",
            "required": True,
        },
    }


@lru_cache(maxsize=8)
def _client(base_url: str) -> httpx.Client:
    """Reuse provider connections and retry only failures before a response exists.

    HTTPTransport's retries cover connect errors/timeouts, not reads or HTTP
    status responses. That distinction matters for paid calls: retrying an
    ambiguous read timeout could bill twice, while reconnecting before a
    response connection exists is safe and fixes transient TLS churn.
    """

    max_connections = max(
        1, min(64, int(os.environ.get("HEALTHBENCH_PROVIDER_MAX_CONNECTIONS", "30")))
    )
    return httpx.Client(
        base_url=base_url,
        transport=httpx.HTTPTransport(retries=2),
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        ),
    )


class HealthBenchRuntime:
    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        platform.step_calls += 1
        seed = int(pin.seed or 0)
        row = load_row(seed)
        log.append("env.episode.opened", {"seed": seed, "world_ref": pin.world_ref})
        if row is None:
            self._fail(pin, log, "unknown_task_instance")
            return
        observation = public_observation(row, seed)
        log.append("observation", observation)
        try:
            config_id = str(pin.policy_ref.get("config") or "groq_llama31_8b")
            policy = platform.policy_configs.get(config_id)
            config = dict(policy.config) if policy is not None else {}
            provider = str(config.get("provider") or "groq")
            model = str(config.get("model") or POLICY_MODEL)
            log.append("span.policy.opened", {"provider": provider, "model": model})
            messages = list(row.get("prompt") or [])
            system_prompt = config.get("system_prompt")
            if isinstance(system_prompt, str) and system_prompt.strip():
                messages = [{"role": "system", "content": system_prompt.strip()}, *messages]
            completion = _chat(config, messages)
            action_payload = {
                "role": "assistant",
                "content": completion["text"],
                "usage": completion["usage"],
            }
            if isinstance(completion.get("training_action"), dict):
                action_payload["training_action"] = completion["training_action"]
            log.append("action", action_payload)
            log.append("span.policy.closed", {"status": "completed", "usage": completion["usage"]})
            grader_model = os.environ.get("HEALTHBENCH_GRADER_MODEL", GRADER_MODEL)
            log.append(
                "span.evaluator.opened",
                {
                    "model": grader_model,
                    "canonical": grader_model == GRADER_MODEL,
                    "evaluation_plan_ref": (
                        "healthbench_eval.v1"
                        if grader_model == GRADER_MODEL
                        else "healthbench_scaled_grader.v1"
                    ),
                },
            )
            grades, grader_usage = _grade(row, completion["text"], log, messages=messages)
            possible = sum(
                float(item.get("points") or 0)
                for item in grades
                if float(item.get("points") or 0) > 0
            )
            if possible <= 0:
                raise RuntimeError("healthbench_positive_rubric_missing")
            achieved = sum(
                float(item.get("points") or 0) for item in grades if item["criteria_met"]
            )
            reward = achieved / possible
            log.append("span.evaluator.closed", {"status": "completed", "usage": grader_usage})
            combined = _combine_usage(completion["usage"], grader_usage)
            log.append(
                "reward_signal",
                {
                    "value": reward,
                    "authority": "healthbench_physician_rubric_grader",
                    "kind": "healthbench_overall_score",
                    "grader_model": grader_model,
                    "canonical_healthbench_grader": grader_model == GRADER_MODEL,
                },
            )
            pin.reward_signals = [reward]
            pin.usage = combined
            pin.status = "completed"
            pin.terminal = True
            log.append("env.episode.closed", {"status": "completed", "seed": seed})
            log.append("status", {"status": "completed"})
        except Exception as exc:
            log.append("span.evaluator.closed", {"status": "failed", "error_code": _safe_code(exc)})
            self._fail(pin, log, "healthbench_rollout_failed")
            return
        self._seal(log)

    def _fail(self, pin: RolloutPin, log: RolloutEventLog, reason: str) -> None:
        pin.reward_signals = [None]
        pin.usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }
        pin.status = "failed"
        pin.terminal = True
        log.append("env.episode.closed", {"status": "failed", "reason": reason})
        log.append("status", {"status": "failed", "reason": reason})
        self._seal(log)

    @staticmethod
    def _seal(log: RolloutEventLog) -> None:
        high_water = log.high_water
        log.append("capture.high_water", {"high_water": high_water})
        log.append("capture.closed", {"high_water": high_water})
        log.mark_closed()


def _chat(config: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sample one completion. Hosted providers stay unrestricted, as before.

    `synth_mlx_rl` is the one provider whose endpoint is checked here, because
    it is the one whose base URL is not a known origin: it goes through the same
    shared admission the Banking77 and ReAct call sites use, on either
    InferenceApiFamily member.
    """
    inference_target = config.get("inference_target")
    if isinstance(inference_target, dict):
        from ...training_rollout import (
            ROLLOUT_ACTION_SCHEMA_VERSION,
            HostedSamplerClient,
            SamplerEndpoint,
        )

        endpoint = SamplerEndpoint(
            url=str(inference_target.get("provider_endpoint_id") or ""),
            bearer_token=str(inference_target.get("auth_bearer") or ""),
            connection_mode=str(inference_target.get("connection_mode") or "keep_alive"),
        )
        policy_version = str(config.get("policy_version") or inference_target.get("checkpoint_id"))
        with HostedSamplerClient(endpoint) as client:
            sampled = client.sample(
                {
                    "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
                    "job_id": config.get("job_id"),
                    "attempt_id": config.get("attempt_id"),
                    "rollout_id": config.get("rollout_id"),
                    "run_id": config.get("job_id"),
                    "checkpoint_id": policy_version,
                    "messages": messages,
                    "max_tokens": int(config.get("max_tokens") or 1536),
                    "temperature": float(config.get("temperature") or 0.2),
                    "policy_version": policy_version,
                },
                idempotency_key=(
                    f"{config.get('rollout_id') or 'healthbench'}:"
                    f"{policy_version}:{canonical_messages_digest(messages)}"
                ),
            )
        return {
            "text": sampled.text,
            "usage": dict(sampled.usage),
            "training_action": {
                "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
                "policy_version": policy_version,
                "prompt_token_ids": list(sampled.prompt_token_ids),
                "token_ids": list(sampled.token_ids),
                "log_probs": list(sampled.log_probs),
            },
        }
    provider = str(config.get("provider") or "groq").lower()
    model = str(config.get("model") or POLICY_MODEL)
    api_family = normalize_api_family(config.get("api_family"))
    base = str(
        config.get("base_url")
        or ("https://api.groq.com/openai/v1" if provider == "groq" else "https://api.openai.com/v1")
    ).rstrip("/")
    key_env = str(
        config.get("api_key_env") or ("GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY")
    )
    api_key = os.environ.get(key_env, "")
    if is_local_provider(provider):
        # A loopback proxy issues no bearer of its own; the URL check is the
        # admission, so a missing key is not a refusal here.
        target = local_endpoint(base, api_family=api_family)
    else:
        if not api_key:
            raise RuntimeError(f"{provider}_api_key_missing")
        target = endpoint_suffix(api_family)
    if api_family == RESPONSES:
        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "temperature": float(config.get("temperature", 0.2)),
            "max_output_tokens": int(config.get("max_tokens", 1536)),
        }
    else:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(config.get("temperature", 0.2)),
            "max_completion_tokens": int(config.get("max_tokens", 1536)),
        }
    response = _client(base).post(
        target,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        json=payload,
        timeout=float(config.get("timeout_seconds", 90)),
    )
    response.raise_for_status()
    body = response.json()
    if api_family == RESPONSES:
        text = _responses_text(body)
    else:
        text = str((body.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    if not text.strip():
        raise RuntimeError("empty_policy_completion")
    return {"text": text, "usage": _usage(provider, model, body.get("usage"))}


def canonical_messages_digest(messages: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _responses_text(body: dict[str, Any]) -> str:
    text = body.get("output_text")
    if isinstance(text, str):
        return text
    fragments: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                fragments.append(str(part.get("text") or ""))
    return "".join(fragments)


def _grade(
    row: dict[str, Any],
    completion: str,
    log: RolloutEventLog,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    roles = model_roles()
    scorer = roles["scorer"]
    conversation_messages = list(messages or row.get("prompt") or []) + [
        {"role": "assistant", "content": completion}
    ]
    conversation = "\n\n".join(
        f"{item.get('role')}: {item.get('content')}" for item in conversation_messages
    )
    usages = []
    grades = []
    for index, rubric in enumerate(row.get("rubrics") or []):
        prompt = (
            "Judge only the final assistant turn against this rubric item. Return exactly JSON "
            "with string explanation and boolean criteria_met. A negative-point item's "
            "criteria_met is true only when that undesirable behavior occurred.\n\n"
            f"Conversation:\n{conversation}\n\nRubric:\n[{rubric.get('points')}] {rubric.get('criterion')}"
        )
        result = _chat(
            {
                "provider": scorer["provider"],
                "model": scorer["model"],
                "base_url": scorer["base_url"],
                "api_key_env": scorer["api_key_env"],
                "temperature": 0,
                "max_tokens": 512,
            },
            [{"role": "user", "content": prompt}],
        )
        parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", result["text"].strip()))
        if not isinstance(parsed.get("criteria_met"), bool):
            raise RuntimeError("healthbench_grader_invalid_json")
        grade = {**rubric, **parsed, "index": index}
        grades.append(grade)
        usages.append(result["usage"])
        log.append("rubric.grade", grade)
    return grades, _sum_usage(usages)


def _usage(provider: str, model: str, raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    prompt = raw.get("prompt_tokens", raw.get("input_tokens"))
    completion = raw.get("completion_tokens", raw.get("output_tokens"))
    prompt = int(prompt) if isinstance(prompt, (int, float)) else None
    completion = int(completion) if isinstance(completion, (int, float)) else None
    price = PRICES.get((provider.lower(), model.lower()))
    cost = None
    if price is not None and prompt is not None and completion is not None:
        cost = (prompt * price[0] + completion * price[1]) / 1_000_000
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion
        if prompt is not None and completion is not None
        else None,
        "cost_usd": cost,
        "cost_kind": "estimated_from_tokens" if cost is not None else None,
        "cost_source": price[2] if cost is not None and price is not None else None,
    }


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def total(key: str) -> int | None:
        return (
            sum(int(row[key]) for row in rows)
            if rows and all(row.get(key) is not None for row in rows)
            else None
        )

    costs = [row.get("cost_usd") for row in rows]
    known_costs = [float(value) for value in costs if value is not None]
    cost = sum(known_costs) if costs and len(known_costs) == len(costs) else None
    return {
        "prompt_tokens": total("prompt_tokens"),
        "completion_tokens": total("completion_tokens"),
        "total_tokens": total("total_tokens"),
        "cost_usd": cost,
        "cost_kind": "estimated_from_tokens" if cost is not None else None,
        "cost_source": "openai_public_price_table_2026-08-13" if cost is not None else None,
        "calls": len(rows),
    }


def _combine_usage(policy: dict[str, Any], grader: dict[str, Any]) -> dict[str, Any]:
    combined = _sum_usage([policy, grader])
    combined["calls"] = 1 + int(grader.get("calls") or 0)
    sources = sorted(
        {str(item.get("cost_source")) for item in (policy, grader) if item.get("cost_source")}
    )
    combined["cost_source"] = sources[0] if len(sources) == 1 else "mixed_public_price_tables"
    combined["cost_sources"] = sources
    combined["policy"] = policy
    combined["grader"] = grader
    return combined


def _safe_code(exc: BaseException) -> str:
    text = str(exc).strip()
    return text if re.fullmatch(r"[a-z0-9_.:-]{1,64}", text) else type(exc).__name__
