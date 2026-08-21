"""GSM8K one-turn solve content. Exact match on the parsed numeric answer.

Harnesses:
- ``dataset_gold`` — the env's own reference solution as the action (container
  authored train traces)
- ``solve`` — live policy from ``policy_ref.config``; ``forced_completion`` is
  test-only. A config may name a provider (``synth_mlx_rl`` over loopback, or
  a hosted one) or carry the typed training boundary's ``inference_target``
  (``/training/rollouts``), which is sampled through the hosted sampler
  contract exactly as Banking77 does.

Every ``action`` carries ``parse_mode`` (``exact`` / ``trailing_number`` /
``unparsed``) next to ``parse_status``: a trial that scored through the
last-number fallback is counted, but it is not format compliance, and a reader
must be able to tell the two apart per trial rather than from a summary.

Three honesty rules, all with a Banking77 precedent:

- the reference answer never enters the public observation;
- a missing completion leaves the reward ``None`` (``_close_missing`` /
  ``reward_signals = [None]``), and ``omit_reward`` does the same;
- a completion the policy *did* produce but that states no parseable number
  scores ``0.0``: the policy attempted the task and failed it. ``parse_status``
  and the raw text are recorded separately so a format failure is still
  distinguishable from a wrong number, but it is counted, not dropped.
  Scoring it ``None`` would be worse than wrong — the eval layer excludes
  missing metrics from the denominator, so a model that rambles instead of
  answering would have its bad trials deleted and its accuracy computed over
  only the subset where it happened to emit a number.

See: workshop/docs/aug_12_update.md (content, not a fold; missing ≠ 0).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ...event_log import RolloutEventLog
from ..gsm8k_world import (
    SOLVE_SYSTEM,
    ParsedAnswer,
    load_row,
    parse_answer,
    public_observation,
    split_from_world_ref,
    user_prompt,
)
from ..local_provider import (
    CHAT_COMPLETIONS,
    RESPONSES,
    is_local_provider,
    local_endpoint,
    normalize_api_family,
    token_capture_ref,
)
from ..state import CompatPlatform, RolloutPin
from ...proxying import WORKSHOP_API_KEY_SENTINEL, workload_proxy_base
from .banking77 import _error_code


_EMPTY_USAGE = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
}


class Gsm8kRuntime:
    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        platform.step_calls += 1
        split = split_from_world_ref(pin.world_ref)
        seed = int(pin.seed or 0)
        row = load_row(split, seed)
        log.append(
            "env.episode.opened",
            {"seed": seed, "split": split, "world_ref": pin.world_ref},
        )
        if row is None:
            self._close_missing(
                pin,
                log,
                reason="unknown_task_instance",
                detail=f"seed {seed} is outside split {split}",
            )
            return

        observation = public_observation(row, seed=seed, split=split)
        log.append("observation", observation)

        harness = str(pin.policy_ref.get("harness") or "").strip()
        if not harness:
            self._close_missing(pin, log, reason="policy_ref_required", detail="harness missing")
            return

        log.append(
            "span.policy.opened",
            {"harness": harness, "config": pin.policy_ref.get("config")},
        )
        try:
            completion, usage, token_capture = self._act(platform, pin, harness, observation)
            training_action = usage.pop("_training_action", None)
            if isinstance(completion, str) and not completion.strip():
                completion = None
        except Exception as exc:
            # The policy raises secret-free codes (`synth_mlx_rl_endpoint_refused`,
            # …). Dropping them leaves the stream saying only "RuntimeError",
            # which a reader cannot act on.
            log.append(
                "span.policy.closed",
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_code": _error_code(exc),
                },
            )
            self._close_missing(
                pin,
                log,
                reason="policy_error",
                detail=type(exc).__name__,
                usage=_EMPTY_USAGE,
            )
            return

        parsed = parse_answer(completion) if completion is not None else ParsedAnswer(None, "absent", "")
        action_payload: dict[str, Any] = {
            # The parsed value and the text the policy actually produced are
            # separate fields on purpose: one is a claim, the other is
            # evidence, and a grader that sees only the first cannot tell an
            # unparseable answer from a wrong one.
            "answer": parsed.value,
            "text": completion,
            "parse_status": "parsed" if parsed.parsed else "unparsed",
            "parse_source": parsed.source,
            # Per trial, so "0% parse failures" can never be read as "100%
            # followed the format": only `exact` trials did.
            "parse_mode": parsed.parse_mode,
            "format_compliant": parsed.format_compliant,
        }
        if isinstance(training_action, dict):
            action_payload["training_action"] = training_action
        log.append("action", action_payload)
        if token_capture is not None:
            # The join key. The proxy owns the authoritative token ids and
            # rollout logprobs; this container owns the reward. Without this
            # event nothing connects the two, and an on-policy trace is
            # unusable no matter how complete each half is on its own.
            log.append("token_capture", dict(token_capture))
        log.append(
            "span.policy.closed",
            {
                "status": "completed" if completion else "empty",
                "parse_status": "parsed" if parsed.parsed else "unparsed",
                "parse_source": parsed.source,
                "parse_mode": parsed.parse_mode,
            },
        )

        reference = row.answer
        if pin.omit_reward or completion is None or not reference:
            # Genuinely absent signal: the policy produced nothing at all, the
            # caller suppressed the reward, or the env's own reference will not
            # parse (an env defect, not a policy failure). None of these is a
            # zero. `_close_missing` is not used here because the episode itself
            # completed — only the signal is absent.
            value: float | None = None
        elif not parsed.parsed:
            # The policy answered, but stated no parseable number. That is a
            # failed attempt, not an absent signal, and it must stay in the
            # denominator — see the module docstring.
            value = 0.0
        else:
            value = 1.0 if parsed.value == reference else 0.0
        log.append(
            "reward_signal",
            {"value": value, "authority": "environment", "kind": "exact_match_numeric"},
        )
        pin.reward_signals = [value]
        pin.usage = dict(usage)
        pin.status = "completed"
        pin.terminal = True
        log.append("env.episode.closed", {"status": "completed", "split": split, "seed": seed})
        log.append("status", {"status": "completed"})
        self._seal_capture(log)

    def _act(
        self,
        platform: CompatPlatform,
        pin: RolloutPin,
        harness: str,
        observation: dict[str, Any],
    ) -> tuple[str | None, dict[str, Any], dict[str, Any] | None]:
        """Returns (completion, usage, token_capture_reference).

        The third element is the join key. It is None for a gold or forced
        action and for a hosted provider: those have no proxy record to join to,
        which is different from having one and losing it.
        """
        if harness == "dataset_gold":
            row = load_row(str(observation["split"]), int(observation["seed"]))
            if row is None:
                return None, dict(_EMPTY_USAGE), None
            return row.answer_text, dict(_EMPTY_USAGE), None

        config_id = str(pin.policy_ref.get("config") or "").strip()
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        forced = config.get("forced_completion")
        if isinstance(forced, str) and forced.strip():
            return forced, dict(_EMPTY_USAGE), None

        sampler_target = _training_sampler_target(config)
        if sampler_target is not None:
            text, usage = _sample_training_sampler(sampler_target, observation, config)
            return text, usage, None

        provider = str(config.get("provider") or "").strip().lower()
        if provider:
            if is_local_provider(provider) and normalize_api_family(config.get("api_family")) == RESPONSES:
                return _sample_responses(observation, config)
            return _sample_chat_completion(observation, config)

        if harness != "solve":
            raise ValueError(f"unknown_gsm8k_harness:{harness}")
        return None, dict(_EMPTY_USAGE), None

    def _close_missing(
        self,
        pin: RolloutPin,
        log: RolloutEventLog,
        *,
        reason: str,
        detail: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        pin.reward_signals = [None]
        pin.usage = dict(usage or _EMPTY_USAGE)
        pin.status = "failed"
        pin.terminal = True
        log.append("env.episode.closed", {"status": "failed", "reason": reason})
        log.append("status", {"status": "failed", "reason": reason, "detail": detail})
        self._seal_capture(log)

    def _seal_capture(self, log: RolloutEventLog) -> None:
        evidence_high_water = log.high_water
        log.append("capture.high_water", {"high_water": evidence_high_water})
        log.append("capture.closed", {"high_water": evidence_high_water})
        log.mark_closed()


_HOSTED_BASES = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

#: GSM8K completions reason before they answer; a 32-token clamp (Banking77's)
#: would cut every trial before the `#### N` line. The boundary already bounds
#: the request at 16_384; this is the container's own ceiling under it.
_SAMPLER_MAX_TOKENS = 4096


def _training_sampler_target(config: dict[str, Any]) -> dict[str, Any] | None:
    """The typed training boundary's target, or None for every other config.

    Only a config the boundary itself stamped (`training_sampler_endpoint` is
    True) is sampled this way. The legacy loopback/allowlisted checkpoint path
    Banking77 still carries is deliberately not admitted here: GSM8K has no
    callers on it and a second admission path is a second thing to audit.
    """
    if config.get("training_sampler_endpoint") is not True:
        return None
    target = config.get("inference_target")
    if not isinstance(target, dict):
        return None
    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        return None
    return dict(target)


def _sample_training_sampler(
    target: dict[str, Any],
    observation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """One sample through the hosted sampler contract (``/v1/training/sample``).

    Same shape as Banking77's `_sample_remote_checkpoint`: the boundary has
    already required an authenticated sampler (HTTPS, or loopback when the
    host explicitly allows it), so this only builds the request and relays the
    token record the contract obliges the container to carry.
    """
    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    if str(target.get("provider") or "").strip().lower() != "tinker":
        raise RuntimeError("remote_checkpoint_provider_unsupported")
    auth_bearer = str(target.get("auth_bearer") or "").strip()
    if not auth_bearer or "\r" in auth_bearer or "\n" in auth_bearer:
        raise RuntimeError("remote_checkpoint_auth_missing")
    run_id = str(target.get("run_id") or "").strip()
    checkpoint_id = str(target.get("checkpoint_id") or "").strip()
    if not run_id or not checkpoint_id:
        raise RuntimeError("remote_checkpoint_identity_missing")
    max_tokens = min(max(int(config.get("max_tokens") or 512), 1), _SAMPLER_MAX_TOKENS)
    messages = [
        {"role": "system", "content": str(observation.get("system") or SOLVE_SYSTEM)},
        {"role": "user", "content": user_prompt(str(observation["question"]))},
    ]
    timeout = min(max(float(config.get("remote_timeout_seconds") or 120.0), 1.0), 600.0)
    from ...training_rollout import (
        ROLLOUT_ACTION_SCHEMA_VERSION,
        HostedSamplerClient,
        SamplerEndpoint,
        canonical_sha256,
    )
    from ..app import allow_loopback_sampler

    message_digest = canonical_sha256({"messages": messages})
    with HostedSamplerClient(
        SamplerEndpoint(
            endpoint,
            auth_bearer,
            str(target.get("connection_mode") or "keep_alive"),
        ),
        timeout_seconds=timeout,
        allow_loopback_http=allow_loopback_sampler(),
    ) as client:
        sampled = client.sample(
            {
                "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
                "job_id": config.get("job_id") or run_id,
                "attempt_id": config.get("attempt_id"),
                "rollout_id": config.get("rollout_id"),
                "run_id": run_id,
                "checkpoint_id": checkpoint_id,
                "policy_version": config.get("policy_version") or checkpoint_id,
                "messages": messages,
                "max_tokens": max_tokens,
                # Honoured, not hardcoded: a group sampled at 0.0 has no reward
                # variance and no optimizer step (the Banking77 canary).
                "temperature": float(config.get("temperature") or 0.0),
            },
            idempotency_key=(
                f"{config.get('rollout_id') or run_id}:{checkpoint_id}:{message_digest}"
            ),
        )
    usage = _usage(
        dict(sampled.usage), input_key="prompt_tokens", output_key="completion_tokens"
    )
    usage["_training_action"] = {
        "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
        "policy_version": config.get("policy_version") or checkpoint_id,
        "prompt_token_ids": list(sampled.prompt_token_ids),
        "token_ids": list(sampled.token_ids),
        "log_probs": list(sampled.log_probs),
    }
    return (sampled.text or None), usage


def _endpoint(config: dict[str, Any], *, api_family: str) -> tuple[str, str]:
    """(endpoint, api_key) for the configured provider. Fails closed."""
    provider = str(config.get("provider") or "openai").strip().lower()
    if is_local_provider(provider):
        # The local proxy carries no bearer of its own; admission is the URL
        # check, not a key. `api_key_env` stays honored so a proxy that does
        # require one is not forced to run open.
        endpoint = local_endpoint(config.get("base_url"), api_family=api_family)
        key_env = str(config.get("api_key_env") or "").strip()
        return endpoint, (os.environ.get(key_env, "").strip() if key_env else "")
    if (proxied_base := workload_proxy_base(config)) is not None:
        key_env = str(config.get("api_key_env") or "OPENAI_API_KEY").strip()
        api_key = os.environ.get(key_env, "").strip() or WORKSHOP_API_KEY_SENTINEL
        suffix = "/responses" if api_family == RESPONSES else "/chat/completions"
        return f"{proxied_base}{suffix}", api_key
    if provider not in _HOSTED_BASES:
        raise RuntimeError("gsm8k_provider_unsupported")
    base_url = str(config.get("base_url") or _HOSTED_BASES[provider]).rstrip("/")
    if base_url != _HOSTED_BASES[provider]:
        raise RuntimeError("gsm8k_chat_endpoint_refused")
    key_env = str(config.get("api_key_env") or "OPENAI_API_KEY").strip()
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise RuntimeError("openai_api_key_missing")
    suffix = "/responses" if api_family == RESPONSES else "/chat/completions"
    return f"{base_url}{suffix}", api_key


def _post(endpoint: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        if "\r" in api_key or "\n" in api_key:
            raise RuntimeError("gsm8k_api_key_invalid")
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(4_194_305)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"gsm8k_http_{exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("gsm8k_transport_error") from exc
    if len(raw) > 4_194_304:
        raise RuntimeError("gsm8k_response_too_large")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("gsm8k_response_invalid") from exc
    if not isinstance(body, dict):
        raise RuntimeError("gsm8k_response_invalid")
    return body


def _sample_chat_completion(
    observation: dict[str, Any], config: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    model = str(config.get("model") or "").strip()
    if not model:
        raise RuntimeError("gsm8k_model_missing")
    endpoint, api_key = _endpoint(config, api_family="chat_completions")
    body = _post(
        endpoint,
        api_key,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": str(observation.get("system") or SOLVE_SYSTEM)},
                {"role": "user", "content": str(observation.get("prompt") or "")},
            ],
            "temperature": float(config.get("temperature", 0)),
            "max_completion_tokens": int(config.get("max_tokens", 512)),
        },
        float(config.get("timeout_seconds", 120)),
    )
    text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return (
        (text or None),
        _usage(body.get("usage"), input_key="prompt_tokens", output_key="completion_tokens"),
        token_capture_ref(body, api_family=CHAT_COMPLETIONS),
    )


def _sample_responses(
    observation: dict[str, Any], config: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    model = str(config.get("model") or "").strip()
    if not model:
        raise RuntimeError("gsm8k_model_missing")
    endpoint, api_key = _endpoint(config, api_family=RESPONSES)
    body = _post(
        endpoint,
        api_key,
        {
            "model": model,
            "instructions": SOLVE_SYSTEM,
            "input": user_prompt(str(observation["question"])),
            "max_output_tokens": int(config.get("max_tokens", 512)),
            "temperature": float(config.get("temperature", 0)),
        },
        float(config.get("timeout_seconds", 120)),
    )
    text = body.get("output_text")
    if not isinstance(text, str):
        fragments: list[str] = []
        for item in body.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    fragments.append(str(part.get("text") or ""))
        text = "".join(fragments)
    return (
        (text or None),
        _usage(body.get("usage"), input_key="input_tokens", output_key="output_tokens"),
        token_capture_ref(body, api_family=RESPONSES),
    )


def _usage(value: Any, *, input_key: str, output_key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return dict(_EMPTY_USAGE)
    mapped = {
        "prompt_tokens": value.get(input_key),
        "completion_tokens": value.get(output_key),
        "total_tokens": value.get("total_tokens"),
    }
    return {
        key: item
        if item is None or (isinstance(item, int) and not isinstance(item, bool) and item >= 0)
        else None
        for key, item in mapped.items()
    }
