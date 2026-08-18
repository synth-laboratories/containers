"""GSM8K one-turn solve content. Exact match on the parsed numeric answer.

Harnesses:
- ``dataset_gold`` — the env's own reference solution as the action (container
  authored train traces)
- ``solve`` — live policy from ``policy_ref.config``; ``forced_completion`` is
  test-only

Three honesty rules, all with a Banking77 precedent:

- the reference answer never enters the public observation;
- a missing completion leaves the reward ``None`` (``_close_missing`` /
  ``reward_signals = [None]``), and ``omit_reward`` does the same;
- an *unparseable* completion is not a wrong answer. It keeps the reward ``None``
  and records ``parse_status`` and the raw text separately, so "the policy never
  stated a number" cannot be read later as "the policy answered zero".

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
    RESPONSES,
    is_local_provider,
    local_endpoint,
    normalize_api_family,
)
from ..state import CompatPlatform, RolloutPin
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
            completion, usage = self._act(platform, pin, harness, observation)
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
        log.append(
            "action",
            {
                # The parsed value and the text the policy actually produced are
                # separate fields on purpose: one is a claim, the other is
                # evidence, and a grader that sees only the first cannot tell an
                # unparseable answer from a wrong one.
                "answer": parsed.value,
                "text": completion,
                "parse_status": "parsed" if parsed.parsed else "unparsed",
                "parse_source": parsed.source,
            },
        )
        log.append(
            "span.policy.closed",
            {
                "status": "completed" if completion else "empty",
                "parse_status": "parsed" if parsed.parsed else "unparsed",
                "parse_source": parsed.source,
            },
        )

        reference = row.answer
        if pin.omit_reward or completion is None or not parsed.parsed or not reference:
            # Absent, unparseable, or an env row whose own reference will not
            # parse: none of these is a zero. `_close_missing` is not used here
            # because the episode itself completed — only the signal is absent.
            value: float | None = None
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
    ) -> tuple[str | None, dict[str, Any]]:
        if harness == "dataset_gold":
            row = load_row(str(observation["split"]), int(observation["seed"]))
            if row is None:
                return None, dict(_EMPTY_USAGE)
            return row.answer_text, dict(_EMPTY_USAGE)

        config_id = str(pin.policy_ref.get("config") or "").strip()
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        forced = config.get("forced_completion")
        if isinstance(forced, str) and forced.strip():
            return forced, dict(_EMPTY_USAGE)

        provider = str(config.get("provider") or "").strip().lower()
        if provider:
            if is_local_provider(provider) and normalize_api_family(config.get("api_family")) == RESPONSES:
                return _sample_responses(observation, config)
            return _sample_chat_completion(observation, config)

        if harness != "solve":
            raise ValueError(f"unknown_gsm8k_harness:{harness}")
        return None, dict(_EMPTY_USAGE)

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
    return (text or None), _usage(body.get("usage"), input_key="prompt_tokens", output_key="completion_tokens")


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
    return (text or None), _usage(body.get("usage"), input_key="input_tokens", output_key="output_tokens")


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
