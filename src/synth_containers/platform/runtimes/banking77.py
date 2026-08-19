"""Banking77 classify content. One-shot env accuracy. Not a Harbor/OpenEnv wrap.

Harnesses:
- ``dataset_gold`` — env gold as the action (container-authored train traces)
- ``classify`` — live policy from ``policy_ref.config``; Tinker sampler if
  ``inference_target`` names one; ``forced_label`` is test-only

Gold never appears in the public observation. Missing prediction stays null
on ``/reward`` (never coerced to 0). ``omit_reward`` is the C1 missing-signal path.

See: workshop/docs/aug_12_update.md (content, not a fold; missing ≠ 0).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from typing import Any, Callable

from ...event_log import RolloutEventLog
from ..banking77_world import (
    CLASSIFY_SYSTEM,
    load_row,
    normalize_label,
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
    validate_local_endpoint,
)
from ..state import CompatPlatform, RolloutPin


_EMPTY_USAGE = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
}


class Banking77Runtime:
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
            predicted, usage = self._act(platform, pin, harness, observation)
            training_action = usage.pop("_training_action", None)
            if isinstance(predicted, str) and not predicted.strip():
                predicted = None
        except Exception as exc:
            # The policy raises secret-free codes (`tinker_sdk_missing`, …).
            # Dropping them left the stream saying only "RuntimeError", which
            # is a failure a reader cannot act on.
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

        action_payload = {"label": predicted, "text": predicted}
        if isinstance(training_action, dict):
            action_payload["training_action"] = training_action
        log.append("action", action_payload)
        log.append("span.policy.closed", {"status": "completed" if predicted else "empty"})

        if pin.omit_reward or predicted is None:
            value: float | None = None
        else:
            value = (
                1.0
                if normalize_label(predicted) == normalize_label(row.label)
                and normalize_label(row.label)
                else 0.0
            )
        log.append(
            "reward_signal",
            {"value": value, "authority": "environment", "kind": "classification_accuracy"},
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
            return row.label, dict(_EMPTY_USAGE)

        config_id = str(pin.policy_ref.get("config") or "").strip()
        policy = platform.policy_configs.get(config_id)
        config = dict(policy.config) if policy is not None else {}
        forced = config.get("forced_label")
        if isinstance(forced, str) and forced.strip():
            return forced.strip(), dict(_EMPTY_USAGE)

        provider = str(config.get("provider") or "").strip()
        if is_local_provider(provider) and normalize_api_family(config.get("api_family")) == RESPONSES:
            # `synth_mlx_rl` is admitted on both InferenceApiFamily members. The
            # family decides the route, so a config cannot name one and be
            # sampled on the other.
            endpoint = local_endpoint(
                config.get("responses_base_url") or config.get("base_url"),
                api_family=RESPONSES,
            )
            return _sample_responses(
                endpoint,
                observation,
                config,
                validate=_validate_local_responses_endpoint,
            )
        if provider:
            return _sample_chat_completion(observation, config)

        responses_endpoint = _responses_endpoint(config)
        if responses_endpoint:
            return _sample_responses(responses_endpoint, observation, config)

        endpoint = _tinker_endpoint(config)
        if endpoint:
            return _sample_tinker(endpoint, observation, config)

        remote_target = _remote_checkpoint_target(config)
        if remote_target is not None:
            return _sample_remote_checkpoint(remote_target, observation, config)

        if harness != "classify":
            raise ValueError(f"unknown_banking77_harness:{harness}")
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


def _sample_chat_completion(
    observation: dict[str, Any], config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    provider = str(config.get("provider") or "openai").strip().lower()
    model = str(config.get("model") or "").strip()
    allowed_bases = {
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
    }
    if is_local_provider(provider):
        # The local MLX proxy is not a hosted origin with a fixed base URL, so
        # the shared validator decides admission instead of an equality check:
        # loopback (and the Docker host alias) over http, or an origin named in
        # SYNTH_MLX_RL_ALLOWED_ENDPOINTS. Everything else is refused before any
        # socket is opened.
        endpoint = local_endpoint(config.get("base_url"), api_family=CHAT_COMPLETIONS)
        key_env = str(config.get("api_key_env") or "").strip()
        api_key = os.environ.get(key_env, "").strip() if key_env else ""
        if not model:
            raise RuntimeError("banking77_model_missing")
    else:
        base_url = str(config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        key_env = str(config.get("api_key_env") or "OPENAI_API_KEY").strip()
        api_key = os.environ.get(key_env, "").strip()
        if provider not in allowed_bases:
            raise RuntimeError("banking77_provider_unsupported")
        if not model:
            raise RuntimeError("banking77_model_missing")
        if not api_key:
            raise RuntimeError("openai_api_key_missing")
        if base_url != allowed_bases[provider]:
            raise RuntimeError("banking77_chat_endpoint_refused")
        endpoint = f"{base_url}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": str(observation.get("system") or CLASSIFY_SYSTEM)},
                {"role": "user", "content": str(observation.get("prompt") or "")},
            ],
            "temperature": float(config.get("temperature", 0)),
            "max_completion_tokens": int(config.get("max_tokens", 32)),
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=float(config.get("timeout_seconds", 90))
        ) as response:
            body = json.loads(response.read(1_000_000))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"openai_http_{exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("openai_transport_error") from exc
    text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    prediction = normalize_label(text)
    if not prediction:
        raise RuntimeError("empty_policy_completion")
    raw_usage = body.get("usage") or {}
    return prediction, {
        "prompt_tokens": raw_usage.get("prompt_tokens"),
        "completion_tokens": raw_usage.get("completion_tokens"),
        "total_tokens": raw_usage.get("total_tokens"),
    }


def _error_code(exc: BaseException) -> str | None:
    """Secret-free failure code. The classify policy raises fixed identifiers
    (`tinker_sdk_missing`, `tinker_base_model_missing`, …); anything with
    whitespace or punctuation is provider prose and is not forwarded."""
    # Provider SDK exception prose can contain request ids, URLs, or other
    # operational details. Prefer a stable, secret-free class/status code and
    # only pass through our own deliberately terse snake-case sentinels.
    if exc.__class__.__module__.split(".", 1)[0] == "tinker":
        if "TINKER_API_KEY" in str(exc) and "must be set" in str(exc):
            return "tinker_api_key_missing"
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return f"tinker_{exc.__class__.__name__.lower()}_{status}"
        return f"tinker_{exc.__class__.__name__.lower()}"
    text = str(exc).strip()
    if not text or len(text) > 64:
        return None
    return text if re.fullmatch(r"[a-z0-9_.:-]+", text) else None


def _responses_endpoint(config: dict[str, Any]) -> str | None:
    if str(config.get("api_family") or "").strip().lower() != "responses":
        return None
    base = str(config.get("responses_base_url") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("responses_base_url_missing")
    return f"{base}/responses"


def _validate_local_responses_endpoint(endpoint: str) -> None:
    validate_local_endpoint(endpoint, api_family=RESPONSES)


def _sample_responses(
    endpoint: str,
    observation: dict[str, Any],
    config: dict[str, Any],
    *,
    validate: Callable[[str], None] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    # The hosted responses lane and the local `synth_mlx_rl` lane share this
    # sampler but not their admission rule, so the validator is chosen by the
    # caller rather than inferred from the URL.
    (validate or _validate_responses_endpoint)(endpoint)
    api_key = str(config.get("responses_api_key") or "").strip()
    idempotency_key = str(config.get("responses_idempotency_key") or "").strip()
    model = str(config.get("responses_model") or "policy").strip()
    if not api_key or "\r" in api_key or "\n" in api_key:
        raise RuntimeError("responses_api_key_missing")
    if not idempotency_key or "\r" in idempotency_key or "\n" in idempotency_key:
        raise RuntimeError("responses_idempotency_key_missing")
    maximum = min(max(int(config.get("max_tokens") or 32), 1), 512)
    body = json.dumps(
        {
            "model": model,
            "instructions": CLASSIFY_SYSTEM,
            "input": user_prompt(str(observation["text"])),
            "max_output_tokens": maximum,
            "temperature": 0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Idempotency-Key": idempotency_key,
            "X-Policy-Pin": str(config.get("policy_pin") or "generation"),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    timeout = min(max(float(config.get("responses_timeout_seconds") or 120.0), 1.0), 300.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(1_048_577)
    except urllib.error.HTTPError as exc:
        mapping = {
            401: "responses_auth_refused",
            404: "responses_policy_not_found",
            409: "responses_policy_conflict",
            422: "responses_request_invalid",
            429: "responses_backpressure",
            502: "responses_sampling_failed",
            503: "responses_sampler_unavailable",
        }
        raise RuntimeError(mapping.get(exc.code, "responses_http_error")) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("responses_transport_error") from exc
    if len(raw) > 1_048_576:
        raise RuntimeError("responses_response_too_large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("responses_response_invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("responses_response_invalid")
    text = payload.get("output_text")
    if not isinstance(text, str):
        fragments: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    fragments.append(str(part.get("text") or ""))
        text = "".join(fragments)
    if not isinstance(text, str):
        raise RuntimeError("responses_response_invalid")
    predicted = text.strip().splitlines()[0].strip() if text.strip() else None
    return predicted, _responses_usage(payload.get("usage"))


def _validate_responses_endpoint(endpoint: str) -> None:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("responses_endpoint_refused") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/responses")
    ):
        raise RuntimeError("responses_endpoint_refused")
    host = parsed.hostname.lower()
    if host in {"127.0.0.1", "localhost", "::1"} and parsed.scheme == "http":
        return
    normalized = f"{parsed.scheme}://{host}"
    if port is not None:
        normalized += f":{port}"
    allowed = {
        item.strip().rstrip("/")
        for item in os.environ.get("SYNTH_RESPONSES_ALLOWED_ENDPOINTS", "").split(",")
        if item.strip()
    }
    if normalized not in allowed:
        raise RuntimeError("responses_endpoint_refused")


def _responses_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return dict(_EMPTY_USAGE)
    mapped = {
        "prompt_tokens": value.get("input_tokens"),
        "completion_tokens": value.get("output_tokens"),
        "total_tokens": value.get("total_tokens"),
    }
    return {
        key: item
        if item is None or (isinstance(item, int) and not isinstance(item, bool) and item >= 0)
        else None
        for key, item in mapped.items()
    }


def _tinker_endpoint(config: dict[str, Any]) -> str | None:
    target = config.get("inference_target")
    if not isinstance(target, dict):
        return None
    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    if endpoint.startswith("tinker:") or endpoint.startswith("tinker://"):
        return endpoint
    return None


def _remote_checkpoint_target(config: dict[str, Any]) -> dict[str, Any] | None:
    target = config.get("inference_target")
    if not isinstance(target, dict):
        return None
    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return dict(target)
    return None


def _sample_remote_checkpoint(
    target: dict[str, Any],
    observation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    if config.get("training_sampler_endpoint") is not True:
        return _sample_remote_checkpoint_legacy(target, observation, config)
    if str(target.get("provider") or "").strip().lower() != "tinker":
        raise RuntimeError("remote_checkpoint_provider_unsupported")
    auth_bearer = str(target.get("auth_bearer") or "").strip()
    if not auth_bearer or "\r" in auth_bearer or "\n" in auth_bearer:
        raise RuntimeError("remote_checkpoint_auth_missing")
    run_id = str(target.get("run_id") or "").strip()
    checkpoint_id = str(target.get("checkpoint_id") or "").strip()
    if not run_id or not checkpoint_id:
        raise RuntimeError("remote_checkpoint_identity_missing")
    max_tokens = min(max(int(config.get("max_tokens") or 32), 1), 512)
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": user_prompt(str(observation["text"]))},
    ]
    timeout = min(max(float(config.get("remote_timeout_seconds") or 60.0), 1.0), 120.0)
    from ...training_rollout import (
        ROLLOUT_ACTION_SCHEMA_VERSION,
        HostedSamplerClient,
        SamplerEndpoint,
        canonical_sha256,
    )

    message_digest = canonical_sha256({"messages": messages})
    with HostedSamplerClient(
        SamplerEndpoint(
            endpoint,
            auth_bearer,
            str(target.get("connection_mode") or "keep_alive"),
        ),
        timeout_seconds=timeout,
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
                "temperature": 0.0,
            },
            idempotency_key=(
                f"{config.get('rollout_id') or run_id}:{checkpoint_id}:{message_digest}"
            ),
        )
    usage = _remote_usage(dict(sampled.usage))
    usage["_training_action"] = {
        "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
        "policy_version": config.get("policy_version") or checkpoint_id,
        "prompt_token_ids": list(sampled.prompt_token_ids),
        "token_ids": list(sampled.token_ids),
        "log_probs": list(sampled.log_probs),
    }
    predicted = sampled.text.strip().splitlines()[0].strip() if sampled.text.strip() else None
    return predicted, usage


def _sample_remote_checkpoint_legacy(
    target: dict[str, Any],
    observation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Preserve the existing loopback/allowlisted SFT checkpoint contract."""

    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    _validate_remote_checkpoint_endpoint(endpoint)
    if str(target.get("provider") or "").strip().lower() != "tinker":
        raise RuntimeError("remote_checkpoint_provider_unsupported")
    auth_bearer = str(target.get("auth_bearer") or "").strip()
    if not auth_bearer or "\r" in auth_bearer or "\n" in auth_bearer:
        raise RuntimeError("remote_checkpoint_auth_missing")
    run_id = str(target.get("run_id") or "").strip()
    checkpoint_id = str(target.get("checkpoint_id") or "").strip()
    if not run_id or not checkpoint_id:
        raise RuntimeError("remote_checkpoint_identity_missing")
    max_tokens = min(max(int(config.get("max_tokens") or 32), 1), 512)
    body = json.dumps(
        {
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "messages": [
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user_prompt(str(observation["text"]))},
            ],
            "max_tokens": max_tokens,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {auth_bearer}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    timeout = min(max(float(config.get("remote_timeout_seconds") or 60.0), 1.0), 120.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(1_048_577)
    except urllib.error.HTTPError as exc:
        mapping = {
            401: "remote_checkpoint_auth_refused",
            404: "remote_checkpoint_not_found",
            409: "remote_checkpoint_unavailable",
            422: "remote_checkpoint_request_invalid",
            502: "remote_checkpoint_sampling_failed",
        }
        raise RuntimeError(mapping.get(exc.code, "remote_checkpoint_http_error")) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("remote_checkpoint_transport_error") from exc
    if len(raw) > 1_048_576:
        raise RuntimeError("remote_checkpoint_response_too_large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("remote_checkpoint_response_invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise RuntimeError("remote_checkpoint_response_invalid")
    text = payload["text"]
    predicted = text.strip().splitlines()[0].strip() if text.strip() else None
    return predicted, _remote_usage(payload.get("usage"))


def _validate_remote_checkpoint_endpoint(endpoint: str) -> None:
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError("remote_checkpoint_endpoint_refused") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise RuntimeError("remote_checkpoint_endpoint_refused")
    host = parsed.hostname.lower()
    if host in {"127.0.0.1", "localhost", "::1"} and parsed.scheme == "http":
        return
    normalized = f"{parsed.scheme}://{host}"
    if port is not None:
        normalized += f":{port}"
    allowed = {
        item.strip().rstrip("/")
        for item in os.environ.get("SYNTH_CHECKPOINT_INFERENCE_ALLOWED_ENDPOINTS", "").split(",")
        if item.strip()
    }
    if normalized not in allowed:
        raise RuntimeError("remote_checkpoint_endpoint_refused")


def _remote_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return dict(_EMPTY_USAGE)
    usage: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        usage[key] = (
            item
            if item is None or (isinstance(item, int) and not isinstance(item, bool) and item >= 0)
            else None
        )
    return usage


def _sample_tinker(
    endpoint: str,
    observation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    try:
        import tinker
    except ImportError as exc:
        raise RuntimeError("tinker_sdk_missing") from exc

    target = (
        config.get("inference_target") if isinstance(config.get("inference_target"), dict) else {}
    )
    base_model = str(target.get("base_model") or config.get("base_model") or "").strip()
    if not base_model:
        raise RuntimeError("tinker_base_model_missing")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers_missing") from exc

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": user_prompt(str(observation["text"]))},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    model_input_cls = getattr(tinker, "ModelInput", None) or tinker.types.ModelInput
    try:
        model_input = model_input_cls.from_ints(tokens=prompt_ids)
    except TypeError:
        model_input = model_input_cls.from_ints(prompt_ids)
    max_tokens = int(config.get("max_tokens") or 32)
    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(model_path=endpoint)
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0)
    result = sampling_client.sample(
        prompt=model_input, num_samples=1, sampling_params=params
    ).result()
    seq = result.sequences[0]
    text = tokenizer.decode(list(map(int, seq.tokens)), skip_special_tokens=True)
    predicted = text.strip().splitlines()[0].strip() if text.strip() else None
    return predicted, dict(_EMPTY_USAGE)
