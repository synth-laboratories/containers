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

import re
from typing import Any

from ...event_log import RolloutEventLog
from ..banking77_world import (
    CLASSIFY_SYSTEM,
    load_row,
    normalize_label,
    public_observation,
    split_from_world_ref,
    user_prompt,
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

        log.append("action", {"label": predicted, "text": predicted})
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

        endpoint = _tinker_endpoint(config)
        if endpoint:
            return _sample_tinker(endpoint, observation, config)

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


def _tinker_endpoint(config: dict[str, Any]) -> str | None:
    target = config.get("inference_target")
    if not isinstance(target, dict):
        return None
    endpoint = str(target.get("provider_endpoint_id") or "").strip()
    if endpoint.startswith("tinker:") or endpoint.startswith("tinker://"):
        return endpoint
    return None


def _sample_tinker(
    endpoint: str,
    observation: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    try:
        import tinker
    except ImportError as exc:
        raise RuntimeError("tinker_sdk_missing") from exc

    target = config.get("inference_target") if isinstance(config.get("inference_target"), dict) else {}
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
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
