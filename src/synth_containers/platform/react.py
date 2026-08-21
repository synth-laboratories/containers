"""ReAct harness over a Craftax-shaped world.

`craftax_engine` uses a scripted planner (no model). That is engine-acceptance,
not a Luna eval. `craftax_react --paid` may call a chat model when a key exists.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .craftax_taxonomy import usage_from_call_identity
from .craftax_world import ACTIONS
from .local_provider import (
    API_FAMILIES,
    CHAT_COMPLETIONS,
    RESPONSES,
    local_endpoint,
)
from .local_provider import PROVIDER_ID as LOCAL_PROVIDER_ID

DeltaCallback = Callable[[dict[str, Any]], None]


# The prompt seeded configs use. Named and exported rather than inlined as a
# fallback, so a config that wants it must say so and one that forgets is refused.
CRAFTAX_REACT_SYSTEM_PROMPT = (
    "You are a careful Craftax ReAct policy. You must call the "
    "choose_actions tool exactly once; do not answer with prose."
)


class PolicyConfigError(ValueError):
    """A policy config omitted a field that defines what is being measured."""


def _required_str(
    config: dict[str, Any], key: str, config_id: str, alias: str | None = None
) -> str:
    value = config.get(key)
    if (not isinstance(value, str) or not value.strip()) and alias is not None:
        value = config.get(alias)
    if not isinstance(value, str) or not value.strip():
        named = repr(key) if alias is None else f"{key!r} (or {alias!r})"
        raise PolicyConfigError(
            f"policy config {config_id!r} must set {named} explicitly; "
            f"it defines the policy under test and is never defaulted"
        )
    return value.strip()


def _optional_temperature(config: dict[str, Any], default: float = 0.0) -> float:
    raw = config.get("temperature", default)
    try:
        temperature = float(raw)
    except (TypeError, ValueError) as exc:
        raise PolicyConfigError(
            f"policy config temperature {raw!r} is not a number"
        ) from exc
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
        raise PolicyConfigError(
            f"policy config temperature={temperature} is outside [0, 2]"
        )
    return temperature


def _hosted_react_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fill ReAct recipe fields the training boundary does not send.

    Hosted CISPO injects a sampler endpoint, temperature, and max_tokens.
    The rest of this class is a Craftax episode loop that still needs a
    model id, compaction window, and system prompt. Those are harness
    constants on this path, not a silent fallback for a paid eval recipe.
    """
    resolved = dict(config)
    resolved.setdefault("model", str(resolved.get("policy_version") or "hosted-sampler"))
    resolved.setdefault("effort", "none")
    if not isinstance(resolved.get("max_tokens"), int) or isinstance(
        resolved.get("max_tokens"), bool
    ):
        resolved["max_tokens"] = 256
    if not isinstance(resolved.get("context_token_budget"), int) or isinstance(
        resolved.get("context_token_budget"), bool
    ):
        resolved["context_token_budget"] = 16_000
    if not isinstance(resolved.get("compact_at"), (int, float)) or isinstance(
        resolved.get("compact_at"), bool
    ):
        resolved["compact_at"] = 0.7
    if not isinstance(resolved.get("keep_recent_messages"), int) or isinstance(
        resolved.get("keep_recent_messages"), bool
    ):
        resolved["keep_recent_messages"] = 8
    if not isinstance(resolved.get("keep_recent_frames"), int) or isinstance(
        resolved.get("keep_recent_frames"), bool
    ):
        resolved["keep_recent_frames"] = 2
    resolved.setdefault("observation_mode", "text")
    resolved.setdefault("provider", "tinker")
    resolved.setdefault("api_key_env", "TINKER_API_KEY")
    if not isinstance(resolved.get("parse_retries"), int) or isinstance(
        resolved.get("parse_retries"), bool
    ):
        resolved["parse_retries"] = 0
    resolved.setdefault("system_prompt", CRAFTAX_REACT_SYSTEM_PROMPT)
    return resolved


def _required_bounded_int(
    config: dict[str, Any], key: str, config_id: str, low: int, high: int
) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise PolicyConfigError(
            f"policy config {config_id!r} must set {key!r} explicitly as an int; "
            f"it changes the policy under test and is never defaulted"
        )
    if not low <= value <= high:
        raise PolicyConfigError(
            f"policy config {config_id!r} {key}={value} is outside [{low}, {high}]"
        )
    return value




def _required_fraction(
    config: dict[str, Any], key: str, config_id: str, low: float, high: float
) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PolicyConfigError(
            f"policy config {config_id!r} must set {key!r} explicitly as a number; "
            f"it changes the policy under test and is never defaulted"
        )
    if not low <= float(value) <= high:
        raise PolicyConfigError(
            f"policy config {config_id!r} {key}={value} is outside [{low}, {high}]"
        )
    return float(value)


def _required_choice(
    config: dict[str, Any], key: str, config_id: str, allowed: tuple[str, ...]
) -> str:
    value = config.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise PolicyConfigError(
            f"policy config {config_id!r} must set {key!r} to one of {allowed}; "
            f"it changes the policy under test and is never defaulted"
        )
    return value


def _text_of(message: dict[str, Any]) -> str:
    """Text of a message whose content may be a multimodal part list.

    Frames are named, never inlined: shipping accumulated base64 to the
    summarizer is enormous, and it cannot use pixels in a text brief anyway.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                parts.append("<frame omitted>")
            elif isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts)
    return ""



class ScriptedReAct:
    """Observe → open span → emit a 5-action plan → close span → execute.

    Plan bounds match the real Craftax ReAct harness (min 5, max 20). The plan
    is authored, not sampled: `do` is always included so env-sum is non-zero
    when the player is on the tree, else movement hunts it.
    """

    plan_min = 5
    plan_max = 20

    def __init__(self, *, config_id: str) -> None:
        self.config_id = config_id
        self.calls = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": "react",
            "kind": "scripted_react",
            "config": self.config_id,
            "plan_min": self.plan_min,
            "plan_max": self.plan_max,
            "graded": False,
            "note": "Engine-acceptance ReAct loop. No model; do not report as a Luna eval.",
        }

    def usage(self) -> dict[str, Any]:
        return usage_from_call_identity(
            calls=self.calls,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            kind="scripted",
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": "synth.containers.scripted-react-checkpoint.v1",
            "calls": self.calls,
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "synth.containers.scripted-react-checkpoint.v1":
            raise RuntimeError("unsupported scripted policy checkpoint schema")
        self.calls = int(state.get("calls") or 0)

    def plan(self, observation: dict[str, Any], on_delta: DeltaCallback | None = None) -> list[str]:
        del on_delta
        self.calls += 1
        tree = tuple(observation.get("tree") or (3, 2))
        x, y = int(observation.get("x") or 0), int(observation.get("y") or 0)
        moves: list[str] = []
        if x < tree[0]:
            moves.append("east")
        elif x > tree[0]:
            moves.append("west")
        if y < tree[1]:
            moves.append("south")
        elif y > tree[1]:
            moves.append("north")
        if not moves:
            moves = ["do"]
        plan = (moves + ["do", "sleep", "noop", "do"])[: self.plan_min]
        while len(plan) < self.plan_min:
            plan.append("noop")
        return [item if item in ACTIONS else "noop" for item in plan]


class UniformEnginePolicy:
    """Seeded walk over legal actions. Transport baseline, not ReAct."""

    def __init__(self, seed: int = 0) -> None:
        import random

        self._rng = random.Random(seed)

    def metadata(self) -> dict[str, Any]:
        return {"harness": "valid_action_uniform", "kind": "baseline", "graded": False}

    def usage(self) -> dict[str, Any]:
        return usage_from_call_identity(
            calls=0,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            kind="scripted",
        )

    def plan(self, observation: dict[str, Any], on_delta: DeltaCallback | None = None) -> list[str]:
        del on_delta
        valid = list(observation.get("valid_actions") or ACTIONS)
        return [self._rng.choice(valid)]


class OpenRouterReAct:
    """Paid ReAct planner for reviewed provider configs.

    The key is read only at request time and never enters metadata or the event
    log. History is the same session across `plan()` calls; compaction
    (default 16) is a harness facet. Token deltas are forwarded only when the
    provider actually streams non-empty chunks — Luna often returns empty
    reasoning, and that absence is left blank.
    """

    plan_min = 5
    plan_max = 20
    compact_keep_turns = 2

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self._inference_target = (
            dict(config["inference_target"])
            if isinstance(config.get("inference_target"), dict)
            else None
        )
        self.temperature = _optional_temperature(config)
        self._last_training_action: dict[str, Any] | None = None
        self.last_call_usage: dict[str, Any] | None = None
        if self._inference_target is not None:
            config = _hosted_react_config(config)
        # Policy identity is never defaulted. A silent fallback here attributes
        # a result to a model, effort or memory window the caller never asked
        # for, and the mistake is invisible in the output — the run simply
        # reports a score for "luna_med" that some other policy produced.
        self.model = _required_str(config, "model", config_id)
        self.reasoning_effort = _required_str(config, "effort", config_id)
        self.max_tokens = _required_bounded_int(config, "max_tokens", config_id, 64, 4096)
        # Compaction knobs mirror craftax_gold.rs. The old `compact_every` fired
        # on a turn count and pasted truncated payloads; that is a different
        # policy from a token-triggered, model-authored summary, and comparing
        # a checkpoint evaluated one way against teacher data collected the
        # other way measures the mismatch rather than the model.
        self.context_token_budget = _required_bounded_int(
            config, "context_token_budget", config_id, 2000, 400_000
        )
        self.compact_at = _required_fraction(config, "compact_at", config_id, 0.1, 0.95)
        self.keep_recent_messages = _required_bounded_int(
            config, "keep_recent_messages", config_id, 2, 64
        )
        self.keep_recent_frames = _required_bounded_int(
            config, "keep_recent_frames", config_id, 1, 16
        )
        self.observation_mode = _required_choice(
            config, "observation_mode", config_id, ("text", "image", "both")
        )
        self.compact_threshold = int(self.context_token_budget * self.compact_at)
        self.last_prompt_tokens = 0
        # Where inference is served is not a detail: defaulting it to OpenRouter
        # sent a `tinker-infer:` checkpoint id to a provider that cannot serve it,
        # and every rollout died on turn 0 with `policy_error` after zero steps.
        # A missing endpoint must say so, not pick one.
        self.provider = _required_choice(
            config, "provider", config_id, ("openrouter", LOCAL_PROVIDER_ID, "tinker")
        )
        self.api_key_env = _required_str(config, "api_key_env", config_id)
        self.parse_retries = _required_bounded_int(config, "parse_retries", config_id, 0, 2)
        # OpenRouter and Tinker speak chat-completions only. `synth_mlx_rl` may
        # name either family and overrides this below.
        self.api_family = CHAT_COMPLETIONS
        self.policy_snapshot_id = str(config.get("policy_snapshot_id") or "").strip()
        if self.provider in {"openrouter", LOCAL_PROVIDER_ID}:
            self.base_url = _required_str(config, "base_url", config_id).rstrip("/")
            self.chat_endpoint = f"{self.base_url}/chat/completions"
            self.sampler_path = ""
            self.tokenizer_id = ""
            self.sampler_ready_timeout_s = 0
            if self.provider == LOCAL_PROVIDER_ID:
                # The local proxy has no fixed origin, so admission is the shared
                # `synth_mlx_rl` check rather than a hosted base URL: loopback or
                # the Docker host alias over http, or an origin named in
                # SYNTH_MLX_RL_ALLOWED_ENDPOINTS. Refusing at config time rather
                # than at sample time keeps a bad endpoint from reading later as
                # a policy that scored zero.
                family = _required_choice(config, "api_family", config_id, API_FAMILIES)
                self.api_family = family
                try:
                    self.chat_endpoint = local_endpoint(self.base_url, api_family=family)
                except RuntimeError as exc:
                    raise PolicyConfigError(
                        f"policy config {config_id!r} names a base_url the local "
                        f"{LOCAL_PROVIDER_ID} provider refuses: {exc}"
                    ) from exc
        else:
            # Hosted CISPO names a SamplerEndpoint, not a Tinker SDK path.
            # Going through the SDK here cannot be pointed at the run's live
            # hosted sampler and bypasses the token receipt the summary needs.
            self.base_url = ""
            self.chat_endpoint = ""
            if self._inference_target is not None:
                self.sampler_path = ""
                self.tokenizer_id = ""
                self.sampler_ready_timeout_s = 0
            else:
                # Tinker samples through its SDK, not an HTTP base_url. The
                # sampler path names the trained weights; the tokenizer must
                # be the base model's, or the chat template renders a prompt
                # the student never saw in training.
                self.sampler_path = _required_str(config, "sampler_path", config_id)
                self.tokenizer_id = _required_str(config, "tokenizer_id", config_id)
                self.sampler_ready_timeout_s = _required_bounded_int(
                    config, "sampler_ready_timeout_s", config_id, 0, 900
                )
            if self.observation_mode != "text":
                raise PolicyConfigError(
                    f"policy config {config_id!r} sets observation_mode="
                    f"{self.observation_mode!r}, but the tinker provider renders a "
                    "text chat template and cannot carry frames; dropping them "
                    "would evaluate a different policy than the one requested"
                )
        self._sampling_client: Any = None
        self._tokenizer: Any = None
        self.calls = 0
        self._compact_count = 0
        self._deltas_emitted = 0
        self._usage: dict[str, int | float | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }
        self._last_trace: dict[str, Any] = {}
        # The system prompt is the policy. Substituting a fallback is how a
        # student gets trained against one prompt and measured against another
        # (see image_input.md) — the mismatch then reads as "no uplift".
        system_prompt = _required_str(
            config, "system_prompt", config_id, alias="react_system_prompt"
        )
        self._messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": "react",
            "kind": "openrouter_react",
            "config": self.config_id,
            "model": self.model,
            "provider": "openrouter",
            "reasoning_effort": self.reasoning_effort,
            "plan_min": self.plan_min,
            "plan_max": self.plan_max,
            "context_token_budget": self.context_token_budget,
            "compact_at": self.compact_at,
            "keep_recent_messages": self.keep_recent_messages,
            "keep_recent_frames": self.keep_recent_frames,
            "observation_mode": self.observation_mode,
            "token_trace": "derived",
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return usage_from_call_identity(
            calls=self.calls,
            prompt_tokens=self._usage.get("prompt_tokens") if isinstance(self._usage.get("prompt_tokens"), int) else None,
            completion_tokens=self._usage.get("completion_tokens") if isinstance(self._usage.get("completion_tokens"), int) else None,
            total_tokens=self._usage.get("total_tokens") if isinstance(self._usage.get("total_tokens"), int) else None,
            cost_usd=self._usage.get("cost_usd") if isinstance(self._usage.get("cost_usd"), (int, float)) else None,
            kind="model",
        )

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    @property
    def last_training_action(self) -> dict[str, Any] | None:
        return self._last_training_action

    def checkpoint_state(self) -> dict[str, Any]:
        """Secret-free policy-session state paired with an environment snapshot."""
        return {
            "schema_version": "synth.containers.react-checkpoint.v1",
            "messages": json.loads(json.dumps(self._messages)),
            "calls": self.calls,
            "compact_count": self._compact_count,
            "deltas_emitted": self._deltas_emitted,
            "usage": dict(self._usage),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "synth.containers.react-checkpoint.v1":
            raise RuntimeError("unsupported policy checkpoint schema")
        messages = state.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RuntimeError("policy checkpoint omitted messages")
        restored = json.loads(json.dumps(messages))
        if not isinstance(restored[0], dict) or restored[0].get("role") != "system":
            raise RuntimeError("policy checkpoint omitted system message")
        # The branch evaluates the NEW candidate prompt while retaining the
        # parent's observation/action history. Never silently restore the parent
        # candidate's system prompt.
        restored[0] = dict(self._messages[0])
        self._messages = restored
        self.calls = int(state.get("calls") or 0)
        self._compact_count = int(state.get("compact_count") or 0)
        self._deltas_emitted = int(state.get("deltas_emitted") or 0)
        usage = state.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("policy checkpoint omitted usage")
        self._usage = dict(usage)

    def plan(
        self,
        observation: dict[str, Any],
        on_delta: DeltaCallback | None = None,
    ) -> list[str]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if self._inference_target is None and not api_key:
            raise RuntimeError(f"paid Craftax policy requires {self.api_key_env}")
        valid = [str(action) for action in observation.get("valid_actions") or []]
        if not valid:
            raise RuntimeError("Craftax observation omitted valid_actions")
        self._maybe_compact(api_key, on_delta)
        text = self._observation_prompt(observation, valid)
        frame = observation.get("frame_png_b64")
        if self.observation_mode in ("image", "both"):
            if not isinstance(frame, str) or not frame:
                # Failing here is the point: silently falling back to text would
                # evaluate a different policy than the one requested, which is
                # exactly the train/eval mismatch this alignment exists to close.
                raise RuntimeError(
                    f"observation_mode={self.observation_mode!r} requires a frame; "
                    "the world supplied none (needs live_frames=native)"
                )
            parts: list[dict[str, Any]] = []
            if self.observation_mode == "both":
                parts.append({"type": "text", "text": text})
            else:
                parts.append({"type": "text", "text": "Current view:"})
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame}"}}
            )
            self._messages.append({"role": "user", "content": parts})
            self._prune_frames()
        else:
            self._messages.append({"role": "user", "content": text})
        prior_attempts: list[dict[str, Any]] = []
        assistant = ""
        reasoning = ""
        tool_arguments = ""
        usage: dict[str, Any] = {}
        parse_error: str | None = None
        fallback = False
        actions: list[str] = []
        deltas_before = self._deltas_emitted
        for attempt in range(self.parse_retries + 1):
            body = self._complete(api_key, valid, on_delta)
            message = (body.get("choices") or [{}])[0].get("message") or {}
            assistant = self._message_text(message.get("content"))
            reasoning = self._message_text(
                message.get("reasoning") or message.get("reasoning_content")
            )
            tool_arguments = self._tool_arguments(message.get("tool_calls"))
            usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
            emitted = bool(usage)
            self._accumulate_usage(usage)
            if usage:
                self.last_call_usage = {
                    key: usage.get(key)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if isinstance(usage.get(key), int)
                } or self.last_call_usage
            policy_output = tool_arguments or assistant
            try:
                actions = self._parse_actions(policy_output, valid)
                parse_error = None
                fallback = False
                break
            except (json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
                prior_attempts.append(
                    {
                        "assistant": assistant,
                        "reasoning": reasoning,
                        "tool_arguments": tool_arguments,
                        "parse_error": str(exc) or exc.__class__.__name__,
                        "usage": {
                            **{
                                key: usage.get(key)
                                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                            },
                            "cost_usd": usage.get("cost"),
                        },
                    }
                )
                if attempt >= self.parse_retries:
                    # A provider can return an empty content field while placing
                    # its reasoning elsewhere, or ignore JSON mode. Do not lose
                    # the real call/usage evidence and do not abort the other
                    # nine lanes. The harness fallback is explicit in the trace
                    # so no UI can present it as a model-authored action.
                    parse_error = str(exc) or exc.__class__.__name__
                    actions = ["do" if "do" in valid else valid[0]]
                    fallback = True
        self.calls += 1
        self._messages.append(
            {
                "role": "assistant",
                "content": tool_arguments or assistant or json.dumps({"actions": actions}),
            }
        )
        provider_cost = usage.get("cost")
        call_deltas = self._deltas_emitted - deltas_before
        self._last_trace = {
            "call": self.calls,
            "model": self.model,
            "provider": "openrouter",
            "assistant": assistant,
            "reasoning": reasoning,
            "tool_arguments": tool_arguments,
            "actions": actions,
            "action_authority": "harness_fallback" if fallback else "policy",
            "fallback": fallback,
            "parse_error": parse_error,
            "context_token_budget": self.context_token_budget,
            "compact_at": self.compact_at,
            "keep_recent_messages": self.keep_recent_messages,
            "keep_recent_frames": self.keep_recent_frames,
            "observation_mode": self.observation_mode,
            "compact_count": self._compact_count,
            "history_turns": self.calls,
            "deltas_emitted": call_deltas,
            "token_trace": "derived" if call_deltas else None,
            "usage": {
                **{
                    key: usage.get(key) if isinstance(usage.get(key), int) else None
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "cost_usd": provider_cost if isinstance(provider_cost, (int, float)) else None,
                "usage_status": "reported" if emitted else "provider_omitted",
            },
        }
        if prior_attempts:
            self._last_trace["prior_attempts"] = prior_attempts
        return actions

    def _observation_prompt(self, observation: dict[str, Any], valid: list[str]) -> str:
        observation_text = str(
            observation.get("observation_text") or observation.get("ascii") or ""
        )
        return (
            "Make measurable Craftax achievement progress while staying alive. "
            f"Choose {self.plan_min}-{min(self.plan_max, len(valid))} sequential actions from the exact legal list.\n\n"
            f"{observation_text}\n\nvalid_actions={json.dumps(valid)}\n"
            'Return JSON only: {"actions":["do","right"]}'
        )

    def _maybe_compact(self, api_key: str, on_delta: DeltaCallback | None) -> None:
        if self.calls == 0:
            return
        pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
        rest = self._messages[1:]
        index = 0
        while index < len(rest):
            message = rest[index]
            if message.get("role") != "user":
                index += 1
                continue
            assistant = None
            if index + 1 < len(rest) and rest[index + 1].get("role") == "assistant":
                assistant = rest[index + 1]
                index += 2
            else:
                index += 1
            if str(message.get("content") or "").startswith("[compacted "):
                continue
            pairs.append((message, assistant))
        # Token-triggered, matching craftax_gold.rs: compaction fires on the
        # real prompt_tokens the provider reported, not on a turn count. A
        # turn count is a poor proxy — turns vary hugely in size once frames
        # and long observations are in the transcript.
        if self.last_prompt_tokens <= self.compact_threshold:
            return
        if len(pairs) <= self.keep_recent_messages:
            return
        keep = pairs[-self.keep_recent_messages :]
        dropped = pairs[: -self.keep_recent_messages]
        if not dropped:
            return

        # Model-authored summary, not a mechanical paste of truncated payloads.
        # The summary is the agent's only memory of these turns, so it has to
        # carry map knowledge and intent rather than the last 200 characters.
        transcript: list[dict[str, Any]] = []
        for user, assistant in dropped:
            transcript.append({"role": "user", "content": _text_of(user)})
            if assistant is not None:
                transcript.append({"role": "assistant", "content": _text_of(assistant)})
        summary = self._summarize(api_key, transcript)

        compact = {
            "role": "user",
            "content": (
                "[context compacted — the turns below are summarized, not verbatim]\n"
                + summary
            ),
        }
        rebuilt: list[dict[str, Any]] = [self._messages[0], compact]
        for user, assistant in keep:
            rebuilt.append(user)
            if assistant is not None:
                rebuilt.append(assistant)
        self._messages = rebuilt
        self._prune_frames()
        self._compact_count += 1
        if on_delta is not None:
            on_delta(
                {
                    "delta": False,
                    "channel": "compact",
                    "trigger": "fixed_threshold",
                    "prompt_tokens_before": self.last_prompt_tokens,
                    "compact_threshold": self.compact_threshold,
                    "compact_count": self._compact_count,
                    "dropped_turns": len(dropped),
                    "kept_turns": len(keep),
                    "summarized": True,
                    "call": self.calls,
                    "model": self.model,
                    "provider": "openrouter",
                }
            )


    def _prune_frames(self) -> None:
        """Keep only the most recent frames as pixels; older ones keep text.

        Frames re-send on every request until compaction. Unbounded, a single
        episode reached ~1M prompt tokens and blew the client timeout — and the
        resulting failures killed long rollouts preferentially, biasing any
        comparison toward early deaths.
        """
        seen = 0
        for message in reversed(self._messages):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if not any(
                isinstance(part, dict) and part.get("type") == "image_url" for part in content
            ):
                continue
            seen += 1
            if seen > self.keep_recent_frames:
                message["content"] = _text_of(message) + " [frame dropped — superseded]"

    def _summarize(self, api_key: str, transcript: list[dict[str, Any]]) -> str:
        """Ask the model to compact its own history into a usable brief.

        Its own request, not `_complete`, which always sends `self._messages`.
        Non-streaming and tool-free: this call produces prose, not actions.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You compact an agent's episode transcript. Write a dense "
                        "factual brief the agent will rely on as its only memory "
                        "of these turns."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize this episode segment for the agent that will keep "
                        "playing. Cover: terrain and resources seen and roughly where "
                        "relative to the player; inventory and achievements gained; "
                        "what was attempted and failed; the current goal. Be concrete "
                        "and brief. No preamble.\n\n" + json.dumps(transcript)
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": 700,
        }
        if self.provider == "tinker":
            # A checkpoint trained to emit `{"actions":[...]}` cannot be asked
            # for prose, and borrowing a second model to summarize would put a
            # policy nobody named inside the loop. Name the gap instead.
            return (
                f"[summary unavailable; provider={self.provider!r} does not summarize; "
                f"{len(transcript)} messages dropped]"
            )
        if self.api_family == RESPONSES:
            # Same prose request on the peer surface. `messages` -> `input`,
            # `max_tokens` -> `max_output_tokens`; a chat body posted to a
            # `/responses` route is a 4xx, and a failed summary silently drops
            # the compacted turns.
            payload = {
                "model": payload["model"],
                "input": [
                    {
                        "role": message["role"],
                        "content": [{"type": "input_text", "text": message["content"]}],
                    }
                    for message in payload["messages"]
                ],
                "temperature": self.temperature,
                "max_output_tokens": 700,
            }
        request = urllib.request.Request(
            self.chat_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            if self.api_family == RESPONSES:
                text = self._responses_output(body)["content"]
            else:
                text = self._message_text(
                    (body.get("choices") or [{}])[0].get("message", {}).get("content")
                )
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            # A failed summary must not end the rollout. Name what was dropped
            # rather than silently losing the turns.
            return f"[summary unavailable; {len(transcript)} messages dropped]"
        return text.strip() or f"[summary empty; {len(transcript)} messages dropped]"

    def _tinker_sample(self, api_key: str) -> dict[str, Any]:
        """Sample the trained checkpoint.

        Returns an OpenAI-shaped body so the caller's parse, retry, usage and
        compaction paths are the same ones the OpenRouter provider exercises.
        Tinker has no tool-calling API, so the sample arrives as assistant text
        — which is exactly the `{"actions":[...]}` shape the SFT targets were
        written in, and `_parse_actions` already reads it.

        A hosted training run names a SamplerEndpoint. Sampling through the
        Tinker SDK instead cannot be pointed at that endpoint, pins
        temperature at whatever this function used to hardcode, and drops the
        per-call token receipt the rollout summary accounts spend from.
        """
        if self._inference_target is not None:
            return self._hosted_sampler_sample()
        if self._sampling_client is None:
            import tinker  # noqa: PLC0415 — optional dependency of the tinker provider
            from transformers import AutoTokenizer  # noqa: PLC0415

            self._tinker = tinker
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)
            # Checkpoint evaluation starts seconds after save_weights_for_sampler
            # returns, and the weights are not necessarily servable yet. Failing
            # on the first attempt reads downstream as "the checkpoint scores
            # zero", which is the most expensive way to learn about a race.
            deadline = time.monotonic() + self.sampler_ready_timeout_s
            attempt = 0
            while True:
                attempt += 1
                try:
                    self._sampling_client = tinker.ServiceClient(
                        api_key=api_key
                    ).create_sampling_client(model_path=self.sampler_path)
                    break
                except Exception as exc:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"tinker sampler {self.sampler_path!r} was not servable after "
                            f"{self.sampler_ready_timeout_s}s and {attempt} attempts: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    time.sleep(min(2.0 * attempt, 10.0))

        tinker = self._tinker
        tokenizer = self._tokenizer
        try:
            prompt = tokenizer.apply_chat_template(
                list(self._messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                list(self._messages), tokenize=False, add_generation_prompt=True
            )
        prompt_ids = [int(value) for value in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        model_input_cls = getattr(tinker, "ModelInput", None) or tinker.types.ModelInput
        try:
            model_input = model_input_cls.from_ints(tokens=prompt_ids)
        except TypeError:
            model_input = model_input_cls.from_ints(prompt_ids)
        deadline = time.monotonic() + self.sampler_ready_timeout_s
        attempt = 0
        while True:
            attempt += 1
            try:
                sequence = (
                    self._sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            max_tokens=self.max_tokens, temperature=self.temperature
                        ),
                    )
                    .result()
                    .sequences[0]
                )
                break
            except Exception as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"tinker sample failed after {attempt} attempts over "
                        f"{self.sampler_ready_timeout_s}s: {type(exc).__name__}: {exc}"
                    ) from exc
                time.sleep(min(2.0 * attempt, 10.0))
        completion_ids = [int(value) for value in sequence.tokens]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(completion_ids),
            "total_tokens": len(prompt_ids) + len(completion_ids),
        }
        self.last_call_usage = dict(usage)
        # Token counts are exact here rather than provider-reported, which is
        # what compaction triggers on.
        return {
            "choices": [{"message": {"content": text, "tool_calls": None}}],
            "usage": usage,
        }

    def _hosted_sampler_sample(self) -> dict[str, Any]:
        """Sample the live hosted checkpoint through SamplerEndpoint."""
        from ..training_rollout import (
            ROLLOUT_ACTION_SCHEMA_VERSION,
            HostedSamplerClient,
            SamplerEndpoint,
            canonical_sha256,
        )
        from .app import allow_loopback_sampler

        target = self._inference_target or {}
        endpoint = str(target.get("provider_endpoint_id") or "").strip()
        auth_bearer = str(target.get("auth_bearer") or "").strip()
        run_id = str(target.get("run_id") or "").strip()
        checkpoint_id = str(
            target.get("checkpoint_id") or self.model or "hosted-sampler"
        ).strip()
        messages = list(self._messages)
        message_digest = canonical_sha256({"messages": messages})
        with HostedSamplerClient(
            SamplerEndpoint(
                endpoint,
                auth_bearer,
                str(target.get("connection_mode") or "keep_alive"),
            ),
            timeout_seconds=90.0,
            allow_loopback_http=allow_loopback_sampler(),
        ) as client:
            sampled = client.sample(
                {
                    "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
                    "job_id": target.get("run_id") or run_id,
                    "attempt_id": target.get("attempt_id"),
                    "rollout_id": target.get("rollout_id"),
                    "run_id": run_id,
                    "checkpoint_id": checkpoint_id,
                    "policy_version": checkpoint_id,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                idempotency_key=f"{target.get('rollout_id') or run_id}:{checkpoint_id}:{message_digest}",
            )
        usage = dict(sampled.usage)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = len(sampled.prompt_token_ids)
        if not isinstance(completion_tokens, int):
            completion_tokens = len(sampled.token_ids)
        accounted = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        self.last_call_usage = dict(accounted)
        self._last_training_action = {
            "schema_version": ROLLOUT_ACTION_SCHEMA_VERSION,
            "policy_version": checkpoint_id,
            "prompt_token_ids": list(sampled.prompt_token_ids),
            "token_ids": list(sampled.token_ids),
            "log_probs": list(sampled.log_probs),
        }
        return {
            "choices": [{"message": {"content": sampled.text, "tool_calls": None}}],
            "usage": accounted,
        }

    def _complete(
        self,
        api_key: str,
        valid: list[str],
        on_delta: DeltaCallback | None,
    ) -> dict[str, Any]:
        if self.provider == "tinker":
            del valid, on_delta  # no tool schema and no token stream on this path
            return self._tinker_sample(api_key)
        if self.api_family == RESPONSES:
            return self._responses_complete(api_key, valid, on_delta)
        payload = {
            "model": self.model,
            "messages": list(self._messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_actions",
                        "description": "Choose the next sequential Craftax actions.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": valid},
                                    "minItems": self.plan_min,
                                    "maxItems": self.plan_max,
                                }
                            },
                            "required": ["actions"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            # Muse currently accepts only `auto` at its provider boundary.
            # The single-tool system instruction plus JSON-text fallback keeps
            # this interoperable without pretending the gateway supports a
            # named forced choice.
            "tool_choice": "auto",
        }
        if self.provider == "openrouter":
            # `reasoning` is an OpenRouter extension, not part of the OpenAI
            # chat schema. Sending it to another provider is a request for a
            # field that provider never defined; a strict server rejects the
            # whole call with a 422 that reads as a policy failure. Found
            # against the local synth_mlx_rl service, which forbids extras.
            payload["reasoning"] = {"effort": self.reasoning_effort}
        request = urllib.request.Request(
            self.chat_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://usesynth.ai",
                "X-Title": "Synth Containers Craftax",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                headers = getattr(response, "headers", None)
                content_type = str(
                    headers.get("Content-Type") if headers is not None else ""
                ).lower()
                if "text/event-stream" in content_type:
                    return self._consume_sse(response, on_delta)
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"OpenRouter policy HTTP {exc.code}: {detail}") from exc
        stripped = raw.lstrip()
        if stripped.startswith("data:"):
            return self._consume_sse_text(raw, on_delta)
        return json.loads(raw)

    # ---- responses family ---------------------------------------------
    #
    # The Responses API is a peer surface, not a translation layer over chat.
    # Three concrete differences, each of which silently breaks a rollout if
    # missed: a tool is FLAT (`type`/`name`/`parameters` at the top level, not
    # nested under `function`), the cap is `max_output_tokens`, and usage is
    # reported as `input_tokens`/`output_tokens`. The last one matters most
    # here — compaction triggers on `prompt_tokens`, so an unmapped usage block
    # reads as a context that never grows and the transcript is never compacted.
    #
    # Everything is normalized back into the same chat-shaped dict the rest of
    # this class consumes, exactly as `_tinker_sample` already does, so parse,
    # retry, usage and compaction stay on one path.

    def _responses_input(self) -> list[dict[str, Any]]:
        """Translate chat `messages` into Responses `input` items."""
        items: list[dict[str, Any]] = []
        for message in self._messages:
            content = message.get("content")
            role = message.get("role")
            if isinstance(content, str):
                items.append({"role": role, "content": [{"type": "input_text", "text": content}]})
                continue
            parts: list[dict[str, Any]] = []
            for part in content or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        parts.append({"type": "input_image", "image_url": url})
                elif isinstance(part.get("text"), str):
                    parts.append({"type": "input_text", "text": part["text"]})
            items.append({"role": role, "content": parts})
        return items

    def _responses_complete(
        self,
        api_key: str,
        valid: list[str],
        on_delta: DeltaCallback | None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": self._responses_input(),
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "name": "choose_actions",
                    "description": "Choose the next sequential Craftax actions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string", "enum": valid},
                                "minItems": self.plan_min,
                                "maxItems": self.plan_max,
                            }
                        },
                        "required": ["actions"],
                        "additionalProperties": False,
                    },
                }
            ],
            "tool_choice": "auto",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.policy_snapshot_id:
            # D9: on this family the pinned snapshot rides in the header, so a
            # mid-episode sampler refresh cannot change what this call sampled.
            headers["X-Policy-Pin"] = self.policy_snapshot_id
        request = urllib.request.Request(
            self.chat_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                response_headers = getattr(response, "headers", None)
                content_type = str(
                    response_headers.get("Content-Type") if response_headers is not None else ""
                ).lower()
                if "text/event-stream" in content_type:
                    return self._consume_responses_sse(response, on_delta)
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"responses policy HTTP {exc.code}: {detail}") from exc
        stripped = raw.lstrip()
        if stripped.startswith("data:") or stripped.startswith("event:"):
            return self._consume_responses_sse_text(raw, on_delta)
        return self._normalize_responses_body(json.loads(raw))

    def _consume_responses_sse(
        self, response: Any, on_delta: DeltaCallback | None
    ) -> dict[str, Any]:
        chunks: list[bytes] = []
        while True:
            piece = response.read(256)
            if not piece:
                break
            chunks.append(piece if isinstance(piece, bytes) else str(piece).encode("utf-8"))
        return self._consume_responses_sse_text(
            b"".join(chunks).decode("utf-8", errors="replace"), on_delta
        )

    def _consume_responses_sse_text(
        self, raw: str, on_delta: DeltaCallback | None
    ) -> dict[str, Any]:
        assistant = ""
        reasoning = ""
        tool_arguments = ""
        usage: dict[str, Any] = {}
        completed: dict[str, Any] | None = None
        for block_text in raw.split("\n\n"):
            data_lines = [
                line[5:].strip() if line.startswith("data:") else line[5:].lstrip()
                for line in block_text.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            payload_text = "\n".join(data_lines).strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            kind = str(event.get("type") or "")
            delta = event.get("delta")
            if kind == "response.output_text.delta" and isinstance(delta, str):
                assistant += delta
                self._emit_delta(on_delta, "content", delta)
            elif kind == "response.function_call_arguments.delta" and isinstance(delta, str):
                tool_arguments += delta
                self._emit_delta(on_delta, "tool", delta)
            elif kind in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            } and isinstance(delta, str):
                reasoning += delta
                self._emit_delta(on_delta, "reasoning", delta)
            elif kind in {"response.completed", "response.incomplete"}:
                body = event.get("response")
                if isinstance(body, dict):
                    completed = body
                    if isinstance(body.get("usage"), dict):
                        usage = body["usage"]

        if completed is not None:
            # A server that streams no argument deltas but does emit a terminal
            # response must still produce a tool call, or the plan is lost.
            final = self._responses_output(completed)
            assistant = assistant or final["content"]
            reasoning = reasoning or final["reasoning"]
            tool_arguments = tool_arguments or final["tool_arguments"]
        return self._responses_shaped(assistant, reasoning, tool_arguments, usage)

    @staticmethod
    def _responses_output(body: dict[str, Any]) -> dict[str, str]:
        """Pull text, reasoning and function-call arguments out of `output`."""
        content = ""
        reasoning = ""
        tool_arguments = ""
        direct = body.get("output_text")
        if isinstance(direct, str):
            content = direct
        for item in body.get("output") or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "function_call":
                if str(item.get("name") or "") == "choose_actions":
                    arguments = item.get("arguments")
                    if isinstance(arguments, str):
                        tool_arguments += arguments
            elif item_type == "reasoning":
                for part in item.get("summary") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        reasoning += part["text"]
            elif item_type == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        if not direct and isinstance(part.get("text"), str):
                            content += part["text"]
        return {"content": content, "reasoning": reasoning, "tool_arguments": tool_arguments}

    def _normalize_responses_body(self, body: dict[str, Any]) -> dict[str, Any]:
        final = self._responses_output(body)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return self._responses_shaped(
            final["content"], final["reasoning"], final["tool_arguments"], usage
        )

    @staticmethod
    def _responses_usage(usage: dict[str, Any]) -> dict[str, Any]:
        """`input_tokens`/`output_tokens` -> the `prompt_tokens` names the rest
        of this class (and compaction in particular) reads."""
        if not usage:
            return {}
        mapped = dict(usage)
        if "prompt_tokens" not in mapped and usage.get("input_tokens") is not None:
            mapped["prompt_tokens"] = usage["input_tokens"]
        if "completion_tokens" not in mapped and usage.get("output_tokens") is not None:
            mapped["completion_tokens"] = usage["output_tokens"]
        if "total_tokens" not in mapped:
            prompt = mapped.get("prompt_tokens")
            completion = mapped.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                mapped["total_tokens"] = prompt + completion
        return mapped

    def _responses_shaped(
        self,
        content: str,
        reasoning: str,
        tool_arguments: str,
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning": reasoning,
                        "tool_calls": (
                            [
                                {
                                    "function": {
                                        "name": "choose_actions",
                                        "arguments": tool_arguments,
                                    }
                                }
                            ]
                            if tool_arguments
                            else []
                        ),
                    }
                }
            ],
            "usage": self._responses_usage(usage),
        }

    def _consume_sse(self, response: Any, on_delta: DeltaCallback | None) -> dict[str, Any]:
        chunks: list[bytes] = []
        while True:
            piece = response.read(256)
            if not piece:
                break
            chunks.append(piece if isinstance(piece, bytes) else str(piece).encode("utf-8"))
        return self._consume_sse_text(b"".join(chunks).decode("utf-8", errors="replace"), on_delta)

    def _consume_sse_text(self, raw: str, on_delta: DeltaCallback | None) -> dict[str, Any]:
        assistant = ""
        reasoning = ""
        tool_arguments = ""
        usage: dict[str, Any] = {}
        for block in raw.split("\n\n"):
            data_lines = [
                line[5:].strip() if line.startswith("data:") else line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            payload_text = "\n".join(data_lines).strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            try:
                chunk = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choice = (chunk.get("choices") or [{}])[0]
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            source = delta or message
            content_piece = self._message_text(source.get("content"))
            reasoning_piece = self._message_text(
                source.get("reasoning") or source.get("reasoning_content")
            )
            tool_piece = self._tool_arguments(source.get("tool_calls"))
            if content_piece:
                assistant += content_piece
                self._emit_delta(on_delta, "content", content_piece)
            if reasoning_piece:
                reasoning += reasoning_piece
                self._emit_delta(on_delta, "reasoning", reasoning_piece)
            if tool_piece:
                tool_arguments += tool_piece
                self._emit_delta(on_delta, "tool", tool_piece)
        return {
            "choices": [
                {
                    "message": {
                        "content": assistant,
                        "reasoning": reasoning,
                        "tool_calls": (
                            [
                                {
                                    "function": {
                                        "name": "choose_actions",
                                        "arguments": tool_arguments,
                                    }
                                }
                            ]
                            if tool_arguments
                            else []
                        ),
                    }
                }
            ],
            "usage": usage,
        }

    def _emit_delta(
        self,
        on_delta: DeltaCallback | None,
        channel: str,
        text: str,
    ) -> None:
        if on_delta is None or not text:
            return
        self._deltas_emitted += 1
        on_delta(
            {
                "delta": True,
                "channel": channel,
                "text": text,
                "call": self.calls + 1,
                "model": self.model,
                "provider": "openrouter",
            }
        )

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                current = self._usage.get(key)
                self._usage[key] = value + (current if isinstance(current, int) else 0)
                if key == "prompt_tokens":
                    # Compaction triggers on the real size of the last request,
                    # so record it per call rather than using the running total.
                    self.last_prompt_tokens = value
        provider_cost = usage.get("cost")
        if isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool):
            current_cost = self._usage.get("cost_usd")
            self._usage["cost_usd"] = float(provider_cost) + (
                float(current_cost) if isinstance(current_cost, (int, float)) else 0.0
            )

    @staticmethod
    def _tool_arguments(value: Any) -> str:
        if not isinstance(value, list):
            return ""
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            # Streaming continuations omit the function name after the first chunk.
            if name not in (None, "", "choose_actions"):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parts.append(arguments)
            elif isinstance(arguments, dict):
                parts.append(json.dumps(arguments, separators=(",", ":")))
        return "".join(parts)

    @staticmethod
    def _message_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
        return ""

    @staticmethod
    def _parse_actions(raw: str, valid: list[str]) -> list[str]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            value = json.loads(raw[start : end + 1]) if start >= 0 and end > start else {}
        requested = value.get("actions") if isinstance(value, dict) else []
        aliases = {
            "north": "up",
            "south": "down",
            "east": "right",
            "west": "left",
        }
        normalized = [
            aliases.get(str(action).strip().lower(), str(action).strip().lower())
            for action in requested or []
        ]
        actions = [action for action in normalized if action in valid][: OpenRouterReAct.plan_max]
        if not actions:
            raise RuntimeError("policy returned no valid Craftax actions")
        return actions
