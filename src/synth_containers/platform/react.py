"""ReAct harness over a Craftax-shaped world.

`craftax_engine` uses a scripted planner (no model). That is engine-acceptance,
not a Luna eval. `craftax_react --paid` may call a chat model when a key exists.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .craftax_world import ACTIONS

DeltaCallback = Callable[[dict[str, Any]], None]


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
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "calls": self.calls,
        }

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
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None, "calls": 0}

    def plan(self, observation: dict[str, Any], on_delta: DeltaCallback | None = None) -> list[str]:
        del on_delta
        valid = list(observation.get("valid_actions") or ACTIONS)
        return [self._rng.choice(valid)]


class OpenRouterReAct:
    """Paid ReAct planner for reviewed provider configs.

    The key is read only at request time and never enters metadata or the event
    log. History is the same session across `plan()` calls; `compact_every`
    (default 16) is a harness facet. Token deltas are forwarded only when the
    provider actually streams non-empty chunks — Luna often returns empty
    reasoning, and that absence is left blank.
    """

    plan_min = 5
    plan_max = 20
    compact_keep_turns = 2

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self.environment_name = str(config.get("environment_name") or "Craftax")
        self.model = str(config.get("model") or "meta/muse-spark-1.1")
        self.reasoning_effort = str(config.get("effort") or "medium")
        self.base_url = str(config.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key_env = str(config.get("api_key_env") or "OPENROUTER_API_KEY")
        self.max_tokens = min(max(int(config.get("max_tokens") or 768), 64), 4096)
        self.parse_retries = min(max(int(config.get("parse_retries") or 0), 0), 2)
        self.transport_retries = min(max(int(config.get("transport_retries") or 2), 0), 4)
        self.compact_every = min(max(int(config.get("compact_every") or 16), 1), 64)
        self.compaction_mode = str(config.get("compaction_mode") or "turn_count")
        self.context_token_budget = min(
            max(int(config.get("context_token_budget") or 16_000), 2_000), 400_000
        )
        self.compact_at = min(max(float(config.get("compact_at") or 0.7), 0.1), 0.95)
        self.compact_threshold = int(self.context_token_budget * self.compact_at)
        self.observation_mode = str(config.get("observation_mode") or "text")
        self.keep_recent_frames = min(
            max(int(config.get("keep_recent_frames") or 2), 1), 16
        )
        self.calls = 0
        self._compact_count = 0
        self._last_prompt_tokens = 0
        self._deltas_emitted = 0
        self._usage: dict[str, int | float | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
        }
        self._last_trace: dict[str, Any] = {}
        system_prompt = str(config.get("system_prompt") or config.get("react_system_prompt") or "").strip()
        if not system_prompt:
            system_prompt = (
                f"You are a careful {self.environment_name} ReAct policy. You must call the "
                "choose_actions tool exactly once; do not answer with prose."
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
            "transport_retries": self.transport_retries,
            "compact_every": self.compact_every,
            "compaction_mode": self.compaction_mode,
            "compact_threshold": self.compact_threshold,
            "observation_mode": self.observation_mode,
            "keep_recent_frames": self.keep_recent_frames,
            "token_trace": "derived",
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {**self._usage, "calls": self.calls}

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def checkpoint_state(self) -> dict[str, Any]:
        """Secret-free policy-session state paired with an environment snapshot."""
        return {
            "schema_version": "synth.containers.react-checkpoint.v1",
            "messages": json.loads(json.dumps(self._messages)),
            "calls": self.calls,
            "compact_count": self._compact_count,
            "last_prompt_tokens": self._last_prompt_tokens,
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
        self._last_prompt_tokens = int(state.get("last_prompt_tokens") or 0)
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
        if not api_key:
            raise RuntimeError(f"paid Craftax policy requires {self.api_key_env}")
        valid = [str(action) for action in observation.get("valid_actions") or []]
        if not valid:
            raise RuntimeError("Craftax observation omitted valid_actions")
        self._maybe_compact(on_delta)
        self._messages.append(
            {"role": "user", "content": self._observation_content(observation, valid)}
        )
        self._expire_old_frames()
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
            self._accumulate_usage(usage)
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
            "compact_every": self.compact_every,
            "compact_count": self._compact_count,
            "compaction_mode": self.compaction_mode,
            "compact_threshold": self.compact_threshold,
            "last_prompt_tokens": self._last_prompt_tokens,
            "observation_mode": self.observation_mode,
            "history_turns": self.calls,
            "deltas_emitted": call_deltas,
            "token_trace": "derived" if call_deltas else None,
            "usage": {
                **{
                    key: usage.get(key)
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
                "cost_usd": provider_cost,
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
            f"Make measurable {self.environment_name} objective progress while staying alive. "
            f"Choose {self.plan_min}-{min(self.plan_max, len(valid))} sequential actions from the exact legal list.\n\n"
            f"{observation_text}\n\nvalid_actions={json.dumps(valid)}\n"
            'Return JSON only: {"actions":["do","right"]}'
        )

    def _observation_content(
        self, observation: dict[str, Any], valid: list[str]
    ) -> str | list[dict[str, Any]]:
        prompt = self._observation_prompt(observation, valid)
        image_url = observation.get("image_data_url")
        if self.observation_mode not in {"image", "both"} or not isinstance(image_url, str):
            return prompt
        parts: list[dict[str, Any]] = []
        if self.observation_mode == "both":
            parts.append({"type": "text", "text": prompt})
        else:
            parts.append(
                {
                    "type": "text",
                    "text": (
                        f"Choose valid {self.environment_name} actions from "
                        f"valid_actions={json.dumps(valid)}."
                    ),
                }
            )
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
        return parts

    def _expire_old_frames(self) -> None:
        image_messages = [
            index
            for index, message in enumerate(self._messages)
            if isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in message["content"]
            )
        ]
        for index in image_messages[: -self.keep_recent_frames]:
            parts = self._messages[index]["content"]
            self._messages[index]["content"] = [
                {"type": "text", "text": "<older frame omitted>"}
                if isinstance(part, dict) and part.get("type") == "image_url"
                else part
                for part in parts
            ]

    def _maybe_compact(self, on_delta: DeltaCallback | None) -> None:
        token_trigger = (
            self.compaction_mode == "token_budget"
            and self._last_prompt_tokens > self.compact_threshold
        )
        turn_trigger = (
            self.compaction_mode != "token_budget"
            and self.calls > 0
            and self.calls % self.compact_every == 0
        )
        if not token_trigger and not turn_trigger:
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
        minimum_pairs = 3 if token_trigger else self.compact_every
        if len(pairs) < minimum_pairs:
            return
        keep = pairs[-self.compact_keep_turns :]
        dropped = pairs[: -self.compact_keep_turns]
        dropped_payloads = [
            str(assistant.get("content") or "")[:200]
            for _, assistant in dropped
            if assistant is not None
        ]
        compact = {
            "role": "user",
            "content": (
                f"[compacted {len(dropped)} earlier ReAct turns; "
                + (
                    f"prompt_tokens={self._last_prompt_tokens} exceeded "
                    f"threshold={self.compact_threshold}. "
                    if token_trigger
                    else f"compact_every={self.compact_every}. "
                )
                + "Mechanical history compact, "
                "not a model-authored summary. Dropped assistant payloads "
                f"(truncated): {json.dumps(dropped_payloads[-8:])}]"
            ),
        }
        rebuilt: list[dict[str, Any]] = [self._messages[0], compact]
        for user, assistant in keep:
            rebuilt.append(user)
            if assistant is not None:
                rebuilt.append(assistant)
        self._messages = rebuilt
        self._compact_count += 1
        if on_delta is not None:
            on_delta(
                {
                    "delta": False,
                    "channel": "compact",
                    "compact_every": self.compact_every,
                    "compaction_mode": self.compaction_mode,
                    "prompt_tokens": self._last_prompt_tokens,
                    "compact_threshold": self.compact_threshold,
                    "compact_count": self._compact_count,
                    "dropped_turns": len(dropped),
                    "kept_turns": len(keep),
                    "call": self.calls,
                    "model": self.model,
                    "provider": "openrouter",
                }
            )

    def _complete(
        self,
        api_key: str,
        valid: list[str],
        on_delta: DeltaCallback | None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": list(self._messages),
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "reasoning": {"effort": self.reasoning_effort},
            "stream": True,
            "stream_options": {"include_usage": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_actions",
                        "description": f"Choose the next sequential {self.environment_name} actions.",
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
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://usesynth.ai",
                "X-Title": "Synth Containers Craftax",
            },
            method="POST",
        )
        raw = ""
        for attempt in range(self.transport_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    headers = getattr(response, "headers", None)
                    content_type = str(
                        headers.get("Content-Type") if headers is not None else ""
                    ).lower()
                    if "text/event-stream" in content_type:
                        return self._consume_sse(response, on_delta)
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"OpenRouter policy HTTP {exc.code}: {detail}") from exc
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt >= self.transport_retries:
                    raise RuntimeError(
                        f"OpenRouter policy transport failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                time.sleep(0.25 * (2**attempt))
        stripped = raw.lstrip()
        if stripped.startswith("data:"):
            return self._consume_sse_text(raw, on_delta)
        return json.loads(raw)

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
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            self._last_prompt_tokens = prompt_tokens
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                current = self._usage.get(key)
                self._usage[key] = value + (current if isinstance(current, int) else 0)
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
