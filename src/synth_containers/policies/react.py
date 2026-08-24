"""ReAct planners over gold-engine observations. Game-agnostic.

Every planner here answers the same duck contract the episode loop calls:

    plan(observation, on_delta) -> list[str]      # actions to execute
    usage() -> dict                                # tokens / calls
    metadata() -> dict                             # harness identity for the trace

Legal actions always come from the observation (`valid_actions`), never from a
per-game constant, so one planner serves Craftax, Rogue, and DungeonGrid alike.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


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
        valid = [str(action) for action in observation.get("valid_actions") or ()]
        if not valid:
            raise RuntimeError("scripted_react_requires_valid_actions")
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
        filler = [action for action in ("do", "sleep", "noop") if action in valid] or valid[:1]
        plan = (moves + filler * self.plan_min)[: self.plan_min]
        while len(plan) < self.plan_min:
            plan.append(filler[0])
        return [item if item in valid else filler[0] for item in plan]


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
        valid = [str(action) for action in observation.get("valid_actions") or ()]
        if not valid:
            raise RuntimeError("uniform_policy_requires_valid_actions")
        return [self._rng.choice(valid)]


class TinkerReAct:
    """ReAct planner backed by an immutable Tinker sampler checkpoint."""

    plan_min = 3
    plan_max = 4

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        target = config.get("inference_target")
        if not isinstance(target, dict):
            raise RuntimeError("tinker_inference_target_missing")
        self.config_id = config_id
        self.env_name = str(config.get("env_name") or "environment")
        self.model_path = str(target.get("provider_endpoint_id") or "").strip()
        self.base_model = str(target.get("base_model") or "").strip()
        if not self.model_path.startswith("tinker://"):
            raise RuntimeError("tinker_sampler_path_missing")
        if not self.base_model:
            raise RuntimeError("tinker_base_model_missing")
        self.calls = 0
        self._total_tokens = 0
        self._tokenizer: Any = None
        self._sampling_client: Any = None

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": "react",
            "kind": "tinker_react",
            "config": self.config_id,
            "model": self.base_model,
            "provider": "tinker",
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": self._total_tokens or None,
            "calls": self.calls,
        }

    def plan(self, observation: dict[str, Any], on_delta: DeltaCallback | None = None) -> list[str]:
        del on_delta
        tokenizer, tinker, client = self._runtime()
        valid = [str(action) for action in observation.get("valid_actions") or ()]
        if not valid:
            raise RuntimeError("observation omitted valid_actions")
        system = (
            f"You are a {self.env_name} teacher policy collecting high-signal long-horizon "
            "trajectories. Return a short useful macro-action with 3-4 legal actions. "
            "Use movement to explore when nothing useful is adjacent."
        )
        user = (
            f"Current {self.env_name} observation:\n{json.dumps(observation, sort_keys=True)}\n\n"
            "Plan a short useful macro-action. Return exactly 4 actions unless the "
            f"environment is already done.\nvalid_actions={json.dumps(valid)}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = list(map(int, tokenizer(prompt, add_special_tokens=False)["input_ids"]))
        model_input_cls = getattr(tinker, "ModelInput", None) or tinker.types.ModelInput
        try:
            model_input = model_input_cls.from_ints(tokens=prompt_ids)
        except TypeError:
            model_input = model_input_cls.from_ints(prompt_ids)
        result = client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(max_tokens=256, temperature=0.0),
        ).result()
        sequence = result.sequences[0]
        token_ids = list(map(int, sequence.tokens))
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        self.calls += 1
        self._total_tokens += len(prompt_ids) + len(token_ids)
        actions = []
        for candidate in re.findall(r'"(?:actions_list|actions)"\s*:\s*\[([^]]+)\]', text):
            actions.extend(re.findall(r'"([A-Za-z0-9_\-]+)"', candidate))
        if not actions:
            actions = [word for word in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", text) if word in valid]
        normalized = [item for item in actions if item in valid][: self.plan_max]
        if not normalized:
            raise RuntimeError("tinker_craftax_actions_missing")
        return normalized

    def _runtime(self) -> tuple[Any, Any, Any]:
        try:
            import tinker
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("tinker_sdk_missing") from exc
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self._sampling_client is None:
            self._sampling_client = tinker.ServiceClient().create_sampling_client(
                model_path=self.model_path
            )
        return self._tokenizer, tinker, self._sampling_client


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
        self.env_name = str(config.get("env_name") or "environment")
        self.objective = str(
            config.get("objective")
            or "Make measurable progress on the environment objective while staying alive."
        )
        self.model = str(config.get("model") or "meta/muse-spark-1.1")
        self.reasoning_effort = str(config.get("effort") or "medium")
        self.base_url = str(config.get("base_url") or "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key_env = str(config.get("api_key_env") or "OPENROUTER_API_KEY")
        self.max_tokens = min(max(int(config.get("max_tokens") or 768), 64), 2048)
        self.parse_retries = min(max(int(config.get("parse_retries") or 0), 0), 2)
        self.compact_every = min(max(int(config.get("compact_every") or 16), 1), 64)
        # How the NEXT observation is carried back to the model.
        #
        #   "user" (default)  a plain user turn, as this harness has always done
        #   "tool"            a tool result answering the model's own call
        #
        # "tool" is the protocol-correct shape when the model actually emitted a
        # tool call, and it is what lets a training proxy keep the sampled tokens
        # as a prefix instead of re-rendering the whole history each turn. It is
        # NOT the default: this harness is shared by every image and provider,
        # and changing the message shape under a running eval lane would move its
        # numbers for reasons that have nothing to do with the model.
        observation_role = str(config.get("observation_role") or "user").strip().lower()
        if observation_role not in {"user", "tool"}:
            raise RuntimeError(f"unsupported observation_role: {observation_role!r}")
        self.observation_role = observation_role
        self._pending_tool_call_id: str | None = None
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
        system_prompt = str(config.get("system_prompt") or config.get("react_system_prompt") or "").strip()
        if not system_prompt:
            system_prompt = (
                "You are a careful ReAct policy for this environment. You must call the "
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
            "compact_every": self.compact_every,
            "observation_role": self.observation_role,
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
            "pending_tool_call_id": self._pending_tool_call_id,
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
        self._pending_tool_call_id = state.get("pending_tool_call_id") or None
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
        if not api_key:
            raise RuntimeError(f"paid policy requires {self.api_key_env}")
        valid = [str(action) for action in observation.get("valid_actions") or []]
        if not valid:
            raise RuntimeError("observation omitted valid_actions")
        self._maybe_compact(on_delta)
        prompt_text = self._observation_prompt(observation, valid)
        if self.observation_role == "tool" and self._pending_tool_call_id:
            # Answer the call the model actually made. Only reachable when the
            # previous turn produced tool_calls; the JSON-content fallback has
            # no call to answer and stays a user turn.
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._pending_tool_call_id,
                    "name": "choose_actions",
                    "content": prompt_text,
                }
            )
        else:
            self._messages.append({"role": "user", "content": prompt_text})
        self._pending_tool_call_id = None
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
        sampled_call = _first_tool_call(message) if self.observation_role == "tool" else None
        if sampled_call is not None:
            # Echo the call structurally rather than flattening it into content,
            # so the next turn is a real tool cycle.
            self._messages.append(
                {"role": "assistant", "content": assistant, "tool_calls": [sampled_call]}
            )
            self._pending_tool_call_id = str(sampled_call.get("id") or "")
        else:
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
            f"{self.objective} "
            f"Choose {self.plan_min}-{min(self.plan_max, len(valid))} sequential actions from the exact legal list.\n\n"
            f"{observation_text}\n\nvalid_actions={json.dumps(valid)}\n"
            'Return JSON only: {"actions":["do","right"]}'
        )

    def _maybe_compact(self, on_delta: DeltaCallback | None) -> None:
        if self.calls == 0 or self.calls % self.compact_every != 0:
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
        if len(pairs) < self.compact_every:
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
                f"compact_every={self.compact_every}. Mechanical history compact, "
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
                        "description": "Choose the next sequential environment actions.",
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
                "X-Title": "Synth Containers",
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
            raise RuntimeError("policy returned no valid actions")
        return actions


def _first_tool_call(message: dict[str, Any]) -> dict[str, Any] | None:
    """The model's own tool call, if it made one.

    Returns None when the model answered with plain JSON content instead — that
    path has no call to answer, so the observation stays a user turn.
    """

    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None
    call = calls[0]
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict) or not function.get("name"):
        return None
    return {
        "id": str(call.get("id") or "call_0"),
        "type": "function",
        "function": {
            "name": str(function.get("name")),
            "arguments": function.get("arguments") or "{}",
        },
    }
