"""Single-LLM-call policy: one request decides the whole episode's action list.

This is the cheapest rung of the policy ladder and the one that isolates the
environment from the harness. There is no loop, no history, and no tool: the
model sees the opening observation once and returns a full action sequence,
which the episode then executes without asking again.

Use it as the floor a ReAct or agentic harness must beat. If a fancier harness
does not beat one call, the harness is not what is producing the score.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

DeltaCallback = Callable[[dict[str, Any]], None]


class SingleCallPolicy:
    """One chat/responses call at episode open; every later ``plan`` replays it."""

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self.env_name = str(config.get("env_name") or "environment")
        self.objective = str(
            config.get("objective")
            or "Make measurable progress on the environment objective while staying alive."
        )
        self.model = str(config.get("model") or "meta/muse-spark-1.1")
        self.api = str(config.get("api") or "chat_completions")
        if self.api not in {"chat_completions", "responses"}:
            raise RuntimeError(f"single_call_unknown_api:{self.api}")
        default_base = (
            "https://api.openai.com/v1"
            if self.api == "responses"
            else "https://openrouter.ai/api/v1"
        )
        self.base_url = str(config.get("base_url") or default_base).rstrip("/")
        self.api_key_env = str(
            config.get("api_key_env")
            or ("OPENAI_API_KEY" if self.api == "responses" else "OPENROUTER_API_KEY")
        )
        self.max_tokens = min(max(int(config.get("max_tokens") or 1024), 64), 32000)
        # The floor rung is meant to be cheap. Reasoning budget is opt-in, and
        # when it is on it must not eat the whole completion cap before the
        # answer: an unparsed 2k-token ramble bills the same as a good plan.
        self.reasoning_effort = str(config.get("effort") or "low")
        self.horizon = min(max(int(config.get("horizon") or 32), 1), 512)
        self.calls = 0
        self._plan: list[str] = []
        self._cursor = 0
        self._usage: dict[str, Any] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self._last_trace: dict[str, Any] = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": "single_call",
            "kind": "single_llm_call",
            "config": self.config_id,
            "model": self.model,
            "api": self.api,
            "horizon": self.horizon,
            "reasoning_effort": self.reasoning_effort,
            "replans": False,
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {**self._usage, "calls": self.calls}

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": "synth.containers.single-call-checkpoint.v1",
            "plan": list(self._plan),
            "cursor": self._cursor,
            "calls": self.calls,
            "usage": dict(self._usage),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "synth.containers.single-call-checkpoint.v1":
            raise RuntimeError("unsupported policy checkpoint schema")
        plan = state.get("plan")
        if not isinstance(plan, list):
            raise RuntimeError("policy checkpoint omitted plan")
        self._plan = [str(item) for item in plan]
        self._cursor = int(state.get("cursor") or 0)
        self.calls = int(state.get("calls") or 0)
        usage = state.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("policy checkpoint omitted usage")
        self._usage = dict(usage)

    def plan(
        self, observation: dict[str, Any], on_delta: DeltaCallback | None = None
    ) -> list[str]:
        valid = [str(action) for action in observation.get("valid_actions") or ()]
        if not valid:
            raise RuntimeError("observation omitted valid_actions")
        if not self._plan:
            self._plan = self._decide(observation, valid, on_delta)
            self._cursor = 0
        if self._cursor >= len(self._plan):
            # The one call is spent. Hold the last legal action rather than
            # silently re-calling: a "single call" policy that calls twice is a
            # different policy and would misattribute the cost.
            return [self._plan[-1] if self._plan else valid[0]]
        chunk = self._plan[self._cursor : self._cursor + 1]
        self._cursor += 1
        return [action if action in valid else valid[0] for action in chunk]

    # ---------------------------------------------------------------- internal

    def _decide(
        self, observation: dict[str, Any], valid: list[str], on_delta: DeltaCallback | None
    ) -> list[str]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"paid policy requires {self.api_key_env}")
        observation_text = str(
            observation.get("observation_text") or observation.get("ascii") or ""
        )
        prompt = (
            f"{self.objective} You get exactly ONE call: return the full action "
            f"sequence to execute, up to {self.horizon} actions, from the legal list.\n\n"
            f"{observation_text}\n\nvalid_actions={json.dumps(valid)}\n"
            'Return JSON only: {"actions":["..."]}'
        )
        raw, usage = (
            self._responses_call(api_key, prompt)
            if self.api == "responses"
            else self._chat_call(api_key, prompt)
        )
        self.calls += 1
        for key, source in (
            ("prompt_tokens", "prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("total_tokens", "total_tokens"),
        ):
            value = usage.get(source)
            if isinstance(value, (int, float)):
                self._usage[key] = float(value)
        self._last_trace = {"raw": raw[:4000], "api": self.api}
        if on_delta is not None and raw:
            on_delta({"kind": "policy.text", "text": raw[:2000]})
        actions = _filter(_load_actions(raw), valid, self.horizon)
        if not actions:
            raise RuntimeError("policy returned no valid actions")
        return actions

    def _chat_call(self, api_key: str, prompt: str) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort and self.reasoning_effort != "none":
            payload["reasoning"] = {"effort": self.reasoning_effort}
        body = self._post("/chat/completions", api_key, payload)
        message = (body.get("choices") or [{}])[0].get("message") or {}
        return str(message.get("content") or ""), dict(body.get("usage") or {})

    def _responses_call(self, api_key: str, prompt: str) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "max_output_tokens": self.max_tokens,
            "store": False,
        }
        if self.reasoning_effort and self.reasoning_effort != "none":
            payload["reasoning"] = {"effort": self.reasoning_effort}
        body = self._post("/responses", api_key, payload)
        text = ""
        for item in body.get("output") or []:
            if isinstance(item, dict) and item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        text += part["text"]
        raw_usage = body.get("usage") or {}
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens"),
            "completion_tokens": raw_usage.get("output_tokens"),
            "total_tokens": raw_usage.get("total_tokens"),
        }
        return text, usage

    def _post(self, path: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"single_call_http_{exc.code}:{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("single_call_unreachable") from exc


def _load_actions(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("actions"), list):
        return [str(item) for item in payload["actions"]]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _filter(requested: list[str], valid: list[str], limit: int) -> list[str]:
    lowered = {action.lower(): action for action in valid}
    out: list[str] = []
    for item in requested:
        match = lowered.get(str(item).strip().lower())
        if match is not None:
            out.append(match)
    return out[:limit]
