"""ReAct planner over the OpenAI **Responses** API (not chat completions).

Two things make this a distinct harness rather than a base-url swap:

* The request carries ``input`` items and a ``tools`` array in Responses shape,
  and multi-turn state is carried by ``previous_response_id`` rather than by
  replaying a message list. That is cheaper and it is what the v5 Responses
  trace adapter (``tracing.adapters.openai_responses``) knows how to read.
* Reasoning models return ``output`` items of several types; the action lives in
  a ``function_call`` item, with a JSON-text fallback for gateways that decline
  to force a tool.

The key is read at request time only and never enters metadata or the log.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

DeltaCallback = Callable[[dict[str, Any]], None]

TOOL_NAME = "choose_actions"


class ResponsesReAct:
    """Responses-API ReAct planner. One tool call per environment observation."""

    plan_min = 1
    plan_max = 20

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self.env_name = str(config.get("env_name") or "environment")
        self.objective = str(
            config.get("objective")
            or "Make measurable progress on the environment objective while staying alive."
        )
        self.model = str(config.get("model") or "gpt-5.6")
        self.base_url = str(config.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
        self.reasoning_effort = str(config.get("effort") or "medium")
        self.max_output_tokens = min(max(int(config.get("max_tokens") or 2048), 64), 32000)
        self.plan_min = min(max(int(config.get("plan_min") or 1), 1), 20)
        self.plan_max = min(max(int(config.get("plan_max") or 5), self.plan_min), 20)
        self.store = bool(config.get("store", False))
        self.calls = 0
        self._previous_response_id: str | None = None
        self._usage: dict[str, Any] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reasoning_tokens": None,
        }
        self._last_trace: dict[str, Any] = {}
        self._system = str(config.get("system_prompt") or "").strip() or (
            f"You are a careful {self.env_name} ReAct policy. Call the "
            f"{TOOL_NAME} tool exactly once per turn; never answer with prose."
        )

    # ------------------------------------------------------------------ facets

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": "responses_react",
            "kind": "openai_responses_react",
            "config": self.config_id,
            "model": self.model,
            "provider": "openai_responses",
            "api": "responses",
            "reasoning_effort": self.reasoning_effort,
            "plan_min": self.plan_min,
            "plan_max": self.plan_max,
            "state_carrier": "previous_response_id",
            "token_trace": "derived",
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {**self._usage, "calls": self.calls}

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": "synth.containers.responses-react-checkpoint.v1",
            "previous_response_id": self._previous_response_id,
            "calls": self.calls,
            "usage": dict(self._usage),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "synth.containers.responses-react-checkpoint.v1":
            raise RuntimeError("unsupported policy checkpoint schema")
        if not self.store:
            # Without server-side storage there is no response chain to resume;
            # refusing beats silently branching from an unrelated context.
            raise RuntimeError("responses_react_restore_requires_store")
        self._previous_response_id = state.get("previous_response_id") or None
        self.calls = int(state.get("calls") or 0)
        usage = state.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("policy checkpoint omitted usage")
        self._usage = dict(usage)

    # -------------------------------------------------------------------- plan

    def plan(
        self, observation: dict[str, Any], on_delta: DeltaCallback | None = None
    ) -> list[str]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"paid policy requires {self.api_key_env}")
        valid = [str(action) for action in observation.get("valid_actions") or ()]
        if not valid:
            raise RuntimeError("observation omitted valid_actions")

        body = self._respond(api_key, self._prompt(observation, valid), valid)
        self.calls += 1
        self._accumulate_usage(body.get("usage") or {})
        if self.store:
            self._previous_response_id = str(body.get("id") or "") or None

        actions, source, raw = _actions_from_output(body.get("output") or [], valid, self.plan_max)
        self._last_trace = {
            "response_id": body.get("id"),
            "action_authority": source,
            "raw": raw[:2000],
            "status": body.get("status"),
        }
        if on_delta is not None and raw:
            on_delta({"kind": "policy.text", "text": raw[:2000]})
        if not actions:
            raise RuntimeError("policy returned no valid actions")
        return actions[: max(self.plan_max, 1)]

    # ---------------------------------------------------------------- internal

    def _prompt(self, observation: dict[str, Any], valid: list[str]) -> str:
        observation_text = str(
            observation.get("observation_text") or observation.get("ascii") or ""
        )
        return (
            f"{self.objective} "
            f"Choose {self.plan_min}-{min(self.plan_max, len(valid))} sequential actions "
            "from the exact legal list.\n\n"
            f"{observation_text}\n\nvalid_actions={json.dumps(valid)}\n"
            'If you cannot call the tool, return JSON only: {"actions":["..."]}'
        )

    def _tools(self, valid: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": TOOL_NAME,
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
            }
        ]

    def _respond(self, api_key: str, prompt: str, valid: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            "tools": self._tools(valid),
            "tool_choice": "auto",
            "max_output_tokens": self.max_output_tokens,
            "store": self.store,
        }
        if self._previous_response_id is not None:
            payload["previous_response_id"] = self._previous_response_id
        else:
            payload["instructions"] = self._system
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"responses_api_http_{exc.code}:{detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("responses_api_unreachable") from exc

    def _accumulate_usage(self, usage: dict[str, Any]) -> None:
        mapping = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        details = usage.get("output_tokens_details")
        if isinstance(details, dict):
            mapping["reasoning_tokens"] = details.get("reasoning_tokens")
        for key, value in mapping.items():
            if isinstance(value, (int, float)):
                current = self._usage.get(key)
                self._usage[key] = float(value) + (current or 0)


def _actions_from_output(
    output: list[Any], valid: list[str], limit: int
) -> tuple[list[str], str, str]:
    """Pull actions from a Responses ``output`` array: tool call first, then text."""

    raw_text = ""
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and item.get("name") == TOOL_NAME:
            arguments = item.get("arguments")
            parsed = _load_actions(arguments if isinstance(arguments, str) else "")
            actions = _filter(parsed, valid, limit)
            if actions:
                return actions, "policy", str(arguments or "")
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    raw_text += part["text"]
    actions = _filter(_load_actions(raw_text), valid, limit)
    return actions, ("policy_text" if actions else "none"), raw_text


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
    if isinstance(payload, dict):
        actions = payload.get("actions")
        if isinstance(actions, list):
            return [str(item) for item in actions]
        if isinstance(payload.get("action"), str):
            return [str(payload["action"])]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _filter(requested: list[str], valid: list[str], limit: int) -> list[str]:
    lowered = {action.lower(): action for action in valid}
    actions = []
    for item in requested:
        match = lowered.get(str(item).strip().lower())
        if match is not None:
            actions.append(match)
    return actions[:limit]
