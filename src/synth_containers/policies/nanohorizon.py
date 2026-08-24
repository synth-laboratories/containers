"""NanoHorizon policy harness: PUT ``policy.py`` + HTTP sampler, experiments ReAct.

Not stock ``react`` (JSON in assistant content). Loads a contest ``Policy`` from
the current policy revision and runs ``Policy.run_episode`` against gold HTTP.
Sampler is ``POST {base_url}/v1/sample`` with thinking and tools.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..gold_episode import _emit_obs, _frame_record
from ..gold_http import StepResult

PROTOCOL = "nanohorizon.policy.v1"
HARNESS = "nanohorizon"
TOOL_NAME = "craftax_interact"
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_BLOCK = re.compile(
    r"<function=(?P<name>[^\s>]+)>(?P<body>.*?)</function>", re.DOTALL
)
PARAM_BLOCK = re.compile(
    r"<parameter=(?P<key>[^\s>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL
)


def _thinking_text(text: str) -> str | None:
    stripped = TOOL_CALL_BLOCK.sub("", text)
    stripped = re.sub(r"<tool_call>.*\Z", "", stripped, flags=re.DOTALL)
    stripped = stripped.strip()
    return stripped or None


def _parse_qwen_tool_xml(text: str) -> dict[str, Any] | None:
    block = TOOL_CALL_BLOCK.search(text)
    if block:
        inner = block.group(1).strip()
    else:
        opened = re.search(r"<tool_call>\s*(.*)\Z", text, re.DOTALL)
        inner = opened.group(1).strip() if opened else ""
        if not inner and "<function=" in text:
            inner = text
    if not inner:
        return None
    if inner.startswith("{"):
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    fn = FUNCTION_BLOCK.search(inner)
    if not fn:
        return None
    arguments: dict[str, Any] = {}
    for param in PARAM_BLOCK.finditer(fn.group("body")):
        raw = param.group("value").strip()
        try:
            arguments[param.group("key")] = json.loads(raw)
        except json.JSONDecodeError:
            arguments[param.group("key")] = raw
    return {"name": fn.group("name"), "arguments": arguments}


def load_policy_class(code: bytes) -> type:
    digest = hashlib.sha256(code).hexdigest()[:12]
    root = Path(tempfile.mkdtemp(prefix="synth-nh-policy-"))
    path = root / "policy.py"
    path.write_bytes(code)
    name = f"nanohorizon_put_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_import_policy:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if getattr(module, "PROTOCOL", None) != PROTOCOL:
        raise RuntimeError(
            f"policy PROTOCOL must be {PROTOCOL!r}, got {getattr(module, 'PROTOCOL', None)!r}"
        )
    policy_cls = getattr(module, "Policy", None)
    if policy_cls is None:
        raise RuntimeError("policy.py must export Policy")
    return policy_cls


def _json_request(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read()
            headers = {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"sampler_http_{exc.code}:{detail}") from exc
    parsed = json.loads(raw.decode("utf-8") or "{}")
    if not isinstance(parsed, dict):
        raise RuntimeError("sampler_response_not_object")
    parsed["_headers"] = headers
    return parsed


def _wire_tool_calls(calls: object) -> list[dict[str, Any]]:
    wired: list[dict[str, Any]] = []
    if not isinstance(calls, list):
        return wired
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(
                arguments if arguments is not None else {}, separators=(",", ":")
            )
        wired.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(function.get("name") or TOOL_NAME),
                    "arguments": arguments,
                },
            }
        )
    return wired


def _wire_message(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or "user")
    row: dict[str, Any] = {"role": role}
    content = message.get("content")
    calls = _wire_tool_calls(message.get("tool_calls"))
    if calls:
        row["tool_calls"] = calls
        if content is not None:
            row["content"] = content
    else:
        row["content"] = "" if content is None else content
    if role == "tool":
        row["tool_call_id"] = str(message.get("tool_call_id") or "")
        if message.get("name"):
            row["name"] = message["name"]
    return row


def _message_from_sample(text: str) -> dict[str, Any]:
    """Lift Qwen's native tool XML into Chat Completions ``tool_calls``."""

    payload = _parse_qwen_tool_xml(text)
    thinking = _thinking_text(text)
    if not payload:
        return {"role": "assistant", "content": text or None}
    name = str(payload.get("name") or TOOL_NAME)
    args = payload.get("arguments") if "arguments" in payload else payload
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_raw": args}
    call = {
        "id": f"call_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, separators=(",", ":")),
        },
    }
    return {"role": "assistant", "content": thinking, "tool_calls": [call]}


def _normalize_base(url: str) -> str:
    base = str(url or "").rstrip("/")
    if not base:
        raise RuntimeError("nanohorizon sampler config requires base_url")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base.rstrip("/")


class HttpSampler:
    """Duck-typed complete / summarize over synth-mlx-rl ``/v1/sample``."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.base_url = _normalize_base(str(config.get("base_url") or ""))
        self.model = str(config.get("model") or "Qwen/Qwen3.5-0.8B")
        self.enable_thinking = bool(config.get("enable_thinking", True))
        self.thinking_budget = int(config.get("thinking_budget") or 256)
        self.answer_max_tokens = int(config.get("answer_max_tokens") or 128)
        # Default hot, and never via `or`: `temperature or 0.0` turns both an
        # omitted key AND an explicit 0.0 into greedy decoding, which is the
        # one setting measured to break this model -- it restates a sentence
        # until the budget is gone instead of emitting the tool call.
        temperature = config.get("temperature")
        self.temperature = float(1.0 if temperature is None else temperature)
        # Greedy decoding makes the 0.8B restate one sentence until the token
        # budget is gone, so it never reaches <tool_call>. Nucleus + top-k are
        # the only anti-repetition levers here: SampleRequest has no
        # repetition_penalty.
        # `or` would swallow top_k=0, which is how you ask for "no top-k".
        top_p = config.get("top_p")
        top_k = config.get("top_k")
        self.top_p = float(0.95 if top_p is None else top_p)
        self.top_k = int(20 if top_k is None else top_k)
        self.timeout = float(config.get("timeout_seconds") or 120.0)
        self.snapshot = str(config.get("policy_snapshot_id") or "").strip() or None

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        max_tokens = self.thinking_budget + self.answer_max_tokens
        payload: dict[str, Any] = {
            "messages": [_wire_message(row) for row in messages],
            "num_samples": 1,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "enable_thinking": self.enable_thinking,
            "seed": int(seed),
            "record_rollout_logprobs": True,
            "api_family": "chat_completions",
            "stop": ["</tool_call>"],
        }
        if tools:
            payload["tools"] = tools
        if self.snapshot:
            payload["policy_snapshot_id"] = self.snapshot
        body = _json_request(f"{self.base_url}/v1/sample", payload, timeout=self.timeout)
        samples = body.get("samples") or []
        sample = samples[0] if samples and isinstance(samples[0], dict) else {}
        proxy = str(
            sample.get("proxy_request_id")
            or (body.get("_headers") or {}).get("x-proxy-request-id")
            or ""
        )
        text = str(sample.get("text") or "")
        return {
            "text": text,
            "message": _message_from_sample(text),
            "proxy_request_id": proxy,
            "prompt_tokens": len(sample.get("prompt_token_ids") or []),
            "completion_tokens": len(sample.get("completion_token_ids") or []),
            "prompt_token_ids": list(sample.get("prompt_token_ids") or []),
            "completion_token_ids": list(sample.get("completion_token_ids") or []),
        }

    def summarize(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "messages": [_wire_message(row) for row in messages],
            "num_samples": 1,
            "max_tokens": int(max_tokens),
            # Same hazard as the action call: a looping summary silently
            # corrupts the compacted state every later turn is planned from.
            "temperature": self.temperature,
            "enable_thinking": False,
            "record_rollout_logprobs": False,
        }
        if self.snapshot:
            payload["policy_snapshot_id"] = self.snapshot
        body = _json_request(f"{self.base_url}/v1/sample", payload, timeout=self.timeout)
        samples = body.get("samples") or []
        sample = samples[0] if samples and isinstance(samples[0], dict) else {}
        return str(sample.get("text") or "").strip()


class NanoHorizonPlanner:
    """Runs the PUT ``Policy.run_episode`` against a gold world."""

    def __init__(self, *, config_id: str, config: dict[str, Any], code: bytes) -> None:
        self.config_id = config_id
        self.config = dict(config)
        self.sampler = HttpSampler(self.config)
        self.policy = load_policy_class(code)(**self.config)
        self._calls = 0
        self._last_events: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = {}
        self._usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": HARNESS,
            "kind": "nanohorizon_react",
            "config": self.config_id,
            "model": self.sampler.model,
            "enable_thinking": self.sampler.enable_thinking,
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {**self._usage, "calls": self._calls}

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def plan(self, observation: dict[str, Any], on_delta: Any = None) -> list[str]:
        del observation, on_delta
        raise RuntimeError("nanohorizon harness does not use gold_episode.plan(); run() instead")

    def run(
        self,
        *,
        world: Any,
        log: Any,
        seed: int,
        max_steps: int,
        omit_reward: bool = False,
    ) -> dict[str, Any]:
        log.append("env.episode.opened", {"seed": seed, "max_steps": max_steps})
        result: StepResult = world.reset(seed, max_steps=max_steps)
        events = world.drain_native_events()
        for event in events:
            kind = event.get("kind")
            if isinstance(kind, str) and kind:
                log.append(kind, {key: value for key, value in event.items() if key != "kind"})
        frames = [_frame_record(result, _emit_obs(log, result, seed=seed))]
        log.append("policy.session.opened", dict(self.metadata()))
        signals: list[float | None] = []
        executed: list[str] = []

        def step_policy(action: str) -> dict[str, Any]:
            nonlocal result
            log.append("span.step.opened", {"action": action, "step": result.env_steps})
            result = world.step(action)
            events = world.drain_native_events()
            self._last_events = events
            executed.append(action)
            for event in events:
                kind = event.get("kind")
                if isinstance(kind, str) and kind:
                    log.append(kind, {key: value for key, value in event.items() if key != "kind"})
            log.append("action", {"step": result.env_steps, "action": action})
            value: float | None = result.reward
            if omit_reward and result.env_steps == 2:
                value = None
            signals.append(value)
            log.append(
                "reward_signal",
                {"step": result.env_steps, "value": value, "authority": "environment"},
            )
            frames.append(_frame_record(result, _emit_obs(log, result, seed=seed)))
            log.append("span.step.closed", {"action": action, "step": result.env_steps})
            observation = dict(result.observation)
            if result.done:
                observation["done"] = True
            return observation

        def drain() -> list[dict[str, Any]]:
            return list(self._last_events)

        def sample(messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            completion = self.sampler.complete(messages, **kwargs)
            self._calls += 1
            self._usage["prompt_tokens"] = int(self._usage["prompt_tokens"] or 0) + int(
                completion.get("prompt_tokens") or 0
            )
            self._usage["completion_tokens"] = int(self._usage["completion_tokens"] or 0) + int(
                completion.get("completion_tokens") or 0
            )
            self._usage["total_tokens"] = int(self._usage["prompt_tokens"] or 0) + int(
                self._usage["completion_tokens"] or 0
            )
            self._usage["calls"] = self._calls
            self._last_trace = {
                "proxy_request_id": completion.get("proxy_request_id"),
                "prompt_tokens": completion.get("prompt_tokens"),
                "text": completion.get("text"),
            }
            log.append("span.policy.opened", {"harness": HARNESS, "call": self._calls})
            log.append("span.policy.data", dict(self._last_trace))
            return completion

        outcome = self.policy.run_episode(
            opening={**result.observation, **({"done": True} if result.done else {})},
            step=step_policy,
            drain=drain,
            sample=sample,
            summarize=self.sampler.summarize,
            is_done=lambda readout: bool(
                (readout.get("private") or {}).get("terminated")
                or (readout.get("private") or {}).get("truncated")
                or readout.get("done")
            ),
            max_steps=max_steps,
        )
        for row in outcome.get("journal") or []:
            kind = str(row.get("kind") or "policy.event")
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
            log.append(kind, payload)
        log.append("span.policy.plan", {"actions": executed, "length": len(executed)})
        log.append("span.policy.closed", {"length": len(executed)})
        log.append(
            "policy.session.closed",
            {
                "calls": self._calls,
                "proxy_request_ids": outcome.get("proxy_request_ids"),
                "achievements": outcome.get("achievements"),
                "reward": outcome.get("reward"),
            },
        )
        log.append("env.episode.closed", {"status": "completed", "steps": result.env_steps})
        log.append("status", {"status": "completed", "steps": result.env_steps})
        log.append("capture.high_water", {"high_water": log.high_water})
        log.append("capture.closed", {"high_water": log.high_water})
        log.mark_closed()
        self._last_trace = {
            **self._last_trace,
            "proxy_request_ids": outcome.get("proxy_request_ids"),
            "achievements": outcome.get("achievements"),
            "reward": outcome.get("reward"),
        }
        return {
            "reward_signals": signals,
            "actions": executed,
            "usage": self.usage(),
            "frame_digest": result.frame_digest,
            "steps": result.env_steps,
            "frames": frames,
            "scheduled_checkpoints": [],
            "episode": outcome,
        }


def build_planner(*, config_id: str, config: dict[str, Any], code: bytes) -> NanoHorizonPlanner:
    if not code:
        raise RuntimeError("nanohorizon_missing_policy_code")
    return NanoHorizonPlanner(config_id=config_id, config=config, code=code)
