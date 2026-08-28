"""NanoHorizon policy harness: PUT ``policy.py`` + HTTP sampler, experiments ReAct.

Not stock ``react`` (JSON in assistant content). Loads a contest ``Policy`` from
the current policy revision and runs ``Policy.run_episode`` against gold HTTP.
Sampler is ``POST {base_url}/v1/sample`` with thinking and tools.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..gold_episode import _emit_obs, _frame_record
from ..gold_http import StepResult
from ..gen_ai import copy_observation, request_observation as gen_ai_request_observation

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


class _RequestPace:
    """Process-wide spacing so parallel rollouts share one OpenRouter budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_ok = 0.0
        self._cool_until = 0.0

    def wait(self, min_interval: float) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                delay = max(0.0, max(self._next_ok, self._cool_until) - now)
                if delay <= 0:
                    gap = max(0.0, float(min_interval))
                    self._next_ok = now + gap
                    return
            time.sleep(min(delay, 1.0))

    def cool(self, seconds: float) -> None:
        wait = max(0.0, float(seconds))
        if wait <= 0:
            return
        with self._lock:
            until = time.monotonic() + wait
            if until > self._cool_until:
                self._cool_until = until


_PACE = _RequestPace()
_RETRY_HTTP = {408, 409, 429, 500, 502, 503, 529}


class NanoHorizonSamplerFailure(RuntimeError):
    """A terminal provider response that cannot produce a policy action."""

    def __init__(self, code: str, *, completion: dict[str, Any]) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = False
        self.completion = copy.deepcopy(completion)


def _retry_after_seconds(exc: urllib.error.HTTPError, detail: str) -> float | None:
    raw = ""
    try:
        raw = str(exc.headers.get("Retry-After") or "").strip()
    except Exception:  # noqa: BLE001
        raw = ""
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        for key in ("retry_after", "retryAfter"):
            value = error.get(key) if isinstance(error, dict) else None
            if value is not None:
                try:
                    return max(0.0, float(value))
                except (TypeError, ValueError):
                    pass
    return None


def _backoff_seconds(attempt: int, *, base: float, cap: float, retry_after: float | None) -> float:
    if retry_after is not None:
        wait = retry_after
    else:
        wait = min(cap, base * (2**attempt))
    jitter = random.uniform(0.0, min(1.0, wait * 0.25) if wait else 0.25)
    return min(cap, wait + jitter)


def _json_request(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    retries: int = 0,
    min_interval: float = 0.0,
    retry_max_wait: float = 90.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        # Groq (Cloudflare 1010) bans Python-urllib's default User-Agent.
        "User-Agent": "NanoHorizon/0.1",
        **(headers or {}),
    }
    last_error: Exception | None = None
    max_wait = max(1.0, float(retry_max_wait))
    for attempt in range(max(0, retries) + 1):
        _PACE.wait(min_interval)
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                raw = response.read()
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
            parsed = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(parsed, dict):
                raise RuntimeError("sampler_response_not_object")
            parsed["_headers"] = response_headers
            return parsed
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last_error = RuntimeError(f"sampler_http_{exc.code}:{detail}")
            if exc.code not in _RETRY_HTTP or attempt >= retries:
                raise last_error from exc
            retry_after = _retry_after_seconds(exc, detail)
            base = 8.0 if exc.code == 429 else 1.0
            wait = _backoff_seconds(
                attempt, base=base, cap=max_wait, retry_after=retry_after
            )
            _PACE.cool(wait)
            print(
                f"nanohorizon sampler http {exc.code}; backing off {wait:.1f}s "
                f"(attempt {attempt + 1}/{retries + 1})",
                file=sys.stderr,
                flush=True,
            )
            continue
        except urllib.error.URLError as exc:
            last_error = RuntimeError(f"sampler_connect:{exc.reason}")
            if attempt >= retries:
                raise last_error from exc
            wait = _backoff_seconds(attempt, base=1.0, cap=min(20.0, max_wait), retry_after=None)
            _PACE.cool(wait)
            print(
                f"nanohorizon sampler connect error; backing off {wait:.1f}s "
                f"(attempt {attempt + 1}/{retries + 1})",
                file=sys.stderr,
                flush=True,
            )
            continue
    raise last_error or RuntimeError("sampler_http_failed")


def resolve_sampler_api(config: dict[str, Any]) -> str:
    raw = str(config.get("api") or config.get("api_family") or "").strip().lower()
    if raw in {"responses"}:
        return "responses"
    return "chat_completions"


def is_local_compat(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    # Workshop's scoped provider proxy is container-reachable through
    # host.docker.internal, but the sampler behind it is still a remote
    # provider. Treating that route as a local model incorrectly adds local
    # decoding knobs and suppresses provider reasoning controls.
    if "/cap/wcap_" in parsed.path:
        return False
    return host in {"127.0.0.1", "localhost", "host.docker.internal"}


def chat_completions_url(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if not base:
        raise RuntimeError("nanohorizon sampler config requires base_url")
    if base.endswith("/chat/completions"):
        return base
    # OpenAI-compatible gateways may scope the base beneath /v1 (for example
    # /v1/providers/<provider>).  In that case /v1 is already present and must
    # not be appended a second time.
    path = urlparse(base).path.rstrip("/")
    if path.endswith("/v1") or "/v1/providers/" in f"{path}/":
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _text_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(value)


def _message_from_chat(message: dict[str, Any]) -> dict[str, Any]:
    content = _text_field(message.get("content"))
    reasoning = _text_field(
        message.get("reasoning") or message.get("reasoning_content")
    )
    calls = _wire_tool_calls(message.get("tool_calls"))
    if calls:
        row: dict[str, Any] = {"role": "assistant", "tool_calls": calls}
        if content:
            row["content"] = content
    elif content:
        row = _message_from_sample(content)
    else:
        row = {"role": "assistant", "content": content or None}
    if reasoning and "reasoning_content" not in row:
        row["reasoning_content"] = reasoning
    return row


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
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        row["reasoning_content"] = _text_field(reasoning)
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


class HttpSampler:
    """OpenAI-compatible chat completions. MLX serve and OpenRouter share this."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.api = resolve_sampler_api(config)
        self.model = str(config.get("model") or "Qwen/Qwen3.5-0.8B")
        if self.model.startswith("openrouter/"):
            self.model = self.model[len("openrouter/") :]
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
        self.chat_url = chat_completions_url(str(config.get("base_url") or ""))
        # Modal / SGLang OpenAI-compatible student: same wire as MLX, not a paid teacher.
        self.local = is_local_compat(self.chat_url) or bool(
            config.get("openai_compatible_local")
        )
        self.timeout = float(
            config.get("timeout_seconds") or (120.0 if self.local else 180.0)
        )
        self.snapshot = str(config.get("policy_snapshot_id") or "").strip() or None
        self.api_key_env = str(config.get("api_key_env") or "").strip()
        if not self.api_key_env and not self.local:
            self.api_key_env = "OPENROUTER_API_KEY"
        self.effort = str(
            config.get("reasoning_effort") or config.get("effort") or "medium"
        )
        self.min_interval = float(config.get("min_request_interval") or 0.0)
        if not self.local and self.min_interval <= 0:
            self.min_interval = 2.0
        self.retries = int(
            config.get("sampler_retries")
            if config.get("sampler_retries") is not None
            else (16 if not self.local else 0)
        )
        self.retry_max_wait = float(config.get("retry_max_wait") or (90.0 if not self.local else 20.0))

    def _auth_headers(self) -> dict[str, str]:
        if not self.api_key_env:
            return {}
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            if self.local:
                return {}
            raise RuntimeError(f"paid sampler requires {self.api_key_env}")
        headers = {"Authorization": f"Bearer {key}"}
        if "openrouter.ai" in self.chat_url:
            headers["HTTP-Referer"] = "https://usesynth.ai"
            headers["X-Title"] = "NanoHorizon"
        return headers

    def wire_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        seed: int = 0,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_wire_message(row) for row in messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": int(
                self.thinking_budget + self.answer_max_tokens
                if max_tokens is None
                else max_tokens
            ),
        }
        if tools:
            payload["tools"] = tools
            # NanoHorizon's policy contract requires exactly one
            # craftax_interact call. Letting the provider choose `auto` can
            # spend the entire completion budget on reasoning and return no
            # action, leaving a nominally completed rollout at step zero.
            payload["tool_choice"] = "required"
        if self.local:
            payload["enable_thinking"] = thinking
            payload["top_k"] = self.top_k
            payload["seed"] = int(seed)
            payload["stop"] = ["</tool_call>"]
            if self.snapshot:
                payload["policy_snapshot_id"] = self.snapshot
        elif thinking:
            if "groq.com" in self.chat_url:
                payload["reasoning_effort"] = self.effort or "low"
            else:
                # OpenRouter's normalized effort contract works across its
                # providers. An exact reasoning-token budget is model-specific
                # and can be translated to a larger minimum allocation by an
                # effort-only provider, so retain the approved total output
                # ceiling and request the configured effort instead.
                payload["reasoning"] = {"effort": self.effort}
        return payload

    def request_observation(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = payload if payload is not None else self.wire_payload([])
        extras: dict[str, Any] = {
            "enableThinking": bool(
                body.get("enable_thinking")
                if "enable_thinking" in body
                else self.enable_thinking
            ),
            "thinkingBudget": self.thinking_budget,
        }
        if not self.local and self.effort:
            extras["effort"] = self.effort
        return gen_ai_request_observation(body, extras=extras)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        payload = self.wire_payload(messages, tools=tools, seed=seed)
        body = _json_request(
            self.chat_url,
            payload,
            timeout=self.timeout,
            headers=self._auth_headers(),
            retries=self.retries,
            min_interval=self.min_interval,
            retry_max_wait=self.retry_max_wait,
        )
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            message = {}
        wired = _message_from_chat(message)
        if not wired.get("tool_calls") and wired.get("content"):
            wired = _message_from_sample(_text_field(wired.get("content")))
            if message.get("reasoning") or message.get("reasoning_content"):
                wired["reasoning_content"] = _text_field(
                    message.get("reasoning") or message.get("reasoning_content")
                )
        reasoning_details = message.get("reasoning_details")
        if isinstance(reasoning_details, list):
            wired["reasoning_details"] = copy.deepcopy(reasoning_details)
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        usage = copy.deepcopy(usage)
        completion_details = (
            usage.get("completion_tokens_details")
            if isinstance(usage.get("completion_tokens_details"), dict)
            else {}
        )
        headers = body.get("_headers") if isinstance(body.get("_headers"), dict) else {}
        completion = {
            "text": _text_field(wired.get("content")),
            "message": wired,
            "finish_reason": str(choice.get("finish_reason") or ""),
            "proxy_request_id": str(
                headers.get("x-proxy-request-id")
                or headers.get("x-request-id")
                or headers.get("x-openrouter-id")
                or ""
            ),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "reasoning_tokens": int(
                completion_details.get("reasoning_tokens")
                or usage.get("reasoning_tokens")
                or 0
            ),
            "usage": usage,
            **self.request_observation(payload),
        }
        tool_calls = wired.get("tool_calls")
        calls = tool_calls if isinstance(tool_calls, list) else []
        exactly_one_action = (
            len(calls) == 1
            and isinstance(calls[0], dict)
            and isinstance(calls[0].get("function"), dict)
            and calls[0]["function"].get("name") == TOOL_NAME
        )
        if not exactly_one_action:
            code = "expected_exactly_one_craftax_interact_tool_call"
            if not calls and completion["finish_reason"] == "length":
                code = "reasoning_budget_exhausted_before_tool"
            raise NanoHorizonSamplerFailure(code, completion=completion)
        return completion

    def summarize(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        payload = self.wire_payload(
            messages,
            max_tokens=int(max_tokens),
            enable_thinking=False,
        )
        if self.local:
            payload["enable_thinking"] = False
            payload.pop("stop", None)
        else:
            payload.pop("reasoning", None)
            payload.pop("reasoning_effort", None)
        body = _json_request(
            self.chat_url,
            payload,
            timeout=self.timeout,
            headers=self._auth_headers(),
            retries=self.retries,
            min_interval=self.min_interval,
            retry_max_wait=self.retry_max_wait,
        )
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        if not isinstance(message, dict):
            return ""
        return _text_field(
            message.get("content") or message.get("reasoning") or ""
        ).strip()


class NanoHorizonPlanner:
    """Runs the PUT ``Policy.run_episode`` against a gold world."""

    def __init__(self, *, config_id: str, config: dict[str, Any], code: bytes) -> None:
        self.config_id = config_id
        policy_cls = load_policy_class(code)
        defaults = getattr(sys.modules[policy_cls.__module__], "SAMPLER", {}) or {}
        if not isinstance(defaults, dict):
            defaults = {}
        self.config = {**defaults, **dict(config)}
        self.sampler = HttpSampler(self.config)
        self.policy = policy_cls(**self.config)
        self._calls = 0
        self._last_events: list[dict[str, Any]] = []
        self._last_trace: dict[str, Any] = {}
        self._call_gen_ai: list[dict[str, Any]] = []
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
            "api": self.sampler.api,
            "enable_thinking": self.sampler.enable_thinking,
            "graded": True,
            **self.sampler.request_observation(),
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
        self._call_gen_ai = []
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
            self._calls += 1
            log.append("span.policy.opened", {"harness": HARNESS, "call": self._calls})
            failure: NanoHorizonSamplerFailure | None = None
            try:
                completion = self.sampler.complete(messages, **kwargs)
            except NanoHorizonSamplerFailure as exc:
                completion = exc.completion
                failure = exc
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
                "phase": "sample",
                "turn_kind": "policy",
                "trainable": True,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(list(kwargs.get("tools") or [])),
                "assistant": copy.deepcopy(completion.get("message") or {}),
                "proxy_request_id": completion.get("proxy_request_id"),
                "finish_reason": completion.get("finish_reason"),
                "prompt_tokens": completion.get("prompt_tokens"),
                "completion_tokens": completion.get("completion_tokens"),
                "reasoning_tokens": completion.get("reasoning_tokens"),
                "usage": copy.deepcopy(completion.get("usage") or {}),
                "text": completion.get("text"),
                **{
                    key: value
                    for key, value in completion.items()
                    if key == "modelParameters" or str(key).startswith("gen_ai.")
                },
            }
            if failure is not None:
                self._last_trace.update(
                    {
                        "error": failure.code,
                        "error_retryable": failure.retryable,
                    }
                )
            self._call_gen_ai.append(copy_observation(self._last_trace))
            log.append("span.policy.data", dict(self._last_trace))
            if failure is not None:
                raise failure
            return completion

        def summarize(messages: list[dict[str, Any]], max_tokens: int) -> str:
            log.append(
                "agent.context_compacting",
                {"call": self._calls, "max_tokens": max_tokens},
            )
            compact_payload = self.sampler.wire_payload(
                messages,
                max_tokens=int(max_tokens),
                enable_thinking=False,
            )
            if self.sampler.local:
                compact_payload.pop("stop", None)
            else:
                compact_payload.pop("reasoning", None)
                compact_payload.pop("reasoning_effort", None)
            text = self.sampler.summarize(messages, max_tokens)
            log.append(
                "span.policy.data",
                {
                    "phase": "compaction",
                    "turn_kind": "summary",
                    "trainable": False,
                    "messages": copy.deepcopy(messages),
                    "assistant": {"role": "assistant", "content": text},
                    **self.sampler.request_observation(compact_payload),
                },
            )
            return text

        outcome = self.policy.run_episode(
            opening={**result.observation, **({"done": True} if result.done else {})},
            step=step_policy,
            drain=drain,
            sample=sample,
            summarize=summarize,
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
            if kind == "span.policy.data" and isinstance(payload, dict):
                payload = dict(payload)
                stamped = copy_observation(payload)
                if not stamped:
                    idx = payload.get("call_index")
                    if isinstance(idx, int) and 0 <= idx < len(self._call_gen_ai):
                        stamped = self._call_gen_ai[idx]
                    elif self._call_gen_ai:
                        stamped = self._call_gen_ai[-1]
                    payload.update(stamped)
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
