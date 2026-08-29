import io
import json
import urllib.error

import pytest

from synth_containers.gold_http import StepResult
from synth_containers.policies import nanohorizon
from synth_containers.policies.nanohorizon import (
    HttpSampler,
    NanoHorizonSamplerFailure,
    NanoHorizonPlanner,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "craftax_interact",
            "parameters": {
                "type": "object",
                "properties": {"actions": {"type": "array", "items": {"type": "string"}}},
                "required": ["actions"],
            },
        },
    }
]


def test_workshop_capability_proxy_keeps_remote_provider_semantics() -> None:
    sampler = HttpSampler(
        {
            "base_url": (
                "http://host.docker.internal:17654/cap/wcap_test/"
                "v1/providers/openrouter"
            ),
            "model": "z-ai/glm-5.3-flash",
            "effort": "medium",
        }
    )

    payload = sampler.wire_payload(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert sampler.local is False
    assert sampler.workshop_capability_proxy is True
    assert sampler.api_key_env == ""
    assert sampler._auth_headers() == {}
    assert payload["tool_choice"] == "required"
    assert payload["max_tokens"] == 384
    assert payload["reasoning"] == {"effort": "medium"}
    assert "enable_thinking" not in payload
    assert "top_k" not in payload
    assert "stop" not in payload


def test_workshop_capability_proxy_never_forwards_explicit_provider_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-cross-the-proxy")
    sampler = HttpSampler(
        {
            "base_url": (
                "http://host.docker.internal:17654/cap/wcap_test/"
                "v1/providers/openrouter"
            ),
            "model": "z-ai/glm-5.3-flash",
            "api_key_env": "OPENROUTER_API_KEY",
        }
    )

    assert sampler.api_key_env == ""
    assert sampler._auth_headers() == {}


def _http_error(
    url: str,
    *,
    status: int,
    code: str,
    origin: str,
) -> urllib.error.HTTPError:
    body = json.dumps({"error": {"code": code, "message": "bounded"}}).encode()
    return urllib.error.HTTPError(
        url,
        status,
        code,
        {"content-type": "application/json", "x-workshop-proxy-origin": origin},
        io.BytesIO(body),
    )


def test_workshop_capability_exhaustion_is_terminal_without_retry(monkeypatch) -> None:
    url = (
        "http://host.docker.internal:17654/cap/wcap_test/"
        "v1/providers/openrouter/chat/completions"
    )
    attempts = 0

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise _http_error(
            url,
            status=429,
            code="budget_exhausted",
            origin="proxy",
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(nanohorizon._PACE, "wait", lambda *_args: None)

    with pytest.raises(
        NanoHorizonSamplerFailure,
        match="workshop_capability_exhausted",
    ) as raised:
        nanohorizon._json_request(url, {"model": "test"}, timeout=1, retries=16)

    assert attempts == 1
    assert raised.value.retryable is False
    assert raised.value.completion["error"] == "workshop_capability_exhausted"
    assert "wcap_" not in str(raised.value)


def test_provider_origin_429_remains_retryable_through_workshop_proxy(monkeypatch) -> None:
    url = (
        "http://host.docker.internal:17654/cap/wcap_test/"
        "v1/providers/openrouter/chat/completions"
    )
    attempts = 0

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"ok":true}'

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _http_error(
                url,
                status=429,
                code="upstream_rate_limited",
                origin="upstream",
            )
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(nanohorizon._PACE, "wait", lambda *_args: None)
    monkeypatch.setattr(nanohorizon._PACE, "cool", lambda *_args: None)
    monkeypatch.setattr(nanohorizon, "_backoff_seconds", lambda *_args, **_kwargs: 0.0)

    assert nanohorizon._json_request(
        url,
        {"model": "test"},
        timeout=1,
        retries=1,
    )["ok"] is True
    assert attempts == 2


def test_sampler_read_timeout_is_retried_within_declared_retry_budget(monkeypatch) -> None:
    attempts = 0

    class Response:
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"ok":true}'

    def urlopen(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timed out")
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(nanohorizon._PACE, "wait", lambda *_args: None)
    monkeypatch.setattr(nanohorizon._PACE, "cool", lambda *_args: None)
    monkeypatch.setattr(nanohorizon, "_backoff_seconds", lambda *_args, **_kwargs: 0.0)

    assert nanohorizon._json_request(
        "https://example.test/v1/chat/completions",
        {"model": "test"},
        timeout=1,
        retries=1,
    )["ok"] is True
    assert attempts == 2


def test_public_openrouter_still_requires_its_declared_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    sampler = HttpSampler(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
        }
    )

    with pytest.raises(RuntimeError, match="paid sampler requires OPENROUTER_API_KEY"):
        sampler._auth_headers()


def test_local_sampler_still_requires_the_declared_tool_call() -> None:
    sampler = HttpSampler(
        {
            "base_url": "http://host.docker.internal:8787/v1",
            "model": "Qwen/Qwen3.5-2B",
        }
    )

    payload = sampler.wire_payload(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert sampler.local is True
    assert payload["tool_choice"] == "required"
    assert payload["enable_thinking"] is True


def test_reasoning_effort_alias_uses_openrouter_normalized_wire() -> None:
    sampler = HttpSampler(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
            "effort": "high",
            "reasoning_effort": "low",
        }
    )

    payload = sampler.wire_payload(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert sampler.effort == "low"
    assert payload["reasoning"] == {"effort": "low"}


def test_completion_preserves_provider_reasoning_and_usage(monkeypatch) -> None:
    sampler = HttpSampler(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
        }
    )
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "reasoning_content": "I should move.",
                    "reasoning_details": [{"type": "summary", "text": "move"}],
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "craftax_interact",
                                "arguments": '{"actions":["do"]}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "cost": 0.0001,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
        "_headers": {"x-proxy-request-id": "proxy-1"},
    }
    monkeypatch.setattr(nanohorizon, "_json_request", lambda *args, **kwargs: response)
    monkeypatch.setattr(sampler, "_auth_headers", lambda: {})

    completion = sampler.complete(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert completion["finish_reason"] == "tool_calls"
    assert completion["proxy_request_id"] == "proxy-1"
    assert completion["reasoning_tokens"] == 5
    assert completion["usage"] == response["usage"]
    assert completion["message"]["reasoning_content"] == "I should move."
    assert completion["message"]["reasoning_details"] == [
        {"type": "summary", "text": "move"}
    ]


def test_completion_retains_generation_id_when_proxy_headers_are_absent(
    monkeypatch,
) -> None:
    sampler = HttpSampler(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
        }
    )
    monkeypatch.setattr(
        nanohorizon,
        "_json_request",
        lambda *args, **kwargs: {
            "id": "gen-accounting-fallback",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None, "tool_calls": []},
                }
            ],
            "usage": {},
        },
    )
    monkeypatch.setattr(sampler, "_auth_headers", lambda: {})

    completion = sampler.complete(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert completion["proxy_request_id"] == "gen-accounting-fallback"
    assert completion["sampler_validation_error"] == (
        "expected_exactly_one_craftax_interact_tool_call"
    )


def test_length_without_tool_flows_to_policy_as_invalid_completion(monkeypatch) -> None:
    sampler = HttpSampler(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
        }
    )
    requests = 0

    def response(*args, **kwargs):
        nonlocal requests
        requests += 1
        return {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": None,
                        "reasoning_content": "Still thinking",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 384,
                "completion_tokens_details": {"reasoning_tokens": 384},
            },
        }

    monkeypatch.setattr(nanohorizon, "_json_request", response)
    monkeypatch.setattr(sampler, "_auth_headers", lambda: {})

    completion = sampler.complete(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert requests == 1
    assert completion["sampler_validation_error"] == (
        "reasoning_budget_exhausted_before_tool"
    )
    assert completion["finish_reason"] == "length"
    assert completion["reasoning_tokens"] == 384
    assert completion["usage"]["completion_tokens_details"] == {
        "reasoning_tokens": 384
    }


@pytest.mark.parametrize(
    "tool_calls",
    [
        [],
        [
            {
                "id": "wrong",
                "type": "function",
                "function": {"name": "other_tool", "arguments": "{}"},
            }
        ],
        [
            {
                "id": "one",
                "type": "function",
                "function": {"name": "craftax_interact", "arguments": "{}"},
            },
            {
                "id": "two",
                "type": "function",
                "function": {"name": "craftax_interact", "arguments": "{}"},
            },
        ],
    ],
)
def test_sampler_marks_any_response_without_exactly_one_action_for_policy_parse(
    monkeypatch, tool_calls
) -> None:
    sampler = HttpSampler(
        {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "z-ai/glm-5.3-flash",
        }
    )
    monkeypatch.setattr(
        nanohorizon,
        "_json_request",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None, "tool_calls": tool_calls},
                }
            ]
        },
    )
    monkeypatch.setattr(sampler, "_auth_headers", lambda: {})

    completion = sampler.complete(
        [{"role": "user", "content": "Act."}], tools=TOOLS
    )

    assert completion["sampler_validation_error"] == (
        "expected_exactly_one_craftax_interact_tool_call"
    )


def test_planner_allows_bounded_policy_recovery_and_counts_each_provider_call(
    monkeypatch,
) -> None:
    invalid_completion = {
        "text": "",
        "message": {"role": "assistant", "reasoning_content": "Still thinking"},
        "finish_reason": "length",
        "proxy_request_id": "proxy-failed",
        "prompt_tokens": 100,
        "completion_tokens": 384,
        "reasoning_tokens": 384,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 384,
            "completion_tokens_details": {"reasoning_tokens": 384},
        },
        "sampler_validation_error": "reasoning_budget_exhausted_before_tool",
    }
    valid_completion = {
        "text": "",
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "craftax_interact",
                        "arguments": '{"actions":["do"]}',
                    },
                }
            ],
        },
        "finish_reason": "tool_calls",
        "proxy_request_id": "proxy-recovered",
        "prompt_tokens": 120,
        "completion_tokens": 20,
        "reasoning_tokens": 5,
        "usage": {"prompt_tokens": 120, "completion_tokens": 20},
    }
    planner = object.__new__(NanoHorizonPlanner)
    planner.config_id = "test"
    planner.config = {}
    planner.sampler = HttpSampler(
        {
            "base_url": "http://host.docker.internal:8787/v1",
            "model": "Qwen/Qwen3.5-2B",
        }
    )

    completions = iter([invalid_completion, valid_completion])

    monkeypatch.setattr(
        planner.sampler,
        "complete",
        lambda *args, **kwargs: next(completions),
    )

    class Policy:
        def run_episode(self, **kwargs):
            first = kwargs["sample"](
                [{"role": "user", "content": "Act."}], tools=TOOLS, seed=1
            )
            assert first["sampler_validation_error"] == (
                "reasoning_budget_exhausted_before_tool"
            )
            second = kwargs["sample"](
                [{"role": "user", "content": "Try again."}], tools=TOOLS, seed=2
            )
            assert second["message"]["tool_calls"][0]["function"]["name"] == (
                "craftax_interact"
            )
            return {
                "journal": [],
                "proxy_request_ids": ["proxy-failed", "proxy-recovered"],
                "achievements": [],
                "reward": 0.0,
            }

    class World:
        def reset(self, seed, *, max_steps):
            return StepResult(
                observation={"private": {}},
                reward=0.0,
                done=False,
                valid_actions=[],
                ascii_map=".",
                frame_digest="frame-0",
                env_steps=0,
            )

        def drain_native_events(self):
            return []

    class Log:
        def __init__(self):
            self.rows = []
            self.closed = False

        def append(self, kind, payload):
            self.rows.append((kind, payload))

        @property
        def high_water(self):
            return len(self.rows)

        def mark_closed(self):
            self.closed = True

        def persist_frame(self, step, frame_bytes):
            return None

    planner.policy = Policy()
    planner._calls = 0
    planner._last_events = []
    planner._last_trace = {}
    planner._call_gen_ai = []
    planner._usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    log = Log()

    planner.run(world=World(), log=log, seed=1, max_steps=10)

    assert planner.usage() == {
        "prompt_tokens": 220,
        "completion_tokens": 404,
        "total_tokens": 624,
        "calls": 2,
    }
    traces = [payload for kind, payload in log.rows if kind == "span.policy.data"]
    assert len(traces) == 2
    assert traces[0]["finish_reason"] == "length"
    assert traces[0]["reasoning_tokens"] == 384
    assert traces[0]["sampler_validation_error"] == (
        "reasoning_budget_exhausted_before_tool"
    )
    assert traces[1]["finish_reason"] == "tool_calls"
    assert traces[1]["sampler_validation_error"] is None


def test_compaction_generation_counts_against_rollout_provider_cap(monkeypatch) -> None:
    planner = object.__new__(NanoHorizonPlanner)
    planner.config_id = "test"
    planner.config = {"max_calls": 2}
    planner.sampler = HttpSampler(
        {
            "base_url": "http://host.docker.internal:8787/v1",
            "model": "Qwen/Qwen3.5-2B",
        }
    )
    planner.policy = None
    planner.max_provider_calls = 2
    planner._calls = 0
    planner._last_events = []
    planner._last_trace = {}
    planner._call_gen_ai = []
    planner._usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    first = {
        "text": "",
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "craftax_interact",
                        "arguments": '{"actions":["do"]}',
                    },
                }
            ],
        },
        "finish_reason": "tool_calls",
        "proxy_request_id": "proxy-action",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "reasoning_tokens": 0,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    monkeypatch.setattr(planner.sampler, "complete", lambda *_args, **_kwargs: first)
    monkeypatch.setattr(planner.sampler, "summarize", lambda *_args, **_kwargs: "summary")

    class Policy:
        def run_episode(self, **kwargs):
            kwargs["sample"]([], tools=TOOLS, seed=1)
            kwargs["step"]("do")
            assert kwargs["summarize"]([], 10) == "summary"
            kwargs["sample"]([], tools=TOOLS, seed=2)
            raise AssertionError("the local provider cap must stop before this point")

    class World:
        def reset(self, seed, *, max_steps):
            return StepResult(
                observation={"private": {}},
                reward=0.0,
                done=False,
                valid_actions=[],
                ascii_map=".",
                frame_digest="frame-0",
                env_steps=0,
            )

        def drain_native_events(self):
            return []

        def step(self, action):
            assert action == "do"
            return StepResult(
                observation={
                    "private": {
                        "total_reward": 1.25,
                        "achievements": {"collect_wood": True, "drink_water": False},
                    }
                },
                reward=1.25,
                done=False,
                valid_actions=[],
                ascii_map=".",
                frame_digest="frame-1",
                env_steps=1,
            )

    class Log:
        def __init__(self):
            self.rows = []
            self.closed = False

        def append(self, kind, payload):
            self.rows.append((kind, payload))

        @property
        def high_water(self):
            return len(self.rows)

        def mark_closed(self):
            self.closed = True

        def persist_frame(self, step, frame_bytes):
            return None

    planner.policy = Policy()
    log = Log()

    outcome = planner.run(world=World(), log=log, seed=1, max_steps=10)

    assert planner.usage()["calls"] == 2
    assert outcome["reward_signals"] == [1.25]
    assert outcome["actions"] == ["do"]
    assert outcome["episode"]["reward"] == 1.25
    assert outcome["episode"]["achievements"] == ["collect_wood"]
    assert outcome["episode"]["proxy_request_ids"] == ["proxy-action"]
    assert outcome["episode"]["policy_calls"] == 1
    assert outcome["episode"]["provider_calls"] == 2
    assert outcome["episode"]["truncated"] is True
    assert outcome["episode"]["stopped_on"] == "provider_call_limit"
    assert log.closed is True
    terminal = [payload for kind, payload in log.rows if kind == "status"][-1]
    assert terminal == {
        "status": "completed",
        "steps": 1,
        "reason": "provider_call_limit",
        "truncated": True,
    }
    assert [payload["phase"] for kind, payload in log.rows if kind == "span.policy.data"] == [
        "sample",
        "compaction",
    ]


def test_premature_workshop_capability_exhaustion_remains_failure(monkeypatch) -> None:
    planner = object.__new__(NanoHorizonPlanner)
    planner.config_id = "test"
    planner.config = {"max_calls": 10}
    planner.sampler = HttpSampler(
        {
            "base_url": (
                "http://host.docker.internal:17654/cap/wcap_test/"
                "v1/providers/openrouter"
            ),
            "model": "z-ai/glm-5.3-flash",
        }
    )
    planner.max_provider_calls = 10
    planner._calls = 0
    planner._last_events = []
    planner._last_trace = {}
    planner._call_gen_ai = []
    planner._usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    failure = NanoHorizonSamplerFailure(
        "workshop_capability_exhausted",
        completion=nanohorizon._failure_completion("workshop_capability_exhausted"),
    )
    monkeypatch.setattr(
        planner.sampler,
        "complete",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    class Policy:
        def run_episode(self, **kwargs):
            kwargs["sample"]([], tools=TOOLS, seed=1)
            raise AssertionError("premature exhaustion must abort the policy")

    class World:
        def reset(self, seed, *, max_steps):
            return StepResult(
                observation={"private": {"total_reward": 2.0}},
                reward=0.0,
                done=False,
                valid_actions=[],
                ascii_map=".",
                frame_digest="frame-0",
                env_steps=0,
            )

        def drain_native_events(self):
            return []

    class Log:
        def __init__(self):
            self.rows = []
            self.closed = False

        def append(self, kind, payload):
            self.rows.append((kind, payload))

        @property
        def high_water(self):
            return len(self.rows)

        def mark_closed(self):
            self.closed = True

        def persist_frame(self, step, frame_bytes):
            return None

    planner.policy = Policy()
    log = Log()

    with pytest.raises(
        NanoHorizonSamplerFailure,
        match="workshop_capability_exhausted",
    ):
        planner.run(world=World(), log=log, seed=1, max_steps=10)

    assert planner.usage()["calls"] == 1
    assert log.closed is False
    assert not [payload for kind, payload in log.rows if kind == "status"]
    trace = [payload for kind, payload in log.rows if kind == "span.policy.data"][-1]
    assert trace["error"] == "workshop_capability_exhausted"
    assert trace["error_retryable"] is False
