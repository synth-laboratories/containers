import pytest

from synth_containers.gold_http import StepResult
from synth_containers.policies import nanohorizon
from synth_containers.policies.nanohorizon import (
    HttpSampler,
    NanoHorizonPlanner,
    NanoHorizonSamplerFailure,
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


def test_length_without_tool_is_terminal_reasoning_budget_failure(monkeypatch) -> None:
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

    with pytest.raises(NanoHorizonSamplerFailure) as caught:
        sampler.complete([{"role": "user", "content": "Act."}], tools=TOOLS)

    assert requests == 1
    assert caught.value.code == "reasoning_budget_exhausted_before_tool"
    assert caught.value.retryable is False
    assert caught.value.completion["finish_reason"] == "length"
    assert caught.value.completion["reasoning_tokens"] == 384
    assert caught.value.completion["usage"]["completion_tokens_details"] == {
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
def test_sampler_rejects_any_response_without_exactly_one_action(
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

    with pytest.raises(
        NanoHorizonSamplerFailure,
        match="expected_exactly_one_craftax_interact_tool_call",
    ):
        sampler.complete([{"role": "user", "content": "Act."}], tools=TOOLS)


def test_planner_records_terminal_sampler_failure_before_aborting(monkeypatch) -> None:
    completion = {
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

    def fail_once(*args, **kwargs):
        raise NanoHorizonSamplerFailure(
            "reasoning_budget_exhausted_before_tool", completion=completion
        )

    monkeypatch.setattr(planner.sampler, "complete", fail_once)

    class Policy:
        def run_episode(self, **kwargs):
            kwargs["sample"](
                [{"role": "user", "content": "Act."}], tools=TOOLS, seed=1
            )
            raise AssertionError("terminal sampler failure must abort the rollout")

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

        def append(self, kind, payload):
            self.rows.append((kind, payload))

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

    with pytest.raises(
        NanoHorizonSamplerFailure,
        match="reasoning_budget_exhausted_before_tool",
    ):
        planner.run(world=World(), log=log, seed=1, max_steps=10)

    assert planner.usage() == {
        "prompt_tokens": 100,
        "completion_tokens": 384,
        "total_tokens": 484,
        "calls": 1,
    }
    traces = [payload for kind, payload in log.rows if kind == "span.policy.data"]
    assert len(traces) == 1
    assert traces[0]["finish_reason"] == "length"
    assert traces[0]["reasoning_tokens"] == 384
    assert traces[0]["error"] == "reasoning_budget_exhausted_before_tool"
    assert traces[0]["error_retryable"] is False
