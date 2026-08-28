import pytest

from synth_containers.policies import nanohorizon
from synth_containers.policies.nanohorizon import (
    HttpSampler,
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
    assert payload["tool_choice"] == "required"
    assert payload["max_tokens"] == 384
    assert payload["reasoning"] == {"effort": "medium"}
    assert "enable_thinking" not in payload
    assert "top_k" not in payload
    assert "stop" not in payload


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
