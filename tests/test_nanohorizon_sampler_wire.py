from synth_containers.policies.nanohorizon import HttpSampler


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
    assert payload["reasoning"] == {"max_tokens": 256}
    assert "effort" not in payload["reasoning"]
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
