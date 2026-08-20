"""End-to-end ReAct and code-policy evals through Containers HTTP."""

from __future__ import annotations

import json

import pytest

from synth_containers.platform.eval_examples import run_code_policy, run_react
from synth_containers.platform.react import (
    OpenRouterReAct,
    PolicyConfigError,
    CRAFTAX_REACT_SYSTEM_PROMPT,
)


def test_react_ten_seeds_through_containers_http() -> None:
    result = run_react(seeds=10)
    board = result["leaderboard"]
    assert len(board) == 10
    assert len({row["rollout_id"] for row in board}) == 10
    assert all(row["first_semantic"] == "trace.opened" for row in board)
    assert all(row["policy_spans"] >= 1 for row in board)
    assert all(row["actions"] >= 5 for row in board)
    assert all(row["status"] in {"scored", "absent"} for row in board)
    assert all(row["reward"] is None or isinstance(row["reward"], (int, float)) for row in board)
    assert result["absent_n"] == 0


def test_code_policy_put_restart_distinct_from_noop() -> None:
    result = run_code_policy()
    assert result["engine_generation"] == 1
    assert result["isolation_receipt"]["sandbox"] == "process"
    assert result["do_policy"]["reward"] == 0.5
    assert result["noop_policy"]["reward"] == 0.0
    assert set(result["noop_policy"]["actions"]) == {"noop"}
    assert "do" in result["do_policy"]["actions"]


def test_openrouter_react_normalizes_craftax_direction_aliases() -> None:
    actions = OpenRouterReAct._parse_actions(
        '{"actions":["North","east","do"]}',
        ["up", "right", "do"],
    )
    assert actions == ["up", "right", "do"]


def test_openrouter_react_binds_candidate_system_prompt() -> None:
    policy = OpenRouterReAct(
        config_id="goex_candidate_test",
        config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024, "context_token_budget": 16000, "compact_at": 0.7, "keep_recent_messages": 8, "keep_recent_frames": 2, "observation_mode": "text", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "parse_retries": 0, "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT, "system_prompt": "Prioritize wood before stone."},
    )
    assert policy._messages[0] == {
        "role": "system",
        "content": "Prioritize wood before stone.",
    }


def test_openrouter_react_preserves_empty_response_as_labeled_fallback(
    monkeypatch,
) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": "", "reasoning": "thinking"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }
            ).encode()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    policy = OpenRouterReAct(config_id="muse_spark_medium", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024, "context_token_budget": 16000, "compact_at": 0.7, "keep_recent_messages": 8, "keep_recent_frames": 2, "observation_mode": "text", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "parse_retries": 0, "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT, })
    actions = policy.plan(
        {
            "valid_actions": ["up", "do"],
            "observation_text": "real observation placeholder",
        }
    )
    assert actions == ["do"]
    trace = policy.trace_data()
    assert trace["action_authority"] == "harness_fallback"
    assert trace["fallback"] is True
    assert trace["assistant"] == ""
    assert trace["reasoning"] == "thinking"
    assert trace["tool_arguments"] == ""
    assert trace["actions"] == ["do"]
    assert trace["context_token_budget"] == 16000
    assert trace["compact_at"] == 0.7
    assert trace["compact_count"] == 0
    assert trace["history_turns"] == 1
    assert trace["deltas_emitted"] == 0
    assert trace["token_trace"] is None
    assert trace["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "cost_usd": None,
        "usage_status": "reported",
    }


def test_openrouter_react_uses_forced_tool_arguments(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "choose_actions",
                                            "arguments": '{"actions":["east","do"]}',
                                        }
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {},
                }
            ).encode()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    policy = OpenRouterReAct(config_id="muse_spark_medium", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024, "context_token_budget": 16000, "compact_at": 0.7, "keep_recent_messages": 8, "keep_recent_frames": 2, "observation_mode": "text", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "parse_retries": 0, "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT, })
    assert policy.plan({"valid_actions": ["right", "do"], "observation_text": "obs"}) == [
        "right",
        "do",
    ]
    trace = policy.trace_data()
    assert trace["action_authority"] == "policy"
    assert trace["fallback"] is False
    assert trace["tool_arguments"] == '{"actions":["east","do"]}'
    assert trace["context_token_budget"] == 16000
    assert trace["compact_at"] == 0.7
    assert len(policy._messages) == 3
    assert policy._messages[0]["role"] == "system"
    assert policy._messages[1]["role"] == "user"
    assert policy._messages[2]["role"] == "assistant"


def _json_response(body: dict) -> object:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(body).encode()

    return Response()


def _tool_body(actions: list[str]) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "choose_actions",
                                "arguments": json.dumps({"actions": actions}),
                            }
                        }
                    ],
                }
            }
        ],
        "usage": {},
    }


def test_openrouter_react_keeps_history_across_turns(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_urlopen(request, timeout=None):
        captured.append(json.loads(request.data.decode()))
        return _json_response(_tool_body(["right", "do"]))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    policy = OpenRouterReAct(config_id="luna_med", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024, "context_token_budget": 16000, "compact_at": 0.7, "keep_recent_messages": 8, "keep_recent_frames": 2, "observation_mode": "text", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "parse_retries": 0, "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT})
    observation = {"valid_actions": ["right", "do"], "observation_text": "first"}
    policy.plan(observation)
    policy.plan({**observation, "observation_text": "second"})
    assert len(captured) == 2
    first_messages = captured[0]["messages"]
    second_messages = captured[1]["messages"]
    assert len(first_messages) == 2
    assert first_messages[0]["role"] == "system"
    assert "first" in first_messages[1]["content"]
    assert len(second_messages) == 4
    assert second_messages[1]["role"] == "user"
    assert second_messages[2]["role"] == "assistant"
    assert "second" in second_messages[3]["content"]
    assert captured[0]["stream"] is True


def test_openrouter_react_compacts_on_token_threshold_with_a_model_summary(monkeypatch) -> None:
    """Compaction fires on real prompt_tokens and the middle is model-written.

    Replaces the old every-16-turns behaviour. A turn count is a poor proxy once
    frames and long observations are in the transcript, and pasting truncated
    payloads is not a summary — it drops the map knowledge the agent needs.
    Mirrors craftax_gold.rs so a checkpoint is evaluated under the same policy
    its teacher data was collected with.
    """
    captured: list[dict] = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode())
        captured.append(body)
        if body.get("tools") is None:
            # the summarizer call
            return _json_response(
                {"choices": [{"message": {"content": "Trees west, stone north. Have wood."}}]}
            )
        # report a prompt size over the threshold (16000 * 0.7 = 11200)
        payload = _tool_body(["do"])
        payload["usage"] = {"prompt_tokens": 12000, "completion_tokens": 5, "total_tokens": 12005}
        return _json_response(payload)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    policy = OpenRouterReAct(
        config_id="luna_med",
        config={
            "model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024,
            "context_token_budget": 16000, "compact_at": 0.7,
            "keep_recent_messages": 8, "keep_recent_frames": 2,
            "observation_mode": "text",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "parse_retries": 0,
            "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT,
        },
    )
    observation = {"valid_actions": ["do"]}
    for index in range(12):
        policy.plan({**observation, "observation_text": f"obs-{index}"})

    assert policy.trace_data()["compact_count"] >= 1
    action_calls = [body for body in captured if body.get("tools") is not None]
    last_messages = action_calls[-1]["messages"]
    assert last_messages[1]["content"].startswith("[context compacted")
    assert "Trees west, stone north" in last_messages[1]["content"]
    # the mechanical paste is gone
    assert "earlier ReAct turns" not in last_messages[1]["content"]
    assert "obs-11" in str(last_messages[-1]["content"])


def test_openrouter_react_streams_token_deltas_and_skips_empty_reasoning(
    monkeypatch,
) -> None:
    sse = (
        'data: {"choices":[{"delta":{"reasoning":""}}]}\n\n'
        'data: {"choices":[{"delta":{"content":""}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"choose_actions","arguments":"{\\"actions\\":[\\"right\\",\\"do\\"]}"}}]}}]}\n\n'
        'data: {"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}\n\n'
        "data: [DONE]\n\n"
    )

    class StreamResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __init__(self) -> None:
            self._buf = sse.encode()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, n: int = -1) -> bytes:
            if n is None or n < 0:
                data, self._buf = self._buf, b""
                return data
            data, self._buf = self._buf[:n], self._buf[n:]
            return data

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: StreamResponse())
    deltas: list[dict] = []
    policy = OpenRouterReAct(config_id="luna_med", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 1024, "context_token_budget": 16000, "compact_at": 0.7, "keep_recent_messages": 8, "keep_recent_frames": 2, "observation_mode": "text", "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY", "parse_retries": 0, "system_prompt": CRAFTAX_REACT_SYSTEM_PROMPT})
    actions = policy.plan(
        {"valid_actions": ["right", "do"], "observation_text": "obs"},
        on_delta=deltas.append,
    )
    assert actions == ["right", "do"]
    assert [item["channel"] for item in deltas] == ["tool"]
    assert deltas[0]["delta"] is True
    assert deltas[0]["text"] == '{"actions":["right","do"]}'
    trace = policy.trace_data()
    assert trace["reasoning"] == ""
    assert trace["token_trace"] == "derived"
    assert trace["deltas_emitted"] == 1
    assert trace["usage"]["total_tokens"] == 13


def _tinker_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model": "tinker-infer:ckpt-4",
        "effort": "medium",
        "max_tokens": 1024,
        "context_token_budget": 16000,
        "compact_at": 0.7,
        "keep_recent_messages": 8,
        "keep_recent_frames": 2,
        "observation_mode": "text",
        "provider": "tinker",
        "api_key_env": "TINKER_API_KEY",
        "parse_retries": 0,
        "system_prompt": "You play Craftax. Reply with JSON only.",
        "sampler_path": "tinker://weights/ckpt-4",
        "tokenizer_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "sampler_ready_timeout_s": 0,
    }
    config.update(overrides)
    return config


def test_tinker_provider_requires_its_own_endpoint_fields() -> None:
    for missing in ("sampler_path", "tokenizer_id", "sampler_ready_timeout_s"):
        config = _tinker_config()
        del config[missing]
        with pytest.raises(PolicyConfigError) as excinfo:
            OpenRouterReAct(config_id="ckpt", config=config)
        assert missing in str(excinfo.value)
    # base_url is meaningless on this path and must not be demanded...
    config = _tinker_config()
    assert OpenRouterReAct(config_id="ckpt", config=config).base_url == ""
    # ...while the OpenRouter path still requires it.
    config = _tinker_config(provider="openrouter")
    with pytest.raises(PolicyConfigError) as excinfo:
        OpenRouterReAct(config_id="ckpt", config=config)
    assert "base_url" in str(excinfo.value)


def test_tinker_provider_refuses_frames_rather_than_dropping_them() -> None:
    # Silently ignoring the frame would measure a text policy and report it as
    # the multimodal one that was asked for.
    for mode in ("image", "both"):
        with pytest.raises(PolicyConfigError) as excinfo:
            OpenRouterReAct(config_id="ckpt", config=_tinker_config(observation_mode=mode))
        assert "cannot carry frames" in str(excinfo.value)


def test_tinker_sample_is_parsed_by_the_shared_action_parser(monkeypatch) -> None:
    policy = OpenRouterReAct(config_id="ckpt", config=_tinker_config())
    # The SFT targets are raw `{"actions":[...]}` text, and Tinker has no
    # tool-calling API — so the sample must survive the same parse path.
    monkeypatch.setattr(
        policy,
        "_tinker_sample",
        lambda api_key: {
            "choices": [{"message": {"content": '{"actions":["left","do","do"]}', "tool_calls": None}}],
            "usage": {"prompt_tokens": 41, "completion_tokens": 9, "total_tokens": 50},
        },
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-only")
    actions = policy.plan({"valid_actions": ["left", "do"], "observation_text": "obs"})
    assert actions == ["left", "do", "do"]
    assert policy.usage()["prompt_tokens"] == 41


def test_tinker_provider_names_the_missing_summary_instead_of_borrowing_a_model(
    monkeypatch,
) -> None:
    policy = OpenRouterReAct(config_id="ckpt", config=_tinker_config())
    called = False

    def _fail(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("summarizer must not issue an HTTP request")

    monkeypatch.setattr("urllib.request.urlopen", _fail)
    summary = policy._summarize("test-only", [{"role": "user", "content": "x"}])
    assert not called
    assert "summary unavailable" in summary and "tinker" in summary


def test_tinker_and_hosted_paths_honour_sampler_temperature() -> None:
    greedy = OpenRouterReAct(config_id="ckpt", config=_tinker_config())
    assert greedy.temperature == 0.0
    sampled = OpenRouterReAct(
        config_id="ckpt", config=_tinker_config(temperature=1.1)
    )
    assert sampled.temperature == 1.1


def test_hosted_sampler_config_does_not_require_tinker_sdk_fields() -> None:
    policy = OpenRouterReAct(
        config_id="train",
        config={
            "inference_target": {
                "provider": "tinker",
                "provider_endpoint_id": "https://sampler.example/v1/sample",
                "auth_bearer": "token-v1",
                "run_id": "cispo1",
                "checkpoint_id": "cispo1:checkpoint:1",
            },
            "temperature": 1.0,
            "max_tokens": 64,
            "policy_version": "cispo1:checkpoint:1",
        },
    )
    assert policy.temperature == 1.0
    assert policy.sampler_path == ""
    assert policy._inference_target is not None


def test_hosted_sampler_records_training_action_and_usage(monkeypatch) -> None:
    from synth_containers.training_rollout import SamplerResult

    policy = OpenRouterReAct(
        config_id="train",
        config={
            "inference_target": {
                "provider": "tinker",
                "provider_endpoint_id": "https://sampler.example/v1/sample",
                "auth_bearer": "token-v1",
                "run_id": "cispo1",
                "checkpoint_id": "cispo1:checkpoint:1",
                "rollout_id": "r1",
            },
            "temperature": 1.0,
            "max_tokens": 64,
            "policy_version": "cispo1:checkpoint:1",
        },
    )
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, endpoint, **kwargs):
            captured["allow_loopback_http"] = kwargs.get("allow_loopback_http")
            captured["url"] = endpoint.url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def sample(self, payload, *, idempotency_key):
            captured["payload"] = dict(payload)
            captured["idempotency_key"] = idempotency_key
            return SamplerResult(
                text='{"actions":["left","do"]}',
                prompt_token_ids=(1, 2, 3),
                token_ids=(4, 5),
                log_probs=(-0.1, -0.2),
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            )

    monkeypatch.setattr(
        "synth_containers.training_rollout.HostedSamplerClient", FakeClient
    )
    body = policy._hosted_sampler_sample()
    assert captured["payload"]["temperature"] == 1.0
    assert captured["url"] == "https://sampler.example/v1/sample"
    assert body["usage"]["prompt_tokens"] == 3
    assert policy.last_call_usage["completion_tokens"] == 2
    assert policy.last_training_action["token_ids"] == [4, 5]
