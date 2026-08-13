"""End-to-end ReAct and code-policy evals through Containers HTTP."""

from __future__ import annotations

import json

from synth_containers.platform.eval_examples import run_code_policy, run_react
from synth_containers.platform.react import OpenRouterReAct


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
        config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 768, "compact_every": 16, "system_prompt": "Prioritize wood before stone."},
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
    policy = OpenRouterReAct(config_id="muse_spark_medium", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 768, "compact_every": 16, })
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
    assert trace["compact_every"] == 16
    assert trace["compact_count"] == 0
    assert trace["history_turns"] == 1
    assert trace["deltas_emitted"] == 0
    assert trace["token_trace"] is None
    assert trace["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "cost_usd": None,
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
    policy = OpenRouterReAct(config_id="muse_spark_medium", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 768, "compact_every": 16, })
    assert policy.plan({"valid_actions": ["right", "do"], "observation_text": "obs"}) == [
        "right",
        "do",
    ]
    trace = policy.trace_data()
    assert trace["action_authority"] == "policy"
    assert trace["fallback"] is False
    assert trace["tool_arguments"] == '{"actions":["east","do"]}'
    assert trace["compact_every"] == 16
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
    policy = OpenRouterReAct(config_id="luna_med", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 768, "compact_every": 16})
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


def test_openrouter_react_compacts_every_sixteen_turns(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_urlopen(request, timeout=None):
        captured.append(json.loads(request.data.decode()))
        return _json_response(_tool_body(["do"]))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    policy = OpenRouterReAct(config_id="luna_med", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 768, "compact_every": 16})
    observation = {"valid_actions": ["do"], "observation_text": "obs"}
    for index in range(17):
        policy.plan({**observation, "observation_text": f"obs-{index}"})
    assert policy.trace_data()["compact_count"] == 1
    last_messages = captured[-1]["messages"]
    assert last_messages[1]["content"].startswith("[compacted 14 earlier ReAct turns")
    assert "compact_every=16" in last_messages[1]["content"]
    user_turns = [message for message in last_messages if message["role"] == "user"]
    assert len(user_turns) == 4  # compact + 2 kept + current
    assert "obs-16" in last_messages[-1]["content"]


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
    policy = OpenRouterReAct(config_id="luna_med", config={"model": "gpt-5.6-luna", "effort": "medium", "max_tokens": 768, "compact_every": 16})
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
