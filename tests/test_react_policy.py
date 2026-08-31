"""`OpenRouterReAct`: action parsing, prompt binding, history, streaming.

The harness is shared by every image that binds a chat-completions policy,
so its tests live with the harness rather than with one image.
"""

from __future__ import annotations

import json

from synth_containers.policies.react import OpenRouterReAct

def test_openrouter_react_normalizes_craftax_direction_aliases() -> None:
    actions = OpenRouterReAct._parse_actions(
        '{"actions":["North","east","do"]}',
        ["up", "right", "do"],
    )
    assert actions == ["up", "right", "do"]


def test_openrouter_react_binds_candidate_system_prompt() -> None:
    policy = OpenRouterReAct(
        config_id="goex_candidate_test",
        config={"system_prompt": "Prioritize wood before stone."},
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
    policy = OpenRouterReAct(config_id="muse_spark_medium", config={})
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
    policy = OpenRouterReAct(config_id="muse_spark_medium", config={})
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
    policy = OpenRouterReAct(config_id="luna_med", config={"compact_every": 16})
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
    policy = OpenRouterReAct(config_id="luna_med", config={"compact_every": 16})
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
    policy = OpenRouterReAct(config_id="luna_med", config={})
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


def _tool_body_with_usage(actions: list[str], prompt_tokens: int) -> dict:
    body = _tool_body(actions)
    body["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": 8}
    return body


def test_openrouter_react_compacts_when_the_prompt_passes_the_token_budget(monkeypatch) -> None:
    """A turn count only predicts context length if every turn is the same size.

    A model that keeps its own reasoning in history grows the prompt by whatever
    it wrote, so the thing that runs out is tokens, not turns.
    """

    captured: list[dict] = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode())
        captured.append(body)
        # A stand-in for a real provider: the prompt costs what the history
        # holds, so a compaction actually brings the number back down.
        return _json_response(_tool_body_with_usage(["do"], 1000 * len(body["messages"])))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    policy = OpenRouterReAct(
        config_id="luna_med", config={"compact_every": 64, "compact_at_tokens": 8000}
    )
    observation = {"valid_actions": ["do"], "observation_text": "obs"}
    for index in range(8):
        policy.plan({**observation, "observation_text": f"obs-{index}"})

    # The turn rule cannot fire in 8 turns; the token budget does, and the
    # history it rebuilds is genuinely shorter than the one it replaced.
    assert policy.trace_data()["compact_count"] >= 1
    lengths = [len(body["messages"]) for body in captured]
    assert any(
        later < earlier for earlier, later in zip(lengths, lengths[1:])
    ), f"compaction never shrank the history: {lengths}"

    notice = captured[-1]["messages"][1]["content"]
    assert notice.startswith("[compacted ")
    assert "trigger=tokens" in notice
    assert "compact_at_tokens=8000" in notice

    # Control: the same eight turns with only the turn rule never compact, and
    # the prompt sails past the budget.
    turns_only = OpenRouterReAct(
        config_id="luna_med", config={"compact_every": 64, "compact_at_tokens": 0}
    )
    for index in range(8):
        turns_only.plan({**observation, "observation_text": f"obs-{index}"})
    assert turns_only.trace_data()["compact_count"] == 0
    assert turns_only.trace_data()["prompt_tokens"] > 8000


def test_openrouter_react_compacts_a_tool_cycle_history(monkeypatch) -> None:
    """Pairing on `user` alone matched nothing when observations are tool results.

    `observation_role="tool"` is what lets a training proxy keep the sampled
    tokens as a prefix, and it silently disabled compaction entirely.
    """

    captured: list[dict] = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode())
        captured.append(body)
        return _json_response(_tool_body_with_usage(["do"], 1000 * len(body["messages"])))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    policy = OpenRouterReAct(
        config_id="luna_med",
        config={"compact_every": 64, "compact_at_tokens": 8000, "observation_role": "tool"},
    )
    observation = {"valid_actions": ["do"], "observation_text": "obs"}
    for index in range(6):
        policy.plan({**observation, "observation_text": f"obs-{index}"})

    assert policy.trace_data()["compact_count"] >= 1
    messages = captured[-1]["messages"]
    # Every surviving tool result must answer a call that is still in history.
    live_calls = {
        str(call.get("id") or "")
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or ())
    }
    orphans = [
        message
        for message in messages
        if message.get("role") == "tool" and str(message.get("tool_call_id") or "") not in live_calls
    ]
    assert orphans == []


def test_openrouter_react_token_budget_survives_a_checkpoint(monkeypatch) -> None:
    def fake_urlopen(request, timeout=None):
        return _json_response(_tool_body_with_usage(["do"], 7777))

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    policy = OpenRouterReAct(config_id="luna_med", config={"compact_at_tokens": 8000})
    policy.plan({"valid_actions": ["do"], "observation_text": "obs"})
    state = policy.checkpoint_state()
    assert state["last_prompt_tokens"] == 7777

    resumed = OpenRouterReAct(config_id="luna_med", config={"compact_at_tokens": 8000})
    resumed.restore_checkpoint_state(state)
    assert resumed._last_prompt_tokens == 7777
    # A checkpoint written before token-triggered compaction still restores.
    legacy = dict(state)
    legacy.pop("last_prompt_tokens")
    resumed.restore_checkpoint_state(legacy)
    assert resumed._last_prompt_tokens == 0
