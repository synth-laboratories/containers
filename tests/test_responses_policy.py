"""`ResponsesReAct`: Responses-shaped request, tool-call authority, chained state.

There is no live OpenAI call here. What is asserted is the wire contract the
harness owes the Responses API — `input`/`tools`, `previous_response_id`
carrying multi-turn state, the `function_call` item outranking message text —
against a stub server. A paid proof is a separate, credentialed run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from synth_containers.policies.responses import TOOL_NAME, ResponsesReAct

OBSERVATION = {"observation_text": "a room", "valid_actions": ["north", "south", "wait"]}


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _stub(monkeypatch: pytest.MonkeyPatch, *payloads: dict[str, Any]) -> list[dict[str, Any]]:
    """Record each request body and reply with the next canned payload."""

    seen: list[dict[str, Any]] = []
    queue = list(payloads)

    def urlopen(request: Any, *_args: object, **_kwargs: object) -> _Response:
        seen.append(json.loads(request.data.decode("utf-8")))
        return _Response(queue.pop(0) if queue else {})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return seen


def _function_call(actions: list[str]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": TOOL_NAME,
                "arguments": json.dumps({"actions": actions}),
            }
        ],
        "usage": {"input_tokens": 40, "output_tokens": 6, "total_tokens": 46},
    }


def _policy(**config: Any) -> ResponsesReAct:
    return ResponsesReAct(config_id="responses_react_gpt", config={"model": "gpt-5.6", **config})


def test_request_is_responses_shaped_not_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = _stub(monkeypatch, _function_call(["north"]))
    assert _policy().plan(OBSERVATION) == ["north"]
    body = seen[0]
    assert "input" in body and "messages" not in body
    assert [tool["name"] for tool in body["tools"]] == [TOOL_NAME]
    assert body["model"] == "gpt-5.6"


def test_function_call_outranks_message_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = _function_call(["south"])
    payload["output"].insert(
        0,
        {
            "type": "message",
            "content": [{"type": "output_text", "text": '{"actions":["north"]}'}],
        },
    )
    _stub(monkeypatch, payload)
    policy = _policy()
    assert policy.plan(OBSERVATION) == ["south"]
    # "policy" is the forced tool call; "policy_text" is the JSON fallback.
    assert policy.trace_data()["action_authority"] == "policy"


def test_message_text_is_the_fallback_when_no_tool_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub(
        monkeypatch,
        {
            "id": "resp_2",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"actions":["wait"]}'}],
                }
            ],
        },
    )
    policy = _policy()
    assert policy.plan(OBSERVATION) == ["wait"]
    assert policy.trace_data()["action_authority"] == "policy_text"


def test_illegal_actions_are_dropped_and_an_empty_plan_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _stub(monkeypatch, _function_call(["fly", "teleport"]))
    with pytest.raises(RuntimeError, match="no valid actions"):
        _policy().plan(OBSERVATION)


def test_state_is_carried_by_previous_response_id_only_when_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = _stub(monkeypatch, _function_call(["north"]), _function_call(["south"]))
    policy = _policy(store=True)
    policy.plan(OBSERVATION)
    policy.plan(OBSERVATION)
    assert "previous_response_id" not in seen[0]
    assert seen[1]["previous_response_id"] == "resp_1"

    seen_unstored = _stub(monkeypatch, _function_call(["north"]), _function_call(["south"]))
    unstored = _policy(store=False)
    unstored.plan(OBSERVATION)
    unstored.plan(OBSERVATION)
    # Without server-side storage there is no chain to point at; sending a stale
    # id would branch the next turn off a context the server no longer holds.
    assert all("previous_response_id" not in body for body in seen_unstored)


def test_a_missing_key_refuses_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = _stub(monkeypatch, _function_call(["north"]))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _policy().plan(OBSERVATION)
    assert seen == []


def test_restore_without_store_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(store=False)
    with pytest.raises(RuntimeError, match="requires_store"):
        policy.restore_checkpoint_state(policy.checkpoint_state())
