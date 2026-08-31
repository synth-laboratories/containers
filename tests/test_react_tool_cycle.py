"""`observation_role="tool"`: the ReAct loop as a real tool cycle.

The default shape — assistant content, then a user observation — is what this
harness has always sent, and it stays the default because every image and
provider shares this file. Opting in makes the loop protocol-correct: the model's
call is echoed structurally and the observation answers it, which is what lets a
training proxy keep the sampled tokens as a prefix instead of re-rendering the
whole history every turn.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from synth_containers.policies import build_planner


class Fake:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []


def _serve(state: Fake) -> tuple[HTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            state.requests.append(body)
            message = state.replies.pop(0) if state.replies else {"content": '{"actions":["do"]}'}
            payload = json.dumps(
                {
                    "choices": [{"index": 0, "message": {"role": "assistant", **message}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def _tool_call(actions: list[str], call_id: str = "call_abc") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "choose_actions",
                    "arguments": json.dumps({"actions": actions}),
                },
            }
        ],
    }


def _observation(step: int) -> dict[str, Any]:
    return {
        "valid_actions": ["do", "left", "right"],
        "observation_text": f"grid at step {step}",
    }


def _planner(base_url: str, **overrides: Any) -> Any:
    config = {
        "model": "gpt-oss-20b",
        "base_url": base_url,
        "api_key_env": "REACT_TEST_KEY",
        "max_tokens": 128,
        "env_name": "craftax",
    }
    config.update(overrides)
    return build_planner("react", config_id="tito_react", config=config)


@pytest.fixture()
def key(monkeypatch) -> None:
    monkeypatch.setenv("REACT_TEST_KEY", "sk-test")


def test_a_tool_call_is_echoed_structurally_and_the_observation_answers_it(key) -> None:
    state = Fake([_tool_call(["do"]), _tool_call(["left"])])
    server, base_url = _serve(state)
    planner = _planner(base_url, observation_role="tool")
    try:
        assert planner.plan(_observation(1)) == ["do"]
        assert planner.plan(_observation(2)) == ["left"]
    finally:
        server.shutdown()

    messages = state.requests[-1]["messages"]
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "tool"]
    # The call is carried as a call, not flattened into content.
    assistant = messages[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "choose_actions"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"actions": ["do"]}
    # The observation answers that exact call.
    tool_message = messages[3]
    assert tool_message["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "grid at step 2" in tool_message["content"]


def test_the_default_shape_is_unchanged(key) -> None:
    """Existing lanes must not move because of this flag."""

    state = Fake([_tool_call(["do"]), _tool_call(["left"])])
    server, base_url = _serve(state)
    planner = _planner(base_url)
    try:
        planner.plan(_observation(1))
        planner.plan(_observation(2))
    finally:
        server.shutdown()

    messages = state.requests[-1]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert "tool_calls" not in messages[2]
    assert planner.metadata()["observation_role"] == "user"


def test_a_json_content_answer_has_no_call_to_answer(key) -> None:
    """The fallback path stays a user turn even when the flag is on."""

    state = Fake([{"content": '{"actions":["do"]}'}, {"content": '{"actions":["left"]}'}])
    server, base_url = _serve(state)
    planner = _planner(base_url, observation_role="tool")
    try:
        planner.plan(_observation(1))
        planner.plan(_observation(2))
    finally:
        server.shutdown()

    messages = state.requests[-1]["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == '{"actions":["do"]}'


def test_the_pending_call_survives_a_checkpoint(key) -> None:
    state = Fake([_tool_call(["do"], call_id="call_xyz")])
    server, base_url = _serve(state)
    planner = _planner(base_url, observation_role="tool")
    try:
        planner.plan(_observation(1))
    finally:
        server.shutdown()

    checkpoint = planner.checkpoint_state()
    assert checkpoint["pending_tool_call_id"] == "call_xyz"

    restored = _planner(base_url, observation_role="tool")
    restored.restore_checkpoint_state(checkpoint)
    assert restored._pending_tool_call_id == "call_xyz"


def test_an_unknown_observation_role_refuses(key) -> None:
    with pytest.raises(RuntimeError, match="unsupported observation_role"):
        _planner("http://x/v1", observation_role="function")
