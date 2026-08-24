"""`mini_swe`: chat completions in, one bash command per turn, workspace out."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from synth_containers.policies import build_planner
from synth_containers.policies.mini_swe import MiniSweAgent, _extract_command


class Fake:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []


def _serve_messages(state: Fake) -> tuple[HTTPServer, str]:
    """Like `_serve`, but each reply is a full assistant MESSAGE object."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            state.requests.append({"path": self.path, "body": body,
                                   "authorization": self.headers.get("Authorization")})
            message = state.replies.pop(0) if state.replies else {"content": "```bash\necho MINI_SWE_DONE\n```"}
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


def _serve(state: Fake) -> tuple[HTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            state.requests.append({"path": self.path, "body": body,
                                   "authorization": self.headers.get("Authorization")})
            content = state.replies.pop(0) if state.replies else "```bash\necho MINI_SWE_DONE\n```"
            payload = json.dumps(
                {
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
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


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "policy.py").write_text("BASELINE = 1\n")
    return tmp_path


def _agent(base_url: str, workspace: Path, **overrides: Any) -> MiniSweAgent:
    config = {
        "model": "laguna-xs",
        "base_url": base_url,
        "api_key_env": "MINI_SWE_TEST_KEY",
        "workspace_root": str(workspace),
        "max_steps": 4,
        "objective": "Improve the policy.",
    }
    config.update(overrides)
    return build_planner("mini_swe", config_id="rogue_mini_swe", config=config)


def test_it_runs_commands_in_the_workspace_and_stops_on_the_marker(monkeypatch, workspace) -> None:
    state = Fake(
        [
            "```bash\ncat policy.py\n```",
            "```bash\nprintf 'BASELINE = 2\\n' > policy.py\n```",
            "```bash\necho MINI_SWE_DONE\n```",
        ]
    )
    server, base_url = _serve(state)
    monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-test")
    agent = _agent(base_url, workspace)
    deltas: list[dict[str, Any]] = []
    try:
        actions = agent.plan(
            {"observation_text": "make it 2", "valid_actions": ["done"], "workspace": str(workspace)},
            on_delta=deltas.append,
        )
    finally:
        server.shutdown()

    assert actions == ["done"]
    assert (workspace / "policy.py").read_text() == "BASELINE = 2\n"
    assert agent.usage() == {"prompt_tokens": 15, "completion_tokens": 6, "total_tokens": 21,
                             "calls": 3, "commands": 2}
    assert agent.trace_data()["finished"] is True
    assert [delta["exit_code"] for delta in deltas if "exit_code" in delta] == [0, 0]
    # Command output is fed back as the next user turn, so the loop is real.
    last_request = state.requests[-1]["body"]["messages"]
    assert any("BASELINE = 1" in str(message.get("content")) for message in last_request)


def test_it_is_a_chat_completions_harness_and_carries_the_bound_credential(monkeypatch, workspace) -> None:
    state = Fake(["```bash\necho MINI_SWE_DONE\n```"])
    server, base_url = _serve(state)
    monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-bound-to-this-prid")
    agent = _agent(base_url, workspace)
    try:
        agent.plan({"observation_text": "t", "valid_actions": ["done"], "workspace": str(workspace)})
    finally:
        server.shutdown()
    assert state.requests[0]["path"].endswith("/chat/completions")
    assert state.requests[0]["authorization"] == "Bearer sk-bound-to-this-prid"
    # No logprob request, no extra_body: the proxy is the training authority.
    assert "logprobs" not in state.requests[0]["body"]
    assert agent.metadata()["wire_api"] == "chat_completions"


def test_a_missing_credential_or_workspace_refuses(monkeypatch, workspace) -> None:
    state = Fake([])
    server, base_url = _serve(state)
    monkeypatch.delenv("MINI_SWE_TEST_KEY", raising=False)
    agent = _agent(base_url, workspace)
    try:
        with pytest.raises(RuntimeError, match="MINI_SWE_TEST_KEY"):
            agent.plan({"observation_text": "t", "valid_actions": ["done"]})
        monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-test")
        with pytest.raises(RuntimeError, match="workspace_missing"):
            agent.plan(
                {"observation_text": "t", "valid_actions": ["done"], "workspace": str(workspace / "nope")}
            )
    finally:
        server.shutdown()


def test_a_reply_without_a_command_is_reprompted_not_executed(monkeypatch, workspace) -> None:
    state = Fake(["I will now think about it.", "```bash\necho MINI_SWE_DONE\n```"])
    server, base_url = _serve(state)
    monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-test")
    agent = _agent(base_url, workspace)
    try:
        agent.plan({"observation_text": "t", "valid_actions": ["done"], "workspace": str(workspace)})
    finally:
        server.shutdown()
    assert agent.usage()["commands"] == 0
    assert any(
        "exactly one fenced bash block" in str(message.get("content"))
        for message in state.requests[-1]["body"]["messages"]
    )


def test_a_hanging_command_times_out_without_killing_the_episode(monkeypatch, workspace) -> None:
    state = Fake(["```bash\nsleep 5\n```", "```bash\necho MINI_SWE_DONE\n```"])
    server, base_url = _serve(state)
    monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-test")
    agent = _agent(base_url, workspace, command_timeout_seconds=0.5)
    try:
        actions = agent.plan(
            {"observation_text": "t", "valid_actions": ["done"], "workspace": str(workspace)}
        )
    finally:
        server.shutdown()
    assert actions == ["done"]
    assert agent.trace_data()["steps"][0]["exit_code"] == 124


def test_the_step_budget_bounds_the_loop(monkeypatch, workspace) -> None:
    state = Fake(["```bash\ntrue\n```"] * 10)
    server, base_url = _serve(state)
    monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-test")
    agent = _agent(base_url, workspace, max_steps=3)
    try:
        agent.plan({"observation_text": "t", "valid_actions": ["done"], "workspace": str(workspace)})
    finally:
        server.shutdown()
    assert agent.usage()["calls"] == 3
    assert agent.trace_data()["finished"] is False


@pytest.mark.parametrize(
    "content,expected",
    [
        ("```bash\nls -la\n```", "ls -la"),
        ("prose\n```\npwd\n```\nmore", "pwd"),
        ("```sh\necho hi\n```", "echo hi"),
        ("no block here", None),
        ("```bash\n\n```", None),
    ],
)
def test_command_extraction(content: str, expected: str | None) -> None:
    assert _extract_command(content) == expected


def test_a_native_tool_call_counts_as_the_command(monkeypatch, workspace) -> None:
    """A model that calls a `bash` tool has already said what to run."""

    state = Fake(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": "printf 'X' > touched.txt"}),
                        },
                    }
                ],
            },
            {"content": "```bash\necho MINI_SWE_DONE\n```"},
        ]
    )
    server, base_url = _serve_messages(state)
    monkeypatch.setenv("MINI_SWE_TEST_KEY", "sk-test")
    agent = _agent(base_url, workspace)
    try:
        agent.plan({"observation_text": "t", "valid_actions": ["done"], "workspace": str(workspace)})
    finally:
        server.shutdown()
    assert (workspace / "touched.txt").read_text() == "X"
    assert agent.usage()["commands"] == 1
