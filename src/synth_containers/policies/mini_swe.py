"""mini-SWE — the smallest real coding agent: chat completions plus bash.

Rung between `react` and `codex_agentic`. One model call per turn, one shell
command per turn, no tool-calling API, no CLI to install, no session state other
than the message list. That minimalism is the point: it is the harness that can
be pointed at any `/v1/chat/completions` endpoint, including a training proxy,
without the endpoint having to reproduce Codex semantics.

Contract with the nested (Harbor) runtime, same as `codex_agentic`:

    observation = {observation_text, valid_actions, workspace}
    plan(...)   mutates the workspace in place and returns the action list

The environment container is the isolation boundary; commands run with the
workspace as cwd, under a per-command timeout and a step budget. This harness
never sees a reward, a token id, or a log-probability — it is the policy half of
the split and nothing else.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

DeltaCallback = Callable[[dict[str, Any]], None]

HARNESS = "mini_swe"
FINISH_MARKER = "MINI_SWE_DONE"

_COMMAND_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """You are a careful software engineer working in a Linux workspace.

Each turn you may run exactly ONE shell command. Reply with a single fenced bash
block and nothing else:

```bash
your command here
```

The command runs with the workspace as the working directory and its combined
output is returned to you. Work in small steps: look before you edit, edit with
a heredoc or a patch, then re-check. When the task is complete, reply with a
fenced block containing exactly:

```bash
echo MINI_SWE_DONE
```
"""


class MiniSweAgent:
    """Chat-completions agent loop. `/v1/responses` is deliberately unsupported."""

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self.model = str(config.get("model") or "")
        if not self.model:
            raise RuntimeError("mini_swe requires policy config `model`")
        self.base_url = str(config.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise RuntimeError("mini_swe requires policy config `base_url`")
        self.api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
        self.workspace = Path(str(config.get("workspace_root") or ".")).expanduser()
        self.max_steps = min(max(int(config.get("max_steps") or 20), 1), 200)
        self.max_tokens = min(max(int(config.get("max_tokens") or 1024), 64), 16384)
        self.command_timeout = float(config.get("command_timeout_seconds") or 120.0)
        self.output_limit = int(config.get("output_limit") or 4000)
        self.request_timeout = float(config.get("timeout_seconds") or 900.0)
        self.objective = str(config.get("objective") or "").strip()
        self.calls = 0
        self.commands = 0
        self._usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._last_trace: dict[str, Any] = {}
        self._transcript: list[dict[str, Any]] = []

    # ------------------------------------------------------------- contract

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": HARNESS,
            "kind": "mini_swe_bash",
            "config": self.config_id,
            "model": self.model,
            "wire_api": "chat_completions",
            "max_steps": self.max_steps,
            "command_timeout_seconds": self.command_timeout,
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {**self._usage, "calls": self.calls, "commands": self.commands}

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    # ----------------------------------------------------------------- loop

    def plan(
        self, observation: dict[str, Any], on_delta: DeltaCallback | None = None
    ) -> list[str]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"mini_swe requires {self.api_key_env}")
        workspace = Path(str(observation.get("workspace") or self.workspace)).expanduser()
        if not workspace.is_dir():
            raise RuntimeError(f"mini_swe_workspace_missing:{workspace}")
        task = str(observation.get("observation_text") or "").strip()
        opening = f"{self.objective}\n\n{task}".strip() if self.objective else task
        self._messages.append(
            {
                "role": "user",
                "content": (
                    f"{opening}\n\nWorkspace: {workspace}\n"
                    f"You have at most {self.max_steps} commands."
                ),
            }
        )

        finished = False
        for step in range(self.max_steps):
            message = self._complete(api_key)
            content = _message_text(message.get("content"))
            command = _command_from_message(message)
            self._transcript.append({"step": step, "assistant": content[:2000], "command": command})
            if command is None:
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            "No command found. Reply with exactly one fenced bash block."
                        ),
                    }
                )
                if on_delta is not None:
                    on_delta({"channel": "mini_swe", "step": step, "parse_error": "no_command"})
                continue
            if FINISH_MARKER in command:
                finished = True
                if on_delta is not None:
                    on_delta({"channel": "mini_swe", "step": step, "event": "finished"})
                break
            result = self._run(command, workspace)
            self.commands += 1
            self._messages.append({"role": "user", "content": _observation_text(result, self.output_limit)})
            self._transcript[-1]["exit_code"] = result["exit_code"]
            if on_delta is not None:
                on_delta(
                    {
                        "channel": "mini_swe",
                        "step": step,
                        "command": command[:400],
                        "exit_code": result["exit_code"],
                        "duration_seconds": result["duration_seconds"],
                    }
                )

        self._last_trace = {
            "harness": HARNESS,
            "model": self.model,
            "calls": self.calls,
            "commands": self.commands,
            "finished": finished,
            "steps": self._transcript[-12:],
            "usage": dict(self._usage),
        }
        valid = [str(action) for action in observation.get("valid_actions") or ("done",)]
        return [valid[0]] if valid else ["done"]

    # -------------------------------------------------------------- helpers

    def _complete(self, api_key: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": list(self._messages),
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"mini_swe policy HTTP {exc.code}: {detail}") from exc
        self.calls += 1
        message = (body.get("choices") or [{}])[0].get("message") or {}
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                self._usage[key] = int(self._usage.get(key) or 0) + value
        self._messages.append({"role": "assistant", "content": _message_text(message.get("content"))})
        return message

    def _run(self, command: str, workspace: Path) -> dict[str, Any]:
        import time

        started = time.time()
        try:
            completed = subprocess.run(  # noqa: S602 - the container is the boundary
                ["bash", "-lc", command],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
            return {
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_seconds": round(time.time() - started, 3),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": 124,
                "stdout": "",
                "stderr": f"command timed out after {self.command_timeout}s",
                "duration_seconds": round(time.time() - started, 3),
                "timed_out": True,
            }


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)


def _command_from_message(message: dict[str, Any]) -> str | None:
    """The command this turn asked for, however the model chose to say it.

    mini-SWE asks for a fenced bash block because that works with any endpoint.
    A model trained to emit a native tool call will do that instead, and
    ignoring it burns the turn on a reprompt for something the model already
    said. Both forms mean the same thing, so both are accepted.
    """

    for call in message.get("tool_calls") or ():
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = str(function.get("name") or "").strip().lower()
        if name not in {"bash", "shell", "sh", "run", "execute"}:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"command": arguments}
        if isinstance(arguments, dict):
            for key in ("command", "cmd", "script", "input"):
                value = arguments.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return _extract_command(_message_text(message.get("content")))


def _extract_command(content: str) -> str | None:
    blocks = _COMMAND_BLOCK.findall(content or "")
    if not blocks:
        return None
    command = blocks[0].strip()
    return command or None


def _observation_text(result: dict[str, Any], limit: int) -> str:
    stdout = _truncate(result["stdout"], limit)
    stderr = _truncate(result["stderr"], limit // 4)
    return (
        f"exit_code={result['exit_code']} duration={result['duration_seconds']}s\n"
        f"stdout:\n{stdout or '(empty)'}\n"
        f"stderr:\n{stderr or '(empty)'}"
    )


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return f"{head}\n... [{len(text) - limit} chars elided] ...\n{tail}"
