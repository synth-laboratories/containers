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
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

DeltaCallback = Callable[[dict[str, Any]], None]

_TINKER_RUNTIMES: dict[str, tuple[Any, Any, Any, Any]] = {}
_TINKER_RUNTIME_LOCK = threading.Lock()

HARNESS = "mini_swe"
FINISH_MARKER = "MINI_SWE_DONE"

_COMMAND_BLOCK = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)
_HARMONY_EXEC_CALL = re.compile(
    r"to=(?:container\.)?exec(?:\s+[^<]*)?<\|message\|>(\{.*?\})(?:<\|call\|>|$)",
    re.DOTALL,
)

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
        self.model_path = str(config.get("model_path") or "").strip()
        self.inference_transport = str(config.get("inference_transport") or "chat_completions")
        if self.inference_transport not in {"chat_completions", "codex_cli", "tinker_sampler"}:
            raise RuntimeError(f"mini_swe_inference_transport_invalid:{self.inference_transport}")
        # Optional provider-side reasoning effort (OpenAI-style `reasoning.effort`,
        # honoured by OpenRouter). Unset means the provider default; the pin must
        # say "medium" explicitly for a luna-medium run to be one.
        self.reasoning_effort = str(config.get("reasoning_effort") or "").strip().lower()
        if self.reasoning_effort and self.reasoning_effort not in {"minimal", "low", "medium", "high", "xhigh"}:
            raise RuntimeError(f"mini_swe_reasoning_effort_invalid:{self.reasoning_effort}")
        if not self.base_url and self.inference_transport != "codex_cli":
            raise RuntimeError("mini_swe requires policy config `base_url`")
        self.api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
        self.workspace = Path(str(config.get("workspace_root") or ".")).expanduser()
        self.max_steps = min(max(int(config.get("max_steps") or 20), 1), 200)
        self.max_recovery_turns = min(
            max(int(config.get("max_recovery_turns") or 12), 0), 50
        )
        self.max_tokens = min(max(int(config.get("max_tokens") or 1024), 64), 16384)
        self.temperature = max(float(config.get("temperature", 0.0)), 0.0)
        self.sampling_seed = int(config.get("sampling_seed", 0))
        self.command_timeout = float(config.get("command_timeout_seconds") or 120.0)
        self.output_limit = int(config.get("output_limit") or 4000)
        self.workspace_aliases = tuple(
            str(item).rstrip("/")
            for item in (config.get("workspace_aliases") or ())
            if str(item).startswith("/") and str(item).rstrip("/")
        )
        self.compaction_threshold_tokens = max(
            int(config.get("compaction_threshold_tokens") or 0), 0
        )
        self.compaction_keep_messages = min(
            max(int(config.get("compaction_keep_messages") or 8), 2), 40
        )
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
        self._last_prompt_tokens = 0
        self._compactions = 0
        self._tinker_runtime: tuple[Any, Any, Any, Any] | None = None
        self._sampling_seconds = 0.0
        self._sampling_calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------- contract

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": HARNESS,
            "kind": "mini_swe_bash",
            "config": self.config_id,
            "model": self.model,
            "model_path_digest": (
                __import__("hashlib").sha256(self.model_path.encode()).hexdigest()
                if self.model_path else None
            ),
            "wire_api": "chat_completions",
            "inference_transport": self.inference_transport,
            "reasoning_effort": self.reasoning_effort or None,
            "max_steps": self.max_steps,
            "command_timeout_seconds": self.command_timeout,
            "compaction_threshold_tokens": self.compaction_threshold_tokens or None,
            "compaction_keep_messages": self.compaction_keep_messages,
            "workspace_aliases": list(self.workspace_aliases),
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        usage = {**self._usage, "calls": self.calls, "commands": self.commands}
        if self._sampling_seconds > 0:
            usage["sampling_seconds"] = round(self._sampling_seconds, 6)
            usage["completion_tokens_per_second"] = round(
                int(self._usage.get("completion_tokens") or 0) / self._sampling_seconds,
                3,
            )
        return usage

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    # ----------------------------------------------------------------- loop

    def plan(
        self, observation: dict[str, Any], on_delta: DeltaCallback | None = None
    ) -> list[str]:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if self.inference_transport != "codex_cli" and not api_key:
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
        command_step = 0
        recovery_turns = 0
        while command_step < self.max_steps:
            try:
                message = self._complete(api_key)
            except Exception as exc:
                recovery_turns += 1
                if on_delta is not None:
                    on_delta(
                        {
                            "channel": "mini_swe",
                            "step": command_step,
                            "inference_error": type(exc).__name__,
                            "inference_error_detail": str(exc)[:400],
                            "recovery_turn": recovery_turns,
                        }
                    )
                if recovery_turns > self.max_recovery_turns:
                    break
                continue
            content = _message_text(message.get("content"))
            command = _command_from_message(message)
            rejection = _command_rejection(command)
            self._transcript.append(
                {"step": command_step, "assistant": content[:2000], "command": command}
            )
            if command is None or rejection is not None:
                recovery_turns += 1
                self._messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply did not contain a safe, recognizable shell "
                            "command. Do not emit prose, raw data, YAML, or file contents as the "
                            "command. Reply with exactly one fenced bash block containing an "
                            "actual shell command."
                        ),
                    }
                )
                if on_delta is not None:
                    on_delta(
                        {
                            "channel": "mini_swe",
                            "step": command_step,
                            "parse_error": rejection or "no_command",
                            "recovery_turn": recovery_turns,
                        }
                    )
                if recovery_turns > self.max_recovery_turns:
                    break
                continue
            if FINISH_MARKER in command:
                finished = True
                if on_delta is not None:
                    on_delta({"channel": "mini_swe", "step": command_step, "event": "finished"})
                break
            result = self._run(command, workspace)
            self.commands += 1
            self._messages.append({"role": "user", "content": _observation_text(result, self.output_limit)})
            self._maybe_compact(on_delta=on_delta, step=command_step)
            self._transcript[-1]["exit_code"] = result["exit_code"]
            if on_delta is not None:
                on_delta(
                    {
                        "channel": "mini_swe",
                        "step": command_step,
                        "command": command[:400],
                        "exit_code": result["exit_code"],
                        "duration_seconds": result["duration_seconds"],
                    }
                )
            command_step += 1

        self._last_trace = {
            "harness": HARNESS,
            "model": self.model,
            "calls": self.calls,
            "commands": self.commands,
            "finished": finished,
            "compactions": self._compactions,
            "recovery_turns": recovery_turns,
            "steps": self._transcript[-12:],
            "usage": dict(self._usage),
            "throughput": {
                "sampling_seconds": round(self._sampling_seconds, 6),
                "completion_tokens_per_second": round(
                    int(self._usage.get("completion_tokens") or 0) / self._sampling_seconds,
                    3,
                ) if self._sampling_seconds > 0 else None,
                "sample_calls": self._sampling_calls[-12:],
            },
        }
        valid = [str(action) for action in observation.get("valid_actions") or ("done",)]
        return [valid[0]] if valid else ["done"]

    # -------------------------------------------------------------- helpers

    def _complete(self, api_key: str) -> dict[str, Any]:
        if self.inference_transport == "codex_cli":
            return self._complete_codex_cli()
        if self.inference_transport == "tinker_sampler":
            return self._complete_tinker_sampler()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(self._messages),
            "max_tokens": self.max_tokens,
        }
        if self.base_url.startswith("https://tinker.thinkingmachines.dev/"):
            # Tinker separates reasoning by default. mini-SWE needs the complete
            # response inline because the shell command may appear in reasoning.
            payload["separate_reasoning"] = False
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
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
        if isinstance(usage.get("prompt_tokens"), int):
            self._last_prompt_tokens = int(usage["prompt_tokens"])
        capture = body.get("synth_capture")
        if isinstance(capture, dict):
            self._sampling_calls.append(dict(capture))
            seconds = capture.get("sampling_seconds")
            if isinstance(seconds, (int, float)):
                self._sampling_seconds += float(seconds)
        self._messages.append({"role": "assistant", "content": _message_text(message.get("content"))})
        return message

    def _complete_tinker_sampler(self) -> dict[str, Any]:
        """Sample a Tinker base model directly; no checkpoint URL is required."""

        if self._tinker_runtime is None:
            with _TINKER_RUNTIME_LOCK:
                runtime_key = self.model_path or self.model
                runtime = _TINKER_RUNTIMES.get(runtime_key)
                if runtime is None:
                    try:
                        import tinker
                    except ImportError as exc:
                        raise RuntimeError("mini_swe_tinker_sdk_missing") from exc
                    service = tinker.ServiceClient()
                    client = (
                        service.create_sampling_client(model_path=self.model_path)
                        if self.model_path
                        else service.create_sampling_client(base_model=self.model)
                    )
                    # A saved sampler path does not carry a resolvable HF model
                    # identity across processes. Tokenization is nevertheless
                    # defined by the immutable base model, while sampling uses
                    # the checkpoint-backed client.
                    tokenizer = (
                        service.create_sampling_client(base_model=self.model).get_tokenizer()
                        if self.model_path else client.get_tokenizer()
                    )
                    try:
                        from renderers import GptOssRendererConfig, create_renderer

                        renderer = create_renderer(
                            tokenizer,
                            GptOssRendererConfig(reasoning_effort="low"),
                        ) if self.model.startswith("openai/gpt-oss-") else create_renderer(tokenizer)
                    except ImportError as exc:
                        raise RuntimeError("mini_swe_prime_renderers_missing") from exc
                    runtime = (tinker, client, tokenizer, renderer)
                    _TINKER_RUNTIMES[runtime_key] = runtime
                self._tinker_runtime = runtime
        tinker, client, tokenizer, renderer = self._tinker_runtime
        prompt_ids = list(map(int, renderer.render_ids(self._messages, add_generation_prompt=True)))
        model_input_cls = getattr(tinker, "ModelInput", None) or tinker.types.ModelInput
        try:
            prompt = model_input_cls.from_ints(tokens=prompt_ids)
        except TypeError:
            prompt = model_input_cls.from_ints(prompt_ids)
        sample_started = time.monotonic()
        sampling_kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.sampling_seed + self.calls,
        }
        if self.model.startswith("openai/gpt-oss-"):
            sampling_kwargs["stop"] = "<|call|>"
        result = client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(**sampling_kwargs),
        ).result()
        sample_seconds = max(time.monotonic() - sample_started, 1e-9)
        token_ids = list(map(int, result.sequences[0].tokens))
        sequence_logprobs = [float(value) for value in (result.sequences[0].logprobs or ())]
        if len(sequence_logprobs) != len(token_ids):
            raise RuntimeError("mini_swe_tinker_logprob_alignment_failed")
        # Parse once to validate the family contract, but preserve the raw
        # sampled surface for the harness's Harmony/tool-call parser.
        renderer.parse_response(token_ids)
        content = tokenizer.decode(token_ids, skip_special_tokens=False)
        self.calls += 1
        self._last_prompt_tokens = len(prompt_ids)
        self._usage["prompt_tokens"] += len(prompt_ids)
        self._usage["completion_tokens"] += len(token_ids)
        self._usage["total_tokens"] += len(prompt_ids) + len(token_ids)
        self._sampling_seconds += sample_seconds
        self._sampling_calls.append(
            {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "sampling_seconds": round(sample_seconds, 6),
                "completion_tokens_per_second": round(len(token_ids) / sample_seconds, 3),
                "prompt_token_ids": prompt_ids,
                "generation_token_ids": token_ids,
                "generation_logprobs": sequence_logprobs,
                "generation_loss_mask": [1] * len(token_ids),
            }
        )
        stored_content = content
        if self.model.startswith("openai/gpt-oss-"):
            # Harmony responses contain channel/control tokens that the HF chat
            # template refuses when they are fed back as ordinary message text.
            # Preserve the raw response in the returned message/trace, but keep
            # only the semantic command in conversational state.
            harmony_command = _extract_harmony_command(content)
            stored_content = (
                f"I requested this shell command:\n```bash\n{harmony_command}\n```"
                if harmony_command
                else _strip_harmony_controls(content)
            )
        self._messages.append({"role": "assistant", "content": stored_content})
        return {"content": content}

    def _maybe_compact(
        self, *, on_delta: DeltaCallback | None, step: int
    ) -> None:
        """Bound long terminal histories while preserving the task and recent work."""

        threshold = self.compaction_threshold_tokens
        if not threshold or self._last_prompt_tokens < threshold:
            return
        keep = self.compaction_keep_messages
        if len(self._messages) <= keep + 2:
            return
        head = self._messages[:2]
        tail = self._messages[-keep:]
        removed = self._messages[2:-keep]
        commands = [
            str(row.get("command") or "")[:240]
            for row in self._transcript
            if row.get("command")
        ][-12:]
        summary = (
            "Context compacted by the mini-SWE harness. The task statement and "
            "recent turns are preserved. Earlier shell commands, oldest first:\n- "
            + "\n- ".join(commands)
            + "\n\nContinue by replying with exactly one fenced bash block containing "
            "one actual shell command. Never put prose, YAML, or raw data in the block."
        )
        self._messages = [*head, {"role": "system", "content": summary}, *tail]
        self._compactions += 1
        # Do not compact every subsequent step solely because the previous
        # request crossed the threshold; wait for fresh provider usage.
        self._last_prompt_tokens = 0
        if on_delta is not None:
            on_delta(
                {
                    "channel": "mini_swe",
                    "step": step,
                    "event": "context_compacted",
                    "removed_messages": len(removed),
                    "retained_messages": len(self._messages),
                }
            )

    def _complete_codex_cli(self) -> dict[str, Any]:
        """Use Codex authentication only as the Luna model transport.

        The mini-SWE loop remains authoritative: Codex receives the accumulated
        chat transcript and may return text only; mini-SWE parses and executes
        exactly one bash command afterward.
        """
        binary = os.environ.get("CODEX_BIN") or "codex"
        if not shutil.which(binary):
            raise RuntimeError("mini_swe codex_cli transport requires codex")
        transcript = "\n\n".join(
            f"{row['role'].upper()}: {_message_text(row.get('content'))}"
            for row in self._messages
        )
        prompt = (
            "Act only as the language model inside a mini-SWE harness. Do not run "
            "tools or inspect the current directory. Continue this transcript by "
            "returning exactly one fenced bash block as the assistant:\n\n" + transcript
        )
        with tempfile.TemporaryDirectory(prefix="mini-swe-luna-") as directory:
            output = Path(directory) / "last.txt"
            completed = subprocess.run(
                [binary, "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--model", self.model, "-c", 'model_reasoning_effort="medium"', "--cd", directory, "--output-last-message", str(output), prompt],
                capture_output=True, text=True, timeout=self.request_timeout, stdin=subprocess.DEVNULL,
            )
            if completed.returncode != 0 or not output.is_file():
                raise RuntimeError(f"mini_swe codex_cli transport failed: exit={completed.returncode}")
            content = output.read_text(encoding="utf-8")
        self.calls += 1
        message = {"content": content}
        self._messages.append({"role": "assistant", "content": content})
        return message

    def _run(self, command: str, workspace: Path) -> dict[str, Any]:
        import time

        for alias in self.workspace_aliases:
            command = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?=/|\b)", str(workspace), command)
        command = _detach_trailing_background_command(command)
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
    content = _message_text(message.get("content"))
    harmony = _extract_harmony_command(content)
    if harmony is not None:
        return harmony
    return _extract_command(content)


def _extract_harmony_command(content: str) -> str | None:
    """Read the first GPT-OSS Harmony ``container.exec`` call."""

    match = _HARMONY_EXEC_CALL.search(content or "")
    if match is None:
        return None
    try:
        arguments = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    command = arguments.get("cmd") if isinstance(arguments, dict) else None
    if isinstance(command, str):
        return command.strip() or None
    if isinstance(command, list) and all(isinstance(item, str) for item in command):
        if len(command) >= 3 and command[0] in {"bash", "/bin/bash", "sh", "/bin/sh"} and command[1] in {"-c", "-lc"}:
            return command[2].strip() or None
        return " ".join(command).strip() or None
    return None


def _strip_harmony_controls(content: str) -> str:
    """Remove GPT-OSS wire control tokens before chat-template reuse."""

    return re.sub(r"<\|[^|]+\|>", "", content or "").strip()


def _extract_command(content: str) -> str | None:
    blocks = _COMMAND_BLOCK.findall(content or "")
    if not blocks:
        return None
    command = blocks[0].strip()
    return command or None


def _command_rejection(command: str | None) -> str | None:
    """Reject common model data/prose emissions before they reach bash."""

    if not command:
        return "no_command"
    stripped = command.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return "no_command"
    first = lines[0]
    if len(lines) >= 2 and all(
        re.fullmatch(r"[A-Za-z_][\w.-]*:\s*.*", line) for line in lines[: min(5, len(lines))]
    ):
        return "looks_like_yaml"
    if len(lines) == 1 and len(first) >= 24 and re.fullmatch(r"[A-Za-z]+", first):
        return "looks_like_raw_data"
    if first.lower().startswith(("the issue is ", "i need to ", "let me ", "we need to ")):
        return "looks_like_prose"
    return None


def _detach_trailing_background_command(command: str) -> str:
    """Prevent a background child from retaining the harness capture pipes."""

    stripped = command.rstrip()
    if not stripped.endswith("&") or stripped.endswith("&&"):
        return command
    foreground = stripped[:-1].rstrip()
    return f"({foreground}) </dev/null >/tmp/mini-swe-background.log 2>&1 &"


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
