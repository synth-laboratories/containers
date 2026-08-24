"""Codex CLI agentic policy — the top rung of the policy ladder.

Unlike the chat/responses harnesses, this policy is an *agent*: each turn it gets
a workspace directory containing the observation, may run tools inside it, and
must finish with a JSON object naming the actions. Continuity across turns is
the Codex session (``codex exec resume``), not a replayed message list.

Contract with the CLI, all first-class flags rather than stdout scraping:

* ``--output-schema``       pins the final message to ``{"actions":[...]}``
* ``--output-last-message`` writes that object to a file we read
* ``--json``                streams ``codex.*`` events, forwarded as policy
                            deltas so they land in the v5 trace

Credentials come from ``CODEX_HOME`` / the forwarded provider key. Nothing here
reads or logs a key.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

DeltaCallback = Callable[[dict[str, Any]], None]

# Strict structured output: every key in `properties` must also appear in
# `required`, so an optional field is spelled as a nullable required one. A
# schema that omits `reason` from `required` is rejected with a 400 before the
# turn runs.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["actions", "reason"],
    "additionalProperties": False,
}


def _failure_detail(completed: "subprocess.CompletedProcess[str]", events: list[dict[str, Any]]) -> str:
    """Why the run failed, in the order the answer is actually carried.

    The CLI reports a refused request as a JSON `error` event on stdout while
    stderr holds only progress chatter, so reading stderr's last line names the
    chatter and hides the cause.
    """

    for event in reversed(events):
        if str(event.get("type")) in {"error", "turn.failed"}:
            message = event.get("message") or (event.get("error") or {}).get("message")
            if message:
                return str(message).replace("\n", " ")[:500]
    lines = (completed.stderr or completed.stdout or "").strip().splitlines()
    return " | ".join(line.strip() for line in lines[-5:] if line.strip())[:500]


class CodexAgenticPolicy:
    """One ``codex exec`` turn per environment observation, session-resumed."""

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self.config_id = config_id
        self.env_name = str(config.get("env_name") or "environment")
        self.objective = str(
            config.get("objective")
            or "Make measurable progress on the environment objective while staying alive."
        )
        self.model = str(config.get("model") or "gpt-5.6-luna")
        self.binary = str(config.get("binary") or os.environ.get("CODEX_BIN") or "codex")
        # Point Codex at an arbitrary Responses endpoint — a training proxy, for
        # instance. Without this the CLI only talks to whatever `CODEX_HOME`
        # was baked with, which a per-rollout session origin cannot be.
        self.base_url = str(
            config.get("base_url") or os.environ.get("OPENAI_BASE_URL") or ""
        ).rstrip("/")
        self.api_key_env = str(config.get("api_key_env") or "OPENAI_API_KEY")
        self.provider_id = str(config.get("provider_id") or "tito")
        self.wire_api = str(config.get("wire_api") or "responses")
        self.sandbox = str(config.get("sandbox") or "workspace-write")
        if self.sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise RuntimeError(f"codex_sandbox_invalid:{self.sandbox}")
        self.timeout_seconds = min(max(float(config.get("timeout_seconds") or 300.0), 10.0), 3600.0)
        self.plan_max = min(max(int(config.get("plan_max") or 5), 1), 64)
        self.resume = bool(config.get("resume", True))
        self.workspace_root = config.get("workspace_root")
        self.calls = 0
        self._session_id: str | None = None
        self._workspace: Path | None = None
        self._owns_workspace = False
        self._usage: dict[str, Any] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self._last_trace: dict[str, Any] = {}

    # ------------------------------------------------------------------ facets

    def metadata(self) -> dict[str, Any]:
        return {
            "harness": "codex_agentic",
            "kind": "codex_cli_agent",
            "wire_api": self.wire_api,
            "provider_id": self.provider_id if self.base_url else None,
            "config": self.config_id,
            "model": self.model,
            "sandbox": self.sandbox,
            "runtime_family": "codex",
            "session_continuity": "codex_session" if self.resume else "none",
            "plan_max": self.plan_max,
            "token_trace": "native",
            "graded": True,
        }

    def usage(self) -> dict[str, Any]:
        return {**self._usage, "calls": self.calls}

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": "synth.containers.codex-agentic-checkpoint.v1",
            "session_id": self._session_id,
            "calls": self.calls,
            "usage": dict(self._usage),
            # The workspace is the other half of a codex checkpoint; the caller
            # snapshots it. Naming it here keeps the pair honest.
            "workspace": str(self._workspace) if self._workspace else None,
            "checkpoint_semantics": "codex_session_workspace_snapshot",
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "synth.containers.codex-agentic-checkpoint.v1":
            raise RuntimeError("unsupported policy checkpoint schema")
        self._session_id = state.get("session_id") or None
        self.calls = int(state.get("calls") or 0)
        usage = state.get("usage")
        if not isinstance(usage, dict):
            raise RuntimeError("policy checkpoint omitted usage")
        self._usage = dict(usage)

    def close(self) -> None:
        if self._owns_workspace and self._workspace is not None:
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    # -------------------------------------------------------------------- plan

    def plan(
        self, observation: dict[str, Any], on_delta: DeltaCallback | None = None
    ) -> list[str]:
        valid = [str(action) for action in observation.get("valid_actions") or ()]
        if not valid:
            raise RuntimeError("observation omitted valid_actions")
        if shutil.which(self.binary) is None:
            raise RuntimeError(f"codex_cli_missing:{self.binary}")

        workspace = self._ensure_workspace()
        (workspace / "observation.json").write_text(
            json.dumps(observation, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        (workspace / "valid_actions.json").write_text(
            json.dumps(valid, indent=2), encoding="utf-8"
        )
        observation_text = str(
            observation.get("observation_text") or observation.get("ascii") or ""
        )
        (workspace / "observation.txt").write_text(observation_text, encoding="utf-8")

        schema_path = workspace / "_actions.schema.json"
        schema_path.write_text(json.dumps(_OUTPUT_SCHEMA), encoding="utf-8")
        last_message = workspace / "_last_message.json"
        last_message.unlink(missing_ok=True)

        prompt = (
            f"You are the acting policy for a {self.env_name} environment.\n"
            f"{self.objective}\n\n"
            "The current observation is in observation.txt / observation.json and the "
            "exact legal actions are in valid_actions.json. Inspect them, think, and "
            f"reply with at most {self.plan_max} actions drawn ONLY from that legal list.\n"
            "You may write scratch notes into this workspace; they persist across turns."
        )

        argv = [self.binary, "exec"]
        if self.resume and self._session_id:
            argv += ["resume", self._session_id]
        argv += self._provider_overrides()
        argv += [
            "--json",
            "--model",
            self.model,
            "--sandbox",
            self.sandbox,
            "--cd",
            str(workspace),
            "--skip-git-repo-check",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(last_message),
            prompt,
        ]

        completed = subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_seconds,
            cwd=str(workspace),
            # The prompt is a positional argument. Without this the CLI waits on
            # an inherited stdin that a container's PID 1 never closes, and the
            # call hangs until the timeout with nothing to show for it.
            stdin=subprocess.DEVNULL,
        )
        self.calls += 1
        events = _parse_events(completed.stdout)
        self._absorb(events, on_delta)
        if completed.returncode != 0 and not last_message.is_file():
            # The last line is often progress chatter, not the cause. Carry the
            # exit code and the tail so a failure names itself.
            raise RuntimeError(
                f"codex_exec_failed:exit={completed.returncode}:{_failure_detail(completed, events)}"
            )

        raw = last_message.read_text(encoding="utf-8") if last_message.is_file() else ""
        actions = _filter(_load_actions(raw), valid, self.plan_max)
        self._last_trace = {
            "session_id": self._session_id,
            "events": len(events),
            "raw": raw[:2000],
            "exit_code": completed.returncode,
        }
        if not actions:
            raise RuntimeError("policy returned no valid actions")
        return actions

    # ---------------------------------------------------------------- internal

    def _ensure_workspace(self) -> Path:
        if self._workspace is not None:
            return self._workspace
        if self.workspace_root:
            workspace = Path(str(self.workspace_root)).expanduser()
            workspace.mkdir(parents=True, exist_ok=True)
            self._owns_workspace = False
        else:
            workspace = Path(tempfile.mkdtemp(prefix="codex-policy-"))
            self._owns_workspace = True
        self._workspace = workspace
        return workspace

    def _provider_overrides(self) -> list[str]:
        """`-c` overrides that bind this turn to a specific Responses endpoint.

        Passed on the command line rather than written into `CODEX_HOME` so two
        rollouts sharing a container cannot read each other's origin.
        """

        if not self.base_url:
            return []
        provider = f"model_providers.{self.provider_id}"
        return [
            "-c", f"model_provider={self.provider_id}",
            "-c", f'{provider}.name="{self.provider_id}"',
            "-c", f'{provider}.base_url="{self.base_url}"',
            "-c", f'{provider}.env_key="{self.api_key_env}"',
            "-c", f'{provider}.wire_api="{self.wire_api}"',
        ]

    def _absorb(self, events: list[dict[str, Any]], on_delta: DeltaCallback | None) -> None:
        for event in events:
            session = _dig(event, "session_id") or _dig(event, "conversation_id")
            if isinstance(session, str) and session:
                self._session_id = session
            usage = _dig(event, "token_usage") or _dig(event, "usage")
            if isinstance(usage, dict):
                for key, source in (
                    ("prompt_tokens", "input_tokens"),
                    ("completion_tokens", "output_tokens"),
                    ("total_tokens", "total_tokens"),
                ):
                    value = usage.get(source, usage.get(key))
                    if isinstance(value, (int, float)):
                        self._usage[key] = float(value)
            if on_delta is not None:
                on_delta({"kind": "codex.event", "event": event})


def _parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _dig(payload: Any, key: str) -> Any:
    """Find ``key`` anywhere in a shallow nest — the CLI moves it between versions."""

    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _dig(value, key)
            if found is not None:
                return found
    return None


def _load_actions(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict) and isinstance(payload.get("actions"), list):
        return [str(item) for item in payload["actions"]]
    return []


def _filter(requested: list[str], valid: list[str], limit: int) -> list[str]:
    lowered = {action.lower(): action for action in valid}
    out: list[str] = []
    for item in requested:
        match = lowered.get(str(item).strip().lower())
        if match is not None:
            out.append(match)
    return out[:limit]
