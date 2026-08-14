"""Client/lifecycle wrapper for the conversational ReAct HTTP policy service."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

from .policy_process import POLICY_CANDIDATE_CONTRACT


class ReactPolicyServiceProcess:
    """Run one checkpointable conversation in a separately restartable process."""

    def __init__(self, *, config_id: str, config: dict[str, Any]) -> None:
        self._sandbox = tempfile.TemporaryDirectory(prefix="synth-react-policy-")
        config_path = Path(self._sandbox.name) / "config.json"
        config_path.write_text(
            json.dumps({"config_id": config_id, "config": config}), encoding="utf-8"
        )
        package_root = str(Path(__file__).resolve().parents[2])
        inherited_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath = os.pathsep.join(
            item for item in (package_root, inherited_pythonpath) if item
        )
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "synth_containers.platform.react_service", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=self._sandbox.name,
            env={
                **os.environ,
                "PYTHONPATH": pythonpath,
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            close_fds=True,
        )
        if self._proc.stdout is None:
            raise RuntimeError("react_policy_service_stdout")
        ready = self._proc.stdout.readline()
        if not ready:
            detail = self._proc.stderr.read() if self._proc.stderr is not None else ""
            raise RuntimeError(f"react_policy_service_startup_failed:{detail[-1000:]}")
        payload = json.loads(ready)
        if payload.get("ok") is not True:
            raise RuntimeError("react_policy_service_not_ready")
        self._base_url = f"http://127.0.0.1:{int(payload['port'])}"
        self._lock = threading.Lock()
        self._last_trace: dict[str, Any] = {}
        self._usage: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        conformance = self._request("/conformance", {})
        if conformance.get("ok") is not True:
            self.close()
            raise RuntimeError("react_policy_service_nonconforming")
        self.isolation_receipt = {
            "contract": POLICY_CANDIDATE_CONTRACT,
            "transport": "http",
            "sandbox": "process",
            "pid": self._proc.pid,
            "conformance": "accepted",
            "service": "conversational_react",
        }

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=660) as response:
                value = json.loads(response.read())
        except Exception as exc:
            detail = getattr(exc, "read", lambda: b"")()
            if detail:
                try:
                    message = json.loads(detail).get("detail")
                    raise RuntimeError(f"react_policy_service:{message}") from exc
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(f"react_policy_service_request_failed:{path}:{exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError("react_policy_service_returned_non_object")
        return value

    def plan(self, observation: dict[str, Any], on_delta: Any = None) -> list[str]:
        request = {
            "contract": POLICY_CANDIDATE_CONTRACT,
            "observation_text": str(
                observation.get("observation_text") or observation.get("ascii") or ""
            ),
            "session": {},
            "valid_actions": list(observation.get("valid_actions") or []),
            "ply": int(observation.get("env_steps") or 0),
            "readout": observation,
        }
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("react_policy_service_dead")
            response = self._request("/act", request)
        for delta in response.get("deltas") or []:
            if on_delta is not None and isinstance(delta, dict):
                on_delta(delta)
        self._last_trace = dict(response.get("trace") or {})
        self._usage = dict(response.get("usage") or {})
        self._metadata = dict(response.get("metadata") or {})
        return [str(item) for item in (response.get("decision") or {}).get("actions") or []]

    def usage(self) -> dict[str, Any]:
        return dict(self._usage)

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata) or {
            "harness": "react",
            "kind": "http_policy_service",
        }

    def trace_data(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def checkpoint_state(self) -> dict[str, Any]:
        return dict(self._request("/checkpoint", {}).get("state") or {})

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        self._request("/restore", {"state": state})

    def close(self) -> None:
        with self._lock:
            if self._proc.poll() is None:
                try:
                    self._request("/shutdown", {})
                    self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
            self._sandbox.cleanup()
