"""Out-of-process code-policy player. Observation/action JSONL, never in-proc import."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

_SERVE = r'''
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path

def _load(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "choose_actions", None)
    act = getattr(module, "act", None)
    if callable(fn):
        return fn
    if callable(act):
        def _wrap(*, observation_text, session, valid_actions, engine, seed, ply, readout):
            raw = act(readout)
            name = str(raw)
            if name not in valid_actions:
                name = "noop"
            return {"actions": [name], "policy_reason": "act()"}
        return _wrap
    raise SystemExit("missing choose_actions")

def main() -> int:
    fn = _load(Path(sys.argv[1]))
    sys.stdout.write(json.dumps({"op": "ready", "ok": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("op") == "close":
            sys.stdout.write(json.dumps({"id": req.get("id"), "ok": True}) + "\n")
            sys.stdout.flush()
            return 0
        decision = fn(
            observation_text=str(req.get("observation_text") or ""),
            session=dict(req.get("session") or {}),
            valid_actions=list(req.get("valid_actions") or []),
            engine=None,
            seed=None,
            ply=int(req.get("ply") or 0),
            readout=dict(req.get("readout") or {}),
        )
        sys.stdout.write(json.dumps({"id": req.get("id"), "ok": True, "decision": decision}) + "\n")
        sys.stdout.flush()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''

DEFAULT_HEURISTIC = '''\
"""Hunt the tree (`T`) then `do`. No model."""

def choose_actions(*, observation_text, session, valid_actions, engine, seed, ply, readout):
    grid = str(readout.get("ascii") or observation_text or "")
    rows = [row for row in grid.splitlines() if row]
    px = py = tx = ty = None
    for y, row in enumerate(rows):
        if "P" in row:
            px, py = row.index("P"), y
        if "T" in row:
            tx, ty = row.index("T"), y
    if px is None:
        return {"actions": ["noop"], "policy_reason": "no_player"}
    if tx is None:
        chosen = "do" if "do" in valid_actions else "noop"
        return {"actions": [chosen], "policy_reason": "collect"}
    if px < tx and "east" in valid_actions:
        return {"actions": ["east"], "policy_reason": "hunt"}
    if px > tx and "west" in valid_actions:
        return {"actions": ["west"], "policy_reason": "hunt"}
    if py < ty and "south" in valid_actions:
        return {"actions": ["south"], "policy_reason": "hunt"}
    if py > ty and "north" in valid_actions:
        return {"actions": ["north"], "policy_reason": "hunt"}
    return {"actions": ["do"], "policy_reason": "collect"}
'''

NOOP_HEURISTIC = '''\
def choose_actions(*, observation_text, session, valid_actions, engine, seed, ply, readout):
    return {"actions": ["noop"], "policy_reason": "always_noop"}
'''


class IsolatedPolicyProcess:
    """Spawn a child Python over JSONL. Isolation receipt is process-level."""

    def __init__(self, code: bytes) -> None:
        self._sandbox = tempfile.TemporaryDirectory(prefix="synth-policy-")
        root = Path(self._sandbox.name)
        policy = root / "policy.py"
        server = root / "serve.py"
        policy.write_bytes(code)
        server.write_text(_SERVE, encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, str(server), str(policy)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(root),
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            close_fds=True,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("isolated_policy_pipes")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._lock = threading.Lock()
        self._request_id = 0
        ready = self._stdout.readline()
        if not ready:
            raise RuntimeError(f"isolated_policy_startup_failed:rc={self._proc.poll()}")
        payload = json.loads(ready)
        if payload.get("op") != "ready" or payload.get("ok") is not True:
            raise RuntimeError("isolated_policy_not_ready")
        self.isolation_receipt = {
            "contract": "process_observation_action.v1",
            "platform": sys.platform,
            "sandbox": "process",
            "network": "shared_with_evaluator",
            "filesystem": "process_cwd_policy_only",
            "suite_visible": False,
            "pid": self._proc.pid,
        }

    def choose(self, observation: dict[str, Any], *, ply: int) -> list[str]:
        request = {
            "op": "choose_actions",
            "observation_text": str(observation.get("ascii") or ""),
            "session": {"ply": ply},
            "valid_actions": list(observation.get("valid_actions") or []),
            "ply": ply,
            "readout": {
                "ascii": observation.get("ascii"),
                "valid_actions": observation.get("valid_actions"),
            },
        }
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("isolated_policy_dead")
            self._request_id += 1
            request["id"] = self._request_id
            self._stdin.write(json.dumps(request) + "\n")
            self._stdin.flush()
            line = self._stdout.readline()
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError("isolated_policy_choose_failed")
        actions = list((response.get("decision") or {}).get("actions") or [])
        return [str(item) for item in actions] or ["noop"]

    def close(self) -> None:
        with self._lock:
            if self._proc.poll() is None:
                try:
                    self._request_id += 1
                    self._stdin.write(json.dumps({"id": self._request_id, "op": "close"}) + "\n")
                    self._stdin.flush()
                    self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
            self._sandbox.cleanup()
