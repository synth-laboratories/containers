"""Out-of-process HTTP code-policy service; candidate code is never imported in-proc."""

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

POLICY_CANDIDATE_CONTRACT = "policy_candidate.v1"

_SERVE = r'''
from __future__ import annotations
import importlib.util, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    raise RuntimeError("missing choose_actions_or_act")

FN = None
SERVER = None

def _decision(req):
        decision = FN(
            observation_text=str(req.get("observation_text") or ""),
            session=dict(req.get("session") or {}),
            valid_actions=list(req.get("valid_actions") or []),
            engine=None,
            seed=None,
            ply=int(req.get("ply") or 0),
            readout=dict(req.get("readout") or {}),
        )
        if not isinstance(decision, dict):
            raise TypeError("decision_must_be_object")
        actions = decision.get("actions")
        if not isinstance(actions, list) or not actions or not all(isinstance(x, str) for x in actions):
            raise TypeError("decision.actions_must_be_nonempty_string_array")
        return decision

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return
    def _send(self, status, payload):
        data = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "contract": "policy_candidate.v1"})
        return self._send(404, {"ok": False, "error": "not_found"})
    def do_POST(self):
        size = int(self.headers.get("content-length", "0"))
        try:
            body = json.loads(self.rfile.read(size) or b"{}")
            if self.path == "/conformance":
                result = _decision({"observation_text":"", "valid_actions":["noop"], "readout":{}, "ply":0})
                return self._send(200, {"ok": True, "contract": "policy_candidate.v1", "probe": result})
            if self.path == "/act":
                return self._send(200, {"ok": True, "contract": "policy_candidate.v1", "decision": _decision(body)})
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                import threading
                threading.Thread(target=SERVER.shutdown, daemon=True).start()
                return
            self._send(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._send(422, {"ok": False, "error": "candidate_contract_violation", "detail": str(exc), "contract": "policy_candidate.v1"})

def main() -> int:
    global FN, SERVER
    FN = _load(Path(sys.argv[1]))
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = SERVER.server_address[1]
    sys.stdout.write(json.dumps({"op": "ready", "ok": True, "port": port, "contract": "policy_candidate.v1"}) + "\n")
    sys.stdout.flush()
    SERVER.serve_forever()
    SERVER.server_close()
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
    """Spawn a candidate as a loopback HTTP service with process isolation."""

    def __init__(self, code: bytes) -> None:
        self._sandbox = tempfile.TemporaryDirectory(prefix="synth-policy-")
        root = Path(self._sandbox.name)
        policy = root / "policy.py"
        server = root / "serve.py"
        policy.write_bytes(code)
        server.write_text(_SERVE, encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, str(server), str(policy)],
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
        if self._proc.stdout is None:
            raise RuntimeError("isolated_policy_stdout")
        self._stdout = self._proc.stdout
        self._lock = threading.Lock()
        ready = self._stdout.readline()
        if not ready:
            raise RuntimeError(f"isolated_policy_startup_failed:rc={self._proc.poll()}")
        payload = json.loads(ready)
        if payload.get("op") != "ready" or payload.get("ok") is not True:
            raise RuntimeError("isolated_policy_not_ready")
        self._base_url = f"http://127.0.0.1:{int(payload['port'])}"
        try:
            conformance = self._request("/conformance", {})
            if conformance.get("ok") is not True:
                raise RuntimeError("candidate_contract_violation")
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=2)
            self._sandbox.cleanup()
            raise
        self.isolation_receipt = {
            "contract": POLICY_CANDIDATE_CONTRACT,
            "transport": "http",
            "platform": sys.platform,
            "sandbox": "process",
            "network": "shared_with_evaluator",
            "filesystem": "process_cwd_policy_only",
            "suite_visible": False,
            "pid": self._proc.pid,
            "conformance": "accepted",
        }

    def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base_url + path,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return dict(json.loads(response.read()))
        except Exception as exc:
            detail = getattr(exc, "read", lambda: b"")()
            if detail:
                try:
                    payload = json.loads(detail)
                    raise RuntimeError(str(payload.get("detail") or payload.get("error"))) from exc
                except json.JSONDecodeError:
                    pass
            raise RuntimeError(f"policy_service_request_failed:{path}:{exc}") from exc

    def choose(self, observation: dict[str, Any], *, ply: int) -> list[str]:
        request = {
            "contract": POLICY_CANDIDATE_CONTRACT,
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
            response = self._request("/act", request)
        if response.get("ok") is not True:
            raise RuntimeError("isolated_policy_choose_failed")
        actions = list((response.get("decision") or {}).get("actions") or [])
        return [str(item) for item in actions] or ["noop"]

    def close(self) -> None:
        with self._lock:
            if self._proc.poll() is None:
                try:
                    self._request("/shutdown", {})
                    self._proc.wait(timeout=2)
                except Exception:
                    self._proc.kill()
            self._sandbox.cleanup()
