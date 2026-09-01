"""Out-of-process host for protocol code. JSONL over stdin/stdout, never in-proc import.

The child gets an empty ``PYTHONPATH`` and a scrubbed environment: a protocol
module is stdlib-only and self-contained, exactly like an isolated code
policy. That is what makes ``update_annotation_protocol=native`` an honest
claim -- a new revision is a new process, and the old one cannot linger as a
cached module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from .contract import PROTOCOL_SCHEMA

_SERVE = r'''
from __future__ import annotations
import importlib.util, json, sys, traceback
from pathlib import Path

EXPECTED = %(schema)r


def _load(path: Path, config: dict):
    spec = importlib.util.spec_from_file_location("live_annotation_protocol", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marker = getattr(module, "PROTOCOL", None)
    if marker != EXPECTED:
        raise SystemExit("protocol_marker_mismatch:%%r" %% (marker,))
    cls = getattr(module, "Protocol", None)
    if cls is None or not callable(cls):
        raise SystemExit("missing_protocol_class")
    protocol_id = str(getattr(module, "PROTOCOL_ID", "") or "")
    return cls(config), protocol_id


def _emissions(value):
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return list(value)


def main() -> int:
    boot = json.loads(sys.stdin.readline())
    if boot.get("op") != "boot":
        raise SystemExit("expected_boot")
    try:
        protocol, protocol_id = _load(Path(sys.argv[1]), dict(boot.get("config") or {}))
    except SystemExit:
        raise
    except Exception:
        sys.stdout.write(json.dumps({"op": "ready", "ok": False, "error": traceback.format_exc()[-2000:]}) + "\n")
        sys.stdout.flush()
        return 1
    sys.stdout.write(json.dumps({"op": "ready", "ok": True, "protocol_id": protocol_id}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        req = json.loads(line)
        op = req.get("op")
        rid = req.get("id")
        try:
            if op == "close":
                hook = getattr(protocol, "on_close", None)
                out = _emissions(hook()) if callable(hook) else []
                sys.stdout.write(json.dumps({"id": rid, "ok": True, "emissions": out}) + "\n")
                sys.stdout.flush()
                return 0
            if op == "event":
                out = _emissions(protocol.on_event(dict(req.get("event") or {})))
            elif op == "model_result":
                hook = getattr(protocol, "on_model_result", None)
                out = (
                    _emissions(hook(str(req.get("request_id")), req.get("result"), req.get("error")))
                    if callable(hook)
                    else []
                )
            else:
                raise ValueError("unknown_op:%%r" %% (op,))
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "emissions": out}, default=str) + "\n")
        except Exception:
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": traceback.format_exc()[-2000:]}) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''' % {"schema": PROTOCOL_SCHEMA}


class ProtocolProcessError(RuntimeError):
    """The child died or refused a request. The runner records and stops feeding."""


class IsolatedProtocolProcess:
    """Spawn a child Python over JSONL. One instance per rollout; never shared."""

    def __init__(self, code: bytes, *, config: dict[str, Any] | None = None) -> None:
        self._sandbox = tempfile.TemporaryDirectory(prefix="synth-annotation-protocol-")
        root = Path(self._sandbox.name)
        protocol = root / "protocol.py"
        server = root / "serve.py"
        protocol.write_bytes(code)
        server.write_text(_SERVE, encoding="utf-8")
        self._proc = subprocess.Popen(
            # -I: isolated (no PYTHONPATH, no user site, no cwd on sys.path).
            # -S: no site-packages at all, so the venv's installed packages --
            #     this one included -- are unreachable. Stdlib only, provably.
            [sys.executable, "-I", "-S", str(server), str(protocol)],
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
            raise ProtocolProcessError("protocol_process_pipes")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self._stdin.write(json.dumps({"op": "boot", "config": dict(config or {})}) + "\n")
        self._stdin.flush()
        ready_line = self._stdout.readline()
        if not ready_line:
            stderr = self._read_stderr()
            self._cleanup()
            raise ProtocolProcessError(
                f"protocol_startup_failed:rc={self._proc.poll()}:{stderr[-500:]}"
            )
        ready = json.loads(ready_line)
        if ready.get("op") != "ready" or ready.get("ok") is not True:
            error = str(ready.get("error") or "not_ready")
            self._cleanup()
            raise ProtocolProcessError(f"protocol_not_ready:{error[-500:]}")
        self.protocol_id = str(ready.get("protocol_id") or "")
        self.isolation_receipt = {
            "contract": "process_event_emission.v1",
            "platform": sys.platform,
            "sandbox": "process",
            "network": "none_expected",
            "filesystem": "process_cwd_protocol_only",
            "policy_visible": False,
            "pid": self._proc.pid,
        }

    @property
    def alive(self) -> bool:
        return not self._closed and self._proc.poll() is None

    def _read_stderr(self) -> str:
        try:
            if self._proc.stderr is not None:
                return self._proc.stderr.read() or ""
        except Exception:  # noqa: BLE001 - diagnostics only
            pass
        return ""

    def _call(self, request: dict[str, Any]) -> list[Any]:
        with self._lock:
            if self._closed or self._proc.poll() is not None:
                raise ProtocolProcessError("protocol_process_dead")
            self._request_id += 1
            request["id"] = self._request_id
            try:
                self._stdin.write(json.dumps(request, default=str) + "\n")
                self._stdin.flush()
                line = self._stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                raise ProtocolProcessError(f"protocol_process_pipe:{exc}") from exc
        if not line:
            raise ProtocolProcessError(f"protocol_process_exited:rc={self._proc.poll()}")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise ProtocolProcessError(str(response.get("error") or "protocol_request_failed"))
        emissions = response.get("emissions")
        return list(emissions) if isinstance(emissions, list) else []

    def on_event(self, event: dict[str, Any]) -> list[Any]:
        return self._call({"op": "event", "event": event})

    def on_model_result(
        self, request_id: str, result: dict[str, Any] | None, error: str | None
    ) -> list[Any]:
        return self._call(
            {"op": "model_result", "request_id": request_id, "result": result, "error": error}
        )

    def close(self) -> list[Any]:
        """Send close, collect the final flush, reap the child. Idempotent."""

        emissions: list[Any] = []
        with self._lock:
            if self._closed:
                return []
            self._closed = True
            if self._proc.poll() is None:
                try:
                    self._request_id += 1
                    self._stdin.write(json.dumps({"id": self._request_id, "op": "close"}) + "\n")
                    self._stdin.flush()
                    line = self._stdout.readline()
                    if line:
                        response = json.loads(line)
                        if response.get("ok") is True and isinstance(response.get("emissions"), list):
                            emissions = list(response["emissions"])
                    self._proc.wait(timeout=2)
                except Exception:  # noqa: BLE001 - closing must not raise
                    self._proc.kill()
        self._cleanup()
        return emissions

    def kill(self) -> None:
        with self._lock:
            self._closed = True
            if self._proc.poll() is None:
                self._proc.kill()
        self._cleanup()

    def _cleanup(self) -> None:
        try:
            self._sandbox.cleanup()
        except Exception:  # noqa: BLE001
            pass
