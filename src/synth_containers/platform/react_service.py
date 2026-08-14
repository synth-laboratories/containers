"""Loopback HTTP host for the trusted conversational ReAct harness."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .policy_process import POLICY_CANDIDATE_CONTRACT
from .react import OpenRouterReAct

PLANNER: OpenRouterReAct
SERVER: ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: Any) -> None:
        return

    def _body(self) -> dict[str, Any]:
        size = int(self.headers.get("content-length", "0"))
        value = json.loads(self.rfile.read(size) or b"{}")
        if not isinstance(value, dict):
            raise TypeError("request_must_be_object")
        return value

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/health":
            self._send(
                200,
                {
                    "ok": True,
                    "contract": POLICY_CANDIDATE_CONTRACT,
                    "service": "conversational_react",
                },
            )
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            body = self._body()
            if self.path == "/conformance":
                self._send(
                    200,
                    {
                        "ok": True,
                        "contract": POLICY_CANDIDATE_CONTRACT,
                        "capabilities": ["act", "checkpoint", "restore"],
                    },
                )
                return
            if self.path == "/act":
                if body.get("contract") != POLICY_CANDIDATE_CONTRACT:
                    raise ValueError("unsupported_policy_contract")
                observation = body.get("readout")
                if not isinstance(observation, dict):
                    raise TypeError("readout_must_be_object")
                observation = dict(observation)
                observation.setdefault("observation_text", body.get("observation_text"))
                observation.setdefault("valid_actions", body.get("valid_actions"))
                deltas: list[dict[str, Any]] = []
                actions = PLANNER.plan(observation, on_delta=deltas.append)
                self._send(
                    200,
                    {
                        "ok": True,
                        "contract": POLICY_CANDIDATE_CONTRACT,
                        "decision": {"actions": actions},
                        "deltas": deltas,
                        "trace": PLANNER.trace_data(),
                        "usage": PLANNER.usage(),
                        "metadata": PLANNER.metadata(),
                    },
                )
                return
            if self.path == "/checkpoint":
                self._send(200, {"ok": True, "state": PLANNER.checkpoint_state()})
                return
            if self.path == "/restore":
                state = body.get("state")
                if not isinstance(state, dict):
                    raise TypeError("state_must_be_object")
                PLANNER.restore_checkpoint_state(state)
                self._send(200, {"ok": True})
                return
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                threading.Thread(target=SERVER.shutdown, daemon=True).start()
                return
            self._send(404, {"ok": False, "error": "not_found"})
        except Exception as exc:
            self._send(
                422,
                {
                    "ok": False,
                    "error": "policy_service_error",
                    "detail": str(exc),
                    "contract": POLICY_CANDIDATE_CONTRACT,
                },
            )


def main() -> int:
    global PLANNER, SERVER
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    PLANNER = OpenRouterReAct(
        config_id=str(config.get("config_id") or "react"),
        config=dict(config.get("config") or {}),
    )
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(
        json.dumps(
            {
                "op": "ready",
                "ok": True,
                "port": SERVER.server_address[1],
                "contract": POLICY_CANDIDATE_CONTRACT,
            }
        ),
        flush=True,
    )
    SERVER.serve_forever()
    SERVER.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
