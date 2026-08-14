from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from synth_containers.platform.react_process import ReactPolicyServiceProcess


def test_conversational_react_runs_out_of_process_over_http(monkeypatch) -> None:
    requests: list[dict] = []

    class ProviderHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            size = int(self.headers.get("content-length", "0"))
            requests.append(json.loads(self.rfile.read(size)))
            payload = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "choose_actions",
                                            "arguments": '{"actions":["noop"]}',
                                        }
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 21,
                        "completion_tokens": 3,
                        "total_tokens": 24,
                    },
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    provider = HTTPServer(("127.0.0.1", 0), ProviderHandler)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("LANE3_TEST_API_KEY", "test-only")
    service = ReactPolicyServiceProcess(
        config_id="service-test",
        config={
            "base_url": f"http://127.0.0.1:{provider.server_address[1]}",
            "api_key_env": "LANE3_TEST_API_KEY",
            "model": "fixture-model",
            "environment_name": "Rogue",
        },
    )
    try:
        assert service.isolation_receipt["transport"] == "http"
        assert service.isolation_receipt["service"] == "conversational_react"
        actions = service.plan(
            {
                "observation_text": "|@..%|",
                "ascii": "|@..%|",
                "valid_actions": ["noop", "h", "j", "k", "l", ">"],
                "env_steps": 0,
            }
        )
        assert actions == ["noop"]
        assert service.usage()["calls"] == 1
        assert service.metadata()["kind"] == "openrouter_react"
        state = service.checkpoint_state()
        assert state["calls"] == 1
        service.restore_checkpoint_state(state)
        assert requests
        assert requests[0]["messages"][0]["content"].startswith(
            "You are a careful Rogue ReAct policy"
        )
    finally:
        service.close()
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=2)
