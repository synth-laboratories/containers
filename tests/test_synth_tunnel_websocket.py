"""Relay-agent side of docs/RELAY_STREAMING_AND_WEBSOCKET.md (stage 2).

Every test drives ``SynthTunnelRelayAgent``'s own receive loop through a fake
relay socket, so the frame handling is exercised without a real relay. The
origin, by contrast, is real: a loopback WebSocket server for the accept path
and a loopback HTTP server for the refusal path.
"""

from __future__ import annotations

import base64
import http.server
import json
import queue
import socketserver
import threading
import time
from typing import Any

import pytest

from synth_containers.tunnels import relay as relay_module
from synth_containers.tunnels.relay import SynthTunnelRelayAgent

websockets_server = pytest.importorskip("websockets.sync.server")


class _FakeRelaySocket:
    """Stands in for the agent's socket to the hosted relay."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self._sent_lock = threading.Lock()
        self._inbound: queue.Queue[str | None] = queue.Queue()
        self._closed = threading.Event()

    # -- agent side ---------------------------------------------------
    def send(self, message: str) -> None:
        payload = json.loads(message)
        with self._sent_lock:
            self.sent.append(payload)
        if payload.get("type") == "ATTACH":
            self.push(
                {
                    "type": "ATTACH_ACK",
                    "accepted_leases": [
                        lease["lease_id"] for lease in payload.get("leases", [])
                    ],
                }
            )

    def recv(self) -> str:
        while True:
            if self._closed.is_set():
                raise OSError("fake relay socket closed")
            try:
                item = self._inbound.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                raise OSError("fake relay socket closed")
            return item

    def close(self) -> None:
        self._closed.set()
        self._inbound.put(None)

    # -- test side ----------------------------------------------------
    def push(self, frame: dict[str, Any]) -> None:
        self._inbound.put(json.dumps(frame))

    def frames(self, frame_type: str) -> list[dict[str, Any]]:
        with self._sent_lock:
            return [item for item in self.sent if item.get("type") == frame_type]

    def wait_for(
        self,
        frame_type: str,
        *,
        count: int = 1,
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            matches = self.frames(frame_type)
            if len(matches) >= count:
                return matches
            time.sleep(0.01)
        with self._sent_lock:
            seen = [item.get("type") for item in self.sent]
        raise AssertionError(f"never saw {count} {frame_type} frame(s); saw {seen}")


def _make_agent(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeRelaySocket,
    *,
    origin_port: int,
    max_request_body_bytes: int = 64 * 1024,
    max_websocket_streams: int = 8,
) -> SynthTunnelRelayAgent:
    monkeypatch.setattr(
        relay_module,
        "_connect_websocket",
        lambda *args, **kwargs: fake,
    )
    agent = SynthTunnelRelayAgent(
        lease_id="lease_test",
        local_target=relay_module._parse_local_target(f"http://127.0.0.1:{origin_port}"),
        agent_connect={
            "transport": "ws",
            "url": "ws://relay.invalid/agent",
            "agent_token": "agent-token",
        },
        max_in_flight_requests=4,
        max_request_body_bytes=max_request_body_bytes,
        max_websocket_streams=max_websocket_streams,
    )
    agent.start(timeout_seconds=5.0)
    return agent


# ---------------------------------------------------------------------------
# origins
# ---------------------------------------------------------------------------

_REFUSAL_BODY = b"the capture proxy is an HTTP/SSE observer"


class _RefusingHandler(http.server.BaseHTTPRequestHandler):
    """Mirrors the Trace V5 capture proxy: a deliberate 426."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(426)
        self.send_header("Connection", "close")
        self.send_header("Upgrade", "websocket")
        self.send_header("Sec-WebSocket-Version", "13")
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(_REFUSAL_BODY)))
        self.end_headers()
        self.wfile.write(_REFUSAL_BODY)

    def log_message(self, *args: Any) -> None:
        return


@pytest.fixture
def refusing_origin_port() -> Any:
    server = socketserver.TCPServer(("127.0.0.1", 0), _RefusingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


class _EchoOrigin:
    def __init__(self) -> None:
        self.request_headers: list[dict[str, str]] = []
        self.closed_with: list[tuple[int | None, str | None]] = []
        self._server: Any = None
        self._thread: threading.Thread | None = None

    def _handler(self, connection: Any) -> None:
        self.request_headers.append(
            {key.lower(): value for key, value in connection.request.headers.raw_items()}
        )
        try:
            for message in connection:
                if isinstance(message, str) and message.startswith("flood:"):
                    connection.send(b"q" * int(message.split(":", 1)[1]))
                    continue
                connection.send(message)
        except Exception:
            pass
        finally:
            self.closed_with.append((connection.close_code, connection.close_reason))

    def start(self, *, subprotocols: list[str] | None = None) -> int:
        self._server = websockets_server.serve(
            self._handler,
            "127.0.0.1",
            0,
            subprotocols=subprotocols,
            compression=None,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()
        return self._server.socket.getsockname()[1]

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()


@pytest.fixture
def echo_origin() -> Any:
    origin = _EchoOrigin()
    try:
        yield origin
    finally:
        origin.stop()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_attach_advertises_the_websocket_capability(
    monkeypatch: pytest.MonkeyPatch,
    refusing_origin_port: int,
) -> None:
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=refusing_origin_port)
    try:
        attach = fake.wait_for("ATTACH")[0]
        assert attach["capabilities"] == {"websocket": True}
        assert attach["leases"] == [{"lease_id": "lease_test"}]
    finally:
        agent.stop()


def test_ws_open_reject_carries_the_origin_refusal_intact(
    monkeypatch: pytest.MonkeyPatch,
    refusing_origin_port: int,
) -> None:
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=refusing_origin_port)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/v1/responses",
                "query": "",
                "headers": [["sec-websocket-version", "13"]],
                "deadline_ms": 10000,
            }
        )
        reject = fake.wait_for("WS_OPEN_REJECT")[0]
        assert reject["rid"] == "r1"
        assert reject["lease_id"] == "lease_test"
        assert reject["status"] == 426
        assert base64.b64decode(reject["body_b64"]) == _REFUSAL_BODY
        headers = {key.lower(): value for key, value in reject["headers"]}
        # The headers that make a client stop retrying the handshake survive.
        assert headers["upgrade"] == "websocket"
        assert headers["sec-websocket-version"] == "13"
        assert headers["content-type"] == "text/plain; charset=utf-8"
        # Framing headers belong to the relay's own response, not the origin's.
        assert "connection" not in headers
        assert "content-length" not in headers
        assert "transfer-encoding" not in headers
        assert fake.frames("WS_OPEN_ACK") == []
    finally:
        agent.stop()


def test_ws_open_ack_then_frames_pump_both_ways(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=port)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "a=1",
                "headers": [],
                "deadline_ms": 10000,
            }
        )
        ack = fake.wait_for("WS_OPEN_ACK")[0]
        assert ack["rid"] == "r1"
        assert ack["headers"] == []

        fake.push(
            {
                "type": "WS_FRAME",
                "lease_id": "lease_test",
                "rid": "r1",
                "opcode": "text",
                "data_b64": base64.b64encode(b"hello").decode("ascii"),
                "fin": True,
            }
        )
        echoed = fake.wait_for("WS_FRAME")[0]
        assert echoed["rid"] == "r1"
        assert echoed["opcode"] == "text"
        assert echoed["fin"] is True
        assert base64.b64decode(echoed["data_b64"]) == b"hello"

        fake.push(
            {
                "type": "WS_FRAME",
                "lease_id": "lease_test",
                "rid": "r1",
                "opcode": "binary",
                "data_b64": base64.b64encode(b"\x00\xff\x10").decode("ascii"),
                "fin": True,
            }
        )
        binary = fake.wait_for("WS_FRAME", count=2)[1]
        assert binary["opcode"] == "binary"
        assert base64.b64decode(binary["data_b64"]) == b"\x00\xff\x10"

        # Ordering for one rid is preserved.
        for index in range(5):
            fake.push(
                {
                    "type": "WS_FRAME",
                    "lease_id": "lease_test",
                    "rid": "r1",
                    "opcode": "text",
                    "data_b64": base64.b64encode(str(index).encode()).decode("ascii"),
                    "fin": True,
                }
            )
        frames = fake.wait_for("WS_FRAME", count=7)
        ordered = [base64.b64decode(item["data_b64"]) for item in frames[2:7]]
        assert ordered == [b"0", b"1", b"2", b"3", b"4"]
    finally:
        agent.stop()


def test_relay_fragments_are_reassembled_before_the_origin_send(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=port)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [],
                "deadline_ms": 10000,
            }
        )
        fake.wait_for("WS_OPEN_ACK")
        for chunk, final in ((b"frag-", False), (b"ment", True)):
            fake.push(
                {
                    "type": "WS_FRAME",
                    "lease_id": "lease_test",
                    "rid": "r1",
                    "opcode": "text",
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                    "fin": final,
                }
            )
        echoed = fake.wait_for("WS_FRAME")[0]
        assert base64.b64decode(echoed["data_b64"]) == b"frag-ment"
        assert echoed["fin"] is True
    finally:
        agent.stop()


def test_credentials_are_stripped_and_subprotocol_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start(subprotocols=["synth.v1"])
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=port)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [
                    ["authorization", "Bearer worker-jwt"],
                    ["x-api-key", "sk-secret"],
                    ["x-api-keys", "sk-secret-2"],
                    ["sec-websocket-key", "AAAAAAAAAAAAAAAAAAAAAA=="],
                    ["sec-websocket-version", "13"],
                    ["sec-websocket-accept", "not-forwarded"],
                    ["sec-websocket-extensions", "permessage-deflate"],
                    ["sec-websocket-protocol", "synth.v1, other.v2"],
                    ["x-passthrough", "kept"],
                ],
                "deadline_ms": 10000,
            }
        )
        ack = fake.wait_for("WS_OPEN_ACK")[0]
        assert ack["headers"] == [["sec-websocket-protocol", "synth.v1"]]

        seen = echo_origin.request_headers[0]
        assert "authorization" not in seen
        assert "x-api-key" not in seen
        assert "x-api-keys" not in seen
        assert "x-sec-websocket-accept" not in seen
        assert "sec-websocket-accept" not in seen
        # The agent handshakes with its own key, not the client's.
        assert seen["sec-websocket-key"] != "AAAAAAAAAAAAAAAAAAAAAA=="
        assert seen.get("sec-websocket-protocol") == "synth.v1, other.v2"
        assert seen["x-passthrough"] == "kept"
    finally:
        agent.stop()


def test_ws_close_from_the_relay_drops_the_stream_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=port)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [],
                "deadline_ms": 10000,
            }
        )
        fake.wait_for("WS_OPEN_ACK")
        fake.push(
            {
                "type": "WS_CLOSE",
                "lease_id": "lease_test",
                "rid": "r1",
                "code": 1000,
                "reason": "done",
            }
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and echo_origin.closed_with == []:
            time.sleep(0.01)
        assert echo_origin.closed_with[0][0] == 1000

        # A second WS_CLOSE, and any later WS_FRAME, are no-ops.
        fake.push(
            {"type": "WS_CLOSE", "lease_id": "lease_test", "rid": "r1", "code": 1000}
        )
        fake.push(
            {
                "type": "WS_FRAME",
                "lease_id": "lease_test",
                "rid": "r1",
                "opcode": "text",
                "data_b64": base64.b64encode(b"late").decode("ascii"),
                "fin": True,
            }
        )
        time.sleep(0.3)
        assert fake.frames("WS_FRAME") == []
        # The relay closed it, so the agent does not echo a WS_CLOSE back.
        assert fake.frames("WS_CLOSE") == []
        assert agent._streams == {}
    finally:
        agent.stop()


def test_oversize_relay_frame_closes_the_stream_with_1009(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(
        monkeypatch,
        fake,
        origin_port=port,
        max_request_body_bytes=1024,
    )
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [],
                "deadline_ms": 10000,
            }
        )
        fake.wait_for("WS_OPEN_ACK")
        fake.push(
            {
                "type": "WS_FRAME",
                "lease_id": "lease_test",
                "rid": "r1",
                "opcode": "binary",
                "data_b64": base64.b64encode(b"x" * 2048).decode("ascii"),
                "fin": True,
            }
        )
        close_frame = fake.wait_for("WS_CLOSE")[0]
        assert close_frame["rid"] == "r1"
        assert close_frame["code"] == 1009
        assert fake.frames("WS_FRAME") == []
    finally:
        agent.stop()


def test_oversize_origin_message_closes_the_stream_with_1009(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(
        monkeypatch,
        fake,
        origin_port=port,
        max_request_body_bytes=4096,
    )
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [],
                "deadline_ms": 10000,
            }
        )
        fake.wait_for("WS_OPEN_ACK")
        # A payload at the limit round-trips.
        fake.push(
            {
                "type": "WS_FRAME",
                "lease_id": "lease_test",
                "rid": "r1",
                "opcode": "binary",
                "data_b64": base64.b64encode(b"y" * 4096).decode("ascii"),
                "fin": True,
            }
        )
        echoed = fake.wait_for("WS_FRAME")[0]
        assert len(base64.b64decode(echoed["data_b64"])) == 4096
        assert fake.frames("WS_CLOSE") == []

        # An origin message over the limit closes the stream with 1009 and never
        # reaches the relay.
        fake.push(
            {
                "type": "WS_FRAME",
                "lease_id": "lease_test",
                "rid": "r1",
                "opcode": "text",
                "data_b64": base64.b64encode(b"flood:9000").decode("ascii"),
                "fin": True,
            }
        )
        close_frame = fake.wait_for("WS_CLOSE")[0]
        assert close_frame["rid"] == "r1"
        assert close_frame["code"] == 1009
        assert len(fake.frames("WS_FRAME")) == 1
    finally:
        agent.stop()


def test_stream_queue_overflow_closes_that_stream_with_1013(
    monkeypatch: pytest.MonkeyPatch,
    refusing_origin_port: int,
) -> None:
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=refusing_origin_port)
    try:
        # Register a stream with no consumer so the bounded queue fills up.
        stream = relay_module._WebSocketStream(
            rid="r1",
            path="/socket",
            query="",
            headers=[],
            deadline_ms=10000,
            connection_generation=agent._connection_generation,
        )
        with agent._streams_lock:
            agent._streams["r1"] = stream
        payload = base64.b64encode(b"z").decode("ascii")
        for _ in range(relay_module._WEBSOCKET_STREAM_QUEUE_FRAMES + 1):
            fake.push(
                {
                    "type": "WS_FRAME",
                    "lease_id": "lease_test",
                    "rid": "r1",
                    "opcode": "text",
                    "data_b64": payload,
                    "fin": True,
                }
            )
        close_frame = fake.wait_for("WS_CLOSE")[0]
        assert close_frame["code"] == 1013
        assert agent._streams == {}
    finally:
        agent.stop()


def test_lease_teardown_closes_open_streams_with_1001(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=port)
    fake.push(
        {
            "type": "WS_OPEN",
            "lease_id": "lease_test",
            "rid": "r1",
            "path": "/socket",
            "query": "",
            "headers": [],
            "deadline_ms": 10000,
        }
    )
    fake.wait_for("WS_OPEN_ACK")
    agent.stop()
    close_frame = fake.wait_for("WS_CLOSE")[0]
    assert close_frame["rid"] == "r1"
    assert close_frame["code"] == 1001
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and echo_origin.closed_with == []:
        time.sleep(0.01)
    assert echo_origin.closed_with[0][0] == 1001


def test_agent_disconnect_closes_every_rid_with_1012(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(monkeypatch, fake, origin_port=port)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [],
                "deadline_ms": 10000,
            }
        )
        fake.wait_for("WS_OPEN_ACK")
        # Losing the relay socket is not a graceful lease close.
        fake.close()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and echo_origin.closed_with == []:
            time.sleep(0.01)
        assert echo_origin.closed_with[0][0] == 1012
    finally:
        agent.stop()


def test_open_sockets_have_their_own_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    echo_origin: _EchoOrigin,
) -> None:
    port = echo_origin.start()
    fake = _FakeRelaySocket()
    agent = _make_agent(
        monkeypatch,
        fake,
        origin_port=port,
        max_websocket_streams=1,
    )
    try:
        for rid in ("r1", "r2"):
            fake.push(
                {
                    "type": "WS_OPEN",
                    "lease_id": "lease_test",
                    "rid": rid,
                    "path": "/socket",
                    "query": "",
                    "headers": [],
                    "deadline_ms": 10000,
                }
            )
        fake.wait_for("WS_OPEN_ACK")
        reject = fake.wait_for("WS_OPEN_REJECT")[0]
        assert reject["rid"] == "r2"
        assert reject["status"] == 503
        # The HTTP request budget is untouched by the open socket.
        assert agent._request_slots._value == 4
    finally:
        agent.stop()


def test_unreachable_origin_is_reported_as_a_rejected_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRelaySocket()
    # Port 1 on loopback refuses connections.
    agent = _make_agent(monkeypatch, fake, origin_port=1)
    try:
        fake.push(
            {
                "type": "WS_OPEN",
                "lease_id": "lease_test",
                "rid": "r1",
                "path": "/socket",
                "query": "",
                "headers": [],
                "deadline_ms": 2000,
            }
        )
        reject = fake.wait_for("WS_OPEN_REJECT")[0]
        assert reject["status"] == 502
        assert b"dial failed" in base64.b64decode(reject["body_b64"])
    finally:
        agent.stop()


def test_local_websocket_url_maps_scheme_and_keeps_query() -> None:
    target = relay_module._parse_local_target("http://127.0.0.1:8123")
    assert (
        relay_module._local_websocket_url(target, "/v1/ws", "a=1&b=2")
        == "ws://127.0.0.1:8123/v1/ws?a=1&b=2"
    )
    secure = relay_module._parse_local_target("https://127.0.0.1:8443")
    assert relay_module._local_websocket_url(secure, "/ws", "") == "wss://127.0.0.1:8443/ws"
