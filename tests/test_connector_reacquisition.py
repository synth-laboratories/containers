"""Controlled ACK races drive the actual agent thread; not production auth proof."""

import json
import queue
import threading
import socket
import time
import base64
import hashlib
from uuid import uuid4

import pytest

from synth_containers.tunnels import relay
from synth_containers.tunnels.connector_control import PreparedAttachment
from synth_containers.tunnels.protocol import SUBPROTOCOL
from synth_containers.tunnels.targets import LocalTargetRegistry


def setup_agent(monkeypatch, *, acknowledged_session=None, acknowledged_generation=2):
    connector, client, session = uuid4(), uuid4(), uuid4()
    operations = []
    sockets = []

    class Provider:
        def prepare(self, *, operation, **kwargs):
            operations.append(operation)
            return PreparedAttachment(
                "ws://127.0.0.1:1/", "local-fixture", session, min(len(operations), 2)
            )

    class Socket:
        subprotocol = SUBPROTOCOL

        def __init__(self):
            self.messages = queue.Queue()
            self.closed = threading.Event()

        def send(self, raw):
            assert json.loads(raw)["type"] == "CONNECTOR_ATTACH"
            self.messages.put(
                json.dumps(
                    {
                        "type": "CONNECTOR_ATTACH_ACK",
                        "connector": str(connector),
                        "session": str(acknowledged_session or session),
                        "server_config": {
                            "protocol": SUBPROTOCOL,
                            "attachment_generation": acknowledged_generation,
                        },
                    }
                )
            )

        def recv(self):
            value = self.messages.get(timeout=5)
            if value is None:
                raise OSError("controlled socket closed")
            return value

        def close(self):
            self.closed.set()
            self.messages.put(None)

    def connect(*args, **kwargs):
        socket = Socket()
        sockets.append(socket)
        return socket

    monkeypatch.setattr(relay, "_connect_websocket", connect)
    agent = relay.SynthTunnelConnectorAgent(
        connector_id=connector,
        client_instance_id=client,
        targets=LocalTargetRegistry(),
        provider=Provider(),
        max_in_flight_requests=4,
        max_request_body_bytes=1024 * 1024,
    )
    return agent, operations, sockets


def test_newer_ack_requires_fresh_prepare_before_admission(monkeypatch):
    agent, operations, sockets = setup_agent(monkeypatch)
    try:
        agent.start(timeout_seconds=3)
        assert agent._ready.is_set() and not agent._fatal.is_set()
        assert agent._prepared.generation == agent._attachment_generation == 2
        assert len(operations) == 2 and operations[0] != operations[1]
        assert sockets[0].closed.is_set(), "superseded transport retained"
    finally:
        agent.stop()
        assert agent._thread is None


def test_wrong_session_ack_still_fails_closed(monkeypatch):
    agent, operations, sockets = setup_agent(
        monkeypatch, acknowledged_session=uuid4(), acknowledged_generation=1
    )
    try:
        with pytest.raises(relay.SynthTunnelRelayError):
            agent.start(timeout_seconds=3)
        assert not agent._ready.is_set() and agent._fatal.is_set()
        assert len(operations) == 1
    finally:
        agent.stop()
        assert sockets[0].closed.is_set() and agent._thread is None


def test_remote_close_does_not_wait_for_backpressured_tcp_handshake():
    """Real WebSocket peer sends Close but deliberately retains its TCP side."""
    from websockets.exceptions import ConnectionClosed

    close_now, release, sent = threading.Event(), threading.Event(), threading.Event()
    errors = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(3)

        def peer():
            try:
                with listener.accept()[0] as transport:
                    transport.settimeout(3)
                    request = bytearray()
                    while b"\r\n\r\n" not in request:
                        chunk = transport.recv(4096)
                        assert chunk and len(request) < 16384
                        request.extend(chunk)
                    headers = dict(line.split(":", 1) for line in request.decode().split("\r\n")[1:] if ":" in line)
                    key = headers["Sec-WebSocket-Key"].strip()
                    accepted = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
                    transport.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + accepted + "\r\nSec-WebSocket-Protocol: " + SUBPROTOCOL + "\r\n\r\n").encode())
                    assert close_now.wait(3)
                    transport.sendall(b"\x88\x02\x03\xe8")
                    sent.set()
                    release.wait(3)
            except BaseException as error:
                errors.append(type(error).__name__)

        thread = threading.Thread(target=peer)
        thread.start()
        connection = None
        try:
            connection = relay._connect_websocket(
                f"ws://127.0.0.1:{listener.getsockname()[1]}/",
                headers={}, max_message_bytes=1024, subprotocols=[SUBPROTOCOL],
            )
            began = time.monotonic()
            close_now.set()
            with pytest.raises(ConnectionClosed):
                connection.recv()
            elapsed = time.monotonic() - began
            assert sent.is_set() and not release.is_set()
            assert elapsed < 0.4, "remote close retained origin admission through TCP handshake"
        finally:
            release.set()
            if connection is not None:
                connection.close()
            thread.join(4)
        assert not thread.is_alive() and not errors
