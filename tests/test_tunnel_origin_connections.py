"""Real HTTP keepalive, scope isolation and retirement barriers."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import pytest

from synth_containers.tunnels.origin_connections import OriginConnections


@pytest.fixture
def origin():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert body == b"effect"
            self.server.effects.append((self.client_address[1], self.headers["x-api-key"]))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.effects = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def effect(connection, credential):
    connection.request("POST", "/", b"effect", {"x-api-key": credential})
    with connection.getresponse() as response:
        assert response.status == 200 and response.read() == b"ok"


def test_reuse_preserves_request_credentials_and_retirement_closes_late_return(origin):
    pool = OriginConnections(capacity=1)
    key = (1, ("origin-a", "service-a", 1), "http", "127.0.0.1", origin.server_port)
    connection, epoch = pool.borrow(key, timeout_seconds=2)
    effect(connection, "first")
    pool.release(key, connection, epoch)
    reused, epoch = pool.borrow(key, timeout_seconds=2)
    effect(reused, "rotated")
    assert [key for _, key in origin.effects] == ["first", "rotated"]
    assert origin.effects[0][0] == origin.effects[1][0]
    # Stop raced the completed request's return. Its old borrow must not reopen
    # the pool after retirement, even with an otherwise reusable socket.
    pool.retire()
    pool.release(key, reused, epoch)
    assert reused.sock is None
    fresh, epoch = pool.borrow(key, timeout_seconds=2)
    try:
        effect(fresh, "after-restart")
        assert origin.effects[-1][0] != origin.effects[0][0]
    finally:
        fresh.close()
        pool.retire()


@pytest.mark.parametrize(
    "second_scope",
    [
        ("origin-b", "service-a", 1),
        ("origin-a", "service-b", 1),
        ("origin-a", "service-a", 2),
    ],
)
def test_target_service_incarnation_isolation_and_capacity(origin, second_scope):
    pool = OriginConnections(capacity=1)
    first_key = (1, ("origin-a", "service-a", 1), "http", "127.0.0.1", origin.server_port)
    second_key = (1, second_scope, "http", "127.0.0.1", origin.server_port)
    first, epoch = pool.borrow(first_key, timeout_seconds=2)
    effect(first, "a")
    pool.release(first_key, first, epoch)
    second, epoch = pool.borrow(second_key, timeout_seconds=2)
    effect(second, "b")
    assert origin.effects[0][0] != origin.effects[1][0]
    pool.release(second_key, second, epoch)
    assert second.sock is None, "idle capacity exceeded"
    pool.retire()
    assert first.sock is None


def test_expired_idle_connection_is_closed_before_next_borrow(origin):
    pool = OriginConnections(capacity=1, idle_seconds=0.02)
    first_key = (1, ("origin-a", "service-a", 1), "http", "127.0.0.1", origin.server_port)
    first, epoch = pool.borrow(first_key, timeout_seconds=2)
    effect(first, "a")
    pool.release(first_key, first, epoch)
    time.sleep(0.03)
    fresh, epoch = pool.borrow(first_key, timeout_seconds=2)
    try:
        effect(fresh, "a-fresh")
        assert first.sock is None
        assert origin.effects[-1][0] != origin.effects[0][0]
    finally:
        fresh.close()
        pool.retire()


def test_peer_closed_idle_connection_is_discarded_before_effect():
    import http.client
    import select
    import socket

    local, peer = socket.socketpair()
    connection = http.client.HTTPConnection("127.0.0.1")
    connection.sock = local
    pool = OriginConnections()
    key = (1, None, "http", "127.0.0.1", 80)
    pool.release(key, connection, 0)
    peer.close()
    assert select.select([local], [], [], 1)[0]
    fresh, _ = pool.borrow(key, timeout_seconds=2)
    assert fresh is not connection
    assert connection.sock is None
    fresh.close()
    pool.retire()
