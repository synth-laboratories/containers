"""Bounded idle origin transports. See tunnels/v2_catalog.md.

The caller exclusively owns borrowed connections. Failed/cancelled operations
are never retried or returned. Pooling carries no credential or access policy.
"""

import http.client
import threading
import time

OriginPoolScope = tuple[str, str, int] | None
OriginConnectionKey = tuple[int, OriginPoolScope, str, str, int | None]


class OriginConnections:
    def __init__(self, capacity: int = 8, idle_seconds: float = 30) -> None:
        assert 1 <= capacity <= 64 and 0 < idle_seconds <= 60
        self.capacity = capacity
        self.idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._idle: list[tuple[OriginConnectionKey, float, http.client.HTTPConnection]] = []
        self._epoch = 0

    def borrow(
        self, key: OriginConnectionKey, *, timeout_seconds: float
    ) -> tuple[http.client.HTTPConnection, int]:
        assert timeout_seconds > 0
        now = time.monotonic()
        selected: http.client.HTTPConnection | None = None
        with self._lock:
            epoch = self._epoch
            retained = []
            for entry_key, released, connection in self._idle:
                if now - released >= self.idle_seconds or connection.sock is None:
                    connection.close()
                elif selected is None and entry_key == key:
                    selected = connection
                else:
                    retained.append((entry_key, released, connection))
            self._idle = retained
        if selected is None:
            _, _, scheme, hostname, port = key
            connection_type = (
                http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            )
            selected = connection_type(hostname, port, timeout=timeout_seconds)
        selected.timeout = timeout_seconds
        if selected.sock is not None:
            selected.sock.settimeout(timeout_seconds)
        return selected, epoch

    def release(
        self, key: OriginConnectionKey, connection: http.client.HTTPConnection, epoch: int
    ) -> None:
        with self._lock:
            if epoch != self._epoch or connection.sock is None or len(self._idle) >= self.capacity:
                connection.close()
            else:
                self._idle.append((key, time.monotonic(), connection))

    def retire(self) -> None:
        with self._lock:
            self._epoch += 1
            for _, _, connection in self._idle:
                connection.close()
            self._idle.clear()
