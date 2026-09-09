"""The relay agent must forward response bytes as they arrive, not in one lump.

Found end to end, not by unit test: with a real relay binary and a real SSE
origin, six events the origin emitted 300ms apart arrived through the tunnel
with a spread of 0.00s. The relay's streaming was correct; the agent upstream of
it was reading with `read(65536)`, which blocks until it has the full 64 KiB or
the stream ends, so nothing left the agent until the origin closed.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from synth_containers.tunnels.relay import SynthTunnelRelayAgent, _PendingRequest, _parse_local_target


class _TricklingResponse:
    """An origin that yields a little at a time.

    `read(n)` blocks until it has all `n` bytes (what a buffered socket reader
    does); `read1(n)` returns whatever has arrived. Only `read1` can stream.
    """

    def __init__(self, pieces: list[bytes]) -> None:
        self._pieces = list(pieces)

    def read(self, _amt: int) -> bytes:
        joined, self._pieces = b"".join(self._pieces), []
        return joined

    def read1(self, _amt: int) -> bytes:
        return self._pieces.pop(0) if self._pieces else b""


class _ReadOnlyResponse:
    """An origin object with no `read1` at all — the fallback must still work."""

    def __init__(self, pieces: list[bytes]) -> None:
        self._pieces = list(pieces)

    def read(self, _amt: int) -> bytes:
        return self._pieces.pop(0) if self._pieces else b""


def _agent() -> SynthTunnelRelayAgent:
    return SynthTunnelRelayAgent(
        lease_id="lease-test",
        local_target=_parse_local_target("http://127.0.0.1:9"),
        agent_connect={"transport": "ws", "url": "ws://127.0.0.1:9/agent", "agent_token": "t"},
        max_in_flight_requests=4,
        max_request_body_bytes=1024 * 1024,
    )


def _capture(agent: SynthTunnelRelayAgent, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    monkeypatch.setattr(agent, "_send_frame", lambda payload, **_: frames.append(payload))
    return frames


def _request() -> _PendingRequest:
    return _PendingRequest(
        method="GET", path="/sse", query="", headers=[], deadline_ms=30_000,
        connection_generation=1, received_monotonic=time.monotonic(),
    )


def test_response_bytes_are_forwarded_as_they_arrive(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    frames = _capture(agent, monkeypatch)
    pieces = [b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"]

    agent._send_response("rid-1", 200, {"content-type": "text/event-stream"},
                         _TricklingResponse(pieces), _request())

    kinds = [f["type"] for f in frames]
    assert kinds[0] == "RESP_HEADERS"
    assert kinds[-1] == "RESP_END"
    body_frames = [f for f in frames if f["type"] == "RESP_BODY"]
    # One frame per piece. Collapsing to a single frame is the bug: the client
    # then sees the whole stream at the end, with every gap gone.
    assert len(body_frames) == len(pieces), f"expected {len(pieces)} chunks, got {len(body_frames)}"


def test_an_origin_without_read1_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    frames = _capture(agent, monkeypatch)

    agent._send_response("rid-2", 200, {}, _ReadOnlyResponse([b"one", b"two"]), _request())

    body_frames = [f for f in frames if f["type"] == "RESP_BODY"]
    assert len(body_frames) == 2
    assert [f["type"] for f in frames][-1] == "RESP_END"
