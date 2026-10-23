"""Multiplexed observation is bounded and preserves original event identities."""

import asyncio
import io
import re
import shutil
from functools import wraps

import pytest

from synth_containers.pools import PoolClientError, PoolRolloutTimeout
from synth_containers.rhodes_watch import RhodesTerminal, watch_rollouts


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


@run_async
async def test_independent_cursors_and_original_events():
    events = []
    seen = []

    class Client:
        async def follow_events(self, identity, *, after_sequence, timeout_seconds):
            seen.append((identity, after_sequence))
            event = {"rollout_id": identity, "sequence": after_sequence + 1,
                     "event_type": "rollout.completed", "score": 0}
            events.append(event)
            yield event

    emitted = []
    await watch_rollouts(Client(), ["a", "b"], cursors={"a": 7, "b": 11},
                         timeout_seconds=1, emit=emitted.append)
    assert sorted(seen) == [("a", 7), ("b", 11)]
    assert sorted(emitted, key=lambda event: event["rollout_id"]) == events
    assert all(actual is original for actual, original in zip(emitted, events))


@run_async
async def test_hundred_watches_bound_live_readers_and_leave_none_running():
    active = peak = closed = 0

    class Client:
        async def follow_events(self, identity, **kwargs):
            nonlocal active, peak, closed
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0)
                yield {"rollout_id": identity, "sequence": 1}
            finally:
                active -= 1
                closed += 1

    received = []
    await watch_rollouts(Client(), [str(i) for i in range(100)], cursors={},
                         timeout_seconds=2, emit=received.append)
    assert len(received) == closed == 100
    assert peak == 32 and active == 0


@run_async
@pytest.mark.parametrize("failure", ["deadline", "consumer", "reader", "external_cancel"])
async def test_failure_detaches_all_reader_tasks(failure):
    active = closed = 0
    started = asyncio.Event()

    class Client:
        async def follow_events(self, identity, **kwargs):
            nonlocal active, closed
            active += 1
            started.set()
            try:
                if failure == "reader" and identity == "a":
                    raise PoolClientError("original-refusal")
                if failure == "consumer":
                    yield {"rollout_id": identity, "sequence": 1}
                await asyncio.sleep(10)
            finally:
                active -= 1
                closed += 1

    def emit(event):
        raise RuntimeError("consumer-refusal")

    job = asyncio.create_task(watch_rollouts(
        Client(), ["a", "b"], cursors={}, timeout_seconds=0.02 if failure == "deadline" else 1,
        emit=emit,
    ))
    await started.wait()
    if failure == "external_cancel":
        job.cancel()
    expected = {
        "deadline": PoolRolloutTimeout, "consumer": RuntimeError,
        "reader": PoolClientError, "external_cancel": asyncio.CancelledError,
    }[failure]
    async with asyncio.timeout(2):
        with pytest.raises(expected):
            await job
    assert active == 0 and closed == 2


@run_async
@pytest.mark.parametrize("ids,cursors", [
    ([], {}), (["a", "a"], {}), ([str(i) for i in range(101)], {}),
    ([""], {}), (["a" * 257], {}), (["a"], {"b": 0}),
    (["a"], {"a": True}), (["a"], {"a": -1}), (["a"], {"a": 2**63}),
])
async def test_invalid_selection_refuses_before_reading(ids, cursors):
    class Client:
        def follow_events(self, *args, **kwargs):
            pytest.fail("invalid watch selection must not perform IO")

    with pytest.raises(ValueError):
        await watch_rollouts(Client(), ids, cursors=cursors, timeout_seconds=1, emit=lambda _: None)


@run_async
async def test_missing_multiplex_identity_refuses():
    class Client:
        async def follow_events(self, *args, **kwargs):
            yield {"sequence": 1}

    with pytest.raises(PoolClientError, match="exact rollout identity"):
        await watch_rollouts(Client(), ["a"], cursors={}, timeout_seconds=1, emit=lambda _: None)


def test_terminal_ten_rows_fixed_sidebar_and_active_tail(monkeypatch):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda _: (100, 24))
    stream = io.StringIO()
    terminal = RhodesTerminal(stream, [f"run-{i}" for i in range(15)])
    terminal.render()
    frame = stream.getvalue()
    assert frame.count(" | ") == 10 and "+5 additional rollouts" in frame
    assert "unobserved" in frame
    assert "0/" not in frame
    for i in range(10):
        terminal({"rollout_id": f"run-{i}", "sequence": 2, "event_type": "rollout.completed"})
    frame = stream.getvalue().rsplit("\x1b[12A", 1)[-1]
    assert "run-14" in frame and frame.count(" | ") == 10
    positions = [line.index(" | ") for line in frame.splitlines() if " | " in line]
    assert len(set(positions)) == 1


def test_terminal_publication_status_is_separate_and_controls_cannot_escape(monkeypatch):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda _: (70, 8))
    stream = io.StringIO()
    identity = "run\x1b]52;c;clipboard\x07\n\u202e"
    terminal = RhodesTerminal(stream, [identity])
    terminal({"rollout_id": identity, "sequence": 1, "event_type": "rollout.running"})
    terminal({"rollout_id": identity, "sequence": 2, "event_type": "rollout.native_service_trace_published",
              "status": "committed", "created_at": "\x1b[2J\r\n"})
    assert terminal.rows[identity]["status"] == "running"
    text = re.sub(r"\x1b\[(?:\d+A|2K)", "", stream.getvalue())
    assert "\x1b" not in text and "\x07" not in text and "\r" not in text and "\u202e" not in text
    assert max(map(len, text.splitlines())) <= 69
    monkeypatch.setattr(shutil, "get_terminal_size", lambda _: (2, 3))
    terminal.render()
