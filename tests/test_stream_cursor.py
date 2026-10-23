"""Replay scope, publication races and crash ambiguity without provider calls."""

import asyncio
import base64
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from synth_containers import StreamCursorError, StreamCursorV1
from synth_containers.event_log import (
    LogEnvelope,
    RolloutEventLog,
    format_sse,
    iter_sse,
    poll_payload as raw_poll_payload,
)
from synth_containers.platform import create_compat_app
from synth_containers.tracing.adapters.sse import SSEDecoder
from synth_containers.stream_cursor import stream_scope

VECTORS = json.loads(
    (Path(__file__).parents[1] / "contracts/fixtures/stream/cursor-v1.json").read_text()
)


def scoped_poll_payload(*args, **kwargs):
    return raw_poll_payload(*args, **kwargs, scoped=True)


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=lambda v: v["name"])
def test_cursor_byte_vectors(vector: dict) -> None:
    scope = stream_scope("stream:fixture", "rollout_id", "fixture", vector["kinds"])
    assert scope.model_dump() == vector["scope"]
    cursor = StreamCursorV1.model_validate(vector["cursor"])
    assert cursor.encode() == vector["token"]
    assert StreamCursorV1.decode(vector["token"]) == cursor


def log_at(path: Path | None = None, *, stream_id="stream:fixture") -> RolloutEventLog:
    return RolloutEventLog(rollout_id="fixture", stream_id=stream_id, journal_path=path)


def test_legacy_cursor_shape_and_explicit_scoped_profile(tmp_path: Path) -> None:
    log = log_at()
    log.append("wanted", {})
    legacy = raw_poll_payload(log)["cursor"]
    keys = {"kind", "after", "high_water", "closed", "next", "has_more"}
    assert set(legacy) == keys
    scoped = raw_poll_payload(log, scoped=True)["cursor"]
    assert set(scoped) > keys and scoped["scanned_through"] == 1
    assert raw_poll_payload(log, cursor=scoped["resume"])["cursor"]["scope"] == scoped["scope"]

    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    app.state.platform.logs["fixture"] = log
    with TestClient(app) as client:
        assert set(client.get("/rollouts/fixture/events").json()["cursor"]) == keys | {"acked", "chain_head"}
        page = client.get("/rollouts/fixture/events", params={"scoped": "true"}).json()
        assert "resume" in page["cursor"] and "scope" in page["cursor"]


def test_filtered_cursor_scans_excluded_tail_without_claiming_delivery() -> None:
    log = log_at()
    log.append("wanted", {"n": 1})
    log.append("excluded", {"n": 2})
    first = scoped_poll_payload(log, kinds=["wanted"], limit=1)
    assert first["cursor"]["next"] == 1
    assert first["cursor"]["scanned_through"] == 2
    assert first["cursor"]["high_water"] == 2
    token = StreamCursorV1.decode(first["cursor"]["resume"])
    assert token.after == 2 and token.chain_head == log.chain_head
    assert first["cursor"]["durability"] == "memory"
    log.append("wanted", {"n": 3})
    second = scoped_poll_payload(log, cursor=token.encode(), kinds=["wanted"])
    assert [row["sequence"] for row in second["events"]] == [3]
    assert second["cursor"]["scanned_through"] == 3


def test_filtered_page_limit_never_skips_an_undelivered_matching_event() -> None:
    log = log_at()
    for kind in ("skip", "wanted", "skip", "wanted", "skip"):
        log.append(kind, {})
    first = scoped_poll_payload(log, kinds=["wanted"], limit=1)
    assert first["cursor"]["scanned_through"] == 2
    assert first["cursor"]["has_more"]
    second = scoped_poll_payload(log, kinds=["wanted"], limit=1, cursor=first["cursor"]["resume"])
    assert [row["sequence"] for row in second["events"]] == [4]
    assert second["cursor"]["scanned_through"] == 5
    assert not second["cursor"]["has_more"]


@pytest.mark.parametrize("change", ["stream", "subject", "filter", "subject-kind"])
def test_cursor_refuses_another_scope(change: str) -> None:
    log = log_at()
    log.append("wanted", {})
    token = scoped_poll_payload(log, kinds=["wanted"])["cursor"]["resume"]
    target = log_at(stream_id="other") if change == "stream" else log
    kwargs = {"kinds": ["wanted"]}
    if change == "filter":
        kwargs["kinds"] = None
    if change == "subject":
        kwargs["subject_id"] = "other"
    if change == "subject-kind":
        kwargs["subject_id_field"] = "job_id"
    with pytest.raises(StreamCursorError, match="stream_cursor_scope_mismatch"):
        scoped_poll_payload(target, cursor=token, **kwargs)


def test_cursor_refuses_changed_history_even_when_stream_names_match() -> None:
    original = log_at()
    original.append("wanted", {"n": 1})
    token = scoped_poll_payload(original)["cursor"]["resume"]
    rewritten = log_at()
    rewritten.append("wanted", {"n": 2})
    with pytest.raises(StreamCursorError, match="stream_cursor_history_mismatch"):
        scoped_poll_payload(rewritten, cursor=token)
    with pytest.raises(StreamCursorError, match="stream_cursor_ahead"):
        scoped_poll_payload(log_at(), cursor=token)


@pytest.mark.parametrize("after", [-1, True, False, 0.0, "0"])
def test_after_is_a_nonnegative_integer(after) -> None:
    with pytest.raises(StreamCursorError):
        scoped_poll_payload(log_at(), after=after)


@pytest.mark.parametrize("token", ["bad", "sc1.", "sc1.====", "sc1." + "a" * 4096])
def test_malformed_cursor_has_a_content_free_error(token: str) -> None:
    with pytest.raises(StreamCursorError, match="^invalid_stream_cursor$"):
        StreamCursorV1.decode(token)


def test_duplicate_keys_and_unknown_profile_refuse() -> None:
    for raw in (b'{"after":0,"after":1}', b'{"schema_version":"unknown"}'):
        token = "sc1." + base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        with pytest.raises(StreamCursorError, match="invalid_stream_cursor"):
            StreamCursorV1.decode(token)


def test_page_and_chain_share_one_snapshot_during_a_writer_interleaving(monkeypatch) -> None:
    log = log_at()
    log.append("wanted", {"n": 1})
    first_head = log.chain_head
    original = LogEnvelope.to_dict
    injected = False

    def serialize(envelope):
        nonlocal injected
        if not injected:
            injected = True
            log.append("wanted", {"n": 2})
            log.seal_capture()
        return original(envelope)

    monkeypatch.setattr(LogEnvelope, "to_dict", serialize)
    page = scoped_poll_payload(log)
    assert log.closed
    assert page["cursor"]["high_water"] == 1
    assert page["cursor"]["chain_head"] == first_head
    assert not page["cursor"]["closed"]
    tail = scoped_poll_payload(log, cursor=page["cursor"]["resume"])
    assert [row["sequence"] for row in tail["events"]] == [2, 3, 4]
    assert tail["cursor"]["closed"]


def test_sse_drains_records_when_writer_closes_between_replay_and_follow() -> None:
    class Request:
        async def is_disconnected(self):
            return False

    async def read():
        log = log_at()
        log.append("wanted", {"n": 1})
        stream = iter_sse(log, Request())
        first = await anext(stream)
        log.append("wanted", {"n": 2})
        log.seal_capture()
        rest = [row async for row in stream]
        return first, rest

    first, rest = asyncio.run(asyncio.wait_for(read(), timeout=2))
    assert "id: 1\n" in first
    assert any("id: 2\n" in row for row in rest)
    assert "event: capture.closed\n" in rest[-1]


def test_sse_control_never_resets_the_browser_cursor() -> None:
    log = log_at()
    control = log.append_control("stream.subscribed", {})
    assert "id:" not in format_sse(control)


def test_sse_refuses_header_injection_and_native_record_overrides(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = log_at(path)
    with pytest.raises(ValueError, match="invalid_stream_event_kind"):
        log.append("observation\nid: forged", {})
    assert not path.exists() and not log._write_failed
    row = log.append("observation", {})
    with pytest.raises(ValueError, match="stream_extra_overrides_native_event"):
        format_sse(row, {"payload": {"forged": True}})


def test_uncertain_append_is_fenced_until_exact_prefix_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    log = log_at(path)
    persist = log._persist

    def uncertain(row):
        persist(row)
        raise OSError("acknowledgment lost after file fsync")

    monkeypatch.setattr(log, "_persist", uncertain)
    with pytest.raises(OSError):
        log.append("wanted", {"n": 1})
    assert log.high_water == 0
    assert scoped_poll_payload(log)["cursor"]["writer_state"] == "recovery_required"
    with pytest.raises(RuntimeError, match="event_log_recovery_required"):
        log.append("wanted", {"n": 2})
    recovered = RolloutEventLog.recover(
        rollout_id="fixture", stream_id=log.stream_id, journal_path=path
    )
    assert recovered.high_water == 1
    assert recovered.append("wanted", {"n": 2}).sequence == 2


def test_recovery_refuses_an_unterminated_tail_and_preserves_bytes(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log_at(path).append("wanted", {})
    raw = path.read_bytes().rstrip(b"\n")
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="event_log_incomplete_tail"):
        RolloutEventLog.recover(rollout_id="fixture", stream_id="stream:fixture", journal_path=path)
    assert path.read_bytes() == raw


def test_missing_released_history_is_explicit_and_memory_only_history_is_not_discarded(
    tmp_path: Path,
) -> None:
    memory = log_at()
    memory.append("wanted", {})
    memory.mark_closed()
    assert memory.compact() == {"compacted": False, "reason": "not_persisted"}
    memory.release()
    assert len(memory.after(0)) == 1
    path = tmp_path / "events.jsonl"
    durable = log_at(path)
    durable.append("wanted", {})
    durable.mark_closed()
    durable.release()
    path.unlink()
    with pytest.raises(StreamCursorError, match="stream_history_unavailable"):
        scoped_poll_payload(durable)


def test_public_envelopes_are_detached_from_durable_history(tmp_path: Path) -> None:
    log = log_at(tmp_path / "events.jsonl")
    returned = log.append("wanted", {"nested": {"n": 1}})
    returned.payload["nested"]["n"] = 2
    log.after(0)[0].payload["nested"]["n"] = 3
    assert scoped_poll_payload(log)["events"][0]["payload"]["nested"]["n"] == 1


def test_corrupt_released_history_is_not_misreported_as_a_page_limit_error(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    path = tmp_path / "events.jsonl"
    log = log_at(path)
    app.state.platform.logs["fixture"] = log
    log.append("wanted", {})
    log.mark_closed()
    log.release()
    path.write_bytes(b"not-json\n")
    with TestClient(app) as client:
        response = client.get("/rollouts/fixture/events")
        assert response.status_code == 409
        assert response.json()["error"] == "stream_history_corrupt"
    assert path.read_bytes() == b"not-json\n"


def test_subject_metadata_cannot_overwrite_a_native_event_field() -> None:
    with pytest.raises(StreamCursorError, match="invalid_stream_scope"):
        scoped_poll_payload(log_at(), subject_id_field="payload")


def test_http_resume_filter_scope_and_invalid_cursor_does_not_ack(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    platform = app.state.platform
    log = log_at(tmp_path / "events.jsonl")
    platform.logs["fixture"] = log
    log.append("wanted", {})
    log.append("excluded", {})
    with TestClient(app) as client:
        page = client.get("/rollouts/fixture/events", params={"kinds": "wanted", "scoped": "true"}).json()
        token = page["cursor"]["resume"]
        assert client.get("/rollouts/fixture/events", params={"cursor": token}).status_code == 409
        assert log.last_acked == 0
        invalid = client.get("/rollouts/fixture/events", params={"cursor": "bad", "ack": 2})
        assert invalid.status_code == 422 and log.last_acked == 0
        resumed = client.get(
            "/rollouts/fixture/events", params={"cursor": token, "kinds": "wanted"}
        ).json()
        assert resumed["events"] == [] and resumed["cursor"]["scanned_through"] == 2


def test_scoped_sse_replays_filtered_history_and_emits_a_typed_watermark(tmp_path: Path) -> None:
    app = create_compat_app("openenv_echo", storage_root=tmp_path)
    with TestClient(app) as client:
        prepared = client.post(
            "/rollouts/prepare",
            json={
                "rollout_id": "fixture",
                "telemetry": {"enabled": True, "transport": "sse"},
            },
        )
        assert prepared.status_code == 200
        log = app.state.platform.logs["fixture"]
        log.append("wanted", {"n": 1})
        first = scoped_poll_payload(log, kinds=["wanted"])["cursor"]["resume"]
        log.append("excluded", {"n": 2})
        log.mark_closed()
        response = client.get(
            "/rollouts/fixture/stream", params={"scoped": True, "kinds": "wanted"}
        )
        assert response.status_code == 200
        decoder = SSEDecoder()
        events = [*decoder.feed(response.content), *decoder.finish()]
        data = next(event for event in events if event.event == "wanted")
        assert StreamCursorV1.decode(data.event_id).after == 1
        assert json.loads(data.data)["event_id"] == "1", "native row identity is preserved"
        watermark = json.loads(events[-1].data)
        assert events[-1].event == "stream.watermark"
        assert watermark["schema_version"] == "synth.stream-watermark.v1"
        assert watermark["cursor"]["after"] == watermark["high_water"] == 2
        assert watermark["closed"] and watermark["durability"] == "local_fsync"
        last = events[-1].event_id
        resumed = client.get("/rollouts/fixture/events", params={"cursor": last, "kinds": "wanted"})
        assert resumed.status_code == 200 and resumed.json()["events"] == []
        # The browser's reconnect position supersedes its original URL cursor.
        follow = client.get(
            "/rollouts/fixture/stream",
            params={"cursor": first, "kinds": "wanted"},
            headers={"Last-Event-ID": last},
        )
        assert follow.status_code == 200
        assert "event: wanted\n" not in follow.text
        assert (
            client.get("/rollouts/fixture/reward/stream", params={"cursor": last}).status_code
            == 409
        )
        for header in ("bad", "999", "-1"):
            assert client.get(
                "/rollouts/fixture/stream", headers={"Last-Event-ID": header}
            ).status_code in (409, 422)


def test_scoped_sse_reports_fenced_writer_without_inventing_capture_closure(monkeypatch) -> None:
    class Request:
        async def is_disconnected(self):
            return False

    log = log_at()

    def fail(_):
        raise OSError("synthetic journal outage")

    monkeypatch.setattr(log, "_persist", fail)
    with pytest.raises(OSError):
        log.append("wanted", {})

    async def read():
        return [row async for row in iter_sse(log, Request(), scoped=True)]

    rows = asyncio.run(read())
    assert len(rows) == 1 and "event: stream.watermark" in rows[0]
    assert '"writer_state":"recovery_required"' in rows[0]
    assert '"closed":false' in rows[0]
    assert "capture.closed" not in rows[0]


def test_ack_is_published_after_persistence_and_uncertainty_fences_the_owner(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "events.jsonl"
    log = log_at(path)
    log.append("wanted", {})
    persist = log._persist_ack

    def uncertain(acked):
        persist(acked)
        raise OSError("synthetic lost ack after rename")

    monkeypatch.setattr(log, "_persist_ack", uncertain)
    with pytest.raises(OSError):
        log.record_ack(1)
    assert log.last_acked == 0
    with pytest.raises(StreamCursorError, match="stream_writer_recovery_required"):
        log.record_ack(1)
    recovered = RolloutEventLog.recover(
        rollout_id="fixture", stream_id=log.stream_id, journal_path=path
    )
    assert recovered.last_acked == recovered.high_water == 1
