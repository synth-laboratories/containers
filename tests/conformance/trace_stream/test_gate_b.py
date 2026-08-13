"""TS-B evidence and ordering against the durable journal."""

from __future__ import annotations

from pathlib import Path

from synth_containers.event_log import RolloutEventLog
from synth_containers.tracing.streaming.lifecycle import semantic


def test_ts_b01_b02_envelopes_verify_and_sequences_are_contiguous(tmp_path: Path) -> None:
    journal = tmp_path / "event_logs" / "roll_ts_b.jsonl"
    log = RolloutEventLog(rollout_id="roll_ts_b", stream_id="stream_ts_b", journal_path=journal)
    log.append("trace.opened", {"rollout_id": "roll_ts_b"})
    log.append("observation", {"step": 0})
    log.append("capture.closed", {"high_water": 2})
    recovered = RolloutEventLog.recover(
        rollout_id="roll_ts_b",
        stream_id="stream_ts_b",
        journal_path=journal,
    )
    events = [item.to_dict() for item in recovered.after(0)]
    sequences = [item["sequence"] for item in semantic(events)]
    assert sequences == list(range(1, len(sequences) + 1))
    for item in events:
        assert item["digest"]
        assert item["event_id"]


def test_ts_b03_duplicate_delivery_is_identical() -> None:
    log = RolloutEventLog(rollout_id="roll_ts_b03", stream_id="stream_ts_b03")
    log.append("trace.opened", {"rollout_id": "roll_ts_b03"})
    first = log.after(0)[0].to_dict()
    again = log.after(0)[0].to_dict()
    assert first["digest"] == again["digest"]
    assert first["event_id"] == again["event_id"]


def test_ts_b04_b05_prefix_is_deterministic_and_monotonic() -> None:
    log = RolloutEventLog(rollout_id="roll_ts_b04", stream_id="stream_ts_b04")
    log.append("trace.opened", {"rollout_id": "roll_ts_b04"})
    prefix = [item.to_dict() for item in log.after(0)]
    log.append("observation", {"step": 0})
    extended = [item.to_dict() for item in log.after(0)]
    assert extended[: len(prefix)] == prefix


def test_ts_b06_journal_is_newline_terminated(tmp_path: Path) -> None:
    journal = tmp_path / "event_logs" / "roll_ts_b06.jsonl"
    log = RolloutEventLog(rollout_id="roll_ts_b06", stream_id="stream_ts_b06", journal_path=journal)
    log.append("trace.opened", {"rollout_id": "roll_ts_b06"})
    text = journal.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert all(line for line in text.splitlines())
