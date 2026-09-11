import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from synth_containers.operator_journal import JournalIntegrityError, OperatorJournal


def test_independent_writers_and_restart_do_not_reuse_event_ids(tmp_path):
    path = tmp_path / "events.jsonl"

    def write(index):
        journal = OperatorJournal(path, run_id="r1")
        return journal.append(lambda seq: {"run_id": "r1", "seq": seq, "index": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(50)))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["seq"] for row in rows] == list(range(1, 51))
    assert write(50)["seq"] == 51


@pytest.mark.parametrize(
    "contents",
    [b'{"seq":1', b"not json\n", b'{"seq":true,"run_id":"r1"}\n', b'{"seq":1,"run_id":"r2"}\n'],
)
def test_damaged_or_foreign_history_is_preserved(tmp_path, contents):
    path = tmp_path / "events.jsonl"
    path.write_bytes(contents)
    with pytest.raises(JournalIntegrityError):
        OperatorJournal(path, run_id="r1").append(lambda seq: {"run_id": "r1", "seq": seq})
    assert path.read_bytes() == contents


def test_event_bound_and_identity_are_checked_before_append(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = OperatorJournal(path, run_id="r1", max_record_bytes=64)
    with pytest.raises(JournalIntegrityError):
        journal.append(lambda seq: {"run_id": "r1", "seq": seq, "text": "x" * 100})
    assert path.read_bytes() == b""
    with pytest.raises(JournalIntegrityError):
        journal.append(lambda seq: {"run_id": "r1", "seq": 100})


def test_replay_pages_resume_after_append_without_rewriting_history(tmp_path):
    from synth_containers.operator_journal import read_operator_events

    path = tmp_path / "events.jsonl"
    journal = OperatorJournal(path, run_id="r1")
    for _ in range(3):
        journal.append(lambda seq: {"run_id": "r1", "seq": seq, "event": "resource.created"})
    first = read_operator_events(path, limit=2)
    assert first["run_id"] == "r1"
    assert [row["seq"] for row in first["events"]] == [1, 2]
    assert first["has_more"] is True
    journal.append(lambda seq: {"run_id": "r1", "seq": seq, "event": "resource.cleanup_confirmed"})
    before = path.read_bytes()
    second = read_operator_events(path, run_id="r1", after_sequence=first["next_sequence"])
    assert [row["seq"] for row in second["events"]] == [3, 4]
    assert second["high_water"] == second["next_sequence"] == 4
    assert second["has_more"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "contents",
    [
        b'{"run_id":"r1","seq":1',
        b'{"run_id":"r1","seq":2}\n',
        b'{"run_id":"r1","seq":true}\n',
        b'{"run_id":"r1","seq":1,"value":NaN}\n',
        b'{"run_id":"r1","seq":1}\n{"run_id":"r2","seq":2}\n',
    ],
)
def test_replay_refuses_damaged_history_without_repairing_it(tmp_path, contents):
    from synth_containers.operator_journal import read_operator_events

    path = tmp_path / "events.jsonl"
    path.write_bytes(contents)
    with pytest.raises(JournalIntegrityError):
        read_operator_events(path)
    assert path.read_bytes() == contents


def test_replay_refuses_future_cursor_and_oversized_journal(tmp_path):
    from synth_containers.operator_journal import read_operator_events

    path = tmp_path / "events.jsonl"
    OperatorJournal(path, run_id="r1").append(lambda seq: {"run_id": "r1", "seq": seq})
    with pytest.raises(JournalIntegrityError, match="ahead"):
        read_operator_events(path, after_sequence=2)
    with pytest.raises(JournalIntegrityError, match="read bound"):
        read_operator_events(path, max_journal_bytes=1)


def test_cli_replays_journal_without_provider_credentials(tmp_path, capsys):
    from synth_containers.cli import main

    path = tmp_path / "events.jsonl"
    OperatorJournal(path, run_id="r1").append(
        lambda seq: {"run_id": "r1", "seq": seq, "event": "resource.cleanup_pending"}
    )
    assert main(["journal", str(path), "--run-id", "r1"]) == 0
    page = json.loads(capsys.readouterr().out)
    assert page["next_sequence"] == 1
    assert page["events"][0]["event"] == "resource.cleanup_pending"


def test_follow_resumes_and_observes_new_committed_event(tmp_path):
    from synth_containers.operator_journal import follow_operator_events

    path = tmp_path / "events.jsonl"
    journal = OperatorJournal(path, run_id="r1")
    journal.append(lambda seq: {"run_id": "r1", "seq": seq})
    journal.append(lambda seq: {"run_id": "r1", "seq": seq})
    stream = follow_operator_events(
        path, after_sequence=1, timeout_seconds=1, poll_interval_seconds=0.001
    )
    assert next(stream)["seq"] == 2
    journal.append(lambda seq: {"run_id": "r1", "seq": seq})
    assert next(stream)["seq"] == 3
    stream.close()
    assert len(path.read_text().splitlines()) == 3


def test_follow_pins_identity_across_polling(tmp_path):
    from synth_containers.operator_journal import follow_operator_events

    path = tmp_path / "events.jsonl"
    OperatorJournal(path, run_id="r1").append(lambda seq: {"run_id": "r1", "seq": seq})
    stream = follow_operator_events(path, timeout_seconds=1, poll_interval_seconds=0.001)
    assert next(stream)["seq"] == 1
    path.write_text('{"run_id":"r2","seq":1}\n')
    with pytest.raises(JournalIntegrityError, match="identity mismatch"):
        next(stream)


def test_follow_writer_contention_obeys_deadline(tmp_path):
    import fcntl

    from synth_containers.operator_journal import follow_operator_events

    path = tmp_path / "events.jsonl"
    OperatorJournal(path, run_id="r1").append(lambda seq: {"run_id": "r1", "seq": seq})
    with path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        assert (
            list(follow_operator_events(path, timeout_seconds=0.01, poll_interval_seconds=0.001))
            == []
        )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, 86401])
def test_follow_rejects_invalid_observation_deadline(tmp_path, timeout):
    from synth_containers.operator_journal import follow_operator_events

    with pytest.raises(ValueError, match="timeout_seconds"):
        list(follow_operator_events(tmp_path / "absent", timeout_seconds=timeout))
