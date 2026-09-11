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
