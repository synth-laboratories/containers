"""Real journal history survives lifetime churn and a crash before archival."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import signal
import subprocess
import sys
from uuid import UUID, uuid4

import pytest

from synth_containers.checkpoints import CheckpointFailure, FileArtifactStore, FileCheckpointJournal


def test_completed_history_does_not_consume_active_capacity(tmp_path: Path):
    store = FileArtifactStore(tmp_path / "blobs", namespace="retention-fixture")
    artifact = store.put(b"immutable checkpoint")
    root = tmp_path / "journal"
    journal = FileCheckpointJournal(root)
    first = None
    for index in range(4097):
        operation = uuid4()
        digest = sha256(str(index).encode()).hexdigest()
        first = first or (operation, digest)
        assert journal.begin(operation, digest) is None
        journal.record(operation, request_sha256=digest, phase="complete", artifact=artifact)
    assert not list((root / "operations").iterdir())
    assert len(list((root / "archive").glob("*/*/state.json"))) == 4097
    restarted = FileCheckpointJournal(root)
    operation, digest = first
    saved = restarted.begin(operation, digest)
    assert saved["phase"] == "complete" and saved["artifact"] == artifact.model_dump(mode="json")
    restarted.record(operation, request_sha256=digest, phase="in_doubt")
    assert restarted.lookup(operation) == saved
    with pytest.raises(CheckpointFailure, match="operation_conflict"):
        restarted.begin(operation, "f" * 64)
    with pytest.raises(CheckpointFailure, match="operation_conflict"):
        restarted.record(operation, request_sha256=digest, phase="complete", artifact=store.put(b"other"))
    assert store.get(artifact) == b"immutable checkpoint"


def test_sigkill_between_completed_fsync_and_archive_never_replays(tmp_path: Path):
    # The child owns its process. Injection occurs at the exact archive boundary,
    # after completed-state fsync, with another unresolved operation still live.
    source = """
import json, os, signal, sys
from pathlib import Path
from uuid import uuid4
from synth_containers.checkpoints import FileArtifactStore, FileCheckpointJournal
root = Path(sys.argv[1])
journal = FileCheckpointJournal(root / 'journal')
artifact = FileArtifactStore(root / 'blobs', namespace='crash-fixture').put(b'saved')
pending, complete = uuid4(), uuid4()
journal.begin(pending, 'a' * 64)
journal.record(pending, request_sha256='a' * 64, phase='in_doubt', artifact=artifact)
journal.begin(complete, 'b' * 64)
def crash(operation, state):
    print(json.dumps({'pending':str(pending), 'complete':str(complete)}), flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
journal._archive_complete = crash
journal.record(complete, request_sha256='b' * 64, phase='complete', artifact=artifact)
"""
    process = subprocess.Popen([sys.executable, "-c", source, str(tmp_path)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        output, errors = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert process.returncode == -signal.SIGKILL, errors
    identities = json.loads(output)
    pending, complete = (UUID(identities[key]) for key in ("pending", "complete"))
    journal = FileCheckpointJournal(tmp_path / "journal")
    saved = journal.lookup(complete)
    assert saved["phase"] == "complete"
    assert journal.begin(complete, "b" * 64) == saved
    assert not (tmp_path / "journal/operations" / str(complete)).exists()
    assert journal.begin(pending, "a" * 64)["phase"] == "in_doubt"
    assert (tmp_path / "journal/operations" / str(pending)).exists()


def test_unresolved_capacity_is_still_bounded_and_completed_requires_artifact(tmp_path: Path):
    journal = FileCheckpointJournal(tmp_path)
    operation = uuid4()
    journal.begin(operation, "a" * 64)
    with pytest.raises(CheckpointFailure, match="complete_requires_artifact"):
        journal.record(operation, request_sha256="a" * 64, phase="complete")
    # Seed the rest of the durable unresolved inventory; no participant or
    # application effects are fabricated by these metadata fixtures.
    state = json.dumps(journal.lookup(operation))
    for _ in range(4095):
        directory = tmp_path / "operations" / str(uuid4())
        directory.mkdir()
        (directory / "state.json").write_text(state)
    with pytest.raises(CheckpointFailure, match="journal_capacity"):
        journal.begin(uuid4(), "c" * 64)
    assert len(list((tmp_path / "operations").iterdir())) == 4096
    assert not list((tmp_path / "archive").iterdir())
