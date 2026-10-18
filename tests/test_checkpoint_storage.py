"""Artifact integrity and immutable publish under competing writers."""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from synth_containers.checkpoints import FileArtifactStore, CheckpointFailure


def test_duplicate_parallel_publish_is_one_immutable_blob(tmp_path: Path):
    store = FileArtifactStore(tmp_path, namespace="owned-fixture")
    payload = b"joint application state" * 1000
    with ThreadPoolExecutor(max_workers=8) as workers:
        references = list(workers.map(store.put, [payload] * 16))
    assert len(set(reference.sha256 for reference in references)) == 1
    assert store.get(references[0]) == payload
    assert len(list((tmp_path / "sha256").iterdir())) == 1


def test_same_length_corruption_and_cross_namespace_are_refused(tmp_path: Path):
    store = FileArtifactStore(tmp_path, namespace="owned-fixture")
    item = store.put(b"saved state")
    (tmp_path / item.object_key).write_bytes(b"other state")
    with pytest.raises(CheckpointFailure, match="digest_mismatch"):
        store.get(item)
    with pytest.raises(CheckpointFailure, match="artifact_scope"):
        store.get(item.model_copy(update={"namespace": "another-tenant"}))
