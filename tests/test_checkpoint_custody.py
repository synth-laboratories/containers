"""Native checkpoint acknowledgment binds immutable facts to durable bytes."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from synth_containers.platform.state import CompatPlatform
from synth_containers.platform.targets import TARGETS


def platform(root):
    return CompatPlatform(TARGETS["openenv_echo"], storage_root=root)


@pytest.mark.parametrize("field,value", [
    ("target_id", "foreign"), ("schema_version", "forged"), ("content_digest", "forged"),
])
def test_producer_cannot_override_checkpoint_authority(tmp_path, field, value):
    store = platform(tmp_path)
    with pytest.raises(ValueError, match="checkpoint_reserved"):
        store.record_checkpoint({"checkpoint_id": "cut", field: value})
    assert not store.checkpoints
    assert not list((tmp_path / "checkpoints").glob("*.json"))


@pytest.mark.parametrize("identity", [None, 17, "", " padded "])
def test_invalid_checkpoint_identity_never_acknowledges_an_unrecoverable_cut(tmp_path, identity):
    with pytest.raises(ValueError, match="checkpoint_id"):
        platform(tmp_path).record_checkpoint({"checkpoint_id": identity})
    assert not list((tmp_path / "checkpoints").glob("*.json"))


def test_input_initial_ack_and_replay_ack_cannot_mutate_retained_state(tmp_path):
    store = platform(tmp_path)
    source = {"checkpoint_id": "cut", "policy_state": {"calls": 1, "messages": ["original"]}}
    original = json.loads(json.dumps(source))
    accepted = store.record_checkpoint(source)
    path = store._checkpoint_path("cut")
    sealed = path.read_bytes()
    source["policy_state"]["messages"].append("changed input")
    accepted["policy_state"]["calls"] = 99
    assert store.checkpoints["cut"]["policy_state"] == original["policy_state"]
    replay = store.record_checkpoint(original)
    replay["policy_state"]["messages"].append("changed replay")
    assert store.checkpoints["cut"]["policy_state"] == original["policy_state"]
    assert path.read_bytes() == sealed
    assert platform(tmp_path).checkpoints == store.checkpoints


def test_native_json_normalization_matches_restart_and_rejects_nonfinite(tmp_path):
    store = platform(tmp_path)
    accepted = store.record_checkpoint({"checkpoint_id": "tuple", "coordinates": (1, 2)})
    assert accepted == platform(tmp_path).checkpoints["tuple"]
    with pytest.raises(ValueError):
        store.record_checkpoint({"checkpoint_id": "nan", "reward": float("nan")})
    assert "nan" not in store.checkpoints


@pytest.mark.parametrize("changed", [False, True])
def test_lost_write_ack_reuses_original_disk_commit_without_overwrite(tmp_path, monkeypatch, changed):
    store = platform(tmp_path)
    write = store._atomic_json
    calls = 0

    def lost_ack(path, body):
        nonlocal calls
        calls += 1
        write(path, body)
        raise OSError("lost acknowledgement after replace")

    monkeypatch.setattr(store, "_atomic_json", lost_ack)
    original = {"checkpoint_id": "cut", "policy_state": {"calls": 1}}
    with pytest.raises(OSError):
        store.record_checkpoint(original)
    sealed = store._checkpoint_path("cut").read_bytes()
    assert "cut" not in store.checkpoints
    if changed:
        with pytest.raises(ValueError, match="checkpoint_identity_conflict"):
            store.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 999}})
    else:
        assert store.record_checkpoint(original) == json.loads(sealed)
    assert calls == 1 and store._checkpoint_path("cut").read_bytes() == sealed


def test_foreign_target_existing_file_is_preserved(tmp_path):
    first = platform(tmp_path)
    first.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 1}})
    sealed = first._checkpoint_path("cut").read_bytes()
    other = CompatPlatform(TARGETS["digbench_mock"], storage_root=tmp_path)
    with pytest.raises(ValueError, match="checkpoint_identity_conflict"):
        other.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 2}})
    assert first._checkpoint_path("cut").read_bytes() == sealed


def test_denied_read_is_not_absence_or_permission_to_write(tmp_path, monkeypatch):
    store = platform(tmp_path)

    def denied(path):
        raise PermissionError("not readable")

    monkeypatch.setattr(store, "_read_checkpoint_record", denied)
    monkeypatch.setattr(store, "_atomic_json", lambda *args: pytest.fail("denied read must not write"))
    with pytest.raises(PermissionError):
        store.record_checkpoint({"checkpoint_id": "cut"})
    assert not store.checkpoints


def test_corrupt_committed_record_is_never_overwritten(tmp_path):
    store = platform(tmp_path)
    source = {"checkpoint_id": "cut", "policy_state": {"calls": 1}}
    accepted = store.record_checkpoint(source)
    accepted["policy_state"]["calls"] = 999
    path = store._checkpoint_path("cut")
    path.write_text(json.dumps(accepted))
    damaged = path.read_bytes()
    with pytest.raises(ValueError, match="checkpoint_digest"):
        store.record_checkpoint(source)
    assert path.read_bytes() == damaged


def test_failed_delete_retains_memory_and_original_identity(tmp_path, monkeypatch):
    store = platform(tmp_path)
    original = store.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 1}})
    unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == store._checkpoint_path("cut"):
            raise PermissionError("not removed")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", denied)
    with pytest.raises(PermissionError):
        store.drop_checkpoint("cut")
    assert store.checkpoints["cut"] == original
    with pytest.raises(ValueError, match="checkpoint_identity_conflict"):
        store.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 999}})


def test_unconfirmed_delete_keeps_identity_and_retry_confirms_absence(tmp_path, monkeypatch):
    store = platform(tmp_path)
    accepted = store.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 1}})
    barrier = store._confirm_checkpoint_directory

    def refused(path):
        raise OSError("directory barrier not confirmed")

    monkeypatch.setattr(store, "_confirm_checkpoint_directory", refused)
    with pytest.raises(OSError):
        store.drop_checkpoint("cut")
    assert not store._checkpoint_path("cut").exists()
    assert store.checkpoints["cut"] == accepted
    with pytest.raises(ValueError, match="checkpoint_retention_subject_missing"):
        store.retain_checkpoint("cut", owner_kind="fork", owner_id="child")
    monkeypatch.setattr(store, "_confirm_checkpoint_directory", barrier)
    store.drop_checkpoint("cut")
    assert "cut" not in store.checkpoints
    assert platform(tmp_path).checkpoints == {}


def test_foreign_target_delete_refuses_and_preserves_file(tmp_path):
    first = platform(tmp_path)
    first.record_checkpoint({"checkpoint_id": "cut"})
    sealed = first._checkpoint_path("cut").read_bytes()
    other = CompatPlatform(TARGETS["digbench_mock"], storage_root=tmp_path)
    with pytest.raises(ValueError, match="checkpoint_identity_conflict"):
        other.drop_checkpoint("cut")
    assert first._checkpoint_path("cut").read_bytes() == sealed


def test_concurrent_conflicting_writes_commit_one_original_cut(tmp_path, monkeypatch):
    store = platform(tmp_path)
    entered, release, second_started, overlap = (threading.Event() for _ in range(4))
    write = store._atomic_json
    writes = 0

    def blocked(path, body):
        nonlocal writes
        writes += 1
        if writes > 1:
            overlap.set()
        entered.set()
        assert release.wait(2)
        write(path, body)

    def second():
        second_started.set()
        return store.record_checkpoint({"checkpoint_id": "cut", "policy_state": {"calls": 2}})

    monkeypatch.setattr(store, "_atomic_json", blocked)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(store.record_checkpoint, {"checkpoint_id": "cut", "policy_state": {"calls": 1}})
        assert entered.wait(2)
        later = executor.submit(second)
        try:
            assert second_started.wait(2)
            raced = overlap.wait(0.05)
        finally:
            release.set()
        assert first.result()["policy_state"]["calls"] == 1
        assert not raced
        with pytest.raises(ValueError, match="checkpoint_identity_conflict"):
            later.result()
    assert writes == 1
    assert platform(tmp_path).checkpoints["cut"]["policy_state"]["calls"] == 1
