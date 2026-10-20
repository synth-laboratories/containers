"""Recovery needs custody, a dead worker, the right daemon and typed absence."""

import fcntl
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("docker")
from docker.errors import NotFound

from synth_containers.harbor_docker_recovery import DockerRecoveryError, reconcile_docker_trial
from synth_containers.operator_journal import OperatorJournal

OWNER = "synth-harbor-" + "a" * 32


def fixture(tmp_path, *, known=True):
    (tmp_path / "resource-create-claim.json").write_text(
        json.dumps(
            {
                "owner": OWNER,
                "provider": "docker",
                "recovery_protocol": "owner_lock.v1",
                "daemon_id": "daemon-one",
            }
        )
    )
    (tmp_path / "resource-owner.lock").touch()
    journal = OperatorJournal(tmp_path / "resource-events.jsonl", run_id=OWNER)
    journal.append(
        lambda seq: {
            "run_id": OWNER,
            "seq": seq,
            "event": "resource.create_requested",
            "provider": "docker",
            "creation_timeout_seconds": 30,
            "occurred_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        }
    )
    if known:
        journal.append(
            lambda seq: {
                "run_id": OWNER,
                "seq": seq,
                "event": "resource.handles_observed",
                "handles": [{"kind": "container", "id": "owned-container"}],
            }
        )
    client = SimpleNamespace(
        api=SimpleNamespace(timeout=5), info=Mock(return_value={"ID": "daemon-one"})
    )
    for kind in ("containers", "networks", "volumes"):
        setattr(
            client,
            kind,
            SimpleNamespace(list=Mock(return_value=[]), get=Mock(side_effect=NotFound("gone"))),
        )
    return client, journal


def rows(journal):
    return [json.loads(row) for row in journal.path.read_text().splitlines()]


def resource(owner=OWNER):
    return SimpleNamespace(
        id="owned-container",
        attrs={"Config": {"Labels": {"com.docker.compose.project": owner}}},
        remove=Mock(),
    )


def test_known_absence_needs_fresh_typed_lookup_and_empty_owner_listing(tmp_path):
    client, journal = fixture(tmp_path)
    result = reconcile_docker_trial(tmp_path, client)
    assert result["event"] == "resource.cleanup_confirmed"
    assert result["handles"] == [{"kind": "container", "id": "owned-container"}]
    assert client.containers.get.call_count == 3
    assert client.containers.list.call_count == 2
    assert rows(journal)[-1] == result


def test_discovered_identity_is_saved_before_delete(tmp_path):
    client, journal = fixture(tmp_path, known=False)
    item = resource()
    client.containers.list.side_effect = [[item], []]
    client.containers.get.side_effect = [item, item, NotFound("gone")]

    def remove(**kwargs):
        assert rows(journal)[-1]["event"] == "resource.cleanup_requested"
        assert rows(journal)[-1]["handles"] == [{"kind": "container", "id": item.id}]
        assert kwargs == {"force": True}

    item.remove.side_effect = remove
    assert reconcile_docker_trial(tmp_path, client)["event"] == "resource.cleanup_confirmed"
    item.remove.assert_called_once()


def test_empty_unknown_creation_remains_pending(tmp_path):
    client, journal = fixture(tmp_path, known=False)
    with pytest.raises(DockerRecoveryError, match="no observed primary"):
        reconcile_docker_trial(tmp_path, client)
    assert rows(journal)[-1]["event"] == "resource.cleanup_pending"


def test_foreign_saved_handle_is_not_deleted(tmp_path):
    client, journal = fixture(tmp_path)
    item = resource("foreign")
    client.containers.get.side_effect = None
    client.containers.get.return_value = item
    with pytest.raises(DockerRecoveryError, match="ownership mismatch"):
        reconcile_docker_trial(tmp_path, client)
    item.remove.assert_not_called()
    assert rows(journal)[-1]["event"] == "resource.cleanup_pending"


def test_transport_failure_cannot_become_absence(tmp_path):
    client, journal = fixture(tmp_path)
    client.containers.get.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        reconcile_docker_trial(tmp_path, client)
    assert rows(journal)[-1]["event"] == "resource.cleanup_pending"


def test_live_worker_lock_prevents_even_provider_reads(tmp_path):
    client, _ = fixture(tmp_path)
    with (tmp_path / "resource-owner.lock").open("r+") as live:
        fcntl.flock(live.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            reconcile_docker_trial(tmp_path, client)
    client.info.assert_not_called()


def test_wrong_daemon_cannot_confirm_absence(tmp_path):
    client, journal = fixture(tmp_path)
    original = journal.path.read_bytes()
    client.info.return_value = {"ID": "other-daemon"}
    with pytest.raises(DockerRecoveryError, match="daemon identity mismatch"):
        reconcile_docker_trial(tmp_path, client)
    client.containers.list.assert_not_called()
    assert journal.path.read_bytes() == original


def test_legacy_claim_does_not_authorize_deletion(tmp_path):
    client, _ = fixture(tmp_path)
    (tmp_path / "resource-create-claim.json").write_text(
        json.dumps({"owner": OWNER, "provider": "docker"})
    )
    with pytest.raises(DockerRecoveryError, match="lifetime-lock custody"):
        reconcile_docker_trial(tmp_path, client)
    client.info.assert_not_called()


def test_creation_grace_cannot_be_waived(tmp_path):
    client, _ = fixture(tmp_path)
    with pytest.raises(DockerRecoveryError, match="has not expired"):
        reconcile_docker_trial(tmp_path, client, now=datetime.now(UTC) - timedelta(hours=1))
    client.containers.list.assert_not_called()
