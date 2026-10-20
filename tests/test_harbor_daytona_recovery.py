"""Expired-trial recovery preserves ownership and ambiguous resource liability."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("daytona")
from daytona.common.errors import DaytonaNotFoundError

from synth_containers.harbor_daytona_recovery import HarborRecoveryError, reconcile_daytona_trial
from synth_containers.operator_journal import OperatorJournal, read_operator_events

OWNER = "synth-harbor-" + "a" * 32
NOW = datetime(2026, 9, 11, 12, tzinfo=UTC)


def trial(root, *, known=True, age_seconds=1000):
    (root / "resource-create-claim.json").write_text(json.dumps({"owner": OWNER}))
    journal = OperatorJournal(root / "resource-events.jsonl", run_id=OWNER)
    journal.append(
        lambda seq: {
            "run_id": OWNER,
            "seq": seq,
            "event": "resource.create_requested",
            "occurred_at": (NOW - timedelta(seconds=age_seconds)).isoformat(),
            "provider": "daytona",
            "ttl_minutes": 5,
        }
    )
    if known:
        journal.append(
            lambda seq: {
                "run_id": OWNER,
                "seq": seq,
                "event": "resource.created",
                "provider_id": "owned",
            }
        )
    return root


class Client:
    def __init__(self, listings=(), gets=()):
        self.listings = iter(listings)
        self.get = AsyncMock(side_effect=list(gets))
        self.delete = AsyncMock()
        self.list_calls = 0

    async def list(self, query, request_timeout):
        assert query.labels == {"ai.synth.harbor.owner": OWNER}
        assert request_timeout == 10
        self.list_calls += 1
        for sandbox in next(self.listings):
            yield sandbox


def sandbox(owner=OWNER, identifier="owned"):
    return SimpleNamespace(id=identifier, labels={"ai.synth.harbor.owner": owner})


def test_active_trial_refused_without_provider_call(tmp_path):
    client = Client()
    with pytest.raises(HarborRecoveryError, match="not expired"):
        asyncio.run(reconcile_daytona_trial(trial(tmp_path, age_seconds=659), client, now=NOW))
    assert client.list_calls == 0
    client.delete.assert_not_awaited()


def test_expired_trial_deletes_owned_resource_and_confirms_absence(tmp_path):
    resource = sandbox()
    client = Client([[resource], []], [resource, DaytonaNotFoundError("gone")])
    result = asyncio.run(reconcile_daytona_trial(trial(tmp_path), client, now=NOW))
    assert result["event"] == "resource.cleanup_confirmed"
    assert result["source"] == "recovery"
    client.delete.assert_awaited_once_with(resource, timeout=20, wait=True)


def test_ambiguous_create_is_discovered_by_exact_owner(tmp_path):
    resource = sandbox()
    client = Client([[resource], []], [resource, DaytonaNotFoundError("gone")])
    asyncio.run(reconcile_daytona_trial(trial(tmp_path, known=False), client, now=NOW))
    events = read_operator_events(tmp_path / "resource-events.jsonl")["events"]
    assert any(row.get("provider_id") == "owned" for row in events)
    assert events[-1]["event"] == "resource.cleanup_confirmed"


def test_empty_listing_does_not_settle_unknown_creation(tmp_path):
    client = Client([[]])
    with pytest.raises(HarborRecoveryError, match="identity remains unconfirmed"):
        asyncio.run(reconcile_daytona_trial(trial(tmp_path, known=False), client, now=NOW))
    client.delete.assert_not_awaited()
    assert (
        read_operator_events(tmp_path / "resource-events.jsonl")["events"][-1]["event"]
        == "resource.cleanup_pending"
    )


def test_known_absence_is_repeatable_without_delete(tmp_path):
    client = Client([[], [], [], []], [DaytonaNotFoundError("gone")] * 4)
    trial(tmp_path)
    for _ in range(2):
        result = asyncio.run(reconcile_daytona_trial(tmp_path, client, now=NOW))
        assert result["event"] == "resource.cleanup_confirmed"
    client.delete.assert_not_awaited()


@pytest.mark.parametrize(
    "failure",
    ["foreign_list", "foreign_get", "multiple", "still_present", "stale_list", "provider_error"],
)
def test_uncertain_cleanup_never_claims_absence(tmp_path, failure):
    resource = sandbox()
    listed = [resource]
    gets = [resource, DaytonaNotFoundError("gone")]
    remaining = []
    if failure == "foreign_list":
        listed = [sandbox("foreign")]
    elif failure == "foreign_get":
        gets[0] = sandbox("foreign")
    elif failure == "multiple":
        listed.append(sandbox(identifier="second"))
    elif failure == "still_present":
        gets[1] = resource
    elif failure == "stale_list":
        remaining = [resource]
    elif failure == "provider_error":
        gets[0] = RuntimeError("private provider details")
    client = Client([listed, remaining], gets)
    with pytest.raises((HarborRecoveryError, RuntimeError)):
        asyncio.run(reconcile_daytona_trial(trial(tmp_path), client, now=NOW))
    rows = read_operator_events(tmp_path / "resource-events.jsonl")["events"]
    assert rows[-1]["event"] == "resource.cleanup_pending"
    assert not any(row["event"] == "resource.cleanup_confirmed" for row in rows)
    assert "private provider details" not in (tmp_path / "resource-events.jsonl").read_text()
    if failure in {"foreign_list", "foreign_get", "multiple", "provider_error"}:
        client.delete.assert_not_awaited()
