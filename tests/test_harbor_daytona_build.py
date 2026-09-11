"""No create retry and no empty-list shortcut after ambiguous image preparation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from synth_containers.harbor_daytona_build import (
    DaytonaSnapshotBuild,
    SnapshotBuildError,
    SnapshotBuildLimits,
)
from synth_containers.harbor_environment import _tree_digest
from synth_containers.harbor_image_context import HarborImageContext


class Absent(Exception):
    pass


def build(tmp_path):
    root = tmp_path / "context"
    root.mkdir()
    (root / "Dockerfile").write_text("FROM scratch\n")
    context = HarborImageContext(root, "sha256:" + "a" * 64, _tree_digest(root))
    provider = SimpleNamespace(
        prepare_context=AsyncMock(return_value={}),
        create=AsyncMock(),
        get=AsyncMock(),
        delete=AsyncMock(),
        is_not_found=lambda e: isinstance(e, Absent),
    )
    operation = DaytonaSnapshotBuild(context, tmp_path / "receipt", provider)
    provider.create.return_value = SimpleNamespace(id="owned", name=operation.owner, state="active")
    return operation, provider


def events(operation):
    return [json.loads(line) for line in operation.journal.path.read_text().splitlines()]


def test_build_claim_and_identity_are_saved_before_readiness(tmp_path):
    operation, provider = build(tmp_path)

    async def create(name, prepared):
        assert (operation.output / "build-claim.json").is_file()
        assert events(operation)[-1]["event"] == "image.create_requested"
        return SimpleNamespace(id="owned", name=name, state="active")

    provider.create.side_effect = create
    snapshot = asyncio.run(operation.build())
    assert snapshot.id == "owned"
    assert [e["event"] for e in events(operation)][-3:] == [
        "image.identity_observed",
        "image.state_observed",
        "image.build_completed",
    ]
    with pytest.raises(FileExistsError):
        asyncio.run(operation.build())
    assert provider.create.await_count == 1


def test_changed_context_prevents_provider_calls(tmp_path):
    operation, provider = build(tmp_path)
    (operation.context.root / "Dockerfile").write_text("changed")
    with pytest.raises(ValueError):
        asyncio.run(operation.build())
    provider.prepare_context.assert_not_awaited()
    provider.create.assert_not_awaited()


def test_snapshot_reference_is_never_promoted_to_verified_image_digest(tmp_path):
    operation, provider = build(tmp_path)
    provider.create.return_value.ref = "https://user:password@example.com/image:mutable"
    asyncio.run(operation.build())
    artifact = events(operation)[-1]["artifact"]
    assert artifact["snapshot_id"] == "owned"
    assert artifact["context_digest"] == operation.context.context_digest
    assert artifact["image_digest"] is None
    assert artifact["image_digest_verified"] is False
    assert artifact["native_pinned_image_launch_eligible"] is False
    assert "password" not in operation.journal.path.read_text()


def test_lost_create_response_recovers_exact_name_then_records_handle_before_delete(tmp_path):
    operation, provider = build(tmp_path)
    provider.create.side_effect = TimeoutError()
    snapshot = SimpleNamespace(id="owned", name=operation.owner, state="building")
    provider.get.side_effect = [snapshot, snapshot, Absent(), Absent()]

    async def delete(identifier):
        assert any(e["event"] == "image.identity_observed" for e in events(operation))
        assert identifier == "owned"

    provider.delete.side_effect = delete
    with pytest.raises(TimeoutError):
        asyncio.run(operation.build())
    assert events(operation)[-1]["event"] == "image.cleanup_confirmed"
    assert provider.create.await_count == 1
    provider.delete.assert_awaited_once()


def test_ambiguous_create_without_observed_identity_remains_pending(tmp_path):
    operation, provider = build(tmp_path)
    provider.create.side_effect = TimeoutError()
    provider.get.side_effect = Absent()
    with pytest.raises(TimeoutError):
        asyncio.run(operation.build())
    assert events(operation)[-1]["event"] == "image.cleanup_pending"
    provider.delete.assert_not_awaited()


def test_foreign_snapshot_is_never_deleted(tmp_path):
    operation, provider = build(tmp_path)
    provider.create.side_effect = TimeoutError()
    provider.get.return_value = SimpleNamespace(id="foreign", name="foreign", state="active")
    with pytest.raises(TimeoutError):
        asyncio.run(operation.build())
    assert events(operation)[-1]["event"] == "image.cleanup_pending"
    provider.delete.assert_not_awaited()


@pytest.mark.parametrize("value", [0, True, 1.5, 1801])
def test_build_deadline_must_be_explicitly_bounded(value):
    with pytest.raises(ValueError):
        SnapshotBuildLimits(build_seconds=value)


def test_recovery_preserves_completed_artifact_until_explicit_release(tmp_path):
    from datetime import UTC, datetime, timedelta

    from synth_containers.harbor_daytona_build import recover_daytona_snapshot_build

    operation, provider = build(tmp_path)
    asyncio.run(operation.build())
    with pytest.raises(SnapshotBuildError, match="not been released"):
        asyncio.run(
            recover_daytona_snapshot_build(
                operation.output, provider, now=datetime.now(UTC) + timedelta(hours=2)
            )
        )
    provider.delete.assert_not_awaited()


def test_recovery_refuses_unexpired_build(tmp_path):
    from synth_containers.harbor_daytona_build import recover_daytona_snapshot_build

    operation, provider = build(tmp_path)
    asyncio.run(operation.build())
    with pytest.raises(SnapshotBuildError, match="not expired"):
        asyncio.run(recover_daytona_snapshot_build(operation.output, provider))
    provider.delete.assert_not_awaited()


def test_recovery_uses_saved_identity_without_another_create(tmp_path):
    from datetime import UTC, datetime, timedelta

    from synth_containers.harbor_daytona_build import recover_daytona_snapshot_build

    operation, provider = build(tmp_path)
    snapshot = asyncio.run(operation.build())
    operation.event("image.cleanup_requested", snapshot_id=snapshot.id)
    provider.get.side_effect = [snapshot, Absent(), Absent()]
    result = asyncio.run(
        recover_daytona_snapshot_build(
            operation.output, provider, now=datetime.now(UTC) + timedelta(hours=2)
        )
    )
    assert result["event"] == "image.cleanup_confirmed"
    provider.create.assert_awaited_once()
    provider.delete.assert_awaited_once_with(snapshot.id)


def test_create_deadline_is_enforced_without_repeating_request(tmp_path):
    operation, provider = build(tmp_path)
    operation.limits = SnapshotBuildLimits(create_seconds=1)

    async def never_returns(*args):
        await asyncio.sleep(30)

    provider.create.side_effect = never_returns
    provider.get.side_effect = Absent()
    with pytest.raises(TimeoutError):
        asyncio.run(operation.build())
    provider.create.assert_awaited_once()
    assert events(operation)[-1]["event"] == "image.cleanup_pending"


def test_expired_recovery_cannot_take_a_live_builder_lock(tmp_path):
    import fcntl
    from datetime import UTC, datetime, timedelta

    from synth_containers.harbor_daytona_build import recover_daytona_snapshot_build

    operation, provider = build(tmp_path)
    asyncio.run(operation.build())
    operation.event("image.cleanup_requested", snapshot_id="owned")
    with (operation.output / "build-owner.lock").open("r+") as live:
        fcntl.flock(live.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            asyncio.run(
                recover_daytona_snapshot_build(
                    operation.output, provider, now=datetime.now(UTC) + timedelta(hours=2)
                )
            )
    provider.delete.assert_not_awaited()
