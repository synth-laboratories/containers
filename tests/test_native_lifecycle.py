"""Recovery and failure custody for the portable native coordinator."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest

from synth_containers.lifecycle_limits import DurableRolloutSupervisor, LifecycleLimits
from synth_containers.native_lifecycle import (
    NativeLifecycleOperations,
    execute_native_lifecycle,
    recover_native_lifecycle,
)


def test_failed_work_still_confirms_cleanup_and_never_replays(tmp_path):
    calls = []

    async def setup():
        calls.append("setup")

    async def work():
        calls.append("work")
        raise RuntimeError("worker failed")

    async def later():
        pytest.fail("verifier/publication must not run after work failure")

    async def cleanup():
        calls.append("cleanup")
        return {"cleanup_status": "confirmed", "provider_id": "fake-owned-resource"}

    operations = NativeLifecycleOperations(setup, work, later, later, cleanup)
    limits = LifecycleLimits(overall_seconds=120)
    with pytest.raises(RuntimeError, match="worker failed"):
        asyncio.run(execute_native_lifecycle(tmp_path, run_id="r1", limits=limits,
                                             operations=operations))
    claim = (tmp_path / "limit-claim.json").read_bytes()
    with pytest.raises(RuntimeError, match="Completed phase cannot be repeated"):
        asyncio.run(execute_native_lifecycle(tmp_path, run_id="r1", limits=limits,
                                             operations=operations))
    assert calls == ["setup", "work", "cleanup", "cleanup"]
    assert (tmp_path / "limit-claim.json").read_bytes() == claim


def test_recovery_requires_expiry_and_only_reconciles_cleanup(tmp_path):
    limits = LifecycleLimits(overall_seconds=120)

    async def claim():
        owner = DurableRolloutSupervisor(tmp_path, "r1", limits)
        owner.close()

    asyncio.run(claim())
    calls = []

    async def cleanup():
        calls.append("cleanup")
        return {"cleanup_status": "confirmed"}

    with pytest.raises(ValueError, match="not expired"):
        asyncio.run(recover_native_lifecycle(tmp_path, cleanup=cleanup))
    assert calls == []
    path = tmp_path / "limit-claim.json"
    body = json.loads(path.read_text())
    body["created_at"] = (datetime.now(UTC) - timedelta(seconds=240)).isoformat()
    body["deadline"] = (datetime.fromisoformat(body["created_at"])
                        + timedelta(seconds=120)).isoformat()
    path.write_text(json.dumps(body))
    assert asyncio.run(recover_native_lifecycle(tmp_path, cleanup=cleanup)) == {
        "cleanup_status": "confirmed"
    }
    assert calls == ["cleanup"]


def test_owner_and_allowances_cannot_be_replaced(tmp_path):
    async def run():
        limits = LifecycleLimits(overall_seconds=120)
        first = DurableRolloutSupervisor(tmp_path, "r1", limits)
        try:
            with pytest.raises(BlockingIOError):
                DurableRolloutSupervisor(tmp_path, "r1", limits)
        finally:
            first.close()
        with pytest.raises(ValueError, match="cannot reset"):
            DurableRolloutSupervisor(tmp_path, "r2", limits)
        with pytest.raises(ValueError, match="cannot reset"):
            DurableRolloutSupervisor(tmp_path, "r1", LifecycleLimits(overall_seconds=121))

    asyncio.run(run())
