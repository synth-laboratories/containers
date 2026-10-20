"""Native provider limits use real Harbor/Daytona types without paid execution."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
pytest.importorskip("daytona")
from daytona import CreateSandboxFromImageParams, Image, Resources
from daytona.common.errors import DaytonaNotFoundError
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from synth_containers.harbor_daytona import BoundedDaytonaEnvironment

IMAGE = "python@sha256:" + "a" * 64


def environment(tmp_path, **kwargs):
    root = tmp_path / "environment"
    root.mkdir(exist_ok=True)
    return BoundedDaytonaEnvironment(
        environment_dir=root,
        environment_name="fixture",
        session_id="session",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(
            docker_image=IMAGE, cpus=1, memory_mb=1024, storage_mb=1024, build_timeout_sec=60
        ),
        **kwargs,
    )


def parameters(**resources):
    return CreateSandboxFromImageParams(
        image=Image.base(IMAGE), resources=Resources(cpu=1, memory=1, disk=1, **resources)
    )


def events(env):
    return [json.loads(line) for line in env._resource_journal.path.read_text().splitlines()]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "5", 361])
def test_invalid_ttl_refused_before_provider(tmp_path, value):
    with pytest.raises(ValueError, match="resource_ttl_minutes"):
        environment(tmp_path, resource_ttl_minutes=value)


def test_creation_claim_is_durable_and_cannot_be_retried(tmp_path):
    env = environment(tmp_path)
    client = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                id="owned",
                cpu=1,
                memory=1,
                disk=1,
                auto_destroy_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
            )
        )
    )
    params = parameters()
    asyncio.run(env._create_sandbox(params, client))
    assert params.ttl_minutes == 5
    assert params.auto_delete_interval == 0
    assert params.labels["ai.synth.harbor.owner"] == env._resource_owner
    assert [item["event"] for item in events(env)] == [
        "resource.create_requested",
        "resource.created",
        "resource.allocation_observed",
    ]
    with pytest.raises(RuntimeError, match="allowance exhausted"):
        asyncio.run(env._create_sandbox(parameters(), client))
    restarted = environment(tmp_path)
    with pytest.raises(FileExistsError):
        asyncio.run(restarted._create_sandbox(parameters(), client))
    client.create.assert_awaited_once()


def test_ambiguous_create_is_not_retried(tmp_path):
    env = environment(tmp_path)
    client = SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("private provider text")))
    with pytest.raises(RuntimeError):
        asyncio.run(env._create_sandbox(parameters(), client))
    assert events(env)[-1]["event"] == "resource.create_unconfirmed"
    assert "private provider text" not in env._resource_journal.path.read_text()
    client.create.assert_awaited_once()


@pytest.mark.parametrize("outcome", ["absent", "present", "delete_failed", "lookup_failed"])
def test_cleanup_requires_typed_absence_and_retains_handle(tmp_path, outcome):
    env = environment(tmp_path)
    resource = SimpleNamespace(id="owned")
    env._sandbox = resource
    client = SimpleNamespace(delete=AsyncMock(), get=AsyncMock())
    env._client_manager = SimpleNamespace(get_client=AsyncMock(return_value=client))
    if outcome == "absent":
        client.get.side_effect = DaytonaNotFoundError("gone")
        asyncio.run(env.stop(delete=True))
        assert env._sandbox is None
        assert events(env)[-1]["event"] == "resource.cleanup_confirmed"
    else:
        if outcome == "delete_failed":
            client.delete.side_effect = RuntimeError("failed")
        if outcome == "lookup_failed":
            client.get.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError):
            asyncio.run(env.stop(delete=True))
        assert env._sandbox is resource
        assert events(env)[-1]["event"] == "resource.cleanup_pending"


def test_oversized_resources_refused_without_claim_or_provider_call(tmp_path):
    env = environment(tmp_path)
    params = parameters()
    params.resources.cpu = 2
    client = SimpleNamespace(create=AsyncMock())
    with pytest.raises(ValueError, match="admitted bounds"):
        asyncio.run(env._create_sandbox(params, client))
    assert not env._resource_journal.path.exists()
    client.create.assert_not_awaited()


@pytest.mark.parametrize(
    "observed", [None, "2099-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", "not-a-time"]
)
def test_unconfirmed_or_excessive_provider_ttl_retains_returned_handle(tmp_path, observed):
    env = environment(tmp_path)
    resource = SimpleNamespace(id="owned", cpu=1, memory=1, disk=1, auto_destroy_at=observed)
    client = SimpleNamespace(create=AsyncMock(return_value=resource))
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        asyncio.run(env._create_sandbox(parameters(), client))
    assert env._sandbox is resource
    assert any(row.get("provider_id") == "owned" for row in events(env))
    client.create.assert_awaited_once()


@pytest.mark.parametrize(
    "kwargs", [{"auto_snapshot": True}, {"snapshot_template_name": "unbounded"}]
)
def test_snapshot_creation_and_selection_are_not_silent_fallbacks(tmp_path, kwargs):
    with pytest.raises(ValueError):
        environment(tmp_path, **kwargs)
