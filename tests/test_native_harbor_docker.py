"""Native Docker must retain custody when Harbor swallows cleanup failures."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("harbor")
pytest.importorskip("docker")
from docker.errors import NotFound
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from synth_containers.harbor_docker import ObservedDockerEnvironment


def environment(tmp_path, **kwargs):
    root = tmp_path / "environment"
    root.mkdir(exist_ok=True)
    return ObservedDockerEnvironment(
        environment_dir=root,
        environment_name="fixture",
        session_id="original",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(
            docker_image="sha256:" + "a" * 64,
            cpus=1,
            memory_mb=1024,
            storage_mb=1024,
            build_timeout_sec=60,
        ),
        **kwargs,
    )


def events(env):
    return [json.loads(line) for line in env._resource_journal.path.read_text().splitlines()]


def test_claim_and_intent_precede_creation_and_survive_restart(tmp_path, monkeypatch):
    env = environment(tmp_path)
    assert env.session_id == env._resource_owner

    async def start(self, force_build):
        assert (self._resource_root / "resource-create-claim.json").is_file()
        assert events(self)[-1]["event"] == "resource.create_requested"

    launch = AsyncMock(side_effect=start)
    monkeypatch.setattr(DockerEnvironment, "start", lambda self, **kw: launch(self, **kw))
    monkeypatch.setattr(env, "_discover", lambda: [{"kind": "container", "id": "owned"}])
    asyncio.run(env.start(False))
    assert events(env)[-1]["event"] == "resource.created"
    with pytest.raises(ValueError, match="one prebuilt"):
        asyncio.run(env.start(False))
    with pytest.raises(FileExistsError):
        asyncio.run(environment(tmp_path).start(False))
    assert launch.await_count == 1


def test_failed_journal_never_launches(tmp_path, monkeypatch):
    env = environment(tmp_path)
    launch = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "start", launch)
    monkeypatch.setattr(env, "_resource_event", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        asyncio.run(env.start(False))
    launch.assert_not_awaited()


@pytest.mark.parametrize("absent", [True, False])
def test_cleanup_checks_absence_after_best_effort_harbor_stop(tmp_path, monkeypatch, absent):
    env = environment(tmp_path)
    env._resource_create_attempted = True
    monkeypatch.setattr(env, "_discover", lambda: [{"kind": "container", "id": "owned"}])

    async def stop(*args, **kwargs):
        assert events(env)[-1]["handles"] == [{"kind": "container", "id": "owned"}]

    monkeypatch.setattr(DockerEnvironment, "stop", stop)
    monkeypatch.setattr(
        env, "_confirm_absence", Mock(side_effect=None if absent else RuntimeError("present"))
    )
    if absent:
        asyncio.run(env.stop(True))
        assert events(env)[-1]["event"] == "resource.cleanup_confirmed"
    else:
        with pytest.raises(RuntimeError):
            asyncio.run(env.stop(True))
        assert events(env)[-1]["event"] == "resource.cleanup_pending"
        assert env._resource_handles


def test_ambiguous_creation_without_primary_handle_stays_pending(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env._resource_create_attempted = True
    monkeypatch.setattr(env, "_discover", list)
    monkeypatch.setattr(DockerEnvironment, "stop", AsyncMock())
    with pytest.raises(RuntimeError, match="cleanup remains pending"):
        asyncio.run(env.stop(True))
    assert events(env)[-1]["event"] == "resource.cleanup_pending"


def client_fixture(monkeypatch, env, *, foreign=False):
    labels = {"com.docker.compose.project": "foreign" if foreign else env._resource_owner}
    container = SimpleNamespace(id="owned", attrs={"Config": {"Labels": labels}})
    client = SimpleNamespace(containers=Mock(), networks=Mock(), volumes=Mock(), close=Mock())
    client.containers.list.return_value = [container]
    client.networks.list.return_value = []
    client.volumes.list.return_value = []
    monkeypatch.setattr("synth_containers.harbor_docker.docker.from_env", lambda **kw: client)
    return client


def test_discovery_is_exact_and_rejects_foreign_resources(tmp_path, monkeypatch):
    env = environment(tmp_path)
    client = client_fixture(monkeypatch, env)
    assert env._discover() == [{"kind": "container", "id": "owned"}]
    client.close.assert_called_once()
    client.containers.list.assert_called_once_with(
        all=True, filters={"label": "com.docker.compose.project=" + env._resource_owner}
    )
    client_fixture(monkeypatch, env, foreign=True)
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        env._discover()


@pytest.mark.parametrize("failure", [NotFound("gone"), RuntimeError("transport")])
def test_only_typed_absence_confirms_cleanup(tmp_path, monkeypatch, failure):
    env = environment(tmp_path)
    client = client_fixture(monkeypatch, env)
    client.containers.get.side_effect = failure
    env._resource_handles = [{"kind": "container", "id": "owned"}]
    monkeypatch.setattr(env, "_discover", list)
    if isinstance(failure, NotFound):
        env._confirm_absence()
    else:
        with pytest.raises(RuntimeError):
            env._confirm_absence()


def test_cleanup_requires_deletion(tmp_path):
    with pytest.raises(ValueError, match="deletion"):
        asyncio.run(environment(tmp_path).stop(False))
