"""Launcher contract: docker-only up/down, run records, sibling reaping."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from synth_containers.launch import (
    HTTP_TASK,
    PARENT_LABEL,
    ImageSpec,
    LaunchError,
    down_image,
    list_run_records,
    load_catalog,
    read_run_record,
    resolve_local_digest,
    up_image,
)

DIGEST = "sha256:" + ("ab" * 32)


def _write_catalog(
    root: Path,
    *,
    contract: str = HTTP_TASK,
    extra: str = "",
    image_id: str = "banking77",
) -> None:
    image_dir = root / image_id
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (image_dir / "image.toml").write_text(
        "\n".join(
            [
                f'contract = "{contract}"',
                'target_id = "banking77_classify"',
                f'image_name = "evals-{image_id}"',
                "port = 8080",
                extra,
            ]
        ),
        encoding="utf-8",
    )
    (root / "catalog.toml").write_text(
        f'[[image]]\nid = "{image_id}"\npath = "{image_id}"\n',
        encoding="utf-8",
    )


class FakeDocker:
    """In-memory daemon: named containers, labels, and image references."""

    def __init__(self, digest: str = DIGEST, *, healthy: bool = True) -> None:
        self.digest = digest
        self.built: list[str] = []
        self.ran: list[tuple[str, tuple[str, ...]]] = []
        self.images: set[str] = {digest, "evals-banking77:local"}
        self.containers: dict[str, tuple[str, ...]] = {}
        self.removed: list[str] = []
        self.healthy = healthy

    def inspect_id(self, reference: str) -> str | None:
        if reference in self.images or reference.endswith(self.digest):
            return self.digest
        return None

    def build(self, spec: ImageSpec, *, tag: str) -> str:
        self.built.append(tag)
        self.images.add(tag)
        self.images.add(spec.pinned_name(self.digest))
        return self.digest

    def pull(self, reference: str) -> str:
        raise LaunchError(f"unexpected_pull:{reference}")

    def run(self, *, reference: str, name: str, args: Sequence[str]) -> str:
        self.ran.append((reference, tuple(args)))
        self.containers[name] = tuple(args)
        return f"cid-{name}"

    def stop(self, name: str) -> None:
        self.containers.pop(name, None)
        self.removed.append(name)

    def logs(self, name: str, *, tail: int = 200) -> str:
        del tail
        return f"logs:{name}"

    def container_ids(self, *, label: str) -> tuple[str, ...]:
        return tuple(name for name, args in self.containers.items() if label in args)

    def remove(self, references: Sequence[str]) -> None:
        for reference in references:
            self.containers.pop(reference, None)
            self.removed.append(reference)

    def exists(self, name: str) -> bool:
        return name in self.containers

    def is_running(self, name: str) -> bool:
        return self.exists(name)

    def spawn_child(self, name: str, parent: str) -> None:
        self.containers[name] = ("--label", f"{PARENT_LABEL}={parent}")


@pytest.fixture(autouse=True)
def _isolated_run_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_CONTAINERS_RUN_ROOT", str(tmp_path / "run"))


@pytest.fixture
def healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "synth_containers.sdk.ContainerHandle.health",
        lambda self, *, timeout_seconds=5.0: {"status": "ok"},
    )


def test_load_catalog_http_task(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    spec = load_catalog(tmp_path)["banking77"]
    assert spec.contract == HTTP_TASK
    assert spec.target_id == "banking77_classify"
    assert spec.image_name == "evals-banking77"
    assert spec.pull is False
    assert spec.nested is None


def test_host_command_is_forbidden(tmp_path: Path) -> None:
    _write_catalog(tmp_path, extra='command = ["python", "-m", "banking77_classify"]')
    with pytest.raises(LaunchError, match="command_forbidden"):
        load_catalog(tmp_path)


def test_latest_tag_is_refused(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    spec = load_catalog(tmp_path)["banking77"]
    with pytest.raises(LaunchError, match="latest_forbidden"):
        resolve_local_digest(spec, image="evals-banking77:latest", build=False, backend=FakeDocker())


def test_resolve_builds_when_missing(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    spec = load_catalog(tmp_path)["banking77"]
    backend = FakeDocker()
    backend.images.clear()
    assert resolve_local_digest(spec, build=True, backend=backend) == DIGEST
    assert backend.built == ["evals-banking77:local"]


def test_harbor_image_cannot_http_up(tmp_path: Path) -> None:
    _write_catalog(tmp_path, contract="harbor_environment")
    with pytest.raises(LaunchError, match="not_http_task"):
        up_image("banking77", catalog=tmp_path, backend=FakeDocker())


def test_rollout_environment_is_never_upped(tmp_path: Path) -> None:
    _write_catalog(tmp_path, contract="rollout_environment")
    with pytest.raises(LaunchError, match="rollout_not_up"):
        up_image("banking77", catalog=tmp_path, backend=FakeDocker())


def test_up_writes_run_record_and_down_clears_it(tmp_path: Path, healthy: None) -> None:
    _write_catalog(tmp_path)
    backend = FakeDocker()
    record = up_image("banking77", catalog=tmp_path, backend=backend, port=8123)
    assert record.container_name == "synth-banking77-8123"
    assert record.url == "http://127.0.0.1:8123"
    assert read_run_record("banking77", 8123) is not None
    assert down_image("banking77", port=8123, catalog=tmp_path, backend=backend) is True
    assert read_run_record("banking77", 8123) is None
    assert backend.containers == {}


def test_second_up_on_same_pair_refuses_without_replace(tmp_path: Path, healthy: None) -> None:
    _write_catalog(tmp_path)
    backend = FakeDocker()
    up_image("banking77", catalog=tmp_path, backend=backend, port=8124)
    with pytest.raises(LaunchError, match="already_up"):
        up_image("banking77", catalog=tmp_path, backend=backend, port=8124)
    replaced = up_image("banking77", catalog=tmp_path, backend=backend, port=8124, replace=True)
    assert replaced.container_name == "synth-banking77-8124"
    assert len(list_run_records("banking77")) == 1


def test_up_recovers_a_stopped_record_without_replace(tmp_path: Path, healthy: None) -> None:
    _write_catalog(tmp_path)

    class StoppedDocker(FakeDocker):
        stopped_name: str | None = None

        def is_running(self, name: str) -> bool:
            return self.exists(name) and name != self.stopped_name

    backend = StoppedDocker()
    first = up_image("banking77", catalog=tmp_path, backend=backend, port=8127)
    backend.stopped_name = first.container_name

    restarted = up_image("banking77", catalog=tmp_path, backend=backend, port=8127)

    assert restarted.container_name == first.container_name
    assert first.container_name in backend.removed


def test_down_reaps_labelled_siblings_first(tmp_path: Path, healthy: None) -> None:
    _write_catalog(tmp_path)
    backend = FakeDocker()
    up_image("banking77", catalog=tmp_path, backend=backend, port=8125)
    backend.spawn_child("trial-agent", "banking77")
    backend.spawn_child("trial-verifier", "banking77")
    down_image("banking77", port=8125, catalog=tmp_path, backend=backend)
    assert "trial-agent" in backend.removed
    assert "trial-verifier" in backend.removed
    assert backend.containers == {}


def test_down_with_no_record_is_a_noop_success(tmp_path: Path) -> None:
    _write_catalog(tmp_path)
    assert down_image("banking77", port=8126, catalog=tmp_path, backend=FakeDocker()) is False


def test_unhealthy_up_leaves_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_catalog(tmp_path)
    backend = FakeDocker()

    def _fail(self, *, timeout_seconds: float = 5.0) -> dict[str, str]:
        raise RuntimeError("nope")

    monkeypatch.setattr("synth_containers.sdk.ContainerHandle.health", _fail)
    with pytest.raises(LaunchError, match="unhealthy"):
        up_image(
            "banking77",
            catalog=tmp_path,
            backend=backend,
            port=8127,
            startup_timeout_seconds=0.5,
        )
    assert backend.containers == {}
    assert read_run_record("banking77", 8127) is None


def test_nested_platform_mounts_socket_and_host_workspace(
    tmp_path: Path, healthy: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTH_CONTAINERS_WORKSPACE_ROOT", str(tmp_path / "work"))
    _write_catalog(tmp_path, extra='nested = "host_docker"')
    backend = FakeDocker()
    record = up_image("banking77", catalog=tmp_path, backend=backend, port=8128)
    args = backend.ran[-1][1]
    assert any(arg.endswith("docker.sock") for arg in args)
    assert record.workspace_host_root == str(tmp_path / "work" / "banking77")
    assert f"SYNTH_WORKSPACE_HOST_ROOT={tmp_path / 'work' / 'banking77'}" in args
    assert "SYNTH_WORKSPACE_ROOT=/work" in args


def test_socket_mount_without_nested_is_refused(tmp_path: Path) -> None:
    _write_catalog(tmp_path, extra="mount_docker_socket = true")
    with pytest.raises(LaunchError, match="nested_required"):
        load_catalog(tmp_path)


def test_evals_images_catalog_loads() -> None:
    root = Path("/Users/joshuapurtell/GitHub/evals/containers/images")
    if not (root / "catalog.toml").is_file():
        pytest.skip("evals checkout images catalog is not present")
    specs = load_catalog(root)
    assert set(specs) >= {
        "banking77",
        "healthbench2",
        "craftax-gamebench-rust",
        "rogue-gold",
        "alfworld",
        "harvey-lab",
    }
    assert all(spec.pull is False for spec in specs.values())
    for spec in specs.values():
        assert spec.dockerfile.name != "http_task.Dockerfile" or spec.contract != HTTP_TASK, (
            f"{spec.id} still uses the generic serve stub"
        )


def test_declared_volume_is_mounted_read_only(tmp_path: Path, healthy: None) -> None:
    secrets = tmp_path / "codex-home"
    secrets.mkdir()
    _write_catalog(
        tmp_path,
        extra=f'[[volumes]]\nsource = "{secrets}"\ntarget = "/root/.codex"\nread_only = true',
    )
    backend = FakeDocker()
    up_image("banking77", catalog=tmp_path, backend=backend, port=8129)
    args = backend.ran[-1][1]
    assert f"{secrets}:/root/.codex:ro" in args


def test_missing_volume_source_is_refused(tmp_path: Path) -> None:
    absent = tmp_path / "never-created"
    _write_catalog(
        tmp_path, extra=f'[[volumes]]\nsource = "{absent}"\ntarget = "/root/.codex"'
    )
    # Docker would create the source as an empty directory and the image would
    # come up silently unauthenticated. Refuse instead. The catalog itself still
    # loads: reading it must not depend on the reader's filesystem, only `up`
    # has to be fail-closed.
    load_catalog(tmp_path)
    with pytest.raises(LaunchError, match="volume_source_missing"):
        up_image("banking77", catalog=tmp_path, backend=FakeDocker(), port=8130)


def test_unset_volume_variable_is_refused(tmp_path: Path, monkeypatch) -> None:
    _write_catalog(
        tmp_path,
        extra='[[volumes]]\nsource = "$SYNTH_TEST_CORPUS"\ntarget = "/corpus"',
    )
    monkeypatch.delenv("SYNTH_TEST_CORPUS", raising=False)
    # Unexpanded, `$SYNTH_TEST_CORPUS` is a relative path the daemon would
    # happily create as an empty directory. Name the variable instead.
    with pytest.raises(LaunchError, match="volume_source_unset"):
        up_image("banking77", catalog=tmp_path, backend=FakeDocker(), port=8131)


def test_volume_variable_expands(tmp_path: Path, monkeypatch, healthy: None) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_catalog(
        tmp_path,
        extra='[[volumes]]\nsource = "$SYNTH_TEST_CORPUS"\ntarget = "/corpus"\nread_only = true',
    )
    monkeypatch.setenv("SYNTH_TEST_CORPUS", str(corpus))
    backend = FakeDocker()
    up_image("banking77", catalog=tmp_path, backend=backend, port=8132)
    assert f"{corpus}:/corpus:ro" in backend.ran[-1][1]
