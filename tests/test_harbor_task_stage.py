"""Prebuilt migration freezes task authority before native Harbor can launch."""

import json
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("toml")
pytest.importorskip("harbor")

from synth_containers.harbor_environment import (
    HarborEnvironmentError,
    HarborProviderCompatibility,
    inspect_harbor_package,
    register_harbor_environment,
)
from synth_containers.harbor_task_stage import stage_native_harbor_task

IMAGE = "registry.example/task@sha256:" + "a" * 64


def release(root: Path, *, provider="daytona"):
    root.mkdir()
    (root / "environment").mkdir()
    (root / "tests").mkdir()
    (root / "instruction.md").write_text("Make the verifier pass.\n")
    (root / "environment/Dockerfile").write_text("FROM scratch\n")
    (root / "tests/test.sh").write_text("exit 0\n")
    (root / "task.toml").write_text("""schema_version = "1.3"
[task]
name = "synth/fixture"
[agent]
timeout_sec = 90
network_mode = "no-network"
[verifier]
timeout_sec = 30
network_mode = "no-network"
environment_mode = "shared"
[environment]
cpus = 1
memory_mb = 1024
storage_mb = 1024
build_timeout_sec = 900
""")
    draft = inspect_harbor_package(root)
    return register_harbor_environment(
        draft,
        agent_image=IMAGE,
        verifier_image=IMAGE,
        provider=HarborProviderCompatibility(
            provider_id=provider, supports_separate_verifier=False
        ),
    )


@pytest.mark.parametrize("provider", ["docker", "daytona"])
def test_stage_changes_only_image_preparation_and_preserves_source(tmp_path, provider):
    bound = release(tmp_path / "source", provider=provider)
    original = (bound.draft.root / "task.toml").read_bytes()
    receipt = stage_native_harbor_task(bound, tmp_path / "staged", creation_timeout_seconds=60)
    task = Path(receipt["task_path"])
    assert (bound.draft.root / "task.toml").read_bytes() == original
    assert (bound.draft.root / "environment/Dockerfile").exists()
    assert not (task / "environment/Dockerfile").exists()
    assert (task / "tests/test.sh").read_bytes() == (
        bound.draft.root / "tests/test.sh"
    ).read_bytes()
    before = tomllib.loads(original.decode())
    after = tomllib.loads((task / "task.toml").read_text())
    before["environment"].update(docker_image=IMAGE, build_timeout_sec=60)
    assert before == after
    assert receipt["source_creation_timeout_seconds"] == 900
    assert receipt["image_build_provenance"] == "operator_bound_not_verified"
    assert json.loads((tmp_path / "staged/stage-receipt.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        stage_native_harbor_task(bound, tmp_path / "staged")


def test_changed_source_refused_before_stage_is_created(tmp_path):
    bound = release(tmp_path / "source")
    (bound.draft.root / "tests/test.sh").write_text("exit 1\n")
    with pytest.raises(HarborEnvironmentError, match="stale"):
        stage_native_harbor_task(bound, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()


def test_stage_cannot_copy_into_its_source_tree(tmp_path):
    bound = release(tmp_path / "source")
    with pytest.raises(HarborEnvironmentError, match="mutate_source"):
        stage_native_harbor_task(bound, bound.draft.root / "staged")
    assert not (bound.draft.root / "staged").exists()


def test_compose_refused_before_any_stage_files(tmp_path):
    bound = release(tmp_path / "source")
    (bound.draft.root / "environment/compose.yaml").write_text("services: {}\n")
    bound = register_harbor_environment(
        inspect_harbor_package(bound.draft.root),
        agent_image=IMAGE,
        verifier_image=IMAGE,
        provider=bound.provider,
    )
    with pytest.raises(HarborEnvironmentError, match="compose_unqualified"):
        stage_native_harbor_task(bound, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()


def test_separate_verifier_is_not_silently_converted_to_shared(tmp_path):
    from dataclasses import replace

    bound = release(tmp_path / "source")
    changed = replace(bound, draft=replace(bound.draft, verifier_environment_mode="separate"))
    with pytest.raises(HarborEnvironmentError, match="separate_verifier_unqualified"):
        stage_native_harbor_task(changed, tmp_path / "staged")


def test_image_mismatch_is_refused(tmp_path):
    from dataclasses import replace

    bound = replace(release(tmp_path / "source"), verifier_image="other@sha256:" + "b" * 64)
    with pytest.raises(HarborEnvironmentError, match="image_mismatch"):
        stage_native_harbor_task(bound, tmp_path / "staged")


@pytest.mark.parametrize("timeout", [0, True, 300.5, 301])
def test_creation_allowance_is_finite_before_staging(tmp_path, timeout):
    bound = release(tmp_path / "source")
    with pytest.raises(HarborEnvironmentError, match="allowance_invalid"):
        stage_native_harbor_task(bound, tmp_path / "staged", creation_timeout_seconds=timeout)
    assert not (tmp_path / "staged").exists()


def test_source_change_during_copy_leaves_no_launch_receipt(tmp_path, monkeypatch):
    from synth_containers import harbor_task_stage as subject

    bound = release(tmp_path / "source")
    copy = subject._copy_bounded_file

    def changed(source, destination, remaining_bytes):
        result = copy(source, destination, remaining_bytes)
        if source.name == "test.sh":
            destination.write_text("corrupted\n")
        return result

    monkeypatch.setattr(subject, "_copy_bounded_file", changed)
    with pytest.raises(HarborEnvironmentError, match="changed_during_stage"):
        stage_native_harbor_task(bound, tmp_path / "staged")
    assert not (tmp_path / "staged/stage-receipt.json").exists()


def test_copy_rechecks_resource_bound_after_inspection(tmp_path, monkeypatch):
    from synth_containers import harbor_task_stage as subject

    bound = release(tmp_path / "source")
    monkeypatch.setattr(subject, "_MAX_TREE_BYTES", 2)
    with pytest.raises(HarborEnvironmentError, match="copy_limit"):
        stage_native_harbor_task(bound, tmp_path / "staged")
    assert not (tmp_path / "staged/stage-receipt.json").exists()


def test_cli_stages_without_provider_access(tmp_path, capsys):
    from synth_containers.cli import main

    bound = release(tmp_path / "source")
    assert (
        main(
            [
                "harbor-stage",
                str(bound.draft.root),
                str(tmp_path / "staged"),
                "--image",
                IMAGE,
                "--provider",
                "daytona",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["environment_release_digest"] == bound.release_digest
    assert Path(receipt["task_path"]).is_dir()
