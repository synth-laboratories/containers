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


def test_stage_receipt_freezes_native_daytona_limits(tmp_path):
    bound = release(tmp_path / 'source')
    receipt = stage_native_harbor_task(bound, tmp_path / 'staged', resource_ttl_minutes=12)
    flags = receipt['native_environment_flags']
    assert 'resource_ttl_minutes=12' in flags
    assert 'maximum_cpu=1' in flags
    assert 'auto_snapshot=false' in flags


@pytest.mark.parametrize('memory,expected', [(1024, 1), (1025, 2)])
def test_shared_native_resource_rounding_is_explicit(tmp_path, memory, expected):
    from synth_containers.harbor_task_stage import native_harbor_environment_flags
    (tmp_path / 'task.toml').write_text(f'[environment]\ncpus=1\nmemory_mb={memory}\nstorage_mb=1024\n')
    flags = native_harbor_environment_flags(tmp_path, provider='daytona', image=IMAGE, resource_ttl_minutes=20)
    assert f'maximum_memory_gib={expected}' in flags


def test_shared_native_resource_policy_refuses_missing_or_excessive_limits(tmp_path):
    from synth_containers.harbor_task_stage import native_harbor_environment_flags
    for text in ('', '[environment]\ncpus=65\nmemory_mb=1024\nstorage_mb=1024\n'):
        (tmp_path / 'task.toml').write_text(text)
        with pytest.raises(HarborEnvironmentError):
            native_harbor_environment_flags(tmp_path, provider='daytona', image=IMAGE, resource_ttl_minutes=20)


def test_explicit_resources_resolve_missing_declarations_without_rewriting_source(tmp_path):
    from synth_containers.harbor_environment import HarborResourceRequest
    source = tmp_path / 'source'
    bound = release(source)
    path = source / 'task.toml'
    path.write_text(path.read_text().replace('cpus = 1\n', '').replace('memory_mb = 1024\n', '').replace('storage_mb = 1024\n', ''))
    original = path.read_bytes()
    bound = register_harbor_environment(inspect_harbor_package(source), agent_image=IMAGE, verifier_image=IMAGE, provider=bound.provider)
    receipt = stage_native_harbor_task(bound, tmp_path / 'staged', resource_request=HarborResourceRequest(cpus=2, memory_mb=2048, storage_mb=4096, gpus=0))
    assert path.read_bytes() == original
    assert receipt['source_resource_request']['cpus'] is None
    assert receipt['resolved_resource_request']=={'cpus':2,'memory_mb':2048,'storage_mb':4096,'gpus':0}
    assert 'maximum_cpu=2' in receipt['native_environment_flags']
    staged = tomllib.loads((Path(receipt['task_path'])/'task.toml').read_text())
    assert staged['environment']['memory_mb']==2048


@pytest.mark.parametrize('cpu,memory,storage', [(None,1024,1024),(True,1024,1024),(65,1024,1024),(1,0,1024),(1,1024,1024*1024+1)])
def test_partial_or_unbounded_resource_overrides_fail_before_materialization(tmp_path,cpu,memory,storage):
    from synth_containers.harbor_environment import HarborResourceRequest
    with pytest.raises(HarborEnvironmentError, match='bounded positive'):
        stage_native_harbor_task(release(tmp_path/'source'),tmp_path/'staged', resource_request=HarborResourceRequest(cpus=cpu,memory_mb=memory,storage_mb=storage,gpus=0))
    assert not (tmp_path/'staged').exists()


def test_stage_records_local_docker_image_scope(tmp_path):
    bound = release(tmp_path / "source", provider="docker")
    image = "sha256:" + "c" * 64
    bound = register_harbor_environment(
        bound.draft, agent_image=image, verifier_image=image, provider=bound.provider
    )
    receipt = stage_native_harbor_task(bound, tmp_path / "staged")
    assert receipt["image_reference_scope"] == "docker_local_image_id"
    assert receipt["native_environment_flags"] == ["--env", "docker"]
    assert receipt["image_build_provenance"] == "operator_bound_not_verified"


def test_native_docker_custody_flags_are_frozen_in_stage(tmp_path):
    bound = release(tmp_path / "source", provider="docker")
    receipt = stage_native_harbor_task(bound, tmp_path / "stage", docker_resource_custody=True,
                                       docker_egress_image=IMAGE)
    assert receipt["docker_resource_custody"] is True
    assert receipt["native_environment_flags"] == [
        "--env", "synth_containers.harbor_docker:ObservedDockerEnvironment",
        "--ek", "egress_control_image=" + IMAGE]


@pytest.mark.parametrize("options", [dict(docker_resource_custody="true"),
                                     dict(docker_egress_image=IMAGE),
                                     dict(docker_resource_custody=True, docker_egress_image="latest")])
def test_invalid_native_docker_custody_options_refused_before_staging(tmp_path, options):
    bound = release(tmp_path / "source", provider="docker")
    with pytest.raises(HarborEnvironmentError):
        stage_native_harbor_task(bound, tmp_path / "stage", **options)
    assert not (tmp_path / "stage").exists()


def test_daytona_refuses_docker_custody_options(tmp_path):
    bound = release(tmp_path / "source")
    with pytest.raises(HarborEnvironmentError, match="require Docker"):
        stage_native_harbor_task(bound, tmp_path / "stage", docker_resource_custody=True)
