"""Static Harbor package inspection and immutable release admission."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_containers.harbor_environment import (
    HarborEnvironmentError,
    HarborProviderCompatibility,
    inspect_harbor_package,
    register_harbor_environment,
)

AGENT = "example.test/agent@sha256:" + "a" * 64
VERIFIER = "example.test/verifier@sha256:" + "b" * 64


def _package(root: Path, *, gpus: int = 0) -> Path:
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "instruction.md").write_text("Make the test pass.\n", encoding="utf-8")
    (root / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (root / "tests" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.3"',
                'artifacts = ["/logs/artifacts/model.patch"]',
                "[task]",
                'name = "example/fix-defaults"',
                'description = "Fix default parsing"',
                "[metadata]",
                'task_id = "fix-defaults"',
                'display_title = "Fix defaults"',
                'language = "go"',
                "[agent]",
                'network_mode = "no-network"',
                "timeout_sec = 300",
                "[verifier]",
                'network_mode = "no-network"',
                'environment_mode = "separate"',
                "timeout_sec = 60",
                "[[verifier.collect]]",
                'command = "git diff > /logs/artifacts/model.patch"',
                "[environment]",
                "cpus = 2",
                "memory_mb = 4096",
                "storage_mb = 8192",
                f"gpus = {gpus}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_inspection_is_static_and_release_is_pinned(tmp_path: Path) -> None:
    package = _package(tmp_path / "package")
    marker = tmp_path / "must-not-run"
    task = package / "task.toml"
    task.write_text(task.read_text(encoding="utf-8") + f"\n# $(touch {marker})\n", encoding="utf-8")

    draft = inspect_harbor_package(package)

    assert not marker.exists()
    assert draft.package_id == "fix-defaults"
    assert draft.verifier_environment_mode == "separate"
    assert draft.candidate_artifacts == ("/logs/artifacts/model.patch",)
    release = register_harbor_environment(
        draft,
        agent_image=AGENT,
        verifier_image=VERIFIER,
        provider=HarborProviderCompatibility(provider_id="local-docker"),
    )
    assert release.validation.valid is True
    assert release.as_dict()["freshness"]["fresh"] is True


def test_release_freshness_refuses_stale_source_evidence(tmp_path: Path) -> None:
    package = _package(tmp_path / "package")
    draft = inspect_harbor_package(package)
    release = register_harbor_environment(
        draft,
        agent_image=AGENT,
        verifier_image=VERIFIER,
        provider=HarborProviderCompatibility(provider_id="local-docker"),
    )
    (package / "instruction.md").write_text("Changed instruction.\n", encoding="utf-8")

    receipt = release.freshness()

    assert receipt.fresh is False
    assert receipt.expected_source_package_digest != receipt.observed_source_package_digest


def test_release_refuses_mutable_images_and_incompatible_gpu(tmp_path: Path) -> None:
    draft = inspect_harbor_package(_package(tmp_path / "package", gpus=1))
    with pytest.raises(HarborEnvironmentError, match="harbor_release_agent_image_unpinned"):
        register_harbor_environment(
            draft,
            agent_image="example.test/agent:latest",
            verifier_image=VERIFIER,
            provider=HarborProviderCompatibility(provider_id="local-docker"),
        )
    with pytest.raises(HarborEnvironmentError, match="harbor_provider_gpu_unsupported"):
        register_harbor_environment(
            draft,
            agent_image=AGENT,
            verifier_image=VERIFIER,
            provider=HarborProviderCompatibility(provider_id="local-docker"),
        )


@pytest.mark.parametrize(
    "field,value", [("cpus", "1.5"), ("cpus", '"4"'), ("memory_mb", "true"), ("storage_mb", "inf")]
)
def test_resource_values_are_not_coerced_before_admission(tmp_path, field, value):
    import re

    package = _package(tmp_path / "package")
    path = package / "task.toml"
    path.write_text(
        re.sub(rf"^{field} = .*$", f"{field} = {value}", path.read_text(), flags=re.MULTILINE)
    )
    with pytest.raises(HarborEnvironmentError):
        inspect_harbor_package(package)


@pytest.mark.parametrize("value", ["true", "inf", "nan", '"300"'])
def test_phase_allowances_require_finite_numbers(tmp_path, value):
    package = _package(tmp_path / "package")
    path = package / "task.toml"
    path.write_text(path.read_text().replace("timeout_sec = 300", f"timeout_sec = {value}"))
    with pytest.raises(HarborEnvironmentError, match="timeout_invalid"):
        inspect_harbor_package(package)


def test_missing_network_policy_remains_unspecified(tmp_path):
    package = _package(tmp_path / "package")
    path = package / "task.toml"
    path.write_text(path.read_text().replace('network_mode = "no-network"\n', ""))
    draft = inspect_harbor_package(package)
    assert draft.agent_network == draft.verifier_network == "unspecified"


def test_modern_optional_task_metadata_uses_package_identity(tmp_path):
    package = _package(tmp_path / "package")
    path = package / "task.toml"
    text = path.read_text()
    start = text.index("[task]")
    end = text.index("[metadata]")
    path.write_text(text[:start] + text[end:])
    assert inspect_harbor_package(package).package_id == "fix-defaults"


@pytest.mark.parametrize("bound", ["file", "tree", "entries"])
def test_inspection_refuses_resource_exhaustion_before_large_reads(tmp_path, monkeypatch, bound):
    from synth_containers import harbor_environment as subject

    package = _package(tmp_path / "package")
    if bound == "file":
        monkeypatch.setattr(subject, "_MAX_FILE_BYTES", 10)
    elif bound == "tree":
        monkeypatch.setattr(subject, "_MAX_TREE_BYTES", 10)
    else:
        monkeypatch.setattr(subject, "_MAX_TREE_ENTRIES", 2)
    with pytest.raises(HarborEnvironmentError, match="too_large|entry_limit"):
        inspect_harbor_package(package)


def test_streamed_tree_digest_preserves_existing_release_identity(tmp_path):
    import hashlib

    from synth_containers.harbor_environment import _tree_digest

    root = _package(tmp_path / "package")
    legacy = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()):
        if path.is_file():
            relative = path.relative_to(root).as_posix().encode()
            contents = path.read_bytes()
            legacy.update(len(relative).to_bytes(8, "big"))
            legacy.update(relative)
            legacy.update(len(contents).to_bytes(8, "big"))
            legacy.update(contents)
    assert _tree_digest(root) == "sha256:" + legacy.hexdigest()


def test_special_files_cannot_block_inspection(tmp_path):
    import os

    package = _package(tmp_path / "package")
    os.mkfifo(package / "pipe")
    with pytest.raises(HarborEnvironmentError, match="special_file"):
        inspect_harbor_package(package)


def test_artifact_objects_are_not_silently_stringified(tmp_path):
    package = _package(tmp_path / "package")
    path = package / "task.toml"
    path.write_text(
        path.read_text().replace(
            'artifacts = ["/logs/artifacts/model.patch"]',
            'artifacts = [{source = "/logs/model.patch"}]',
        )
    )
    with pytest.raises(HarborEnvironmentError, match="artifacts_invalid"):
        inspect_harbor_package(package)


@pytest.mark.parametrize('kind', ['oversize', 'symlink', 'fifo'])
def test_task_metadata_reader_refuses_unbounded_or_nonregular_inputs(tmp_path, kind):
    import os
    from synth_containers.harbor_environment import read_harbor_task_toml
    path = tmp_path / 'task.toml'
    if kind == 'oversize':
        path.write_bytes(b'x' * (1024 * 1024 + 1))
    elif kind == 'symlink':
        target = tmp_path / 'other.toml'
        target.write_text('[task]\nname="fixture"\n')
        path.symlink_to(target)
    else:
        os.mkfifo(path)
    with pytest.raises(HarborEnvironmentError):
        read_harbor_task_toml(tmp_path)
