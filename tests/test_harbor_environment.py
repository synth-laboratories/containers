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
    task.write_text(task.read_text(encoding="utf-8") + f'\n# $(touch {marker})\n', encoding="utf-8")

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
