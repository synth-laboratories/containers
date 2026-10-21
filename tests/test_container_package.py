"""Executable, provider-free package/build protocol and refusal qualification."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_containers import (
    ContainerArtifactDescriptor,
    ContainerPackageError,
    ImageArtifactIdentity,
    PackageBuildPlan,
    build_container_package,
    plan_container_package,
    validate_container_build_output,
)
from synth_containers.cli import main

DIGEST = "sha256:" + "ab" * 32
BUILDER = """#!/usr/bin/env python3
import json, os
from pathlib import Path
plan = json.loads(Path(os.environ["SYNTH_CONTAINER_BUILD_PLAN"]).read_text())
assert Path("payload/data.txt").read_text() == "retained input"
assert not Path("undeclared.txt").exists()
assert "SYNTH_TEST_UNDECLARED_SECRET" not in os.environ
print("resolver progress, retained outside machine stdout")
Path(os.environ["SYNTH_CONTAINER_BUILD_OUTPUT"]).write_text(json.dumps({
    "schema_version": "synth.container-build-output.v1",
    "action": "resolve", "input_digest": plan["input_digest"],
    "artifact": {"kind": "oci_image", "digest_kind": "oci_manifest",
        "reference": "registry.example/eval@DIGEST", "digest": "DIGEST",
        "platform": "linux/amd64"}
}))
""".replace("DIGEST", DIGEST)


def package(root: Path, *, builder: str = BUILDER) -> Path:
    root.mkdir(parents=True)
    (root / "build.py").write_text(builder)
    (root / "build.py").chmod(0o755)
    (root / "payload/empty").mkdir(parents=True)
    (root / "payload/data.txt").write_text("retained input")
    (root / "undeclared.txt").write_text("excluded")
    path = root / "container.toml"
    path.write_text("""schema_version = "synth.container-package.v1"
package_id = "generic-cli"
[build]
entrypoint = "build.py"
inputs = ["payload"]
platform = "linux/amd64"
[runtime]
profile = "command"
command = ["python", "run.py"]
[runtime.readiness]
kind = "process"
[[credentials]]
name = "MODEL_API_KEY"
reference = "secretref://org/model"
""")
    return path


def output(plan: PackageBuildPlan) -> dict:
    return {
        "schema_version": "synth.container-build-output.v1",
        "action": "resolve",
        "input_digest": plan.input_digest,
        "artifact": {
            "kind": "oci_image",
            "digest_kind": "oci_manifest",
            "reference": "registry.example/eval@" + DIGEST,
            "digest": DIGEST,
            "platform": "linux/amd64",
        },
    }


def test_plan_is_portable_and_does_not_execute_package_code(tmp_path: Path) -> None:
    manifest = package(tmp_path / "a", builder="#!/bin/sh\nexit 99\n")
    first = plan_container_package(manifest)
    shutil.copytree(manifest.parent, tmp_path / "b")
    second = plan_container_package(tmp_path / "b/container.toml")
    assert first == second
    assert str(tmp_path) not in first.model_dump_json()
    assert "payload/empty" in first.directories
    assert "undeclared.txt" not in {f.path for f in first.files}
    assert PackageBuildPlan.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize("change", ["file", "mode", "empty-directory", "manifest", "entrypoint"])
def test_complete_input_identity_changes(tmp_path: Path, change: str) -> None:
    manifest = package(tmp_path / "source")
    before = plan_container_package(manifest).input_digest
    if change == "file":
        (manifest.parent / "payload/data.txt").write_text("changed")
    elif change == "mode":
        (manifest.parent / "payload/data.txt").chmod(0o755)
    elif change == "empty-directory":
        (manifest.parent / "payload/another-empty").mkdir()
    elif change == "manifest":
        manifest.write_text(manifest.read_text().replace('"run.py"', '"different.py"'))
    else:
        (manifest.parent / "build.py").write_text(BUILDER + "\n# changed\n")
    assert plan_container_package(manifest).input_digest != before


@pytest.mark.parametrize(
    "bad_path", ["../outside", "/absolute", "payload/../build.py", "./build.py", "a\\b"]
)
def test_manifest_refuses_nonportable_paths(tmp_path: Path, bad_path: str) -> None:
    manifest = package(tmp_path / "source")
    manifest.write_text(
        manifest.read_text().replace('inputs = ["payload"]', "inputs = " + json.dumps([bad_path]))
    )
    with pytest.raises(ContainerPackageError, match="manifest_invalid"):
        plan_container_package(manifest)


def test_unknown_fields_and_schema_versions_refuse(tmp_path: Path) -> None:
    manifest = package(tmp_path / "source")
    original = manifest.read_text()
    for value in [original + "\nunknown = true\n", original.replace("package.v1", "package.v2")]:
        manifest.write_text(value)
        with pytest.raises(ContainerPackageError, match="manifest_invalid"):
            plan_container_package(manifest)


@pytest.mark.parametrize("location", ["file", "directory"])
def test_symlink_inputs_refuse(tmp_path: Path, location: str) -> None:
    manifest = package(tmp_path / "source")
    if location == "file":
        target = manifest.parent / "payload/data.txt"
        target.unlink()
        target.symlink_to(manifest.parent / "undeclared.txt")
    else:
        (manifest.parent / "payload/link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ContainerPackageError, match="input_unavailable"):
        plan_container_package(manifest)


def test_entrypoint_must_be_executable(tmp_path: Path) -> None:
    manifest = package(tmp_path / "source")
    (manifest.parent / "build.py").chmod(0o644)
    with pytest.raises(ContainerPackageError, match="manifest_invalid"):
        plan_container_package(manifest)


@pytest.mark.parametrize("field", ["input_digest", "platform", "digest", "schema_version"])
def test_foreign_or_incompatible_build_output_refuses(tmp_path: Path, field: str) -> None:
    plan = plan_container_package(package(tmp_path / "source"))
    value = output(plan)
    if field == "input_digest":
        value[field] = "sha256:" + "cd" * 32
    elif field == "platform":
        value["artifact"][field] = "linux/arm64"
    elif field == "digest":
        value["artifact"][field] = "sha256:" + "cd" * 32
    else:
        value[field] = "future.v2"
    with pytest.raises(ContainerPackageError, match="build_output_invalid"):
        validate_container_build_output(plan, value)


def test_daemon_config_and_registry_identity_remain_distinct(tmp_path: Path) -> None:
    plan = plan_container_package(package(tmp_path / "source"))
    value = output(plan)
    value["artifact"].update(digest_kind="oci_config", reference=DIGEST)
    descriptor = validate_container_build_output(plan, value)
    assert descriptor.output.artifact.reference == DIGEST
    with pytest.raises(ValidationError):
        ImageArtifactIdentity.model_validate({**value["artifact"], "reference": "repo@" + DIGEST})
    value["artifact"]["digest_kind"] = "oci_manifest"
    with pytest.raises(ContainerPackageError):
        validate_container_build_output(plan, value)


def test_real_entrypoint_runs_from_frozen_inputs_and_retains_protocol(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("SYNTH_TEST_UNDECLARED_SECRET", "must-not-be-forwarded")
    manifest = package(tmp_path / "source")
    destination = tmp_path / "receipt"
    result = asyncio.run(build_container_package(manifest, destination))
    assert result.artifact_verification == "not_checked"
    assert result.runtime_validation == "not_run"
    assert (destination / "build.log").read_text().startswith("resolver progress")
    assert (
        ContainerArtifactDescriptor.model_validate_json((destination / "artifact.json").read_text())
        == result
    )
    assert not list(destination.glob("inputs-*"))
    with pytest.raises(ContainerPackageError, match="output_exists"):
        asyncio.run(build_container_package(manifest, destination))


@pytest.mark.parametrize("kind", ["timeout", "output", "failure", "missing", "invalid-json"])
def test_failed_build_keeps_plan_and_typed_failure(tmp_path: Path, kind: str) -> None:
    statements = {
        "timeout": "import time; time.sleep(30)",
        "output": "print('x' * 10000)",
        "failure": "raise SystemExit(7)",
        "missing": "pass",
        "invalid-json": "import os; from pathlib import Path; Path(os.environ['SYNTH_CONTAINER_BUILD_OUTPUT']).write_text('not json')",
    }
    manifest = package(
        tmp_path / "source", builder="#!" + sys.executable + "\n" + statements[kind] + "\n"
    )
    destination = tmp_path / "receipt"
    with pytest.raises(ContainerPackageError):
        asyncio.run(
            build_container_package(manifest, destination, timeout_seconds=0.15, max_log_bytes=100)
        )
    assert (destination / "plan.json").exists()
    assert (destination / "failure.json").exists()
    assert not (destination / "artifact.json").exists()
    assert (destination / "build.log").stat().st_size <= 100


def test_cli_machine_output_and_failure_are_json(tmp_path: Path, capsys) -> None:
    manifest = package(tmp_path / "source")
    assert main(["package", "plan", str(manifest)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "synth.container-build-plan.v1"
    assert main(["package", "build", str(manifest), "--output", str(tmp_path / "receipt")]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "synth.container-artifact.v1"
    assert main(["package", "plan", str(tmp_path / "missing.toml")]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "container_package_input_unavailable"


def test_json_authoring_and_schema_are_available_without_building(tmp_path: Path, capsys) -> None:
    manifest = package(tmp_path / "source", builder="#!/bin/sh\nexit 99\n")
    declaration = plan_container_package(manifest).manifest
    json_manifest = manifest.with_suffix(".json")
    json_manifest.write_text(declaration.model_dump_json())
    assert plan_container_package(json_manifest).manifest == declaration
    assert main(["package", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "synth.container-package.v1"


def test_duplicate_json_keys_refuse(tmp_path: Path) -> None:
    manifest = package(tmp_path / "source").with_suffix(".json")
    manifest.write_text('{"package_id":"first","package_id":"second"}')
    with pytest.raises(ContainerPackageError, match="manifest_invalid"):
        plan_container_package(manifest)


def test_changed_inputs_refuse_before_entrypoint_execution(tmp_path: Path, monkeypatch) -> None:
    from synth_containers import package_build

    manifest = package(tmp_path / "source")
    original = package_build.materialize_inputs

    def change_before_copy(root, plan, destination):
        (root / "payload/data.txt").write_text("changed after acceptance")
        return original(root, plan, destination)

    monkeypatch.setattr(package_build, "materialize_inputs", change_before_copy)
    destination = tmp_path / "receipt"
    with pytest.raises(ContainerPackageError, match="input_changed"):
        asyncio.run(build_container_package(manifest, destination))
    assert not (destination / "build.log").exists()
    assert json.loads((destination / "failure.json").read_text())["error"].endswith("input_changed")


def test_cancelled_build_retains_disposition(tmp_path: Path) -> None:
    manifest = package(
        tmp_path / "source", builder="#!" + sys.executable + "\nimport time; time.sleep(30)\n"
    )
    destination = tmp_path / "receipt"

    async def cancel():
        task = asyncio.create_task(build_container_package(manifest, destination))
        for _ in range(100):
            if (destination / "build.log").exists():
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert json.loads((destination / "failure.json").read_text())["error"].endswith("cancelled")
    assert not (destination / "artifact.json").exists()


@pytest.mark.parametrize("limit", ["entries", "file", "total"])
def test_input_resource_bounds_are_enforced(tmp_path: Path, monkeypatch, limit: str) -> None:
    from synth_containers import package_files

    manifest = package(tmp_path / "source")
    if limit == "entries":
        monkeypatch.setattr(package_files, "MAX_ENTRIES", 3)
    else:
        (manifest.parent / "payload/data.txt").write_bytes(b"x" * 4096)
        monkeypatch.setattr(
            package_files, "MAX_FILE_BYTES" if limit == "file" else "MAX_PACKAGE_BYTES", 2048
        )
    with pytest.raises(ContainerPackageError, match="limit"):
        plan_container_package(manifest)


def test_builder_cannot_change_its_plan_receipt(tmp_path: Path) -> None:
    builder = (
        BUILDER
        + """
p = Path(os.environ["SYNTH_CONTAINER_BUILD_PLAN"])
p.chmod(0o644)
p.write_text("{}")
"""
    )
    manifest = package(tmp_path / "source", builder=builder)
    destination = tmp_path / "receipt"
    with pytest.raises(ContainerPackageError, match="build_plan_changed"):
        asyncio.run(build_container_package(manifest, destination))
    assert PackageBuildPlan.model_validate_json((destination / "plan.json").read_text())
    assert not (destination / "artifact.json").exists()
