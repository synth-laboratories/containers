"""Execution declarations are bound to frozen package bytes before preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_containers.execution_package import (
    ExecutionPackageCode,
    ExecutionPackageError,
    ExecutionPackagePlan,
    plan_execution_package,
)
from synth_containers.package_build import plan_container_package


def package(root: Path, *, declared: bool = True, platform: str = "linux/amd64") -> Path:
    root.mkdir()
    (root / "build.py").write_text(
        "#!/usr/bin/env python3\nraise RuntimeError('must not execute')\n"
    )
    (root / "build.py").chmod(0o755)
    inputs = '["execution-requirements.toml"]' if declared else "[]"
    manifest = root / "container.toml"
    manifest.write_text(f'''schema_version = "synth.container-package.v1"
package_id = "scorer"
[build]
entrypoint = "build.py"
inputs = {inputs}
platform = "{platform}"
[runtime]
profile = "command"
command = ["python", "run.py"]
[runtime.readiness]
kind = "process"
''')
    (
        root / "execution-requirements.toml"
    ).write_text("""schema_version = "synth.container-execution-requirements.v1"
platform = "linux/amd64"
max_wall_seconds = 600
privileged = false
verifier_isolation = "self_reported"
[resources]
cpu_millis = 2000
memory_bytes = 4294967296
disk_bytes = 0
gpu_kind = "none"
gpu_count = 0
[network]
mode = "none"
hosts = []
""")
    return manifest


def test_plan_binds_source_bytes_and_semantic_requirements_without_executing(tmp_path):
    manifest = package(tmp_path / "package")
    historical = plan_container_package(manifest)
    planned = plan_execution_package(manifest)
    assert planned.build_plan == historical
    assert planned.execution_requirements_digest == planned.execution_requirements.digest()
    source = manifest.parent / "execution-requirements.toml"
    assert (
        planned.requirements_file.digest
        == "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert planned.requirements_file in historical.files
    assert ExecutionPackagePlan.model_validate_json(planned.model_dump_json()) == planned


def test_sidecar_outside_the_declared_build_snapshot_is_refused(tmp_path):
    manifest = package(tmp_path / "package", declared=False)
    with pytest.raises(ExecutionPackageError) as caught:
        plan_execution_package(manifest)
    assert caught.value.refusal.code is ExecutionPackageCode.REQUIREMENTS_NOT_FROZEN
    assert caught.value.refusal.package_id == "scorer"
    assert caught.value.refusal.stage == "execution_package_plan"


def test_build_platform_cannot_be_relabelled_by_requirements(tmp_path):
    manifest = package(tmp_path / "package", platform="linux/arm64")
    with pytest.raises(ExecutionPackageError) as caught:
        plan_execution_package(manifest)
    assert caught.value.refusal.code is ExecutionPackageCode.PLATFORM_MISMATCH


def test_requirements_change_between_snapshot_and_read_is_refused(tmp_path, monkeypatch):
    from synth_containers import execution_package

    manifest = package(tmp_path / "package")
    reader = execution_package.read_package_file

    def changed(root, relative, *, maximum):
        path = root / relative
        path.write_text(path.read_text() + "\n# changed bytes, identical semantics\n")
        return reader(root, relative, maximum=maximum)

    monkeypatch.setattr(execution_package, "read_package_file", changed)
    with pytest.raises(ExecutionPackageError) as caught:
        plan_execution_package(manifest)
    assert caught.value.refusal.code is ExecutionPackageCode.REQUIREMENTS_CHANGED


def test_a_retained_plan_cannot_claim_another_semantic_digest(tmp_path):
    plan = plan_execution_package(package(tmp_path / "package"))
    payload = plan.model_dump(mode="json")
    payload["execution_requirements_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError, match="execution_package_requirements_digest_mismatch"):
        ExecutionPackagePlan.model_validate(payload)


@pytest.mark.parametrize("content", [b"\xff", b"platform = ", b'platform = "linux/amd64"'])
def test_invalid_source_is_a_typed_refusal(tmp_path, content):
    manifest = package(tmp_path / "package")
    (manifest.parent / "execution-requirements.toml").write_bytes(content)
    with pytest.raises(ExecutionPackageError) as caught:
        plan_execution_package(manifest)
    assert caught.value.refusal.code is ExecutionPackageCode.REQUIREMENTS_INVALID


def test_historical_package_planning_never_requires_a_new_sidecar(tmp_path):
    manifest = package(tmp_path / "package", declared=False)
    (manifest.parent / "execution-requirements.toml").unlink()
    assert plan_container_package(manifest).manifest.package_id == "scorer"


def test_requirements_size_bound_has_a_distinct_refusal(tmp_path):
    from synth_containers.execution_limits import MAXIMUM_REQUIREMENTS_BYTES

    manifest = package(tmp_path / "package")
    (manifest.parent / "execution-requirements.toml").write_bytes(
        b"#" * (MAXIMUM_REQUIREMENTS_BYTES + 1)
    )
    with pytest.raises(ExecutionPackageError) as caught:
        plan_execution_package(manifest)
    assert caught.value.refusal.code is ExecutionPackageCode.REQUIREMENTS_TOO_LARGE


def test_cli_emits_one_structured_refusal_without_source_contents(tmp_path, capsys):
    from synth_containers.package_cli import main

    manifest = package(tmp_path / "package", declared=False)
    assert main(["execution-plan", str(manifest)]) == 2
    output = capsys.readouterr()
    assert output.err == ""
    assert len(output.out.splitlines()) == 1
    refusal = json.loads(output.out)
    assert refusal == {
        "code": "execution_package_requirements_not_frozen",
        "stage": "execution_package_plan",
        "cause": "requirements_file_not_in_build_inputs",
        "package_id": "scorer",
        "requirements_path": "execution-requirements.toml",
    }


def test_cli_retains_the_compound_plan_without_running_the_builder(tmp_path, capsys):
    from synth_containers.package_cli import main

    manifest = package(tmp_path / "package")
    assert main(["execution-plan", str(manifest)]) == 0
    plan = ExecutionPackagePlan.model_validate_json(capsys.readouterr().out)
    assert plan.execution_requirements_digest == plan.execution_requirements.digest()
