"""Materialize a pinned native Harbor task without executing task/build code."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .harbor_environment import (
    _IO_CHUNK_BYTES,
    _MAX_FILE_BYTES,
    _MAX_TREE_BYTES,
    HarborEnvironmentError,
    HarborEnvironmentRelease,
    _tree_digest,
    _tree_files,
)


def _copy_bounded_file(source: Path, target: Path, remaining_bytes: int) -> int:
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as reader:
        info = os.fstat(reader.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise HarborEnvironmentError("harbor_native_special_file_refused")
        maximum = min(_MAX_FILE_BYTES, remaining_bytes)
        if info.st_size > maximum:
            raise HarborEnvironmentError("harbor_native_copy_limit_exceeded")
        total = 0
        with target.open("xb") as writer:
            while chunk := reader.read(min(_IO_CHUNK_BYTES, maximum - total + 1)):
                total += len(chunk)
                if total > maximum:
                    raise HarborEnvironmentError("harbor_native_copy_limit_exceeded")
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    shutil.copystat(source, target, follow_symlinks=False)
    return total


def stage_native_harbor_task(
    release: HarborEnvironmentRelease,
    destination: Path,
    *,
    creation_timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Stage one immutable prebuilt-image task, retaining source/release identity.

    This first native slice supports a shared verifier on the same image. It
    refuses Compose and alternate verifier images before copying. A destination
    is exclusively created; an interrupted or failed stage is never reused.
    The receipt is committed last, after source and copied-tree reconciliation.
    This binds an operator-selected image; it does not prove how it was built.
    """
    if type(creation_timeout_seconds) is not int or not 1 <= creation_timeout_seconds <= 300:
        raise HarborEnvironmentError("harbor_native_creation_allowance_invalid")
    if not release.validation.valid or not release.freshness().fresh:
        raise HarborEnvironmentError("harbor_native_source_release_stale")
    if release.draft.verifier_environment_mode not in {"shared", "same"}:
        raise HarborEnvironmentError("harbor_native_separate_verifier_unqualified")
    if release.agent_image != release.verifier_image:
        raise HarborEnvironmentError("harbor_native_shared_verifier_image_mismatch")
    if release.provider.provider_id not in {"docker", "daytona"}:
        raise HarborEnvironmentError("harbor_native_provider_unqualified")
    source = release.draft.root
    for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
        if (source / "environment" / name).exists():
            raise HarborEnvironmentError("harbor_native_compose_unqualified")
    destination = destination.expanduser().absolute()
    if destination.resolve().is_relative_to(source):
        raise HarborEnvironmentError("harbor_native_stage_cannot_mutate_source")
    try:
        import toml
        from harbor.models.task.config import TaskConfig
    except ImportError as error:
        raise HarborEnvironmentError("harbor_native_stage_requires_harbor_extra") from error
    if version("harbor") != "0.22.0":
        raise HarborEnvironmentError("harbor_native_stage_requires_qualified_harbor_022")
    original = (source / "task.toml").read_text()
    manifest = tomllib.loads(original)
    environment = manifest["environment"]
    if environment.get("mounts") or environment.get("volumes"):
        raise HarborEnvironmentError("harbor_native_host_mounts_unqualified")
    environment["docker_image"] = release.agent_image
    environment["build_timeout_sec"] = creation_timeout_seconds
    staged_toml = toml.dumps(manifest)
    try:
        TaskConfig.model_validate_toml(staged_toml)
    except ValueError as error:
        raise HarborEnvironmentError("harbor_native_task_contract_invalid") from error
    staged = tomllib.loads(staged_toml)
    if staged != manifest:
        raise HarborEnvironmentError("harbor_native_task_serialization_changed_values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    task = destination / "task"
    # Freeze a bounded inventory, then bound every copy again. A source growing
    # after inspection cannot turn preparation into an unbounded disk write.
    total = 0
    for source_file in _tree_files(source):
        target = task / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        total += _copy_bounded_file(source_file, target, _MAX_TREE_BYTES - total)
    if _tree_digest(task) != release.draft.source_package_digest or not release.freshness().fresh:
        raise HarborEnvironmentError("harbor_native_source_changed_during_stage")
    (destination / "source-task.toml").write_text(original)
    (task / "task.toml").write_text(staged_toml)
    (task / "environment" / "Dockerfile").unlink()
    receipt = {
        "schema_version": "synth.harbor-native-task-stage.v1",
        "task_path": str(task),
        "source_package_digest": release.draft.source_package_digest,
        "staged_package_digest": _tree_digest(task),
        "environment_release_id": release.release_id,
        "environment_release_digest": release.release_digest,
        "image": release.agent_image,
        "provider": release.provider.provider_id,
        "creation_timeout_seconds": creation_timeout_seconds,
        "source_creation_timeout_seconds": tomllib.loads(original)["environment"].get(
            "build_timeout_sec"
        ),
        "excluded_build_files": ["environment/Dockerfile"],
        "image_build_provenance": "operator_bound_not_verified",
        "release": release.as_dict(),
    }
    if not receipt["release"]["freshness"]["fresh"]:
        raise HarborEnvironmentError("harbor_native_source_changed_during_stage")
    # Flush copied file contents and directory entries before the receipt. A
    # partial stage without this receipt has no authority to launch execution.
    for directory, _, files in os.walk(destination, topdown=False, followlinks=False):
        for name in files:
            with (Path(directory) / name).open("rb") as handle:
                os.fsync(handle.fileno())
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    with (destination / "stage-receipt.json").open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for directory in (destination, destination.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return receipt
