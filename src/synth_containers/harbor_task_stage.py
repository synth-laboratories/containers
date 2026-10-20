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
    HarborResourceRequest,
    _tree_digest,
    _tree_files,
    is_pinned_harbor_image,
    read_harbor_task_toml,
)
from .harbor_phase_limits import NativeHarborPhaseLimits
from .native_limit_admission import require_native_limit_capabilities


def native_harbor_environment_flags(
    package: Path,
    *,
    provider: str,
    image: str,
    resource_ttl_minutes: int,
    resource_request: HarborResourceRequest | None = None,
    docker_resource_custody: bool = False,
    docker_egress_image: str | None = None,
) -> list[str]:
    """Resolve qualified native backend arguments without widening task resources."""
    if not is_pinned_harbor_image(image, provider):
        raise HarborEnvironmentError("native image must be digest-pinned")
    if type(resource_ttl_minutes) is not int or not 1 <= resource_ttl_minutes <= 360:
        raise HarborEnvironmentError(
            "native resource TTL must be an integer from 1 through 360 minutes"
        )
    if type(docker_resource_custody) is not bool:
        raise HarborEnvironmentError("native Docker resource custody must be boolean")
    if (docker_resource_custody or docker_egress_image is not None) and provider != "docker":
        raise HarborEnvironmentError("native Docker custody options require Docker")
    if docker_egress_image is not None and (
        not docker_resource_custody or not is_pinned_harbor_image(docker_egress_image, "docker")
    ):
        raise HarborEnvironmentError(
            "native Docker egress image requires custody and an immutable image"
        )
    overrides = native_harbor_resource_overrides(resource_request)
    if provider == "docker":
        if not docker_resource_custody:
            return ["--env", "docker"]
        flags = ["--env", "synth_containers.harbor_docker:ObservedDockerEnvironment"]
        if docker_egress_image is not None:
            flags += ["--ek", f"egress_control_image={docker_egress_image}"]
        return flags
    if provider != "daytona":
        raise HarborEnvironmentError("native image provider must be docker or daytona")
    config = tomllib.loads(read_harbor_task_toml(package))
    environment = config.get("environment")
    if not isinstance(environment, dict):
        raise HarborEnvironmentError("native Daytona requires an environment resource table")
    environment = {**environment, **overrides}
    values = [environment.get(field) for field in ("cpus", "memory_mb", "storage_mb")]
    if any(type(value) is not int or value < 1 for value in values):
        raise HarborEnvironmentError(
            "native Daytona requires explicit positive CPU, memory and storage requests"
        )
    cpus, memory, storage = values
    memory_gib, storage_gib = (memory + 1023) // 1024, (storage + 1023) // 1024
    if cpus > 64 or memory_gib > 512 or storage_gib > 1024 or environment.get("gpus", 0):
        raise HarborEnvironmentError(
            "native Daytona task request exceeds qualified resource ceilings"
        )
    return [
        "--env",
        "synth_containers.harbor_daytona:BoundedDaytonaEnvironment",
        "--ek",
        "auto_snapshot=false",
        "--ek",
        f"resource_ttl_minutes={resource_ttl_minutes}",
        "--ek",
        f"maximum_cpu={cpus}",
        "--ek",
        f"maximum_memory_gib={memory_gib}",
        "--ek",
        f"maximum_disk_gib={storage_gib}",
    ]


def native_harbor_resource_overrides(request: HarborResourceRequest | None) -> dict[str, int]:
    """Explicit overrides are complete and bounded; absent means preserve source."""
    if request is None:
        return {}
    values = request.as_dict()
    maxima = {"cpus": 64, "memory_mb": 512 * 1024, "storage_mb": 1024 * 1024}
    for field, maximum in maxima.items():
        value = values[field]
        if type(value) is not int or not 1 <= value <= maximum:
            raise HarborEnvironmentError(
                "native resource overrides require bounded positive CPU, memory and storage"
            )
    if type(request.gpus) is not int or request.gpus != 0:
        raise HarborEnvironmentError("native GPU resource overrides are not qualified")
    return {field: int(values[field]) for field in maxima}


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
    resource_ttl_minutes: int = 20,
    environment_transfer: str = "preserve",
    resource_request: HarborResourceRequest | None = None,
    docker_resource_custody: bool = False,
    docker_egress_image: str | None = None,
    phase_limits: NativeHarborPhaseLimits | None = None,
    required_limit_capabilities: object = (),
) -> dict[str, Any]:
    """Stage one immutable prebuilt-image task, retaining source/release identity.

    This first native slice supports a shared verifier on the same image. It
    refuses Compose and alternate verifier images before copying. A destination
    is exclusively created; an interrupted or failed stage is never reused.
    The receipt is committed last, after source and copied-tree reconciliation.
    This binds an operator-selected image; it does not prove how it was built.
    """
    admitted_capabilities = require_native_limit_capabilities(required_limit_capabilities)
    if environment_transfer not in {"preserve", "image_only"}:
        raise HarborEnvironmentError("harbor_native_environment_transfer_invalid")
    if phase_limits is not None and not isinstance(phase_limits, NativeHarborPhaseLimits):
        raise HarborEnvironmentError("harbor_native_phase_limits_invalid")
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
    environment_flags = native_harbor_environment_flags(
        source,
        provider=release.provider.provider_id,
        image=release.agent_image,
        resource_ttl_minutes=resource_ttl_minutes,
        resource_request=resource_request,
        docker_resource_custody=docker_resource_custody,
        docker_egress_image=docker_egress_image,
    )
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
    original = read_harbor_task_toml(source)
    manifest = tomllib.loads(original)
    environment = manifest["environment"]
    if environment.get("mounts") or environment.get("volumes"):
        raise HarborEnvironmentError("harbor_native_host_mounts_unqualified")
    environment.update(native_harbor_resource_overrides(resource_request))
    environment["docker_image"] = release.agent_image
    environment["build_timeout_sec"] = creation_timeout_seconds
    phase_receipt = phase_limits.apply(manifest) if phase_limits is not None else None
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
    excluded = ["environment/Dockerfile"]
    if environment_transfer == "image_only":
        excluded = [str(path.relative_to(task)) for path in _tree_files(task / "environment")]
        shutil.rmtree(task / "environment")
        (task / "environment").mkdir()
    else:
        (task / "environment" / "Dockerfile").unlink()
    receipt = {
        "schema_version": "synth.harbor-native-task-stage.v1",
        "task_path": str(task),
        "native_environment_flags": environment_flags,
        "docker_resource_custody": docker_resource_custody,
        "docker_egress_image": docker_egress_image,
        "source_resource_request": release.draft.resource_request.as_dict(),
        "resolved_resource_request": {
            field: environment.get(field, 0 if field == "gpus" else None)
            for field in ("cpus", "memory_mb", "storage_mb", "gpus")
        },
        "source_package_digest": release.draft.source_package_digest,
        "staged_package_digest": _tree_digest(task),
        "environment_release_id": release.release_id,
        "environment_release_digest": release.release_digest,
        "image": release.agent_image,
        "image_reference_scope": (
            "docker_local_image_id"
            if release.agent_image.startswith("sha256:")
            else "registry_manifest_digest"
        ),
        "provider": release.provider.provider_id,
        "creation_timeout_seconds": creation_timeout_seconds,
        "phase_limits": phase_receipt,
        "required_limit_capabilities": admitted_capabilities,
        "source_creation_timeout_seconds": tomllib.loads(original)["environment"].get(
            "build_timeout_sec"
        ),
        "environment_transfer": environment_transfer,
        "excluded_build_files": excluded,
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
