"""Freeze/execute a package-owned build entrypoint; no provider or runtime admission.

See docs/specs/container-package-v1.md. Execution is a trusted local build,
not a sandbox. Rust entrypoints must use the owner's managed mbx build path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import tomllib
from pathlib import Path

from pydantic import ValidationError

from .bounded_process import ProcessOutputLimit, run_bounded_process
from .package_files import (
    ContainerPackageError,
    materialize_inputs,
    read_package_file,
    snapshot_inputs,
)
from .package_models import (
    ContainerArtifactDescriptor,
    ContainerPackageManifest,
    PackageBuildOutput,
    PackageBuildPlan,
    content_digest,
)

MAX_DESCRIPTOR_BYTES = 1024 * 1024


def _retain(path: Path, data: bytes) -> None:
    """Publish complete, fsynced receipt bytes without replacing an existing record."""
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def plan_container_package(manifest_path: str | Path) -> PackageBuildPlan:
    """Provider-free preflight; planning never executes package code."""
    path = Path(manifest_path).absolute()
    try:
        raw = read_package_file(path.parent, path.name, maximum=MAX_DESCRIPTOR_BYTES)
        if path.suffix == ".toml":
            payload = tomllib.loads(raw.decode("utf-8"))
        elif path.suffix == ".json":
            payload = json.loads(raw, object_pairs_hook=_unique_object)
        else:
            raise ContainerPackageError("container_package_format_unsupported")
        manifest = ContainerPackageManifest.model_validate(payload)
        files, directories = snapshot_inputs(
            path.parent, (path.name, manifest.build.entrypoint, *manifest.build.inputs)
        )
        # The parsed declaration and retained manifest bytes must name the same snapshot.
        retained = next(entry for entry in files if entry.path == path.name)
        if retained.digest != "sha256:" + hashlib.sha256(raw).hexdigest():
            raise ContainerPackageError("container_package_input_changed")
        identity = {
            "manifest": manifest.model_dump(mode="json"),
            "files": [entry.model_dump(mode="json") for entry in files],
            "directories": list(directories),
        }
        return PackageBuildPlan(
            schema_version="synth.container-build-plan.v1",
            manifest=manifest,
            files=files,
            directories=directories,
            input_digest=content_digest(identity),
        )
    except (ValueError, UnicodeError, OSError) as error:
        if isinstance(error, ContainerPackageError):
            raise
        raise ContainerPackageError("container_package_manifest_invalid") from None


def validate_container_build_output(
    plan: PackageBuildPlan, output: PackageBuildOutput | dict
) -> ContainerArtifactDescriptor:
    """Bind builder-declared artifact identity to the exact frozen input plan."""
    try:
        output = PackageBuildOutput.model_validate(output)
        return ContainerArtifactDescriptor(
            schema_version="synth.container-artifact.v1",
            plan=plan,
            output=output,
        )
    except ValidationError:
        raise ContainerPackageError("container_package_build_output_invalid") from None


async def build_container_package(
    manifest_path: str | Path,
    output_directory: str | Path,
    *,
    timeout_seconds: float = 300,
    max_log_bytes: int = 4 * 1024 * 1024,
) -> ContainerArtifactDescriptor:
    """Execute the entrypoint against copied, hash-checked inputs with bounded logs.

    The output directory is exclusive and retains the plan/log/failure receipt.
    A descriptor validates the protocol; image existence, startup, capabilities,
    provider support and current permission still require their own checks.
    """
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ContainerPackageError("container_package_timeout_invalid")
    if type(max_log_bytes) is not int or max_log_bytes <= 0:
        raise ContainerPackageError("container_package_log_limit_invalid")
    plan = plan_container_package(manifest_path)
    destination = Path(output_directory).absolute()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ContainerPackageError("container_package_output_exists") from None
    except OSError:
        raise ContainerPackageError("container_package_output_unavailable") from None
    try:
        _retain(destination / "plan.json", (plan.model_dump_json(indent=2) + "\n").encode())
        with tempfile.TemporaryDirectory(prefix="inputs-", dir=destination) as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            protocol = Path(temporary) / "protocol"
            protocol.mkdir()
            plan_bytes = (plan.model_dump_json(indent=2) + "\n").encode()
            (protocol / "plan.json").write_bytes(plan_bytes)
            (protocol / "plan.json").chmod(0o444)
            materialize_inputs(Path(manifest_path).absolute().parent, plan, root)
            # Explicit protocol env carries paths, not credential values. The build
            # has ordinary local authority; this allowlist is not an isolation claim.
            environment = {
                key: os.environ[key]
                for key in ("PATH", "LANG", "LC_ALL", "TMPDIR")
                if key in os.environ
            }
            environment.update(
                SYNTH_CONTAINER_BUILD_PLAN=str(protocol / "plan.json"),
                SYNTH_CONTAINER_BUILD_OUTPUT=str(protocol / "build-output.json"),
                SYNTH_CONTAINER_PACKAGE_ROOT=str(root),
            )
            async with asyncio.timeout(timeout_seconds):
                code = await run_bounded_process(
                    [str(root / plan.manifest.build.entrypoint), *plan.manifest.build.arguments],
                    output=destination / "build.log",
                    max_output_bytes=max_log_bytes,
                    cwd=root,
                    env=environment,
                )
            if code != 0:
                raise ContainerPackageError("container_package_build_failed")
            if read_package_file(protocol, "plan.json", maximum=len(plan_bytes)) != plan_bytes:
                raise ContainerPackageError("container_package_build_plan_changed")
            raw = read_package_file(protocol, "build-output.json", maximum=MAX_DESCRIPTOR_BYTES)
            _retain(destination / "build-output.json", raw)
            descriptor = validate_container_build_output(
                plan, json.loads(raw, object_pairs_hook=_unique_object)
            )
            _retain(
                destination / "artifact.json",
                (descriptor.model_dump_json(indent=2) + "\n").encode(),
            )
            return descriptor
    except asyncio.CancelledError:
        _retain(
            destination / "failure.json",
            (json.dumps({"error": "container_package_build_cancelled"}) + "\n").encode(),
        )
        raise
    except (OSError, ValueError, ProcessOutputLimit, TimeoutError) as error:
        if isinstance(error, ContainerPackageError):
            code = str(error)
        elif isinstance(error, TimeoutError):
            code = "container_package_build_timeout"
        elif isinstance(error, ProcessOutputLimit):
            code = "container_package_build_log_limit"
        elif isinstance(error, OSError):
            code = "container_package_build_io_failed"
        else:
            code = "container_package_build_output_invalid"
        try:
            _retain(destination / "failure.json", (json.dumps({"error": code}) + "\n").encode())
        except OSError:
            raise ContainerPackageError("container_package_failure_receipt_unavailable") from None
        raise ContainerPackageError(code) from None
