"""Portable package/build contracts; see docs/specs/container-package-v1.md."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")]
Platform = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+/[a-z0-9_]+(?:/[a-zA-Z0-9_.-]+)?$")]


def content_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def portable_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("Expected a canonical package-relative path")
    return value


class PackageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class BuildEntrypoint(PackageModel):
    entrypoint: str
    inputs: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    platform: Platform

    @model_validator(mode="after")
    def validate_paths(self) -> BuildEntrypoint:
        for path in (self.entrypoint, *self.inputs):
            portable_path(path)
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Duplicate build input")
        if any("\x00" in arg for arg in self.arguments):
            raise ValueError("Invalid build argument")
        return self


class ReadinessDeclaration(PackageModel):
    kind: Literal["process", "http", "adapter"]
    path: str | None = None
    port: Annotated[int, Field(strict=True, ge=1, le=65535)] | None = None
    adapter: Name | None = None

    @model_validator(mode="after")
    def validate_probe(self) -> ReadinessDeclaration:
        if self.kind == "http":
            if not self.path or not self.path.startswith("/") or self.port is None:
                raise ValueError("HTTP readiness requires a path and port")
            if any(ord(c) < 32 for c in self.path):
                raise ValueError("Invalid readiness path")
        elif self.path is not None or self.port is not None:
            raise ValueError("Only HTTP readiness declares path/port")
        if (self.kind == "adapter") != (self.adapter is not None):
            raise ValueError("Adapter readiness requires an adapter identity")
        return self


class RuntimeDeclaration(PackageModel):
    profile: Literal["command", "http", "native"]
    command: Annotated[tuple[str, ...], Field(min_length=1)]
    working_directory: str = "/"
    readiness: ReadinessDeclaration

    @model_validator(mode="after")
    def validate_command(self) -> RuntimeDeclaration:
        if not self.command[0] or any("\x00" in part for part in self.command):
            raise ValueError("Runtime command requires an executable and NUL-free argv")
        directory = PurePosixPath(self.working_directory)
        if (
            not directory.is_absolute()
            or ".." in directory.parts
            or str(directory) != self.working_directory
            or "\\" in self.working_directory
        ):
            raise ValueError("Runtime working directory must be canonical and absolute")
        return self


class ContractDeclaration(PackageModel):
    name: Name
    version: Name


class CredentialReference(PackageModel):
    name: Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
    reference: Annotated[str, StringConstraints(pattern=r"^secretref://[A-Za-z0-9._/-]+$")]


class DependencyIdentity(PackageModel):
    name: Name
    reference: Annotated[str, StringConstraints(min_length=1)]
    digest: Digest


class ContainerPackageManifest(PackageModel):
    schema_version: Literal["synth.container-package.v1"]
    package_id: Name
    build: BuildEntrypoint
    runtime: RuntimeDeclaration
    contracts: tuple[ContractDeclaration, ...] = ()
    credentials: tuple[CredentialReference, ...] = ()
    dependencies: tuple[DependencyIdentity, ...] = ()

    @model_validator(mode="after")
    def validate_names(self) -> ContainerPackageManifest:
        for declarations in (self.contracts, self.credentials, self.dependencies):
            names = [entry.name for entry in declarations]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate declaration identity")
        return self


class PackageFileIdentity(PackageModel):
    path: str
    digest: Digest
    size_bytes: Annotated[int, Field(strict=True, ge=0)]
    executable: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def validate_path(self) -> PackageFileIdentity:
        portable_path(self.path)
        return self


class PackageBuildPlan(PackageModel):
    schema_version: Literal["synth.container-build-plan.v1"]
    manifest: ContainerPackageManifest
    files: tuple[PackageFileIdentity, ...]
    directories: tuple[str, ...] = ()
    input_digest: Digest

    @model_validator(mode="after")
    def validate_identity(self) -> PackageBuildPlan:
        paths = [entry.path for entry in self.files]
        if paths != sorted(set(paths)) or not paths:
            raise ValueError("Build plan files must be nonempty, sorted and unique")
        if list(self.directories) != sorted(set(self.directories)):
            raise ValueError("Build plan directories must be sorted and unique")
        for directory in self.directories:
            portable_path(directory)
        if set(paths) & set(self.directories):
            raise ValueError("A package path cannot be both file and directory")
        all_paths = set(paths) | set(self.directories)
        if any(
            str(parent) in paths for path in all_paths for parent in PurePosixPath(path).parents
        ):
            raise ValueError("A package file cannot be a parent directory")
        entrypoint = next((f for f in self.files if f.path == self.manifest.build.entrypoint), None)
        if entrypoint is None or not entrypoint.executable:
            raise ValueError("Build entrypoint must be a retained executable file")
        if self.input_digest != content_digest(self.identity_payload()):
            raise ValueError("Build plan digest mismatch")
        return self

    def identity_payload(self) -> dict:
        return {
            "manifest": self.manifest.model_dump(mode="json"),
            "files": [entry.model_dump(mode="json") for entry in self.files],
            "directories": list(self.directories),
        }


class ImageArtifactIdentity(PackageModel):
    kind: Literal["oci_image"]
    digest_kind: Literal["oci_manifest", "oci_index", "oci_config"]
    reference: str
    digest: Digest
    platform: Platform

    @model_validator(mode="after")
    def validate_reference(self) -> ImageArtifactIdentity:
        if self.digest_kind == "oci_config":
            if self.reference != self.digest:
                raise ValueError("Local config identity must use its daemon digest")
        elif not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}", self.reference
        ) or not self.reference.endswith("@" + self.digest):
            raise ValueError("Registry identity requires a matching digest-pinned reference")
        return self


class PackageBuildOutput(PackageModel):
    schema_version: Literal["synth.container-build-output.v1"]
    action: Literal["build", "resolve"]
    input_digest: Digest
    artifact: ImageArtifactIdentity


class ContainerArtifactDescriptor(PackageModel):
    schema_version: Literal["synth.container-artifact.v1"]
    plan: PackageBuildPlan
    output: PackageBuildOutput
    artifact_verification: Literal["not_checked"] = "not_checked"
    runtime_validation: Literal["not_run"] = "not_run"

    @model_validator(mode="after")
    def validate_binding(self) -> ContainerArtifactDescriptor:
        if self.output.input_digest != self.plan.input_digest:
            raise ValueError("Build output belongs to different inputs")
        if self.output.artifact.platform != self.plan.manifest.build.platform:
            raise ValueError("Build output platform differs from requested platform")
        return self
