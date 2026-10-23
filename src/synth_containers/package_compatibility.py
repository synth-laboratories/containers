"""Offline package declaration matching, separate from qualification/admission."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .package_models import (
    ContainerArtifactDescriptor,
    ContractDeclaration,
    Digest,
    Name,
    PackageModel,
    Platform,
    content_digest,
)


class PackageRuntimeSupport(PackageModel):
    """Runtime owner's explicit support policy, not package-supplied capabilities.

    Named contracts use their existing names/versions. Optimizer compatibility
    still uses RuntimeCapabilitySurface and evaluate_consumer_support after the
    runtime handshake; this policy does not introduce a second capability system.
    """

    schema_version: Literal["synth.package-runtime-support.v1"]
    runtime_id: Name
    platforms: tuple[Platform, ...]
    runtime_profiles: tuple[Literal["command", "http", "native"], ...]
    readiness_kinds: tuple[Literal["process", "http", "adapter"], ...]
    readiness_adapters: tuple[Name, ...] = ()
    contracts: tuple[ContractDeclaration, ...] = ()
    image_digest_kinds: tuple[Literal["oci_manifest", "oci_index", "oci_config"], ...] = (
        "oci_manifest", "oci_index",
    )

    @model_validator(mode="after")
    def validate_unique_declarations(self) -> PackageRuntimeSupport:
        for entries in (
            self.platforms, self.runtime_profiles, self.readiness_kinds,
            self.readiness_adapters, self.image_digest_kinds,
            tuple((c.name, c.version) for c in self.contracts),
        ):
            if len(entries) != len(set(entries)):
                raise ValueError("Duplicate runtime support declaration")
        if self.readiness_adapters and "adapter" not in self.readiness_kinds:
            raise ValueError("Readiness adapters require adapter readiness support")
        return self


class PackageCompatibilityIssue(PackageModel):
    code: Literal[
        "platform_unsupported", "runtime_profile_unsupported", "readiness_kind_unsupported",
        "readiness_adapter_unsupported", "contract_unsupported", "image_digest_kind_unsupported",
    ]
    field: str
    requested: str


class PackageCompatibilityReport(PackageModel):
    schema_version: Literal["synth.package-compatibility.v1"]
    artifact_descriptor_digest: Digest
    runtime_support_digest: Digest
    runtime_id: Name
    declaration_supported: bool
    issues: tuple[PackageCompatibilityIssue, ...]
    artifact_verification: Literal["not_checked"] = "not_checked"
    runtime_qualification: Literal["not_run"] = "not_run"
    admission: Literal["not_checked"] = "not_checked"
    readiness: Literal["not_observed"] = "not_observed"

    @model_validator(mode="after")
    def validate_result(self) -> PackageCompatibilityReport:
        if self.declaration_supported != (not self.issues):
            raise ValueError("Declaration support must agree with compatibility issues")
        return self


def check_package_compatibility(
    artifact: ContainerArtifactDescriptor,
    support: PackageRuntimeSupport,
) -> PackageCompatibilityReport:
    """Check frozen declarations without starting code or making provider calls.

    A successful check cannot authorize execution or certify that the image exists.
    Runtime readiness, qualification, credential binding and resource grants are
    checked by their owners at execution admission, against these pinned identities.
    """
    # Revalidate even programmatically constructed/copied instances at this boundary.
    artifact = ContainerArtifactDescriptor.model_validate(artifact)
    support = PackageRuntimeSupport.model_validate(support)
    manifest, image = artifact.plan.manifest, artifact.output.artifact
    issues: list[PackageCompatibilityIssue] = []

    def require(ok: bool, code: str, field: str, requested: str) -> None:
        if not ok:
            issues.append(PackageCompatibilityIssue.model_validate({
                "code": code, "field": field, "requested": requested,
            }))

    require(image.platform in support.platforms,
            "platform_unsupported", "output.artifact.platform", image.platform)
    require(image.digest_kind in support.image_digest_kinds,
            "image_digest_kind_unsupported", "output.artifact.digest_kind", image.digest_kind)
    runtime = manifest.runtime
    require(runtime.profile in support.runtime_profiles,
            "runtime_profile_unsupported", "plan.manifest.runtime.profile", runtime.profile)
    require(runtime.readiness.kind in support.readiness_kinds,
            "readiness_kind_unsupported", "plan.manifest.runtime.readiness.kind", runtime.readiness.kind)
    if runtime.readiness.adapter is not None:
        require(runtime.readiness.adapter in support.readiness_adapters,
                "readiness_adapter_unsupported", "plan.manifest.runtime.readiness.adapter",
                runtime.readiness.adapter)
    contracts = {(c.name, c.version) for c in support.contracts}
    for index, contract in enumerate(manifest.contracts):
        require((contract.name, contract.version) in contracts,
                "contract_unsupported", f"plan.manifest.contracts.{index}",
                f"{contract.name}@{contract.version}")
    return PackageCompatibilityReport(
        schema_version="synth.package-compatibility.v1",
        artifact_descriptor_digest=content_digest(artifact.model_dump(mode="json")),
        runtime_support_digest=content_digest(support.model_dump(mode="json")),
        runtime_id=support.runtime_id,
        declaration_supported=not issues,
        issues=tuple(issues),
    )
