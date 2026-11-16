"""Package requirements and exact provider preflight; no provider effects.

Requirements describe customer work. Offer capabilities must come from the
authenticated placement owner, never from the package or caller. A supported
preflight is neither supplier qualification nor a financial reservation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from .execution_limits import (
    MAXIMUM_CPU_MILLIS,
    MAXIMUM_DISK_BYTES,
    MAXIMUM_GPU_COUNT,
    MAXIMUM_MEMORY_BYTES,
    MAXIMUM_NETWORK_HOSTS,
    MAXIMUM_WALL_SECONDS,
)
from .package_models import Digest, ImageArtifactIdentity, PackageModel, content_digest

__all__ = [
    "ArtifactVerification",
    "ExecutionArtifact",
    "ExecutionNetwork",
    "ExecutionPlatform",
    "ExecutionPreflight",
    "ExecutionProvider",
    "ExecutionRequirements",
    "ExecutionResources",
    "GPUKind",
    "NetworkMode",
    "OfferQualification",
    "PreflightCode",
    "PreflightResult",
    "ScoringOfferCapabilities",
    "VerifierIsolation",
    "preflight_execution",
]


class ExecutionPlatform(StrEnum):
    LINUX_AMD64 = "linux/amd64"
    LINUX_ARM64 = "linux/arm64"


class ExecutionProvider(StrEnum):
    DOCKER = "docker"
    MODAL = "modal"
    DAYTONA = "daytona"
    EXE_DEV = "exe_dev"
    BOAT_DEV = "boat_dev"


class NetworkMode(StrEnum):
    NONE = "none"
    ALLOWLIST = "allowlist"


class GPUKind(StrEnum):
    NONE = "none"
    T4 = "t4"
    L4 = "l4"
    H100 = "h100"


class VerifierIsolation(StrEnum):
    SELF_REPORTED = "self_reported"
    SHARED = "shared"
    ISOLATED = "isolated"


class ArtifactVerification(StrEnum):
    NOT_CHECKED = "not_checked"
    VERIFIED = "verified"


class ExecutionArtifact(PackageModel):
    identity: ImageArtifactIdentity
    verification: ArtifactVerification
    # Absent means no trusted inspection has been performed, never success.
    inspection_receipt_digest: Digest | None

    @model_validator(mode="after")
    def explicit_inspection_authority(self) -> Self:
        if (self.verification is ArtifactVerification.VERIFIED) != (
            self.inspection_receipt_digest is not None
        ):
            raise ValueError("Verified image identity requires its trusted inspection receipt")
        return self


Host = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=253,
        pattern=r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$",
    ),
]


class ExecutionNetwork(PackageModel):
    mode: NetworkMode
    hosts: Annotated[tuple[Host, ...], Field(max_length=MAXIMUM_NETWORK_HOSTS)]

    @model_validator(mode="after")
    def explicit_access(self) -> Self:
        if tuple(sorted(set(self.hosts))) != self.hosts:
            raise ValueError("Network hosts must be sorted and distinct")
        if (self.mode is NetworkMode.NONE) != (len(self.hosts) == 0):
            raise ValueError("No-network requires no hosts; allow-list requires named hosts")
        return self


class ExecutionResources(PackageModel):
    cpu_millis: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_CPU_MILLIS)]
    memory_bytes: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_MEMORY_BYTES)]
    disk_bytes: Annotated[int, Field(strict=True, ge=0, le=MAXIMUM_DISK_BYTES)]
    gpu_kind: GPUKind
    gpu_count: Annotated[int, Field(strict=True, ge=0, le=MAXIMUM_GPU_COUNT)]

    @model_validator(mode="after")
    def paired_gpu(self) -> Self:
        if (self.gpu_kind is GPUKind.NONE) != (self.gpu_count == 0):
            raise ValueError("GPU kind and count must identify the same requirement")
        return self


class ExecutionRequirements(PackageModel):
    schema_version: Literal["synth.container-execution-requirements.v1"]
    platform: ExecutionPlatform
    resources: ExecutionResources
    max_wall_seconds: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_WALL_SECONDS)]
    network: ExecutionNetwork
    privileged: Annotated[bool, Field(strict=True)]
    verifier_isolation: VerifierIsolation

    def digest(self) -> str:
        """SHA256 of sorted compact UTF-8 JSON, without a trailing newline."""
        return content_digest(self.model_dump(mode="json"))


class OfferQualification(StrEnum):
    DRAFT = "draft"
    QUALIFIED = "qualified"


class ScoringOfferCapabilities(PackageModel):
    """Authenticated owner readback, including the actual adapter revision.

    Resource values are upper bounds for the exact offer, not a provider-wide
    promise. Network modes name enforced adapter behavior. Listing a hostname
    in the package cannot grant an unsupported network mode to the adapter.
    """

    provider: ExecutionProvider
    offer_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    option_id: Annotated[str, StringConstraints(pattern=r"^sbo1_[0-9a-f]{16}_[0-9a-f]{64}$")]
    catalog_digest: Digest
    descriptor_digest: Digest
    tariff_revision: Digest
    adapter_revision: Digest
    qualification: OfferQualification
    release_admitted: Annotated[bool, Field(strict=True)]
    platforms: Annotated[tuple[ExecutionPlatform, ...], Field(min_length=1, max_length=2)]
    resources: ExecutionResources
    max_wall_seconds: Annotated[int, Field(strict=True, ge=1, le=MAXIMUM_WALL_SECONDS)]
    network_modes: Annotated[tuple[NetworkMode, ...], Field(min_length=1, max_length=2)]
    network_policy_digest: Digest
    allowed_hosts: Annotated[tuple[Host, ...], Field(max_length=MAXIMUM_NETWORK_HOSTS)]
    verifier_modes: Annotated[tuple[VerifierIsolation, ...], Field(min_length=1, max_length=3)]
    privileged: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def distinct_capabilities(self) -> Self:
        if len(set(self.platforms)) != len(self.platforms):
            raise ValueError("Duplicate execution platform")
        if len(set(self.network_modes)) != len(self.network_modes):
            raise ValueError("Duplicate enforced network mode")
        if tuple(sorted(set(self.allowed_hosts))) != self.allowed_hosts:
            raise ValueError("Authorized network hosts must be sorted and distinct")
        if len(set(self.verifier_modes)) != len(self.verifier_modes):
            raise ValueError("Duplicate verifier isolation mode")
        return self


class PreflightResult(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNQUALIFIED = "unqualified"
    BLOCKED = "blocked"


class PreflightCode(StrEnum):
    SUPPORTED = "execution_supported"
    REQUIREMENTS_DIGEST_MISMATCH = "execution_requirements_digest_mismatch"
    ARTIFACT_UNVERIFIED = "execution_artifact_unverified"
    ARTIFACT_PLATFORM_MISMATCH = "execution_artifact_platform_mismatch"
    OFFER_NOT_WHITELISTED = "execution_offer_not_whitelisted"
    OFFER_UNQUALIFIED = "execution_offer_unqualified"
    PLATFORM_UNSUPPORTED = "execution_platform_unsupported"
    RESOURCES_UNSUPPORTED = "execution_resources_unsupported"
    WALL_TIME_UNSUPPORTED = "execution_wall_time_unsupported"
    NETWORK_UNSUPPORTED = "execution_network_unsupported"
    NETWORK_HOST_NOT_ALLOWED = "execution_network_host_not_allowed"
    VERIFIER_ISOLATION_UNSUPPORTED = "execution_verifier_isolation_unsupported"
    PRIVILEGE_UNSUPPORTED = "execution_privilege_unsupported"


class ExecutionPreflight(PackageModel):
    schema_version: Literal["synth.container-execution-preflight.v1"]
    result: PreflightResult
    code: PreflightCode
    cause: str
    stage: Literal["scoring_preflight"]
    requirements_digest: Digest
    requested_requirements_digest: Digest
    provider: ExecutionProvider
    offer_id: str
    option_id: Annotated[str, StringConstraints(pattern=r"^sbo1_[0-9a-f]{16}_[0-9a-f]{64}$")]
    catalog_digest: Digest
    descriptor_digest: Digest
    tariff_revision: Digest
    adapter_revision: Digest
    network_policy_digest: Digest
    artifact_digest: Digest
    artifact_platform: str
    # Present only when a trusted artifact inspection was performed.
    artifact_inspection_receipt_digest: Digest | None


def preflight_execution(
    requirements: ExecutionRequirements,
    *,
    expected_digest: str,
    artifact: ExecutionArtifact,
    offer: ScoringOfferCapabilities,
) -> ExecutionPreflight:
    """Refuse unsupported requirements without substitution or provider I/O."""
    digest = requirements.digest()
    result, code = PreflightResult.SUPPORTED, PreflightCode.SUPPORTED
    cause = "All declared requirements fit the exact qualified release offer"
    if digest != expected_digest:
        result, code = PreflightResult.BLOCKED, PreflightCode.REQUIREMENTS_DIGEST_MISMATCH
        cause = f"Requirements digest {digest} differs from retained {expected_digest}"
    elif artifact.verification is not ArtifactVerification.VERIFIED:
        result, code = PreflightResult.BLOCKED, PreflightCode.ARTIFACT_UNVERIFIED
        cause = f"Image {artifact.identity.digest} has no trusted platform inspection receipt"
    elif artifact.identity.platform != requirements.platform:
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.ARTIFACT_PLATFORM_MISMATCH
        cause = f"Inspected image platform {artifact.identity.platform} differs from required {requirements.platform}"
    elif not offer.release_admitted:
        result, code = PreflightResult.BLOCKED, PreflightCode.OFFER_NOT_WHITELISTED
        cause = f"Offer {offer.offer_id} is outside the environment release allow-list"
    elif offer.qualification is not OfferQualification.QUALIFIED:
        result, code = PreflightResult.UNQUALIFIED, PreflightCode.OFFER_UNQUALIFIED
        cause = f"Offer {offer.offer_id} has no qualified adapter/capability packet"
    elif requirements.platform not in offer.platforms:
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.PLATFORM_UNSUPPORTED
        cause = f"Platform {requirements.platform} is unsupported by offer {offer.offer_id}"
    elif (
        requirements.resources.cpu_millis > offer.resources.cpu_millis
        or requirements.resources.memory_bytes > offer.resources.memory_bytes
        or requirements.resources.disk_bytes > offer.resources.disk_bytes
        or requirements.resources.gpu_kind is not offer.resources.gpu_kind
        or requirements.resources.gpu_count > offer.resources.gpu_count
    ):
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.RESOURCES_UNSUPPORTED
        cause = f"Requested resources exceed or differ from exact offer {offer.offer_id}"
    elif requirements.max_wall_seconds > offer.max_wall_seconds:
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.WALL_TIME_UNSUPPORTED
        cause = f"Requested wall time {requirements.max_wall_seconds}s exceeds {offer.max_wall_seconds}s"
    elif requirements.network.mode not in offer.network_modes:
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.NETWORK_UNSUPPORTED
        cause = f"Adapter {offer.adapter_revision} does not enforce {requirements.network.mode} networking"
    elif any(host not in offer.allowed_hosts for host in requirements.network.hosts):
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.NETWORK_HOST_NOT_ALLOWED
        denied_hosts = tuple(
            host for host in requirements.network.hosts if host not in offer.allowed_hosts
        )
        cause = f"Network policy {offer.network_policy_digest} refuses hosts {denied_hosts}"
    elif requirements.verifier_isolation not in offer.verifier_modes:
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.VERIFIER_ISOLATION_UNSUPPORTED
        cause = f"Adapter {offer.adapter_revision} does not enforce {requirements.verifier_isolation} verifier isolation"
    elif requirements.privileged and not offer.privileged:
        result, code = PreflightResult.UNSUPPORTED, PreflightCode.PRIVILEGE_UNSUPPORTED
        cause = f"Offer {offer.offer_id} cannot enforce the requested privilege contract"
    assert (result is PreflightResult.SUPPORTED) == (code is PreflightCode.SUPPORTED)
    return ExecutionPreflight(
        schema_version="synth.container-execution-preflight.v1",
        result=result,
        code=code,
        cause=cause,
        stage="scoring_preflight",
        requirements_digest=digest,
        requested_requirements_digest=expected_digest,
        provider=offer.provider,
        offer_id=offer.offer_id,
        option_id=offer.option_id,
        catalog_digest=offer.catalog_digest,
        descriptor_digest=offer.descriptor_digest,
        tariff_revision=offer.tariff_revision,
        adapter_revision=offer.adapter_revision,
        network_policy_digest=offer.network_policy_digest,
        artifact_digest=artifact.identity.digest,
        artifact_platform=artifact.identity.platform,
        artifact_inspection_receipt_digest=artifact.inspection_receipt_digest,
    )
