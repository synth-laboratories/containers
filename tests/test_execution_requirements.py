"""Preflight refusals preserve requirements and exact execution authority."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from synth_containers.execution_requirements import (
    ArtifactVerification,
    ExecutionArtifact,
    ExecutionNetwork,
    ExecutionPlatform,
    ExecutionProvider,
    ExecutionRequirements,
    ExecutionResources,
    GPUKind,
    NetworkMode,
    OfferQualification,
    PreflightCode,
    PreflightResult,
    ScoringOfferCapabilities,
    VerifierIsolation,
    preflight_execution,
)
from synth_containers.package_models import ImageArtifactIdentity


def requirements(**changes: object) -> ExecutionRequirements:
    fields = {
        "schema_version": "synth.container-execution-requirements.v1",
        "platform": "linux/amd64",
        "resources": {
            "cpu_millis": 2000,
            "memory_bytes": 4 << 30,
            "disk_bytes": 0,
            "gpu_kind": "none",
            "gpu_count": 0,
        },
        "max_wall_seconds": 600,
        "network": {"mode": "none", "hosts": []},
        "privileged": False,
        "verifier_isolation": "self_reported",
    }
    fields.update(changes)
    return ExecutionRequirements.model_validate(fields)


def offer(**changes: object) -> ScoringOfferCapabilities:
    # Fixture authority tests the checker only; no real supplier is qualified.
    fields = {
        "provider": ExecutionProvider.MODAL,
        "offer_id": "modal:2vcpu-4gib:us",
        "option_id": "sbo1_" + "5" * 16 + "_" + "6" * 64,
        "catalog_digest": "sha256:" + "1" * 64,
        "descriptor_digest": "sha256:" + "2" * 64,
        "tariff_revision": "sha256:" + "7" * 64,
        "adapter_revision": "sha256:" + "3" * 64,
        "qualification": OfferQualification.QUALIFIED,
        "release_admitted": True,
        "platforms": [ExecutionPlatform.LINUX_AMD64],
        "resources": {
            "cpu_millis": 2000,
            "memory_bytes": 4 << 30,
            "disk_bytes": 0,
            "gpu_kind": "none",
            "gpu_count": 0,
        },
        "max_wall_seconds": 600,
        "network_modes": [NetworkMode.NONE],
        "privileged": False,
        "network_policy_digest": "sha256:" + "4" * 64,
        "allowed_hosts": [],
        "verifier_modes": [VerifierIsolation.SELF_REPORTED],
    }
    fields.update(changes)
    return ScoringOfferCapabilities.model_validate(fields)


def artifact(*, platform: ExecutionPlatform = ExecutionPlatform.LINUX_AMD64) -> ExecutionArtifact:
    digest = "sha256:" + "8" * 64
    return ExecutionArtifact(
        identity=ImageArtifactIdentity(
            kind="oci_image",
            digest_kind="oci_manifest",
            reference="registry.example/scorer@" + digest,
            digest=digest,
            platform=platform,
        ),
        verification=ArtifactVerification.VERIFIED,
        inspection_receipt_digest="sha256:" + "9" * 64,
    )


def test_digest_matches_existing_sorted_compact_package_contract() -> None:
    declared = requirements()
    wire = declared.model_dump(mode="json")
    canonical = json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert not canonical.endswith(b"\n")
    assert declared.digest() == "sha256:" + hashlib.sha256(canonical).hexdigest()
    reordered = dict(reversed(tuple(wire.items())))
    assert ExecutionRequirements.model_validate(reordered).digest() == declared.digest()


def test_supported_receipt_binds_requirements_catalog_offer_and_adapter() -> None:
    declared, selected = requirements(), offer()
    receipt = preflight_execution(
        declared,
        expected_digest=declared.digest(),
        artifact=artifact(platform=declared.platform),
        offer=selected,
    )
    assert receipt.result is PreflightResult.SUPPORTED
    assert receipt.code is PreflightCode.SUPPORTED
    assert receipt.requirements_digest == declared.digest()
    assert receipt.catalog_digest == selected.catalog_digest
    assert receipt.descriptor_digest == selected.descriptor_digest
    assert receipt.adapter_revision == selected.adapter_revision
    assert receipt.network_policy_digest == selected.network_policy_digest
    assert receipt.offer_id == selected.offer_id
    assert receipt.option_id == selected.option_id
    assert receipt.tariff_revision == selected.tariff_revision


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"platform": "linux/arm64"}, PreflightCode.PLATFORM_UNSUPPORTED),
        ({"max_wall_seconds": 601}, PreflightCode.WALL_TIME_UNSUPPORTED),
        (
            {"network": {"mode": "allowlist", "hosts": ["api.groq.com"]}},
            PreflightCode.NETWORK_UNSUPPORTED,
        ),
        ({"privileged": True}, PreflightCode.PRIVILEGE_UNSUPPORTED),
        ({"verifier_isolation": "isolated"}, PreflightCode.VERIFIER_ISOLATION_UNSUPPORTED),
        (
            {
                "resources": {
                    "cpu_millis": 2001,
                    "memory_bytes": 4 << 30,
                    "disk_bytes": 0,
                    "gpu_kind": "none",
                    "gpu_count": 0,
                }
            },
            PreflightCode.RESOURCES_UNSUPPORTED,
        ),
    ],
)
def test_qualified_offer_reaches_semantic_refusal(
    changes: dict[str, object], code: PreflightCode
) -> None:
    declared = requirements(**changes)
    receipt = preflight_execution(
        declared,
        expected_digest=declared.digest(),
        artifact=artifact(platform=declared.platform),
        offer=offer(),
    )
    assert receipt.result is PreflightResult.UNSUPPORTED
    assert receipt.code is code
    assert receipt.cause
    assert receipt.requirements_digest == declared.digest()


def test_draft_qualification_is_not_a_success() -> None:
    declared = requirements()
    receipt = preflight_execution(
        declared,
        expected_digest=declared.digest(),
        artifact=artifact(platform=declared.platform),
        offer=offer(qualification=OfferQualification.DRAFT),
    )
    assert receipt.result is PreflightResult.UNQUALIFIED
    assert receipt.code is PreflightCode.OFFER_UNQUALIFIED


def test_each_requested_host_needs_the_environment_policy_authority() -> None:
    declared = requirements(network={"mode": "allowlist", "hosts": ["api.groq.com", "modal.com"]})
    selected = offer(network_modes=[NetworkMode.ALLOWLIST], allowed_hosts=["api.groq.com"])
    receipt = preflight_execution(
        declared,
        expected_digest=declared.digest(),
        artifact=artifact(platform=declared.platform),
        offer=selected,
    )
    assert receipt.result is PreflightResult.UNSUPPORTED
    assert receipt.code is PreflightCode.NETWORK_HOST_NOT_ALLOWED
    assert "modal.com" in receipt.cause
    assert receipt.network_policy_digest == selected.network_policy_digest


def test_authorized_hosts_are_preserved_without_permission_widening() -> None:
    declared = requirements(network={"mode": "allowlist", "hosts": ["api.groq.com"]})
    selected = offer(network_modes=[NetworkMode.ALLOWLIST], allowed_hosts=["api.groq.com"])
    receipt = preflight_execution(
        declared,
        expected_digest=declared.digest(),
        artifact=artifact(platform=declared.platform),
        offer=selected,
    )
    assert receipt.result is PreflightResult.SUPPORTED
    assert declared.network.hosts == ("api.groq.com",)


def test_separate_is_not_silently_normalized_to_shared_or_isolated() -> None:
    with pytest.raises(ValidationError):
        requirements(verifier_isolation="separate")


def test_off_list_offer_is_blocked_before_any_provider_effect() -> None:
    declared = requirements()
    receipt = preflight_execution(
        declared,
        expected_digest=declared.digest(),
        artifact=artifact(platform=declared.platform),
        offer=offer(release_admitted=False),
    )
    assert receipt.result is PreflightResult.BLOCKED
    assert receipt.code is PreflightCode.OFFER_NOT_WHITELISTED


def test_changed_requirements_never_reuse_an_old_preflight_identity() -> None:
    original, changed = requirements(), requirements(max_wall_seconds=599)
    receipt = preflight_execution(
        changed,
        expected_digest=original.digest(),
        artifact=artifact(platform=changed.platform),
        offer=offer(),
    )
    assert receipt.result is PreflightResult.BLOCKED
    assert receipt.code is PreflightCode.REQUIREMENTS_DIGEST_MISMATCH
    assert receipt.requested_requirements_digest == original.digest()
    assert receipt.requirements_digest == changed.digest()


def test_actual_arm64_artifact_cannot_be_relabelled_as_required_amd64() -> None:
    declared = requirements()
    receipt = preflight_execution(
        declared, expected_digest=declared.digest(),
        artifact=artifact(platform=ExecutionPlatform.LINUX_ARM64), offer=offer(),
    )
    assert receipt.result is PreflightResult.UNSUPPORTED
    assert receipt.code is PreflightCode.ARTIFACT_PLATFORM_MISMATCH
    assert receipt.artifact_platform == "linux/arm64"
    assert declared.platform is ExecutionPlatform.LINUX_AMD64


def test_uninspected_artifact_never_counts_as_ready() -> None:
    declared = requirements()
    inspected = artifact()
    uninspected = ExecutionArtifact(
        identity=inspected.identity, verification=ArtifactVerification.NOT_CHECKED,
        inspection_receipt_digest=None,
    )
    receipt = preflight_execution(
        declared, expected_digest=declared.digest(), artifact=uninspected, offer=offer(),
    )
    assert receipt.result is PreflightResult.BLOCKED
    assert receipt.code is PreflightCode.ARTIFACT_UNVERIFIED
    assert receipt.artifact_inspection_receipt_digest is None


@pytest.mark.parametrize(
    "wire",
    [
        {"mode": "none", "hosts": ["api.groq.com"]},
        {"mode": "allowlist", "hosts": []},
        {"mode": "allowlist", "hosts": ["z.example", "a.example"]},
        {"mode": "allowlist", "hosts": ["api.groq.com", "api.groq.com"]},
        {"mode": "allowlist", "hosts": ["*.example"]},
    ],
)
def test_network_contract_never_normalizes_invalid_access(wire: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ExecutionNetwork.model_validate(wire)


def test_resources_reject_unpaired_gpu_and_boolean_cpu() -> None:
    for cpu, kind, count in [
        (True, GPUKind.NONE, 0),
        (2000, GPUKind.T4, 0),
        (2000, GPUKind.NONE, 1),
    ]:
        with pytest.raises(ValidationError):
            ExecutionResources(
                cpu_millis=cpu, memory_bytes=4 << 30, disk_bytes=0, gpu_kind=kind, gpu_count=count
            )


def test_requirements_are_frozen_and_unknown_fields_refuse() -> None:
    declared = requirements()
    with pytest.raises(ValidationError):
        declared.max_wall_seconds = 1
    with pytest.raises(ValidationError):
        requirements(provider_substitution=True)


@pytest.mark.parametrize("provider", [ExecutionProvider.MODAL, ExecutionProvider.DOCKER])
def test_local_daemon_image_is_not_a_modal_registry_artifact(provider):
    declared = requirements()
    digest = "sha256:" + "8" * 64
    local = ExecutionArtifact(
        identity=ImageArtifactIdentity(kind="oci_image", digest_kind="oci_config",
                                      reference=digest, digest=digest, platform="linux/amd64"),
        verification=ArtifactVerification.VERIFIED,
        inspection_receipt_digest="sha256:" + "9" * 64,
    )
    receipt = preflight_execution(declared, expected_digest=declared.digest(), artifact=local,
                                 offer=offer(provider=provider))
    assert receipt.code is (
        PreflightCode.ARTIFACT_REFERENCE_UNSUPPORTED if provider is ExecutionProvider.MODAL
        else PreflightCode.SUPPORTED
    )
