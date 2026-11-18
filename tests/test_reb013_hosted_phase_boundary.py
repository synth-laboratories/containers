"""REB 013 hosted phases: artifact reference and isolated verifier boundary (SYN-5685).

Provider-free. The phase requirements and digests are the retained REB 013
translation (SYN-5647 W04 receipt). The offers are fixtures for the checker;
no supplier is qualified here. Hosted 4 vCPU / 8 GiB / 20 GiB qualification
is the SYN-5654 lane.
"""

from __future__ import annotations

import pytest

from synth_containers.execution_requirements import (
    ArtifactVerification,
    ExecutionArtifact,
    ExecutionPlatform,
    ExecutionProvider,
    ExecutionRequirements,
    NetworkMode,
    OfferQualification,
    PreflightCode,
    ScoringOfferCapabilities,
    VerifierIsolation,
    preflight_execution,
)
from synth_containers.package_models import ImageArtifactIdentity

GIB = 1 << 30
REB013 = {
    "execution": ("sha256:638a115a666249efff29afd32732359e5bfb991b5acd6af3abcb2873fbfa4d5d", 1800, "self_reported"),
    "verifier": ("sha256:b019ab9f24105f712bc2e8f7201df15730ccd098b929b4ca321a38ab97744646", 3600, "isolated"),
}
CONFIG = "sha256:" + "8" * 64
MANIFEST = "sha256:" + "a" * 64


def phase(name: str) -> tuple[ExecutionRequirements, str]:
    digest, wall, isolation = REB013[name]
    declared = ExecutionRequirements.model_validate({
        "schema_version": "synth.container-execution-requirements.v1",
        "platform": "linux/amd64",
        "resources": {"cpu_millis": 4000, "memory_bytes": 8 * GIB, "disk_bytes": 20 * GIB,
                      "gpu_kind": "none", "gpu_count": 0},
        "max_wall_seconds": wall,
        "network": {"mode": "none", "hosts": []},
        "privileged": False,
        "verifier_isolation": isolation,
    })
    return declared, digest


def artifact(digest_kind: str) -> ExecutionArtifact:
    digest = CONFIG if digest_kind == "oci_config" else MANIFEST
    reference = digest if digest_kind == "oci_config" else "ghcr.io/synth-laboratories/reb-013@" + digest
    return ExecutionArtifact(
        identity=ImageArtifactIdentity(kind="oci_image", digest_kind=digest_kind, reference=reference,
                                       digest=digest, platform="linux/amd64"),
        verification=ArtifactVerification.VERIFIED,
        inspection_receipt_digest="sha256:" + "9" * 64,
    )


def offer(*, cpu: int, memory_gib: int, disk_gib: int, wall: int, verifier_modes) -> ScoringOfferCapabilities:
    return ScoringOfferCapabilities.model_validate({
        "provider": ExecutionProvider.MODAL,
        "offer_id": f"modal:{cpu}vcpu-{memory_gib}gib:us",
        "option_id": "sbo1_" + "5" * 16 + "_" + "6" * 64,
        "catalog_digest": "sha256:" + "1" * 64,
        "descriptor_digest": "sha256:" + "2" * 64,
        "tariff_revision": "sha256:" + "7" * 64,
        "adapter_revision": "sha256:" + "3" * 64,
        "qualification": OfferQualification.QUALIFIED,
        "release_admitted": True,
        "platforms": [ExecutionPlatform.LINUX_AMD64],
        "resources": {"cpu_millis": cpu * 1000, "memory_bytes": memory_gib * GIB,
                      "disk_bytes": disk_gib * GIB, "gpu_kind": "none", "gpu_count": 0},
        "max_wall_seconds": wall,
        "network_modes": [NetworkMode.NONE],
        "privileged": False,
        "network_policy_digest": "sha256:" + "4" * 64,
        "allowed_hosts": [],
        "verifier_modes": list(verifier_modes),
    })


RETAINED = offer(cpu=2, memory_gib=2, disk_gib=0, wall=1980, verifier_modes=[VerifierIsolation.SELF_REPORTED])
SIZED_SELF = offer(cpu=4, memory_gib=8, disk_gib=20, wall=3600, verifier_modes=[VerifierIsolation.SELF_REPORTED])
SIZED_ISOLATED = offer(cpu=4, memory_gib=8, disk_gib=20, wall=3600,
                       verifier_modes=[VerifierIsolation.SELF_REPORTED, VerifierIsolation.ISOLATED])


@pytest.mark.parametrize("name", sorted(REB013))
def test_retained_phase_digests_are_pinned(name):
    declared, digest = phase(name)
    assert declared.digest() == digest


@pytest.mark.parametrize("name", sorted(REB013))
@pytest.mark.parametrize("target", [RETAINED, SIZED_SELF, SIZED_ISOLATED], ids=["retained", "sized", "isolated"])
def test_daemon_config_identity_is_refused_before_any_capacity_check(name, target):
    declared, digest = phase(name)
    receipt = preflight_execution(declared, expected_digest=digest, artifact=artifact("oci_config"), offer=target)
    assert receipt.code is PreflightCode.ARTIFACT_REFERENCE_UNSUPPORTED


def test_registry_manifest_clears_reference_gate_and_stops_at_capacity():
    declared, digest = phase("execution")
    receipt = preflight_execution(declared, expected_digest=digest, artifact=artifact("oci_manifest"), offer=RETAINED)
    assert receipt.code is PreflightCode.RESOURCES_UNSUPPORTED


def test_sized_offer_admits_execution_but_self_reported_never_stands_in_for_isolation():
    execution, execution_digest = phase("execution")
    verifier, verifier_digest = phase("verifier")
    manifest = artifact("oci_manifest")
    assert preflight_execution(execution, expected_digest=execution_digest, artifact=manifest,
                               offer=SIZED_SELF).code is PreflightCode.SUPPORTED
    assert preflight_execution(verifier, expected_digest=verifier_digest, artifact=manifest,
                               offer=SIZED_SELF).code is PreflightCode.VERIFIER_ISOLATION_UNSUPPORTED
    assert preflight_execution(verifier, expected_digest=verifier_digest, artifact=manifest,
                               offer=SIZED_ISOLATED).code is PreflightCode.SUPPORTED


def test_pinned_digest_refuses_resized_phase():
    declared, _ = phase("verifier")
    _, execution_digest = phase("execution")
    receipt = preflight_execution(declared, expected_digest=execution_digest, artifact=artifact("oci_manifest"),
                                  offer=SIZED_ISOLATED)
    assert receipt.code is PreflightCode.REQUIREMENTS_DIGEST_MISMATCH
