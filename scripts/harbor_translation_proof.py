"""Translate exact Harbor bundle sources and run the canonical execution preflight.

Usage:
    python scripts/harbor_translation_proof.py BUNDLE POLICY OUTPUT_DIR --source-ref REF

The bundle must be an exported tree at an exact source commit (for example
`git archive <sha> <path>`). OUTPUT_DIR receives the emitted manifest,
requirements, translated Dockerfile, build script, translation receipt and
`preflight.json`; the copied context is left in a temporary directory.

Both phases are preflighted on their own: the packaged training container
(self-reported training scores) and the unpackaged held-out verifier phase.
The offers are fixed fixture capability packets written independently of the
bundle, not derived from its requirements. They are labelled as fixtures: no
live producer of `ScoringOfferCapabilities` exists yet, so no live supplier,
adapter, image or egress policy is qualified by this script. No build,
provider call or paid execution happens.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from synth_containers.execution_package import plan_execution_package
from synth_containers.execution_requirements import (
    ArtifactVerification,
    ExecutionArtifact,
    ExecutionRequirements,
    NetworkMode,
    OfferQualification,
    ScoringOfferCapabilities,
    VerifierIsolation,
    preflight_execution,
)
from synth_containers.harbor_translation import load_translation_policy, translate_harbor_bundle
from synth_containers.package_models import ImageArtifactIdentity

FIXTURE_LABEL = (
    "fixture capability packets; no live ScoringOfferCapabilities producer exists, "
    "so no supplier, adapter, image or egress policy is qualified"
)
COPIED = ("container.toml", "execution-requirements.toml", "Dockerfile", "build.sh")
GIB = 1 << 30


def fixture_offer(
    offer_id: str, verifier_mode: VerifierIsolation, max_wall_seconds: int, **changes: object
) -> ScoringOfferCapabilities:
    """A fixed amd64 4 vCPU / 8 GiB / 20 GiB, no-network, CPU-only offer shape."""
    fields: dict[str, object] = {
        "provider": "daytona",
        "offer_id": offer_id,
        "option_id": "sbo1_" + "0" * 16 + "_" + "0" * 64,
        "catalog_digest": "sha256:" + "1" * 64,
        "descriptor_digest": "sha256:" + "2" * 64,
        "tariff_revision": "sha256:" + "3" * 64,
        "adapter_revision": "sha256:" + "4" * 64,
        "qualification": OfferQualification.QUALIFIED,
        "release_admitted": True,
        "platforms": ["linux/amd64"],
        "resources": {
            "cpu_millis": 4000,
            "memory_bytes": 8 * GIB,
            "disk_bytes": 20 * GIB,
            "gpu_kind": "none",
            "gpu_count": 0,
        },
        "max_wall_seconds": max_wall_seconds,
        "network_modes": [NetworkMode.NONE],
        "network_policy_digest": "sha256:" + "5" * 64,
        "allowed_hosts": [],
        "verifier_modes": [verifier_mode],
        "privileged": False,
    }
    fields.update(changes)
    return ScoringOfferCapabilities.model_validate(fields)


TRAINING_OFFER = fixture_offer(
    "fixture:amd64-4c-8g-20g-1h-nonet-self-reported", VerifierIsolation.SELF_REPORTED, 3600
)
VERIFIER_OFFER = fixture_offer(
    "fixture:amd64-4c-8g-20g-2h-nonet-isolated", VerifierIsolation.ISOLATED, 7200
)


def fixture_artifact(platform: str) -> ExecutionArtifact:
    digest = "sha256:" + "a" * 64
    return ExecutionArtifact(
        identity=ImageArtifactIdentity(
            kind="oci_image",
            digest_kind="oci_manifest",
            reference="registry.invalid/fixture@" + digest,
            digest=digest,
            platform=platform,
        ),
        verification=ArtifactVerification.VERIFIED,
        inspection_receipt_digest="sha256:" + "b" * 64,
    )


def run_cases(
    requirements: ExecutionRequirements, digest: str, offer: ScoringOfferCapabilities
) -> dict[str, dict[str, object]]:
    other_modes = [
        mode for mode in VerifierIsolation if mode is not requirements.verifier_isolation
    ]
    cases = {
        "supported": (fixture_artifact("linux/amd64"), offer),
        "artifact_platform_mismatch": (fixture_artifact("linux/arm64"), offer),
        "verifier_isolation_unavailable": (
            fixture_artifact("linux/amd64"),
            offer.model_copy(update={"verifier_modes": tuple(other_modes)}),
        ),
        "network_mode_unenforced": (
            fixture_artifact("linux/amd64"),
            offer.model_copy(update={"network_modes": (NetworkMode.ALLOWLIST,)}),
        ),
        "wall_time_cap_below_required": (
            fixture_artifact("linux/amd64"),
            offer.model_copy(update={"max_wall_seconds": requirements.max_wall_seconds - 1}),
        ),
        "offer_unqualified": (
            fixture_artifact("linux/amd64"),
            offer.model_copy(update={"qualification": OfferQualification.DRAFT}),
        ),
    }
    return {
        name: preflight_execution(
            requirements, expected_digest=digest, artifact=art, offer=off
        ).model_dump(mode="json")
        for name, (art, off) in cases.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("policy", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-ref", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as scratch:
        package = Path(scratch) / "package"
        receipt, plan = translate_harbor_bundle(
            args.bundle, package, load_translation_policy(args.policy)
        )
        replanned = plan_execution_package(package / "container.toml")
        assert replanned == plan, "re-planning the emitted package changed its identity"
        for name in COPIED:
            shutil.copy2(package / name, args.output / name)
        (args.output / "translation-receipt.json").write_text(
            receipt.model_dump_json(indent=2) + "\n"
        )
        (args.output / "execution-plan.json").write_text(plan.model_dump_json(indent=2) + "\n")
    phases = {
        "training": run_cases(
            plan.execution_requirements, plan.execution_requirements_digest, TRAINING_OFFER
        ),
        "verifier": run_cases(
            receipt.verifier_requirements, receipt.verifier_requirements_digest, VERIFIER_OFFER
        ),
    }
    summary = {
        "schema_version": "synth.harbor-translation-proof.v1",
        "label": FIXTURE_LABEL,
        "source_ref": args.source_ref,
        "bundle_id": receipt.bundle_id,
        "source_dialect": receipt.source_dialect.value,
        "translation_receipt_digest": receipt.digest(),
        "build_input_digest": plan.build_plan.input_digest,
        "execution_requirements_digest": plan.execution_requirements_digest,
        "verifier_requirements_digest": receipt.verifier_requirements_digest,
        "offers": {
            "training": TRAINING_OFFER.model_dump(mode="json"),
            "verifier": VERIFIER_OFFER.model_dump(mode="json"),
        },
        "packaged_paths": [entry.path for entry in plan.build_plan.files],
        "excluded_paths": [entry.path for entry in receipt.excluded_inputs],
        "preflight": {
            phase: {name: (r["result"], r["code"]) for name, r in results.items()}
            for phase, results in phases.items()
        },
        "preflight_receipts": phases,
    }
    (args.output / "preflight.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("bundle_id", "preflight")}, indent=2))
    ok = all(
        results["supported"][0] == "supported"
        and all(
            result != "supported" for name, (result, _) in results.items() if name != "supported"
        )
        for results in summary["preflight"].values()
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
