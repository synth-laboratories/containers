"""The held-out verifier phase of a translated Harbor package (SYN-5685).

A Harbor verifier never runs in the candidate's container. The translation
receipt already freezes what the task declares about it: its entrypoint, its
isolation, network and wall time (`verifier_requirements`), and the hidden
verifier inputs. This module derives the single verifier phase a scoring host
executes from that receipt, and binds an uploaded phase back to it.

Nothing here is chosen by the uploader: image lineage, command, wall time,
network, the submission files and the result file all follow from the
receipt, the frozen package plan and the verifier image's own config. A
phase that differs from the derivation is a typed refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, ValidationError

from .execution_package import ExecutionPackagePlan
from .execution_requirements import ExecutionRequirements, VerifierIsolation
from .harbor_translation import HarborDialect, HarborTranslationReceipt
from .package_models import Digest, ImageArtifactIdentity, PackageModel

PHASE_SCHEMA = "synth.container-verifier-phase.v1"
PHASE_FILENAME = "verifier-phase.json"
# Harbor uploads tests/ to /tests for verification and reads the reward here.
VERIFIER_TESTS_ROOT = "/tests"
VERIFIER_RESULT_PATH = "/logs/verifier/reward.txt"
VERIFIER_SUBMISSION_ROOT = "/submission"
MAX_SUBMISSION_BYTES = 16 * 1024 * 1024
# The verifier image is built FROM the candidate image (OCI base-image annotation).
BASE_DIGEST_LABEL = "org.opencontainers.image.base.digest"

SandboxPath = Annotated[
    str,
    StringConstraints(
        min_length=2, max_length=512, pattern=r"^/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+$"
    ),
]


class SubmissionFile(PackageModel):
    """One final candidate file copied into the verifier sandbox, and nothing else."""

    candidate_path: SandboxPath
    verifier_path: SandboxPath
    max_bytes: Annotated[int, Field(strict=True, ge=1, le=MAX_SUBMISSION_BYTES)]


class VerifierPhase(PackageModel):
    """The held-out verifier: its own image, requirements, command and result file."""

    schema_version: Literal["synth.container-verifier-phase.v1"]
    translation_receipt_digest: Digest
    requirements: ExecutionRequirements
    requirements_digest: Digest
    image: ImageArtifactIdentity
    command: Annotated[tuple[str, ...], Field(min_length=1, max_length=256)]
    workdir: SandboxPath
    submission: Annotated[tuple[SubmissionFile, ...], Field(min_length=1, max_length=16)]
    result_path: SandboxPath


class VerifierPhaseCode(StrEnum):
    INVALID = "execution_verifier_phase_invalid"
    RECEIPT_MISMATCH = "execution_verifier_receipt_mismatch"
    NOT_ISOLATED = "execution_verifier_not_isolated"
    SUBMISSION_UNDECLARED = "execution_verifier_submission_undeclared"
    NOT_DISTINCT = "execution_verifier_not_distinct"
    IMAGE_LINEAGE_MISMATCH = "execution_verifier_image_lineage_mismatch"
    PHASE_MISMATCH = "execution_verifier_phase_mismatch"


@dataclass(frozen=True, slots=True)
class VerifierPhaseRefusal:
    code: VerifierPhaseCode
    cause: str
    stage: str = "verifier_phase"


class VerifierPhaseError(ValueError):
    def __init__(self, code: VerifierPhaseCode, cause: str) -> None:
        super().__init__(code.value)
        self.refusal = VerifierPhaseRefusal(code, cause)


def _submission(receipt: HarborTranslationReceipt, workdir: str) -> tuple[SubmissionFile, ...]:
    """The task's declared final file. Only the REB dialect declares one today."""
    declared = receipt.source_declarations
    if receipt.source_dialect is not HarborDialect.REB_TASK_METADATA:
        raise VerifierPhaseError(
            VerifierPhaseCode.SUBMISSION_UNDECLARED, "Harbor 1.0 tasks declare no submission file"
        )
    name = (declared.get("task") or {}).get("submission_file")
    workspace = (declared.get("paths") or {}).get("workspace")
    if not isinstance(name, str) or not isinstance(workspace, str):
        raise VerifierPhaseError(
            VerifierPhaseCode.SUBMISSION_UNDECLARED, "task.submission_file and paths.workspace"
        )
    relative = PurePosixPath(workspace) / name
    if relative.is_absolute() or ".." in relative.parts or PurePosixPath(name).name != name:
        raise VerifierPhaseError(VerifierPhaseCode.SUBMISSION_UNDECLARED, str(relative))
    try:
        return (
            SubmissionFile(
                candidate_path=str(PurePosixPath(workdir) / relative),
                verifier_path=f"{VERIFIER_SUBMISSION_ROOT}/{name}",
                max_bytes=MAX_SUBMISSION_BYTES,
            ),
        )
    except ValidationError:
        raise VerifierPhaseError(VerifierPhaseCode.SUBMISSION_UNDECLARED, str(relative)) from None


def derive_verifier_phase(
    receipt: HarborTranslationReceipt,
    plan: ExecutionPackagePlan,
    *,
    candidate: ImageArtifactIdentity,
    verifier: ImageArtifactIdentity,
    verifier_config: dict[str, Any],
) -> VerifierPhase:
    """The only verifier phase this receipt, package and verifier image admit."""
    if (
        receipt.execution_requirements_digest != plan.execution_requirements_digest
        or receipt.build_input_digest != plan.build_plan.input_digest
        or receipt.verifier_requirements.digest() != receipt.verifier_requirements_digest
    ):
        raise VerifierPhaseError(
            VerifierPhaseCode.RECEIPT_MISMATCH, "translation receipt is not this package's"
        )
    requirements = receipt.verifier_requirements
    if requirements.verifier_isolation is not VerifierIsolation.ISOLATED:
        raise VerifierPhaseError(
            VerifierPhaseCode.NOT_ISOLATED, str(requirements.verifier_isolation)
        )
    if verifier.digest == candidate.digest or verifier.reference == candidate.reference:
        raise VerifierPhaseError(VerifierPhaseCode.NOT_DISTINCT, verifier.reference)
    labels = (verifier_config.get("config") or {}).get("Labels") or {}
    if (
        labels.get(BASE_DIGEST_LABEL) != candidate.digest
        or verifier.platform != candidate.platform
    ):
        raise VerifierPhaseError(
            VerifierPhaseCode.IMAGE_LINEAGE_MISMATCH,
            f"verifier image must be built FROM {candidate.digest}",
        )
    [phase] = (p for p in receipt.phases if p.name == "verifier")
    workdir = plan.build_plan.manifest.runtime.working_directory
    if not workdir:
        raise VerifierPhaseError(VerifierPhaseCode.SUBMISSION_UNDECLARED, "working_directory")
    return VerifierPhase(
        schema_version=PHASE_SCHEMA,
        translation_receipt_digest=receipt.digest(),
        requirements=requirements,
        requirements_digest=receipt.verifier_requirements_digest,
        image=verifier,
        command=(f"{VERIFIER_TESTS_ROOT}/{PurePosixPath(phase.entrypoint).relative_to('tests')}",),
        workdir=workdir,
        submission=_submission(receipt, workdir),
        result_path=VERIFIER_RESULT_PATH,
    )


def bind_verifier_phase(
    raw: bytes,
    receipt: HarborTranslationReceipt,
    plan: ExecutionPackagePlan,
    *,
    candidate: ImageArtifactIdentity,
    verifier_config: dict[str, Any],
) -> VerifierPhase:
    """Accept an uploaded phase only when it is exactly the receipt's derivation."""
    try:
        phase = VerifierPhase.model_validate_json(raw)
    except ValidationError:
        raise VerifierPhaseError(VerifierPhaseCode.INVALID, PHASE_FILENAME) from None
    expected = derive_verifier_phase(
        receipt, plan, candidate=candidate, verifier=phase.image, verifier_config=verifier_config
    )
    if phase != expected:
        differing = sorted(
            name
            for name in VerifierPhase.model_fields
            if getattr(phase, name) != getattr(expected, name)
        )
        raise VerifierPhaseError(VerifierPhaseCode.PHASE_MISMATCH, ",".join(differing))
    return phase
