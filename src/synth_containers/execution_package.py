"""Bind execution requirements to frozen package inputs; no build or provider I/O.

See docs/specs/container-execution-package-v1.md. Historical manifest identities
remain unchanged; this explicit planning path adds a companion requirements seal.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import ValidationError, model_validator

from .execution_limits import MAXIMUM_REQUIREMENTS_BYTES
from .execution_requirements import ExecutionRequirements
from .package_build import plan_container_package
from .package_files import ContainerPackageError, read_package_file
from .package_models import Digest, PackageBuildPlan, PackageFileIdentity, PackageModel

__all__ = [
    "EXECUTION_REQUIREMENTS_FILENAME",
    "ExecutionPackageCause",
    "ExecutionPackageCode",
    "ExecutionPackageError",
    "ExecutionPackagePlan",
    "ExecutionPackageRefusal",
    "ExecutionPackageStage",
    "plan_execution_package",
]

EXECUTION_REQUIREMENTS_FILENAME = "execution-requirements.toml"


class ExecutionPackageCode(StrEnum):
    REQUIREMENTS_NOT_FROZEN = "execution_package_requirements_not_frozen"
    REQUIREMENTS_UNAVAILABLE = "execution_package_requirements_unavailable"
    REQUIREMENTS_CHANGED = "execution_package_requirements_changed"
    REQUIREMENTS_INVALID = "execution_package_requirements_invalid"
    REQUIREMENTS_TOO_LARGE = "execution_package_requirements_too_large"
    PLATFORM_MISMATCH = "execution_package_platform_mismatch"


class ExecutionPackageStage(StrEnum):
    PLAN = "execution_package_plan"


class ExecutionPackageCause(StrEnum):
    NOT_DECLARED = "requirements_file_not_in_build_inputs"
    UNAVAILABLE = "requirements_input_unavailable"
    SOURCE_CHANGED = "requirements_input_changed_after_snapshot"
    INVALID_ENCODING = "requirements_utf8_invalid"
    INVALID_TOML = "requirements_toml_invalid"
    INVALID_CONTRACT = "requirements_contract_invalid"
    SIZE_EXCEEDED = "requirements_input_size_exceeded"
    PLATFORM_DISAGREEMENT = "build_and_execution_platform_disagree"


@dataclass(frozen=True, slots=True)
class ExecutionPackageRefusal:
    code: ExecutionPackageCode
    stage: ExecutionPackageStage
    cause: ExecutionPackageCause
    package_id: str
    requirements_path: str

    @classmethod
    def for_package(
        cls, plan: PackageBuildPlan, code: ExecutionPackageCode, cause: ExecutionPackageCause
    ) -> Self:
        return cls(
            code=code,
            stage=ExecutionPackageStage.PLAN,
            cause=cause,
            package_id=plan.manifest.package_id,
            requirements_path=EXECUTION_REQUIREMENTS_FILENAME,
        )


class ExecutionPackageError(ValueError):
    """Exception transport for an immutable, secret-free refusal payload."""

    def __init__(self, refusal: ExecutionPackageRefusal) -> None:
        self.refusal = refusal
        super().__init__(refusal.code.value)


class ExecutionPackagePlan(PackageModel):
    schema_version: Literal["synth.container-execution-package.v1"]
    build_plan: PackageBuildPlan
    requirements_file: PackageFileIdentity
    execution_requirements: ExecutionRequirements
    execution_requirements_digest: Digest

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.requirements_file.path != EXECUTION_REQUIREMENTS_FILENAME:
            raise ValueError("execution_package_requirements_path_invalid")
        if self.requirements_file not in self.build_plan.files:
            raise ValueError("execution_package_requirements_not_frozen")
        if self.execution_requirements_digest != self.execution_requirements.digest():
            raise ValueError("execution_package_requirements_digest_mismatch")
        if self.build_plan.manifest.build.platform != self.execution_requirements.platform:
            raise ValueError("execution_package_platform_mismatch")
        return self

    @classmethod
    def bind(
        cls,
        plan: PackageBuildPlan,
        source: PackageFileIdentity,
        requirements: ExecutionRequirements,
    ) -> Self:
        return cls(
            schema_version="synth.container-execution-package.v1",
            build_plan=plan,
            requirements_file=source,
            execution_requirements=requirements,
            execution_requirements_digest=requirements.digest(),
        )


def _refused(
    plan: PackageBuildPlan, code: ExecutionPackageCode, cause: ExecutionPackageCause
) -> ExecutionPackageError:
    return ExecutionPackageError(ExecutionPackageRefusal.for_package(plan, code, cause))


def plan_execution_package(manifest_path: str | Path) -> ExecutionPackagePlan:
    """Freeze declared bytes and reject changed or contradictory requirements.

    This checks package identity, not artifact architecture or provider support.
    The placement owner still supplies an inspected artifact and authenticated
    offer capabilities to the canonical execution preflight.
    """
    path = Path(manifest_path).absolute()
    plan = plan_container_package(path)
    # Absence here means the sidecar was not part of the declared build snapshot.
    source = next(
        (entry for entry in plan.files if entry.path == EXECUTION_REQUIREMENTS_FILENAME), None
    )
    if source is None:
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_NOT_FROZEN, ExecutionPackageCause.NOT_DECLARED
        )
    if source.size_bytes > MAXIMUM_REQUIREMENTS_BYTES:
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_TOO_LARGE, ExecutionPackageCause.SIZE_EXCEEDED
        )
    try:
        raw = read_package_file(
            path.parent, EXECUTION_REQUIREMENTS_FILENAME, maximum=MAXIMUM_REQUIREMENTS_BYTES
        )
    except ContainerPackageError as error:
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_UNAVAILABLE, ExecutionPackageCause.UNAVAILABLE
        ) from error
    if (
        len(raw) != source.size_bytes
        or "sha256:" + hashlib.sha256(raw).hexdigest() != source.digest
    ):
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_CHANGED, ExecutionPackageCause.SOURCE_CHANGED
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_INVALID, ExecutionPackageCause.INVALID_ENCODING
        ) from error
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_INVALID, ExecutionPackageCause.INVALID_TOML
        ) from error
    try:
        requirements = ExecutionRequirements.model_validate(payload)
    except ValidationError as error:
        raise _refused(
            plan, ExecutionPackageCode.REQUIREMENTS_INVALID, ExecutionPackageCause.INVALID_CONTRACT
        ) from error
    if plan.manifest.build.platform != requirements.platform:
        raise _refused(
            plan,
            ExecutionPackageCode.PLATFORM_MISMATCH,
            ExecutionPackageCause.PLATFORM_DISAGREEMENT,
        )
    return ExecutionPackagePlan.bind(plan, source, requirements)
