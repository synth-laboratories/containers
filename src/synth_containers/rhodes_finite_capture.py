"""Finite stdio-only native execution contract, not an arbitrary HTTP claim."""

from typing import Annotated, Literal
from pathlib import PurePosixPath
from uuid import UUID

from pydantic import Field, model_validator
from .rhodes_catalog import Contract, EvidenceReference


class FiniteCaptureContract(Contract):
    schema_version: Literal["synth.rhodes-finite-capture.v1"]
    profile: Literal["isolated_stdio_v1"]
    build_operation_id: str
    source: EvidenceReference
    image_archive: EvidenceReference
    grader_source_path: str
    policy_source_path: str
    command: tuple[str, ...] = Field(min_length=1, max_length=64)
    recorder_cohort_id: str = Field(min_length=1, max_length=128)
    maximum_output_bytes: Annotated[int, Field(strict=True, ge=1024, le=256 * 1024)]
    cpu_cores: Annotated[int, Field(strict=True, ge=1, le=16)]
    memory_mb: Annotated[int, Field(strict=True, ge=256, le=8192)]
    pids_limit: Literal[64]

    @model_validator(mode="after")
    def finite_shape(self):
        from .package_models import portable_path

        if str(UUID(self.build_operation_id)) != self.build_operation_id:
            raise ValueError("Finite capture requires original qualified build identity")
        portable_path(self.grader_source_path)
        portable_path(self.policy_source_path)
        executable = PurePosixPath(self.command[0])
        if not executable.is_absolute() or ".." in executable.parts:
            raise ValueError("Finite command requires an absolute in-image executable")
        if any("\0" in part or len(part) > 4096 for part in self.command):
            raise ValueError("Finite argv exceeds its bound")
        return self
