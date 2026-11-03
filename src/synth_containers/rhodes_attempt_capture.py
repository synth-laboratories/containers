"""Opt-in per-epoch capture/export contracts; no caller claim grants custody.

Spec: notes/specifications/rhodes-v024-service.md. A full closure requires
independent current owner proofs, not object labels or an observed image digest.
"""

from typing import Annotated, Literal
from uuid import UUID
import re

from pydantic import Field, model_validator

from .rhodes_catalog import Contract, EvidenceReference
from .package_models import portable_path
from .rhodes_finite_capture import FiniteCaptureContract

CaptureRole = Literal["input", "configuration", "source", "grader", "policy", "image", "dependency", "output", "log"]


class CaptureAdmissionReference(Contract):
    authority: Literal["artifact-platform"]
    kind: Literal["capture_admission_request"]
    record_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    revision: Literal["1"]
    digest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def same_request(self):
        if self.record_id != self.digest_sha256:
            raise ValueError("Capture admission reference digest differs")
        return self


class AttemptCaptureRequest(Contract):
    schema_version: Literal["synth.rhodes-attempt-capture-request.v1"]
    mode: Literal["available", "required_full"]
    publish_input: Literal[True]
    publish_configuration: Literal[True]
    input_data_classification: Literal["owner_declared_nonsecret_except_opaque_paths"]
    allow_private_grader_export: bool = False
    credential_handling: Literal["opaque_nonreplayable_references"]
    # Nonsecret reference only. The owner resolves original intent and current
    # original-caller authority; caller bodies cannot supply that proof.
    intent_reference: CaptureAdmissionReference
    declared_secret_paths: tuple[str, ...] = Field(default=(), max_length=128)
    required_roles: tuple[CaptureRole, ...] = Field(default=(
        "input", "configuration", "source", "grader", "policy", "image",
    ), max_length=9)
    maximum_bytes: Annotated[int, Field(strict=True, ge=1024, le=16 * 1024 * 1024)] = 8 * 1024 * 1024

    @model_validator(mode="after")
    def exact_declaration(self):
        if not self.required_roles or len(set(self.required_roles)) != len(self.required_roles):
            raise ValueError("Capture roles must be nonempty and unique")
        for pointer in self.declared_secret_paths:
            if not pointer.startswith("/") or len(pointer) > 1024:
                raise ValueError("Secret path requires a bounded JSON pointer")
            if "~" in pointer.replace("~0", "").replace("~1", ""):
                raise ValueError("Secret path has an invalid JSON pointer escape")
        return self


class DeclaredCaptureReference(Contract):
    role: CaptureRole
    reference: EvidenceReference


class AttemptCaptureExportPolicy(Contract):
    """Task-owner declaration, independently authorized by the custody owner."""

    schema_version: Literal["synth.rhodes-attempt-capture-export.v1"]
    project_id: str
    revision: Annotated[int, Field(strict=True, ge=1)]
    private_grader_export: bool = False
    private_grader_contains_no_credentials: bool = False
    input_data_classification: Literal["owner_declared_nonsecret_except_opaque_paths"]
    command_contains_no_credentials: Literal[True]
    retained_objects: tuple[DeclaredCaptureReference, ...] = Field(default=(), max_length=128)
    finite_contract: FiniteCaptureContract | None = None

    @model_validator(mode="after")
    def canonical_project(self):
        if str(UUID(self.project_id)) != self.project_id:
            raise ValueError("Capture policy requires canonical project identity")
        if self.private_grader_export and not self.private_grader_contains_no_credentials:
            raise ValueError("Private grader export requires owner noncredential declaration")
        return self


class CaptureObject(Contract):
    logical_path: str
    role: CaptureRole
    digest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=16 * 1024 * 1024)]
    media_type: str
    origin: Literal["executor_handoff", "resolved_configuration", "private_harbor_phase", "execution_policy"]

    @model_validator(mode="after")
    def portable_object(self):
        portable_path(self.logical_path)
        return self


class AttemptCaptureReceipt(Contract):
    schema_version: Literal["synth.rhodes-attempt-capture.v1"]
    execution_epoch: str
    status: Literal["pending", "committed", "refused"]
    scope: Literal["pre_execution_inputs_and_declarations", "terminal_observed_output"]
    inventory: tuple[CaptureObject, ...] = Field(max_length=256)
    inventory_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    archive_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    archive_size_bytes: Annotated[int, Field(strict=True, ge=0, le=16 * 1024 * 1024)]
    publication_size_bytes: Annotated[int, Field(strict=True, ge=0, le=32 * 1024 * 1024)]
    publication_id: str
    collection_id: str
    manifest_digest: str | None = None
    logical_path: Literal["attempt.zip"] = "attempt.zip"
    source_references: tuple[DeclaredCaptureReference, ...] = Field(default=(), max_length=130)
    opaque_credential_references: tuple[dict[str, str], ...] = Field(default=(), max_length=256)
    gaps: tuple[str, ...] = Field(max_length=256)
    full_attempt_capture_verified: bool = False
    owner_custody: dict[str, str] | None = None
    role_closures: dict[str, str] = Field(default_factory=dict)
    terminal_publication: dict | None = None
    terminal_native: dict | None = None

    @model_validator(mode="after")
    def closure_is_owned(self):
        if self.publication_size_bytes < self.archive_size_bytes:
            raise ValueError("Capture publication byte accounting is invalid")
        if self.full_attempt_capture_verified and (
            self.status != "committed" or self.gaps or not self.owner_custody
            or self.opaque_credential_references
        ):
            raise ValueError("Complete capture requires committed gap-free owner custody")
        if self.full_attempt_capture_verified:
            required = {"input", "configuration", "source", "image", "dependency", "grader", "policy", "image_config_digest", "image_platform"}
            if (self.scope != "pre_execution_inputs_and_declarations"
                or not required <= self.role_closures.keys()
                or any(re.fullmatch(r"sha256:[0-9a-f]{64}", self.role_closures[key]) is None for key in required)
                or self.terminal_publication is None or self.terminal_native is None):
                raise ValueError("Complete capture requires exact finite role and terminal closures")
            terminal = type(self).model_validate(self.terminal_publication)
            if (terminal.scope != "terminal_observed_output" or terminal.status != "committed"
                or terminal.execution_epoch != self.execution_epoch or terminal.gaps
                or not terminal.owner_custody or terminal.opaque_credential_references):
                raise ValueError("Complete capture requires committed same-epoch terminal custody")
            objects = {item.logical_path: item for item in terminal.inventory}
            if not {"output/stdout.bin", "output/stderr.bin", "output/stdio-journal.json",
                    "output/recorder-proof.json", "output/native-trace.zip"} <= objects.keys():
                raise ValueError("Complete capture requires every actual terminal object")
            native = self.terminal_native
            if ((native.get("authority"), native.get("kind"), native.get("revision")) !=
                    ("mnemosyne", "capture_native_trace", "1")
                or not isinstance(native.get("record_id"), str)
                or re.fullmatch(r"mct_[0-9a-f]{64}", native["record_id"]) is None
                or native.get("digest_sha256") != objects["output/native-trace.zip"].digest_sha256):
                raise ValueError("Complete capture requires exact retained terminal native archive")
        return self
