"""Typed contracts for Rhodes environment build/import operations (SYN-4839).

Mirrors backend ``packages/rhodes/build_operations.py`` and the routes under
``/v1/environment_build_operations``. Build success is not a release, a
validation or ready capacity; those remain separate operations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .rhodes_client import RhodesRequestError

BUILD_OPERATION_SCHEMA = "synth.rhodes-environment-build.v1"
BUILD_STATES = ("pending", "running", "succeeded", "failed", "recovery_required")
TERMINAL_STATES = frozenset({"succeeded", "failed", "recovery_required"})
RETRYABLE_STATES = frozenset({"failed", "recovery_required"})

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_ENV = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_IMAGE = re.compile(r"^[A-Za-z0-9._/:@-]{1,512}$")

#: Refusal codes the backend emits for build operations and their bindings.
BUILD_REFUSAL_CODES = frozenset(
    "rhodes.build." + name
    for name in (
        "executor_unavailable",
        "executor_not_permitted",
        "artifact_provider_mismatch",
        "source_missing",
        "manifest_mismatch",
        "source_forbidden",
        "source_invalid",
        "source_unavailable",
        "digest_mismatch",
        "digest_unpinned",
        "operation_conflict",
        "not_found",
        "not_retryable",
        "invalid_id",
        "invalid_limit",
    )
)


class BuildOperationRefused(RhodesRequestError):
    """A typed ``rhodes.build.*`` refusal; ``code`` is the backend code."""

    @property
    def reason(self) -> str:
        return self.code.removeprefix("rhodes.build.")


class ExecutorUnavailable(BuildOperationRefused): ...
class ExecutorNotPermitted(BuildOperationRefused): ...
class ArtifactProviderMismatch(BuildOperationRefused): ...
class SourceMissing(BuildOperationRefused): ...
class ManifestMismatch(BuildOperationRefused): ...
class SourceForbidden(BuildOperationRefused): ...
class SourceInvalid(BuildOperationRefused): ...


_REFUSALS: dict[str, type[BuildOperationRefused]] = {
    "rhodes.build.executor_unavailable": ExecutorUnavailable,
    "rhodes.build.executor_not_permitted": ExecutorNotPermitted,
    "rhodes.build.artifact_provider_mismatch": ArtifactProviderMismatch,
    "rhodes.build.source_missing": SourceMissing,
    "rhodes.build.manifest_mismatch": ManifestMismatch,
    "rhodes.build.source_forbidden": SourceForbidden,
    "rhodes.build.source_invalid": SourceInvalid,
}


def typed_refusal(error: RhodesRequestError) -> RhodesRequestError:
    """Return the typed refusal for a build code, else the error unchanged."""
    if error.code not in BUILD_REFUSAL_CODES:
        return error
    cls = _REFUSALS.get(error.code, BuildOperationRefused)
    return cls(error.status_code, error.code, error.retry_after, error.identifiers)


def _check(pattern: re.Pattern[str], value: str, label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}")
    return value


@dataclass(frozen=True)
class SourceArtifact:
    artifact_id: str
    digest: str
    kind: Literal["source_artifact"] = "source_artifact"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "artifact_id": _check(_ID, self.artifact_id, "artifact_id"),
            "digest": _check(_DIGEST, self.digest, "digest"),
        }


@dataclass(frozen=True)
class RegistryImage:
    image_ref: str
    expected_digest: str | None = None
    kind: Literal["registry_image"] = "registry_image"

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "kind": self.kind,
            "image_ref": _check(_IMAGE, self.image_ref, "image_ref"),
        }
        if self.expected_digest is not None:
            body["expected_digest"] = _check(_DIGEST, self.expected_digest, "expected_digest")
        return body


@dataclass(frozen=True)
class BuildOperationRequest:
    operation_id: str
    environment: str
    source: SourceArtifact | RegistryImage
    manifest_digest: str

    @property
    def operation_kind(self) -> Literal["build", "import"]:
        return "build" if isinstance(self.source, SourceArtifact) else "import"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": BUILD_OPERATION_SCHEMA,
            "operation_id": _check(_ID, self.operation_id, "operation_id"),
            "environment": _check(_ENV, self.environment, "environment"),
            "source": self.source.to_json(),
            "manifest_digest": _check(_DIGEST, self.manifest_digest, "manifest_digest"),
        }


@dataclass(frozen=True)
class FailureRecord:
    attempt: int
    code: str
    retryable: bool
    observed_at: str
    log_ref: str | None = None

    @classmethod
    def from_json(cls, body: Mapping[str, Any]) -> FailureRecord:
        return cls(
            attempt=int(body["attempt"]),
            code=str(body["code"]),
            retryable=bool(body["retryable"]),
            observed_at=str(body["observed_at"]),
            log_ref=body.get("log_ref"),
        )


@dataclass(frozen=True)
class BuildOperation:
    build_operation_id: str
    operation_id: str
    kind: str
    environment: str
    source: Mapping[str, Any]
    manifest_digest: str
    state: str
    attempt: int
    failure_history: tuple[FailureRecord, ...] = ()
    lease_expires_at: str | None = None
    log_ref: str | None = None
    artifact_ref: Mapping[str, str] = field(default_factory=dict)
    resolved_digest: str | None = None
    artifact_kind: str | None = None
    release_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    reused: bool | None = None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def retryable(self) -> bool:
        return self.state in RETRYABLE_STATES

    @classmethod
    def from_json(cls, body: Mapping[str, Any], *, reused: bool | None = None) -> BuildOperation:
        try:
            return cls(
                build_operation_id=str(body["build_operation_id"]),
                operation_id=str(body["operation_id"]),
                kind=str(body["kind"]),
                environment=str(body["environment"]),
                source=dict(body["source"]),
                manifest_digest=str(body["manifest_digest"]),
                state=str(body["state"]),
                attempt=int(body["attempt"]),
                failure_history=tuple(
                    FailureRecord.from_json(item) for item in body.get("failure_history") or ()
                ),
                lease_expires_at=body.get("lease_expires_at"),
                log_ref=body.get("log_ref"),
                artifact_ref=dict(body.get("artifact_ref") or {}),
                resolved_digest=body.get("resolved_digest"),
                artifact_kind=body.get("artifact_kind"),
                release_id=body.get("release_id"),
                created_at=body.get("created_at"),
                updated_at=body.get("updated_at"),
                started_at=body.get("started_at"),
                completed_at=body.get("completed_at"),
                reused=reused,
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            from .pools import PoolClientError

            raise PoolClientError("Rhodes returned a malformed build operation") from error
