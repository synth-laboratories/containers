"""Portable service declarations and observed bindings. See docs/specs/services-v1.md.

Declarations carry requirements, never authority. A deployment owner supplies the
binding after admission and protocol negotiation. Native jobs may group operations;
their protocol-specific configuration is deliberately outside this contract.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Annotated, Literal
from urllib.parse import unquote, urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[a-f0-9]{64}$")]


class ServiceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


def service_path(value: str) -> str:
    decoded = unquote(value)
    path = PurePosixPath(decoded)
    if (
        not value.startswith("/")
        or value.startswith("//")
        or decoded != value
        or ".." in path.parts
        or "\\" in value
        or "?" in value
        or "#" in value
        or any(ord(c) <= 32 for c in value)
        or str(path) != value
    ):
        raise ValueError("service_path_invalid")
    return value


class ServiceInterface(ServiceModel):
    name: Name
    protocol: Name
    version: Name
    port: int = Field(strict=True, ge=1, le=65535)
    base_path: str = "/"
    transport: Literal["http", "mcp.streamable-http"] = "http"
    schema_path: str | None = None
    authentication: Literal["scoped-bearer"] = "scoped-bearer"

    @model_validator(mode="after")
    def paths(self) -> ServiceInterface:
        service_path(self.base_path)
        if self.schema_path is not None:
            service_path(self.schema_path)
        return self


class ServiceState(ServiceModel):
    name: Name
    mount_path: str
    persistence: Literal["deployment"] = "deployment"
    access: Literal["single-writer"] = "single-writer"
    retention: Literal["retain"] = "retain"

    @model_validator(mode="after")
    def path(self) -> ServiceState:
        service_path(self.mount_path)
        if self.mount_path == "/":
            raise ValueError("service_state_root_mount_refused")
        return self


class ServiceResourceLimits(ServiceModel):
    cpu_cores: int = Field(strict=True, ge=1, le=64)
    memory_mb: int = Field(strict=True, ge=128, le=262144)
    storage_mb: int = Field(strict=True, ge=128, le=1048576)


class ServiceInferencePolicy(ServiceModel):
    models: tuple[str, ...] = Field(min_length=1, max_length=32)
    ceiling_micros: int = Field(strict=True, ge=1, le=50_000_000)


class ServiceInferenceRequest(ServiceModel):
    model: str = Field(min_length=1, max_length=200)
    wire: Literal["responses", "chat_completions"]


class ServiceExecution(ServiceModel):
    protocol: Name
    version: Name
    interface: Name
    admission_unit: Literal["operation", "trial"] = "operation"
    concurrency: int = Field(strict=True, ge=1, le=4096)
    resource_limits: ServiceResourceLimits = ServiceResourceLimits(
        cpu_cores=4, memory_mb=8192, storage_mb=32768
    )
    max_timeout_seconds: int = Field(strict=True, ge=1, le=3600, default=3600)
    max_queued: int = Field(strict=True, ge=1, le=4096, default=4096)
    inference: ServiceInferencePolicy | None = None
    pause: Literal["unsupported", "queued"] = "unsupported"
    resume: Literal["unsupported", "queued"] = "unsupported"
    checkpoint: Literal["unsupported"] = "unsupported"
    recovery: Literal["reconcile"] = "reconcile"
    cancellation: Literal["request-and-reconcile"] = "request-and-reconcile"


class ServiceDeclaration(ServiceModel):
    schema_version: Literal["synth.container-services.v1"] = "synth.container-services.v1"
    interfaces: tuple[ServiceInterface, ...] = Field(min_length=1, max_length=16)
    state: tuple[ServiceState, ...] = Field(default=(), max_length=16)
    execution: ServiceExecution | None = None

    @model_validator(mode="after")
    def identities(self) -> ServiceDeclaration:
        names = [item.name for item in self.interfaces]
        if len(names) != len(set(names)):
            raise ValueError("service_interface_duplicate")
        mounts = [item.mount_path for item in self.state]
        if len({item.name for item in self.state}) != len(self.state):
            raise ValueError("service_state_duplicate")
        if len(set(mounts)) != len(mounts):
            raise ValueError("service_mount_duplicate")
        for index, mount in enumerate(mounts):
            if any(
                PurePosixPath(mount) in PurePosixPath(other).parents
                for other in mounts[index + 1 :]
            ) or any(
                PurePosixPath(other) in PurePosixPath(mount).parents
                for other in mounts[index + 1 :]
            ):
                raise ValueError("service_mount_overlap")
        if self.execution is not None and self.execution.interface not in names:
            raise ValueError("service_execution_interface_missing")
        return self


class ServiceEndpoint(ServiceModel):
    name: Name
    protocol: Name
    version: Name
    url: str
    transport: Literal["http", "mcp.streamable-http"] = "http"
    declaration_digest: Digest | None = None

    @model_validator(mode="after")
    def public_url(self) -> ServiceEndpoint:
        url = urlsplit(self.url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("service_endpoint_url_invalid")
        service_path(url.path.rstrip("/") or "/")
        return self


class ServiceBinding(ServiceModel):
    schema_version: Literal["synth.service-binding.v1"] = "synth.service-binding.v1"
    deployment_id: Name
    revision: Digest
    origin_instance: UUID
    generation: UUID
    declaration_digest: Digest
    endpoints: tuple[ServiceEndpoint, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities(self) -> ServiceBinding:
        if not self.origin_instance.int or len({e.name for e in self.endpoints}) != len(
            self.endpoints
        ):
            raise ValueError("service_binding_identity_invalid")
        return self


class ExternalOperationBinding(ServiceModel):
    schema_version: Literal["synth.external-operation.v1"] = "synth.external-operation.v1"
    operation_id: Name
    input_digest: Digest
    deployment_id: Name
    deployment_revision: Digest
    origin_instance: UUID
    interface: Name
    native_id: str = Field(min_length=1, max_length=256)
    admission_id: str = Field(min_length=1, max_length=256)
    epoch: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def non_nil(self) -> ExternalOperationBinding:
        if not self.origin_instance.int:
            raise ValueError("service_origin_instance_nil")
        return self


class ServiceAttempt(ServiceModel):
    """The resolved, bounded work a service asks its deployment owner to admit."""

    schema_version: Literal["synth.service-attempt.v1"] = "synth.service-attempt.v1"
    native_id: Name
    job_id: Name
    retry_of: Name | None = None
    input_digest: Digest
    timeout_seconds: int = Field(strict=True, ge=1, le=3600)
    limits: ServiceResourceLimits
    inference: ServiceInferenceRequest | None = None
    images: tuple[
        Annotated[
            str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[a-f0-9]{64}$")
        ],
        ...,
    ] = Field(min_length=1, max_length=2)
