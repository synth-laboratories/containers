"""Declarative connector configuration with local-only secret references."""

from __future__ import annotations
from pathlib import Path
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .targets import LocalOrigin, LocalTargetRegistry, OriginSecretHeader


class SecretHeaderProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    header: str
    secret_name: str


class TargetProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    key: str
    url: str
    service_identity: str
    credentials: tuple[SecretHeaderProfile, ...] = Field(default=(), max_length=8)


class ConnectorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    connector: UUID
    client_instance: UUID
    backend_url: str
    gateway_url: str
    backend_credential_env: str = "SYNTH_API_KEY"
    targets: tuple[TargetProfile, ...] = Field(min_length=1, max_length=64)
    max_inflight: int = Field(default=128, ge=1, le=2048)
    max_request_bytes: int = Field(default=50_000_000, ge=1, le=50_000_000)

    @field_validator("connector", "client_instance")
    @classmethod
    def non_nil(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("nil connector profile identity")
        return value

    @classmethod
    def load(cls, path: Path) -> ConnectorProfile:
        if path.stat().st_size > 65536:
            raise ValueError("connector profile exceeds 64 KiB")
        return cls.model_validate_json(path.read_text())

    def registry(self) -> LocalTargetRegistry:
        registry = LocalTargetRegistry()
        for target in self.targets:
            registry.register(
                target.key,
                LocalOrigin(
                    target.url,
                    target.service_identity,
                    tuple(
                        OriginSecretHeader(item.header, item.secret_name)
                        for item in target.credentials
                    ),
                ),
                expected_generation=0,
            )
        return registry
