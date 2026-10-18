"""Operation-scoped origin observation/cleanup. See docs/specs/tunnel-v2-connector.md."""
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OriginOperationBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_instance: UUID
    input_sha256: str = Field(strict=True, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")

    @field_validator("origin_instance")
    @classmethod
    def non_nil(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("origin_operation_instance_nil")
        return value
