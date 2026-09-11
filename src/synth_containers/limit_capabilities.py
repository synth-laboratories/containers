"""Typed enforcement requirements, distinct from the requested limit values.

A provider's allocation control is not a sampled usage threshold, and neither
implies an actuator that survives supervisor loss. Missing guarantees fail closed.
"""

from dataclasses import dataclass
from enum import StrEnum


class LimitDimension(StrEnum):
    WORK_TIME = "work_time"
    OUTPUT_BYTES = "output_bytes"
    WORKSPACE_BYTES = "workspace_bytes"
    CPU_ALLOCATION = "cpu_allocation"
    MEMORY_BYTES = "memory_bytes"
    PROVIDER_SPEND = "provider_spend"


class LimitEnforcement(StrEnum):
    OBSERVED_THRESHOLD = "observed_threshold"
    NATIVE_CONTROL = "native_control"
    RESERVED_ALLOWANCE = "reserved_allowance"


@dataclass(frozen=True, slots=True)
class LimitCapability:
    dimension: LimitDimension
    enforcement: LimitEnforcement
    survives_supervisor_loss: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, LimitDimension):
            raise ValueError("limit dimension must be a known LimitDimension")
        if not isinstance(self.enforcement, LimitEnforcement):
            raise ValueError("limit enforcement must be a known LimitEnforcement")
        if type(self.survives_supervisor_loss) is not bool:
            raise ValueError("survives_supervisor_loss must be a boolean")

    def to_payload(self) -> dict[str, str | bool]:
        return {
            "dimension": self.dimension.value,
            "enforcement": self.enforcement.value,
            "survives_supervisor_loss": self.survives_supervisor_loss,
        }


def unsupported_limit_capabilities(
    required: tuple[LimitCapability, ...], available: tuple[LimitCapability, ...]
) -> tuple[LimitCapability, ...]:
    """Compare exact mechanisms; do not infer an ordering between guarantees."""
    if not isinstance(required, tuple) or len(required) > 32:
        raise ValueError("required limit capabilities must be a tuple of at most 32 entries")
    if any(not isinstance(item, LimitCapability) for item in (*required, *available)):
        raise ValueError("limit capabilities must be typed LimitCapability entries")
    return tuple(
        requirement for requirement in required
        if not any(
            capability.dimension == requirement.dimension
            and capability.enforcement == requirement.enforcement
            and (not requirement.survives_supervisor_loss or capability.survives_supervisor_loss)
            for capability in available
        )
    )
