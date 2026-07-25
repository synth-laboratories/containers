"""Optional token-level capture without forcing training payloads into every trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from synth_containers.serde import JsonDataclassMixin


class TokenCaptureProvenance(StrEnum):
    OBSERVED_PROVIDER = "observed_provider"
    OBSERVED_HARNESS = "observed_harness"
    IMPORTED = "imported"
    DERIVED_RETOKENIZED = "derived_retokenized"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class TokenSequenceRefV1(JsonDataclassMixin):
    token_ids: tuple[int, ...] = ()
    artifact_id: str | None = None
    count: int | None = None
    digest: str | None = None
    encoding: str = "int-array"

    def __post_init__(self) -> None:
        if self.token_ids and self.artifact_id:
            raise ValueError("token sequence cannot be both inline and artifact-backed")
        if self.count is not None and self.token_ids and self.count != len(self.token_ids):
            raise ValueError("token sequence count does not match inline token ids")


@dataclass(frozen=True, slots=True)
class TokenCaptureV5(JsonDataclassMixin):
    provenance: TokenCaptureProvenance | str
    level: str
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    prompt: TokenSequenceRefV1 | None = None
    completion: TokenSequenceRefV1 | None = None
    completion_logprobs: tuple[float, ...] = ()
    logprobs_artifact_id: str | None = None
    unavailable_fields: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        completion_count = self.completion.count if self.completion else None
        if completion_count is None and self.completion and self.completion.token_ids:
            completion_count = len(self.completion.token_ids)
        if (
            completion_count is not None
            and self.completion_logprobs
            and completion_count != len(self.completion_logprobs)
        ):
            raise ValueError("completion logprobs must align one-to-one with completion tokens")


__all__ = [
    "TokenCaptureProvenance",
    "TokenCaptureV5",
    "TokenSequenceRefV1",
]
