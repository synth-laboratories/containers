from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serde import JsonDataclassMixin


GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"


@dataclass(slots=True)
class PromptModule(JsonDataclassMixin):
    module_id: str
    role: str = ""
    content: str = ""
    mutable: bool = False
    candidate_field: str = ""
    template_variables: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TargetModule(JsonDataclassMixin):
    module_id: str
    candidate_field: str = ""
    objective: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PromptProgram(JsonDataclassMixin):
    version: str = "prompt_program.v1"
    program_id: str = ""
    modules: list[PromptModule] = field(default_factory=list)
    target_modules: list[TargetModule] = field(default_factory=list)
    seed_candidate: dict[str, str] = field(default_factory=dict)
    rollout_overlay_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def mutable_field_ids(self) -> list[str]:
        return [
            module.candidate_field
            for module in self.modules
            if module.mutable and module.candidate_field
        ]


@dataclass(slots=True)
class CandidateOverlay(JsonDataclassMixin):
    candidate: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def gepa_optimizer_contract() -> dict[str, str]:
    return {
        "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
        "program_route": "/program",
        "dataset_route": "/dataset",
        "dataset_rows_route": "/dataset/rows",
        "rollout_route": "/rollout",
    }
