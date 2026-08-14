"""Frozen search task families shared by GELO and OHCO.

The split is part of the benchmark identity: callers may select rows, but they
cannot move a seed between train and heldout by changing a request field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .targets import TargetRuntimeKind, TargetSpec


@dataclass(frozen=True, slots=True)
class SearchTaskFamily:
    family_id: str
    task_id: str
    objective: str
    train_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    objective_definition: dict[str, Any]

    def seeds(self, split: str) -> tuple[int, ...]:
        if split == "train":
            return self.train_seeds
        if split == "heldout":
            return self.heldout_seeds
        raise ValueError("split must be train or heldout")


CRAFTAX_SEARCH_V1 = SearchTaskFamily(
    family_id="craftax_singleplayer_search_v1",
    task_id="craftax.singleplayer",
    objective="craftax_env_sum",
    train_seeds=tuple(range(101, 133)),
    heldout_seeds=tuple(range(501, 521)),
    objective_definition={
        "version": "graded_objective.v1",
        "primary_metric": "craftax_env_sum",
        "direction": "maximize",
        "supporting_metrics": ["achievement_union", "survival", "env_steps"],
    },
)

ROGUE_SEARCH_V1 = SearchTaskFamily(
    family_id="rogue_singleplayer_search_v1",
    task_id="rogue.singleplayer",
    objective="rogue_graded_progress",
    train_seeds=tuple(range(1, 33)),
    heldout_seeds=tuple(range(101, 121)),
    objective_definition={
        "version": "graded_objective.v1",
        "primary_metric": "rogue_graded_progress",
        "direction": "maximize",
        "source_field": "readout.progress_metrics.synth_shaped_reward",
        "components": {
            "tile_seen": 1.0,
            "depth_reached": 100.0,
            "gold": 0.1,
            "experience": 0.05,
            "identity_learned": 5.0,
            "item_class_acquired": 5.0,
            "monster_type_killed": 10.0,
        },
        "supporting_metrics": ["max_level", "survival", "achievement_union"],
    },
)


def family_for_target(spec: TargetSpec) -> SearchTaskFamily | None:
    if (
        spec.runtime_family is TargetRuntimeKind.CRAFTAX
        and spec.environment_ref == "env:craftax_gold"
    ):
        return CRAFTAX_SEARCH_V1
    if spec.runtime_family is TargetRuntimeKind.ROGUE and spec.environment_ref == "env:rogue_gold":
        return ROGUE_SEARCH_V1
    return None


def dataset_manifest(spec: TargetSpec, family: SearchTaskFamily) -> dict[str, Any]:
    return {
        "version": "search_dataset.v1",
        "dataset_id": family.family_id,
        "task_family": family.family_id,
        "environment_ref": spec.environment_ref,
        "world_ref": spec.world_ref,
        "is_reference_world": True,
        "objective": family.objective_definition,
        "splits": {
            "train": {"count": len(family.train_seeds), "seeds": list(family.train_seeds)},
            "heldout": {
                "count": len(family.heldout_seeds),
                "seeds": list(family.heldout_seeds),
                "frozen": True,
            },
        },
        "seed_policy": "explicit_membership_v1",
    }


def dataset_rows(
    spec: TargetSpec,
    family: SearchTaskFamily,
    *,
    split: str,
    requested_seeds: list[int] | None,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    allowed = family.seeds(split)
    if requested_seeds is None:
        selected = allowed[max(0, offset) : max(0, offset) + max(0, limit)]
    else:
        invalid = sorted(set(requested_seeds).difference(allowed))
        if invalid:
            raise ValueError(f"seeds are not members of frozen {split} split: {invalid}")
        selected = tuple(requested_seeds[: max(0, limit)])
    return [
        {
            "row_id": f"{family.family_id}:{split}:{seed}",
            "task_id": family.task_id,
            "task_family": family.family_id,
            "task_instance_id": f"{family.family_id}:{split}:seed:{seed}",
            "split": split,
            "seed": seed,
            "objective": family.objective,
            "example": {
                "task_id": family.task_id,
                "task_family": family.family_id,
                "task_instance_id": f"{family.family_id}:{split}:seed:{seed}",
                "seed": seed,
                "objective": family.objective,
                "world_ref": spec.world_ref,
                "environment_ref": spec.environment_ref,
                "is_reference_world": True,
            },
            "metadata": {"is_reference_world": True, "frozen_split": split == "heldout"},
        }
        for seed in selected
    ]
