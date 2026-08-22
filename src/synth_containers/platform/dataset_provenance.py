"""Immutable dataset and generated-workload identities for training admission."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def manifest_for_spec(spec: Any) -> dict[str, Any] | None:
    family = spec.runtime_family.value
    if family == "gsm8k":
        from .gsm8k_world import dataset_manifest

        return dataset_manifest()
    if family == "banking77":
        from .banking77_world import (
            HELDOUT_SPLIT,
            TRAIN_SPLIT,
            load_row,
            source_name,
            split_size,
        )

        splits: dict[str, list[dict[str, str]]] = {}
        for split in (TRAIN_SPLIT, HELDOUT_SPLIT):
            splits[split] = [
                {"text": row.text, "label": row.label}
                for seed in range(split_size(split))
                if (row := load_row(split, seed)) is not None
            ]
        return {
            "schema_version": "banking77.dataset-manifest.v1",
            "source": source_name(),
            "splits": splits,
        }
    if family == "healthbench":
        from .healthbench_world import DATASET_URL, rows

        return {
            "schema_version": "healthbench.dataset-manifest.v1",
            "source": DATASET_URL,
            "rows": list(rows()),
        }
    if family == "craftax":
        return {
            "schema_version": "craftax.workload-manifest.v1",
            "world_ref": spec.world_ref,
            "environment_ref": spec.environment_ref,
            "evaluation_plan_ref": spec.evaluation_plan_ref,
            "max_episode_steps": spec.max_episode_steps,
        }
    return None


def fallback_manifest_for_spec(spec: Any) -> dict[str, Any]:
    return {
        "schema_version": "training.workload-manifest.v1",
        "target_id": spec.target_id,
        "runtime_family": spec.runtime_family.value,
        "world_ref": spec.world_ref,
        "environment_ref": spec.environment_ref,
        "evaluation_plan_ref": spec.evaluation_plan_ref,
    }


def digest_for_spec(spec: Any) -> str:
    manifest = manifest_for_spec(spec) or fallback_manifest_for_spec(spec)
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()
