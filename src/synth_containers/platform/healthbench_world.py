"""HealthBench task world backed by OpenAI's public physician-rubric release."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx


DATASET_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "2025-05-07-06-14-12_oss_eval.jsonl"
)


@lru_cache(maxsize=1)
def rows() -> tuple[dict[str, Any], ...]:
    override = os.environ.get("SYNTH_HEALTHBENCH_DATASET_PATH", "").strip()
    if override:
        text = Path(override).read_text(encoding="utf-8")
    else:
        cache = Path(
            os.environ.get(
                "SYNTH_HEALTHBENCH_CACHE_PATH",
                str(Path.home() / ".synth" / "datasets" / "healthbench_eval.jsonl"),
            )
        )
        if not cache.is_file():
            response = httpx.get(DATASET_URL, timeout=120.0, follow_redirects=True)
            response.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            temporary.replace(cache)
        text = cache.read_text(encoding="utf-8")
    parsed = tuple(json.loads(line) for line in text.splitlines() if line.strip())
    if not parsed:
        raise RuntimeError("healthbench_dataset_empty")
    return parsed


def load_row(seed: int) -> dict[str, Any] | None:
    dataset = rows()
    return dataset[seed] if 0 <= seed < len(dataset) else None


def public_observation(row: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "seed": seed,
        "prompt_id": str(row.get("prompt_id") or ""),
        "messages": row.get("prompt") or [],
        "rubric_count": len(row.get("rubrics") or []),
        "example_tags": row.get("example_tags") or [],
    }
