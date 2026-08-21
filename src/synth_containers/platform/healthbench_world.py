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
    parsed = _parse_json_records(text)
    if not parsed:
        raise RuntimeError("healthbench_dataset_empty")
    return parsed


def _parse_json_records(text: str) -> tuple[dict[str, Any], ...]:
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index].isspace():
            index += 1
        if index >= length:
            break
        record, end = decoder.raw_decode(text, index)
        if not isinstance(record, dict):
            raise RuntimeError("healthbench_dataset_invalid_record")
        records.append(record)
        index = end
    return tuple(records)


def load_row(seed: int) -> dict[str, Any] | None:
    dataset = rows()
    return dataset[seed] if 0 <= seed < len(dataset) else None


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the public chat turns. Professional rows use conversation.messages."""
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        return [item for item in prompt if isinstance(item, dict)]
    conversation = row.get("conversation")
    if isinstance(conversation, dict):
        messages = conversation.get("messages")
        if isinstance(messages, list):
            return [item for item in messages if isinstance(item, dict)]
    messages = row.get("messages")
    if isinstance(messages, list):
        return [item for item in messages if isinstance(item, dict)]
    return []


def rubric_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize OSS `rubrics` and Professional `rubric_items` to criterion/points."""
    raw = row.get("rubrics") or row.get("rubric_items") or []
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        criterion = item.get("criterion") or item.get("criterion_text")
        normalized.append({**item, "criterion": criterion})
    return normalized


def public_observation(row: dict[str, Any], seed: int) -> dict[str, Any]:
    messages = prompt_messages(row)
    return {
        "seed": seed,
        "prompt_id": str(row.get("prompt_id") or row.get("id") or ""),
        "messages": messages,
        "rubric_count": len(rubric_items(row)),
        "example_tags": row.get("example_tags") or [],
    }
