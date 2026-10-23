"""Explicit local owner roots; custody registration does not grant execution."""

from __future__ import annotations

import json
from pathlib import Path

from .package_models import content_digest, content_bytes
from .tracing.atomic import durable_write, sync_directory


def _claim(checkpoint_id: str, owner_kind: str, owner_id: str) -> dict:
    if owner_kind not in {"continuation", "fork", "archive"} or not checkpoint_id or not owner_id:
        raise ValueError("checkpoint_retention_owner_invalid")
    return {"schema_version": "synth.checkpoint-retention-root.v1", "checkpoint_id": checkpoint_id,
            "owner_kind": owner_kind, "owner_id": owner_id}


def retain_checkpoint(root: Path, checkpoint_id: str, *, owner_kind: str, owner_id: str) -> str:
    claim = _claim(checkpoint_id, owner_kind, owner_id)
    digest = content_digest(claim)
    path = root / "checkpoint_roots" / content_digest(checkpoint_id)[7:] / (digest[7:] + ".json")
    durable_write(path, content_bytes(claim))
    return digest


def release_checkpoint_root(root: Path, checkpoint_id: str, *, owner_kind: str, owner_id: str):
    claim = _claim(checkpoint_id, owner_kind, owner_id)
    path = root / "checkpoint_roots" / content_digest(checkpoint_id)[7:] / (content_digest(claim)[7:] + ".json")
    if path.exists():
        if json.loads(path.read_bytes()) != claim:
            raise ValueError("checkpoint_retention_identity_conflict")
        path.unlink()
        sync_directory(path.parent)


def checkpoint_is_retained(root: Path, checkpoint_id: str) -> bool:
    paths = list((root / "checkpoint_roots" / content_digest(checkpoint_id)[7:]).glob("*.json"))
    for path in paths:
        claim = json.loads(path.read_bytes())
        if claim["checkpoint_id"] != checkpoint_id or content_digest(claim)[7:] != path.stem:
            raise ValueError("checkpoint_retention_registry_corrupt")
    return bool(paths)
