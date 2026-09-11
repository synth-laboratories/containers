"""Freeze the exact bounded build context used to prepare a Harbor image."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from .harbor_environment import (
    _MAX_TREE_BYTES,
    HarborEnvironmentDraft,
    HarborEnvironmentError,
    _tree_digest,
    _tree_files,
)
from .harbor_task_stage import _copy_bounded_file


@dataclass(frozen=True)
class HarborImageContext:
    root: Path
    source_package_digest: str
    context_digest: str

    def verify(self) -> None:
        if not (self.root / "Dockerfile").is_file():
            raise HarborEnvironmentError("harbor_image_context_dockerfile_missing")
        if _tree_digest(self.root) != self.context_digest:
            raise HarborEnvironmentError("harbor_image_context_changed")

    def as_dict(self) -> dict:
        return {
            "schema_version": "synth.harbor-image-context.v1",
            "root": str(self.root),
            "source_package_digest": self.source_package_digest,
            "context_digest": self.context_digest,
        }


def freeze_harbor_image_context(
    draft: HarborEnvironmentDraft, destination: Path
) -> HarborImageContext:
    """Copy only environment/, refuse mutation, and persist a receipt last.

    This does not rewrite Dockerfiles, resolve mutable base tags, run task code,
    or establish that an existing image was built from this context.
    """
    source = draft.root / "environment"
    destination = destination.expanduser().absolute()
    if destination.resolve().is_relative_to(draft.root):
        raise HarborEnvironmentError("harbor_image_context_cannot_mutate_source")
    if _tree_digest(draft.root) != draft.source_package_digest:
        raise HarborEnvironmentError("harbor_image_source_changed")
    if not (source / "Dockerfile").is_file():
        raise HarborEnvironmentError("harbor_image_context_dockerfile_missing")
    expected = _tree_digest(source)
    destination.mkdir(parents=True, exist_ok=False)
    context = destination / "context"
    context.mkdir()
    copied = 0
    for item in _tree_files(source):
        target = context / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        copied += _copy_bounded_file(item, target, _MAX_TREE_BYTES - copied)
    frozen = HarborImageContext(context, draft.source_package_digest, expected)
    frozen.verify()
    if _tree_digest(draft.root) != draft.source_package_digest:
        raise HarborEnvironmentError("harbor_image_source_changed_during_copy")
    for directory, _, _ in os.walk(context, topdown=False):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    with (destination / "context-receipt.json").open("x") as handle:
        json.dump(frozen.as_dict(), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return frozen
