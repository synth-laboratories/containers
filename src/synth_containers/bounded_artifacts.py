"""Admit immutable artifact payloads before retaining bytes in a fresh directory."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


class ArtifactAllowanceExceeded(ValueError):
    pass


def collect_bounded_artifacts(
    sources: Mapping[str, Path], destination: Path, *, max_bytes: int, max_files: int = 1024
) -> dict:
    """Copy regular files without following final symlinks, committing manifest last.

    Payload bytes alone consume the allowance; bounded claim/manifest metadata is
    separately retained. Partial copies survive failures and are never a committed
    result. Source mutation invalidates publication. The destination is exclusively
    created, so retries cannot reset an existing artifact collection's allowance.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Artifact byte allowance must be positive")
    if type(max_files) is not int or not 1 <= max_files <= 10000:
        raise ValueError("Artifact file allowance must be from 1 through 10000")
    if len(sources) > max_files:
        raise ArtifactAllowanceExceeded("Artifact count exceeds admission")
    names = []
    for name in sources:
        path = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 1024
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or str(path) != name
            or name in {"manifest.json", "claim.json"}
        ):
            raise ValueError("Invalid artifact logical path")
        names.append(path)
    for path in names:
        if any(parent in names for parent in path.parents):
            raise ValueError("Artifact paths collide as file and directory")
    destination.mkdir(parents=True, exist_ok=False)
    claim = {
        "schema_version": "synth.artifact-collection-claim.v1",
        "max_bytes": max_bytes,
        "max_files": max_files,
        "enforcement": "collection_admission",
    }

    def durable_json(path, payload):
        with path.open("x") as handle:
            json.dump(payload, handle, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())

    durable_json(destination / "claim.json", claim)
    consumed = 0
    files = []
    for name, source in sources.items():
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as reader:
            original = os.fstat(reader.fileno())
            if not stat.S_ISREG(original.st_mode):
                raise ValueError("Only regular artifact files can be collected")
            if original.st_size > max_bytes - consumed:
                raise ArtifactAllowanceExceeded("Artifact payload exceeds remaining bytes")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as writer:
                while True:
                    body = reader.read(min(65536, max_bytes - consumed + 1))
                    if not body:
                        break
                    if len(body) > max_bytes - consumed:
                        raise ArtifactAllowanceExceeded("Artifact grew past remaining bytes")
                    writer.write(body)
                    consumed += len(body)
                    size += len(body)
                    digest.update(body)
                writer.flush()
                os.fsync(writer.fileno())
            final = os.fstat(reader.fileno())
            if (original.st_size, original.st_mtime_ns, original.st_ctime_ns) != (
                final.st_size,
                final.st_mtime_ns,
                final.st_ctime_ns,
            ):
                raise ValueError("Artifact source changed during collection")
            files.append({"path": name, "size_bytes": size, "sha256": digest.hexdigest()})
    manifest = {
        "schema_version": "synth.artifact-collection.v1",
        "files": files,
        "payload_bytes": consumed,
        "max_payload_bytes": max_bytes,
        "enforcement": "collection_admission",
        "complete": True,
    }
    # Sync every directory before committing the publication pointer.
    for root, _, _ in os.walk(destination, topdown=False):
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    durable_json(destination / "manifest.json", manifest)
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest
