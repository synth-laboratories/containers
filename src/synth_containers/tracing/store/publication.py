"""Portable publication outbox over existing generic BlobStore custody.

No rollout or grader is callable from this path. Hosted services own authentication,
retention registration, admission, indexing policy and discovery of commit roots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field

from synth_containers.package_models import Digest, Name, PackageModel, portable_path

from ..atomic import durable_write
from ..canonical import bytes_digest, canonical_bytes, content_digest
from .base import BlobStore
from .bundle import LocalTraceBundle
from .filesystem import FilesystemBlobStore


class PublicationPlanV1(PackageModel):
    schema_version: Literal["synth.trace-publication.v1"] = "synth.trace-publication.v1"
    operation_id: Name
    attribution: dict[str, str]
    bundle_id: str
    bundle_manifest_digest: Digest
    archive_digest: Digest
    archive_bytes: int = Field(strict=True, ge=0)
    trace_digests: tuple[Digest, ...]
    evidence_digests: tuple[Digest, ...]
    retention_roots: tuple[Digest, ...]

    @property
    def commit_digest(self) -> str:
        return bytes_digest(canonical_bytes(self.model_dump(mode="json")))


class PublicationReceiptV1(PackageModel):
    schema_version: Literal["synth.trace-publication-receipt.v1"] = "synth.trace-publication-receipt.v1"
    operation_id: Name
    commit_digest: Digest
    local_durable: bool = True
    uploaded: bool
    manifest_committed: bool
    indexed: bool
    blocked_stage: Literal["upload", "commit", "index"] | None = None
    error_type: str | None = None
    # Do not retain arbitrary exception text: remote errors can contain credentials.


def load_publication(outbox: Path, operation_id: str) -> PublicationPlanV1:
    portable_path(operation_id)
    plan = PublicationPlanV1.model_validate_json((outbox / f"{operation_id}.json").read_bytes())
    if plan.operation_id != operation_id:
        raise ValueError("publication operation identity mismatch")
    return plan


def prepare_publication(
    bundle: LocalTraceBundle, outbox: Path, *, operation_id: str, attribution: dict[str, str],
    max_archive_bytes: int = 64 * 1024 * 1024,
) -> PublicationPlanV1:
    """Freeze a verified self-contained cut; the outbox survives source cleanup."""
    portable_path(operation_id)
    manifest = bundle.read_manifest()
    if type(max_archive_bytes) is not int or not 1 <= max_archive_bytes <= 1024 * 1024 * 1024:
        raise ValueError("publication archive allowance outside supported bounds")
    objects = manifest.get("objects") or []
    if len(objects) > 10000 or sum(item["byte_size"] for item in objects) > max_archive_bytes:
        raise ValueError("publication object closure exceeds allowance")
    if not manifest.get("self_contained", False):
        raise ValueError("publication requires self-contained retained evidence")
    archive = bundle.archive_bytes()
    if len(archive) > max_archive_bytes:
        raise ValueError("publication archive exceeds allowance")
    digest = bytes_digest(archive)
    plan = PublicationPlanV1(operation_id=operation_id, attribution=attribution,
        bundle_id=bundle.bundle_id, bundle_manifest_digest=content_digest(manifest),
        archive_digest=digest, archive_bytes=len(archive),
        trace_digests=tuple(item["trace_digest"] for item in manifest.get("traces") or ()),
        evidence_digests=tuple(item["bundle_digest"] for item in manifest.get("evidence") or ()),
        retention_roots=(digest,))
    store = FilesystemBlobStore(outbox / "objects")
    if store.put(archive) != digest:
        raise ValueError("publication archive identity mismatch")
    durable_write(outbox / f"{operation_id}.json", canonical_bytes(plan.model_dump(mode="json")))
    return plan


def publish(
    plan: PublicationPlanV1, outbox: Path, remote: BlobStore, *,
    index: Callable[[PublicationPlanV1, BlobStore], None],
) -> PublicationReceiptV1:
    """Replay upload/commit/index independently, with digest-verified readback.

    Index implementations must upsert by commit_digest. An index outage never
    invalidates an immutable commit, and never invokes execution again.
    """
    plan = PublicationPlanV1.model_validate(plan.model_dump(mode="python"))
    retained = load_publication(outbox, plan.operation_id)
    if retained != plan:
        raise ValueError("publication outbox identity conflict")
    local = FilesystemBlobStore(outbox / "objects")
    archive = local.get(plan.archive_digest)
    if len(archive) != plan.archive_bytes:
        raise ValueError("publication archive size mismatch")
    uploaded = committed = indexed = False
    stage: Literal["upload", "commit", "index"] = "upload"
    error_type = None
    try:
        result = remote.put_if_absent(archive, media_type="application/zip")
        if result.digest != plan.archive_digest or remote.get(plan.archive_digest) != archive:
            raise ValueError("uploaded archive readback mismatch")
        uploaded = True
        stage = "commit"
        commit = canonical_bytes(plan.model_dump(mode="json"))
        result = remote.put_if_absent(commit, media_type="application/vnd.synth.trace-publication+json")
        if result.digest != plan.commit_digest or remote.get(plan.commit_digest) != commit:
            raise ValueError("publication commit readback mismatch")
        committed = True
        stage = "index"
        index(plan, remote)
        indexed = True
    except Exception as exc:
        error_type = type(exc).__name__
    receipt = PublicationReceiptV1(operation_id=plan.operation_id, commit_digest=plan.commit_digest,
        uploaded=uploaded, manifest_committed=committed, indexed=indexed,
        blocked_stage=stage if not indexed else None, error_type=error_type)
    body = canonical_bytes(receipt.model_dump(mode="json"))
    durable_write(outbox / "receipts" / f"{bytes_digest(body)[7:]}.json", body)
    return receipt


def read_publication(remote: BlobStore, commit_digest: str, target: Path) -> LocalTraceBundle:
    """Restore only from committed roots, without any original source paths."""
    body = remote.get(commit_digest)
    if bytes_digest(body) != commit_digest:
        raise ValueError("publication commit digest mismatch")
    plan = PublicationPlanV1.model_validate(json.loads(body))
    if plan.commit_digest != commit_digest or plan.retention_roots != (plan.archive_digest,):
        raise ValueError("publication retention roots mismatch")
    archive = remote.get(plan.archive_digest)
    if bytes_digest(archive) != plan.archive_digest or len(archive) != plan.archive_bytes:
        raise ValueError("publication archive is missing or corrupt")
    # An owned sibling scratch file is removed even if safe extraction refuses.
    import tempfile
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".read-publication-", dir=target.parent) as scratch:
        source = Path(scratch) / "bundle.zip"
        source.write_bytes(archive)
        bundle = LocalTraceBundle.extract_archive(source, target)
    if content_digest(bundle.read_manifest()) != plan.bundle_manifest_digest:
        raise ValueError("publication bundle manifest identity mismatch")
    return bundle
