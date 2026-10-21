"""Read-only closure checks and portable export; no decoder or origin is invoked.

See docs/specs/portable-checkpoint-v1.md. This adds metadata and byte custody to
the existing ApplicationCheckpoint, not another checkpoint execution authority.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from ..package_files import ContainerPackageError, read_package_file
from ..package_models import content_digest
from .contracts import ApplicationCheckpoint, ArtifactReference, CheckpointFailure
from .portable_models import (
    CheckpointInspectionV1,
    CheckpointRestoreReportV1,
    CheckpointRestoreRequestV1,
    CheckpointRuntimeSupportV1,
    PortableCheckpointManifestV1,
)
from .storage import ArtifactReader, FileArtifactStore, verified

MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_checkpoint_json_key")
        value[key] = item
    return value


def _json(payload: bytes) -> object:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise CheckpointFailure("checkpoint_manifest_capacity")
    try:
        return json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError):
        raise CheckpointFailure("checkpoint_descriptor_invalid") from None


def read_checkpoint_descriptor(path: Path) -> object:
    """Bounded JSON only, with no symlinks or content-bearing error messages."""
    path = path.absolute()
    try:
        return _json(read_package_file(path.parent, path.name, maximum=MAX_MANIFEST_BYTES))
    except ContainerPackageError:
        raise CheckpointFailure("checkpoint_descriptor_unavailable") from None


class CheckpointArtifactReader:
    """Read-only existing local CAS. Construction never creates directories."""

    _prefix = ""

    def __init__(self, root: Path, *, namespace: str):
        self.root = root.absolute()
        self.namespace = namespace

    def put(self, payload: bytes) -> ArtifactReference:
        raise CheckpointFailure("checkpoint_bundle_read_only")

    def get(self, item: ArtifactReference) -> bytes:
        item = ArtifactReference.model_validate(item.model_dump(mode="python"))
        if item.namespace != self.namespace:
            raise CheckpointFailure("checkpoint_artifact_scope")
        try:
            data = read_package_file(self.root, self._prefix + item.object_key)
        except ContainerPackageError:
            raise CheckpointFailure("checkpoint_blob_unavailable") from None
        return verified(item, data)


class CheckpointBundleStore(CheckpointArtifactReader):
    """Read-only portable bundle; check every directory below its configured root."""

    _prefix = "blobs/"


def _closure(
    manifest: PortableCheckpointManifestV1, store: ArtifactReader
) -> tuple[tuple[ArtifactReference, ...], int]:
    # Reparse instances, including nested legacy artifact references. model_copy
    # and model_construct are not validation boundaries.
    manifest = PortableCheckpointManifestV1.model_validate(manifest.model_dump(mode="python"))
    application = ApplicationCheckpoint.model_validate(_json(verified(
        manifest.application, store.get(manifest.application)
    )))
    if application.checkpoint != manifest.checkpoint:
        raise CheckpointFailure("checkpoint_application_identity")
    if {row.cut.participant.name for row in application.participants} != {
        row.participant for row in manifest.components
    }:
        raise CheckpointFailure("checkpoint_participant_coverage")
    references = (
        manifest.application, manifest.execution.configuration, manifest.execution.policies,
        *(row.artifact for row in application.participants),
    )
    unique: dict[str, ArtifactReference] = {}
    for item in references:
        if item.namespace != manifest.application.namespace:
            raise CheckpointFailure("checkpoint_artifact_scope")
        previous = unique.setdefault(item.sha256, item)
        if previous != item:
            raise CheckpointFailure("checkpoint_artifact_reference_conflict")
    total = sum(item.size_bytes for item in unique.values())
    if total > MAX_BUNDLE_BYTES:
        raise CheckpointFailure("checkpoint_bundle_capacity")
    for item in unique.values():
        verified(item, store.get(item))
    return tuple(unique.values()), total


def inspect_portable_checkpoint(
    manifest: PortableCheckpointManifestV1, store: ArtifactReader
) -> CheckpointInspectionV1:
    manifest = PortableCheckpointManifestV1.model_validate(manifest.model_dump(mode="python"))
    references, total = _closure(manifest, store)
    return CheckpointInspectionV1(
        manifest=manifest, manifest_digest=manifest.digest(),
        verified_blobs=len(references), verified_bytes=total,
    )


def load_checkpoint_bundle(root: Path) -> CheckpointInspectionV1:
    manifest = PortableCheckpointManifestV1.model_validate(
        read_checkpoint_descriptor(root / "manifest.json")
    )
    return inspect_portable_checkpoint(
        manifest, CheckpointBundleStore(root, namespace=manifest.application.namespace)
    )


def _sync(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_checkpoint_bundle(
    manifest: PortableCheckpointManifestV1, store: ArtifactReader, destination: Path
) -> CheckpointInspectionV1:
    """Publish a new private directory: verified blobs first, discoverable manifest last.

    A failure leaves owned partial output for inspection; an existing destination
    is never overwritten or silently resumed. Generic custody remains with store.
    """
    manifest = PortableCheckpointManifestV1.model_validate(manifest.model_dump(mode="python"))
    references, _ = _closure(manifest, store)
    destination = destination.absolute()
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        raise CheckpointFailure("checkpoint_bundle_destination_exists") from None
    try:
        target = FileArtifactStore(destination / "blobs", namespace=manifest.application.namespace)
        for item in references:
            payload = verified(item, store.get(item))
            if target.put(payload) != item:
                raise CheckpointFailure("checkpoint_artifact_reference_conflict")
            (destination / "blobs" / item.object_key).chmod(0o600)
            with (destination / "blobs" / item.object_key).open("rb") as handle:
                os.fsync(handle.fileno())
        # Reopen via the portable reader before publishing the discoverability marker.
        inspection = inspect_portable_checkpoint(
            manifest, CheckpointBundleStore(destination, namespace=manifest.application.namespace)
        )
        _sync(destination / "blobs")
        _sync(destination)
        payload = json.dumps(manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        # Link a fully fsynced file: readers never see a partially written manifest.
        temporary = destination / ".manifest.pending"
        with temporary.open("xb") as handle:
            temporary.chmod(0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination / "manifest.json")
        _sync(destination)
        _sync(destination.parent)
        temporary.unlink()
        _sync(destination)
        return inspection
    except OSError:
        raise CheckpointFailure("checkpoint_bundle_publication_in_doubt") from None


def check_checkpoint_restore(
    manifest: PortableCheckpointManifestV1,
    store: ArtifactReader,
    request: CheckpointRestoreRequestV1,
    runtime_support: CheckpointRuntimeSupportV1,
) -> CheckpointRestoreReportV1:
    """Preflight declarations and bytes without deserializing state or admitting work."""
    manifest = PortableCheckpointManifestV1.model_validate(manifest.model_dump(mode="python"))
    request = CheckpointRestoreRequestV1.model_validate(request.model_dump(mode="python"))
    runtime_support = CheckpointRuntimeSupportV1.model_validate(runtime_support.model_dump(mode="python"))
    blockers: list[str] = []
    verified_blobs = verified_bytes = 0
    try:
        inspection = inspect_portable_checkpoint(manifest, store)
        verified_blobs, verified_bytes = inspection.verified_blobs, inspection.verified_bytes
    except CheckpointFailure as error:
        blockers.append(str(error))
    except (ValidationError, OSError):
        blockers.append("checkpoint_closure_invalid")
    if request.commit.manifest != manifest.digest() or (
        request.commit.sequence != manifest.sequence or request.commit.chain_head != manifest.chain_head
    ):
        blockers.append("checkpoint_commit_mismatch")
    if request.source_run_id != manifest.run_id or request.source_lineage_id != manifest.lineage_id:
        blockers.append("checkpoint_source_lineage_mismatch")
    if request.mode not in manifest.modes or request.mode not in runtime_support.modes:
        blockers.append("checkpoint_mode_unsupported")
    restore = request.mode in ("strict_continue", "same_policy_fork", "edited_policy_fork")
    if restore:
        if manifest.terminal:
            blockers.append("checkpoint_terminal_not_resumable")
        if manifest.unresolved_effects:
            blockers.append("checkpoint_unresolved_effects")
        if manifest.clock_policy != runtime_support.clock_policy:
            blockers.append("checkpoint_clock_policy_mismatch")
        source = manifest.execution.model_dump(mode="json")
        target = request.execution.model_dump(mode="json")
        if request.mode == "edited_policy_fork":
            source.pop("policies")
            target.pop("policies")
        if source != target:
            blockers.append("checkpoint_execution_identity_mismatch")
    if request.mode != "inspect":
        actual = {item.participant: item for item in manifest.components}
        required = {item.participant: item for item in runtime_support.components}
        if actual.keys() != required.keys():
            blockers.append("checkpoint_participant_set_mismatch")
        for name, expected in required.items():
            component = actual.get(name)
            if component is None:
                continue
            if component.codec != expected.codec:
                blockers.append("checkpoint_codec_incompatible:" + name)
            if not set(expected.coverage).issubset(component.coverage):
                blockers.append("checkpoint_component_coverage_missing:" + name)
            if component.codec not in runtime_support.trusted_decoders:
                blockers.append("checkpoint_decoder_untrusted:" + name)
        if not set(runtime_support.required_cursors).issubset(item.name for item in manifest.cursors):
            blockers.append("checkpoint_cursor_coverage_missing")
    # An anchor varies per candidate; all other request/policy facts must match
    # before reports can participate in one strict-position selection.
    selection = request.model_dump(mode="json", exclude={"commit"})
    return CheckpointRestoreReportV1(
        manifest=manifest.digest(), request=content_digest(request.model_dump(mode="json")),
        runtime_support=content_digest(runtime_support.model_dump(mode="json")),
        selection_scope=content_digest({"request": selection, "runtime_support": runtime_support.model_dump(mode="json")}),
        mode=request.mode, checkpoint=manifest.checkpoint, sequence=manifest.sequence,
        chain_head=manifest.chain_head, declaration_supported=not blockers,
        blockers=tuple(sorted(set(blockers))), verified_blobs=verified_blobs,
        verified_bytes=verified_bytes, retained_liabilities=manifest.liabilities,
    )


def select_strict_checkpoint(
    reports: Sequence[CheckpointRestoreReportV1],
) -> CheckpointRestoreReportV1 | None:
    """Choose the latest compatible owner-anchored position, never reward or wall time.

    Reports are local preflight values, not signed execution grants. The owning
    runtime must still reconcile the suffix, resources, grants and live fences.
    """
    checked = [CheckpointRestoreReportV1.model_validate(row.model_dump(mode="python")) for row in reports]
    if any(row.mode != "strict_continue" for row in checked):
        raise CheckpointFailure("checkpoint_selection_requires_strict_mode")
    if len({row.selection_scope for row in checked}) > 1:
        raise CheckpointFailure("checkpoint_selection_scope_mismatch")
    eligible = [row for row in checked if row.declaration_supported]
    if not eligible:
        return None
    position = max(row.sequence for row in eligible)
    latest = {row.manifest: row for row in eligible if row.sequence == position}
    if len(latest) != 1:
        raise CheckpointFailure("checkpoint_committed_position_ambiguous")
    return next(iter(latest.values()))
