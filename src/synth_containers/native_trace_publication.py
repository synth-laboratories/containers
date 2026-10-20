"""Publish existing sealed native Trace V5 through the public trace-store API.

This adapter never constructs a trace from operator events or rewrites a sealed
identity to redact it. Native frame attachments remain part of the original
bundle; normal-result publication refuses missing/partial required evidence.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .bounded_artifacts import collect_bounded_artifacts
from .platform.trace_bundle import _verify_frame_artifact_bindings
from .tracing.capture.redaction import assert_no_secrets
from .tracing.inspection import inspect_trace_input
from .tracing.store.bundle import LocalTraceBundle


class NativeTraceStore(Protocol):
    """Implemented by the SDK's AsyncFactoryTraceStoreAPI; credentials stay there."""

    factory_id: str

    async def upload_bundle(self, root: Path, **kwargs: Any) -> Any: ...
    async def query(self, **kwargs: Any) -> Any: ...
    async def bundle_download(self, publication_id: str, **kwargs: Any) -> Any: ...


def _json(path: Path, maximum: int = 1024 * 1024) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Native evidence metadata exceeds bound")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Native evidence metadata must be an object")
    return value


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Native evidence object path is not bundle-relative")
    return value


def _durable(path: Path, payload: Mapping[str, Any]):
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("Existing native publication receipt differs")
        return
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def freeze_native_trace_evidence(
    source: Path,
    destination: Path,
    *,
    expected_manifest_digest: str,
    expected_trace_digests: tuple[str, ...],
    max_bytes: int,
    required_visual_digests: tuple[str, ...] = (),
    allow_interrupted: bool = False,
) -> dict:
    """Bounded exact-byte copy plus semantic validation; no synthetic completion.

    Interrupted bundles retain their true capture status. Only explicit
    allow_interrupted permits them, and the returned receipt stays incomplete.
    Missing required visuals always fail because their source cannot be invented.
    """
    for digest in (expected_manifest_digest, *expected_trace_digests, *required_visual_digests):
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError("Native evidence requires explicit SHA-256 identities")
    if not expected_trace_digests or len(set(expected_trace_digests)) != len(
        expected_trace_digests
    ):
        raise ValueError("Native evidence requires unique trace identities")
    source = source.resolve()
    pointer = _json(source / "manifest.json")
    manifest = (
        _json(source / _relative(pointer["relative_path"]))
        if "relative_path" in pointer
        else pointer
    )
    if manifest.get("content_digest") != expected_manifest_digest:
        raise ValueError("Native evidence manifest pin mismatch")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or not objects or len(objects) > 9998:
        raise ValueError("Native evidence requires a bounded object inventory")
    paths = {"manifest.json"}
    if "relative_path" in pointer:
        paths.add(_relative(pointer["relative_path"]))
    for item in objects:
        if not isinstance(item, dict) or item.get("immutable") is not True:
            raise ValueError("Native evidence requires immutable declared objects")
        paths.add(_relative(item["path"]))
    sources = {}
    for name in paths:
        path = source / name
        if not path.resolve().is_relative_to(source):
            raise ValueError("Native evidence object escapes source bundle")
        sources["bundle/" + name] = path
    collection = collect_bounded_artifacts(
        sources, destination, max_bytes=max_bytes, max_files=10000
    )
    return _validate_frozen(
        destination,
        expected_manifest_digest=expected_manifest_digest,
        expected_trace_digests=expected_trace_digests,
        max_bytes=max_bytes,
        required_visual_digests=required_visual_digests,
        allow_interrupted=allow_interrupted,
        collection=collection,
    )


def _validate_frozen(
    destination,
    *,
    expected_manifest_digest,
    expected_trace_digests,
    max_bytes,
    required_visual_digests,
    allow_interrupted,
    collection,
):
    bundle_root = destination / "bundle"
    inspection = inspect_trace_input(bundle_root)
    if not (
        inspection.trusted
        and inspection.validation.valid
        and inspection.self_contained
        and inspection.compatibility == "native"
        and inspection.bundle_digest == expected_manifest_digest
    ):
        raise ValueError("Native evidence did not pass sealed bundle validation")
    if {item.trace_digest for item in inspection.traces} != set(expected_trace_digests):
        raise ValueError("Native evidence trace pins mismatch")
    bundle = LocalTraceBundle(bundle_root)
    available_visuals = set()
    incomplete = []
    for trace in inspection.traces:
        document = bundle.read_trace(trace.trace_digest)
        assert_no_secrets(document, where="native sealed trace publication")
        _verify_frame_artifact_bindings(bundle, document)
        for artifact in document.get("artifacts", []):
            if str(artifact.get("media_type", "")).startswith(("image/", "video/")):
                digest = artifact.get("digest")
                if digest in required_visual_digests:
                    if artifact.get("uri") != bundle.blobs.uri(digest):
                        raise ValueError("Required native visual is not embedded in bundle custody")
                    body = bundle.blobs.get(digest)
                    if artifact.get("size_bytes") != len(body):
                        raise ValueError(
                            "Required native visual size differs from sealed declaration"
                        )
                available_visuals.add(digest)
        if trace.capture_status != "complete" or trace.lifecycle_status not in {
            "completed",
            "failed",
        }:
            incomplete.append(trace.trace_digest)
    if set(required_visual_digests) - available_visuals:
        raise ValueError("Native evidence required visual attachments are missing")
    if incomplete and not allow_interrupted:
        raise ValueError("Normal result requires complete terminal native trace capture")
    receipt = {
        "schema_version": "synth.native-evidence-freeze.v1",
        "manifest_digest": expected_manifest_digest,
        "trace_digests": sorted(expected_trace_digests),
        "required_visual_digests": sorted(required_visual_digests),
        "incomplete_trace_digests": incomplete,
        "complete": not incomplete,
        "payload_bytes": collection["payload_bytes"],
        "max_bytes": max_bytes,
    }
    _durable(destination / "freeze-receipt.json", receipt)
    return receipt


async def _reconcile_committed(
    store, *, run_id, project_id, manifest_digest, trace_digests, bundle_id, freeze
):
    def wire(value):
        return value.to_wire() if hasattr(value, "to_wire") else value

    async with asyncio.timeout(60):
        query = wire(await store.query(run_id=run_id, project_id=project_id, limit=100))
        if query.get("factory_id") != store.factory_id or query.get("count", 101) > 100:
            raise ValueError("Native publication lookup is ambiguous or exceeds bound")
        matches = [
            row
            for row in query.get("traces", [])
            if row.get("manifest_digest") == manifest_digest and row.get("run_id") == run_id
        ]
        if {row.get("trace_digest") for row in matches} != set(trace_digests):
            raise TimeoutError("Expired publication has no complete committed run binding")
        publication_ids = {row["publication_id"] for row in matches}
        if len(publication_ids) != 1:
            raise ValueError("Native publication lookup has conflicting identities")
        publication_id = publication_ids.pop()
        # The backend download endpoint only returns committed publications and
        # verifies store/tenant ownership; URLs returned here are never retained.
        descriptor = wire(await store.bundle_download(publication_id, expires_in_seconds=60))
    if (
        descriptor.get("publication_id") != publication_id
        or descriptor.get("manifest_digest") != manifest_digest
        or descriptor.get("bundle_id") != bundle_id
        or descriptor.get("receipt", {}).get("factory_id") != store.factory_id
    ):
        raise ValueError("Committed publication lookup differs from the original pins")
    result = {
        "schema_version": "synth.native-trace-publication.v1",
        "run_id": run_id,
        "project_id": project_id,
        "factory_id": store.factory_id,
        "manifest_digest": manifest_digest,
        "trace_digests": sorted(trace_digests),
        "status": "committed",
        "evidence_complete": freeze["complete"],
        "promotion_receipt_missing": True,
        "publication": {
            "publication_id": publication_id,
            "bundle_id": bundle_id,
            "manifest_digest": manifest_digest,
            "reconciled_read_only": True,
            "access_receipt": descriptor["receipt"],
        },
    }
    assert_no_secrets(result, where="reconciled native publication custody")
    return result


async def publish_native_trace_evidence(
    source: Path,
    output: Path,
    *,
    store: NativeTraceStore,
    run_id: str,
    expected_manifest_digest: str,
    expected_trace_digests: tuple[str, ...],
    max_bytes: int,
    timeout_seconds: float = 120,
    required_visual_digests: tuple[str, ...] = (),
    allow_interrupted: bool = False,
    project_id: str | None = None,
) -> dict:
    """Freeze, then use existing SDK prepare/upload/finalize and commit receipt last.

    Transfer metadata and credentials never enter this receipt directory. An
    interrupted upload can be retried under the SDK's manifest idempotency key;
    it does not re-run work or manufacture a successful publication receipt.
    """
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 1800
    ):
        raise ValueError("Native publication timeout must be at most 1800 seconds")
    factory_id = getattr(store, "factory_id", None)
    if (
        not isinstance(factory_id, str)
        or not factory_id
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise ValueError("Native publication requires bound factory and run identities")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "publication-owner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = {
            "schema_version": "synth.native-trace-publication-claim.v1",
            "run_id": run_id,
            "manifest_digest": expected_manifest_digest,
            "trace_digests": sorted(expected_trace_digests),
            "max_bytes": max_bytes,
            "required_visual_digests": sorted(required_visual_digests),
            "allow_interrupted": allow_interrupted,
            "project_id": project_id,
            "factory_id": factory_id,
        }
        assert_no_secrets(claim, where="native evidence publication claim")
        claim_path = output / "publication-claim.json"
        claim["timeout_seconds"] = timeout_seconds
        if claim_path.exists():
            retained = _json(claim_path)
            if any(retained.get(key) != value for key, value in claim.items()):
                raise ValueError("Native publication cannot reset its admitted inputs or allowance")
            deadline = datetime.fromisoformat(retained["deadline"])
            if deadline.tzinfo is None:
                raise ValueError("Native publication expiry lacks a timezone")
        else:
            deadline = datetime.now(UTC) + timedelta(seconds=timeout_seconds)
            claim["deadline"] = deadline.isoformat()
            _durable(claim_path, claim)
        monotonic_deadline = (
            asyncio.get_running_loop().time() + (deadline - datetime.now(UTC)).total_seconds()
        )
        frozen = output / "frozen"
        if frozen.exists():
            # A crash after all payload writes but before the freeze receipt is
            # recoverable from the pinned sealed bundle, without copying again.
            bundle_root = frozen / "bundle"
            bundle = LocalTraceBundle(bundle_root)
            frozen_manifest = bundle.read_manifest()
            if frozen_manifest.get("content_digest") != expected_manifest_digest:
                raise ValueError("Native publication frozen manifest changed")
            pointer = _json(bundle_root / "manifest.json")
            paths = {"manifest.json"}
            if "relative_path" in pointer:
                paths.add(_relative(pointer["relative_path"]))
            for item in frozen_manifest.get("objects", []):
                paths.add(_relative(item["path"]))
            if len(paths) > 10000:
                raise ValueError("Frozen object inventory exceeds admission")
            payload_bytes = 0
            for name in paths:
                path = bundle_root / name
                if path.is_symlink() or not path.resolve().is_relative_to(bundle_root.resolve()):
                    raise ValueError("Frozen object escapes custody")
                payload_bytes += path.stat().st_size
            if payload_bytes > max_bytes:
                raise ValueError("Frozen evidence exceeds admitted bytes")
            freeze = _validate_frozen(
                frozen,
                expected_manifest_digest=expected_manifest_digest,
                expected_trace_digests=expected_trace_digests,
                max_bytes=max_bytes,
                required_visual_digests=required_visual_digests,
                allow_interrupted=allow_interrupted,
                collection={"payload_bytes": payload_bytes},
            )
        else:
            freeze = freeze_native_trace_evidence(
                source,
                frozen,
                expected_manifest_digest=expected_manifest_digest,
                expected_trace_digests=expected_trace_digests,
                max_bytes=max_bytes,
                required_visual_digests=required_visual_digests,
                allow_interrupted=allow_interrupted,
            )
        receipt_path = output / "publication-receipt.json"
        if receipt_path.exists():
            receipt = _json(receipt_path)
            if (
                receipt.get("manifest_digest") != expected_manifest_digest
                or receipt.get("run_id") != run_id
                or receipt.get("project_id") != project_id
                or receipt.get("factory_id") != factory_id
            ):
                raise ValueError("Native publication receipt pin mismatch")
            return receipt
        remaining = min(
            (deadline - datetime.now(UTC)).total_seconds(),
            monotonic_deadline - asyncio.get_running_loop().time(),
        )
        promotion_path = output / "promotion-custody.json"
        if remaining <= 0 and not promotion_path.exists():
            receipt = await _reconcile_committed(
                store,
                run_id=run_id,
                project_id=project_id,
                manifest_digest=expected_manifest_digest,
                trace_digests=expected_trace_digests,
                bundle_id=LocalTraceBundle(frozen / "bundle").read_manifest()["bundle_id"],
                freeze=freeze,
            )
            _durable(receipt_path, receipt)
            return receipt
        if promotion_path.exists():
            value = _json(promotion_path)
        else:
            async with asyncio.timeout(remaining):
                committed = await store.upload_bundle(
                    frozen / "bundle",
                    project_id=project_id,
                    run_id=run_id,
                    metadata={
                        "native_evidence_complete": freeze["complete"],
                        "incomplete_trace_digests": freeze["incomplete_trace_digests"],
                    },
                    transfer_timeout_seconds=remaining,
                )
            value = committed.to_wire() if hasattr(committed, "to_wire") else committed
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "synth.trace-promotion-receipt.v1"
            or value.get("manifest_digest") != expected_manifest_digest
            or value.get("factory_id") != factory_id
            or value.get("bundle_id")
            != LocalTraceBundle(frozen / "bundle").read_manifest().get("bundle_id")
            or set(value.get("trace_digests", [])) != set(expected_trace_digests)
            or not value.get("committed_at")
            or not value.get("receipt_digest")
        ):
            raise ValueError("Trace store did not confirm the pinned native publication")
        assert_no_secrets(value, where="native trace promotion receipt")
        # Artifact custody survives catalog/run-link outages independently. This
        # original receipt has no invented run/project attestation.
        _durable(promotion_path, value)
        binding = await _reconcile_committed(
            store,
            run_id=run_id,
            project_id=project_id,
            manifest_digest=expected_manifest_digest,
            trace_digests=expected_trace_digests,
            bundle_id=value["bundle_id"],
            freeze=freeze,
        )
        if binding["publication"]["publication_id"] != value.get("publication_id"):
            raise ValueError("Trace promotion is not bound to the requested run/project")
        receipt = {
            "schema_version": "synth.native-trace-publication.v1",
            "run_id": run_id,
            "manifest_digest": expected_manifest_digest,
            "trace_digests": sorted(expected_trace_digests),
            "status": "committed",
            "evidence_complete": freeze["complete"],
            "factory_id": factory_id,
            "project_id": project_id,
            "publication": value,
            "run_binding_access_receipt": binding["publication"]["access_receipt"],
        }
        _durable(receipt_path, receipt)
        return receipt
