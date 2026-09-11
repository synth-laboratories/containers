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

    async def upload_bundle(self, root: Path, **kwargs: Any) -> Any: ...


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
            freeze = _json(frozen / "freeze-receipt.json")
            if freeze.get("manifest_digest") != expected_manifest_digest or freeze.get(
                "trace_digests"
            ) != sorted(expected_trace_digests):
                raise ValueError("Native publication frozen receipt changed")
            inspection = inspect_trace_input(frozen / "bundle")
            if not inspection.trusted or inspection.bundle_digest != expected_manifest_digest:
                raise ValueError("Native publication frozen bundle changed")
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
            if receipt.get("manifest_digest") != expected_manifest_digest:
                raise ValueError("Native publication receipt pin mismatch")
            return receipt
        remaining = min(
            (deadline - datetime.now(UTC)).total_seconds(),
            monotonic_deadline - asyncio.get_running_loop().time(),
        )
        if remaining <= 0:
            raise TimeoutError("Native publication original allowance expired; reconcile custody")
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
            or set(value.get("trace_digests", [])) != set(expected_trace_digests)
            or not value.get("committed_at")
            or not value.get("receipt_digest")
        ):
            raise ValueError("Trace store did not confirm the pinned native publication")
        assert_no_secrets(value, where="native trace promotion receipt")
        receipt = {
            "schema_version": "synth.native-trace-publication.v1",
            "run_id": run_id,
            "manifest_digest": expected_manifest_digest,
            "trace_digests": sorted(expected_trace_digests),
            "status": "committed",
            "evidence_complete": freeze["complete"],
            "publication": value,
        }
        _durable(receipt_path, receipt)
        return receipt
