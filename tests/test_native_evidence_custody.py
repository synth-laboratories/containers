"""Partial copies cannot masquerade as committed native evidence."""

import pytest

from synth_containers.bounded_artifacts import (
    ArtifactAllowanceExceeded,
    collect_bounded_artifacts,
)
from synth_containers.native_trace_publication import freeze_native_trace_evidence
from synth_containers.tracing.store.bundle import LocalTraceBundle
from tests.test_harbor_trace_bundle import _frame_bundle
from tests.test_trace_v5_inspection import _bundle


def test_over_budget_collection_keeps_prefix_without_committing(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"abc")
    second.write_bytes(b"def")
    destination = tmp_path / "retained"
    with pytest.raises(ArtifactAllowanceExceeded):
        collect_bounded_artifacts({"a": first, "b": second}, destination, max_bytes=5)
    assert (destination / "a").read_bytes() == b"abc"
    assert not (destination / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        collect_bounded_artifacts({"a": first}, destination, max_bytes=5)


def test_native_freeze_preserves_exact_sealed_identity(tmp_path):
    archive = _frame_bundle(tmp_path)
    bundle = LocalTraceBundle.extract_archive(archive, tmp_path / "source")
    manifest = bundle.read_manifest()
    digest = manifest["traces"][0]["trace_digest"]
    receipt = freeze_native_trace_evidence(
        bundle.root, tmp_path / "frozen",
        expected_manifest_digest=manifest["content_digest"],
        expected_trace_digests=(digest,), max_bytes=1024 * 1024,
    )
    assert receipt["trace_digests"] == [digest]
    assert receipt["manifest_digest"] == manifest["content_digest"]
    assert (tmp_path / "frozen/bundle/manifest.json").read_bytes() == (
        bundle.root / "manifest.json"
    ).read_bytes()


def test_partial_capture_requires_opt_in_and_stays_incomplete(tmp_path):
    bundle = _bundle(tmp_path / "source")
    manifest = bundle.read_manifest()
    args = {
        "expected_manifest_digest": manifest["content_digest"],
        "expected_trace_digests": (manifest["traces"][0]["trace_digest"],),
        "max_bytes": 1024 * 1024,
    }
    with pytest.raises(ValueError, match="complete terminal native trace capture"):
        freeze_native_trace_evidence(bundle.root, tmp_path / "refused", **args)
    receipt = freeze_native_trace_evidence(
        bundle.root, tmp_path / "partial", allow_interrupted=True, **args
    )
    assert receipt["complete"] is False
    assert receipt["incomplete_trace_digests"] == list(args["expected_trace_digests"])


def test_required_visuals_cannot_be_invented(tmp_path):
    bundle = _bundle(tmp_path / "source")
    manifest = bundle.read_manifest()
    with pytest.raises(ValueError, match="required visual attachments are missing"):
        freeze_native_trace_evidence(
            bundle.root, tmp_path / "frozen",
            expected_manifest_digest=manifest["content_digest"],
            expected_trace_digests=(manifest["traces"][0]["trace_digest"],),
            required_visual_digests=("sha256:" + "a" * 64,), max_bytes=1024 * 1024,
        )
    assert not (tmp_path / "frozen/freeze-receipt.json").exists()
