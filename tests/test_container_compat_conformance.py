"""PR CI wrapper for containers-compat C0–C8 (in-process stubs, no --paid)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from synth_containers.platform import PR_TARGETS, create_compat_app
from tests.conformance.container_compat.run import receipt_from_suite, run_against_client


@pytest.mark.parametrize("target", PR_TARGETS)
def test_container_compat_pr_target(target: str, tmp_path) -> None:
    with TestClient(create_compat_app(target, storage_root=tmp_path / "p0")) as client:
        suite = run_against_client(client, target, paid=False)
    receipt = receipt_from_suite(suite)
    failed = receipt["failed"]
    assert not failed, json.dumps({"target": target, "failed": failed, "details": receipt["details"]}, indent=2)
    assert receipt.get("stream_descriptor_digest"), json.dumps(
        {"target": target, "missing": "stream_descriptor_digest"}, indent=2
    )
    assert receipt.get("trace_v5_digest"), json.dumps(
        {"target": target, "missing": "trace_v5_digest"}, indent=2
    )
