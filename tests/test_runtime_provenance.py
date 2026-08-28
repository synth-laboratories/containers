"""Immutable runtime provenance advertised by launched HTTP containers."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from synth_containers.metadata import attach_runtime_provenance
from synth_containers.platform.app import create_compat_app


IMAGE_DIGEST = "sha256:" + ("ab" * 32)
PRODUCER_REVISION = "cd" * 20


def test_runtime_provenance_uses_stable_top_level_info_paths() -> None:
    payload = attach_runtime_provenance(
        {"capabilities": {}},
        environ={
            "SYNTH_CONTAINER_IMAGE_DIGEST": IMAGE_DIGEST,
            "SYNTH_CONTAINER_PRODUCER_SOURCE_REVISION": PRODUCER_REVISION,
        },
    )

    assert payload["imageDigest"] == IMAGE_DIGEST
    assert payload["producerSourceRevision"] == PRODUCER_REVISION


@pytest.mark.parametrize(
    ("environment", "error"),
    [
        ({"SYNTH_CONTAINER_IMAGE_DIGEST": "sha256:not-a-digest"}, "container_image_digest_invalid"),
        ({"SYNTH_CONTAINER_PRODUCER_SOURCE_REVISION": "main"}, "container_producer_source_revision_invalid"),
    ],
)
def test_runtime_provenance_rejects_unverifiable_identity(
    environment: dict[str, str], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        attach_runtime_provenance({}, environ=environment)


def test_compat_info_advertises_injected_runtime_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SYNTH_CONTAINER_IMAGE_DIGEST", IMAGE_DIGEST)
    monkeypatch.setenv("SYNTH_CONTAINER_PRODUCER_SOURCE_REVISION", PRODUCER_REVISION)

    response = TestClient(create_compat_app("openenv_echo", storage_root=tmp_path)).get("/info")

    assert response.status_code == 200
    assert response.json()["imageDigest"] == IMAGE_DIGEST
    assert response.json()["producerSourceRevision"] == PRODUCER_REVISION
