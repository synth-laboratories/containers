"""Docker 29 successor metadata verifies without treating iidfile as config."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from synth_containers import docker_build_receipt
from synth_containers.docker_build_receipt import ImageReceiptError, inspect_docker_export
from synth_containers.harbor_translation import _build_script

FIXTURE = Path(__file__).parent / "fixtures/docker_build_receipt"
INDEX = "sha256:83dd767072db183910469afdf9502e35a12a5593a25ed0abd95238084155e955"
MANIFEST = "sha256:3909dbe95c7ab7ce375514d8aa7d878bae97e8932fd3143a96ab8a317cac9e97"
CONFIG = "sha256:a7dd95aba9d29cc83c34a59f0687d3febc4ff71353c3c30e8fc4ad03b7c1b5c1"


def _archive(*, single_manifest: bool = False, corrupt_config: bool = False) -> io.BytesIO:
    index = (FIXTURE / "image-index.json").read_bytes()
    layout = (FIXTURE / "archive-index.json").read_bytes()
    config = (FIXTURE / "config.json").read_bytes()
    if single_manifest:
        descriptor = next(
            row for row in json.loads(index)["manifests"] if row["digest"] == MANIFEST
        )
        layout = json.dumps({"schemaVersion": 2, "manifests": [descriptor]}).encode()
    if corrupt_config:
        config = bytes([config[0] ^ 1]) + config[1:]
    entries = {
        "blobs/sha256/" + INDEX[7:]: index,
        "blobs/sha256/" + MANIFEST[7:]: (FIXTURE / "manifest.json").read_bytes(),
        "blobs/sha256/" + CONFIG[7:]: config,
        "index.json": layout,
    }
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w") as archive:
        for name, data in entries.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    result.seek(0)
    return result


@pytest.mark.parametrize(
    ("single_manifest", "iid"), [(False, INDEX), (True, MANIFEST), (True, CONFIG)]
)
def test_build_id_projects_exact_successor_config(single_manifest, iid):
    artifact = inspect_docker_export(
        _archive(single_manifest=single_manifest), image_id=iid, platform="linux/amd64"
    )
    assert artifact.image_id == iid
    assert artifact.manifest_digest == MANIFEST
    assert artifact.config_digest == CONFIG


@pytest.mark.parametrize(
    ("single_manifest", "iid", "platform", "message"),
    [
        (False, INDEX, "linux/arm64", "one requested platform"),
        (True, MANIFEST, "linux/arm64", "config platform"),
        (False, "sha256:" + "f" * 64, "linux/amd64", "absent from"),
    ],
)
def test_build_receipt_refuses_unbound_platform_or_identity(
    single_manifest, iid, platform, message
):
    with pytest.raises(ImageReceiptError, match=message):
        inspect_docker_export(
            _archive(single_manifest=single_manifest), image_id=iid, platform=platform
        )


def test_build_receipt_refuses_tampered_config_bytes():
    with pytest.raises(ImageReceiptError, match="digest does not match"):
        inspect_docker_export(_archive(corrupt_config=True), image_id=INDEX, platform="linux/amd64")


def test_emitted_entrypoint_writes_verified_config_receipt(tmp_path):
    (tmp_path / "docker_build_receipt.py").write_bytes(
        Path(docker_build_receipt.__file__).read_bytes()
    )
    archive = tmp_path / "export.tar"
    archive.write_bytes(_archive().getvalue())
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"input_digest": "sha256:" + "1" * 64}))
    output = tmp_path / "build-output.json"
    mock_docker = tmp_path / "docker"
    mock_docker.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        'if [ "$1" = build ]; then\n'
        '  while [ "$1" != --iidfile ]; do shift; done\n'
        f"  printf '%s\\n' '{INDEX}' > \"$2\"\n"
        "else\n"
        f"  test \"$*\" = 'image save {INDEX}'\n"
        '  cat "$MOCK_DOCKER_EXPORT"\n'
        "fi\n"
    )
    mock_docker.chmod(0o755)
    script = tmp_path / "build.sh"
    script.write_text(_build_script("linux/amd64", ()))
    completed = subprocess.run(
        ["/bin/sh", str(script)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "MOCK_DOCKER_EXPORT": str(archive),
            "SYNTH_CONTAINER_PACKAGE_ROOT": str(tmp_path),
            "SYNTH_CONTAINER_BUILD_PLAN": str(plan),
            "SYNTH_CONTAINER_BUILD_OUTPUT": str(output),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    actual = json.loads(output.read_bytes())
    assert actual["input_digest"] == "sha256:" + "1" * 64
    assert actual["artifact"] == {
        "kind": "oci_image",
        "digest_kind": "oci_config",
        "reference": CONFIG,
        "digest": CONFIG,
        "platform": "linux/amd64",
    }
    assert INDEX in completed.stdout and MANIFEST in completed.stdout
