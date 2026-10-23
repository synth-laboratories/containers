"""Verify the frozen native engine context before runtime image assembly."""

import hashlib
import json
from pathlib import Path
import platform
import re
import sys


def verify(root):
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise ValueError("prepared Craftax engine requires an ARM64 runtime image")
    root = Path(root)
    manifest = json.loads((root / "build-inputs.json").read_bytes())
    if (
        manifest.get("schema_version") != "synth.craftax.compiler-artifact.v1"
        or manifest.get("platform") != "linux/arm64"
        or manifest.get("mbx_sha256") != "3b7408c71fae119b09c506ab47daff6cba6476c5e7140111959824bd72e63331"
        or manifest.get("rust_image") != "rust:1.97-slim-bookworm@sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3"
        or not re.fullmatch("[0-9a-f]{40}", manifest.get("source_revision", ""))
    ):
        raise ValueError("qualified Craftax compiler manifest required")
    expected = {"craftax_gold": manifest["binary_sha256"]}
    for path, digest in manifest["source_input_sha256"].items():
        if path.startswith(("defaults/", "shared/assets/craftax/")):
            expected[path] = digest
    names = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if names != set(expected) | {"build-inputs.json"}:
        raise ValueError("engine artifact file closure differs")
    for relative, digest in expected.items():
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink():
            raise ValueError("unsafe engine artifact path")
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("engine artifact bytes differ: " + relative)
    with (root / "craftax_gold").open("rb") as stream:
        header = stream.read(20)
    if (
        len(header) != 20
        or header[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(header[18:20], "little") != 183
    ):
        raise ValueError("Linux ARM64 engine required")
    return manifest


if __name__ == "__main__":
    verify(sys.argv[1])
