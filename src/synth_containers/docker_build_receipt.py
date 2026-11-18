"""Standalone receipt writer copied into translated Harbor build inputs.

Docker's image ID can name an index, a manifest or a platform config. The
archive identifies which, and binds the config to the selected platform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

_MAX_DESCRIPTOR_BYTES = 64 * 1024
_MAX_ARCHIVE_ENTRIES = 4096
_MAX_ARCHIVE_BYTES = 64 * 1024**3
_MAX_METADATA_BYTES = 8 * 1024**2
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class MediaType(StrEnum):
    OCI_INDEX = "application/vnd.oci.image.index.v1+json"
    DOCKER_INDEX = "application/vnd.docker.distribution.manifest.list.v2+json"
    OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
    DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
    OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
    DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"


_INDEX_TYPES = (MediaType.OCI_INDEX, MediaType.DOCKER_INDEX)
_MANIFEST_TYPES = (MediaType.OCI_MANIFEST, MediaType.DOCKER_MANIFEST)
_CONFIG_TYPES = (MediaType.OCI_CONFIG, MediaType.DOCKER_CONFIG)


class ImageReceiptError(ValueError):
    """Docker export did not establish the package image's platform config."""


@dataclass(frozen=True, slots=True)
class Descriptor:
    kind: MediaType
    digest: str
    size: int

    @classmethod
    def parse(cls, value: dict[str, Any]) -> Descriptor:
        result = cls(MediaType(value["mediaType"]), value["digest"], value["size"])
        if (
            not _SHA256.fullmatch(result.digest)
            or type(result.size) is not int
            or not 0 < result.size <= _MAX_DESCRIPTOR_BYTES
        ):
            raise ImageReceiptError("invalid image metadata descriptor")
        return result


@dataclass(frozen=True, slots=True)
class PlatformConfig:
    image_id: str
    manifest_digest: str
    config_digest: str


def _metadata(stream: IO[bytes]) -> dict[str, bytes]:
    """Scan a single Docker export without extracting or retaining layer bytes."""
    documents: dict[str, bytes] = {}
    retained = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for number, entry in enumerate(archive, start=1):
            if number > _MAX_ARCHIVE_ENTRIES or entry.offset_data + entry.size > _MAX_ARCHIVE_BYTES:
                raise ImageReceiptError("Docker export exceeds image receipt scan limits")
            if not entry.isfile():
                continue
            if entry.name != "index.json" and not entry.name.startswith("blobs/sha256/"):
                continue
            if entry.size > _MAX_DESCRIPTOR_BYTES:
                continue
            if entry.name in documents:
                raise ImageReceiptError("duplicate image metadata in Docker export")
            retained += entry.size
            if retained > _MAX_METADATA_BYTES:
                raise ImageReceiptError("Docker export exceeds image receipt metadata limit")
            member = archive.extractfile(entry)
            if member is None:
                raise ImageReceiptError("Docker export metadata is unreadable")
            with member:
                documents[entry.name] = member.read()
    return documents


def _matches_platform(document: dict[str, Any], platform: str) -> bool:
    return f"{document.get('os')}/{document.get('architecture')}" == platform


def _verified_blob(documents: dict[str, bytes], descriptor: Descriptor) -> dict[str, Any]:
    raw = documents.get("blobs/sha256/" + descriptor.digest[7:])
    if raw is None or len(raw) != descriptor.size:
        raise ImageReceiptError("Docker export metadata size does not match descriptor")
    if "sha256:" + hashlib.sha256(raw).hexdigest() != descriptor.digest:
        raise ImageReceiptError("Docker export metadata digest does not match descriptor")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ImageReceiptError("Docker export metadata must be a JSON object")
    return value


def _layout_descriptors(index: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        index.get("schemaVersion") != 2
        or index.get("mediaType", MediaType.OCI_INDEX) not in _INDEX_TYPES
    ):
        raise ImageReceiptError("Docker export contains an unsupported image index")
    descriptors = index["manifests"]
    if not isinstance(descriptors, list) or any(
        not isinstance(value, dict) for value in descriptors
    ):
        raise ImageReceiptError("invalid image index descriptor list")
    return descriptors


def inspect_docker_export(stream: IO[bytes], *, image_id: str, platform: str) -> PlatformConfig:
    if not _SHA256.fullmatch(image_id):
        raise ImageReceiptError("invalid Docker build identity")
    try:
        documents = _metadata(stream)
        layout = _layout_descriptors(json.loads(documents["index.json"]))
        if len(layout) != 1:
            raise ImageReceiptError("Docker export must contain exactly one image root")
        root = Descriptor.parse(layout[0])
        manifest = root
        if root.kind in _INDEX_TYPES:
            index = _verified_blob(documents, root)
            if index.get("mediaType") != root.kind:
                raise ImageReceiptError("image index media type does not match descriptor")
            matching = [
                value
                for value in _layout_descriptors(index)
                if _matches_platform(value.get("platform", {}), platform)
            ]
            if len(matching) != 1:
                raise ImageReceiptError(
                    "image index must contain one requested platform descriptor"
                )
            manifest = Descriptor.parse(matching[0])
        if manifest.kind not in _MANIFEST_TYPES:
            raise ImageReceiptError("selected image descriptor must be an image manifest")
        image = _verified_blob(documents, manifest)
        if (
            image.get("schemaVersion") != 2
            or image.get("mediaType") != manifest.kind
            or not isinstance(image.get("layers"), list)
        ):
            raise ImageReceiptError("image manifest does not match its descriptor kind")
        config = Descriptor.parse(image["config"])
        if config.kind not in _CONFIG_TYPES:
            raise ImageReceiptError("manifest config descriptor has an unsupported kind")
        platform_config = _verified_blob(documents, config)
        if not _matches_platform(platform_config, platform):
            raise ImageReceiptError("image config platform does not match the package build")
        if image_id not in (root.digest, manifest.digest, config.digest):
            raise ImageReceiptError("Docker build identity is absent from the verified image")
        return PlatformConfig(image_id, manifest.digest, config.digest)
    except ImageReceiptError:
        raise
    except (AttributeError, KeyError, OSError, TypeError, ValueError, tarfile.TarError) as error:
        raise ImageReceiptError("Docker export image metadata is missing or invalid") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("iidfile", type=Path)
    parser.add_argument("platform")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image_id = args.iidfile.read_text().strip()
    if not _SHA256.fullmatch(image_id):
        parser.exit(1, "docker_image_receipt_invalid: invalid Docker build identity\n")
    try:
        with subprocess.Popen(
            ["docker", "image", "save", image_id], stdout=subprocess.PIPE
        ) as save:
            assert save.stdout is not None
            try:
                identity = inspect_docker_export(
                    save.stdout, image_id=image_id, platform=args.platform
                )
                # Drain tar padding before waiting for the producer.
                if len(save.stdout.read(tarfile.RECORDSIZE + 1)) > tarfile.RECORDSIZE:
                    raise ImageReceiptError("excess bytes after Docker export archive")
            except BaseException:
                save.kill()
                raise
            if save.wait() != 0:
                raise ImageReceiptError("Docker image export failed")
    except (ImageReceiptError, OSError) as error:
        parser.exit(1, f"docker_image_receipt_invalid: {error}\n")

    plan = json.loads(args.plan.read_bytes())
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "synth.container-build-output.v1",
                "action": "build",
                "input_digest": plan["input_digest"],
                "artifact": {
                    "kind": "oci_image",
                    "digest_kind": "oci_config",
                    "reference": identity.config_digest,
                    "digest": identity.config_digest,
                    "platform": args.platform,
                },
            }
        )
    )
    print(
        f"Docker image {identity.image_id}, manifest {identity.manifest_digest}, config {identity.config_digest}"
    )


if __name__ == "__main__":
    main()
