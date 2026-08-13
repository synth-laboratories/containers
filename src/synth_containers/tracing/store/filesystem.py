"""Filesystem ``BlobStore``: immutable, content-addressed bodies under ``blobs/``."""

from __future__ import annotations

from pathlib import Path
import re

from ..canonical import bytes_digest
from .base import BlobMetadataV1, BlobPutResultV1


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class FilesystemBlobStore:
    """Stores bodies at ``<root>/<algorithm>/<first-two>/<digest>``.

    Writes are temp-file-plus-rename and the final file is read-only, so a reader can
    never observe a partially written body.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        algorithm, _, hexed = digest.partition(":")
        if algorithm != "sha256" or not _SHA256_HEX.fullmatch(hexed):
            raise ValueError(f"malformed digest: {digest!r}")
        return self.root / algorithm / hexed[:2] / hexed

    def put(self, content: bytes) -> str:
        return self.put_if_absent(content).digest

    def put_if_absent(
        self,
        content: bytes,
        *,
        media_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> BlobPutResultV1:
        digest = bytes_digest(content)
        path = self.path_for(digest)
        if path.exists():
            existing = self.get(digest)
            return BlobPutResultV1(
                digest=digest,
                created=False,
                metadata=BlobMetadataV1(
                    digest=digest,
                    byte_size=len(existing),
                    media_type=media_type,
                    uri=self.uri(digest),
                    metadata=dict(metadata or {}),
                ),
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(content)
        temp.replace(path)
        path.chmod(0o444)
        return BlobPutResultV1(
            digest=digest,
            created=True,
            metadata=BlobMetadataV1(
                digest=digest,
                byte_size=len(content),
                media_type=media_type,
                uri=self.uri(digest),
                metadata=dict(metadata or {}),
            ),
        )

    def get(
        self,
        digest: str,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        path = self.path_for(digest)
        if not path.exists():
            raise FileNotFoundError(f"blob {digest} is not present under {self.root}")
        content = path.read_bytes()
        actual = bytes_digest(content)
        if actual != digest:
            raise ValueError(f"blob digest mismatch: expected {digest}, found {actual}")
        if byte_range is None:
            return content
        start, end = byte_range
        if start < 0 or end < start:
            raise ValueError(f"invalid byte range: {byte_range!r}")
        return content[start : end + 1]

    def has(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def head(self, digest: str) -> BlobMetadataV1:
        path = self.path_for(digest)
        content = self.get(digest)
        return BlobMetadataV1(
            digest=digest,
            byte_size=len(content),
            uri=str(path.relative_to(self.root.parent)),
        )

    def uri(self, digest: str) -> str:
        return str(self.path_for(digest).relative_to(self.root.parent))


__all__ = ["FilesystemBlobStore"]
