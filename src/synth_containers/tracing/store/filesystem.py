"""Filesystem ``BlobStore``: immutable, content-addressed bodies under ``blobs/``."""

from __future__ import annotations

from pathlib import Path

from ..canonical import bytes_digest


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
        if not algorithm or not hexed:
            raise ValueError(f"malformed digest: {digest!r}")
        return self.root / algorithm / hexed[:2] / hexed

    def put(self, content: bytes) -> str:
        digest = bytes_digest(content)
        path = self.path_for(digest)
        if path.exists():
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(content)
        temp.replace(path)
        path.chmod(0o444)
        return digest

    def get(self, digest: str) -> bytes:
        path = self.path_for(digest)
        if not path.exists():
            raise FileNotFoundError(f"blob {digest} is not present under {self.root}")
        content = path.read_bytes()
        actual = bytes_digest(content)
        if actual != digest:
            raise ValueError(f"blob digest mismatch: expected {digest}, found {actual}")
        return content

    def has(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def uri(self, digest: str) -> str:
        return str(self.path_for(digest).relative_to(self.root.parent))


__all__ = ["FilesystemBlobStore"]
