"""S3 content-addressed blobs and CAS manifest pointers.

The client is injected (boto3-compatible) so the core package does not require an
AWS SDK. ETags are transport metadata only and are never treated as content digests.
"""

from __future__ import annotations

import re
from typing import Any

from ..canonical import bytes_digest
from .base import BlobMetadataV1, BlobPutResultV1


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_POINTER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RESERVED_DIGEST_METADATA = "synth-content-digest"


class S3BlobStore:
    def __init__(self, client: Any, *, bucket: str, prefix: str = "traces") -> None:
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, digest: str) -> str:
        algorithm, _, hexed = digest.partition(":")
        if algorithm != "sha256" or not _SHA256_HEX.fullmatch(hexed):
            raise ValueError(f"malformed digest: {digest!r}")
        return f"{self.prefix}/blobs/{algorithm}/{hexed[:2]}/{hexed}"

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
        key = self._key(digest)
        requested_metadata = dict(metadata or {})
        if any(
            str(name).lower() == _RESERVED_DIGEST_METADATA
            for name in requested_metadata
        ):
            raise ValueError(
                f"{_RESERVED_DIGEST_METADATA!r} is reserved store metadata"
            )
        stored_metadata = {
            **requested_metadata,
            _RESERVED_DIGEST_METADATA: digest,
        }
        created = True
        try:
            result = self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=media_type,
                Metadata=stored_metadata,
                IfNoneMatch="*",
            )
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = ((response.get("ResponseMetadata") or {}).get("HTTPStatusCode"))
            if status not in {409, 412}:
                raise
            created = False
            result = self.client.head_object(Bucket=self.bucket, Key=key)
        observed = self.get(digest)
        if observed != content:
            raise ValueError(f"S3 object collision for {digest}")
        actual_metadata = (
            stored_metadata
            if created
            else {
                str(name): str(value)
                for name, value in (result.get("Metadata") or {}).items()
            }
        )
        declared = actual_metadata.get(_RESERVED_DIGEST_METADATA)
        if declared and declared != digest:
            raise ValueError(f"S3 metadata digest mismatch for {digest}")
        return BlobPutResultV1(
            digest=digest,
            created=created,
            metadata=BlobMetadataV1(
                digest=digest,
                byte_size=(
                    len(content)
                    if created
                    else int(result.get("ContentLength", len(content)))
                ),
                media_type=(
                    media_type
                    if created
                    else str(result.get("ContentType") or media_type)
                ),
                etag=str(result.get("ETag") or "").strip() or None,
                uri=self.uri(digest),
                metadata=actual_metadata,
            ),
        )

    def get(
        self,
        digest: str,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> bytes:
        kwargs: dict[str, Any] = {"Bucket": self.bucket, "Key": self._key(digest)}
        if byte_range is not None:
            start, end = byte_range
            if start < 0 or end < start:
                raise ValueError(f"invalid byte range: {byte_range!r}")
            kwargs["Range"] = f"bytes={start}-{end}"
        response = self.client.get_object(**kwargs)
        body = response["Body"].read()
        if byte_range is None and bytes_digest(body) != digest:
            raise ValueError(f"S3 blob digest mismatch for {digest}")
        return body

    def has(self, digest: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(digest))
        except Exception as exc:
            status = (
                (getattr(exc, "response", {}).get("ResponseMetadata") or {}).get(
                    "HTTPStatusCode"
                )
            )
            if status == 404:
                return False
            raise
        return True

    def head(self, digest: str) -> BlobMetadataV1:
        result = self.client.head_object(Bucket=self.bucket, Key=self._key(digest))
        metadata = dict(result.get("Metadata") or {})
        declared = metadata.get(_RESERVED_DIGEST_METADATA)
        if declared and declared != digest:
            raise ValueError(f"S3 metadata digest mismatch for {digest}")
        return BlobMetadataV1(
            digest=digest,
            byte_size=int(result["ContentLength"]),
            media_type=str(result.get("ContentType") or "application/octet-stream"),
            etag=str(result.get("ETag") or "").strip() or None,
            uri=self.uri(digest),
            metadata=metadata,
        )

    def uri(self, digest: str) -> str:
        return f"s3://{self.bucket}/{self._key(digest)}"

    def compare_and_swap_pointer(
        self,
        *,
        name: str,
        content: bytes,
        expected_etag: str | None,
    ) -> str:
        if not _POINTER_NAME.fullmatch(name):
            raise ValueError(f"invalid pointer name: {name!r}")
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": f"{self.prefix}/pointers/{name}.json",
            "Body": content,
            "ContentType": "application/json",
        }
        if expected_etag is None:
            kwargs["IfNoneMatch"] = "*"
        else:
            kwargs["IfMatch"] = expected_etag
        result = self.client.put_object(**kwargs)
        return str(result.get("ETag") or "").strip()


def replicate_objects(source: Any, destination: Any, digests: tuple[str, ...]) -> dict[str, int]:
    copied = 0
    present = 0
    for digest in digests:
        if destination.has(digest):
            destination.get(digest)
            present += 1
            continue
        content = source.get(digest)
        destination.put_if_absent(content)
        destination.get(digest)
        copied += 1
    return {"copied": copied, "already_present": present}


__all__ = ["S3BlobStore", "replicate_objects"]
