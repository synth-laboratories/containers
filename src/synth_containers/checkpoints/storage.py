"""Immutable content-addressed checkpoint blobs; vendor clients end here.

See docs/specs/tunnel-v2-connector.md. Blob bytes never enter the tunnel actor.
"""

from __future__ import annotations
from hashlib import sha256
import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4
from .contracts import ArtifactReference, CheckpointFailure

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class ArtifactReader(Protocol):
    def get(self, reference: ArtifactReference, /) -> bytes: ...


class ArtifactStore(ArtifactReader, Protocol):
    def put(self, payload: bytes) -> ArtifactReference: ...


def reference(namespace: str, payload: bytes) -> ArtifactReference:
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise CheckpointFailure("checkpoint_artifact_capacity")
    digest = sha256(payload).hexdigest()
    return ArtifactReference(
        namespace=namespace, object_key="sha256/" + digest, sha256=digest, size_bytes=len(payload)
    )


def verified(reference: ArtifactReference, payload: bytes) -> bytes:
    if len(payload) != reference.size_bytes or sha256(payload).hexdigest() != reference.sha256:
        raise CheckpointFailure("checkpoint_artifact_digest_mismatch")
    return payload


class FileArtifactStore:
    """Local immutable blobs with atomic publish and digest-checked reads."""

    def __init__(self, root: Path, *, namespace: str):
        self.root = root.resolve()
        self.namespace = namespace
        (self.root / "sha256").mkdir(parents=True, exist_ok=True)

    def _path(self, item: ArtifactReference) -> Path:
        if item.namespace != self.namespace or item.object_key != "sha256/" + item.sha256:
            raise CheckpointFailure("checkpoint_artifact_scope")
        return self.root / item.object_key

    def put(self, payload: bytes) -> ArtifactReference:
        item = reference(self.namespace, payload)
        destination = self._path(item)
        temporary = destination.parent / (".publish-" + uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self.get(item)
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        return item

    def get(self, item: ArtifactReference) -> bytes:
        path = self._path(item)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if os.fstat(stream.fileno()).st_size != item.size_bytes:
                raise CheckpointFailure("checkpoint_artifact_size_mismatch")
            return verified(item, stream.read(MAX_ARTIFACT_BYTES + 1))


class S3ArtifactStore:
    """Injected boto3-compatible client; MinIO/Wasabi configuration stays outside core."""

    def __init__(self, client, *, bucket: str, namespace: str):
        self._client = client
        self._bucket = bucket
        self.namespace = namespace

    def _key(self, item: ArtifactReference) -> str:
        if item.namespace != self.namespace or item.object_key != "sha256/" + item.sha256:
            raise CheckpointFailure("checkpoint_artifact_scope")
        return self.namespace + "/" + item.object_key

    def put(self, payload: bytes) -> ArtifactReference:
        item = reference(self.namespace, payload)
        key = self._key(item)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=payload,
                IfNoneMatch="*",
                Metadata={"sha256": item.sha256},
            )
        except Exception as error:
            status = (
                getattr(error, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode")
            )
            if status != 412:
                raise CheckpointFailure("checkpoint_blob_write_in_doubt") from None
            self.get(item)
        return item

    def get(self, item: ArtifactReference) -> bytes:
        key = self._key(item)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception:
            raise CheckpointFailure("checkpoint_blob_read_unavailable") from None
        body = response["Body"]
        try:
            if response["ContentLength"] != item.size_bytes:
                raise CheckpointFailure("checkpoint_artifact_size_mismatch")
            return verified(item, body.read(MAX_ARTIFACT_BYTES + 1))
        finally:
            body.close()
