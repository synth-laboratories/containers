"""Version-pinned Daytona API seam that returns identity before build polling."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

from daytona import Image
from daytona._async.snapshot import AsyncSnapshotService
from daytona_api_client_async import CreateBuildInfo, CreateSnapshot
from daytona_api_client_async.exceptions import NotFoundException

from .harbor_image_context import HarborImageContext


class DaytonaSnapshotProvider:
    """Bypass only the SDK's unbounded polling, retaining its context uploader.

    These resource settings describe sandboxes created from the snapshot. They
    are not a quota or charge ceiling on Daytona's internal image builder.
    """

    # API resource fields apply to the resulting sandbox, not the builder.
    # Hard builder spend/resource/remote-context-expiry requests fail admission.
    build_capabilities = frozenset()
    build_architectures = ()

    def __init__(
        self,
        client: Any,
        *,
        cpu: int = 1,
        memory_gib: int = 1,
        disk_gib: int = 2,
        region_id: str | None = None,
        qualified_architecture: str | None = None,
    ):
        if version("daytona") != "0.210.0":
            raise ValueError("Snapshot preparation requires qualified Daytona 0.210.0")
        for field, value, ceiling in (
            ("cpu", cpu, 64),
            ("memory_gib", memory_gib, 512),
            ("disk_gib", disk_gib, 1024),
        ):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{field} exceeds snapshot resource bounds")
        if qualified_architecture is not None:
            if qualified_architecture not in {"amd64", "arm64"}:
                raise ValueError("Invalid qualified provider architecture")
            # Operator configuration must come from provider/region qualification;
            # this is not an OCI inspection or cross-architecture build switch.
            self.build_architectures = (qualified_architecture,)
        service = client.snapshot
        self.api = service._AsyncSnapshotService__snapshots_api
        self.storage_api = service._AsyncSnapshotService__object_storage_api
        self.region_id = region_id or service._AsyncSnapshotService__default_region_id
        self.resources = {"cpu": cpu, "gpu": 0, "memory": memory_gib, "disk": disk_gib}

    async def prepare_context(self, context: HarborImageContext):
        context.verify()
        image = Image.from_dockerfile(context.root / "Dockerfile")
        if len(image._context_list) > 32:
            raise ValueError("Daytona build context reference count exceeds bound")
        root = context.root.resolve()
        for item in image._context_list:
            path = Path(item.source_path).resolve()
            archive = PurePosixPath(item.archive_path)
            if not path.is_relative_to(root) or archive.is_absolute() or ".." in archive.parts:
                raise ValueError("Daytona build context reference escapes the frozen context")
        context.verify()
        hashes = await AsyncSnapshotService.process_image_context(self.storage_api, image)
        context.verify()
        if (
            not isinstance(hashes, list)
            or len(hashes) > 32
            or any(not isinstance(value, str) or not 1 <= len(value) <= 512 for value in hashes)
        ):
            raise ValueError("Daytona context upload returned invalid identities")
        return CreateBuildInfo(dockerfile_content=image.dockerfile(), context_hashes=hashes)

    async def create(self, name: str, prepared: CreateBuildInfo):
        return await self.api.create_snapshot(
            CreateSnapshot(
                name=name, build_info=prepared, region_id=self.region_id, **self.resources
            ),
            _request_timeout=15,
        )

    async def get(self, identifier: str):
        return await self.api.get_snapshot(identifier, _request_timeout=10)

    async def delete(self, identifier: str):
        return await self.api.remove_snapshot(identifier, _request_timeout=15)

    @staticmethod
    def is_not_found(error: Exception) -> bool:
        return isinstance(error, NotFoundException)
