"""Internal MCP adapters for the same authenticated Rhodes SDK contracts.

The caller supplies current admitted role/resource authorization. Registration
never grants actor publishing privileges or bypasses the API's tenant boundary.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict
from typing import Any, Annotated

from pydantic import Field

from .package_models import PackageModel
from .rhodes_catalog import CatalogArchive, CatalogKind, CatalogPut, CatalogQuery, EnvironmentName, OperationId, Revision


class ReadArguments(PackageModel):
    kind: CatalogKind
    name: EnvironmentName
    revision: Annotated[int, Field(strict=True, ge=1, le=2**53 - 1)] | None = None


class ListArguments(PackageModel):
    kind: CatalogKind
    query: CatalogQuery = Field(default_factory=CatalogQuery)


class PutArguments(PackageModel):
    name: EnvironmentName
    request: CatalogPut


class ArchiveArguments(PackageModel):
    kind: CatalogKind
    name: EnvironmentName
    request: CatalogArchive


class HistoryArguments(PackageModel):
    kind: CatalogKind
    name: EnvironmentName
    after_revision: Revision = 0
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 100


class OperationArguments(PackageModel):
    operation_id: OperationId


class BuildArguments(PackageModel):
    build_operation_id: Annotated[str, Field(min_length=1, max_length=128)]


class BuildLogArguments(BuildArguments):
    after_attempt: Annotated[int, Field(strict=True, ge=0)] = 0
    limit: Annotated[int, Field(strict=True, ge=1, le=200)] = 100


_SCHEMAS = {
    "rhodes_catalog_get": ReadArguments, "rhodes_catalog_list": ListArguments,
    "rhodes_catalog_put": PutArguments, "rhodes_catalog_archive": ArchiveArguments,
    "rhodes_catalog_history": HistoryArguments, "rhodes_catalog_operation": OperationArguments,
    "rhodes_build_get": BuildArguments, "rhodes_build_logs": BuildLogArguments,
}


class RhodesCatalogTools:
    def __init__(self, client: Any, *, authorize: Callable[[str, Mapping[str, Any]], Awaitable[None]]):
        self.client, self.authorize = client, authorize

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return [{"name": name, "description": name.replace("_", " "), "inputSchema": model.model_json_schema()}
                for name, model in _SCHEMAS.items()]

    async def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name not in _SCHEMAS:
            raise ValueError("Unknown Rhodes catalogue tool")
        parsed = _SCHEMAS[name].model_validate(arguments)
        # Distinct admitted operations remain distinct from ordinary run/read.
        capability = "rhodes.catalog.write" if name in {"rhodes_catalog_put", "rhodes_catalog_archive"} else "rhodes.catalog.read"
        if name in {"rhodes_build_get", "rhodes_build_logs"}:
            capability = "rhodes.build.read"
        await self.authorize(capability, parsed.model_dump(mode="json"))
        if isinstance(parsed, BuildLogArguments):
            return asdict(await self.client.list_build_log_publications(parsed.build_operation_id, after_attempt=parsed.after_attempt, limit=parsed.limit))
        if isinstance(parsed, BuildArguments):
            return asdict(await self.client.get_build_operation(parsed.build_operation_id))
        catalog = self.client.catalog
        if isinstance(parsed, ReadArguments): result = await catalog.inspect(parsed.kind, parsed.name, revision=parsed.revision)
        elif isinstance(parsed, ListArguments): result = await catalog.list(parsed.kind, parsed.query)
        elif isinstance(parsed, PutArguments): result = await catalog.put(parsed.name, parsed.request)
        elif isinstance(parsed, ArchiveArguments): result = await catalog.archive(parsed.kind, parsed.name, parsed.request)
        elif isinstance(parsed, HistoryArguments): result = await catalog.history(parsed.kind, parsed.name, after_revision=parsed.after_revision, limit=parsed.limit)
        else: result = await catalog.operation(parsed.operation_id)
        return result.model_dump(mode="json")
