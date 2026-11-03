"""Draft v1 task/resource registry contracts; rhodes-v024-service.md.

Field names are an implementation proposal for SYN-4839's accepted registry
scope. Registration records declarations and grants no execution or budget.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from decimal import Decimal
from typing import Annotated, Literal, Any, Protocol
from urllib.parse import quote

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from .package_models import Digest, PackageModel, portable_path, content_digest
from .rhodes_client import RhodesRequestError

Contract = PackageModel
Uuid = Annotated[str, Field(pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")]
EnvironmentName = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.-]*$")]
OperationId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]
Revision = Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]


class EvidenceReference(Contract):
    publication_id: Uuid
    manifest_digest: Digest
    logical_path: str
    digest: Digest
    size_bytes: Annotated[int, Field(strict=True, ge=0, le=64 * 1024 * 1024)]

    @model_validator(mode="after")
    def portable_coordinates(self):
        portable_path(self.logical_path)
        return self


def environment_name(value: str) -> str:
    return TypeAdapter(EnvironmentName).validate_python(value)


Token = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/@+-]+$")]
Text = Annotated[str, Field(min_length=1, max_length=4096)]
Amount = Annotated[Decimal, Field(ge=0, max_digits=24, decimal_places=9, allow_inf_nan=False)]
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class KnownClassification(Contract):
    status: Literal["known"]
    value: Token


class UnknownClassification(Contract):
    status: Literal["unknown"]
    reason: Text


Classification = Annotated[KnownClassification | UnknownClassification, Field(discriminator="status")]


class TaskMetadata(Contract):
    taxonomy: Token
    taxonomy_revision: Token
    source: EvidenceReference
    source_revision: Token
    mapping_revision: Token
    classifications: Annotated[dict[Token, Classification], Field(min_length=1, max_length=64)]
    description: Annotated[str, Field(max_length=16384)] = ""


class CurrencyBudget(Contract):
    currency: Currency
    amount: Amount


class Quantity(Contract):
    amount: Amount
    unit: Token


class ExpectedCost(Contract):
    status: Literal["known", "unknown"]
    currency: Currency
    amount: Amount | None = None
    source_revision: Token | None = None
    unknown_reason: Text | None = None

    @model_validator(mode="after")
    def explicit_knowledge(self):
        if self.status == "known":
            if self.amount is None or self.source_revision is None or self.unknown_reason is not None:
                raise ValueError("Known costs require amount/source revision and no unknown reason")
        elif self.amount is not None or self.unknown_reason is None:
            raise ValueError("Unknown cost requires an explicit reason and no invented amount")
        return self


class EnforcementCoverage(Contract):
    # Only a future authenticated enforcement-owner attestation may assert
    # enforcement. A registry write cannot manufacture that authority.
    quantity: Literal["declared", "not_enforced", "unknown"]
    currency: Literal["declared", "not_enforced", "unknown"]
    authority: Token | None = None
    limitations: Annotated[list[Text], Field(min_length=1, max_length=16)]


class ResourceRequirement(Contract):
    phase: Token
    role: Token
    resource_class: Token
    provider: Token
    quantity: Quantity
    hard_quantity_budget: Quantity
    expected_cost: ExpectedCost
    hard_currency_budget: CurrencyBudget
    enforcement: EnforcementCoverage

    @model_validator(mode="after")
    def consistent_limits(self):
        if self.quantity.unit != self.hard_quantity_budget.unit or self.quantity.amount > self.hard_quantity_budget.amount:
            raise ValueError("Quantity and its hard budget require the same unit and a sufficient bound")
        if self.expected_cost.currency != self.hard_currency_budget.currency:
            raise ValueError("Expected cost and its hard budget require the same currency")
        return self


class ResourcePlanTemplate(Contract):
    schema_version: Literal["synth.rhodes-resource-plan.draft.v1"]
    kind: Literal["resource_plan"]
    title: Annotated[str, Field(min_length=1, max_length=256)]
    providers: Annotated[list[Token], Field(min_length=1, max_length=16)]
    requirements: Annotated[list[ResourceRequirement], Field(min_length=1, max_length=128)]
    hard_currency_budgets: Annotated[list[CurrencyBudget], Field(min_length=1, max_length=16)]
    source_revision: Token

    @model_validator(mode="after")
    def provider_and_aggregate_coverage(self):
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("Provider declarations must be unique")
        budgets = {budget.currency: budget.amount for budget in self.hard_currency_budgets}
        if len(budgets) != len(self.hard_currency_budgets):
            raise ValueError("One aggregate hard budget per currency is required")
        totals: dict[str, Decimal] = {}
        identities = set()
        for requirement in self.requirements:
            identity = (requirement.phase, requirement.role, requirement.resource_class, requirement.provider)
            if identity in identities or requirement.provider not in self.providers:
                raise ValueError("Each phase/role/class/provider requirement must be unique and declared")
            identities.add(identity)
            currency = requirement.hard_currency_budget.currency
            totals[currency] = totals.get(currency, Decimal(0)) + requirement.hard_currency_budget.amount
        if any(currency not in budgets or amount > budgets[currency] for currency, amount in totals.items()):
            raise ValueError("Aggregate currency budgets must cover all hard requirement budgets")
        return self


class CatalogPin(Contract):
    name: EnvironmentName
    revision: Annotated[int, Field(strict=True, ge=1, le=2**53 - 1)]
    definition_digest: Digest


class QualifiedReleaseBinding(Contract):
    release_id: Uuid
    manifest_digest: Digest
    provider: Token
    validation_id: Uuid
    validation_receipt_digest: Digest


class TaskDefinition(Contract):
    schema_version: Literal["synth.rhodes-task-definition.draft.v1"]
    kind: Literal["task"]
    title: Annotated[str, Field(min_length=1, max_length=256)]
    task_id: Token
    task_version: Token
    metadata: TaskMetadata
    release_bindings: Annotated[list[QualifiedReleaseBinding], Field(min_length=1, max_length=16)]
    resource_plan: CatalogPin

    @model_validator(mode="after")
    def unique_bindings(self):
        identities = {(binding.release_id, binding.provider) for binding in self.release_bindings}
        if len(identities) != len(self.release_bindings):
            raise ValueError("Release/provider bindings must be unique")
        return self


CatalogDefinition = Annotated[TaskDefinition | ResourcePlanTemplate, Field(discriminator="kind")]
CatalogKind = Literal["task", "resource_plan"]


class CatalogPut(Contract):
    schema_version: Literal["synth.rhodes-catalog-put.draft.v1"]
    operation_id: OperationId
    expected_revision: Revision
    definition: CatalogDefinition

    @model_validator(mode="after")
    def bounded_storable_definition(self):
        values = [self.definition.model_dump(mode="json")]
        encoded = json.dumps(values[0], ensure_ascii=False).encode("utf-8")
        if len(encoded) > 256 * 1024:
            raise ValueError("Catalogue definition exceeds the retained bound")
        for _ in range(32768):
            if not values:
                return self
            value = values.pop()
            if isinstance(value, str) and "\x00" in value:
                raise ValueError("Catalogue strings cannot contain NUL")
            if isinstance(value, dict):
                values.extend(value.values())
            elif isinstance(value, list):
                values.extend(value)
        raise ValueError("Catalogue definition exceeds the structural bound")


class CatalogArchive(Contract):
    schema_version: Literal["synth.rhodes-catalog-archive.draft.v1"]
    operation_id: OperationId
    expected_revision: Revision


class CatalogEntryView(Contract):
    schema_version: Literal["synth.rhodes-catalog-entry.draft.v1"]
    entry_id: Uuid
    org_id: Uuid
    kind: CatalogKind
    name: EnvironmentName
    revision: Annotated[int, Field(strict=True, ge=1, le=2**53 - 1)]
    status: Literal["active", "archived"]
    definition: CatalogDefinition
    definition_digest: Digest
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def exact_definition_identity(self):
        if self.definition.kind != self.kind or content_digest(self.definition.model_dump(mode="json")) != self.definition_digest:
            raise ValueError("Catalogue definition differs from its retained kind/digest")
        return self


class CatalogReceipt(Contract):
    schema_version: Literal["synth.rhodes-catalog-receipt.draft.v1"]
    operation_id: OperationId
    request_digest: Digest
    action: Literal["put", "archive"]
    entry: CatalogEntryView


class CatalogMutationResponse(Contract):
    receipt: CatalogReceipt
    reused: bool


class CatalogPage(Contract):
    items: Annotated[list[CatalogEntryView], Field(max_length=100)]
    next_name: EnvironmentName | None


class CatalogHistoryPage(Contract):
    items: Annotated[list[CatalogReceipt], Field(max_length=100)]
    next_revision: Annotated[int, Field(strict=True, ge=1, le=2**53 - 1)] | None


class CatalogQuery(Contract):
    after_name: EnvironmentName | None = None
    include_archived: bool = False
    provider: Token | None = None
    resource_class: Token | None = None
    taxonomy: Token | None = None
    taxonomy_revision: Token | None = None
    source_revision: Token | None = None
    mapping_revision: Token | None = None
    classification_key: Token | None = None
    classification_value: Token | None = None
    classification_unknown: bool | None = None
    limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 100


def metadata_from_source(source_document: Mapping[str, object], *, taxonomy: str,
                         taxonomy_revision: str, source: EvidenceReference,
                         source_revision: str, mapping_revision: str,
                         field_mapping: Mapping[str, str]) -> TaskMetadata:
    """Map selected REB/Task Factory fields with explicit unknown provenance.

    No taxonomy value is guessed. The exact source remains in existing custody,
    and the caller pins this mapping revision independently of the source.
    """
    if not 1 <= len(field_mapping) <= 64:
        raise ValueError("Metadata mapping requires one to 64 explicit dimensions")
    classifications: dict[str, Classification] = {}
    for dimension, source_field in field_mapping.items():
        dimension = TypeAdapter(Token).validate_python(dimension)
        source_field = TypeAdapter(Token).validate_python(source_field)
        value = source_document.get(source_field)
        if value is None or value == "":
            classifications[dimension] = UnknownClassification(status="unknown", reason=f"source_field_absent:{source_field}")
            continue
        try:
            token = TypeAdapter(Token).validate_python(value)
        except ValidationError:
            classifications[dimension] = UnknownClassification(status="unknown", reason=f"source_value_not_taxonomy_token:{source_field}")
        else:
            classifications[dimension] = KnownClassification(status="known", value=token)
    return TaskMetadata(taxonomy=taxonomy, taxonomy_revision=taxonomy_revision, source=source,
                        source_revision=source_revision, mapping_revision=mapping_revision,
                        classifications=classifications)


class CatalogTransport(Protocol):
    async def request(self, method: str, path: str, *, json: Mapping[str, Any] | None = None,
                      params: Mapping[str, Any] | None = None) -> Mapping[str, Any]: ...


class RhodesCatalogClient:
    """Typed task/resource-plan CRUD, immutable pins, query and receipt readback."""

    def __init__(self, transport: CatalogTransport):
        self._transport = transport

    def _path(self, kind: CatalogKind, name: str | None = None) -> str:
        collection = {"task": "task_definitions", "resource_plan": "resource_plan_templates"}[kind]
        base = "/v1/rhodes/catalog/" + collection
        return base if name is None else base + "/" + quote(environment_name(name), safe="")

    async def put(self, name: str, request: CatalogPut) -> CatalogMutationResponse:
        response = await self._transport.request("PUT", self._path(request.definition.kind, name),
                                                  json=request.model_dump(mode="json"))
        return CatalogMutationResponse.model_validate(response)

    async def archive(self, kind: CatalogKind, name: str, request: CatalogArchive) -> CatalogMutationResponse:
        response = await self._transport.request("POST", self._path(kind, name) + "/archive",
                                                  json=request.model_dump(mode="json"))
        return CatalogMutationResponse.model_validate(response)

    async def inspect(self, kind: CatalogKind, name: str, *, revision: int | None = None) -> CatalogEntryView:
        params = {} if revision is None else {"revision": revision}
        response = await self._transport.request("GET", self._path(kind, name), params=params)
        return CatalogEntryView.model_validate(response)

    async def list(self, kind: CatalogKind, query: CatalogQuery | None = None) -> CatalogPage:
        response = await self._transport.request("GET", self._path(kind),
                                                  params=(query or CatalogQuery()).model_dump(mode="json", exclude_none=True))
        return CatalogPage.model_validate(response)

    async def history(self, kind: CatalogKind, name: str, *, after_revision: int = 0, limit: int = 100) -> CatalogHistoryPage:
        response = await self._transport.request("GET", self._path(kind, name) + "/history",
                                                  params={"after_revision": after_revision, "limit": limit})
        return CatalogHistoryPage.model_validate(response)

    async def operation(self, operation_id: str) -> CatalogReceipt:
        operation_id = TypeAdapter(OperationId).validate_python(operation_id)
        response = await self._transport.request("GET", "/v1/rhodes/catalog/operations/" + quote(operation_id, safe=""))
        return CatalogReceipt.model_validate(response)


class _RhodesTransport:
    def __init__(self, client: Any):
        self.client = client

    async def request(self, method: str, path: str, *, json=None, params=None):
        try:
            return await self.client._request(method, path.removeprefix("/v1"), payload=json, params=params,
                                              replay_safe=method == "GET" or bool(json and json.get("operation_id")))
        except RhodesRequestError as error:
            if not error.code.startswith("rhodes.catalog."):
                raise
            raise CatalogRefused(error.status_code, error.code, error.retry_after, error.identifiers) from error


def catalog_for(client: Any) -> RhodesCatalogClient:
    return RhodesCatalogClient(_RhodesTransport(client))


class CatalogRefused(RhodesRequestError):
    """A typed registry refusal retaining the exact stable service code."""
