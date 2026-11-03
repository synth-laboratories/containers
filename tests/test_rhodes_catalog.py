"""Draft SDK/catalogue compatibility assertions authored for later execution."""
from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from synth_containers.package_models import content_digest
from synth_containers.rhodes_catalog import CatalogEntryView, CatalogPut, RhodesCatalogClient
from synth_containers.rhodes_catalog_tools import RhodesCatalogTools


def plan():
    return {"schema_version": "synth.rhodes-resource-plan.draft.v1", "kind": "resource_plan", "title": "Bounded evaluator", "providers": ["docker"],
        "source_revision": "plan-1", "hard_currency_budgets": [{"currency": "USD", "amount": "2"}],
        "requirements": [{"phase": "evaluate", "role": "evaluator", "resource_class": "cpu", "provider": "docker",
            "quantity": {"amount": "1", "unit": "cpu_second"}, "hard_quantity_budget": {"amount": "60", "unit": "cpu_second"},
            "expected_cost": {"status": "unknown", "currency": "USD", "unknown_reason": "No quote"}, "hard_currency_budget": {"currency": "USD", "amount": "2"},
            "enforcement": {"quantity": "declared", "currency": "unknown", "limitations": ["Declaration only"]}}]}


def entry(definition):
    return {"schema_version": "synth.rhodes-catalog-entry.draft.v1", "entry_id": "11111111-1111-1111-1111-111111111111",
        "org_id": "22222222-2222-2222-2222-222222222222", "kind": "resource_plan", "name": "bounded-plan", "revision": 1,
        "status": "active", "definition": definition, "definition_digest": content_digest(definition),
        "created_at": "2026-11-03T00:00:00+00:00", "updated_at": "2026-11-03T00:00:00+00:00"}


def test_catalogue_client_keeps_original_operation_and_exact_retained_decimal_definition():
    seen = []
    request = CatalogPut(schema_version="synth.rhodes-catalog-put.draft.v1", operation_id="create-1", expected_revision=0, definition=plan())

    class Transport:
        async def request(self, method, path, *, json=None, params=None):
            seen.append((method, path, json))
            return {"receipt": {"schema_version": "synth.rhodes-catalog-receipt.draft.v1", "operation_id": json["operation_id"],
                "request_digest": content_digest({"kind": "resource_plan", "name": "bounded-plan", "request": json}),
                "action": "put", "entry": entry(json["definition"])}, "reused": False}

    result = asyncio.run(RhodesCatalogClient(Transport()).put("bounded-plan", request))
    assert seen[0][0:2] == ("PUT", "/v1/rhodes/catalog/resource_plan_templates/bounded-plan")
    assert seen[0][2]["operation_id"] == "create-1"
    assert result.receipt.entry.definition_digest == content_digest(request.definition.model_dump(mode="json"))


def test_catalogue_decoder_refuses_changed_retained_definition():
    valid = CatalogPut(schema_version="synth.rhodes-catalog-put.draft.v1", operation_id="create-1", expected_revision=0, definition=plan())
    body = entry(valid.definition.model_dump(mode="json"))
    body["definition"]["title"] = "Changed after digest"
    with pytest.raises(ValidationError):
        CatalogEntryView.model_validate(body)


def test_internal_catalogue_write_requires_current_admitted_authority_before_client_effect():
    class Client:
        @property
        def catalog(self):
            raise AssertionError("A refused grant must not reach the client")

    async def refuse(capability, arguments):
        assert capability == "rhodes.catalog.write"
        raise PermissionError("revoked")

    arguments = {"name": "bounded-plan", "request": {"schema_version": "synth.rhodes-catalog-put.draft.v1",
        "operation_id": "create-1", "expected_revision": 0, "definition": plan()}}
    with pytest.raises(PermissionError):
        asyncio.run(RhodesCatalogTools(Client(), authorize=refuse).call("rhodes_catalog_put", arguments))
