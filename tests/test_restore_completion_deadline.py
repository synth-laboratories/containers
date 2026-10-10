"""Completion recovery cannot renew origin execution or rewrite canonical intent."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from synth_ai.core.errors import HTTPError
from synth_ai.core.http.transport import raise_http_error
import httpx
from synth_containers.checkpoints.tunnel_authority import HttpSessionAuthority


@pytest.mark.parametrize("registered", [False, True])
def test_expired_origin_completion_has_stable_control_custody(registered):
    async def run():
        expired = datetime.now(timezone.utc) - timedelta(minutes=1)
        origin = SimpleNamespace(operation=uuid4(), connector=uuid4(), session=uuid4(), deadline_at=expired)
        result = SimpleNamespace(fence=SimpleNamespace(generation=4), checkpoint=uuid4(),
                                 checkpoint_sha256="a" * 64,
                                 receipts=[SimpleNamespace(restored_origin_instance=uuid4())])
        class Catalog:
            saved = {"deadline_at": expired.isoformat()} if registered else None
            requests = []
            async def session_operation(self, connector, operation):
                if self.saved is None:
                    raise_http_error(httpx.Response(404, json={"detail": "unknown_session_operation"}, request=httpx.Request("GET", "http://catalog/session-operations/operation")))
                return self.saved
            async def register_session_operation(self, connector, request):
                self.requests.append(request)
                if self.saved is None:
                    self.saved = {"deadline_at": request.deadline_at.isoformat()}
                assert datetime.fromisoformat(self.saved["deadline_at"]) == request.deadline_at
        catalog = Catalog()
        authority = HttpSessionAuthority(catalog=catalog, gateway_url="http://gateway", service_identity="fixture")
        async def gateway(*args, **kwargs):
            return {"state": "applied", "result": {"generation": 4, "phase": "open"}}
        authority._gateway = gateway
        try:
            before = datetime.now(timezone.utc)
            await authority.complete_restore(origin, result)
            await authority.complete_restore(origin, result)
            first, second = catalog.requests
            assert first == second
            assert origin.deadline_at == expired
            assert first.restart_operation == origin.operation
            if registered:
                assert first.deadline_at == expired
            else:
                assert before < first.deadline_at <= datetime.now(timezone.utc) + timedelta(seconds=60)
        finally:
            await authority.close()
    asyncio.run(run())


@pytest.mark.parametrize("status,detail", [(503, "unavailable"), (403, "forbidden"), (404, "unknown_connector")])
def test_unavailable_custody_cannot_create_completion(status, detail):
    async def run():
        class Catalog:
            async def session_operation(self, *args):
                raise_http_error(httpx.Response(status, json={"detail": detail}, request=httpx.Request("GET", "http://catalog/session-operations/operation")))
            async def register_session_operation(self, *args):
                raise AssertionError("created command without authoritative absence")
        authority = HttpSessionAuthority(catalog=Catalog(), gateway_url="http://gateway", service_identity="fixture")
        try:
            request = SimpleNamespace(operation=uuid4(), connector=uuid4())
            result = SimpleNamespace(fence=SimpleNamespace(generation=4), receipts=[object()])
            with pytest.raises(HTTPError):
                await authority.complete_restore(request, result)
        finally:
            await authority.close()
    asyncio.run(run())
