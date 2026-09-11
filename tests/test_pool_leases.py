import asyncio
import json

import httpx
import pytest

from synth_containers.pools import PoolClient, PoolClientError

PROJECT = '00000000-0000-0000-0000-000000000001'


def assign(client):
    return client.assign_lease(project_id=PROJECT, image_kind='synth_sdk',
                               substrate='docker', idempotency_key='stable', ttl_seconds=300)


def test_assignment_requires_server_resolved_placement():
    sent = []
    def handle(request):
        sent.append(json.loads(request.content))
        return httpx.Response(201, json={'lease': {'lease_id':'lease-1', 'project_id':PROJECT,
                              'image_kind':'synth_sdk', 'execution_substrate':'docker', 'status':'active'}, 'reused':False})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="https://fixture") as transport:
            result = await assign(PoolClient(api_key='fixture', client=transport))
            assert result['lease']['execution_substrate']=='docker'
    asyncio.run(run())
    assert sent == [{'project_id':PROJECT,'image_kind':'synth_sdk','substrate':'docker',
                     'idempotency_key':'stable','ttl_seconds':300}]


@pytest.mark.parametrize('change', [
    {'execution_substrate':None, 'metadata':{'execution_substrate':'docker'}},
    {'execution_substrate':'daytona'}, {'project_id':'other'}, {'status':'expired'},
])
def test_legacy_or_mismatched_acknowledgement_is_not_silent_success(change):
    sent=[]
    def handle(request):
        sent.append(request)
        return httpx.Response(201,json={'lease':{'lease_id':'lease-1','project_id':PROJECT,
                              'image_kind':'synth_sdk','execution_substrate':'docker','status':'active',**change}})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="https://fixture") as transport:
            with pytest.raises(PoolClientError, match="lease_id='lease-1'"):
                await assign(PoolClient(api_key='fixture', client=transport))
    asyncio.run(run())
    assert len(sent)==1


def test_uncertain_assignment_never_retries_automatically():
    sent=[]
    def handle(request):
        sent.append(request)
        raise httpx.ReadTimeout('uncertain',request=request)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="https://fixture") as transport:
            with pytest.raises(PoolClientError):
                await assign(PoolClient(api_key='fixture',client=transport,max_retries=3))
    asyncio.run(run())
    assert len(sent)==1


@pytest.mark.parametrize('ttl', [0,30,True,3601,1.5])
def test_ttl_cannot_be_silently_widened_by_server_clamping(ttl):
    with pytest.raises(PoolClientError):
        PoolClient._validate_lease_ttl(ttl)
