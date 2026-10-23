"""Real interface mismatches found by the pinned release typing profiles."""

from __future__ import annotations

import asyncio
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest

from synth_containers.tunnels.profiles import ConnectorProfile, TargetProfile
from synth_containers.tunnels.relay import _SynthTunnelAgent


def test_response_headers_preserve_repeated_fields():
    # A Message is intentionally not converted to dict: Set-Cookie and other
    # repeated fields must survive both success and HTTPError forwarding.
    headers = Message()
    headers["Set-Cookie"] = "first=one"
    headers["Set-Cookie"] = "second=two"
    headers["Content-Length"] = "5"
    headers["Connection"] = "close"
    frames = []
    agent = object.__new__(_SynthTunnelAgent)
    agent._protocol = "legacy"
    agent._identity_fields = lambda: {}
    agent._send_frame = lambda frame, **kwargs: frames.append(frame)
    agent._send_response("fixture", 200, headers, BytesIO(b"hello"), SimpleNamespace(connection_generation=1))
    assert frames[0]["headers"] == [["Set-Cookie", "first=one"], ["Set-Cookie", "second=two"]]
    assert frames[-1]["type"] == "RESP_END"


@pytest.mark.parametrize("fail", [False, True])
def test_connector_async_issuer_always_uses_sdk_aclose(monkeypatch, fail):
    import synth_ai
    import synth_containers.tunnels.service as module

    closed = []
    suppliers = []

    class SyncClient:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed.append("sync")

    class AsyncClient:
        def __init__(self, **kwargs):
            self.tunnels = self

        async def attachment_capability(self, connector, client):
            if fail:
                raise RuntimeError("fixture_lookup_failed")
            return SimpleNamespace(token=SimpleNamespace(get_secret_value=lambda: "fixture-token"))

        async def aclose(self):
            closed.append("async")

    class Provider:
        def __init__(self, *, async_token_supplier, **kwargs):
            suppliers.append(async_token_supplier)

    class Agent:
        def __init__(self, **kwargs):
            pass

        def stop(self):
            closed.append("agent")

    monkeypatch.setattr(synth_ai, "SynthClient", SyncClient)
    monkeypatch.setattr(synth_ai, "AsyncSynthClient", AsyncClient)
    monkeypatch.setattr(module, "HttpConnectorAttachmentProvider", Provider)
    monkeypatch.setattr(module, "SynthTunnelConnectorAgent", Agent)
    monkeypatch.setenv("RELEASE_FIXTURE_API_KEY", "synthetic")
    profile = ConnectorProfile(
        connector=uuid4(), client_instance=uuid4(),
        backend_url="http://127.0.0.1:1", gateway_url="http://127.0.0.1:1",
        backend_credential_env="RELEASE_FIXTURE_API_KEY",
        targets=(TargetProfile(key="fixture", url="http://127.0.0.1:2", service_identity="fixture"),),
    )
    service = module.ConnectorService(profile)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="fixture_lookup_failed"):
                asyncio.run(suppliers[0](profile.connector, profile.client_instance))
        else:
            assert asyncio.run(suppliers[0](profile.connector, profile.client_instance)) == "fixture-token"
        assert closed == ["async"]
    finally:
        service.close()
    assert closed == ["async", "agent", "sync"]
