"""Foreground connector runtime; the host's existing supervisor owns restarts."""

from __future__ import annotations
import os
import threading
from .profiles import ConnectorProfile
from .connector_control import HttpConnectorAttachmentProvider
from .relay import SynthTunnelConnectorAgent


class ConnectorService:
    def __init__(self, profile: ConnectorProfile) -> None:
        credential = os.environ.get(profile.backend_credential_env)
        if not credential:
            raise ValueError("connector_backend_credential_unavailable")
        from synth_ai import SynthClient, AsyncSynthClient

        self.profile = profile
        targets = profile.registry()
        self._client = SynthClient(api_key=credential, base_url=profile.backend_url)

        async def attachment_token(connector, client):
            issuer = AsyncSynthClient(api_key=credential, base_url=profile.backend_url)
            try:
                return (
                    await issuer.tunnels.attachment_capability(connector, client)
                ).token.get_secret_value()
            finally:
                await issuer.aclose()

        provider = HttpConnectorAttachmentProvider(
            gateway_url=profile.gateway_url,
            async_token_supplier=attachment_token,
        )
        try:
            self._agent = SynthTunnelConnectorAgent(
                connector_id=profile.connector,
                client_instance_id=profile.client_instance,
                targets=targets,
                provider=provider,
                max_in_flight_requests=profile.max_inflight,
                max_request_body_bytes=profile.max_request_bytes,
            )
        except BaseException:
            self._client.close()
            raise
        self._closed = False

    def start(self, *, timeout_seconds: float = 30) -> None:
        self._agent.start(timeout_seconds=timeout_seconds)

    def status(self) -> dict:
        catalog = self._client.tunnels.status(self.profile.connector)
        return {
            "connector": str(self.profile.connector),
            "attached": self._agent._ready.is_set(),
            "catalog": catalog.model_dump(mode="json"),
        }

    def wait(self, stop: threading.Event) -> None:
        while not stop.wait(0.25):
            thread = self._agent._thread
            if thread is None or not thread.is_alive():
                raise RuntimeError("connector_agent_stopped")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._agent.stop()
        finally:
            self._client.close()
            self._closed = True
