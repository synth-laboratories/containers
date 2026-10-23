"""``SynthTunnelControlPlane`` backed by the Synth API's lease endpoints.

Lease CRUD and worker-token mint live on the Python API (``/api/v1/synthtunnel/leases``);
the relay that serves ``public_url`` is a separate Railway service.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any
import urllib.error
import urllib.request

DEFAULT_API_BASE = "https://api-dev.usesynth.ai"
LEASES_PATH = "/api/v1/synthtunnel/leases"


class SynthTunnelControlPlaneError(RuntimeError):
    """A lease request was refused or the API could not be reached."""


class ApiControlPlane:
    def __init__(self, *, api_key: str, api_base: str = DEFAULT_API_BASE) -> None:
        if not api_key:
            raise SynthTunnelControlPlaneError("a Synth API key is required to mint a lease")
        self._api_key = api_key
        self._api_base = api_base.rstrip("/")

    def _request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self._api_base + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                **({"content-type": "application/json"} if data else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read()[:400].decode("utf-8", "replace")
            raise SynthTunnelControlPlaneError(
                f"SynthTunnel {method} {path} failed: HTTP {error.code} {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise SynthTunnelControlPlaneError(
                f"SynthTunnel {method} {path} could not reach {self._api_base}: {error.reason}"
            ) from error

    def create_synth_lease(
        self,
        *,
        client_instance_id: str,
        local_host: str,
        local_port: int,
        requested_ttl_seconds: int,
        metadata: dict[str, Any],
        capabilities: dict[str, Any],
    ) -> Mapping[str, Any]:
        return self._request(
            "POST",
            LEASES_PATH,
            {
                "client_instance_id": client_instance_id,
                "local_target": {"host": local_host, "port": local_port},
                "requested_ttl_seconds": requested_ttl_seconds,
                "metadata": metadata,
                "capabilities": capabilities,
            },
        )

    def close_synth_lease(self, lease_id: str) -> object:
        return self._request("DELETE", f"{LEASES_PATH}/{lease_id}")
