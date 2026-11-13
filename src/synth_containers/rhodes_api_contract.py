"""Rhodes HTTP compatibility, independent of package/build/readiness versions."""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from typing import Any

from .pools import PoolClientError

DISCOVERY_PATH = "/.well-known/synth-api"
REQUEST_HEADER = "Synth-API-Major"
VERSION_HEADER = "Synth-API-Version"
SUPPORTED_MAJOR = 2
MAX_DISCOVERY_BYTES = 65536
_NUMBER = r"(?:0|[1-9][0-9]*)"
_SEMVER = re.compile(
    rf"({_NUMBER})\.({_NUMBER})\.({_NUMBER})"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
)


class APIContractError(PoolClientError):
    """A bounded, credential-free compatibility refusal; no fallback is safe."""

    def __init__(self, message: str, *, code: str = "api_contract_invalid"):
        self.code = code
        super().__init__(message)


def compatible_version(value: Any) -> bool:
    """Accept stable >=2.0.0,<3.0.0, including SemVer build metadata."""
    if not isinstance(value, str) or len(value) > 128:
        return False
    match = _SEMVER.fullmatch(value)
    return bool(match and match[1] == str(SUPPORTED_MAJOR) and match[4] is None)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate discovery field")
        result[key] = value
    return result


def validate_discovery(
    content: bytes,
    response_version: str | None,
    required_capabilities: Collection[str],
) -> dict[str, Any]:
    try:
        body = json.loads(content, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise APIContractError("Rhodes API discovery is not valid unambiguous JSON") from error
    if not isinstance(body, dict):
        raise APIContractError("Rhodes API discovery must be an object")
    api = body.get("api")
    if (
        body.get("schema") != "synth.api-discovery.v1"
        or body.get("service") != "rhodes-api"
        or body.get("authority") != "synth-backend.core"
        or not isinstance(api, dict)
        or api.get("name") != "synth.rhodes"
    ):
        raise APIContractError("Endpoint does not identify the Rhodes API contract")
    version = api.get("version")
    if not compatible_version(version):
        raise APIContractError(
            "Rhodes API requires stable >=2.0.0,<3.0.0", code="api_version_unsupported"
        )
    supported = api.get("supported_versions")
    if (
        response_version != version
        or not isinstance(supported, list)
        or version not in supported
        or any(not isinstance(item, str) for item in supported)
    ):
        raise APIContractError("Rhodes API discovery has contradictory version metadata")
    capabilities = body.get("capabilities")
    if not isinstance(capabilities, dict) or any(
        not isinstance(key, str) or type(value) is not bool for key, value in capabilities.items()
    ):
        raise APIContractError("Rhodes API capabilities must be explicit booleans")
    if any(capabilities.get(name) is not True for name in required_capabilities):
        raise APIContractError(
            "Rhodes API does not support the required operations", code="api_capability_unsupported"
        )
    return body
