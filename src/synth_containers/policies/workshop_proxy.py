"""Public-sentinel authentication for Workshop capability proxy routes.

The capability is carried by the unguessable, run-scoped URL. Containers must
not receive the underlying provider credential. A fixed public bearer keeps
OpenAI-compatible clients happy without turning it into a secret.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

_CAPABILITY_PATH = re.compile(r"^/cap/wcap_[A-Za-z0-9_-]+/v1/providers/[A-Za-z0-9._-]+(?:/|$)")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "host.docker.internal"}


def is_workshop_capability_proxy(base_url: str) -> bool:
    parsed = urlparse(str(base_url or ""))
    return (
        parsed.scheme == "http"
        and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS
        and bool(_CAPABILITY_PATH.match(parsed.path.rstrip("/") + "/"))
    )


def public_proxy_bearer(base_url: str) -> str | None:
    """Return the non-secret bearer only for a structurally valid local route."""

    return "workshop-proxy" if is_workshop_capability_proxy(base_url) else None
