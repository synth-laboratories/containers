"""Admission for the local ``synth_mlx_rl`` inference proxy. Fails closed.

One validator, three call sites (``runtimes/banking77.py``, ``react.py``,
``runtimes/healthbench.py``) so a loopback MLX server is admitted on exactly the
same terms everywhere. The shape is copied from ``_validate_responses_endpoint``
in ``runtimes/banking77.py``, which is the in-repo precedent:

- reject userinfo, query and fragment outright — a base URL carrying a password
  is a credential in a config field that is echoed back on ``/policy-configs``;
- permit ``http://`` unconditionally for ``127.0.0.1`` / ``localhost`` / ``::1``,
  matching the precedent exactly;
- require every other origin to be named in ``SYNTH_MLX_RL_ALLOWED_ENDPOINTS``.
  That includes ``host.docker.internal``, which is deliberately NOT blanket
  loopback: it resolves to the host from inside a container, so admitting it on
  any port would hand a policy config a container-to-host probe across every
  service on the machine. The Docker case is enabled by naming the one origin,
  e.g. ``SYNTH_MLX_RL_ALLOWED_ENDPOINTS=http://host.docker.internal:8787``,
  which is an explicit act rather than a default;
- refuse with terse, secret-free snake_case codes so ``_error_code`` in
  ``runtimes/banking77.py`` forwards them instead of dropping them as prose.

Both ``InferenceApiFamily`` members are first class: the endpoint path must end
in ``/chat/completions`` or ``/responses`` according to the family the caller
declares, so a config cannot name one family and be sampled on the other.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit


PROVIDER_ID = "synth_mlx_rl"
ALLOWLIST_ENV = "SYNTH_MLX_RL_ALLOWED_ENDPOINTS"

CHAT_COMPLETIONS = "chat_completions"
RESPONSES = "responses"
API_FAMILIES: tuple[str, ...] = (CHAT_COMPLETIONS, RESPONSES)

_ENDPOINT_SUFFIX = {
    CHAT_COMPLETIONS: "/chat/completions",
    RESPONSES: "/responses",
}

# Loopback only. `host.docker.internal` is the name a container uses to reach the
# host, which is exactly why it is not in this set: blanket-admitting it would
# admit every port on the host, not just the proxy's. It goes through the
# allowlist instead, where the port is named.
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DOCKER_HOST_ALIAS = "host.docker.internal"

ENDPOINT_REFUSED = f"{PROVIDER_ID}_endpoint_refused"
API_FAMILY_UNSUPPORTED = f"{PROVIDER_ID}_api_family_unsupported"
BASE_URL_MISSING = f"{PROVIDER_ID}_base_url_missing"


def is_local_provider(value: object) -> bool:
    return isinstance(value, str) and value.strip().lower() == PROVIDER_ID


def normalize_api_family(value: object) -> str:
    """``chat_completions`` when unset; anything unrecognised is refused.

    Defaulting is safe here in a way it is not for ``provider`` — the two
    families have different request bodies, and an unnamed family that silently
    became ``responses`` would send a chat payload to a route that cannot read
    it. Unset means "the family this call site already speaks".
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return CHAT_COMPLETIONS
    if not isinstance(value, str):
        raise RuntimeError(API_FAMILY_UNSUPPORTED)
    family = value.strip().lower()
    if family not in API_FAMILIES:
        raise RuntimeError(API_FAMILY_UNSUPPORTED)
    return family


def endpoint_suffix(api_family: str) -> str:
    return _ENDPOINT_SUFFIX[normalize_api_family(api_family)]


def allowed_origins() -> frozenset[str]:
    return frozenset(
        item.strip().rstrip("/")
        for item in os.environ.get(ALLOWLIST_ENV, "").split(",")
        if item.strip()
    )


def validate_local_endpoint(endpoint: str, *, api_family: str = CHAT_COMPLETIONS) -> None:
    """Refuse anything that is not a local proxy or an explicitly allowed origin."""
    family = normalize_api_family(api_family)
    suffix = _ENDPOINT_SUFFIX[family]
    try:
        parsed = urlsplit(endpoint or "")
        port = parsed.port
    except ValueError as exc:
        raise RuntimeError(ENDPOINT_REFUSED) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(suffix)
    ):
        raise RuntimeError(ENDPOINT_REFUSED)
    host = parsed.hostname.lower()
    if host in _LOCAL_HOSTS and parsed.scheme == "http":
        return
    normalized = f"{parsed.scheme}://{host}"
    if port is not None:
        normalized += f":{port}"
    if normalized not in allowed_origins():
        raise RuntimeError(ENDPOINT_REFUSED)


def local_endpoint(base_url: object, *, api_family: str = CHAT_COMPLETIONS) -> str:
    """Validated endpoint for ``base_url`` on ``api_family``.

    The suffix is appended here rather than at each call site so a base URL that
    already carries the route is not doubled into ``/v1/responses/responses``.
    """
    family = normalize_api_family(api_family)
    suffix = _ENDPOINT_SUFFIX[family]
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError(BASE_URL_MISSING)
    endpoint = base if base.endswith(suffix) else f"{base}{suffix}"
    validate_local_endpoint(endpoint, api_family=family)
    return endpoint
