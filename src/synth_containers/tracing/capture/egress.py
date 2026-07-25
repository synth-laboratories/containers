"""Optional TLS interception planning and fail-closed egress assertions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class EgressAssertion:
    allowed_hosts: tuple[str, ...]
    observed_hosts: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def assert_egress(
    observed_urls: tuple[str, ...],
    *,
    allowed_hosts: tuple[str, ...],
) -> EgressAssertion:
    observed = tuple(
        sorted({urlparse(item).hostname or item for item in observed_urls if item})
    )
    allowed = set(allowed_hosts)
    violations = tuple(item for item in observed if item not in allowed)
    return EgressAssertion(tuple(sorted(allowed)), observed, violations)


def mitm_environment(
    *,
    proxy_url: str,
    ca_bundle_path: str,
    base: Mapping[str, str] | None = None,
    no_proxy_hosts: tuple[str, ...] = ("127.0.0.1", "localhost"),
) -> dict[str, str]:
    """Build the child-only environment for an explicitly selected scoped MITM.

    ``ca_bundle_path`` is the public-only trust bundle mounted into the child. The
    function never exposes the mitmproxy configuration directory or a private key.
    Provider hosts are deliberately absent from ``NO_PROXY`` so a parent shell's
    bypass list cannot silently defeat capture.
    """

    parsed = urlparse(proxy_url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise ValueError("scoped MITM proxy URL must be an absolute http URL")
    ca_path = str(ca_bundle_path)
    if not ca_path or not PurePosixPath(ca_path).is_absolute():
        raise ValueError("scoped MITM public CA bundle path must be absolute")
    bypass = ",".join(
        sorted(
            {
                str(item).strip().lower().rstrip(".")
                for item in no_proxy_hosts
                if str(item).strip()
            }
        )
    )
    values = {
        **{
            key: value
            for key, value in dict(base or {}).items()
            if key
            not in {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "http_proxy",
                "https_proxy",
                "NO_PROXY",
                "no_proxy",
                "SSL_CERT_FILE",
                "REQUESTS_CA_BUNDLE",
                "NODE_EXTRA_CA_CERTS",
            }
        },
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "NO_PROXY": bypass,
        "no_proxy": bypass,
        "SSL_CERT_FILE": ca_path,
        "REQUESTS_CA_BUNDLE": ca_path,
        "NODE_EXTRA_CA_CERTS": ca_path,
    }
    return values


__all__ = ["EgressAssertion", "assert_egress", "mitm_environment"]
