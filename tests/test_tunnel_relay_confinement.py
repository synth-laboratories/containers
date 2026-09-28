"""Relay forwarding confinement keeps legitimate relative forwarding intact.

The SSRF fix confines worker-supplied forward paths to the leased target. These
cases pin that ordinary forwarding (paths, queries, path-prefixed and IPv6
leases, trailing slashes) still resolves to the same URL, and that escapes fail
closed to the lease root.
"""

from __future__ import annotations

import urllib.request

import pytest

from synth_containers.tunnels import relay


@pytest.mark.parametrize(
    ("local_url", "path", "query", "expected"),
    [
        ("http://127.0.0.1:8123", "/v1/chat", "", "http://127.0.0.1:8123/v1/chat"),
        ("http://127.0.0.1:8123", "/v1/chat", "a=1&b=2", "http://127.0.0.1:8123/v1/chat?a=1&b=2"),
        ("http://127.0.0.1:8123", "/", "", "http://127.0.0.1:8123/"),
        ("http://127.0.0.1:8123", "/rollouts/", "", "http://127.0.0.1:8123/rollouts/"),
        ("https://127.0.0.1:8443", "/health", "", "https://127.0.0.1:8443/health"),
        ("http://127.0.0.1:8123/api", "/v1/x", "q=1", "http://127.0.0.1:8123/api/v1/x?q=1"),
        ("http://[::1]:8123", "/v1/x", "", "http://[::1]:8123/v1/x"),
    ],
    ids=["path", "query", "root", "trailing-slash", "https", "path-prefixed-lease", "ipv6"],
)
def test_legitimate_forwarding_is_preserved(
    local_url: str, path: str, query: str, expected: str
) -> None:
    target = relay._parse_local_target(local_url)
    assert relay._local_upstream_url(target, path, query) == expected


@pytest.mark.parametrize(
    "path",
    [
        "/http://evil.example/steal",
        "/file:///etc/passwd",
        "/ws://evil.example/x",
        "/../../etc/passwd",
    ],
    ids=["absolute-url", "file-scheme", "ws-scheme", "dotdot"],
)
def test_escaping_paths_fail_closed_to_the_lease_root(path: str) -> None:
    target = relay._parse_local_target("http://127.0.0.1:8123/api")
    assert relay._local_upstream_url(target, path, "") == "http://127.0.0.1:8123/api/"


def test_authority_shaped_path_stays_on_the_leased_host() -> None:
    # Leading slashes are stripped, so ``//evil.example/x`` becomes a plain path
    # segment under the lease rather than a new authority.
    target = relay._parse_local_target("http://127.0.0.1:8123/api")
    url = relay._local_upstream_url(target, "//evil.example/x", "")
    assert url == "http://127.0.0.1:8123/api/evil.example/x"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1/x"])
def test_open_upstream_refuses_non_http_schemes(url: str) -> None:
    with pytest.raises(relay.SynthTunnelRelayError):
        relay._open_upstream(urllib.request.Request(url), timeout=1.0)
