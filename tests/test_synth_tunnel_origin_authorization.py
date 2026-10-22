"""Opt-in origin authorization: a declared header becomes the origin's Authorization.

The relay consumes ``Authorization`` (the lease's worker token) and the agent never
forwards caller credentials. An origin that authenticates each caller itself -- the
inference gateway's rollout session -- receives one declared header instead.
"""

from __future__ import annotations

import pytest

from synth_containers.tunnels.relay import (
    SynthTunnelProvider,
    SynthTunnelRelayAgent,
    _parse_local_target,
)


def _agent(declared: str | None) -> SynthTunnelRelayAgent:
    return SynthTunnelRelayAgent(
        lease_id="lease-test",
        local_target=_parse_local_target("http://127.0.0.1:9"),
        agent_connect={"transport": "ws", "url": "ws://127.0.0.1:9/agent", "agent_token": "t"},
        max_in_flight_requests=4,
        max_request_body_bytes=1024 * 1024,
        origin_authorization_header=declared,
    )


PAYLOAD = {
    "headers": [
        ["Authorization", "Bearer worker-jwt"],
        ["X-Synth-Origin-Authorization", "Bearer gw_session"],
        ["Content-Type", "application/json"],
    ]
}


def test_declared_header_becomes_origin_authorization_and_is_not_forwarded():
    agent = _agent("x-synth-origin-authorization")
    _target, origin_headers, _scope = agent._bind_origin(PAYLOAD)
    assert origin_headers == (("Authorization", "Bearer gw_session"),)
    assert {"authorization", "x-synth-origin-authorization"} <= agent._local_only_headers


def test_without_a_declaration_no_caller_credential_reaches_the_origin():
    agent = _agent(None)
    _target, origin_headers, _scope = agent._bind_origin(PAYLOAD)
    assert origin_headers == ()
    assert "x-synth-origin-authorization" not in agent._local_only_headers


@pytest.mark.parametrize("declared", ["authorization", "x-api-key", "host", "Not-Custom", "x-" + "a" * 65])
def test_only_custom_headers_may_be_declared(declared):
    provider = SynthTunnelProvider(control_plane=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        provider.open_synth_tunnel(
            "http://127.0.0.1:9", requested_ttl_seconds=60, metadata={}, capabilities={},
            wait_ready=False, origin_authorization_header=declared,
        )
