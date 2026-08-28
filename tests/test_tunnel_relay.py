"""Regression coverage for long-lived SynthTunnel credential rotation."""

from __future__ import annotations

import pytest

from synth_containers.tunnels.relay import (
    AttachedSynthTunnelLease,
    SynthTunnelRelayError,
)


class _ControlPlane:
    def close_synth_lease(self, lease_id: str) -> None:
        del lease_id


class _Agent:
    def __init__(self) -> None:
        self.agent_tokens: list[str] = []

    def update_agent_token(self, token: str) -> None:
        self.agent_tokens.append(token)

    def stop(self) -> None:
        pass


def _lease(agent: _Agent) -> AttachedSynthTunnelLease:
    return AttachedSynthTunnelLease(
        lease_id="lease-1",
        public_url="https://relay.example/lease-1",
        worker_token="worker-old",
        expires_at="2026-08-13T23:00:00Z",
        connector_mode="relay",
        _control_plane=_ControlPlane(),
        _agent=agent,
        _local_health_url="http://127.0.0.1:8097/health",
        _attach_timeout_seconds=30.0,
        _ready_timeout_seconds=60.0,
    )


def test_attached_lease_adopts_heartbeat_rotated_credentials() -> None:
    agent = _Agent()
    lease = _lease(agent)

    lease.update_credentials(
        worker_token="worker-new",
        expires_at="2026-08-14T00:00:00Z",
        agent_token="agent-new",
    )

    assert lease.worker_token == "worker-new"
    assert lease.expires_at == "2026-08-14T00:00:00Z"
    assert agent.agent_tokens == ["agent-new"]


def test_credential_rotation_fails_closed_without_mutating_a_closed_lease() -> None:
    agent = _Agent()
    lease = _lease(agent)
    lease._closed = True

    with pytest.raises(SynthTunnelRelayError, match="lease is closed"):
        lease.update_credentials(
            worker_token="worker-new",
            expires_at=None,
            agent_token="agent-new",
        )

    assert lease.worker_token == "worker-old"
    assert lease.expires_at == "2026-08-13T23:00:00Z"
    assert agent.agent_tokens == []
