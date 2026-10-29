"""SynthTunnel relay error shared by the agent and its origin I/O."""

from __future__ import annotations


class SynthTunnelRelayError(RuntimeError):
    """The SynthTunnel relay could not attach or proxy a request safely."""


__all__ = ["SynthTunnelRelayError"]
