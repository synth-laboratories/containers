"""Declared physical attachment scopes and origin snapshots.

Legacy lease attachment and shared connector attachment are explicit modes;
connector identity is never passed as a pretend lease identifier.
"""

from __future__ import annotations
from dataclasses import dataclass
from uuid import UUID
from .targets import LocalTargetRegistry


@dataclass(frozen=True, slots=True)
class LocalTarget:
    base_url: str
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class LeaseAttachment:
    lease_id: str
    target: LocalTarget
    # The relay consumes ``Authorization`` (the worker token) and the agent never
    # forwards it. An origin that authenticates each caller itself (the inference
    # gateway's rollout session) receives this declared request header as its
    # ``Authorization`` instead; it is opt-in per lease and never forwarded as-is.
    origin_authorization_header: str | None = None


@dataclass(frozen=True, slots=True)
class ConnectorAttachment:
    connector: UUID
    client_instance: UUID
    targets: LocalTargetRegistry

    def __post_init__(self) -> None:
        if self.connector.int == 0 or self.client_instance.int == 0:
            raise ValueError("nil connector attachment identity")


Attachment = LeaseAttachment | ConnectorAttachment
