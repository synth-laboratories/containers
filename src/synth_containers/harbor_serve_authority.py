"""Managed Harbor's deployment-scoped authority port; no provider keys enter trials."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol

import httpx


class HarborAuthority(Protocol):
    async def call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpHarborAuthority:
    def __init__(self, *, base_url: str, token: str, client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.client = client

    async def call(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action not in {"prepare", "start", "finish", "observe", "recover", "sandbox"}:
            raise ValueError("harbor_authority_action_invalid")
        response = await self.client.post(
            f"{self.base_url}/{action}",
            json=payload,
            headers={"Authorization": f"Bearer {self.token}"},
            follow_redirects=False,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError("harbor_authority_response_invalid")
        return value


@dataclass(frozen=True)
class TrialAuthority:
    authority: HarborAuthority
    trial_id: str


trial_authority: ContextVar[TrialAuthority] = ContextVar("managed_harbor_trial_authority")
