"""Reacquire a connector attachment after transport or actor restart."""

from __future__ import annotations
import asyncio
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from typing import Protocol
import threading
from uuid import UUID
from urllib.parse import urlparse
import httpx


@dataclass(frozen=True, slots=True)
class PreparedAttachment:
    url: str
    token: str = field(repr=False)
    session: UUID
    generation: int

    def __post_init__(self) -> None:
        parsed = urlparse(self.url)
        if (
            parsed.scheme not in {"ws", "wss"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or self.session.int == 0
            or type(self.generation) is not int
            or not 0 < self.generation <= 9_007_199_254_740_991
        ):
            raise ValueError("invalid prepared connector attachment")


class ConnectorAttachmentProvider(Protocol):
    def prepare(
        self, *, connector: UUID, client_instance: UUID, operation: UUID
    ) -> PreparedAttachment: ...


class HttpConnectorAttachmentProvider:
    """Use the gateway and an explicitly supplied attachment-token issuer.

    The token supplier can use project-local credentials or an existing secrets
    proxy. This component never reads a credential registry or Keychain.
    """

    def __init__(
        self,
        *,
        gateway_url: str,
        token_supplier: Callable[[UUID, UUID], str] | None = None,
        async_token_supplier: Callable[[UUID, UUID], Awaitable[str]] | None = None,
        timeout_seconds: float = 10,
    ) -> None:
        parsed = urlparse(gateway_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not 0 < timeout_seconds <= 30
            or (token_supplier is None) == (async_token_supplier is None)
        ):
            raise ValueError("invalid connector control endpoint")
        self._gateway = gateway_url.rstrip("/")
        self._token_supplier = token_supplier
        self._async_token_supplier = async_token_supplier
        self._timeout_seconds = timeout_seconds
        self._cancelled = threading.Event()
        self._preparation_lock = threading.Lock()
        self._preparation_loop = None
        self._preparation_task = None

    def cancel(self) -> None:
        """Retire this provider's in-flight I/O, preserving unknown admission intent."""
        self._cancelled.set()
        with self._preparation_lock:
            loop, task = self._preparation_loop, self._preparation_task
            if loop is not None and task is not None:
                try:
                    loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    if not loop.is_closed():
                        raise

    def reset(self) -> None:
        """Allow an explicit restart only after the prior prepare relinquished I/O."""
        with self._preparation_lock:
            if self._preparation_task is not None:
                raise RuntimeError("connector_prepare_still_running")
            self._cancelled.clear()

    def prepare(
        self, *, connector: UUID, client_instance: UUID, operation: UUID
    ) -> PreparedAttachment:
        if self._cancelled.is_set():
            raise RuntimeError("connector_prepare_cancelled")
        try:
            return asyncio.run(self._prepare(connector, client_instance, operation))
        except asyncio.CancelledError:
            raise RuntimeError("connector_prepare_cancelled") from None

    async def _prepare(self, connector, client_instance, operation):
        with self._preparation_lock:
            self._preparation_loop = asyncio.get_running_loop()
            self._preparation_task = asyncio.current_task()
        try:
            if self._cancelled.is_set():
                raise asyncio.CancelledError()
            async with asyncio.timeout(self._timeout_seconds):
                if self._async_token_supplier is not None:
                    token = await self._async_token_supplier(connector, client_instance)
                else:
                    # Legacy suppliers must bound their own synchronous I/O.
                    # ConnectorService uses the cancellable async issuer.
                    token = await asyncio.to_thread(
                        self._token_supplier, connector, client_instance
                    )
                async with httpx.AsyncClient(
                    timeout=self._timeout_seconds, follow_redirects=False
                ) as client:
                    response = await client.post(
                        f"{self._gateway}/v2/connectors/{connector}/attachment",
                        headers={"authorization": "Bearer " + token},
                        json={"operation": str(operation)},
                    )
                if response.status_code != 200:
                    # Do not embed arbitrary response bodies or credential headers
                    # in a retained startup error or log line.
                    raise RuntimeError(f"connector_prepare_http_{response.status_code}")
                value = response.json()
        finally:
            with self._preparation_lock:
                self._preparation_loop = None
                self._preparation_task = None
        if (
            value.get("connector") != str(connector)
            or value.get("protocol") != "synth-tunnel.v2"
            or value.get("transport") != "ws"
        ):
            raise ValueError("connector preparation scope mismatch")
        return PreparedAttachment(value["url"], token, UUID(value["session"]), value["generation"])
