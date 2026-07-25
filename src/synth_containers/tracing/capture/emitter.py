"""Dependency-light client for emitting application facts from a workload."""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from ..models.actors import SessionStatus
from ..models.identity import TraceContextV1


class TraceEmitter:
    """Send application events and artifacts to the bound capture collector."""

    def __init__(
        self,
        base_url: str,
        context: TraceContextV1,
        timeout: float = 10.0,
        collector_token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.context = context
        self.collector_token = collector_token
        self._client = httpx.Client(timeout=timeout)
        self._registered_context_tokens: dict[str, str] = {}

    @classmethod
    def from_environment(cls, *, timeout: float = 10.0) -> "TraceEmitter":
        context = TraceContextV1.from_environment()
        if context is None:
            raise RuntimeError("Synth trace context is not present in the environment")
        if not context.collector_url:
            raise RuntimeError("SYNTH_TRACE_COLLECTOR_URL is not set")
        return cls(
            context.collector_url,
            context,
            timeout=timeout,
            collector_token=str(os.environ.get("SYNTH_TRACE_COLLECTOR_TOKEN") or "") or None,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-synth-trace-id": self.context.trace_id,
            "x-synth-capture-id": self.context.capture_id,
            "x-synth-actor-id": self.context.actor_id,
            "x-synth-session-id": self.context.actor_session_id,
            "content-type": "application/json",
        }
        if self.collector_token:
            headers["authorization"] = f"Bearer {self.collector_token}"
        return headers

    def provider_headers(self) -> dict[str, str]:
        """Headers for attributing provider-proxy calls to this actor context."""

        if not self.collector_token:
            raise RuntimeError("provider call attribution requires the collector token")
        return {
            "x-synth-trace-id": self.context.trace_id,
            "x-synth-capture-id": self.context.capture_id,
            "x-synth-actor-id": self.context.actor_id,
            "x-synth-session-id": self.context.actor_session_id,
            "x-synth-context-token": self.collector_token,
        }

    def event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        occurred_at: str | None = None,
        caused_by: tuple[str, ...] = (),
        structural: dict[str, Any] | None = None,
        actor_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        response = self._client.post(
            f"{self.base_url}/v1/events",
            headers=self._headers(),
            json={
                "event_type": event_type,
                "payload": payload,
                "occurred_at": occurred_at,
                "caused_by": list(caused_by),
                "structural": structural,
                "actor_id": actor_id or self.context.actor_id,
                "session_id": session_id or self.context.actor_session_id,
            },
        )
        response.raise_for_status()
        return str(response.json()["envelope_id"])

    def artifact(
        self,
        role: str,
        media_type: str,
        content: bytes,
        logical_name: str,
        *,
        visibility: str = "private",
        actor_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        response = self._client.post(
            f"{self.base_url}/v1/artifacts",
            headers=self._headers(),
            json={
                "role": role,
                "media_type": media_type,
                "content_base64": base64.b64encode(content).decode("ascii"),
                "logical_name": logical_name,
                "visibility": visibility,
                "actor_id": actor_id or self.context.actor_id,
                "session_id": session_id or self.context.actor_session_id,
            },
        )
        response.raise_for_status()
        return str(response.json()["artifact_id"])

    def register_context(
        self,
        child: TraceContextV1,
        *,
        actor: dict[str, Any],
        session: dict[str, Any],
    ) -> str:
        response = self._client.post(
            f"{self.base_url}/v1/contexts",
            headers=self._headers(),
            json={"context": child.to_dict(), "actor": actor, "session": session},
        )
        response.raise_for_status()
        receipt = response.json()
        capture_id = str(receipt["capture_id"])
        collector_token = str(receipt.get("collector_token") or "")
        if not collector_token:
            raise RuntimeError("child registration did not return a collector capability")
        self._registered_context_tokens[capture_id] = collector_token
        return capture_id

    def registered_context_token(
        self,
        child: TraceContextV1 | str,
    ) -> str:
        """Return the ephemeral capability minted by ``register_context``."""

        capture_id = child.capture_id if isinstance(child, TraceContextV1) else child
        try:
            return self._registered_context_tokens[capture_id]
        except KeyError as exc:
            raise ValueError("child context was not registered by this emitter") from exc

    def emitter_for_registered_context(
        self,
        child: TraceContextV1,
        *,
        timeout: float = 10.0,
    ) -> "TraceEmitter":
        """Create a child emitter with only that child's scoped capability."""

        return TraceEmitter(
            self.base_url,
            child,
            timeout=timeout,
            collector_token=self.registered_context_token(child),
        )

    def finish(
        self,
        *,
        status: SessionStatus | str = SessionStatus.COMPLETED,
        ended_at: str | None = None,
    ) -> str:
        """Durably finish this delegated child session.

        Root-session lifecycle remains owned by ``CaptureSupervisor.finalize``.
        Repeating the same terminal fact is idempotent; a conflicting terminal
        status fails at the collector authority.
        """

        response = self._client.post(
            f"{self.base_url}/v1/sessions/finish",
            headers=self._headers(),
            json={"status": str(status), "ended_at": ended_at},
        )
        response.raise_for_status()
        return str(response.json()["envelope_id"])

    def finish_session(
        self,
        *,
        status: SessionStatus | str = SessionStatus.COMPLETED,
        ended_at: str | None = None,
    ) -> str:
        """Compatibility spelling for ``finish``."""

        return self.finish(status=status, ended_at=ended_at)

    def flush(self) -> None:
        """The synchronous transport has no buffered writes."""

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TraceEmitter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["TraceEmitter"]
