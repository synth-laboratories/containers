from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI


class HTTPContainerClient:
    def __init__(
        self,
        *,
        base_url: str = "http://testserver",
        client: httpx.AsyncClient,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    @classmethod
    def from_app(
        cls,
        app: FastAPI,
        *,
        base_url: str = "http://testserver",
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
    ) -> "HTTPContainerClient":
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url=base_url)
        return cls(
            base_url=base_url,
            client=client,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
    ) -> "HTTPContainerClient":
        client = httpx.AsyncClient(base_url=url.rstrip("/"))
        return cls(
            base_url=url,
            client=client,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HTTPContainerClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        optional: bool = False,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method.upper(), path, json=payload)
                if optional and response.status_code == 404:
                    return {}
                response.raise_for_status()
                if not str(response.text or "").strip():
                    return {}
                body = response.json()
                if isinstance(body, dict):
                    return body
                return {"value": body}
            except httpx.HTTPStatusError as exc:
                if optional and exc.response.status_code == 404:
                    return {}
                body = str(exc.response.text or "").strip()
                detail = f"{exc.response.status_code} {exc.response.reason_phrase}"
                if body:
                    detail = f"{detail}: {body[:1000]}"
                last_error = RuntimeError(detail)
            except (httpx.RequestError, ValueError) as exc:
                last_error = exc
            if optional:
                return {}
            if attempt < self._max_retries:
                await asyncio.sleep(self._retry_backoff_seconds * (attempt + 1))
        raise RuntimeError(f"container request failed {method.upper()} {path}: {last_error}") from last_error

    async def _get(self, path: str, *, optional: bool = False) -> dict[str, Any]:
        return await self._request("GET", path, optional=optional)

    async def _post(self, path: str, payload: dict[str, Any] | None = None, *, optional: bool = False) -> dict[str, Any]:
        return await self._request("POST", path, payload=payload or {}, optional=optional)

    async def root(self) -> dict[str, Any]:
        return await self._get("/")

    async def health(self) -> dict[str, Any]:
        return await self._get("/health")

    async def metadata(self) -> dict[str, Any]:
        return await self._get("/metadata")

    async def info(self) -> dict[str, Any]:
        return await self._get("/info", optional=True)

    async def task_info(self) -> dict[str, Any]:
        return await self._get("/task_info", optional=True)

    async def task_catalog(self) -> dict[str, Any]:
        return await self._get("/task_catalog", optional=True)

    async def program(self) -> dict[str, Any]:
        return await self._get("/program", optional=True)

    async def dataset(self) -> dict[str, Any]:
        return await self._get("/dataset", optional=True)

    async def dataset_rows(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._post("/dataset/rows", payload or {}, optional=True)

    async def compatibility(self, target: str | None = None) -> dict[str, Any]:
        if target is None or not str(target).strip():
            return await self._get("/compatibility", optional=True)
        encoded_target = quote(str(target).strip(), safe="")
        return await self._get(f"/compatibility?target={encoded_target}", optional=True)

    async def rollout(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.setdefault("submission_mode", "sync")
        return await self._post("/rollout", body)

    async def rollouts(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.setdefault("submission_mode", "async")
        return await self._post("/rollouts", body)

    async def get_rollout(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}", optional=True)

    async def get_state(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/state", optional=True)

    async def summary(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/summary", optional=True)

    async def usage(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/usage", optional=True)

    async def artifacts(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/artifacts", optional=True)

    async def events(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/events", optional=True)

    async def trace(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/trace", optional=True)

    async def pause(self, rollout_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._post(f"/rollouts/{rollout_id}/pause", payload)

    async def terminate(self, rollout_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._post(f"/rollouts/{rollout_id}/terminate", payload)

    async def checkpoint(self, rollout_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._post(f"/rollouts/{rollout_id}/checkpoints", payload)

    async def list_rollout_checkpoints(self, rollout_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/checkpoints", optional=True)

    async def get_rollout_checkpoint(self, rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
        return await self._get(f"/rollouts/{rollout_id}/checkpoints/{checkpoint_id}", optional=True)

    async def list_checkpoints(self) -> dict[str, Any]:
        return await self._get("/checkpoints", optional=True)

    async def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        return await self._get(f"/checkpoints/{checkpoint_id}", optional=True)

    async def update_checkpoint_labels(
        self,
        checkpoint_id: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._post(f"/checkpoints/{checkpoint_id}/labels", payload, optional=True)

    async def resume(self, rollout_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._post(f"/rollouts/{rollout_id}/resume", payload)

    async def fork(self, rollout_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._post(f"/rollouts/{rollout_id}/fork", payload)
