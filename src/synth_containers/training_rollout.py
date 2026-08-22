"""Versioned task-container contract for hosted on-policy training.

The environment calls the hosted sampler over direct HTTPS. The SynthTunnel
is only the cloud-to-local rollout transport.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx


ROLLOUT_REQUEST_SCHEMA_VERSION = "training.rollout.request.v1"
ROLLOUT_ACTION_SCHEMA_VERSION = "training.rollout.action.v1"
ROLLOUT_REWARD_SCHEMA_VERSION = "training.rollout.reward.v1"
ROLLOUT_SUMMARY_SCHEMA_VERSION = "training.rollout.summary.v1"
ROLLOUT_CAPABILITIES_SCHEMA_VERSION = "training.rollout.capabilities.v1"


class TrainingRolloutError(RuntimeError):
    """Secret-free, stable failure at the training rollout boundary."""


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def training_capabilities(
    *,
    target_id: str,
    runtime_family: str,
    container_digest: str,
    dataset_digest: str,
    max_concurrency: int,
) -> dict[str, Any]:
    if not target_id.strip() or not container_digest.strip() or max_concurrency < 1:
        raise TrainingRolloutError("invalid_training_capability_identity")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", dataset_digest) is None:
        raise TrainingRolloutError("invalid_training_dataset_digest")
    payload: dict[str, Any] = {
        "schema_version": ROLLOUT_CAPABILITIES_SCHEMA_VERSION,
        "container_id": target_id,
        "container_digest": container_digest,
        "dataset_digest": dataset_digest,
        "task_id": runtime_family,
        "protocol_versions": [
            ROLLOUT_REQUEST_SCHEMA_VERSION,
            ROLLOUT_ACTION_SCHEMA_VERSION,
            ROLLOUT_REWARD_SCHEMA_VERSION,
            ROLLOUT_SUMMARY_SCHEMA_VERSION,
        ],
        "operations": ["rollout", "reward", "heartbeat"],
        "max_concurrency": max_concurrency,
        "supports_idempotency": True,
        "supports_sampler_https": True,
        # A tunnel relay can require close while the direct sampler can retain
        # connections. The job requirement chooses one explicitly.
        "connection_modes": ["close", "keep_alive"],
    }
    payload["capability_hash"] = canonical_sha256(payload)
    return payload


@dataclass(frozen=True, slots=True)
class SamplerEndpoint:
    url: str
    bearer_token: str = field(repr=False)
    connection_mode: str = "keep_alive"

    def validate(self, *, allow_loopback_http: bool = False) -> None:
        parsed = urlparse(self.url)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (allow_loopback_http and loopback):
            raise TrainingRolloutError("sampler_endpoint_requires_https")
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.hostname
            or not parsed.path
        ):
            raise TrainingRolloutError("sampler_endpoint_invalid")
        if not self.bearer_token.strip() or any(ch in self.bearer_token for ch in "\r\n"):
            raise TrainingRolloutError("sampler_auth_missing")
        if self.connection_mode not in {"close", "keep_alive"}:
            raise TrainingRolloutError("sampler_connection_mode_invalid")


@dataclass(frozen=True, slots=True)
class SamplerResult:
    text: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    log_probs: tuple[float, ...]
    usage: Mapping[str, Any]


class HostedSamplerClient:
    """REB-059 request discipline for the direct hosted sampler leg.

    Transport retries keep the idempotency key stable. A 401 causes at most
    one synchronized refresh for the observed token generation.
    """

    def __init__(
        self,
        endpoint: SamplerEndpoint,
        *,
        refresh: Callable[[], str] | None = None,
        transport: httpx.BaseTransport | None = None,
        max_transport_attempts: int = 3,
        timeout_seconds: float = 90.0,
        allow_loopback_http: bool = False,
    ) -> None:
        endpoint.validate(allow_loopback_http=allow_loopback_http)
        if max_transport_attempts < 1 or max_transport_attempts > 5:
            raise TrainingRolloutError("sampler_retry_budget_invalid")
        self._url = endpoint.url
        self._token = endpoint.bearer_token
        self._connection_mode = endpoint.connection_mode
        self._refresh = refresh
        self._refresh_lock = threading.Lock()
        self._token_generation = 0
        self._max_transport_attempts = max_transport_attempts
        self._timeout_seconds = timeout_seconds
        self._client = httpx.Client(transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HostedSamplerClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample(self, payload: Mapping[str, Any], *, idempotency_key: str) -> SamplerResult:
        if not idempotency_key.strip() or any(ch in idempotency_key for ch in "\r\n"):
            raise TrainingRolloutError("sampler_idempotency_key_required")
        observed_generation = self._token_generation
        refreshed = False
        transport_attempt = 0
        while transport_attempt < self._max_transport_attempts:
            transport_attempt += 1
            try:
                response = self._client.post(
                    self._url,
                    json=dict(payload),
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Idempotency-Key": idempotency_key,
                        "Accept": "application/json",
                        "Connection": (
                            "close" if self._connection_mode == "close" else "keep-alive"
                        ),
                    },
                    timeout=self._timeout_seconds,
                )
            except (httpx.TransportError, TimeoutError) as exc:
                if transport_attempt >= self._max_transport_attempts:
                    raise TrainingRolloutError("sampler_transport_exhausted") from exc
                time.sleep(min(0.05 * (2 ** (transport_attempt - 1)), 0.25))
                continue
            if response.status_code == 401 and not refreshed and self._refresh is not None:
                self._refresh_token(observed_generation)
                refreshed = True
                # Auth refusal is not counted against bounded transport retry.
                transport_attempt -= 1
                continue
            if response.status_code in {408, 425, 429, 500, 502, 503, 504}:
                if transport_attempt < self._max_transport_attempts:
                    time.sleep(min(0.05 * (2 ** (transport_attempt - 1)), 0.25))
                    continue
                raise TrainingRolloutError(f"sampler_http_retry_exhausted_{response.status_code}")
            if response.status_code == 401:
                raise TrainingRolloutError("sampler_auth_refused")
            if response.status_code >= 400:
                raise TrainingRolloutError(f"sampler_http_{response.status_code}")
            try:
                body = response.json()
            except ValueError as exc:
                raise TrainingRolloutError("sampler_response_invalid") from exc
            return _parse_sampler_result(body)
        raise TrainingRolloutError("sampler_transport_exhausted")

    def _refresh_token(self, observed_generation: int) -> None:
        assert self._refresh is not None
        with self._refresh_lock:
            if self._token_generation != observed_generation:
                return
            token = self._refresh()
            if not isinstance(token, str) or not token.strip() or any(ch in token for ch in "\r\n"):
                raise TrainingRolloutError("sampler_refresh_invalid")
            self._token = token
            self._token_generation += 1


def _parse_sampler_result(body: Any) -> SamplerResult:
    if not isinstance(body, dict):
        raise TrainingRolloutError("sampler_response_invalid")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise TrainingRolloutError("sampler_response_text_missing")
    token_ids = body.get("token_ids") or []
    prompt_token_ids = body.get("prompt_token_ids") or []
    log_probs = body.get("log_probs") or []
    if not isinstance(token_ids, list) or not all(isinstance(value, int) for value in token_ids):
        raise TrainingRolloutError("sampler_response_token_ids_invalid")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(value, int) for value in prompt_token_ids
    ):
        raise TrainingRolloutError("sampler_response_prompt_token_ids_invalid")
    if not isinstance(log_probs, list) or not all(
        isinstance(value, int | float) and not isinstance(value, bool) for value in log_probs
    ):
        raise TrainingRolloutError("sampler_response_log_probs_invalid")
    if not prompt_token_ids or not token_ids or len(log_probs) != len(token_ids):
        raise TrainingRolloutError("sampler_response_token_alignment_invalid")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return SamplerResult(
        text=text,
        prompt_token_ids=tuple(prompt_token_ids),
        token_ids=tuple(token_ids),
        log_probs=tuple(float(value) for value in log_probs),
        usage=usage,
    )
