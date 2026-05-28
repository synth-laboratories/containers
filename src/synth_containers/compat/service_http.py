from __future__ import annotations

from typing import Any

import httpx


class ServiceRuntimeError(RuntimeError):
    pass


def _as_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _to_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_service_headers(target: Any, org_api_key: str | None = None) -> dict[str, str]:
    token = str(getattr(target, "auth_token", "") or org_api_key or "").strip()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["x-api-key"] = token
    return headers


async def request_service_json(
    *,
    target: Any,
    path: str,
    method: str = "GET",
    org_api_key: str | None = None,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    error_label: str,
) -> dict[str, Any]:
    service_url = str(getattr(target, "service_url", "") or "").rstrip("/")
    if not service_url:
        raise ServiceRuntimeError(f"{error_label} is missing service_url.")

    limits = getattr(target, "limits", {})
    timeout_s = max(
        _to_int(limits.get("timeout_s") if isinstance(limits, dict) else None, default=600),
        30,
    )
    url = f"{service_url}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as client:
        response = await client.request(
            method,
            url,
            headers=build_service_headers(target, org_api_key),
            json=json_body,
            params=params,
        )
        response.raise_for_status()
        if not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise ServiceRuntimeError(f"{error_label} returned a non-object payload for {path}.")
        return dict(payload)


async def fetch_service_health(*, target: Any, error_label: str) -> dict[str, Any]:
    return await request_service_json(target=target, path="/health", error_label=error_label)


async def fetch_service_info(*, target: Any, error_label: str) -> dict[str, Any]:
    return await request_service_json(target=target, path="/info", error_label=error_label)


async def fetch_service_metadata(*, target: Any, error_label: str) -> dict[str, Any]:
    return await request_service_json(target=target, path="/metadata", error_label=error_label)


async def fetch_service_task_info(
    *,
    target: Any,
    seeds: list[int],
    error_label: str,
) -> dict[str, Any] | list[dict[str, Any]]:
    seed_values = seeds or [0]
    payloads = [
        await request_service_json(
            target=target,
            path="/task_info",
            params={"seed": seed},
            error_label=error_label,
        )
        for seed in seed_values
    ]
    if len(payloads) == 1:
        return payloads[0]
    return payloads


def _first_numeric(values: list[Any]) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return None


def derive_service_rollout_score(
    result: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> float | None:
    metrics_payload = dict(metrics or result.get("metrics") or {})
    reward_info = _as_dict(result.get("reward_info"))
    outcome_objectives = _as_dict(reward_info.get("outcome_objectives"))
    metadata = _as_dict(result.get("metadata"))
    rollout_payload = _as_dict(result.get("rollout"))
    rollout_metadata = _as_dict(rollout_payload.get("metadata"))
    reward_details = _as_dict(reward_info.get("details"))
    return _first_numeric(
        [
            result.get("score"),
            metrics_payload.get("outcome_reward"),
            metrics_payload.get("outcome_score"),
            reward_info.get("outcome_reward"),
            reward_info.get("outcome_score"),
            reward_info.get("shaped_outcome_reward"),
            outcome_objectives.get("reward"),
            outcome_objectives.get("score"),
            metadata.get("total_reward"),
            rollout_metadata.get("total_reward"),
            reward_details.get("total_reward"),
        ]
    )


def normalize_service_rollout_result(
    *,
    result: dict[str, Any],
    rollout_id: str,
    trace_correlation_id: str | None,
) -> dict[str, Any]:
    metrics = dict(result.get("metrics") or {})
    score = derive_service_rollout_score(result, metrics)
    if score is not None:
        metrics.setdefault("outcome_reward", float(score))
        result["score"] = float(score)
    result["metrics"] = metrics
    result.setdefault("rollout_id", rollout_id)
    result.setdefault("trace_correlation_id", trace_correlation_id)
    result.setdefault("success", not bool(result.get("error")))
    return result


async def execute_service_rollout(
    *,
    target: Any,
    rollout_id: str,
    input_payload: dict[str, Any],
    org_api_key: str | None,
    error_label: str,
) -> dict[str, Any]:
    result = await request_service_json(
        target=target,
        path="/rollout",
        method="POST",
        org_api_key=org_api_key,
        json_body=input_payload,
        error_label=error_label,
    )
    return normalize_service_rollout_result(
        result=result,
        rollout_id=rollout_id,
        trace_correlation_id=input_payload.get("trace_correlation_id"),
    )
