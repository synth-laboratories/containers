"""HTTP trust-boundary parsers for the containers-compat façade.

JSON bodies become dataclasses here. `CompatPlatform.start_rollout` reads
`CreateRolloutRequest` fields. `to_platform_dict` is an explicit dump, not the
platform entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

from ..event_log import validate_rollout_id

_CREATE = "POST /rollouts"
_PREPARE = "POST /rollouts/prepare"
_REWARD = "POST /reward"
_COMBINE = "POST /reward/combine"
_POLICY_CONFIG = "POST /policy-configs"
_PUT_POLICY = "PUT /policy"

_TRANSPORTS = frozenset({"poll", "sse", "websocket", "auto"})
_REWARD_MODES = frozenset({"terminal", "provisional"})
_SUBMISSION_MODES = frozenset({"sync", "async"})
ISOLATED_POLICY_HARNESS = "isolated_policy_process"


class RequestParseError(ValueError):
    """Typed parse failure at the HTTP edge. `status_code` is the HTTP mapping."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class TelemetrySpec:
    enabled: bool
    transport: str
    retention: Optional[str]


@dataclass(frozen=True)
class PolicyRefSpec:
    harness: Optional[str]
    config: Optional[str]
    code: Any


@dataclass(frozen=True)
class CreateRolloutRequest:
    rollout_id: Optional[str]
    telemetry: TelemetrySpec
    policy_ref: PolicyRefSpec
    recipe: Optional[dict[str, Any]]
    task_instance_id: Optional[str]
    evaluation_plan_ref: Optional[str]
    world_ref: Optional[str]
    submission_mode: str
    omit_reward: bool
    outcome: Optional[str]
    slot: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RewardPostRequest:
    rollout_id: Optional[str]
    evidence: Optional[dict[str, Any]]
    mode: str
    rescore: bool
    evaluation_plan_ref: Optional[str]
    after_sequence: Optional[int]


@dataclass(frozen=True)
class CombineRewardRequest:
    bases: dict[str, Optional[float]]
    required: list[str]


@dataclass(frozen=True)
class PolicyConfigRequest:
    config_id: str
    harness: Optional[str]
    config: dict[str, Any]


@dataclass(frozen=True)
class PutPolicyRequest:
    code: Any
    harness: Optional[str]


def require_json_object(body: object, *, operation: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise RequestParseError(
            f"{operation}: JSON body must be an object, got {type(body).__name__}"
        )
    return cast(dict[str, Any], body)


def parse_create_rollout(
    body: dict,
    *,
    operation: str = _CREATE,
    require_policy_pin: bool = True,
) -> CreateRolloutRequest:
    raw = require_json_object(body, operation=operation)
    telemetry = _parse_telemetry(raw, operation=operation)
    policy_ref = _parse_policy_ref(raw, operation=operation)
    if require_policy_pin:
        _require_policy_pin(policy_ref, operation=operation)
    recipe = _optional_object(raw, "recipe", operation=operation)
    metadata = _optional_object(raw, "metadata", operation=operation)
    if metadata is None:
        metadata = {}
    task_instance_id = _optional_str(raw, "task_instance_id", operation=operation)
    if task_instance_id is None:
        task_instance_id = _task_seed_id(raw, operation=operation)
    omit_reward = _optional_bool(raw, "omit_reward", operation=operation)
    if omit_reward is None:
        omit_reward = False
    if "omit_reward" in metadata:
        meta_flag = metadata["omit_reward"]
        if not isinstance(meta_flag, bool):
            raise RequestParseError(
                f"{operation}: metadata.omit_reward must be a boolean, got {type(meta_flag).__name__}"
            )
        omit_reward = omit_reward or meta_flag
    slot = _optional_str(raw, "slot", operation=operation)
    if slot is None:
        slot = _optional_str(raw, "stream_slot", operation=operation)
    if slot is None:
        slot = "stream"
    submission_mode = _optional_str(raw, "submission_mode", operation=operation)
    if submission_mode is None:
        submission_mode = "sync"
    elif submission_mode not in _SUBMISSION_MODES:
        raise RequestParseError(
            f"{operation}: submission_mode must be sync or async, got {submission_mode!r}"
        )
    rollout_id = _optional_str(raw, "rollout_id", operation=operation)
    if rollout_id == "":
        rollout_id = None
    if rollout_id is not None:
        try:
            rollout_id = validate_rollout_id(rollout_id)
        except ValueError as exc:
            raise RequestParseError(f"{operation}: {exc}") from exc
    return CreateRolloutRequest(
        rollout_id=rollout_id,
        telemetry=telemetry,
        policy_ref=policy_ref,
        recipe=recipe,
        task_instance_id=task_instance_id,
        evaluation_plan_ref=_optional_str(raw, "evaluation_plan_ref", operation=operation),
        world_ref=_optional_str(raw, "world_ref", operation=operation),
        submission_mode=submission_mode,
        omit_reward=omit_reward,
        outcome=_optional_str(raw, "outcome", operation=operation) or None,
        slot=slot,
        metadata=metadata,
    )


def parse_prepare_rollout(body: dict) -> CreateRolloutRequest:
    # Prepare reserves stream identity. The caller names the policy at start.
    return parse_create_rollout(body, operation=_PREPARE, require_policy_pin=False)


def to_platform_dict(req: CreateRolloutRequest) -> dict[str, Any]:
    """Explicit dump of a parsed create-rollout request. Prefer the dataclass."""
    return {
        "rollout_id": req.rollout_id,
        "telemetry": {
            "enabled": req.telemetry.enabled,
            "transport": req.telemetry.transport,
            "retention": req.telemetry.retention,
        },
        "policy_ref": {
            "harness": req.policy_ref.harness,
            "config": req.policy_ref.config,
            "code": req.policy_ref.code,
        },
        "policy_config": req.policy_ref.config,
        "recipe": req.recipe,
        "task_instance_id": req.task_instance_id,
        "task": None,
        "evaluation_plan_ref": req.evaluation_plan_ref,
        "world_ref": req.world_ref,
        "submission_mode": req.submission_mode,
        "omit_reward": req.omit_reward,
        "outcome": req.outcome,
        "slot": req.slot,
        "stream_slot": req.slot,
        "metadata": req.metadata,
    }


def parse_reward_post(body: dict) -> RewardPostRequest:
    raw = require_json_object(body, operation=_REWARD)
    rollout_id = _optional_str(raw, "rollout_id", operation=_REWARD)
    if rollout_id == "":
        rollout_id = None
    evidence = _optional_object(raw, "evidence", operation=_REWARD)
    if (rollout_id is None) == (evidence is None):
        raise RequestParseError(f"{_REWARD}: requires exactly one of rollout_id or evidence")
    mode = _optional_str(raw, "mode", operation=_REWARD)
    if mode is None:
        mode = "terminal"
    elif mode not in _REWARD_MODES:
        raise RequestParseError(f"{_REWARD}: mode must be terminal or provisional, got {mode!r}")
    rescore = _optional_bool(raw, "rescore", operation=_REWARD)
    if rescore is None:
        rescore = False
    return RewardPostRequest(
        rollout_id=rollout_id,
        evidence=evidence,
        mode=mode,
        rescore=rescore,
        evaluation_plan_ref=_optional_str(raw, "evaluation_plan_ref", operation=_REWARD),
        after_sequence=_optional_int(raw, "after_sequence", operation=_REWARD),
    )


def parse_combine_reward(body: dict) -> CombineRewardRequest:
    raw = require_json_object(body, operation=_COMBINE)
    bases_raw = _optional_object(raw, "bases", operation=_COMBINE)
    if bases_raw is None:
        bases_raw = {}
    bases: dict[str, Optional[float]] = {}
    for name, value in bases_raw.items():
        key = str(name)
        if value is None:
            bases[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestParseError(
                f"{_COMBINE}: bases[{key!r}] must be a number or null, got {type(value).__name__}"
            )
        bases[key] = float(value)
    if "required" not in raw or raw["required"] is None:
        required = list(bases.keys())
    else:
        required_raw = raw["required"]
        if not isinstance(required_raw, list):
            raise RequestParseError(
                f"{_COMBINE}: required must be an array of strings, got {type(required_raw).__name__}"
            )
        required = []
        for item in required_raw:
            if not isinstance(item, str):
                raise RequestParseError(
                    f"{_COMBINE}: required entries must be strings, got {type(item).__name__}"
                )
            required.append(item)
    return CombineRewardRequest(bases=bases, required=required)


def parse_policy_config(body: dict, *, path_config_id: Optional[str] = None) -> PolicyConfigRequest:
    raw = require_json_object(body, operation=_POLICY_CONFIG)
    config_id = path_config_id or _optional_str(raw, "config_id", operation=_POLICY_CONFIG)
    if not config_id:
        raise RequestParseError(f"{_POLICY_CONFIG}: config_id required")
    config = _optional_object(raw, "config", operation=_POLICY_CONFIG)
    if config is None:
        config = {}
    return PolicyConfigRequest(
        config_id=config_id,
        harness=_optional_str(raw, "harness", operation=_POLICY_CONFIG),
        config=config,
    )


def to_policy_config_dict(req: PolicyConfigRequest) -> dict[str, Any]:
    return {
        "config_id": req.config_id,
        "harness": req.harness,
        "config": req.config,
    }


def parse_put_policy(body: dict) -> PutPolicyRequest:
    raw = require_json_object(body, operation=_PUT_POLICY)
    if "code" not in raw:
        raise RequestParseError(f"{_PUT_POLICY}: code is required")
    return PutPolicyRequest(
        code=raw["code"],
        harness=_optional_str(raw, "harness", operation=_PUT_POLICY),
    )


def to_put_policy_dict(req: PutPolicyRequest) -> dict[str, Any]:
    return {"code": req.code, "harness": req.harness}


def _parse_telemetry(body: dict[str, Any], *, operation: str) -> TelemetrySpec:
    if "telemetry" not in body or body["telemetry"] is None:
        return TelemetrySpec(enabled=True, transport="sse", retention=None)
    raw = body["telemetry"]
    if not isinstance(raw, dict):
        raise RequestParseError(
            f"{operation}: telemetry must be an object, got {type(raw).__name__}"
        )
    enabled = _optional_bool(raw, "enabled", operation=operation, field="telemetry.enabled")
    if enabled is None:
        enabled = True
    transport = _optional_str(raw, "transport", operation=operation, field="telemetry.transport")
    if transport is None:
        transport = "sse"
    if enabled and transport not in _TRANSPORTS:
        raise RequestParseError(
            f"{operation}: telemetry.transport must be poll, sse, websocket, or auto when "
            f"enabled, got {transport!r}"
        )
    if not enabled and transport not in _TRANSPORTS:
        raise RequestParseError(
            f"{operation}: telemetry.transport must be poll, sse, websocket, or auto, got {transport!r}"
        )
    retention = _optional_str(raw, "retention", operation=operation, field="telemetry.retention")
    return TelemetrySpec(enabled=enabled, transport=transport, retention=retention)


def _parse_policy_ref(body: dict[str, Any], *, operation: str) -> PolicyRefSpec:
    sibling_config = _optional_str(body, "policy_config", operation=operation)
    if "policy_ref" not in body or body["policy_ref"] is None:
        return PolicyRefSpec(harness=None, config=sibling_config, code=None)
    raw = body["policy_ref"]
    if not isinstance(raw, dict):
        raise RequestParseError(
            f"{operation}: policy_ref must be an object, got {type(raw).__name__}"
        )
    config = _optional_str(raw, "config", operation=operation, field="policy_ref.config")
    if config is None:
        config = sibling_config
    code = raw["code"] if "code" in raw else None
    return PolicyRefSpec(
        harness=_optional_str(raw, "harness", operation=operation, field="policy_ref.harness"),
        config=config,
        code=code,
    )


def _require_policy_pin(policy_ref: PolicyRefSpec, *, operation: str) -> None:
    """Start must name the policy. Missing is not luna_med or the target default.

    # See: workshop/docs/container_compat.md (policy_ref = harness + config + optional code)
    """
    harness = (policy_ref.harness or "").strip()
    if not harness:
        raise RequestParseError(
            f"{operation}: policy_ref.harness is required; the platform does not pick a recipe"
        )
    if harness == ISOLATED_POLICY_HARNESS:
        return
    config = (policy_ref.config or "").strip()
    if not config:
        raise RequestParseError(
            f"{operation}: policy_ref.config is required; the platform does not default luna_med"
        )


def _task_seed_id(body: dict[str, Any], *, operation: str) -> Optional[str]:
    if "task" not in body or body["task"] is None:
        return None
    task = body["task"]
    if not isinstance(task, dict):
        raise RequestParseError(f"{operation}: task must be an object, got {type(task).__name__}")
    if "seed" not in task or task["seed"] is None:
        return None
    seed = task["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, float, str)):
        raise RequestParseError(
            f"{operation}: task.seed must be a string or number, got {type(seed).__name__}"
        )
    return f"seed:{seed}"


def _optional_str(
    body: dict[str, Any],
    key: str,
    *,
    operation: str,
    field: Optional[str] = None,
) -> Optional[str]:
    label = field or key
    if key not in body or body[key] is None:
        return None
    value = body[key]
    if not isinstance(value, str):
        raise RequestParseError(f"{operation}: {label} must be a string, got {type(value).__name__}")
    return value


def _optional_bool(
    body: dict[str, Any],
    key: str,
    *,
    operation: str,
    field: Optional[str] = None,
) -> Optional[bool]:
    label = field or key
    if key not in body or body[key] is None:
        return None
    value = body[key]
    if not isinstance(value, bool):
        raise RequestParseError(
            f"{operation}: {label} must be a boolean, got {type(value).__name__}"
        )
    return value


def _optional_int(body: dict[str, Any], key: str, *, operation: str) -> Optional[int]:
    if key not in body or body[key] is None:
        return None
    value = body[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestParseError(
            f"{operation}: {key} must be an integer, got {type(value).__name__}"
        )
    return value


def _optional_object(
    body: dict[str, Any],
    key: str,
    *,
    operation: str,
) -> Optional[dict[str, Any]]:
    if key not in body or body[key] is None:
        return None
    value = body[key]
    if not isinstance(value, dict):
        raise RequestParseError(
            f"{operation}: {key} must be an object, got {type(value).__name__}"
        )
    return value
