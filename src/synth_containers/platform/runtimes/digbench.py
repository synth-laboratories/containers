"""dig.bench content. Mock dungeon for PR CI; live relay for A8. No frames."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any

from ...event_log import RolloutEventLog
from ..state import CompatPlatform, RolloutPin

LIVE_ENVIRONMENT = "env:digbench_relay"
MOCK_ENVIRONMENT = "env:digbench_mock"
DEFAULT_LIVE_URL = "https://api.digbench.ai/api/agent"
WORLD_PREFIX = "world:digbench:"
DEFAULT_GAME = "P-1"
CHECKPOINT_KINDS = frozenset({"true_checkpoint", "restore", "fork", "checkpoint", "state"})

_EMPTY_USAGE = {
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
}


class DigbenchRelayError(RuntimeError):
    """Fail-closed live relay error. ``error_type`` is the public log field."""

    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


class DigbenchHttpError(DigbenchRelayError):
    def __init__(self, code: int) -> None:
        super().__init__("HTTPError")
        self.code = code


class DigbenchRuntime:
    def simulate(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        if platform.spec.environment_ref == LIVE_ENVIRONMENT:
            self._live(platform, pin, log)
            return
        self._mock(platform, pin, log)

    def _mock(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        platform.start_session_calls += 1
        game_id = _game_from_world_ref(pin.world_ref)
        pin.world_ref = f"{WORLD_PREFIX}{game_id}"
        raw_state = {
            "observation": "A locked door. Legal: inspect, wait.",
            "level": 1,
            "lives": 3,
            "steps_remaining": 20,
            "actions": ["inspect", "wait"],
            "status": "running",
        }
        log.append("start_session", {"game": game_id, "session_id": pin.rollout_id})
        log.append("session.opened", {"game": game_id, "session_id": pin.rollout_id, "raw": raw_state})
        log.append("observation", {"text": raw_state["observation"], "raw": raw_state})
        log.append("legal_actions", {"actions": raw_state["actions"]})
        log.append("stats", {"level": 1, "lives": 3, "steps_remaining": 20})
        log.append("invalid_action", {"action": "fly", "reason": "not_legal"})
        if _agentic_mcp(platform, pin):
            log.append(
                "span.mcp.opened",
                {
                    "tool": "step",
                    "server": "digbench-mcp",
                    "evidence_class": "simulated",
                },
            )
            log.append(
                "action",
                {
                    "action": "inspect",
                    "step_index": 1,
                    "action_authority": "harness_stub",
                    "harness": pin.policy_ref.get("harness"),
                },
            )
            log.append(
                "span.mcp.closed",
                {
                    "tool": "step",
                    "evidence_class": "simulated",
                },
            )
        else:
            log.append(
                "action",
                {
                    "action": "inspect",
                    "step_index": 1,
                    "action_authority": "harness_stub",
                    "harness": pin.policy_ref.get("harness"),
                },
            )
        outcome = pin.outcome or "completed"
        log.append("status", {"status": outcome})
        pin.status = outcome
        pin.terminal = True
        _apply_env_status_reward(pin, outcome)
        pin.usage = dict(_EMPTY_USAGE)
        log.mark_closed()

    def _live(self, platform: CompatPlatform, pin: RolloutPin, log: RolloutEventLog) -> None:
        token = (os.environ.get("DIGBENCH_API_TOKEN") or "").strip()
        if not token:
            _fail_closed(pin, log, "credential_missing")
            return
        base_url = (os.environ.get("SYNTH_DIGBENCH_URL") or DEFAULT_LIVE_URL).rstrip("/")
        try:
            catalog = _request(base_url, token, "GET", "/games")
            game_id = _freeze_game(_game_ids(catalog), _game_from_world_ref(pin.world_ref))
            pin.world_ref = f"{WORLD_PREFIX}{game_id}"
            created = _request(
                base_url,
                token,
                "POST",
                "/sessions",
                {
                    "game": game_id,
                    "model_name": _model_name(platform, pin),
                },
            )
            platform.start_session_calls += 1
            session_id = str(created.get("session_id") or "").strip()
            if not session_id:
                raise DigbenchRelayError("digbench_session_omitted")
            log.append("start_session", {"game": game_id, "session_id": session_id})
            state = _require_state(created)
            step_index = _require_step_index(created)
            log.append(
                "session.opened",
                {
                    "game": game_id,
                    "session_id": session_id,
                    "raw": _public_payload(created, token),
                },
            )
            log.append(
                "observation",
                {
                    "text": state["observation"],
                    "raw": _public_payload(state, token),
                },
            )
            legal = [str(item) for item in state["actions"]]
            log.append("legal_actions", {"actions": list(legal)})
            log.append(
                "stats",
                {
                    "level": state.get("level"),
                    "lives": state.get("lives_left"),
                    "steps_remaining": state.get("steps_remaining"),
                },
            )
            if not legal:
                raise DigbenchRelayError("digbench_actions_empty")
            illegal = _illegal_probe(legal)
            next_index = step_index + 1
            probed = _live_step(base_url, token, session_id, illegal, next_index)
            if probed.get("invalid_action") is True:
                log.append("invalid_action", {"action": illegal, "reason": "not_legal"})
            else:
                raise DigbenchRelayError("digbench_illegal_accepted")
            action = legal[0]
            agentic = _agentic_mcp(platform, pin)
            if agentic:
                log.append(
                    "span.mcp.opened",
                    {
                        "tool": "step",
                        "server": "digbench-mcp",
                        "evidence_class": "simulated",
                    },
                )
            try:
                stepped = _live_step(base_url, token, session_id, action, next_index)
                if stepped.get("invalid_action") is True:
                    raise DigbenchRelayError("digbench_legal_rejected")
                log.append(
                    "action",
                    {
                        "action": action,
                        "step_index": _require_step_index(stepped),
                        "action_authority": "relay_stub",
                        "harness": pin.policy_ref.get("harness"),
                    },
                )
            finally:
                if agentic:
                    log.append(
                        "span.mcp.closed",
                        {
                            "tool": "step",
                            "evidence_class": "simulated",
                        },
                    )
            # get_session is reconnect, not a checkpoint restore. Do not emit
            # restore / true_checkpoint / fork / state kinds.
            view = _live_get_session(base_url, token, session_id)
            outcome = _map_status(_require_state(view), done=view.get("done"))
            log.append("status", {"status": outcome})
            pin.status = outcome
            pin.terminal = outcome in {"completed", "game_over"}
            _apply_env_status_reward(pin, outcome)
            pin.usage = dict(_EMPTY_USAGE)
            log.mark_closed()
        except Exception as exc:
            error_type = getattr(exc, "error_type", None) or type(exc).__name__
            _fail_closed(
                pin,
                log,
                "digbench_relay_error",
                error_type=_scrub(str(error_type), token),
            )


def _agentic_mcp(platform: CompatPlatform, pin: RolloutPin) -> bool:
    if str(pin.policy_ref.get("harness") or "") != "codex":
        return False
    if platform.spec.mcp_bind == "unused":
        return False
    config_id = str(pin.policy_ref.get("config") or "")
    policy = platform.policy_configs.get(config_id)
    bind = None
    if policy is not None:
        bind = policy.config.get("mcp_bind")
    if bind in {None, "", "unused"}:
        bind = platform.spec.mcp_bind
    return bind not in {None, "", "unused"}


def _model_name(platform: CompatPlatform, pin: RolloutPin) -> str:
    config_id = str(pin.policy_ref.get("config") or "")
    policy = platform.policy_configs.get(config_id)
    if policy is not None:
        model = str(policy.config.get("model") or "").strip()
        if model:
            return model
    harness = str(pin.policy_ref.get("harness") or "").strip()
    return harness or "synth-containers"


def _game_from_world_ref(world_ref: str | None) -> str:
    raw = (world_ref or "").strip()
    if raw.startswith(WORLD_PREFIX):
        game = raw[len(WORLD_PREFIX) :].strip()
        if game:
            return game
    return DEFAULT_GAME


def _game_ids(payload: dict[str, Any]) -> list[str]:
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise DigbenchRelayError("digbench_games_omitted")
    ids: list[str] = []
    for item in games:
        if not isinstance(item, str) or not item.strip():
            raise DigbenchRelayError("digbench_games_shape")
        ids.append(item.strip())
    return ids


def _freeze_game(ids: list[str], requested: str) -> str:
    if requested in ids:
        return requested
    if requested == DEFAULT_GAME:
        return ids[0]
    raise DigbenchRelayError("digbench_game_missing")


def _require_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("state")
    if not isinstance(state, dict):
        raise DigbenchRelayError("digbench_state_omitted")
    if not isinstance(state.get("observation"), str):
        raise DigbenchRelayError("digbench_observation_omitted")
    if not isinstance(state.get("actions"), list):
        raise DigbenchRelayError("digbench_actions_omitted")
    return state


def _require_step_index(payload: dict[str, Any]) -> int:
    step_index = payload.get("step_index")
    if isinstance(step_index, bool) or not isinstance(step_index, int):
        raise DigbenchRelayError("digbench_step_index_omitted")
    return step_index


def _map_status(state: dict[str, Any], *, done: Any = None) -> str:
    status = state.get("status")
    if status == "in_progress":
        return "running"
    if status in {"running", "completed", "game_over"}:
        return str(status)
    if status is None and done is False:
        return "running"
    raise DigbenchRelayError("digbench_status_shape")


def _illegal_probe(legal: list[str]) -> str:
    for candidate in ("fly", "__not_legal__"):
        if candidate not in legal:
            return candidate
    return f"not_legal:{legal[0]}"


def _apply_env_status_reward(pin: RolloutPin, outcome: str) -> None:
    if outcome == "completed":
        pin.native_script_reward = 1.0
    elif outcome == "game_over":
        pin.native_script_reward = 0.0
    else:
        pin.native_script_reward = None


def _fail_closed(
    pin: RolloutPin,
    log: RolloutEventLog,
    reason: str,
    *,
    error_type: str | None = None,
) -> None:
    pin.status = "failed"
    pin.terminal = True
    pin.native_script_reward = None
    pin.usage = dict(_EMPTY_USAGE)
    payload: dict[str, Any] = {"status": "failed", "reason": reason}
    if error_type is not None:
        payload["error_type"] = error_type
    log.append("status", payload)
    log.mark_closed()


def _live_step(
    base_url: str,
    token: str,
    session_id: str,
    action: str,
    step_index: int,
) -> dict[str, Any]:
    try:
        payload = _request(
            base_url,
            token,
            "POST",
            f"/sessions/{session_id}/step",
            {"action": action, "step_index": step_index},
        )
    except DigbenchHttpError as exc:
        if exc.code == 400:
            return {"invalid_action": True}
        raise
    if not isinstance(payload, dict):
        raise DigbenchRelayError("digbench_step_shape")
    return payload


def _live_get_session(base_url: str, token: str, session_id: str) -> dict[str, Any]:
    payload = _request(base_url, token, "GET", f"/sessions/{session_id}")
    if not isinstance(payload, dict):
        raise DigbenchRelayError("digbench_session_shape")
    return payload


def _request(
    base_url: str,
    token: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=encoded,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        _read_and_discard(exc, token)
        raise DigbenchHttpError(exc.code) from None
    except urllib.error.URLError:
        raise DigbenchRelayError("URLError") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DigbenchRelayError("digbench_non_json") from exc
    if not isinstance(payload, dict):
        raise DigbenchRelayError("digbench_non_object")
    return _public_payload(payload, token)


def _read_and_discard(exc: urllib.error.HTTPError, token: str) -> None:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return
    _scrub(body, token)


def _public_payload(payload: dict[str, Any], token: str) -> dict[str, Any]:
    return _scrub_value(deepcopy(payload), token)


def _scrub_value(value: Any, token: str) -> Any:
    if isinstance(value, str):
        return _scrub(value, token)
    if isinstance(value, list):
        return [_scrub_value(item, token) for item in value]
    if isinstance(value, dict):
        return {str(key): _scrub_value(item, token) for key, item in value.items()}
    return value


def _scrub(text: str, token: str) -> str:
    cleaned = text.replace(token, "<redacted>") if token else text
    cleaned = re.sub(r"(?i)Bearer\s+\S+", "<redacted>", cleaned)
    cleaned = cleaned.replace("DIGBENCH_API_TOKEN", "<redacted>")
    return cleaned
