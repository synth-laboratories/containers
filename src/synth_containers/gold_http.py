"""Generic EnvironmentService over a baked GameBench-style gold HTTP engine.

The engine speaks its own wire (``/rollouts``, ``/rollouts/{id}/step``,
``/rollouts/{id}/event_log``, checkpoint/restore, optional PNG frames). This
module is the relay every rust-engine image shares: Craftax, Rogue, DungeonGrid.
It names no game. The per-game part is the **task payload builder** the image
supplies.

Relay invariants (unchanged from the Craftax original):

* The producer's ``nev_cursor`` never leaves this adapter — the relay slices the
  whole-log poll locally and fails closed if the prefix mutated or shrank.
* ``require_frames`` means a missing PNG is an error, not an ASCII fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .pid1 import probe_http_health

__all__ = [
    "ACTION_KEY",
    "GoldEventLogCorrupt",
    "GoldFrameMissing",
    "probe_gold_ready",
    "GoldHttpWorld",
    "PNG_MAGIC",
    "StepResult",
    "facade_health_for",
    "probe_gold_health",
]

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ACTION_KEY = "action"


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float | None
    done: bool
    valid_actions: list[str]
    ascii_map: str
    frame_digest: str
    env_steps: int
    frame_url: str | None = None
    frame_bytes: bytes | None = None


class GoldFrameMissing(RuntimeError):
    """``live_frames=native`` requires a PNG copy into the relay. Missing is not ASCII."""


class GoldEventLogCorrupt(RuntimeError):
    """Gold whole-log poll rewrote or dropped already-relayed NEV rows."""


def probe_gold_health(base_url: str, *, timeout: float = 1.0) -> dict[str, Any]:
    """GET gold ``/health``. Raises if the engine is down. Missing is never ok."""

    try:
        return probe_http_health(base_url, path="/health", timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — normalise to the adapter's error text
        raise RuntimeError(f"gold_unreachable:{base_url.rstrip('/')}/health") from exc


def probe_gold_ready(base_url: str, *, timeout: float = 5.0) -> None:
    """Verify the engine can actually open an episode, not merely that it answers.

    A stub sidecar that has not been attached to a real gold service answers
    ``/health`` with ``status: ok`` and refuses ``/reset`` with a 503. Liveness
    is not readiness: probing only ``/health`` lets such an image come up and
    report healthy, and every rollout against it then fails with a 500.

    The throwaway rollout this opens is the cheapest honest question to ask.
    """

    body = json.dumps({"seed": 0, "max_steps": 1}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/reset",
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"gold_not_ready:{exc.code}:{detail}") from exc
    except Exception as exc:  # noqa: BLE001 — normalise to the adapter's error text
        raise RuntimeError(f"gold_not_ready:{base_url.rstrip('/')}/reset") from exc


def facade_health_for(url_env: str, *, engine: str) -> Callable[[], dict[str, Any]]:
    """Build the ``TargetSpec.health_probe`` for an image whose engine is a child.

    Fail closed: an unreachable engine makes the container's ``/health`` 503, so
    ``synth-containers up`` refuses rather than handing back a half-up URL.
    """

    def _probe() -> dict[str, Any]:
        raw = os.environ.get(url_env, "").strip().rstrip("/")
        if not raw:
            return {"status": "unhealthy", "reason": f"{engine}_url_missing", "gold_ok": False}
        from urllib.parse import urlparse

        parsed = urlparse(raw)
        host = parsed.hostname or ""
        public = f"{host}:{parsed.port}" if parsed.port else host
        try:
            probe_gold_health(raw, timeout=1.0)
        except Exception:  # noqa: BLE001
            return {
                "status": "unhealthy",
                "reason": f"{engine}_unreachable",
                "gold_ok": False,
                "gold_url": public,
            }
        return {"gold_ok": True, "gold_url": public}

    return _probe


def _event_digest(event: dict[str, Any]) -> str:
    blob = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class GoldHttpWorld:
    """reset / step / checkpoint / restore / NEV drain against a gold HTTP engine.

    ``task_payload`` receives ``(seed, max_steps)`` and returns the engine's task
    object. That function is the only game-specific thing here.
    """

    max_steps: int
    task_payload: Callable[[int, int], dict[str, Any]]
    base_url: str | None = None
    url_env: str = "SYNTH_GOLD_URL"
    engine: str = "gold"
    require_frames: bool = False
    frame_path: str = "/rollouts/{rollout_id}/frames/{env_steps}.png"
    request_timeout_seconds: float = 60.0
    rollout_id: str | None = field(default=None, init=False)
    previous_total_reward: float = field(default=0.0, init=False)
    _native_digests: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"{self.engine} gold max_steps must be a positive pin")
        resolved = (self.base_url or os.environ.get(self.url_env) or "").strip().rstrip("/")
        if not resolved:
            raise RuntimeError(f"{self.engine}_gold_url_missing")
        self.base_url = resolved

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: int, *, max_steps: int | None = None) -> StepResult:
        self.max_steps = int(max_steps or self.max_steps)
        payload = self._request(
            "POST",
            "/rollouts",
            {
                "task": self.task_payload(int(seed), self.max_steps),
                "seed": int(seed),
                "telemetry": {"enabled": True},
            },
        )
        self.rollout_id = str(payload.get("rollout_id") or "")
        if not self.rollout_id:
            raise RuntimeError(f"{self.engine} gold reset omitted rollout_id")
        self.previous_total_reward = 0.0
        self._native_digests = []
        return self._result(payload)

    def step(self, action: str) -> StepResult:
        if self.rollout_id is None:
            raise RuntimeError(f"{self.engine} gold step before reset")
        payload = self._request(
            "POST", f"/rollouts/{self.rollout_id}/step", {ACTION_KEY: action}
        )
        return self._result(payload)

    def checkpoint(self) -> dict[str, Any]:
        if self.rollout_id is None:
            raise RuntimeError(f"{self.engine} gold checkpoint before reset")
        payload = self._request("POST", f"/rollouts/{self.rollout_id}/checkpoint", {})
        blob = payload.get("blob")
        if not isinstance(blob, str) or not blob:
            raise RuntimeError(f"{self.engine} gold checkpoint omitted blob")
        if not isinstance(payload.get("checkpoint_id"), str):
            raise RuntimeError(f"{self.engine} gold checkpoint omitted checkpoint_id")
        return payload

    def restore(self, blob: str) -> StepResult:
        if self.rollout_id is None:
            raise RuntimeError(f"{self.engine} gold restore before reset")
        if not blob:
            raise RuntimeError(f"{self.engine} gold restore requires checkpoint blob")
        payload = self._request("POST", f"/rollouts/{self.rollout_id}/restore", {"blob": blob})
        state = payload.get("state")
        if not isinstance(state, dict):
            raise RuntimeError(f"{self.engine} gold restore omitted state")
        return self._result(state)

    # -------------------------------------------------------------------- NEV

    def drain_native_events(self) -> list[dict[str, Any]]:
        """New NEV rows since the last drain. Gold has no ``since``; cursor is local."""

        if self.rollout_id is None:
            return []
        payload = self._request("GET", f"/rollouts/{self.rollout_id}/event_log")
        if "nev_cursor" in payload:
            payload = {key: value for key, value in payload.items() if key != "nev_cursor"}
        events = payload.get("events")
        if not isinstance(events, list):
            raise GoldEventLogCorrupt(f"{self.engine} gold event_log omitted events")
        if len(events) < len(self._native_digests):
            raise GoldEventLogCorrupt(f"{self.engine} gold event_log shrank")
        for index, digest in enumerate(self._native_digests):
            row = events[index]
            if not isinstance(row, dict) or _event_digest(row) != digest:
                raise GoldEventLogCorrupt(f"{self.engine} gold event_log prefix mutated")
        new_events: list[dict[str, Any]] = []
        for row in events[len(self._native_digests) :]:
            if not isinstance(row, dict):
                raise GoldEventLogCorrupt(f"{self.engine} gold NEV row is not an object")
            self._native_digests.append(_event_digest(row))
            new_events.append(row)
        return new_events

    # ---------------------------------------------------------------- internal

    def _result(self, payload: dict[str, Any]) -> StepResult:
        readout = payload.get("readout") if isinstance(payload.get("readout"), dict) else {}
        observation = (
            readout.get("observation") if isinstance(readout.get("observation"), dict) else {}
        )
        public = readout.get("public") if isinstance(readout.get("public"), dict) else {}
        private = readout.get("private") if isinstance(readout.get("private"), dict) else {}
        progress = payload.get("progress") if isinstance(payload.get("progress"), dict) else {}
        total_reward = private.get("total_reward")
        reward = private.get("reward_last")
        if not isinstance(reward, (int, float)) and isinstance(total_reward, (int, float)):
            reward = float(total_reward) - self.previous_total_reward
        if isinstance(total_reward, (int, float)):
            self.previous_total_reward = float(total_reward)
        if not isinstance(reward, (int, float)):
            reward = None
        ascii_map = str(readout.get("ascii") or "")
        env_steps = int(progress.get("env_steps") or private.get("step_index") or 0)
        valid_actions = [str(action) for action in readout.get("valid_actions") or []]
        enriched = {
            **observation,
            "observation_text": readout.get("observation_text"),
            "ascii": ascii_map,
            "valid_actions": valid_actions,
            "public": public,
            "private": private,
            "env_steps": env_steps,
        }
        digest = (
            str(readout.get("grid_hash") or "")
            or hashlib.sha256(
                f"{self.rollout_id}:{env_steps}:{ascii_map}".encode("utf-8")
            ).hexdigest()[:16]
        )
        done = bool(payload.get("terminated") or payload.get("truncated") or public.get("done"))
        frame_url: str | None = None
        frame_bytes: bytes | None = None
        if self.frame_path and self.rollout_id is not None:
            frame_url = self.base_url + self.frame_path.format(
                rollout_id=self.rollout_id, env_steps=env_steps
            )
            frame_bytes = self._request_frame(frame_url)
        return StepResult(
            observation=enriched,
            reward=reward,
            done=done,
            valid_actions=valid_actions,
            ascii_map=ascii_map,
            frame_digest=digest,
            env_steps=env_steps,
            frame_url=frame_url,
            frame_bytes=frame_bytes,
        )

    def _request_frame(self, url: str) -> bytes | None:
        """Copy the transient gold frame into the relay before the next step."""

        try:
            request = urllib.request.Request(url, headers={"Accept": "image/png"}, method="GET")
            with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
                payload = response.read()
        except Exception as exc:  # noqa: BLE001
            if self.require_frames:
                raise GoldFrameMissing(f"{self.engine} gold frame missing at {url}") from exc
            return None
        if payload.startswith(PNG_MAGIC):
            return payload
        if self.require_frames:
            raise GoldFrameMissing(f"{self.engine} gold frame at {url} is not a PNG")
        return None

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=encoded, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=self.request_timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{self.engine} gold unreachable at {self.base_url}{path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{self.engine} gold returned non-object JSON")
        return payload
