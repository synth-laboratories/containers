"""Craftax EnvironmentService over rust gold HTTP.

Gold speaks GameBench wire internally. The Containers stream is a sequence log:
NEV kinds are copied verbatim after each step; producer `nev_cursor` never
leaves this adapter. Gold has no `since` — the relay slices locally and fails
closed if the prefix mutates or shrinks.

See: workshop/docs/aug_12_update.md §2.3 GameBench rust HTTP.
     workshop/docs/container_compat.md (EnvironmentService façade over gold).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .craftax_world import StepResult

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class GoldWorldUnreachable(RuntimeError):
    """The world service could not be reached. Not a policy failure.

    Reported separately because an unreachable world dies in reset(), before
    the policy is ever called, and labelling that `policy_error` sent two
    diagnoses chasing the model instead of the address.
    """


class GoldFrameMissing(RuntimeError):
    """live_frames=native requires a PNG copy into the relay. Missing is not ASCII."""


class GoldEventLogCorrupt(RuntimeError):
    """Gold whole-log poll rewrote or dropped already-relayed NEV rows."""


def _event_digest(event: dict[str, Any]) -> str:
    blob = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class GoldCraftaxWorld:
    """EnvironmentService: reset/step/frames/NEV drain against rust gold HTTP."""

    def __init__(
        self,
        *,
        max_steps: int,
        base_url: str | None = None,
        require_frames: bool = True,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("Craftax gold max_steps must be a positive pin, not a silent default")
        self.max_steps = int(max_steps)
        # No default address. Pointing at a port nothing serves produced a
        # rollout that died in reset() before the policy was ever called, and
        # the failure surfaced downstream as `policy_error` with zero steps —
        # so an unconfigured world address read as "this checkpoint scores
        # zero" and cost several runs to trace. An unset address must say so.
        resolved = base_url or os.environ.get("SYNTH_CRAFTAX_URL")
        if not resolved or not resolved.strip():
            raise ValueError(
                "Craftax gold world requires an address: pass base_url or set "
                "SYNTH_CRAFTAX_URL. It is never defaulted, because a wrong "
                "address is indistinguishable from a failing policy downstream."
            )
        self.base_url = resolved.strip().rstrip("/")
        self.require_frames = require_frames
        self.rollout_id: str | None = None
        self.previous_total_reward = 0.0
        self._native_digests: list[str] = []

    def reset(self, seed: int, *, max_steps: int | None = None) -> StepResult:
        self.max_steps = int(max_steps or self.max_steps)
        payload = self._request(
            "POST",
            "/rollouts",
            {
                "task": {
                    "schema": "gamebench.task.craftax.v1",
                    "task_id": "synth_containers_craftax_react",
                    "scenario_id": f"seed-{int(seed)}",
                    "max_steps": self.max_steps,
                    "world": {
                        "use_default": "policy_dev_small",
                        "seed": int(seed),
                        "max_steps": self.max_steps,
                    },
                    "rules": {"base": "symbolic_no_homeostasis"},
                    "readouts": {"profile": "symbolic_compact"},
                },
                "seed": int(seed),
                "telemetry": {"enabled": True},
            },
        )
        self.rollout_id = str(payload.get("rollout_id") or "")
        if not self.rollout_id:
            raise RuntimeError("Craftax gold reset omitted rollout_id")
        self.previous_total_reward = 0.0
        self._native_digests = []
        return self._result(payload)

    def drain_native_events(self) -> list[dict[str, Any]]:
        """New NEV rows since last drain. Gold has no `since`; cursor is local."""
        if self.rollout_id is None:
            return []
        payload = self._request("GET", f"/rollouts/{self.rollout_id}/event_log")
        if "nev_cursor" in payload:
            # Producer cursor stays inside gold. Never copy it onto the relay stream.
            payload = {key: value for key, value in payload.items() if key != "nev_cursor"}
        events = payload.get("events")
        if not isinstance(events, list):
            raise GoldEventLogCorrupt("Craftax gold event_log omitted events")
        if len(events) < len(self._native_digests):
            raise GoldEventLogCorrupt("Craftax gold event_log shrank")
        for index, digest in enumerate(self._native_digests):
            row = events[index]
            if not isinstance(row, dict) or _event_digest(row) != digest:
                raise GoldEventLogCorrupt("Craftax gold event_log prefix mutated")
        new_events: list[dict[str, Any]] = []
        for row in events[len(self._native_digests) :]:
            if not isinstance(row, dict):
                raise GoldEventLogCorrupt("Craftax gold NEV row is not an object")
            self._native_digests.append(_event_digest(row))
            new_events.append(row)
        return new_events

    def step(self, action: str) -> StepResult:
        if self.rollout_id is None:
            raise RuntimeError("Craftax gold step before reset")
        payload = self._request(
            "POST",
            f"/rollouts/{self.rollout_id}/step",
            {"action": action},
        )
        return self._result(payload)

    def checkpoint(self) -> dict[str, Any]:
        if self.rollout_id is None:
            raise RuntimeError("Craftax gold checkpoint before reset")
        payload = self._request("POST", f"/rollouts/{self.rollout_id}/checkpoint", {})
        blob = payload.get("blob")
        if not isinstance(blob, str) or not blob:
            raise RuntimeError("Craftax gold checkpoint omitted blob")
        if not isinstance(payload.get("checkpoint_id"), str):
            raise RuntimeError("Craftax gold checkpoint omitted checkpoint_id")
        return payload

    def restore(self, blob: str) -> StepResult:
        if self.rollout_id is None:
            raise RuntimeError("Craftax gold restore before reset")
        if not blob:
            raise RuntimeError("Craftax gold restore requires checkpoint blob")
        payload = self._request(
            "POST", f"/rollouts/{self.rollout_id}/restore", {"blob": blob}
        )
        state = payload.get("state")
        if not isinstance(state, dict):
            raise RuntimeError("Craftax gold restore omitted state")
        return self._result(state)

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
        frame_url = (
            f"{self.base_url}/rollouts/{self.rollout_id}/frames/{env_steps}.png"
            if self.rollout_id is not None
            else None
        )
        frame_bytes = self._request_frame(frame_url) if frame_url is not None else None
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
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = response.read()
        except Exception as exc:
            if self.require_frames:
                raise GoldFrameMissing(f"Craftax gold frame missing at {url}") from exc
            return None
        if payload.startswith(PNG_MAGIC):
            return payload
        if self.require_frames:
            raise GoldFrameMissing(f"Craftax gold frame at {url} is not a PNG")
        return None

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=encoded,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise GoldWorldUnreachable(
                f"Craftax gold unreachable at {self.base_url}{path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Craftax gold returned non-object JSON")
        return payload
