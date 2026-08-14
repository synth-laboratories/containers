"""Rogue EnvironmentService adapter over the rust-gold HTTP contract."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Any

from .craftax_world import StepResult


class GoldRogueWorld:
    """Expose Rogue's reset/step/checkpoint/restore as the common episode world."""

    def __init__(self, *, max_steps: int, base_url: str | None = None) -> None:
        if max_steps <= 0:
            raise ValueError("Rogue gold max_steps must be positive")
        self.max_steps = int(max_steps)
        self.base_url = (
            base_url or os.environ.get("SYNTH_ROGUE_URL", "http://127.0.0.1:8101")
        ).rstrip("/")
        self.rollout_id: str | None = None
        self.previous_total_reward = 0.0
        self.previous_graded_reward = 0.0
        self._native_digests: list[str] = []

    def reset(self, seed: int, *, max_steps: int | None = None) -> StepResult:
        self.max_steps = int(max_steps or self.max_steps)
        payload = self._request(
            "POST",
            "/rollouts",
            {
                "seed": int(seed),
                "task": {
                    "task_id": "synth_containers_rogue_react",
                    "seed": int(seed),
                    "rules": {
                        "base": "modern_rogue_core",
                        "overrides": {"max_steps": self.max_steps},
                    },
                    "objective": "descend",
                },
            },
        )
        self.rollout_id = str(payload.get("rollout_id") or "")
        if not self.rollout_id:
            raise RuntimeError("Rogue gold reset omitted rollout_id")
        self.previous_total_reward = 0.0
        self.previous_graded_reward = 0.0
        self._native_digests = []
        return self._result(payload)

    def step(self, action: str) -> StepResult:
        if self.rollout_id is None:
            raise RuntimeError("Rogue gold step before reset")
        return self._result(
            self._request("POST", f"/rollouts/{self.rollout_id}/step", {"action": action})
        )

    def checkpoint(self) -> dict[str, Any]:
        if self.rollout_id is None:
            raise RuntimeError("Rogue gold checkpoint before reset")
        return self._request("POST", f"/rollouts/{self.rollout_id}/checkpoint", {})

    def restore(self, blob: str) -> StepResult:
        if self.rollout_id is None:
            raise RuntimeError("Rogue gold restore before reset")
        payload = self._request("POST", f"/rollouts/{self.rollout_id}/restore", {"blob": blob})
        readout = payload.get("readout")
        if not isinstance(readout, dict):
            raise RuntimeError("Rogue gold restore omitted readout")
        return self._result({"readout": readout})

    def drain_native_events(self) -> list[dict[str, Any]]:
        if self.rollout_id is None:
            return []
        payload = self._request("GET", f"/rollouts/{self.rollout_id}/event_log")
        rows = payload.get("events")
        if not isinstance(rows, list):
            raise RuntimeError("Rogue gold event_log omitted events")
        if not all(isinstance(row, dict) for row in rows):
            raise RuntimeError("Rogue gold event_log contains a non-object event")
        digests = [self._digest(row) for row in rows]
        if digests[: len(self._native_digests)] != self._native_digests:
            raise RuntimeError("Rogue gold event_log prefix mutated or shrank")
        new_rows = rows[len(self._native_digests) :]
        self._native_digests = digests
        return list(new_rows)

    def _result(self, payload: dict[str, Any]) -> StepResult:
        readout = payload.get("readout") if isinstance(payload.get("readout"), dict) else payload
        public = readout.get("public") if isinstance(readout.get("public"), dict) else {}
        private = readout.get("private") if isinstance(readout.get("private"), dict) else {}
        progress_metrics = (
            readout.get("progress_metrics")
            if isinstance(readout.get("progress_metrics"), dict)
            else {}
        )
        ascii_map = str(readout.get("ascii") or public.get("ascii") or public.get("grid") or "")
        valid_actions = [
            str(item) for item in readout.get("valid_actions") or public.get("valid_actions") or []
        ]
        graded_total = progress_metrics.get(
            "synth_shaped_reward", private.get("synth_shaped_reward")
        )
        binary_total = payload.get("reward", private.get("total_reward"))
        total = graded_total if isinstance(graded_total, (int, float)) else binary_total
        reward = None
        if isinstance(total, (int, float)):
            reward = float(total) - self.previous_graded_reward
            self.previous_graded_reward = float(total)
        if isinstance(binary_total, (int, float)):
            self.previous_total_reward = float(binary_total)
        steps = int(
            private.get("step_index") or private.get("turn") or payload.get("nev_cursor") or 0
        )
        done = bool(
            payload.get("terminated")
            or payload.get("truncated")
            or private.get("terminated")
            or private.get("truncated")
        )
        observation = {
            "observation_text": readout.get("observation_text") or ascii_map,
            "ascii": ascii_map,
            "valid_actions": valid_actions,
            "public": public,
            "private": private,
            "progress_metrics": progress_metrics,
            "env_steps": steps,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            valid_actions=valid_actions,
            ascii_map=ascii_map,
            frame_digest=self._digest(
                {"rollout_id": self.rollout_id, "step": steps, "ascii": ascii_map}
            ),
            env_steps=steps,
        )

    @staticmethod
    def _digest(value: Any) -> str:
        blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=encoded,
            headers={
                "accept": "application/json",
                **({"content-type": "application/json"} if encoded else {}),
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Rogue gold unreachable at {self.base_url}{path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Rogue gold returned non-object JSON")
        return payload
