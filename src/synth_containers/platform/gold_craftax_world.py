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
import urllib.error
import urllib.request
from typing import Any

from .craftax_taxonomy import GOLD_URL_CONFIG_KEY
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


class GoldConnectionError(RuntimeError):
    """Gold HTTP failed. The attempted URL and the pinned config key travel with it."""

    def __init__(self, *, attempted_url: str, config_key: str, cause: BaseException) -> None:
        self.attempted_url = attempted_url
        self.config_key = config_key
        super().__init__(
            f"Craftax gold unreachable at {attempted_url} (config_key={config_key}): {cause}"
        )


class GoldHTTPError(GoldConnectionError):
    """The pinned gold origin answered, but rejected the gold-world contract."""

    def __init__(
        self,
        *,
        attempted_url: str,
        config_key: str,
        status_code: int,
        cause: BaseException,
    ) -> None:
        super().__init__(
            attempted_url=attempted_url,
            config_key=config_key,
            cause=cause,
        )
        self.status_code = int(status_code)


class GoldCraftaxWorld:
    """EnvironmentService: reset/step/frames/NEV drain against rust gold HTTP."""

    def __init__(
        self,
        *,
        max_steps: int,
        base_url: str,
        require_frames: bool = True,
        config_key: str = GOLD_URL_CONFIG_KEY,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("Craftax gold max_steps must be a positive pin, not a silent default")
        pinned = str(base_url or "").strip()
        if not pinned:
            raise ValueError(
                f"Craftax gold address must be pinned on {config_key}; it is not an env var"
            )
        self.max_steps = int(max_steps)
        self.base_url = pinned.rstrip("/")
        self.config_key = config_key
        self.require_frames = require_frames
        self.rollout_id: str | None = None
        self.previous_total_reward = 0.0
        self._native_digests: list[str] = []
        self._idempotency_key: str | None = None
        self._reset_payload: dict[str, Any] | None = None
        self._reset_identity: dict[str, Any] | None = None

    def reset(self, seed: int, *, max_steps: int | None = None) -> StepResult:
        self.max_steps = int(max_steps or self.max_steps)
        identity = {
            "task_id": "synth_containers_craftax_react",
            "seed": int(seed),
            "max_steps": self.max_steps,
        }
        self._idempotency_key = "sha256:" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            self._reset_payload is not None
            and self.rollout_id
            and self._reset_identity == identity
        ):
            return self._result(self._reset_payload)
        self._reset_identity = identity
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
                "idempotency_key": self._idempotency_key,
            },
        )
        self.rollout_id = str(payload.get("rollout_id") or "")
        if not self.rollout_id:
            raise RuntimeError("Craftax gold reset omitted rollout_id")
        self.previous_total_reward = 0.0
        self._native_digests = []
        self._reset_payload = payload
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
        terminated = bool(payload.get("terminated") or public.get("terminated"))
        truncated = bool(payload.get("truncated") or public.get("truncated"))
        if done and not terminated and not truncated:
            truncated = True
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
            terminated=terminated,
            truncated=truncated,
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
        if method == "POST" and path == "/rollouts" and self._idempotency_key:
            headers["Idempotency-Key"] = self._idempotency_key
        attempted = f"{self.base_url}{path}"
        last_error: BaseException | None = None
        for _attempt in range(3):
            request = urllib.request.Request(
                attempted,
                data=encoded,
                headers=headers,
                method=method,
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RuntimeError("Craftax gold returned non-object JSON")
                return payload
            except urllib.error.HTTPError as exc:
                # A healthy process at the pinned address is not necessarily a
                # Craftax gold world. Keep provider traffic out of the diagnosis:
                # reset happens before the first policy call, so an HTTP status
                # here is an environment-contract failure, never a model failure.
                # Do not persist the response body; upstream services may echo
                # request material in error details.
                raise GoldHTTPError(
                    attempted_url=attempted,
                    config_key=self.config_key,
                    status_code=exc.code,
                    cause=exc,
                ) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                continue
        assert last_error is not None
        raise GoldConnectionError(
            attempted_url=attempted,
            config_key=self.config_key,
            cause=last_error,
        ) from last_error
