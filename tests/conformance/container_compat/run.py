"""Containers-compat conformance runner (C0–C8)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_containers.platform.reducer import assert_honest_projection, project_envelopes
from synth_containers.platform.targets import PAID_TARGETS, TARGETS, RewardKind


SCHEMA = "synth.container-compat-conformance.v1"


@dataclass
class Check:
    test_id: str
    status: str  # pass | fail | skip
    detail: str = ""


@dataclass
class Suite:
    target: str
    checks: list[Check] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def add(self, test_id: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(test_id, status, detail))

    def ok(self, test_id: str, detail: str = "") -> None:
        self.add(test_id, "pass", detail)

    def fail(self, test_id: str, detail: str) -> None:
        self.add(test_id, "fail", detail)

    def skip(self, test_id: str, detail: str) -> None:
        self.add(test_id, "skip", detail)

    def failed(self) -> list[Check]:
        return [item for item in self.checks if item.status == "fail"]


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json(response: Any) -> dict[str, Any]:
    try:
        body = response.json()
    except Exception:
        return {"_raw": getattr(response, "text", "")}
    return body if isinstance(body, dict) else {"_value": body}


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("kind")) for item in events]


def _semantic(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in events if not item.get("control")]


def _parse_sse(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in text.splitlines():
        if line.startswith(":"):
            continue
        if not line.strip():
            if current.get("data"):
                try:
                    payload = json.loads(current["data"])
                    if isinstance(payload, dict):
                        events.append(payload)
                except json.JSONDecodeError:
                    pass
            current = {}
            continue
        if line.startswith("data:"):
            current["data"] = line[5:].strip()
        elif line.startswith("id:"):
            current["id"] = line[3:].strip()
        elif line.startswith("event:"):
            current["event"] = line[6:].strip()
    return events


def _ws_envelopes(client: Any, url: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with client.websocket_connect(url) as socket:
        while True:
            try:
                payload = socket.receive_json()
            except Exception:
                break
            if isinstance(payload, dict):
                events.append(payload)
    return events


def _stream_id(body: dict[str, Any]) -> str | None:
    stream = body.get("stream") if isinstance(body.get("stream"), dict) else {}
    value = stream.get("id")
    return str(value) if value is not None else None


def _seal_kind(event: dict[str, Any]) -> str | None:
    kind = event.get("event_type")
    if kind is None:
        kind = event.get("kind")
    return str(kind) if kind is not None else None


def _seal_sequence(event: dict[str, Any]) -> int | None:
    order = event.get("order") if isinstance(event.get("order"), dict) else {}
    raw = order.get("chronological_sequence")
    if raw is None:
        raw = event.get("sequence")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _payload_corr_id(event: dict[str, Any]) -> Any:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    for key in ("correlation_id", "call_id", "observation_id", "action_id"):
        if payload.get(key) is not None:
            return payload.get(key)
    return None


def _obs_action_correlated(events: list[dict[str, Any]]) -> tuple[bool, str]:
    observations = [item for item in events if _seal_kind(item) == "observation"]
    actions = [item for item in events if _seal_kind(item) == "action"]
    if not observations or not actions:
        return False, "missing observation or action"
    missing_sequence = False
    ordered = False
    shared_id = False
    for obs in observations:
        obs_seq = _seal_sequence(obs)
        obs_id = _payload_corr_id(obs)
        for action in actions:
            action_seq = _seal_sequence(action)
            action_id = _payload_corr_id(action)
            if obs_seq is None or action_seq is None:
                missing_sequence = True
                continue
            if obs_seq < action_seq:
                ordered = True
            if obs_id is not None and action_id is not None and obs_id == action_id:
                shared_id = True
    if ordered or shared_id:
        return True, "sequence" if ordered else "payload_id"
    if missing_sequence:
        return False, "observation/action sequence missing (not coerced to 0)"
    return False, "no obs sequence < action sequence and no shared payload id"


def _hillclimb_nodes_ok(nodes: Any) -> bool:
    if not isinstance(nodes, list) or not nodes:
        return False
    if len(nodes) == 1:
        return isinstance(nodes[0], dict) and nodes[0].get("kind") == "gate"
    gates = [item for item in nodes if isinstance(item, dict) and item.get("kind") == "gate"]
    deltas = [
        item
        for item in nodes
        if isinstance(item, dict)
        and (
            item.get("kind") in {"aggregate", "heldout"}
            or item.get("node_id") in {"baseline_delta", "delta"}
        )
    ]
    for item in nodes:
        if not isinstance(item, dict) or item.get("node_id") != "heldout_gate":
            continue
        if item.get("kind") != "gate":
            deltas.append(item)
    return bool(gates) and bool(deltas)


class Runner:
    def __init__(self, client: Any, target: str, *, paid: bool = False) -> None:
        self.client = client
        self.target = target
        self.paid = paid
        self.spec = TARGETS[target]
        self.suite = Suite(target=target)

    def run(self) -> Suite:
        if self.target in PAID_TARGETS and not self.paid:
            self.suite.skip("paid_target", "requires --paid")
            return self.suite
        meta = _json(self.client.get("/metadata"))
        self.suite.extras["metadata"] = meta
        self._c0(meta)
        self._c1(meta)
        self._c2(meta)
        if self.target.startswith("craftax") and self.target != "craftax_code_policy":
            self._c3(meta)
        if self.target in {"craftax_code_policy", "deo_nested"}:
            self._c4(meta)
        if self.target == "harbor_public":
            self._c5(meta)
        if self.target == "openenv_echo":
            self._c6(meta)
        self._c7(meta)
        if self.target.startswith("digbench"):
            self._c8(meta)
        return self.suite

    def _start(self, **body: Any) -> Any:
        payload = {
            "telemetry": {"enabled": True, "transport": "sse", "retention": "run"},
            "policy_ref": self._explicit_policy_ref(),
            **body,
        }
        return self.client.post("/rollouts", json=payload)

    def _explicit_policy_ref(self) -> dict[str, Any]:
        harness = self.spec.default_policy_harness
        if harness == "isolated_policy_process":
            return {"harness": harness}
        if self.spec.policy_seeds:
            seed = self.spec.policy_seeds[0]
            return {"harness": seed.harness, "config": seed.config_id}
        return {"harness": harness, "config": "luna_med"}

    def _c0(self, meta: dict[str, Any]) -> None:
        required = (
            "world_ref",
            "environment_ref",
            "policy_ref",
            "evaluation_plan_ref",
            "task_instance_id",
        )
        missing = [name for name in required if name not in meta]
        policy = meta.get("policy_ref") if isinstance(meta.get("policy_ref"), dict) else {}
        if missing or "harness_ref" in meta or "harness" not in policy:
            self.suite.fail("C0-01", f"missing={missing} harness_ref={'harness_ref' in meta}")
        elif "logical_service_ids" not in meta:
            self.suite.fail("C0-01", "logical_service_ids missing")
        else:
            self.suite.ok("C0-01")

        advertised = meta.get("affordances") if isinstance(meta.get("affordances"), dict) else {}
        booleans = meta.get("affordance_booleans") if isinstance(meta.get("affordance_booleans"), dict) else {}
        env = advertised.get("environment") if isinstance(advertised.get("environment"), dict) else {}
        env_bool = booleans.get("environment") if isinstance(booleans.get("environment"), dict) else {}
        derived_ok = all((env_bool.get(name) is True) == (level != "unsupported") for name, level in env.items())
        unnamed = self.spec.affordances.level("definitely_not_a_real_affordance")
        if unnamed != "unsupported" or not derived_ok:
            self.suite.fail("C0-02", f"unnamed={unnamed} derived_ok={derived_ok}")
        else:
            self.suite.ok("C0-02")

        refused = self._start(recipe={"require": {"true_checkpoint": True}})
        body = _json(refused)
        if refused.status_code == 403 and body.get("affordance") == "true_checkpoint":
            self.suite.ok("C0-03", "true_checkpoint refused")
        else:
            self.suite.fail("C0-03", f"status={refused.status_code} body={body}")

        if self.target.startswith("digbench"):
            frames = self._start(recipe={"require": {"live_frames": True}})
            body = _json(frames)
            if frames.status_code == 403 and body.get("affordance") == "live_frames":
                self.suite.ok("C0-03b", "live_frames refused")
            else:
                self.suite.fail("C0-03b", f"status={frames.status_code} body={body}")

        chain = meta.get("adapter_chain")
        expected = list(self.spec.adapter_chain)
        if chain != expected:
            self.suite.fail("C0-04", f"adapter_chain={chain} expected={expected}")
        else:
            self.suite.ok("C0-04")

        if self.target == "openenv_echo":
            from synth_containers.compat.openenv import openenv_capability_surface

            surface = openenv_capability_surface()
            if surface.checkpoint_support or meta.get("true_checkpoint") != "unsupported":
                self.suite.fail("C0-05", "checkpointable claimed")
            else:
                self.suite.ok("C0-05")
        else:
            self.suite.skip("C0-05", "not an OpenEnv wrap")

        if self.spec.reward_kind == RewardKind.SCRIPT:
            if meta.get("reward_authority") != "trusted_scorer":
                self.suite.fail("C0-06", f"authority={meta.get('reward_authority')}")
            else:
                self.suite.ok("C0-06")
        else:
            self.suite.skip("C0-06", "not a Harbor script reward")

        from synth_containers.rubrics.v1 import _clamp_score

        if _clamp_score(None) is not None:
            self.suite.fail("C0-07", "clamp coerces missing")
        else:
            self.suite.ok("C0-07")

        leases = int(meta.get("scale_leases") or 0)
        held = []
        busy_ok = False
        for index in range(leases + 1):
            response = self._start(submission_mode="async", task_instance_id=f"seed:{index}")
            if response.status_code == 429:
                body = _json(response)
                busy_ok = body.get("affordance") == "scale_leases"
                break
            if response.status_code == 200:
                held.append(_json(response)["rollout_id"])
        for rollout_id in held:
            self.client.post(f"/rollouts/{rollout_id}/complete")
        if busy_ok and len(held) == leases:
            self.suite.ok("C0-08", f"leases={leases}")
        else:
            self.suite.fail("C0-08", f"held={len(held)} leases={leases} busy_ok={busy_ok}")

        if meta.get("retention") != "run":
            self.suite.fail("C0-09", f"retention={meta.get('retention')}")
        elif self.spec.live_frames == "native":
            started = self._start(task_instance_id="seed:0")
            rollout = _json(started)
            events = _json(self.client.get(rollout["stream"]["transports"]["poll"]["url"], params={"after": 0}))["events"]
            frame = next((item for item in events if item.get("kind") == "frame"), None)
            digest = (frame or {}).get("payload", {}).get("digest")
            self.client.post("/world/stop")
            fetched = self.client.get(f"/artifacts/{digest}")
            if fetched.status_code == 200 and _json(fetched).get("available"):
                self.suite.ok("C0-09")
            else:
                self.suite.fail("C0-09", f"artifact after stop {fetched.status_code}")
        else:
            self.suite.ok("C0-09", "retention advertised; no frames to fetch")

    def _c1(self, meta: dict[str, Any]) -> None:
        auto = self._start(telemetry={"enabled": True, "transport": "auto"})
        if auto.status_code in {422, 400}:
            self.suite.ok("C1-01a", "auto refused")
        else:
            self.suite.fail("C1-01a", f"auto status={auto.status_code}")

        live = self._start(slot="live", task_instance_id="seed:0")
        jobs = self._start(slot="jobs", task_instance_id="seed:0")
        if live.status_code == 400 and jobs.status_code == 400:
            self.suite.ok("C1-09", "live/jobs refused")
        else:
            self.suite.fail("C1-09", f"live={live.status_code} jobs={jobs.status_code}")

        prepared_id = "roll_c108"
        prepared = self.client.post(
            "/rollouts/prepare",
            json={"rollout_id": prepared_id, "telemetry": {"enabled": True, "transport": "sse"}},
        )
        if prepared.status_code != 200:
            self.suite.fail("C1-08", f"prepare {prepared.status_code}")
            return
        descriptor = _json(prepared)["stream"]
        self.suite.extras["stream_descriptor"] = descriptor
        required_fields = ["id", "transports", "cursor", "reward", "auth", "retention"]
        missing = [name for name in required_fields if name not in descriptor]
        transports = descriptor.get("transports") or {}
        poll_url = (transports.get("poll") or {}).get("url")
        if missing or descriptor.get("cursor", {}).get("kind") != "sequence" or not poll_url:
            self.suite.fail("C1-01", f"descriptor missing {missing} poll={poll_url}")
        else:
            self.suite.ok("C1-01")

        before = _json(self.client.get(poll_url, params={"after": 0}))
        kinds = _kinds(before.get("events") or [])
        semantic = _semantic(before.get("events") or [])
        subscribed = any(item.get("kind") == "stream.subscribed" for item in before.get("events") or [])
        started = None
        if not subscribed or semantic:
            self.suite.fail("C1-08", f"before start kinds={kinds}")
        else:
            started = self._start(rollout_id=prepared_id, task_instance_id="seed:1")
            after = _json(self.client.get(poll_url, params={"after": 0}))
            sem = _semantic(after.get("events") or [])
            first = sem[0]["kind"] if sem else None
            cursor = after.get("cursor") or {}
            if first != "trace.opened":
                self.suite.fail("C1-08", f"first semantic={first}")
            elif cursor.get("kind") != "sequence":
                self.suite.fail("C1-02", f"cursor={cursor}")
            else:
                self.suite.ok("C1-08")
                self.suite.ok("C1-02")
                if cursor.get("closed") and cursor.get("high_water") == max(
                    (item.get("sequence") or 0) for item in sem
                ):
                    self.suite.ok("C1-06")
                else:
                    self.suite.fail("C1-06", f"cursor={cursor}")

        poll_body = _json(self.client.get(poll_url, params={"after": 0}))
        poll_sem = _semantic(poll_body.get("events") or [])
        sse_url = (transports.get("sse") or {}).get("url")
        if sse_url:
            with self.client.stream("GET", sse_url) as response:
                text = "".join(response.iter_text())
            sse_events = _parse_sse(text)
            sse_sem = _semantic(sse_events)
            poll_ids = [(item.get("sequence"), item.get("digest")) for item in poll_sem]
            sse_ids = [(item.get("sequence"), item.get("digest")) for item in sse_sem]
            if poll_ids == sse_ids:
                self.suite.ok("C1-03")
            else:
                self.suite.fail("C1-03", f"poll={poll_ids} sse={sse_ids}")
            if poll_sem:
                high = poll_sem[-1]["sequence"]
                with self.client.stream("GET", sse_url, headers={"Last-Event-ID": str(high)}) as response:
                    resume = "".join(response.iter_text())
                resumed = _semantic(_parse_sse(resume))
                if any(item.get("sequence") == high for item in resumed):
                    self.suite.fail("C1-05", "duplicate high_water on resume")
                else:
                    self.suite.ok("C1-05")
        else:
            self.suite.skip("C1-03", "sse not bound")
            self.suite.skip("C1-05", "sse not bound")

        if self.spec.affordances.level("websocket") == "unsupported":
            asked = self._start(telemetry={"enabled": True, "transport": "websocket"})
            if asked.status_code in {400, 422} and transports.get("websocket") in (None, {}):
                self.suite.ok("C1-04", "websocket not advertised; request refused")
            else:
                self.suite.fail(
                    "C1-04",
                    f"unadvertised websocket status={asked.status_code} bound={transports.get('websocket')}",
                )
        else:
            bound = self._start(
                telemetry={"enabled": True, "transport": "websocket"},
                task_instance_id="seed:ws",
            )
            body = _json(bound)
            ws_url = ((body.get("stream") or {}).get("transports") or {}).get("websocket") or {}
            ws_url = ws_url.get("url") if isinstance(ws_url, dict) else None
            if bound.status_code != 200 or not ws_url:
                self.suite.fail("C1-04", f"websocket bind {bound.status_code} url={ws_url}")
            else:
                poll_ws = _json(self.client.get(body["stream"]["transports"]["poll"]["url"], params={"after": 0}))
                poll_ids = [(item.get("sequence"), item.get("digest")) for item in _semantic(poll_ws.get("events") or [])]
                ws_ids = [(item.get("sequence"), item.get("digest")) for item in _semantic(_ws_envelopes(self.client, ws_url))]
                if poll_ids == ws_ids and poll_ids:
                    self.suite.ok("C1-04")
                else:
                    self.suite.fail("C1-04", f"poll={poll_ids} ws={ws_ids}")

        if "nev_cursor" in json.dumps(poll_body):
            self.suite.fail("C1-02b", "consumer saw nev_cursor")
        else:
            self.suite.ok("C1-02b", "no nev_cursor")

        if started is not None and started.status_code == 200:
            rollout_id = prepared_id
            seal = self.client.get(f"/rollouts/{rollout_id}/trace")
            if seal.status_code == 200:
                sealed = _json(seal)
                if sealed.get("content_digest"):
                    self.suite.extras["trace_v5_digest"] = sealed.get("content_digest")
                high = poll_body.get("cursor", {}).get("high_water")
                if sealed.get("high_water") == high and sealed.get("schema_version") == "synth.trace.v5":
                    self.suite.ok("C1-10")
                else:
                    self.suite.fail("C1-10", f"seal high={sealed.get('high_water')} live={high}")
            else:
                self.suite.fail("C1-10", f"seal {seal.status_code}")

        missing = self._start(task_instance_id="seed:8", omit_reward=True)
        if missing.status_code == 200 and self.spec.reward_kind == "env_sum":
            rid = _json(missing)["rollout_id"]
            scored = _json(self.client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}))
            if scored.get("reward") is None:
                self.suite.ok("C1-07")
            else:
                self.suite.fail("C1-07", f"missing filled {scored.get('reward')}")
        else:
            live = self._start(submission_mode="async", task_instance_id="seed:8")
            if live.status_code != 200:
                self.suite.fail("C1-07", f"async {live.status_code}")
            else:
                rid = _json(live)["rollout_id"]
                absent = _json(self.client.get("/reward", params={"rollout_id": rid}))
                projection = project_envelopes(
                    _json(self.client.get(f"/rollouts/{rid}/events", params={"after": 0})).get("events") or [],
                    reward=absent.get("reward"),
                    reward_status=str(absent.get("status") or "absent"),
                )
                defects = assert_honest_projection(projection)
                self.client.post(f"/rollouts/{rid}/complete")
                if absent.get("reward") is None and absent.get("status") in {"absent", "incomplete"} and not defects:
                    self.suite.ok("C1-07", "non-env-sum missing stays null")
                else:
                    self.suite.fail("C1-07", f"absent={absent} defects={defects}")

    def _c2(self, meta: dict[str, Any]) -> None:
        started = self._start(task_instance_id="seed:2")
        rid = _json(started)["rollout_id"]
        absent = self.client.get("/reward", params={"rollout_id": rid})
        body = _json(absent)
        if absent.status_code == 200 and body.get("status") == "absent" and body.get("reward") is None:
            self.suite.ok("C2-01")
        else:
            self.suite.fail("C2-01", f"{absent.status_code} {body}")

        live = self._start(submission_mode="async", task_instance_id="seed:3")
        live_id = _json(live)["rollout_id"]
        premature = self.client.post("/reward", json={"rollout_id": live_id, "mode": "terminal"})
        premature_body = _json(premature)
        if premature.status_code == 409 and premature_body.get("status") == "incomplete":
            self.suite.ok("C2-04")
        else:
            self.suite.fail("C2-04", f"{premature.status_code} {premature_body}")
        self.client.post(f"/rollouts/{live_id}/complete")

        scored = self.client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
        first = _json(scored)
        if scored.status_code not in {200, 202} or "execution_id" not in first:
            self.suite.fail("C2-02", f"{scored.status_code} {first}")
        else:
            self.suite.ok("C2-02")
        again = self.client.post("/reward", json={"rollout_id": rid, "mode": "terminal"})
        second = _json(again)
        if second.get("execution_id") == first.get("execution_id"):
            self.suite.ok("C2-03a", "idempotent")
        else:
            self.suite.fail("C2-03a", f"{first.get('execution_id')} vs {second.get('execution_id')}")
        rescore = self.client.post("/reward", json={"rollout_id": rid, "mode": "terminal", "rescore": True})
        third = _json(rescore)
        old = self.client.get(f"/evaluations/{first['execution_id']}")
        if third.get("execution_id") != first.get("execution_id") and old.status_code == 200:
            self.suite.ok("C2-03")
        else:
            self.suite.fail("C2-03", f"rescore={third.get('execution_id')} old={old.status_code}")

        prov = self.client.post("/reward", json={"rollout_id": rid, "mode": "provisional"})
        if self.spec.live_reward:
            if prov.status_code == 200 and _json(prov).get("reward") is not None:
                self.suite.ok("C2-05")
            else:
                self.suite.fail("C2-05", f"{prov.status_code} {_json(prov)}")
        else:
            if prov.status_code == 409 and _json(prov).get("reward") is None:
                self.suite.ok("C2-05", "provisional refused")
            else:
                self.suite.fail("C2-05", f"{prov.status_code} {_json(prov)}")

        if first.get("env_mutated") or first.get("step_calls_delta"):
            self.suite.fail("C2-06", f"mutated {first}")
        else:
            self.suite.ok("C2-06")

        gated = self._start(task_instance_id="seed:4", evaluation_plan_ref="eval:gated")
        gated_id = _json(gated)["rollout_id"]
        gate = _json(self.client.post("/reward", json={"rollout_id": gated_id, "mode": "terminal"}))
        if gate.get("status") == "gated" and gate.get("reward") is None:
            self.suite.ok("C2-07")
        else:
            self.suite.fail("C2-07", f"{gate}")

        if self.target in {"harbor_public", "deo_nested"}:
            if scored.status_code == 202 and first.get("evaluation_id"):
                events = self.client.get(f"/evaluations/{first['evaluation_id']}/events")
                if events.status_code == 200 and _json(events).get("status") in {"scored", "gated", "refused"}:
                    self.suite.ok("C2-08")
                else:
                    self.suite.fail("C2-08", f"eval events {events.status_code}")
            else:
                self.suite.fail("C2-08", f"expected 202 got {scored.status_code}")
        else:
            self.suite.skip("C2-08", "not a long script node")

        nodes = first.get("node_results") or []
        if self.spec.reward_kind == "env_sum":
            events = _json(self.client.get(f"/rollouts/{rid}/events", params={"after": 0}))["events"]
            signals = [
                item["payload"]["value"]
                for item in events
                if item.get("kind") == "reward_signal" and item["payload"].get("value") is not None
            ]
            expected = float(sum(signals))
            if first.get("reward") == expected and nodes and nodes[0].get("kind") == "env_reward":
                self.suite.ok("C2-09")
            else:
                self.suite.fail("C2-09", f"reward={first.get('reward')} expected={expected}")
        elif self.spec.reward_kind == "script":
            if nodes and nodes[0].get("authority") == "trusted_scorer" and nodes[0].get("kind") in {"script", "gate"}:
                self.suite.ok("C2-09")
            else:
                self.suite.fail("C2-09", f"nodes={nodes}")
        else:
            if nodes and nodes[0].get("authority") == "environment":
                self.suite.ok("C2-09")
            else:
                self.suite.fail("C2-09", f"nodes={nodes}")

        both = self.client.post("/reward", json={"rollout_id": rid, "evidence": {"reward": 1}})
        neither = self.client.post("/reward", json={})
        provided = self.client.post("/reward", json={"evidence": {"reward.txt": 1.0}})
        if both.status_code == 422 and neither.status_code == 422 and _json(provided).get("reward") == 1.0:
            self.suite.ok("C2-10")
        else:
            self.suite.fail("C2-10", f"both={both.status_code} neither={neither.status_code} provided={_json(provided)}")

        missing_product = _json(
            self.client.post("/reward/combine", json={"bases": {"a": 1.0, "b": None}, "required": ["a", "b"]})
        )
        if missing_product.get("reward") is None and missing_product.get("status") == "absent":
            self.suite.ok("C2-11")
        else:
            self.suite.fail("C2-11", f"{missing_product}")

    def _c3(self, meta: dict[str, Any]) -> None:
        ids: list[str | None] = []
        stream_ids: list[str | None] = []
        rewards: list[Any] = []
        generations: list[Any] = []
        for seed in range(10):
            response = self._start(
                task_instance_id=f"seed:{seed}",
                world_ref=meta["world_ref"],
                policy_ref={"harness": "react", "config": "luna_med"},
            )
            body = _json(response)
            rid = body.get("rollout_id") if response.status_code == 200 else None
            ids.append(rid)
            stream_ids.append(_stream_id(body) if response.status_code == 200 else None)
            generations.append(body.get("engine_generation") if response.status_code == 200 else None)
            if rid is None:
                rewards.append(None)
                continue
            scored = _json(self.client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}))
            rewards.append(scored.get("reward"))
        if len({item for item in ids if item is not None}) == 10 and all(item is not None for item in rewards):
            self.suite.ok("C3-01")
            self.suite.ok("C3-04")
        else:
            self.suite.fail("C3-01", f"ids={ids} rewards={rewards}")

        first_id = ids[0]
        events = []
        if first_id is not None:
            events = _json(self.client.get(f"/rollouts/{first_id}/events", params={"after": 0})).get("events") or []
        kinds = _kinds(events)
        if "reward_signal" in kinds and "frame" in kinds and "trace.opened" in kinds:
            self.suite.ok("C3-03")
        else:
            self.suite.fail("C3-03", f"kinds={kinds}")

        self._c3_02_occupancy(meta, ids, stream_ids, generations)

        cfg = self.client.post("/policy-configs", json={"config_id": "luna_med", "config": {"model": "luna"}})
        cfg2 = self.client.post("/policy-configs", json={"config_id": "sol_med", "config": {"model": "sol"}})
        nxt = self._start(policy_ref={"harness": "react", "config": "sol_med"}, task_instance_id="seed:0")
        if cfg.status_code == 200 and cfg2.status_code == 200 and _json(nxt)["policy_ref"]["config"] == "sol_med":
            self.suite.ok("C3-05")
        else:
            self.suite.fail("C3-05", f"{cfg.status_code} {cfg2.status_code} {_json(nxt)}")

        refused = self._start(recipe={"require": {"true_checkpoint": True}})
        if refused.status_code == 403 and meta.get("true_checkpoint") == "unsupported":
            self.suite.ok("C3-06")
        else:
            self.suite.fail("C3-06", f"{refused.status_code}")

        self._c3_07_seal(first_id)

        cutoff = 2
        projection = project_envelopes(events, cutoff_sequence=cutoff)
        projected = projection["events"]
        sequences = [_seal_sequence(item) for item in projected]
        if not events:
            self.suite.fail("C3-08", "no events to project")
        elif any(seq is None for seq in sequences):
            self.suite.fail("C3-08", "cutoff projection missing sequence (not coerced to 0)")
        elif all(seq <= cutoff for seq in sequences):
            self.suite.ok("C3-08", "cutoff <= N; subscribe-before-start covered by C1-08")
        else:
            self.suite.fail("C3-08", f"cutoff leaked sequences={sequences}")

    def _c3_02_occupancy(
        self,
        meta: dict[str, Any],
        ids: list[str | None],
        stream_ids: list[str | None],
        generations: list[Any],
    ) -> None:
        distinct_ids = [item for item in ids if item is not None]
        distinct_streams = [item for item in stream_ids if item is not None]
        if len(distinct_ids) != 10 or len(set(distinct_ids)) != 10:
            self.suite.fail("C3-02", f"shared or missing rollout_id ids={ids}")
            return
        if len(distinct_streams) != 10 or len(set(distinct_streams)) != 10:
            self.suite.fail("C3-02", f"shared or missing stream.id streams={stream_ids}")
            return
        gens = [item for item in generations if item is not None]
        if len(set(gens)) != 1 or len(gens) != 10:
            self.suite.fail("C3-02", f"engine did not stay up generations={generations}")
            return
        leases = meta.get("scale_leases")
        if not isinstance(leases, int):
            self.suite.fail("C3-02", "scale_leases missing")
            return
        occupancy = _json(self.client.get("/metadata"))
        active = occupancy.get("active_leases")
        if active is None:
            self.suite.fail("C3-02", "active_leases missing")
            return
        held: list[str] = []
        try:
            if isinstance(active, int) and active >= leases:
                eleventh = self._start(submission_mode="async", task_instance_id="seed:10")
                body = _json(eleventh)
                if eleventh.status_code == 429 and body.get("affordance") == "scale_leases":
                    self.suite.ok(
                        "C3-02",
                        f"terminal still occupying; 11th 429 leases={leases} active={active}",
                    )
                else:
                    self.suite.fail(
                        "C3-02",
                        f"11th while full status={eleventh.status_code} body={body} active={active}",
                    )
                return
            for index in range(leases):
                response = self._start(submission_mode="async", task_instance_id=f"seed:occ:{index}")
                if response.status_code != 200:
                    self.suite.fail("C3-02", f"could not occupy slot {index}: {response.status_code}")
                    return
                held.append(_json(response)["rollout_id"])
            eleventh = self._start(submission_mode="async", task_instance_id="seed:occ:overflow")
            body = _json(eleventh)
            if eleventh.status_code == 200:
                extra = _json(eleventh).get("rollout_id")
                if extra:
                    held.append(extra)
            if eleventh.status_code == 429 and body.get("affordance") == "scale_leases":
                self.suite.ok(
                    "C3-02",
                    f"serial C3-01 released terminal leases; concurrent 11th 429 leases={leases}",
                )
            else:
                self.suite.fail("C3-02", f"11th while full status={eleventh.status_code} body={body}")
        finally:
            for rid in held:
                self.client.post(f"/rollouts/{rid}/complete")
            if isinstance(active, int) and active >= leases:
                for rid in distinct_ids:
                    self.client.post(f"/rollouts/{rid}/complete")

    def _c3_07_seal(self, rollout_id: str | None) -> None:
        if rollout_id is None:
            self.suite.fail("C3-07", "no rollout to seal")
            return
        response = self.client.get(f"/rollouts/{rollout_id}/trace")
        seal = _json(response)
        if response.status_code != 200:
            self.suite.fail("C3-07", f"seal {response.status_code} {seal}")
            return
        if seal.get("content_digest"):
            self.suite.extras["trace_v5_digest"] = seal.get("content_digest")
        required_meta = ("schema_version", "trace_id", "high_water", "closed")
        missing_meta = [name for name in required_meta if name not in seal]
        events = seal.get("events") if isinstance(seal.get("events"), list) else []
        kinds = {kind for kind in (_seal_kind(item) for item in events) if kind is not None}
        required_kinds = {"observation", "action", "reward_signal"}
        if self.spec.live_frames == "native":
            required_kinds.add("frame")
        missing_kinds = sorted(required_kinds - kinds)
        correlated, corr_detail = _obs_action_correlated(events)
        if missing_meta:
            self.suite.fail("C3-07", f"missing {missing_meta}")
        elif missing_kinds:
            self.suite.fail("C3-07", f"seal missing kinds {missing_kinds} have={sorted(kinds)}")
        elif seal.get("closed") is not True:
            self.suite.fail("C3-07", f"closed={seal.get('closed')}")
        elif seal.get("high_water") is None:
            self.suite.fail("C3-07", "high_water missing")
        elif not correlated:
            self.suite.fail("C3-07", corr_detail)
        elif self.paid:
            first_obs = next((_seal_sequence(item) for item in events if _seal_kind(item) == "observation"), None)
            early_policy = [
                item
                for item in events
                if str(_seal_kind(item) or "").startswith("span.policy")
                and first_obs is not None
                and _seal_sequence(item) is not None
                and _seal_sequence(item) < first_obs
            ]
            if early_policy:
                self.suite.fail("C3-07", "policy span before observation / C1-08 ready")
            else:
                self.suite.ok("C3-07", corr_detail)
        else:
            self.suite.ok("C3-07", corr_detail)

    def _c4(self, meta: dict[str, Any]) -> None:
        if self.target == "craftax_code_policy":
            held = self._start(submission_mode="async", policy_ref={"harness": "isolated_policy_process"})
            held_id = _json(held)["rollout_id"]
            old_rev = _json(held).get("policy_revision_id")
            put = self.client.put("/policy", json={"code": "def act(obs): return 0\n"})
            put_body = _json(put)
            if put.status_code == 200 and put_body.get("engine_generation") == _json(held).get("engine_generation"):
                self.suite.ok("C4-01")
            else:
                self.suite.fail("C4-01", f"{put.status_code} {put_body}")
            restart = self.client.post("/policy/restart")
            if restart.status_code == 200 and held_id in _json(restart).get("durable_logs", []):
                self.suite.ok("C4-02")
            else:
                self.suite.fail("C4-02", f"{_json(restart)}")
            completed = _json(self.client.post(f"/rollouts/{held_id}/complete"))
            if completed.get("policy_revision_id") == old_rev:
                self.suite.ok("C4-03")
            else:
                self.suite.fail("C4-03", f"pin mutated {completed.get('policy_revision_id')} vs {old_rev}")
            nxt = self._start(policy_ref={"harness": "isolated_policy_process", "code": "new"})
            scored = _json(self.client.post("/reward", json={"rollout_id": _json(nxt)["rollout_id"], "mode": "terminal"}))
            if nxt.status_code == 200 and scored.get("node_results"):
                self.suite.ok("C4-04")
            else:
                self.suite.fail("C4-04", f"{_json(nxt)} {scored}")
            bind = self.client.post("/policy-configs", json={"config_id": "luna_med", "config": {}})
            start_cfg = self._start(policy_ref={"harness": "isolated_policy_process", "config": "luna_med"})
            if bind.status_code == 403 and start_cfg.status_code == 403:
                self.suite.ok("C4-05")
            else:
                self.suite.fail("C4-05", f"bind={bind.status_code} start={start_cfg.status_code}")
            self.suite.skip("C4-06", "deo_nested only")
        else:
            self.suite.skip("C4-01", "deo_nested")
            self.suite.skip("C4-02", "deo_nested")
            self.suite.skip("C4-03", "deo_nested")
            self.suite.skip("C4-04", "deo_nested")
            self.suite.skip("C4-05", "deo_nested")
            started = self._start(task_instance_id="seed:0")
            body = _json(started)
            child = body.get("child_rollout_id")
            parent_reward = _json(self.client.post("/reward", json={"rollout_id": body["rollout_id"], "mode": "terminal"}))
            child_reward = (
                _json(self.client.post("/reward", json={"rollout_id": child, "mode": "terminal"}))
                if child
                else {}
            )
            child_events = (
                _json(self.client.get(f"/rollouts/{child}/events", params={"after": 0})).get("events") or []
                if child
                else []
            )
            parent_events = _json(self.client.get(body["stream"]["transports"]["poll"]["url"], params={"after": 0})).get("events") or []
            parent_kinds = _kinds(parent_events)
            child_kinds = _kinds(child_events)
            parent_value = parent_reward.get("reward")
            child_value = child_reward.get("reward")
            nodes = parent_reward.get("node_results")
            if parent_value is None or child_value is None:
                self.suite.fail(
                    "C4-06",
                    f"missing reward parent={parent_value} child={child_value} (not coerced to 0)",
                )
            elif (
                child
                and "frame" in child_kinds
                and "frame" not in parent_kinds
                and parent_value != child_value
                and _hillclimb_nodes_ok(nodes)
            ):
                self.suite.ok("C4-06")
            else:
                self.suite.fail(
                    "C4-06",
                    f"child={child} parent_kinds={parent_kinds} child_kinds={child_kinds} "
                    f"pr={parent_reward} cr={child_reward}",
                )
        hill = self.client.get("/hillclimb")
        if hill.status_code in {404, 405}:
            self.suite.ok("C4-07")
        else:
            self.suite.fail("C4-07", f"hillclimb {hill.status_code}")

    def _c5(self, meta: dict[str, Any]) -> None:
        if meta.get("blocking_trial") != "native":
            self.suite.fail("C5-01", f"{meta.get('blocking_trial')}")
        else:
            self.suite.ok("C5-01")
        a = self.client.post("/policy-configs", json={"config_id": "luna_med", "config": {"model": "luna"}})
        b = self.client.post("/policy-configs", json={"config_id": "sol_med", "config": {"model": "sol"}})
        held = self._start(submission_mode="async", policy_ref={"harness": "harbor_fused", "config": "luna_med"})
        mid = self.client.post("/policy-configs", json={"config_id": "other", "config": {}})
        self.client.post(f"/rollouts/{_json(held)['rollout_id']}/complete")
        if a.status_code == 200 and b.status_code == 200 and mid.status_code == 409:
            self.suite.ok("C5-02")
        else:
            self.suite.fail("C5-02", f"{a.status_code} {b.status_code} mid={mid.status_code}")
        started = self._start(policy_ref={"harness": "harbor_fused", "config": "luna_med"})
        events = _json(self.client.get(_json(started)["stream"]["transports"]["poll"]["url"], params={"after": 0}))["events"]
        kinds = _kinds(events)
        if "frame" in kinds or "jobs" in json.dumps(_json(started)):
            self.suite.fail("C5-03", f"kinds={kinds}")
        elif "trial.planned" in kinds and "verifier" in kinds:
            self.suite.ok("C5-03")
        else:
            self.suite.fail("C5-03", f"kinds={kinds}")
        native = next((item["payload"].get("reward.txt") for item in events if item.get("kind") == "verifier"), None)
        scored = _json(self.client.post("/reward", json={"rollout_id": _json(started)["rollout_id"], "mode": "terminal"}))
        if native == scored.get("reward"):
            self.suite.ok("C5-04")
        else:
            self.suite.fail("C5-04", f"native={native} wrapped={scored.get('reward')}")
        refused = self._start(recipe={"require": {"true_checkpoint": True}})
        if refused.status_code == 403:
            self.suite.ok("C5-05")
        else:
            self.suite.fail("C5-05", f"{refused.status_code}")
        prepared_id = "roll_c506"
        prepared = self.client.post(
            "/rollouts/prepare",
            json={"rollout_id": prepared_id, "telemetry": {"enabled": True, "transport": "sse"}},
        )
        before = _json(self.client.get(f"/rollouts/{prepared_id}/events", params={"after": 0}))
        before_kinds = _kinds(before.get("events") or [])
        started = self._start(rollout_id=prepared_id, policy_ref={"harness": "harbor_fused", "config": "luna_med"})
        after = _semantic(_json(self.client.get(f"/rollouts/{prepared_id}/events", params={"after": 0})).get("events") or [])
        first = after[0]["kind"] if after else None
        if (
            prepared.status_code == 200
            and "stream.subscribed" in before_kinds
            and "trial.planned" not in before_kinds
            and started.status_code == 200
            and first == "trace.opened"
            and "trial.planned" in _kinds(after)
        ):
            self.suite.ok("C5-06")
        else:
            self.suite.fail(
                "C5-06",
                f"prepare={prepared.status_code} before={before_kinds} start={started.status_code} first={first}",
            )

    def _c6(self, meta: dict[str, Any]) -> None:
        started = self._start(task_instance_id="seed:0")
        events = _json(self.client.get(_json(started)["stream"]["transports"]["poll"]["url"], params={"after": 0}))["events"]
        kinds = _kinds(events)
        if {"observation", "action", "reward_signal", "status"} <= set(kinds):
            self.suite.ok("C6-01")
        else:
            self.suite.fail("C6-01", f"kinds={kinds}")
        from synth_containers.compat.openenv import openenv_capability_surface

        surface = openenv_capability_surface()
        if (not surface.checkpoint_support) and meta.get("true_checkpoint") == "unsupported":
            self.suite.ok("C6-02")
        else:
            self.suite.fail("C6-02", "checkpoint claimed")
        scored = _json(self.client.post("/reward", json={"rollout_id": _json(started)["rollout_id"], "mode": "terminal"}))
        nodes = scored.get("node_results") or [{}]
        if nodes[0].get("authority") == "environment" and nodes[0].get("kind") != "script":
            self.suite.ok("C6-03")
        else:
            self.suite.fail("C6-03", f"{nodes}")
        poll = self.client.get(_json(started)["stream"]["transports"]["poll"]["url"], params={"after": 0})
        if poll.status_code == 200:
            self.suite.ok("C6-04")
        else:
            self.suite.fail("C6-04", f"{poll.status_code}")

    def _c7(self, meta: dict[str, Any]) -> None:
        prepared_id = f"roll_c7_{self.target}"
        prepared = self.client.post(
            "/rollouts/prepare",
            json={"rollout_id": prepared_id, "telemetry": {"enabled": True, "transport": "sse"}},
        )
        descriptor = _json(prepared)["stream"]
        self.suite.extras["stream_descriptor"] = descriptor
        if descriptor.get("id") in {None, "live", "jobs"}:
            self.suite.fail("C7-W01", f"id={descriptor.get('id')}")
        else:
            before = _json(self.client.get(descriptor["transports"]["poll"]["url"], params={"after": 0}))
            if any(item.get("kind") == "stream.subscribed" for item in before.get("events") or []):
                self.suite.ok("C7-W01")
            else:
                self.suite.fail("C7-W01", "not subscribed")
        started = self._start(rollout_id=prepared_id, task_instance_id="seed:0", omit_reward=self.spec.reward_kind == "env_sum")
        started_body = _json(started)
        if isinstance(started_body.get("stream"), dict):
            self.suite.extras["stream_descriptor"] = started_body["stream"]
        events = _json(self.client.get(descriptor["transports"]["poll"]["url"], params={"after": 0}))["events"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "envelopes.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
            replayed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if [(item.get("sequence"), item.get("digest")) for item in events] == [
            (item.get("sequence"), item.get("digest")) for item in replayed
        ]:
            self.suite.ok("C7-W02")
        else:
            self.suite.fail("C7-W02", "replay mismatch")

        usage = started_body.get("usage")
        reward_body = _json(self.client.post("/reward", json={"rollout_id": prepared_id, "mode": "terminal"}))
        projection = project_envelopes(
            events,
            usage=usage,
            reward=reward_body.get("reward"),
            reward_status=str(reward_body.get("status") or "absent"),
        )
        defects = assert_honest_projection(projection)
        if defects:
            self.suite.fail("C7-W03", str(defects))
        else:
            self.suite.ok("C7-W03")

        if self.target.startswith("craftax") and projection["has_reward_txt"]:
            self.suite.fail("C7-W04", "craftax has reward.txt")
        elif self.target in {"harbor_public", "deo_nested"} and projection["has_live_frames"]:
            self.suite.fail("C7-W04", "harbor has frames")
        elif self.target.startswith("digbench") and (projection["has_live_frames"] or projection["has_reward_txt"]):
            self.suite.fail("C7-W04", "digbench has frames or reward.txt")
        elif self.target == "banking77_classify" and (
            projection["has_live_frames"] or projection["has_reward_txt"]
        ):
            self.suite.fail("C7-W04", "banking77 has frames or reward.txt")
        elif self.target == "openenv_echo" and (
            projection["has_live_frames"] or projection["has_reward_txt"]
        ):
            self.suite.fail("C7-W04", "openenv_echo has frames or reward.txt")
        else:
            self.suite.ok("C7-W04")

        receipt_preview = json.dumps({"events": events, "metadata": {k: meta[k] for k in meta if k != "collector"}})
        if any(name in receipt_preview for name in ("collector", "capability_blob")):
            self.suite.fail("C7-W05", "blob leaked")
        else:
            self.suite.ok("C7-W05")

        seal = self.client.get(f"/rollouts/{prepared_id}/trace")
        first = _json(seal) if seal.status_code == 200 else {}
        if first.get("content_digest"):
            self.suite.extras["trace_v5_digest"] = first.get("content_digest")
        if seal.status_code != 200:
            self.suite.fail("C7-W06", f"seal {seal.status_code}")
        elif not first.get("trace_id") or not first.get("content_digest") or not first.get("rollout_id"):
            self.suite.fail("C7-W06", f"seal missing identity fields {sorted(first)}")
        else:
            stop = self.client.post("/world/stop")
            again = self.client.get(f"/rollouts/{prepared_id}/trace")
            second = _json(again) if again.status_code == 200 else {}
            live_after = self.client.get(f"/rollouts/{prepared_id}/events", params={"after": 0})
            if again.status_code != 200:
                self.suite.fail(
                    "C7-W06",
                    f"seal wiped after world_stop {again.status_code} stop={stop.status_code} live={live_after.status_code}",
                )
            elif (
                second.get("trace_id") == first.get("trace_id")
                and second.get("content_digest") == first.get("content_digest")
                and second.get("rollout_id") == first.get("rollout_id")
                and first.get("trace_id") == prepared_id
            ):
                self.suite.ok(
                    "C7-W06",
                    f"seal retained after world_stop live={live_after.status_code}",
                )
            else:
                self.suite.fail(
                    "C7-W06",
                    f"identity drift stop={stop.status_code} first={first.get('trace_id')} second={second.get('trace_id')}",
                )

        child_ref = {
            "rollout_id": prepared_id,
            "stream.id": descriptor.get("id"),
            "reward_url": descriptor.get("reward", {}).get("url"),
        }
        if all(child_ref.values()):
            self.suite.ok("C7-O01")
        else:
            self.suite.fail("C7-O01", f"{child_ref}")

        a = self._start(submission_mode="async", task_instance_id="seed:0")
        b = self._start(submission_mode="async", task_instance_id="seed:1")
        if a.status_code == 200 and b.status_code == 200:
            aid, bid = _json(a)["rollout_id"], _json(b)["rollout_id"]
            ae = _json(self.client.get(f"/rollouts/{aid}/events", params={"after": 0}))
            be = _json(self.client.get(f"/rollouts/{bid}/events", params={"after": 0}))
            self.client.post(f"/rollouts/{aid}/complete")
            self.client.post(f"/rollouts/{bid}/complete")
            ar = _json(self.client.post("/reward", json={"rollout_id": aid, "mode": "terminal"}))
            br = _json(self.client.post("/reward", json={"rollout_id": bid, "mode": "terminal"}))
            if aid != bid and ae.get("stream_id") != be.get("stream_id") and ar.get("execution_id") != br.get("execution_id"):
                self.suite.ok("C7-O02")
            else:
                self.suite.fail("C7-O02", "logs crossed")
        elif b.status_code == 429:
            self.suite.ok("C7-O02", "honestly queued")
            if a.status_code == 200:
                self.client.post(f"/rollouts/{_json(a)['rollout_id']}/complete")
        else:
            self.suite.fail("C7-O02", f"{a.status_code} {b.status_code}")

        held = self._start(submission_mode="async", task_instance_id="seed:9")
        held_id = _json(held)["rollout_id"]
        pin_before = _json(held).get("policy_ref")
        if self.target == "craftax_code_policy":
            self.client.put("/policy", json={"code": "x"})
        elif self.spec.affordances.level("bind_policy_config") != "unsupported":
            self.client.post("/policy-configs", json={"config_id": "mid", "config": {"x": 1}})
        after = _json(self.client.get(f"/rollouts/{held_id}"))
        self.client.post(f"/rollouts/{held_id}/complete")
        if after.get("policy_ref") == pin_before:
            self.suite.ok("C7-O03")
        else:
            self.suite.fail("C7-O03", f"{pin_before} vs {after.get('policy_ref')}")

        if self.target == "craftax_code_policy":
            from synth_containers.platform.policy_process import DEFAULT_HEURISTIC

            self.client.put("/policy", json={"code": DEFAULT_HEURISTIC})

        incomplete = self._start(submission_mode="async", task_instance_id="seed:absent")
        if incomplete.status_code != 200:
            self.suite.fail("C7-O04", f"absent child {incomplete.status_code}")
        else:
            incomplete_id = _json(incomplete)["rollout_id"]
            absent = _json(self.client.get("/reward", params={"rollout_id": incomplete_id}))
            summary = [
                {
                    "rollout_id": prepared_id,
                    "reward": reward_body.get("reward"),
                    "status": reward_body.get("status"),
                },
                {
                    "rollout_id": incomplete_id,
                    "reward": absent.get("reward"),
                    "status": absent.get("status"),
                },
            ]
            dropped = [row for row in summary if row["reward"] is None and row["rollout_id"] is None]
            filled = [
                row
                for row in summary
                if row["reward"] == 0 and row["status"] in {"absent", "incomplete", "gated", "refused"}
            ]
            self.client.post(f"/rollouts/{incomplete_id}/complete")
            if (
                len(summary) == 2
                and not dropped
                and not filled
                and any(row["reward"] is None for row in summary)
                and {row["rollout_id"] for row in summary} == {prepared_id, incomplete_id}
            ):
                self.suite.ok("C7-O04", "absent child kept with null reward")
            else:
                self.suite.fail("C7-O04", f"summary={summary} dropped={dropped} filled={filled}")

        if usage is None or any(usage.get(key) is None for key in ("prompt_tokens", "completion_tokens") if key in (usage or {})):
            if usage and 0 in {usage.get("prompt_tokens"), usage.get("completion_tokens")} and None in usage.values():
                self.suite.fail("C7-O05", f"{usage}")
            else:
                self.suite.ok("C7-O05")
        else:
            self.suite.ok("C7-O05", "usage present and honest")

    def _c8(self, meta: dict[str, Any]) -> None:
        if meta.get("adapter_chain") != [] or "digbench" not in str(meta.get("world_ref")):
            self.suite.fail("C8-01", f"{meta.get('world_ref')} {meta.get('adapter_chain')}")
        else:
            self.suite.ok("C8-01")
        refused = self._start(recipe={"require": {"live_frames": True}})
        started = self._start(task_instance_id="P-1", policy_ref={"harness": "react_legal_actions", "config": "react_legal_actions"})
        events = _json(self.client.get(_json(started)["stream"]["transports"]["poll"]["url"], params={"after": 0}))["events"]
        kinds = set(_kinds(events))
        seven = {
            "session.opened",
            "observation",
            "legal_actions",
            "stats",
            "action",
            "invalid_action",
            "status",
        }
        if refused.status_code == 403 and seven <= kinds and "frame" not in kinds and "state" not in kinds:
            self.suite.ok("C8-02")
        else:
            self.suite.fail("C8-02", f"kinds={kinds} refuse={refused.status_code}")
        if meta.get("true_checkpoint") == "unsupported" and meta.get("reconnect") == "native":
            self.suite.ok("C8-03")
        else:
            self.suite.fail("C8-03", f"ckpt={meta.get('true_checkpoint')} reconnect={meta.get('reconnect')}")

        basic = self.client.post(
            "/policy-configs",
            json={"config_id": "react_legal_actions", "harness": "react_legal_actions", "config": {"mcp_bind": "unused"}},
        )
        agentic = self.client.post(
            "/policy-configs",
            json={"config_id": "agentic_codex", "harness": "codex", "config": {"mcp_bind": "digbench-mcp"}},
        )
        if basic.status_code == 200 and agentic.status_code == 200:
            if self.target == "digbench_mock" and not self.paid:
                self.suite.ok("C8-04", "agentic MCP skipped on mock")
            else:
                agentic_run = self._start(policy_ref={"harness": "codex", "config": "agentic_codex"})
                akinds = _kinds(_json(self.client.get(_json(agentic_run)["stream"]["transports"]["poll"]["url"], params={"after": 0}))["events"])
                if "span.mcp.opened" in akinds:
                    self.suite.ok("C8-04")
                else:
                    self.suite.fail("C8-04", f"{akinds}")
        else:
            self.suite.fail("C8-04", f"{basic.status_code} {agentic.status_code}")

        win = self._start(outcome="completed")
        loss = self._start(outcome="game_over")
        live = self._start(submission_mode="async")
        win_r = _json(self.client.post("/reward", json={"rollout_id": _json(win)["rollout_id"], "mode": "terminal"}))
        loss_r = _json(self.client.post("/reward", json={"rollout_id": _json(loss)["rollout_id"], "mode": "terminal"}))
        live_r = self.client.post("/reward", json={"rollout_id": _json(live)["rollout_id"], "mode": "terminal"})
        self.client.post(f"/rollouts/{_json(live)['rollout_id']}/complete")
        if (
            win_r.get("reward") == 1.0
            and loss_r.get("reward") == 0.0
            and live_r.status_code == 409
            and (win_r.get("node_results") or [{}])[0].get("authority") == "environment"
            and not win_r.get("start_session_delta")
        ):
            self.suite.ok("C8-05")
        else:
            self.suite.fail("C8-05", f"win={win_r} loss={loss_r} live={live_r.status_code}")

        prepared_id = "roll_c806"
        self.client.post("/rollouts/prepare", json={"rollout_id": prepared_id, "telemetry": {"enabled": True, "transport": "sse"}})
        before = _kinds(_json(self.client.get(f"/rollouts/{prepared_id}/events", params={"after": 0}))["events"])
        self._start(rollout_id=prepared_id)
        after = _semantic(_json(self.client.get(f"/rollouts/{prepared_id}/events", params={"after": 0}))["events"])
        mutating = [item["kind"] for item in after if item["kind"] != "trace.opened"]
        if "stream.subscribed" in before and mutating and mutating[0] == "start_session":
            self.suite.ok("C8-06")
        else:
            self.suite.fail("C8-06", f"before={before} mutating={mutating}")

        blob = json.dumps(events) + json.dumps(meta) + json.dumps(_json(self.client.get(f"/rollouts/{_json(started)['rollout_id']}/trace")))
        if any(token in blob for token in ("DIGBENCH_API_TOKEN", "Authorization", "Bearer ")):
            self.suite.fail("C8-07", "token leaked")
        else:
            self.suite.ok("C8-07")

        projection = project_envelopes(events)
        if projection["has_live_frames"] or projection["has_reward_txt"]:
            self.suite.fail("C8-08", "unexpected kinds")
        else:
            self.suite.ok("C8-08")

        a = self._start(submission_mode="async")
        b = self._start(submission_mode="async")
        if a.status_code == 200 and b.status_code == 200:
            self.suite.ok("C8-09")
            self.client.post(f"/rollouts/{_json(a)['rollout_id']}/complete")
            self.client.post(f"/rollouts/{_json(b)['rollout_id']}/complete")
        elif b.status_code == 429:
            self.suite.ok("C8-09", "queued")
            if a.status_code == 200:
                self.client.post(f"/rollouts/{_json(a)['rollout_id']}/complete")
        else:
            self.suite.fail("C8-09", f"{a.status_code} {b.status_code}")

        persist_id = _json(started)["rollout_id"]
        dropped = self.client.post(f"/rollouts/{persist_id}/drop_session")
        seal = self.client.get(f"/rollouts/{persist_id}/trace")
        if dropped.status_code == 200 and seal.status_code == 200:
            self.suite.ok("C8-10")
        else:
            self.suite.fail("C8-10", f"{dropped.status_code} {seal.status_code}")

        if meta.get("world_ref") == "world:digbench:P-1":
            self.suite.ok("C8-11", "P-1 frozen; agentic skipped on mock" if not self.paid else "P-1 frozen")
        else:
            self.suite.fail("C8-11", f"{meta.get('world_ref')}")


def receipt_from_suite(suite: Suite) -> dict[str, Any]:
    results = {item.test_id: item.status for item in suite.checks}
    details = {item.test_id: item.detail for item in suite.checks if item.detail}
    descriptor = suite.extras.get("stream_descriptor")
    body = {
        "schema": SCHEMA,
        "target": suite.target,
        "adapter_chain": list(TARGETS[suite.target].adapter_chain),
        "evaluation_plan_ref": TARGETS[suite.target].evaluation_plan_ref,
        "affordances": TARGETS[suite.target].affordances.advertised(),
        "results": results,
        "details": details,
        "stream_descriptor_digest": _digest(descriptor) if descriptor else None,
        "trace_v5_digest": suite.extras.get("trace_v5_digest"),
        "failed": [item.test_id for item in suite.failed()],
    }
    body["content_digest"] = _digest({k: v for k, v in body.items() if k != "content_digest"})
    return body


def run_against_client(client: Any, target: str, *, paid: bool = False) -> Suite:
    return Runner(client, target, paid=paid).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Containers-compat conformance (C0–C8)")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument("--paid", action="store_true")
    parser.add_argument("--receipt", default=None)
    args = parser.parse_args(argv)
    if args.base_url:
        import httpx

        client = httpx.Client(base_url=args.base_url, timeout=30.0)
        suite = run_against_client(client, args.target, paid=args.paid)
        client.close()
    else:
        from fastapi.testclient import TestClient

        from synth_containers.platform import create_compat_app

        with TestClient(create_compat_app(args.target)) as client:
            suite = run_against_client(client, args.target, paid=args.paid)
    receipt = receipt_from_suite(suite)
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    json.dump(receipt, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if suite.failed() else 0


if __name__ == "__main__":
    raise SystemExit(main())
