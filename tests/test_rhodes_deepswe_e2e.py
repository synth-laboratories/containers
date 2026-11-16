"""Lifecycle qualification without provider/model calls."""

import argparse
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from synth_containers import rhodes_deepswe_e2e as e2e
from synth_containers.pools import PoolRolloutTimeout
from synth_containers.rhodes_client import RhodesRequestError

AGENT = {"name": "codex", "model_name": "gpt-6-luna", "kwargs": {"reasoning_effort": "low"}}
OWNERS = ["synth-harbor-" + "a" * 32, "synth-harbor-" + "b" * 32]
SNAPSHOT = "synth-harbor-" + "c" * 32


def inputs(tmp_path: Path):
    root = tmp_path / "corpus" / "tasks" / "demo"
    (root / "tests").mkdir(parents=True)
    (root / "solution").mkdir()
    (root / "task.toml").write_text("""
[metadata]
base_commit_hash = "abc123"
[agent]
timeout_sec = 60
[verifier]
timeout_sec = 30
[verifier.environment]
cpus = 2
memory_mb = 8192
""")
    (root / "instruction.md").write_text("Fix the bug.")
    (root / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "solution/solution.patch").write_text("reference must not enter agent build")
    pins = tmp_path / "pins.toml"
    pins.write_text('["deepswe/demo"]\nagent="registry/task@sha256:' + "d" * 64 + '"\n')
    return argparse.Namespace(
        deepswe_root=tmp_path / "corpus",
        pins=pins,
        task="demo",
        seed=0,
        project_id=None,
        rollout_seconds=10,
        queue_seconds=0.05,
        cleanup_seconds=1,
        poll_seconds=0.001,
        require_solved=False,
    )


def body(reward=0):
    return {
        "status": "completed",
        "metadata": {"agent_name": "codex", "model_name": "gpt-6-luna"},
        "execution": {
            "launches": [{"role": role, "provider": "daytona"} for role in ("agent", "verifier")]
        },
        "result": {
            "success": True,
            "metrics": {
                "outcome_reward": reward,
                "details": {
                    "agent_exit_code": 0,
                    "verifier_exit_code": 0,
                    "verifier_environment_mode": "isolated",
                },
            },
            "execution_metadata": {
                "snapshot_id": SNAPSHOT,
                "sandbox_cleanup": [
                    {"role": role, "owner": owner, "state": "confirmed_absent"}
                    for role, owner in zip(("agent", "verifier"), OWNERS, strict=True)
                ],
            },
        },
    }


def events():
    return [
        {
            "sequence": 1,
            "event_type": "rollout.phase",
            "phase": "agent",
            "state": "exited",
            "exit_code": 0,
        },
        {
            "sequence": 2,
            "event_type": "rollout.phase",
            "phase": "verification",
            "state": "exited",
            "exit_code": 0,
        },
        {
            "sequence": 3,
            "event_type": "rollout.inference_accounting",
            "accounting_status": "settled",
            "invocation_id": "inv-1",
        },
    ]


class Client:
    async def negotiate_api(self, *, required_capabilities):
        assert set(required_capabilities) == {"customer.pool.create", "customer.rollout.create"}
        return {"api": {"version": "2.0.0"}}

    def __init__(self, result=None, timeout=False, fail_build=False, lose_create=False):
        self.body = result or body()
        self.timeout, self.fail_build, self.lose_create = timeout, fail_build, lose_create
        self.calls = []
        self.created = None
        self._client = SimpleNamespace(get=self.health)

    async def health(self, path):
        return httpx.Response(
            200,
            json={
                "service": "rhodes-api",
                "build": {
                    "image_digest": "sha256:" + "e" * 64,
                },
            },
            request=httpx.Request("GET", "https://rhodes.invalid" + path),
        )

    async def _request(self, method, path, *, payload, replay_safe):
        self.calls.append((method, path))
        assert replay_safe is False
        self.created = copy.deepcopy(payload)
        if self.lose_create:
            raise OSError("secret provider response must not be logged")
        return {"id": payload["pool_id"]}

    async def get_pool(self, pool_id):
        assert pool_id == self.created["pool_id"]
        return self.created

    async def create_task(self, pool_id, payload):
        self.calls.append(("create_task", payload))
        if self.fail_build:
            raise RuntimeError("secret build response must not be logged")
        assert payload["harbor_agent"] == AGENT
        assert payload["harbor_native_phases"]["verifier_workspace_transfer"] == "model_patch"
        return {
            **payload,
            "deployment_id": "deployment-owned",
            "harbor_agent": {
                **payload["harbor_agent"],
                "env": {},
            },
        }

    async def submit(self, pool_id, payload):
        self.calls.append(("submit", copy.deepcopy(payload)))
        return "rollout-owned"

    async def wait_for(self, rollout_id, **kwargs):
        if self.timeout:
            self.body["status"] = "running"
            raise PoolRolloutTimeout("observation timed out")
        return SimpleNamespace(payload=self.body)

    async def get_rollout(self, rollout_id):
        return self.body

    async def event_page(self, rollout_id, **kwargs):
        return {"events": events(), "has_more": False}

    async def usage(self, rollout_id):
        return {"rollout_id": rollout_id, "usage": {"producer_reported": {"status": "unavailable"}}}

    async def cancel(self, rollout_id):
        self.calls.append(("cancel", rollout_id))
        self.body["status"] = "cancelled"
        return {"status": "cancelled"}

    async def delete_pool(self, pool_id):
        self.calls.append(("delete_pool", pool_id))
        return {"status": "archived", "provider_cleanup": {"complete": True}}


class Observer:
    def __init__(self):
        self.samples = 0
        self.observed_owners = []

    async def snapshots(self):
        self.samples += 1
        # Another actor's newly created snapshot must not fail our cleanup.
        if self.samples == 1:
            return {"shared-snapshot"}
        if self.samples == 2:
            return {"shared-snapshot", SNAPSHOT}
        return {"shared-snapshot", "foreign-new"}

    async def remaining_owners(self, owners):
        self.observed_owners.extend(owners)
        return []


def run(tmp_path, client=None, **options):
    args = inputs(tmp_path)
    for name, value in options.items():
        setattr(args, name, value)
    output = tmp_path / "receipt"
    output.mkdir()
    client, observer = client or Client(), Observer()
    report = asyncio.run(e2e.E2ERun(client, observer, args, output).execute())
    return report, client, observer


def test_complete_lifecycle_with_zero_reward_is_not_a_claim_of_solving(tmp_path):
    report, client, observer = run(tmp_path)
    assert report["passed"] is True
    assert report["reward"] == 0 and report["task_solved"] is False
    submissions = [payload for name, payload in client.calls if name == "submit"]
    assert len(submissions) == 2 and submissions[0] == submissions[1]
    assert submissions[0]["policy"]["model"] == "gpt-6-luna"
    assert observer.observed_owners == OWNERS
    assert {"upload_and_build", "run_agent_and_verifier", "teardown", "total"} <= report[
        "timings_seconds"
    ].keys()
    saved = json.loads((tmp_path / "receipt/report.json").read_text())
    assert saved["passed"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent_exit_code", 1),
        ("verifier_exit_code", 1),
        ("verifier_environment_mode", "shared"),
    ],
)
def test_unexpected_native_result_fails_but_deletes_pool(tmp_path, field, value):
    result = body()
    result["result"]["metrics"]["details"][field] = value
    report, client, _ = run(tmp_path, Client(result))
    assert report["passed"] is False
    assert any(name == "delete_pool" for name, _ in client.calls)


def test_missing_reward_and_inference_receipts_never_count_as_success():
    result = body(None)
    checks = e2e.result_checks(result, events()[:2], AGENT)
    assert checks["native_reward_is_finite"] is False
    assert checks["inference_settled"] is False


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True])
def test_invalid_reward_still_writes_failure_receipt_and_tears_down(tmp_path, reward):
    report, client, _ = run(tmp_path, Client(body(reward)))
    assert report["passed"] is False
    assert report["checks"]["native_reward_is_finite"] is False
    assert any(name == "delete_pool" for name, _ in client.calls)
    saved = json.loads((tmp_path / "receipt/report.json").read_text())
    assert saved["passed"] is False


def test_strict_solved_option_reports_zero_reward_as_failure(tmp_path):
    report, _, _ = run(tmp_path, require_solved=True)
    assert report["passed"] is False and report["checks"]["task_solved"] is False


def test_timeout_explicitly_cancels_original_attempt_before_teardown(tmp_path):
    report, client, _ = run(tmp_path, Client(timeout=True))
    assert report["passed"] is False
    assert client.calls.index(("cancel", "rollout-owned")) < next(
        i for i, call in enumerate(client.calls) if call[0] == "delete_pool"
    )
    assert sum(name == "submit" for name, _ in client.calls) == 1


@pytest.mark.parametrize("failure", ["fail_build", "lose_create"])
def test_failure_still_attempts_owned_pool_teardown_and_redacts_errors(tmp_path, failure):
    report, client, _ = run(tmp_path, Client(**{failure: True}))
    assert report["passed"] is False
    assert any(name == "delete_pool" for name, _ in client.calls)
    assert "secret" not in (tmp_path / "receipt/report.json").read_text()


def test_foreign_pool_is_never_deleted(tmp_path):
    class ForeignClient(Client):
        async def get_pool(self, pool_id):
            return {"metadata": {"e2e_run_id": "another-owner"}}

    report, client, _ = run(tmp_path, ForeignClient())
    assert report["checks"]["teardown_confirmed"] is False
    assert not any(name == "delete_pool" for name, _ in client.calls)


def test_cleanup_deadline_produces_failure_receipt(tmp_path):
    class HeldClient(Client):
        async def delete_pool(self, pool_id):
            raise RhodesRequestError(409, "held", None)

    report, _, _ = run(tmp_path, HeldClient(), cleanup_seconds=0.01)
    assert report["passed"] is False
    assert report["checks"]["teardown_confirmed"] is False
    assert report["errors"][-1]["type"] == "TimeoutError"


def test_sigterm_style_cancellation_still_tears_down(tmp_path):
    class CancelledClient(Client):
        async def wait_for(self, *args, **kwargs):
            self.body["status"] = "running"
            raise asyncio.CancelledError()

    report, client, _ = run(tmp_path, CancelledClient())
    assert report["passed"] is False
    assert ("cancel", "rollout-owned") in client.calls
    assert any(name == "delete_pool" for name, _ in client.calls)


def test_dry_run_needs_no_credentials_or_provider(tmp_path, monkeypatch, capsys):
    args = inputs(tmp_path)
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    code = e2e.main(
        [
            "--dry-run",
            "--deepswe-root",
            str(args.deepswe_root),
            "--pins",
            str(args.pins),
            "--task",
            "demo",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["network_requests"] == 0 and plan["reasoning_effort"] == "low"


def test_event_cursor_gaps_refuse_qualification(tmp_path):
    class GappedClient(Client):
        async def event_page(self, *args, **kwargs):
            return {"events": [{"sequence": 3}], "has_more": False}

    report, _, _ = run(tmp_path, GappedClient())
    assert report["passed"] is False
    assert report["errors"][0]["type"] == "ValueError"


def test_snapshot_still_present_cannot_pass_cleanup(tmp_path):
    args = inputs(tmp_path)
    output = tmp_path / "receipt"
    output.mkdir()
    args.cleanup_seconds = 0.01

    class LeakingObserver(Observer):
        async def snapshots(self):
            self.samples += 1
            return set() if self.samples == 1 else {SNAPSHOT}

    report = asyncio.run(e2e.E2ERun(Client(), LeakingObserver(), args, output).execute())
    assert report["passed"] is False
    assert report["remaining_snapshots"] == [SNAPSHOT]


def test_real_client_uses_public_wire_paths_and_original_mutation_ids(tmp_path):
    from synth_containers.rhodes_client import RhodesClient

    args = inputs(tmp_path)
    output = tmp_path / "wire-receipt"
    output.mkdir()
    seen, state = [], {}

    def respond(request):
        path, method = request.url.path, request.method
        payload = json.loads(request.content) if request.content else None
        seen.append((method, path, payload))
        if path == "/.well-known/synth-api":
            return httpx.Response(200, json={
                "schema": "synth.api-discovery.v1", "service": "rhodes-api",
                "authority": "synth-backend.core",
                "api": {"name": "synth.rhodes", "version": "2.0.0", "supported_versions": ["2.0.0"]},
                "capabilities": {"customer.pool.create": True, "customer.rollout.create": True},
            }, headers={"Synth-API-Version": "2.0.0"})
        if path in {"/healthz", "/readyz"}:
            return httpx.Response(
                200,
                json={
                    "service": "rhodes-api",
                    "build": {
                        "image_digest": "sha256:" + "e" * 64,
                    },
                },
            )
        if method == "POST" and path == "/v1/pools":
            state["pool"] = payload
            return httpx.Response(200, json={"pool_id": payload["pool_id"]})
        if method == "GET" and path == "/v1/pools/" + state["pool"]["pool_id"]:
            return httpx.Response(200, json=state["pool"])
        if path.endswith("/tasks"):
            return httpx.Response(200, json={**payload, "harbor_agent": {**AGENT, "env": {}}})
        if method == "POST" and path.endswith("/rollouts"):
            return httpx.Response(200, json={"rollout_id": "rollout-owned"})
        if path == "/v1/rollouts/rollout-owned":
            return httpx.Response(200, json=body())
        if path.endswith("/events"):
            assert request.url.params["format"] == "json"
            return httpx.Response(200, json={"events": events(), "has_more": False})
        if path.endswith("/usage"):
            return httpx.Response(200, json={"rollout_id": "rollout-owned", "usage": {}})
        if method == "DELETE":
            return httpx.Response(
                200, json={"status": "archived", "provider_cleanup": {"complete": True}}
            )
        raise AssertionError((method, path))

    async def exercise():
        async with (
            httpx.AsyncClient(
                transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid"
            ) as transport,
            RhodesClient(api_key="fixture", client=transport) as client,
        ):
            return await e2e.E2ERun(client, Observer(), args, output).execute()

    report = asyncio.run(exercise())
    assert report["passed"] is True
    creates = [
        payload for method, path, payload in seen if method == "POST" and path == "/v1/pools"
    ]
    assert len(creates) == 1 and creates[0]["pool_id"] == report["pool_id"]
    submits = [
        payload for method, path, payload in seen if method == "POST" and path.endswith("/rollouts")
    ]
    assert len(submits) == 2 and submits[0] == submits[1]


def test_waits_for_pending_accounting_without_replaying_events(tmp_path):
    class PendingClient(Client):
        async def event_page(self, rollout_id, *, after_sequence, **kwargs):
            if after_sequence == 0:
                return {"events": events(), "has_more": False, "inference_pending": True}
            assert after_sequence == 3
            return {"events": [], "has_more": False, "inference_pending": False}

    report, _, _ = run(tmp_path, PendingClient())
    assert report["passed"] is True


def test_provider_cleanup_receipt_must_complete(tmp_path):
    class PendingCleanupClient(Client):
        async def delete_pool(self, pool_id):
            return {"status": "archived", "provider_cleanup": {"complete": False}}

    report, _, _ = run(tmp_path, PendingCleanupClient(), cleanup_seconds=0.01)
    assert report["passed"] is False
    assert report["checks"]["teardown_confirmed"] is False


def test_snapshot_absence_without_prior_observation_is_not_a_cleanup_proof(tmp_path):
    class UnobservedSnapshot(Observer):
        async def snapshots(self):
            return set()

    args = inputs(tmp_path)
    output = tmp_path / "receipt"
    output.mkdir()
    report = asyncio.run(e2e.E2ERun(Client(), UnobservedSnapshot(), args, output).execute())
    assert report["passed"] is False
    assert report["checks"]["snapshot_observed_before_teardown"] is False


def test_legacy_v1_endpoint_is_refused_before_pool_creation(tmp_path):
    class LegacyClient(Client):
        async def health(self, path):
            return httpx.Response(
                200,
                json={"ok": True},
                request=httpx.Request("GET", "https://legacy.invalid" + path),
            )

    report, client, observer = run(tmp_path, LegacyClient())
    assert report["passed"] is False
    assert report["errors"][0]["type"] == "APIContractError"
    assert not client.calls and observer.samples == 0


def test_hosted_capability_refusal_precedes_provider_inventory_and_upload(tmp_path):
    args = inputs(tmp_path)
    output = tmp_path / "capability-refusal"
    output.mkdir()
    seen = []

    def respond(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, headers={"Synth-API-Version": "2.0.0"}, json={
            "schema": "synth.api-discovery.v1", "service": "rhodes-api",
            "authority": "synth-backend.core",
            "api": {"name": "synth.rhodes", "version": "2.0.0", "supported_versions": ["2.0.0"]},
            "capabilities": {"customer.pool.create": False, "customer.rollout.create": False},
        })

    observer = Observer()

    async def exercise():
        from synth_containers.rhodes_client import RhodesClient
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), base_url="https://rhodes.invalid") as transport:
            return await e2e.E2ERun(RhodesClient(api_key="fixture", client=transport), observer, args, output).execute()

    report = asyncio.run(exercise())
    assert report["passed"] is False
    assert report["errors"] == [{"stage": "preflight", "type": "APIContractError"}]
    assert observer.samples == 0
    assert seen == [("GET", "/.well-known/synth-api")]


def test_queue_timeout_cancels_without_waiting_for_agent_budget(tmp_path):
    class StuckQueue(Client):
        def __init__(self):
            super().__init__()
            self.body["status"] = "queued"

        async def wait_for(self, *args, **kwargs):
            raise AssertionError("The agent observer must not run for a stuck queue")

    report, client, _ = run(tmp_path, StuckQueue(), queue_seconds=0.01)
    assert report["passed"] is False
    assert report["errors"][0]["stage"] == "queue_wait"
    assert ("cancel", "rollout-owned") in client.calls


def test_definite_pool_refusal_does_not_claim_unknown_provider_liability(tmp_path):
    class RefusedClient(Client):
        async def _request(self, *args, **kwargs):
            raise RhodesRequestError(
                403,
                "rhodes_customer_route_refused",
                None,
                {"service": "rhodes", "stage": "public_origin_guard"},
            )

        async def get_pool(self, pool_id):
            raise AssertionError("A definite pre-creation refusal created no pool")

    report, client, _ = run(tmp_path, RefusedClient())
    assert report["passed"] is False
    assert report["checks"]["pool_not_created"] is True
    assert report["checks"]["teardown_confirmed"] is True
    assert report["errors"][0]["code"] == "rhodes_customer_route_refused"
    assert report["errors"][0]["upstream"]["stage"] == "public_origin_guard"
    assert not client.calls
