"""The lifecycle driver reports observations and always tears down what it created."""

import asyncio
import json

import pytest

from synth_containers import rhodes_lifecycle
from synth_containers.rhodes_client import RhodesRequestError


class FakeRhodes:
    def __init__(self, *, refuse=None, not_ready_reads=1, rollout_seconds=0.05):
        self.refuse = refuse or {}
        self.not_ready = not_ready_reads
        self.rollout_seconds = rollout_seconds
        self.calls = []
        self.pools = {}
        self.deployments = {}
        self.leases = {}
        self.rollouts = {}
        self.cancelled = []

    def _call(self, name, **details):
        self.calls.append((name, details))
        if name in self.refuse:
            raise RhodesRequestError(409, self.refuse[name], None)

    async def create_project_pool(self, body):
        self._call("create_pool", body=body)
        pool_id = f"pool-{len(self.pools)}"
        self.pools[pool_id] = "active"
        return {"pool_id": pool_id}

    async def delete_pool(self, pool_id):
        self._call("delete_pool", pool_id=pool_id)
        self.pools[pool_id] = "archived"
        return {"status": "archived"}

    async def deployment_operation(self, pool_id, task_id, *, project_id, operation,
                                   idempotency_key, expected_revision=None, payload=None):
        self._call(operation + "_deployment", key=idempotency_key, payload=payload)
        if operation == "create":
            self.deployments[(pool_id, task_id)] = "r1"
            return {"status": "completed", "revision": "r1"}
        assert self.deployments.pop((pool_id, task_id)) == expected_revision
        return {"status": "completed", "provider_cleanup": {"status": "confirmed_removed"}}

    async def get_deployment(self, pool_id, task_id, *, project_id):
        if (pool_id, task_id) not in self.deployments:
            raise RhodesRequestError(404, "not_found", None)
        return {"revision": self.deployments[(pool_id, task_id)]}

    async def task_readiness(self, pool_id, task_id):
        self._call("readiness")
        if self.not_ready > 0:
            self.not_ready -= 1
            raise RhodesRequestError(409, "rhodes.readiness.health_refused", None)
        return {"instance_id": "i-" + task_id, "service_url": "http://127.0.0.1:1"}

    async def acquire_observed_lease(self, pool_id, task_id, *, image_kind, idempotency_key,
                                     project_id, ttl_seconds):
        self._call("acquire_lease", key=idempotency_key)
        lease_id = "lease-" + task_id
        self.leases[lease_id] = "active"
        return {"lease_id": lease_id, "reused": False}

    async def release_lease(self, lease_id):
        self._call("release_lease")
        self.leases[lease_id] = "released"
        return {"lease_id": lease_id}

    async def submit(self, pool_id, payload):
        self._call("submit", payload=payload)
        rollout_id = "run-" + str(payload["seed"])
        self.rollouts[rollout_id] = "running"
        return rollout_id

    async def follow_events(self, rollout_id, *, timeout_seconds):
        yield {"sequence": 1, "type": "started"}
        waited = 0.0
        while self.rollouts[rollout_id] == "running" and waited < self.rollout_seconds:
            await asyncio.sleep(0.01)
            waited += 0.01
        if self.rollouts[rollout_id] == "running":
            self.rollouts[rollout_id] = "completed"
        yield {"sequence": 2, "type": self.rollouts[rollout_id]}

    async def get_rollout(self, rollout_id):
        status = self.rollouts[rollout_id]
        return {"status": status, "reward_status": "scored" if status == "completed" else None,
                "score": 1.0}

    async def usage(self, rollout_id):
        return {"rollout_id": rollout_id, "settlement_status": "unresolved"}

    async def cancel(self, rollout_id):
        self.cancelled.append(rollout_id)
        self.rollouts[rollout_id] = "cancelled"
        return {"status": "cancelled"}

    async def wait_for(self, rollout_id, *, timeout_seconds):
        return None


def lifecycle(tmp_path, client, *extra):
    args = rhodes_lifecycle._parser().parse_args([
        "--backend-url", "http://rhodes.invalid", "--project-id", "project",
        "--ready-seconds", "5", "--progress-seconds", "60", *extra,
    ])
    out = tmp_path / "run"
    out.mkdir()
    return rhodes_lifecycle.Lifecycle(client, args, out)


def test_full_lifecycle_passes_and_tears_down_everything(tmp_path):
    client = FakeRhodes(not_ready_reads=0, rollout_seconds=0.2)
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "0-3", "--parallel", "2").run())
    assert report["passed"], report["checks"]
    assert report["observed_rollout_overlap"] == 2
    assert set(client.pools.values()) == {"archived"}
    assert client.deployments == {} and set(client.leases.values()) == {"released"}
    # One pool per lane; every seed gets its own fresh deployment and lease.
    assert sum(1 for name, _ in client.calls if name == "create_pool") == 2
    assert sum(1 for name, _ in client.calls if name == "create_deployment") == 4
    submitted = [d["payload"] for name, d in client.calls if name == "submit"]
    assert {p["seed"] for p in submitted} == {0, 1, 2, 3}
    assert all(p["capacity_lease_id"] == "lease-seed-" + str(p["seed"]) for p in submitted)
    assert all(p["policy_ref"] == {"harness": "scripted_react", "config": "engine_acceptance"}
               for p in submitted)
    summary = (tmp_path / "run" / "summary.txt").read_text()
    assert "passed=True" in summary
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert len(events) == 8


def test_idempotency_keys_are_stable_per_run_and_seed(tmp_path):
    client = FakeRhodes()
    driver = lifecycle(tmp_path, client, "--seeds", "5", "--parallel", "1")
    asyncio.run(driver.run())
    keys = [d["key"] for name, d in client.calls if "key" in d]
    assert keys == [
        "rhodes-lifecycle:run:deploy:5", "rhodes-lifecycle:run:lease:5",
        "rhodes-lifecycle:run:undeploy:5",
    ]


def test_lease_refusal_still_undeploys_and_fails_the_report(tmp_path):
    client = FakeRhodes(refuse={"acquire_lease": "rhodes.lease.capacity_held"})
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "0", "--parallel", "1").run())
    assert not report["passed"]
    assert not any(name == "submit" for name, _ in client.calls)
    seed = report["seeds"][0]
    assert seed["errors"] == [
        {"stage": "lease", "error": "rhodes.lease.capacity_held", "http_status": 409}]
    assert seed["deployment_deleted"] and seed["absent_after_delete"]
    assert client.pools == {"pool-0": "archived"}


def test_pool_refusal_reports_every_seed_without_deploying(tmp_path):
    client = FakeRhodes(refuse={"create_pool": "rhodes.project_surface_uncovered"})
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "0-1", "--parallel", "1").run())
    assert not report["passed"]
    assert [s["stage"] for s in report["seeds"]] == ["pool", "pool"]
    assert not any(name == "create_deployment" for name, _ in client.calls)


def test_readiness_waits_through_not_ready_but_stops_on_refusal(tmp_path):
    client = FakeRhodes(not_ready_reads=3)
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "0", "--parallel", "1").run())
    assert report["passed"]
    assert sum(1 for name, _ in client.calls if name == "readiness") == 4


def test_interrupt_cancels_only_owned_active_rollouts_then_tears_down(tmp_path):
    client = FakeRhodes(rollout_seconds=5)
    driver = lifecycle(tmp_path, client, "--seeds", "0-1", "--parallel", "2")

    async def run():
        task = asyncio.create_task(driver.run())
        async with asyncio.timeout(10):
            while len(client.rollouts) < 2:
                await asyncio.sleep(0.01)
        driver.interrupted = True
        await driver.cancel_active()
        return await task

    report = asyncio.run(run())
    assert sorted(client.cancelled) == ["run-0", "run-1"]
    assert report["checks"]["not_interrupted"] is False and not report["passed"]
    assert client.deployments == {} and set(client.pools.values()) == {"archived"}


class RegistryRhodes(FakeRhodes):
    def __init__(self, verdict, **kwargs):
        super().__init__(not_ready_reads=0, **kwargs)
        self.verdict = verdict

    async def import_environment(self, manifest):
        self._call("import", manifest=manifest)
        return {"release_id": "release-1"}

    async def environment_compatibility(self, release_id, *, provider):
        self._call("compatibility", release_id=release_id, provider=provider)
        return {"status": self.verdict, "provider": provider, "reasons": [], "validation": "not_run"}


def test_unqualified_release_never_runs_without_explicit_acceptance(tmp_path):
    client = RegistryRhodes("unqualified")
    report = asyncio.run(lifecycle(tmp_path, client, "--release-id", "release-1", "--seeds", "0").run())
    assert not report["passed"] and report["release_support"] == "unqualified"
    assert not any(name == "create_pool" for name, _ in client.calls)


def test_accepted_unqualified_import_deploys_the_registered_release(tmp_path):
    manifest = tmp_path / "release.json"
    manifest.write_text('{"schema_version":"fixture"}')
    client = RegistryRhodes("unqualified")
    report = asyncio.run(lifecycle(
        tmp_path, client, "--import", str(manifest), "--allow-unqualified", "--seeds", "0",
    ).run())
    assert report["passed"], report["checks"]
    # Acceptance does not upgrade the claim.
    assert report["release_support"] == "unqualified"
    assert ("import", {"manifest": {"schema_version": "fixture"}}) in client.calls
    payload = next(d["payload"] for name, d in client.calls if name == "create_deployment")
    assert payload["environment_release_id"] == "release-1"
    assert "image_ref" not in payload and "port" not in payload


class ValidatingRhodes(RegistryRhodes):
    def __init__(self, outcome):
        super().__init__("unqualified")
        self.outcome = outcome

    async def validate_environment(self, release_id, *, operation_id, case):
        self._call("validate", release_id=release_id, key=operation_id, case=case)
        return {"validation_id": "val-1", "state": "pending"}

    async def wait_for_validation(self, release_id, validation_id, *, timeout_seconds):
        if self.outcome == "passed":
            self.verdict = "supported"
        return {"validation_id": validation_id, "state": self.outcome}


def test_validate_qualifies_the_release_before_any_pool(tmp_path):
    client = ValidatingRhodes("passed")
    report = asyncio.run(lifecycle(
        tmp_path, client, "--release-id", "release-1", "--validate", "--seeds", "0",
    ).run())
    assert report["passed"] and report["release_support"] == "supported"
    names = [name for name, _ in client.calls]
    assert names.index("validate") < names.index("compatibility") < names.index("create_pool")
    validate = next(d for name, d in client.calls if name == "validate")
    assert validate["key"] == "rhodes-lifecycle:run:validate"
    assert validate["case"]["policy_ref"] == {"harness": "scripted_react", "config": "engine_acceptance"}


def test_failed_validation_leaves_release_unqualified_and_refuses(tmp_path):
    client = ValidatingRhodes("failed")
    report = asyncio.run(lifecycle(
        tmp_path, client, "--release-id", "release-1", "--validate", "--seeds", "0",
    ).run())
    assert not report["passed"] and report["registry"]["validation"]["state"] == "failed"
    assert not any(name == "create_pool" for name, _ in client.calls)


def test_validate_without_a_release_refuses_instead_of_ignoring_the_flag(tmp_path):
    client = FakeRhodes()
    report = asyncio.run(lifecycle(tmp_path, client, "--validate", "--seeds", "0").run())
    assert not report["passed"]
    assert report["registry"]["error"] == {"error": "rhodes.lifecycle.validate_requires_release"}
    assert client.calls == []


@pytest.mark.parametrize("verdict", ["unsupported", "blocked"])
def test_unsupported_or_blocked_release_refuses_even_when_accepting_unqualified(tmp_path, verdict):
    client = RegistryRhodes(verdict)
    report = asyncio.run(lifecycle(
        tmp_path, client, "--release-id", "release-1", "--allow-unqualified", "--seeds", "0",
    ).run())
    assert not report["passed"]
    assert not any(name == "create_pool" for name, _ in client.calls)


@pytest.mark.parametrize("value,expected", [
    ("0-3", [0, 1, 2, 3]), ("7", [7]), ("1,4-5", [1, 4, 5]),
])
def test_seed_ranges(value, expected):
    assert rhodes_lifecycle._seeds(value) == expected


@pytest.mark.parametrize("value", ["3-1", "-1", "1,1", ""])
def test_invalid_seed_ranges_refuse(value):
    with pytest.raises((ValueError, Exception)):
        rhodes_lifecycle._seeds(value)


def test_main_refuses_without_backend_and_project(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("SYNTH_BACKEND_URL", raising=False)
    monkeypatch.delenv("SYNTH_PROJECT_ID", raising=False)
    assert rhodes_lifecycle.main(["--out", str(tmp_path)]) == 2
    assert "backend_and_project_required" in capsys.readouterr().err
