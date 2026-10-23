"""The lifecycle driver reports observations and always tears down what it created."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

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
        self.kinds = {}
        self.leases = {}
        self.rollouts = {}
        self.submitted = {}
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
            self.kinds[(pool_id, task_id)] = set(payload.get("image_kinds") or ())
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
        # Mirrors observed_leases: the task must declare the leased kind.
        if image_kind not in self.kinds.get((pool_id, task_id), set()):
            raise RhodesRequestError(422, "rhodes.capacity.task_image_kind_mismatch", None)
        lease_id = "lease-" + task_id
        self.leases[lease_id] = "active"
        return {"lease_id": lease_id, "reused": False}

    async def release_lease(self, lease_id):
        self._call("release_lease")
        self.leases[lease_id] = "released"
        return {"lease_id": lease_id}

    async def submit(self, pool_id, payload):
        self._call("submit", payload=payload)
        # The original idempotency key attaches to the original attempt.
        if payload["idempotency_key"] in self.submitted:
            return self.submitted[payload["idempotency_key"]]
        rollout_id = "run-" + str(payload["seed"])
        self.submitted[payload["idempotency_key"]] = rollout_id
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
        # Real wire shape (Rhodes v0.24): the scored outcome is nested under "result".
        return {"status": status, "score": 1.0, "result": {
            "status": status, "reward_status": "scored" if status == "completed" else None},
            "metadata": {"native_service_trace": {
                "status": "committed", "publication_id": "pub-" + rollout_id,
                "archive_sha256": hashlib.sha256(self.archive(rollout_id)).hexdigest()}}}

    async def usage(self, rollout_id):
        return {"rollout_id": rollout_id, "usage": {"container_pool": {
            "observed_through": "2026-10-22T00:00:00+00:00",
            "receipts": [{"usage_fact_id": "fact-" + rollout_id, "duplicate": False}]}}}

    @staticmethod
    def archive(rollout_id):
        return ("trace " + rollout_id).encode()

    async def publication_asset(self, publication_id, asset):
        self._call("publication_asset", publication_id=publication_id, asset=asset)
        return self.archive(publication_id.removeprefix("pub-"))

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
    # The real inspector opens a Harbor bundle; the fake archive names its rollout.
    return rhodes_lifecycle.Lifecycle(
        client, args, out,
        inspect_trace=lambda path: SimpleNamespace(trace_id=path.read_bytes().decode()[6:]),
    )


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


class HealthBenchRhodes(ValidatingRhodes):
    """Canned native HTTP chat results; no model or provider is contacted."""

    async def get_rollout(self, rollout_id):
        result = await super().get_rollout(rollout_id)
        result["score"] = 0.75
        result["result"].update(
            reward=0.75,
            task_instance_id="healthbench2:" + rollout_id.removeprefix("run-"),
            policy_ref={"harness": "chat_completion", "config": "healthbench_canned"},
        )
        return result


def test_healthbench_http_release_preserves_seeds_and_chat_policy_through_cleanup(tmp_path):
    client = HealthBenchRhodes("passed")
    report = asyncio.run(lifecycle(
        tmp_path, client, "--release-id", "healthbench-release", "--validate",
        "--validation-seed", "2", "--image-kind", "healthbench_eval",
        "--harness", "chat_completion", "--policy-config", "healthbench_canned",
        "--seeds", "0-2", "--parallel", "1", "--max-steps", "1",
    ).run())
    assert report["passed"], report["checks"]
    validate = next(body for name, body in client.calls if name == "validate")
    assert validate["case"]["seed"] == 2
    assert validate["case"]["policy_ref"] == {
        "harness": "chat_completion", "config": "healthbench_canned",
    }
    submitted = [body["payload"] for name, body in client.calls if name == "submit"]
    assert {body["seed"] for body in submitted} == {0, 1, 2}
    assert all(body["policy_ref"] == validate["case"]["policy_ref"] for body in submitted)
    assert all(body["max_steps"] == 1 and "task" not in body for body in submitted)
    deployments = [body["payload"] for name, body in client.calls if name == "create_deployment"]
    assert all(body["environment_release_id"] == "healthbench-release" for body in deployments)
    assert all(body["image_kinds"] == ["healthbench_eval"] and "env_vars" not in body for body in deployments)
    assert [seed["score"] for seed in report["seeds"]] == [0.75, 0.75, 0.75]
    assert report["checks"]["trace_archives_reopen_after_cleanup"]
    names = [name for name, _ in client.calls]
    assert names.index("publication_asset") > max(i for i, name in enumerate(names) if name == "delete_pool")
    assert client.deployments == {} and set(client.leases.values()) == {"released"}


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


class ReplayedCreateRhodes(FakeRhodes):
    """The first create's response is lost; its replay meets the still-pending original."""

    def __init__(self, *, blocker=None, final_state="completed", **kwargs):
        super().__init__(not_ready_reads=0, **kwargs)
        self.blocker = blocker
        self.final_state = final_state
        self.intents = {}
        self.lookups = 0

    async def deployment_operation(self, pool_id, task_id, *, project_id, operation,
                                   idempotency_key, expected_revision=None, payload=None):
        if operation == "create" and idempotency_key not in self.intents:
            receipt = await super().deployment_operation(
                pool_id, task_id, project_id=project_id, operation=operation,
                idempotency_key=idempotency_key, payload=payload)
            self.intents[idempotency_key] = receipt
            raise RhodesRequestError(409, "DEPLOYMENT_RECOVERY_REQUIRED", None, {
                "operation_id": self.blocker or "op-" + idempotency_key})
        return await super().deployment_operation(
            pool_id, task_id, project_id=project_id, operation=operation,
            idempotency_key=idempotency_key, expected_revision=expected_revision, payload=payload)

    async def find_deployment_operation(self, pool_id, task_id, *, project_id, idempotency_key):
        self.lookups += 1
        state = "pending" if self.lookups == 1 else self.final_state
        return {"operation_id": "op-" + idempotency_key, "state": state,
                "result": self.intents[idempotency_key] if state == "completed" else None}


def test_replayed_create_is_reconciled_by_key_not_repeated(tmp_path):
    client = ReplayedCreateRhodes()
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "7", "--parallel", "1").run())
    assert report["passed"], report["checks"]
    # The provider effect ran once; the driver read the pending original back by key.
    assert sum(1 for name, _ in client.calls if name == "create_deployment") == 1
    assert client.lookups == 2
    assert client.deployments == {} and set(client.pools.values()) == {"archived"}


def test_foreign_blocker_or_recovery_required_stays_a_refusal(tmp_path):
    for client in (ReplayedCreateRhodes(blocker="op-someone-else"),
                   ReplayedCreateRhodes(final_state="recovery_required")):
        out = tmp_path / str(id(client))
        out.mkdir()
        report = asyncio.run(lifecycle(out, client, "--seeds", "7", "--parallel", "1").run())
        assert not report["passed"]
        (seed,) = report["seeds"]
        assert seed["errors"][0]["error"] == "DEPLOYMENT_RECOVERY_REQUIRED"
        assert seed["errors"][0]["operation_id"].startswith("op-")


def test_retained_evidence_is_read_back_after_all_cleanup(tmp_path):
    client = FakeRhodes(not_ready_reads=0)
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "0-1", "--parallel", "1").run())
    assert report["passed"], report["checks"]
    for name in ("replay_attaches_to_original", "results_retained_after_cleanup",
                 "usage_recorded_after_cleanup", "trace_archives_reopen_after_cleanup"):
        assert report["checks"][name] is True
    # Archives are fetched only after every pool is archived.
    names = [name for name, _ in client.calls]
    assert names.index("publication_asset") > max(
        i for i, name in enumerate(names) if name == "delete_pool")
    assert (tmp_path / "run" / "archives" / "seed1.zip").read_bytes() == b"trace run-1"


class RerunningRhodes(FakeRhodes):
    """A server that starts a new attempt for a replayed key and serves a foreign archive."""

    async def submit(self, pool_id, payload):
        key = payload["idempotency_key"]
        if key in self.submitted:
            return self.submitted[key] + "-again"
        return await super().submit(pool_id, payload)

    async def publication_asset(self, publication_id, asset):
        return b"tampered"


def test_replay_and_archive_mismatches_fail_the_run(tmp_path):
    client = RerunningRhodes(not_ready_reads=0)
    report = asyncio.run(lifecycle(tmp_path, client, "--seeds", "3", "--parallel", "1").run())
    assert not report["passed"]
    assert report["checks"]["replay_attaches_to_original"] is False
    assert report["checks"]["trace_archives_reopen_after_cleanup"] is False
    assert report["checks"]["results_retained_after_cleanup"] is True
    assert report["checks"]["usage_recorded_after_cleanup"] is True


def test_hosted_provider_and_limits_reach_pool_deploy_and_report(tmp_path):
    client = FakeRhodes(not_ready_reads=0, rollout_seconds=0.05)
    report = asyncio.run(lifecycle(
        tmp_path, client, "--seeds", "0-1", "--parallel", "2",
        "--provider", "daytona", "--cpu-cores", "1", "--memory-mb", "1024",
    ).run())
    assert report["passed"], report["checks"]
    assert report["provider"] == "daytona"
    pools = [d["body"] for name, d in client.calls if name == "create_pool"]
    assert {body["provider"] for body in pools} == {"daytona"}
    deploys = [d["payload"] for name, d in client.calls if name == "create_deployment"]
    assert all(p["provider"] == "daytona" for p in deploys)
    assert all(p["limits"] == {"cpu_cores": 1.0, "memory_mb": 1024} for p in deploys)


def test_docker_default_sends_no_limits(tmp_path):
    client = FakeRhodes(not_ready_reads=0, rollout_seconds=0.05)
    asyncio.run(lifecycle(tmp_path, client, "--seeds", "0", "--parallel", "1").run())
    deploys = [d["payload"] for name, d in client.calls if name == "create_deployment"]
    assert deploys and all(p["provider"] == "docker" and "limits" not in p for p in deploys)
