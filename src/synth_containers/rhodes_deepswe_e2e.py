"""Upload, run and tear down one DeepSWE/Codex Luna 6 low attempt on Rhodes.

See docs/rhodes-deepswe-e2e.md. Uses the public Rhodes API for all mutations;
Daytona is observed read-only. PASS describes the lifecycle, not a solved task.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .pools import PoolClientError, PoolRolloutTimeout
from .rhodes_client import RhodesClient, RhodesRequestError
from .rhodes_api_contract import APIContractError
from .rhodes_harbor_daytona import load_task, task_payload
from .tracing.capture.redaction import redact_payload

MODEL = "gpt-6-luna"
EFFORT = "low"
OWNER = re.compile(r"synth-harbor-[0-9a-f]{32}")
TERMINAL = {"completed", "failed", "cancelled"}


async def api_preflight(client: RhodesClient) -> dict[str, Any]:
    profile = await client.negotiate_api(required_capabilities=(
        "customer.pool.create", "customer.rollout.create",
    ))
    health = await client._client.get("/healthz")
    health.raise_for_status()
    identity = health.json()
    build = identity.get("build") or {}
    if identity.get("service") != "rhodes-api" or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(build.get("image_digest", ""))
    ):
        raise APIContractError("Endpoint does not identify the current Rhodes API")
    ready = await client._client.get("/readyz")
    if ready.status_code != 200:
        raise APIContractError("Rhodes control plane is not ready")
    return {
        "contract": profile,
        "service": identity["service"],
        "build": build,
        "requested_api_path": "/v1",
        "version_negotiation": "major_2",
        "control_plane_ready": True,
        "worker_ready": "not_observed",
    }


class ScopedDaytonaObserver:
    """Read only resources named by this attempt's backend cleanup receipts."""

    def __init__(self, client: Any):
        self.client = client

    async def snapshots(self) -> set[str]:
        names: set[str] = set()
        for page in range(1, 101):
            listed = await self.client.snapshot.list(page=page, limit=100)
            names.update(item.name for item in listed.items)
            if page >= listed.total_pages:
                return names
        raise ValueError("Snapshot inventory exceeds the bounded read")

    async def remaining_owners(self, owners: list[str]) -> list[str]:
        from daytona import ListSandboxesQuery

        remaining = []
        for owner in owners:
            if not OWNER.fullmatch(owner):
                raise ValueError("Invalid Harbor ownership receipt")
            async for sandbox in self.client.list(
                ListSandboxesQuery(labels={"ai.synth.harbor.owner": owner}, limit=2),
                request_timeout=10,
            ):
                if sandbox.labels.get("ai.synth.harbor.owner") != owner:
                    raise ValueError("Daytona listing returned a foreign owner")
                remaining.append(owner)
                break  # One live resource already proves cleanup is incomplete.
        return remaining


def _write(path: Path, body: Any) -> None:
    """Keep a private, atomically replaced receipt, including before mutations."""
    staging = path.with_suffix(path.suffix + ".tmp")
    with staging.open("w", encoding="utf-8") as stream:
        os.chmod(staging, 0o600)
        redacted, _ = redact_payload(_json_safe(body))
        json.dump(redacted, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)


def _json_safe(value: Any) -> Any:
    # Invalid provider numbers must fail qualification without preventing the
    # failure receipt itself from being saved or bypassing teardown.
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid": "nonfinite_number"}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _error(error: BaseException) -> dict[str, Any]:
    # Exception bodies can contain credentials, task content or provider URLs.
    return {
        "type": type(error).__name__,
        **(
            {"http_status": error.status_code, "code": error.code, "upstream": error.identifiers}
            if isinstance(error, RhodesRequestError)
            else {}
        ),
    }


def _result(body: dict[str, Any]) -> tuple[dict, dict, dict]:
    result = body.get("result") or {}
    metrics = result.get("metrics") or body.get("metrics") or {}
    execution = result.get("execution_metadata") or body.get("metadata") or {}
    return result, metrics, execution


def checked_task(args: argparse.Namespace):
    task = load_task(args.deepswe_root, args.pins, args.task)
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", task.image):
        raise ValueError("DeepSWE requires an immutable registry image digest")
    if "test.sh" not in task.tests:
        raise ValueError("DeepSWE requires its native verifier test.sh")
    if any(
        value <= 0
        for value in (
            task.cpus,
            task.memory_mb,
            task.storage_mb,
            task.agent_timeout_s,
            task.verifier_timeout_s,
        )
    ):
        raise ValueError("Task resources and phase ceilings must be positive")
    return task


def result_checks(body: dict, events: list[dict], requested_agent: dict) -> dict[str, bool]:
    """Require independent execution, grading, inference and resource facts."""
    result, metrics, execution = _result(body)
    details = metrics.get("details") or {}
    reward = metrics.get("outcome_reward")
    receipts = execution.get("sandbox_cleanup") or []
    launches = (body.get("execution") or {}).get("launches") or []
    phases = {
        e.get("phase")
        for e in events
        if e.get("event_type") == "rollout.phase"
        and e.get("state") == "exited"
        and type(e.get("exit_code")) is int
        and e["exit_code"] == 0
    }
    settled = [
        e
        for e in events
        if e.get("event_type") == "rollout.inference_accounting"
        and e.get("accounting_status") == "settled"
    ]
    metadata = body.get("metadata") or {}
    return {
        "rollout_completed": body.get("status") == "completed",
        "task_pins_luna6_low": requested_agent
        == {"name": "codex", "model_name": MODEL, "kwargs": {"reasoning_effort": EFFORT}},
        "backend_records_codex_luna6": metadata.get("agent_name") == "codex"
        and metadata.get("model_name") == MODEL,
        "inference_settled": bool(settled),
        "agent_exited_zero": type(details.get("agent_exit_code")) is int
        and details["agent_exit_code"] == 0,
        "verifier_exited_zero": type(details.get("verifier_exit_code")) is int
        and details["verifier_exit_code"] == 0,
        "native_reward_is_finite": isinstance(reward, (int, float))
        and not isinstance(reward, bool)
        and math.isfinite(reward),
        "isolated_verifier": details.get("verifier_environment_mode") == "isolated",
        "agent_and_verifier_launched_on_daytona": len(launches) == 2
        and {x.get("role") for x in launches} == {"agent", "verifier"}
        and all(x.get("provider") == "daytona" for x in launches),
        "agent_and_verifier_phases_observed": {"agent", "verification"} <= phases,
        "sandbox_cleanup_receipts_confirmed": len(receipts) == 2
        and {x.get("role") for x in receipts} == {"agent", "verifier"}
        and all(
            x.get("state") == "confirmed_absent" and OWNER.fullmatch(str(x.get("owner", "")))
            for x in receipts
        ),
        "native_success": result.get("success") is True,
    }


class E2ERun:
    def __init__(
        self,
        client: RhodesClient,
        observer: ScopedDaytonaObserver,
        args: argparse.Namespace,
        output: Path,
    ):
        self.client, self.observer, self.args, self.output = client, observer, args, output
        self.started = time.perf_counter()
        self.report: dict[str, Any] = {
            "schema": "synth.rhodes-deepswe-e2e.v1",
            "run_id": output.name,
            "pool_id": "pool_deepswe_" + uuid4().hex,
            "started_at": datetime.now(UTC).isoformat(),
            "model": MODEL,
            "reasoning_effort": EFFORT,
            "timings_seconds": {},
            "checks": {},
            "errors": [],
        }
        self.body: dict[str, Any] = {}
        self.submission: dict[str, Any] | None = None
        self.pool_attempted = False
        self.before_snapshots: set[str] = set()
        self.current_stage = "preflight"

    def save(self) -> None:
        _write(self.output / "report.json", self.report)

    @contextmanager
    def stage(self, name: str):
        self.current_stage = name
        print(f"{name} ...", flush=True)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = round(time.perf_counter() - started, 3)
            self.report["timings_seconds"][name] = elapsed
            self.save()
            print(f"{name}: {elapsed:.3f}s", flush=True)

    async def event_history(self, rollout_id: str) -> list[dict]:
        events: list[dict] = []
        cursor = 0
        for _ in range(100):
            page = await self.client.event_page(rollout_id, after_sequence=cursor, limit=1000)
            rows = page.get("events") or []
            for row in rows:
                sequence = row.get("sequence")
                if type(sequence) is not int or sequence != cursor + 1:
                    raise ValueError("Noncontiguous committed event history")
                if row.get("rollout_id", rollout_id) != rollout_id:
                    raise ValueError("Foreign rollout event")
                cursor = sequence
                events.append(row)
            if not page.get("has_more"):
                if not any(
                    page.get(key)
                    for key in ("publication_pending", "cleanup_pending", "inference_pending")
                ):
                    return events
                await asyncio.sleep(self.args.poll_seconds)
                continue
            if not rows:
                raise ValueError("Event pagination made no progress")
        raise ValueError("Event history exceeds the bounded read")

    async def execute(self) -> dict[str, Any]:
        """Persist identities first; always attempt owned cleanup after any failure."""
        self.save()
        try:
            with self.stage("preflight"):
                task = checked_task(self.args)
                self.report["api_profile"] = await api_preflight(self.client)
                self.before_snapshots = await self.observer.snapshots()
                self.report.update(task="deepswe/" + task.slug, image=task.image)
                agent = {
                    "name": "codex",
                    "model_name": MODEL,
                    "kwargs": {"reasoning_effort": EFFORT},
                }
                payload = task_payload(task, task_id=task.slug, agent=agent)
                self.report["agent_timeout_seconds"] = task.agent_timeout_s
                self.report["verifier_timeout_seconds"] = task.verifier_timeout_s
            with self.stage("create_pool"):
                self.pool_attempted = True
                # The API accepts caller pool_id; keeping it before POST permits
                # exact lookup/teardown after a lost creation acknowledgement.
                pool = await self.client._request(
                    "POST",
                    "/pools",
                    replay_safe=False,
                    payload={
                        "pool_id": self.report["pool_id"],
                        "name": self.output.name,
                        "backend": "harbor",
                        "compute_provider": "daytona",
                        "capacity": 1,
                        "concurrency": 1,
                        "metadata": {"e2e_run_id": self.output.name},
                        **({"project_id": self.args.project_id} if self.args.project_id else {}),
                    },
                )
                if (pool.get("pool_id") or pool.get("id")) != self.report["pool_id"]:
                    raise ValueError("Pool creation changed the retained identity")
            with self.stage("upload_and_build"):
                created = await self.client.create_task(self.report["pool_id"], payload)
                # Retain only safe preparation facts, never test bytes or credentials.
                self.report["deployment_id"] = created.get("deployment_id")
                configured = created.get("harbor_agent") or {}
                self.report["checks"]["task_configuration_retained"] = (
                    configured.get("name") == "codex"
                    and configured.get("model_name") == MODEL
                    and (configured.get("kwargs") or {}).get("reasoning_effort") == EFFORT
                    and created.get("lifecycle_profile") == "deepswe"
                )
            self.submission = {
                "task_id": task.slug,
                "seed": self.args.seed,
                "mode": "queued",
                "idempotency_key": self.output.name,
                "logical_intent_id": self.output.name,
                "policy": {"model": MODEL},
            }
            self.save()
            with self.stage("submit"):
                self.report["rollout_id"] = await self.client.submit(
                    self.report["pool_id"], self.submission
                )
            observation_started = time.perf_counter()
            observation_seconds = self.args.rollout_seconds or payload["limits"]["timeout_s"] + 600
            with self.stage("queue_wait"):
                while True:
                    observed = await self.client.get_rollout(self.report["rollout_id"])
                    if observed.get("status") != "queued":
                        break
                    if time.perf_counter() - observation_started >= min(
                        self.args.queue_seconds, observation_seconds
                    ):
                        raise PoolRolloutTimeout(
                            "Queued rollout did not start within the queue allowance"
                        )
                    await asyncio.sleep(self.args.poll_seconds)
            with self.stage("run_agent_and_verifier"):
                outcome = await self.client.wait_for(
                    self.report["rollout_id"],
                    timeout_seconds=max(
                        0.001, observation_seconds - (time.perf_counter() - observation_started)
                    ),
                    poll_interval_seconds=self.args.poll_seconds,
                )
                self.body = dict(outcome.payload)
            with self.stage("verify_result"):
                async with asyncio.timeout(self.args.cleanup_seconds):
                    events = await self.event_history(self.report["rollout_id"])
                self.report["checks"]["event_history_drained"] = True
                self.report["checks"].update(result_checks(self.body, events, agent))
                _, metrics, execution = _result(self.body)
                self.report.update(
                    status=self.body.get("status"),
                    reward=metrics.get("outcome_reward"),
                    execution=execution,
                )
                self.report["task_solved"] = self.report["reward"] == 1
                if self.args.require_solved:
                    self.report["checks"]["task_solved"] = self.report["task_solved"]
                # Events carry operator facts only; do not write raw native logs.
                _write(self.output / "events.json", events)
                _write(
                    self.output / "result.json",
                    {
                        "status": self.body.get("status"),
                        "metrics": metrics,
                        "execution": self.body.get("execution"),
                        "requested_agent": agent,
                    },
                )
                _write(
                    self.output / "usage.json", await self.client.usage(self.report["rollout_id"])
                )
                replay = await self.client.submit(self.report["pool_id"], self.submission)
                self.report["checks"]["replay_attaches_to_original"] = (
                    replay == self.report["rollout_id"]
                )
        except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 - retain failure and clean owned resources
            if (
                self.current_stage == "create_pool"
                and isinstance(error, RhodesRequestError)
                and error.status_code in {400, 401, 403, 404, 405, 406, 410, 415, 422}
            ):
                self.pool_attempted = False
                self.report["checks"]["pool_not_created"] = True
            self.report["errors"].append({"stage": self.current_stage, **_error(error)})
        finally:
            with self.stage("teardown"):
                try:
                    async with asyncio.timeout(self.args.cleanup_seconds):
                        await self.teardown()
                except (Exception, asyncio.CancelledError) as error:  # noqa: BLE001 - cleanup failure must produce a receipt
                    self.report["errors"].append({"stage": "teardown", **_error(error)})
                    self.report["checks"]["teardown_confirmed"] = False
        self.report["timings_seconds"]["total"] = round(time.perf_counter() - self.started, 3)
        self.report["finished_at"] = datetime.now(UTC).isoformat()
        self.report["passed"] = (
            bool(self.report["checks"])
            and all(self.report["checks"].values())
            and not self.report["errors"]
        )
        self.save()
        print("\n" + ("E2E PASS" if self.report["passed"] else "E2E FAIL"), flush=True)
        for name, passed in self.report["checks"].items():
            print(f"  {'PASS' if passed else 'FAIL'} {name}", flush=True)
        for error in self.report["errors"]:
            print(f"  ERROR {json.dumps(error, sort_keys=True)}", flush=True)
        print(f"  reward={self.report.get('reward')} task_solved={self.report.get('task_solved')}")
        print(f"  total={self.report['timings_seconds']['total']:.3f}s")
        print(f"  receipt={self.output / 'report.json'}")
        return self.report

    async def teardown(self) -> None:
        if not self.pool_attempted:
            self.report["checks"]["teardown_confirmed"] = True
            return
        pool_id = self.report["pool_id"]
        try:
            pool = await self.client.get_pool(pool_id)
        except RhodesRequestError as error:
            if error.status_code != 404:
                raise
            # A lost creation acknowledgement cannot be certified by absence of
            # an active pool; an archived row can also read as 404.
            raise ValueError("Owned pool creation/archival outcome is unavailable") from error
        if (pool.get("metadata") or {}).get("e2e_run_id") != self.output.name:
            raise ValueError("Pool ownership changed; refusing teardown")
        if self.submission is not None and not self.report.get("rollout_id"):
            # Retry the same service identity to recover a lost submit response.
            # This attaches to or accepts this one intended attempt, then cancels it.
            self.report["rollout_id"] = await self.client.submit(pool_id, self.submission)
            self.save()
        rollout_id = self.report.get("rollout_id")
        if rollout_id:
            self.body = await self.client.get_rollout(rollout_id)
            if self.body.get("status") not in TERMINAL:
                try:
                    await self.client.cancel(rollout_id)
                except RhodesRequestError as error:
                    if error.status_code != 409:
                        raise
            _, _, execution = _result(self.body)
            self.report["execution"] = execution
        snapshot = (self.report.get("execution") or {}).get("snapshot_id")
        predelete_snapshots = await self.observer.snapshots()
        self.report["checks"]["snapshot_observed_before_teardown"] = isinstance(snapshot, str) and (
            snapshot.startswith("image_ref:") or snapshot in predelete_snapshots
        )
        # Delete refuses while a running child still holds capacity. Retry only
        # the same owned pool, allowing its server-side recovery to finish.
        while True:
            try:
                deleted = await self.client.delete_pool(pool_id)
                if (deleted.get("provider_cleanup") or {}).get("complete") is True:
                    break
                await asyncio.sleep(self.args.poll_seconds)
            except RhodesRequestError as error:
                if error.status_code != 409:
                    raise
                await asyncio.sleep(self.args.poll_seconds)
        self.report["pool_delete"] = deleted
        self.report["checks"]["pool_archived"] = deleted.get("status") == "archived"
        if rollout_id:
            self.body = await self.client.get_rollout(rollout_id)
            _, _, execution = _result(self.body)
            self.report["execution"] = execution
        execution = self.report.get("execution") or {}
        owners = [r["owner"] for r in execution.get("sandbox_cleanup") or [] if r.get("owner")]
        snapshot = execution.get("snapshot_id")
        self.report["checks"]["snapshot_identity_retained"] = bool(snapshot)
        # Shared pre-existing snapshots are retained; only this run's newly
        # created exact snapshot is expected to disappear.
        direct_image = isinstance(snapshot, str) and snapshot.startswith("image_ref:")
        owned_snapshots = (
            {snapshot} - self.before_snapshots if snapshot and not direct_image else set()
        )
        self.report["snapshot_disposition"] = (
            "direct_image"
            if direct_image
            else "new"
            if owned_snapshots
            else "preexisting_or_unknown"
        )
        while True:
            remaining_owners = await self.observer.remaining_owners(owners)
            remaining_snapshots = owned_snapshots & await self.observer.snapshots()
            self.report["remaining_owners"] = remaining_owners
            self.report["remaining_snapshots"] = sorted(remaining_snapshots)
            self.save()
            if not remaining_owners and not remaining_snapshots:
                break
            await asyncio.sleep(self.args.poll_seconds)
        self.report["checks"]["owned_daytona_resources_absent"] = len(set(owners)) == 2
        self.report["checks"]["teardown_confirmed"] = (
            self.report["checks"]["pool_archived"]
            and self.report["checks"]["snapshot_identity_retained"]
            and self.report["checks"]["snapshot_observed_before_teardown"]
            and self.report["checks"]["owned_daytona_resources_absent"]
        )


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 86400:
        raise argparse.ArgumentTypeError("expected finite seconds in (0, 86400]")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend-url", default=os.environ.get("SYNTH_BACKEND_URL"))
    parser.add_argument("--project-id", default=os.environ.get("SYNTH_PROJECT_ID"))
    parser.add_argument("--core-url", default=os.environ.get("SYNTH_CORE_URL"), help="use Core-owned hosted Harbor execution")
    parser.add_argument("--budget", default="30", help="Core parent run maximum charge in USD")
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--deepswe-root", type=Path, required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--task", default="anko-default-function-arguments")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-seconds", type=_positive)
    parser.add_argument("--queue-seconds", type=_positive, default=120)
    parser.add_argument("--cleanup-seconds", type=_positive, default=300)
    parser.add_argument("--request-seconds", type=_positive, default=1800)
    parser.add_argument("--poll-seconds", type=_positive, default=5)
    parser.add_argument("--require-solved", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate inputs without network or credentials"
    )
    parser.add_argument("--out", type=Path, default=Path(".local/rhodes-deepswe-e2e"))
    args = parser.parse_args(argv)
    from decimal import Decimal, InvalidOperation
    try:
        budget = Decimal(args.budget)
        if not budget.is_finite() or not 0 < budget <= 39 or not 60 <= args.wall_seconds <= 3600:
            raise ValueError("Core limits out of range")
    except (InvalidOperation, ValueError):
        parser.error("--budget must be positive and at most 39 USD; --wall-seconds must be 60..3600")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.task):
        parser.error("--task must be a corpus task slug")
    try:
        task = checked_task(args)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "task": task.slug,
                        "image": task.image,
                        "agent": "codex",
                        "model": MODEL,
                        "reasoning_effort": EFFORT,
                        "provider": "daytona",
                        "verifier": "isolated_model_patch",
                        "agent_timeout_seconds": task.agent_timeout_s,
                        "verifier_timeout_seconds": task.verifier_timeout_s,
                        "network_requests": 0,
                    },
                    indent=2,
                )
            )
            return 0
        if (
            not args.backend_url
            or not os.environ.get("SYNTH_API_KEY")
            or not os.environ.get("DAYTONA_API_KEY")
        ):
            parser.error("set SYNTH_BACKEND_URL, SYNTH_API_KEY and DAYTONA_API_KEY")
        output = args.out / f"deepswe-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        output.mkdir(parents=True, mode=0o700)

        async def run():
            from importlib.metadata import version

            from daytona import AsyncDaytona, DaytonaConfig

            if version("daytona") != "0.210.0":
                raise ValueError("Provider observation requires Daytona 0.210.0")
            async with (
                RhodesClient(
                    api_key=os.environ["SYNTH_API_KEY"],
                    backend_url=args.backend_url,
                    timeout_seconds=args.request_seconds,
                ) as client,
                AsyncDaytona(
                    DaytonaConfig(
                        api_key=os.environ["DAYTONA_API_KEY"],
                        api_url=os.environ.get("DAYTONA_API_URL"),
                    )
                ) as provider,
            ):
                current = asyncio.current_task()
                loop = asyncio.get_running_loop()
                if current is not None:
                    loop.add_signal_handler(signal.SIGTERM, current.cancel)
                try:
                    if args.core_url:
                        import httpx
                        from .rhodes_core_harbor_e2e import CoreHarborRun
                        async with (
                            httpx.AsyncClient(base_url=args.core_url, timeout=60, headers={"Authorization": "Bearer " + os.environ["SYNTH_API_KEY"]}) as core,
                            httpx.AsyncClient(timeout=60) as storage,
                        ):
                            report = await CoreHarborRun(core, client, storage, ScopedDaytonaObserver(provider), args, output).execute()
                        print("E2E " + ("PASS" if report["passed"] else "FAIL"))
                        print(json.dumps({k: report.get(k) for k in ("run_id", "rollout_id", "reward", "timings_seconds", "checks", "errors")}, indent=2))
                        print("receipt=" + str(output / "report.json"))
                        return report
                    return await E2ERun(
                        client, ScopedDaytonaObserver(provider), args, output
                    ).execute()
                finally:
                    loop.remove_signal_handler(signal.SIGTERM)

        report = asyncio.run(run())
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, PoolClientError, ImportError) as error:
        print(json.dumps({"error": _error(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
