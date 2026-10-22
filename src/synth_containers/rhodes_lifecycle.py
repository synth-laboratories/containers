"""Drive one complete Rhodes lifecycle over the public API and report what was observed.

Spec: docs/specs/rhodes-lifecycle-driver-v1.md. Per seed: server-provisioned
deployment → observed readiness → exclusive capacity lease → admitted rollout →
committed events → results/usage → lease release → deployment delete → absence
readback. The observed-capacity profile excludes a whole pool while a lease is
held and never reuses a used instance, so each parallel lane owns one pool and
each seed gets a fresh deployment. Teardown always runs, including on failure or
interrupt, and its outcome is reported separately from execution.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable
from uuid import uuid4

from .platform.trace_bundle import inspect_harbor_trace_bundle
from .pools import PoolClientError, PoolRolloutTimeout
from .rhodes_client import RhodesClient, RhodesRequestError

TERMINAL = {"completed", "failed", "cancelled"}
# The server bounds one deployment execution at 120 s; read back a little longer.
DEPLOYMENT_RECONCILE_SECONDS = 150
# The cached native Craftax runtime used by the retained native cohort receipts.
DEFAULT_CRAFTAX_IMAGE = "sha256:8f33bf128a3c48a457e9357688b7a41288b4b4714722a6ad725cfdcefa59a329"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for part in value.split(","):
        start, _, end = part.strip().partition("-")
        first, last = int(start), int(end or start)
        if first < 0 or last < first:
            raise argparse.ArgumentTypeError("seeds must be N, N-M or a comma list")
        seeds.extend(range(first, last + 1))
    if not seeds or len(seeds) > 1000 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be 1-1000 distinct values")
    return seeds


def _manifest(value: str) -> dict[str, Any]:
    from .rhodes_cli import _payload

    try:
        return _payload(Path(value))
    except (ValueError, OSError) as error:
        raise argparse.ArgumentTypeError("import declaration must be a bounded JSON/TOML object") from error


def _refusal(error: BaseException) -> dict[str, Any]:
    if isinstance(error, RhodesRequestError):
        return {"error": error.code, "http_status": error.status_code, **error.identifiers}
    # Never echo bodies; they can carry credentials or terminal control bytes.
    return {"error": type(error).__name__}


@dataclass
class SeedRecord:
    seed: int
    lane: int
    task_id: str
    stage: str = "pending"
    deployment_created: bool = False
    deployment_revision: str | None = None
    instance_id: str | None = None
    lease_id: str | None = None
    lease_released: bool = False
    rollout_id: str | None = None
    status: str | None = None
    reward_status: str | None = None
    score: Any = None
    events: int = 0
    usage: dict[str, Any] | None = None
    started_at: float | None = None
    finished_at: float | None = None
    deployment_deleted: bool = False
    provider_cleanup: Any = None
    absent_after_delete: bool | None = None
    replay_rollout_id: str | None = None
    # Read back after every pool, lease and container is gone.
    retained: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)


class Lifecycle:
    def __init__(
        self,
        client: RhodesClient,
        args: argparse.Namespace,
        out: Path,
        *,
        inspect_trace: Callable[[Path], Any] = inspect_harbor_trace_bundle,
    ) -> None:
        self.client = client
        self.inspect_trace = inspect_trace
        self.args = args
        self.out = out
        self.run_id = out.name
        self.pools: dict[int, dict[str, Any]] = {}
        self.records: list[SeedRecord] = []
        self.release_id: str | None = args.release_id
        self.registry: dict[str, Any] = {}
        self.interrupted = False
        self._log = (out / "stream.log").open("a")
        self._events = (out / "events.jsonl").open("a")

    def log(self, message: str) -> None:
        line = f"{_now()} {message}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def _deployed_image(self) -> str:
        # A registered release pins its own image server-side; --image is unused then.
        return f"release:{self.release_id}" if self.release_id else self.args.image

    def key(self, *parts: object) -> str:
        return ":".join(["rhodes-lifecycle", self.run_id, *map(str, parts)])

    async def _deployment_operation(
        self, pool_id: str, task_id: str, *, operation: str, idempotency_key: str, **request: Any
    ) -> dict[str, Any]:
        """Run one deployment mutation, reconciling our own uncertain replay by its key.

        A transport retry of an identified mutation can reach the server while the
        first attempt still executes; the server then answers 409
        DEPLOYMENT_RECOVERY_REQUIRED naming that same operation. Read it back by
        key (never repeat the effect) until it completes, then use its retained
        receipt. Any other blocker, or an operation that needs recovery, stays a
        refusal.
        """
        try:
            return await self.client.deployment_operation(
                pool_id, task_id, project_id=self.args.project_id, operation=operation,
                idempotency_key=idempotency_key, **request,
            )
        except RhodesRequestError as refusal:
            if refusal.code != "DEPLOYMENT_RECOVERY_REQUIRED":
                raise
            deadline = time.monotonic() + DEPLOYMENT_RECONCILE_SECONDS
            while True:
                try:
                    intent = await self.client.find_deployment_operation(
                        pool_id, task_id, project_id=self.args.project_id,
                        idempotency_key=idempotency_key,
                    )
                except RhodesRequestError as lookup:
                    if lookup.status_code == 404:
                        raise refusal from None  # the blocker is not our operation
                    raise
                blocker = refusal.identifiers.get("operation_id")
                if blocker is not None and blocker != intent.get("operation_id"):
                    raise refusal
                if intent.get("state") == "completed" and isinstance(intent.get("result"), dict):
                    self.log(f"{task_id} {operation} reconciled {intent.get('operation_id')}")
                    return intent["result"]
                if intent.get("state") != "pending" or time.monotonic() >= deadline:
                    raise refusal
                await asyncio.sleep(2)

    async def _readiness(self, pool_id: str, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.ready_seconds
        while True:
            try:
                return await self.client.task_readiness(pool_id, task_id)
            except RhodesRequestError as error:
                # 409/503 are honest "not ready yet" answers; others are refusals.
                if error.status_code not in {409, 503} or time.monotonic() >= deadline:
                    raise
            await asyncio.sleep(1)

    async def _setup_pool(self, lane: int) -> dict[str, Any]:
        body = {
            "name": f"rhodes-lifecycle-{self.run_id}-lane{lane}",
            "backend": "arbitrary",
            "project_id": self.args.project_id,
            "provider": "docker",
            "metadata": {"rhodes_lifecycle_run": self.run_id, "lane": lane},
        }
        pool = await self.client.create_project_pool(body)
        pool_id = str(pool.get("pool_id") or pool.get("id") or "")
        if not pool_id:
            raise PoolClientError("Rhodes pool response lacks a pool identity")
        self.pools[lane] = {"pool_id": pool_id, "deleted": False}
        self.log(f"lane{lane} pool created {pool_id}")
        return self.pools[lane]

    async def _seed(self, pool_id: str, record: SeedRecord) -> None:
        args = self.args
        try:
            record.stage = "deploy"
            receipt = await self._deployment_operation(
                pool_id, record.task_id, operation="create",
                idempotency_key=self.key("deploy", record.seed),
                payload={
                    "provider": "docker",
                    "interface_mode": "synth_http",
                    # Observed leases admit only a kind the task itself declares.
                    "image_kinds": [args.image_kind],
                    # A registered release pins image, port and health path server-side.
                    **(
                        {"environment_release_id": self.release_id}
                        if self.release_id
                        else {"image_ref": args.image, "port": args.port}
                    ),
                    "metadata": {"rhodes_lifecycle_run": self.run_id, "seed": record.seed},
                },
            )
            record.deployment_created = True
            record.deployment_revision = receipt.get("revision")
            record.stage = "ready"
            observed = await self._readiness(pool_id, record.task_id)
            record.instance_id = observed.get("instance_id")
            record.stage = "lease"
            lease = await self.client.acquire_observed_lease(
                pool_id, record.task_id, image_kind=args.image_kind,
                idempotency_key=self.key("lease", record.seed), project_id=args.project_id,
                ttl_seconds=args.lease_ttl_seconds,
            )
            record.lease_id = lease["lease_id"]
            if self.interrupted:
                return
            record.stage = "rollout"
            record.started_at = time.monotonic()
            submission = {
                "idempotency_key": self.key("rollout", record.seed),
                "capacity_lease_id": record.lease_id,
                "task_id": record.task_id,
                "seed": record.seed,
                "max_steps": args.max_steps,
                "policy_ref": {"harness": args.harness, "config": args.policy_config},
            }
            record.rollout_id = await self.client.submit(pool_id, submission)
            self.log(f"lane{record.lane} seed{record.seed} rollout {record.rollout_id}")
            try:
                async for event in self.client.follow_events(
                    record.rollout_id, timeout_seconds=args.rollout_seconds
                ):
                    record.events += 1
                    self._events.write(json.dumps(
                        {"seed": record.seed, "rollout_id": record.rollout_id, "event": event},
                        sort_keys=True, default=str,
                    ) + "\n")
                    self._events.flush()
            except PoolRolloutTimeout:
                record.errors.append({"stage": "rollout", "error": "observation_timeout"})
            result = await self.client.get_rollout(record.rollout_id)
            record.finished_at = time.monotonic()
            record.status = result.get("status")
            # The scored outcome is nested; the top level carries only status/score.
            outcome = result.get("result")
            record.reward_status = (
                outcome.get("reward_status") if isinstance(outcome, dict) else None
            )
            record.score = result.get("score")
            (self.out / f"result-seed{record.seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
            )
            record.usage = await self.client.usage(record.rollout_id)
            # Exact replay: the original key must attach to the original attempt, not run again.
            record.stage = "replay"
            record.replay_rollout_id = await self.client.submit(pool_id, submission)
            record.stage = "done"
        except (RhodesRequestError, PoolClientError, ValueError, OSError) as error:
            record.errors.append({"stage": record.stage, **_refusal(error)})
            self.log(f"lane{record.lane} seed{record.seed} {record.stage} refused {_refusal(error)}")
        finally:
            await self._teardown_seed(pool_id, record)

    async def _teardown_seed(self, pool_id: str, record: SeedRecord) -> None:
        args = self.args
        if record.rollout_id and record.status not in TERMINAL:
            try:
                await self.client.cancel(record.rollout_id)
                await self.client.wait_for(record.rollout_id, timeout_seconds=120)
            except (RhodesRequestError, PoolClientError) as error:
                record.errors.append({"stage": "cancel", **_refusal(error)})
        if record.lease_id:
            try:
                await self.client.release_lease(record.lease_id)
                record.lease_released = True
            except (RhodesRequestError, PoolClientError) as error:
                record.errors.append({"stage": "release", **_refusal(error)})
        if not record.deployment_created:
            return
        try:
            current = await self.client.get_deployment(
                pool_id, record.task_id, project_id=args.project_id
            )
            receipt = await self._deployment_operation(
                pool_id, record.task_id, operation="delete",
                idempotency_key=self.key("undeploy", record.seed),
                expected_revision=current["revision"],
            )
            record.deployment_deleted = receipt.get("status") == "completed"
            record.provider_cleanup = receipt.get("provider_cleanup")
        except (RhodesRequestError, PoolClientError, KeyError) as error:
            record.errors.append({"stage": "undeploy", **_refusal(error)})
            return
        try:
            await self.client.get_deployment(pool_id, record.task_id, project_id=args.project_id)
            record.absent_after_delete = False
        except RhodesRequestError as error:
            record.absent_after_delete = error.status_code == 404

    async def _lane(self, lane: int, seeds: list[int]) -> None:
        try:
            pool = await self._setup_pool(lane)
        except (RhodesRequestError, PoolClientError, ValueError) as error:
            self.log(f"lane{lane} pool refused {_refusal(error)}")
            for seed in seeds:
                record = SeedRecord(seed, lane, f"seed-{seed}", stage="pool")
                record.errors.append({"stage": "pool", **_refusal(error)})
                self.records.append(record)
            return
        try:
            for seed in seeds:
                record = SeedRecord(seed, lane, f"seed-{seed}")
                self.records.append(record)
                if self.interrupted:
                    record.errors.append({"stage": "pending", "error": "interrupted"})
                    continue
                await self._seed(pool["pool_id"], record)
        finally:
            try:
                await self.client.delete_pool(pool["pool_id"])
                pool["deleted"] = True
            except (RhodesRequestError, PoolClientError) as error:
                pool["delete_error"] = _refusal(error)

    async def cancel_active(self) -> None:
        """Cancel only rollouts this run admitted; their followers then end at terminal."""
        for record in list(self.records):
            if record.rollout_id and record.stage == "rollout":
                try:
                    await self.client.cancel(record.rollout_id)
                except (RhodesRequestError, PoolClientError) as error:
                    record.errors.append({"stage": "cancel", **_refusal(error)})

    async def _progress(self) -> None:
        while True:
            await asyncio.sleep(self.args.progress_seconds)
            stages: dict[str, int] = {}
            for record in self.records:
                stages[record.stage] = stages.get(record.stage, 0) + 1
            active = sum(1 for r in self.records if r.stage == "rollout")
            self.log(f"PROGRESS stages={dict(sorted(stages.items()))} active_rollouts={active}")

    async def _registry(self) -> bool:
        """Import or select a release and require an explicit support verdict first."""
        args = self.args
        if args.import_manifest is None and self.release_id is None:
            self.registry = {"mode": "image_digest", "image": args.image}
            if args.validate:
                # Only a registered release has an identity a validation can qualify.
                self.registry["error"] = {"error": "rhodes.lifecycle.validate_requires_release"}
                self.log("registry refused: --validate needs --release-id or --import")
                return False
            return True
        try:
            if args.import_manifest is not None:
                imported = await self.client.import_environment(dict(args.import_manifest))
                self.release_id = str(imported.get("release_id") or "")
                if not self.release_id:
                    raise PoolClientError("Rhodes import response lacks a release identity")
                self.registry["imported"] = True
                self.log(f"registry imported release {self.release_id}")
            assert self.release_id is not None
            if args.validate:
                started = await self.client.validate_environment(
                    self.release_id, operation_id=self.key("validate"), case={
                        "seed": args.validation_seed, "max_steps": args.max_steps,
                        "policy_ref": {"harness": args.harness, "config": args.policy_config},
                    },
                )
                self.log(f"registry validation {started['validation_id']} {started['state']}")
                finished = await self.client.wait_for_validation(
                    self.release_id, started["validation_id"],
                    timeout_seconds=args.validation_seconds,
                )
                self.registry["validation"] = finished
                self.log(f"registry validation {finished['state']}")
            verdict = await self.client.environment_compatibility(self.release_id, provider="docker")
        except (RhodesRequestError, PoolClientError, ValueError, PoolRolloutTimeout) as error:
            self.registry.update(mode="release", release_id=self.release_id, error=_refusal(error))
            self.log(f"registry refused {_refusal(error)}")
            return False
        self.registry.update(mode="release", release_id=self.release_id, compatibility=verdict)
        status = verdict.get("status")
        self.log(f"registry compatibility docker={status} validation={verdict.get('validation')}")
        # Unsupported/blocked never execute. Unqualified runs only when the caller
        # explicitly accepts it, and the report keeps saying unqualified.
        return status == "supported" or (status == "unqualified" and args.allow_unqualified)

    async def run(self) -> dict[str, Any]:
        seeds = self.args.seeds
        lanes = min(self.args.parallel, len(seeds))
        plan = {lane: seeds[lane::lanes] for lane in range(lanes)}
        self.log(
            f"START run={self.run_id} seeds={len(seeds)} lanes={lanes} image={self._deployed_image()}"
        )
        if not await self._registry():
            for seed in seeds:
                record = SeedRecord(seed, -1, f"seed-{seed}", stage="registry")
                record.errors.append({"stage": "registry", "error": "release_not_admitted"})
                self.records.append(record)
            return self.report(lanes)
        progress = asyncio.create_task(self._progress())
        try:
            await asyncio.gather(*(self._lane(lane, plan[lane]) for lane in plan))
        finally:
            progress.cancel()
        if not self.interrupted:
            await asyncio.gather(*(self._verify_retained(r) for r in self.records if r.rollout_id))
        return self.report(lanes)

    async def _verify_retained(self, record: SeedRecord) -> None:
        """After all owned capacity is gone, the result, usage and trace archive must remain."""
        assert record.rollout_id is not None
        original_path = self.out / f"result-seed{record.seed}.json"
        try:
            later = await self.client.get_rollout(record.rollout_id)
            original = json.loads(original_path.read_text()) if original_path.exists() else {}
            record.retained["result_unchanged"] = bool(original) and all(
                later.get(key) == original.get(key) for key in ("status", "score", "result")
            )
            usage = (await self.client.usage(record.rollout_id)).get("usage") or {}
            pool_usage = usage.get("container_pool") or {}
            receipts = pool_usage.get("receipts") or []
            before = ((record.usage or {}).get("usage") or {}).get("container_pool") or {}
            record.retained["usage_recorded"] = bool(
                receipts
                and pool_usage.get("observed_through")
                and not any(receipt.get("duplicate") for receipt in receipts)
                and {r.get("usage_fact_id") for r in receipts}
                == {r.get("usage_fact_id") for r in before.get("receipts") or []}
            )
            trace = (later.get("metadata") or {}).get("native_service_trace") or {}
            if trace.get("status") != "committed" or not trace.get("publication_id"):
                record.retained["archive_reopened"] = False
                record.errors.append({"stage": "retained", "error": "trace_not_committed"})
                return
            archive = await self.client.publication_asset(trace["publication_id"], "trace.zip")
            path = self.out / "archives" / f"seed{record.seed}.zip"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(archive)
            inspected = await asyncio.to_thread(self.inspect_trace, path)
            record.retained["archive_reopened"] = (
                hashlib.sha256(archive).hexdigest() == trace.get("archive_sha256")
                and getattr(inspected, "trace_id", None) == record.rollout_id
            )
        except (RhodesRequestError, PoolClientError, ValueError, OSError) as error:
            record.errors.append({"stage": "retained", **_refusal(error)})

    def report(self, lanes: int) -> dict[str, Any]:
        records = sorted(self.records, key=lambda r: r.seed)
        spans = [(r.started_at, r.finished_at) for r in records if r.started_at and r.finished_at]
        overlap = max(
            (sum(1 for s, e in spans if s <= point < e) for point, _ in spans), default=0
        )
        checks = {
            "all_seeds_completed": all(r.status == "completed" for r in records),
            "all_scored": all(r.reward_status == "scored" for r in records),
            "every_lease_released": all(r.lease_released for r in records if r.lease_id),
            "every_deployment_deleted": all(
                r.deployment_deleted and r.absent_after_delete for r in records
                if r.deployment_created
            ),
            "every_pool_deleted": all(p["deleted"] for p in self.pools.values()),
            "observed_overlap_reaches_lanes": overlap >= lanes,
            "replay_attaches_to_original": all(
                r.replay_rollout_id == r.rollout_id for r in records if r.status == "completed"
            ),
            # Read back after every lease, deployment and pool is gone.
            "results_retained_after_cleanup": all(
                r.retained.get("result_unchanged") for r in records if r.rollout_id
            ),
            "usage_recorded_after_cleanup": all(
                r.retained.get("usage_recorded") for r in records if r.rollout_id
            ),
            "trace_archives_reopen_after_cleanup": all(
                r.retained.get("archive_reopened") for r in records if r.rollout_id
            ),
            "not_interrupted": not self.interrupted,
        }
        report = {
            "run_id": self.run_id,
            "backend_url": self.args.backend_url,
            "image": self._deployed_image(),
            "lanes": lanes,
            "observed_rollout_overlap": overlap,
            "checks": checks,
            "passed": all(checks.values()),
            "pools": self.pools,
            "registry": self.registry,
            # Only a "supported" verdict is a support claim; unqualified stays unqualified.
            "release_support": (self.registry.get("compatibility") or {}).get("status"),
            "seeds": [asdict(r) for r in records],
            # This driver observes the public API only. It does not qualify
            # hosted providers, warm reset, money settlement or Swarms adoption.
            "scope": "public_api_lifecycle_observation",
        }
        (self.out / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
        )
        lines = [
            f"SUMMARY run={self.run_id} passed={report['passed']} overlap={overlap}/{lanes} "
            f"source={self.registry.get('mode')} release_support={report['release_support']}"
        ]
        lines += [f"  {name}={value}" for name, value in checks.items()]
        for r in records:
            lines.append(
                f"  seed{r.seed:<4} {str(r.status):<10} reward={r.reward_status} score={r.score} "
                f"events={r.events} lease_released={r.lease_released} "
                f"undeployed={r.deployment_deleted} absent={r.absent_after_delete} "
                f"replay_same={r.replay_rollout_id == r.rollout_id} "
                f"retained={sorted(k for k, v in r.retained.items() if v)}"
                + (f" errors={r.errors}" if r.errors else "")
            )
        (self.out / "summary.txt").write_text("\n".join(lines) + "\n")
        for line in lines:
            self.log(line)
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m synth_containers.rhodes_lifecycle")
    parser.add_argument("--backend-url", default=os.environ.get("SYNTH_BACKEND_URL"))
    parser.add_argument("--project-id", default=os.environ.get("SYNTH_PROJECT_ID"))
    parser.add_argument("--image", default=DEFAULT_CRAFTAX_IMAGE)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--release-id", help="registered environment release to deploy")
    source.add_argument(
        "--import", dest="import_manifest", type=_manifest,
        help="environment release import declaration (JSON/TOML) to register first",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="run one bounded server validation of the release before the verdict",
    )
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--validation-seconds", type=float, default=900)
    parser.add_argument(
        "--allow-unqualified", action="store_true",
        help="run an unqualified release; the report still records it as unqualified",
    )
    parser.add_argument("--image-kind", default="craftax_eval")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seeds", type=_seeds, default=_seeds("0-9"))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--harness", default="scripted_react")
    parser.add_argument("--policy-config", default="engine_acceptance")
    parser.add_argument("--ready-seconds", type=float, default=180)
    parser.add_argument("--rollout-seconds", type=float, default=1500)
    parser.add_argument("--lease-ttl-seconds", type=int, default=1800)
    parser.add_argument("--progress-seconds", type=float, default=15)
    parser.add_argument("--out", type=Path, default=Path("rhodes-runs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.backend_url or not args.project_id:
        print('{"error":"rhodes.lifecycle.backend_and_project_required"}', file=sys.stderr)
        return 2
    if not 1 <= args.parallel <= 32:
        print('{"error":"rhodes.lifecycle.parallel_out_of_range"}', file=sys.stderr)
        return 2
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"{stamp}-{uuid4().hex[:8]}"
    out.mkdir(parents=True)

    async def run() -> dict[str, Any]:
        client = RhodesClient(api_key=os.environ.get("SYNTH_API_KEY", ""), backend_url=args.backend_url)
        async with client:
            lifecycle = Lifecycle(client, args, out)
            loop = asyncio.get_running_loop()

            def interrupt() -> None:
                # First Ctrl-C: stop admitting, cancel this run's rollouts, tear down.
                lifecycle.interrupted = True
                lifecycle.log("INTERRUPT: cancelling owned rollouts and tearing down")
                loop.remove_signal_handler(signal.SIGINT)
                pending.add(loop.create_task(lifecycle.cancel_active()))

            pending: set[asyncio.Task] = set()

            loop.add_signal_handler(signal.SIGINT, interrupt)
            return await lifecycle.run()

    try:
        report = asyncio.run(run())
    except PoolClientError as error:
        print(json.dumps({"error": "rhodes.lifecycle.client_refused", **_refusal(error)}), file=sys.stderr)
        return 2
    print(f"report: {out / 'report.json'}")
    if report["checks"]["not_interrupted"] is False:
        return 130
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
