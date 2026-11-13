"""Core-owned Harbor launch and observation; no direct hosted pool mutations.

The parent Swarms run owns its pool, deployment, rollout and cleanup. All paid
work remains under its explicit budget and single-rollout resource grant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

import httpx

from .rhodes_deepswe_e2e import (
    EFFORT,
    MODEL,
    OWNER,
    TERMINAL,
    _error,
    _result,
    _write,
    checked_task,
    result_checks,
)
from .rhodes_harbor_daytona import task_payload
from .rhodes_client import _http_refusal


async def core_request(client, method: str, path: str, *, key: str | None = None, body=None):
    response = await client.request(
        method, path, json=body, headers={"Idempotency-Key": key} if key else {}
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        raise _http_refusal(error) from error
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("Core response must be an object")
    return value


async def upload_task(core, storage, content: bytes, *, key: str, name: str) -> dict[str, str]:
    digest = hashlib.sha256(content).hexdigest()
    prepared = await core_request(
        core,
        "POST",
        "/v1/uploads",
        key=key,
        body={
            "name": name,
            "media_type": "application/json",
            "size_bytes": len(content),
            "digest_sha256": digest,
        },
    )
    grant = prepared.get("put") or {}
    target = urlsplit(grant.get("url", ""))
    if target.scheme != "https" or not target.hostname or target.username or target.password:
        raise ValueError("Core upload requires a credential-free HTTPS target")
    headers = grant.get("headers")
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ValueError("Core upload headers are malformed")
    # A separate client carries neither the Core bearer nor its cookies.
    request = storage.build_request("PUT", grant["url"], content=content, headers=headers)
    for name in ("authorization", "x-api-key", "cookie", "proxy-authorization"):
        request.headers.pop(name, None)
    response = await storage.send(request, auth=None, follow_redirects=False)
    response.raise_for_status()
    finalized = await core_request(
        core, "POST", f"/v1/uploads/{prepared['upload_id']}/finalize", key=key + "-finalize"
    )
    file = finalized.get("file") or {}
    if file.get("digest_sha256") != digest or file.get("size_bytes") != len(content):
        raise ValueError("Core did not attest the uploaded task bytes")
    return {key: file[key] for key in ("file_id", "revision_id", "digest_sha256")}


def launch_request(file: dict, *, task_id: str, seed: int, budget: str, wall_seconds: int) -> dict:
    message = f"""Run exactly one native Harbor task as a provisioning and execution check.
Use one worker. Do not implement or solve the task in the worker: the immutable
attachment contains its configured Codex {MODEL} agent, reasoning_effort={EFFORT},
and native isolated verifier. Do not edit the attachment, add an oracle, or use
the generic command_job interface. The worker must use the available pool tools:
1. pool_create with name harbor-check and image_kind deepswe.
2. Find the attached JSON task in context.project.files (public file reference
{file["file_id"]}). Call pool_deployment_create with that pool_id,
task_id={task_id}, container_subtype=harbor and harbor_task_file_id equal to the
attachment's UUID file_id from context.project.files (select by name {task_id}.json,
not by the public file reference above). No interface_mode,
image_ref, source_files or other deployment fields.
3. Poll pool_deployment_get until ready or failed. A failed build is a failure;
do not create another deployment, pool or rollout.
4. Call pool_rollout_create exactly once with that pool_id, task_id={task_id},
seed={seed}, timeout_seconds=3600 and inputs={{}}. Its native agent runs the task.
5. Poll pool_rollout_get for the SAME rollout_id until terminal. Report the
actual rollout_id, pool_id, native reward, phase exit codes and any refusal.
6. Record the summary and finish this workflow. Leave the pool to the parent
run's Core-owned termination/archival path, so the external observer can read
its exact snapshot before cleanup. Do not retry or create replacement resources.
The native verifier's score is the result; do not infer success from a tool's HTTP status.
"""
    return {
        "profile": "swarms-v2",
        "message": message,
        "inputs": {"files": [file]},
        "limits": {
            "maximum_charge": {"amount": budget, "currency": "USD"},
            "wall_time_seconds": wall_seconds,
            "parallel_workers": 1,
        },
        "routing": {"providers": {"only": ["openai"]}, "models": {"only": [MODEL]}},
        "resources": [
            {
                "plugin": "pool",
                "roles": ["worker"],
                "selection": {},
                "limits": {"rollouts": 1, "spend_micros": int(Decimal(budget) * 1_000_000)},
            }
        ],
    }


class CoreHarborRun:
    def __init__(self, core, rhodes, storage, observer, args, output):
        self.core, self.rhodes, self.storage, self.observer = core, rhodes, storage, observer
        self.args, self.output = args, output
        self.report: dict[str, Any] = {
            "passed": False,
            "checks": {},
            "errors": [],
            "timings_seconds": {},
        }
        self.body: dict[str, Any] = {}
        self.started = time.perf_counter()
        self.key = output.name
        self.before_snapshots: set[str] = set()

    def save(self):
        _write(self.output / "report.json", self.report)

    async def events(self, rollout_id):
        events, cursor = [], 0
        async with asyncio.timeout(self.args.cleanup_seconds):
            for _ in range(1000):
                page = await self.rhodes.event_page(rollout_id, after_sequence=cursor)
                rows = page.get("events") or []
                for row in rows:
                    if row.get("sequence") != cursor + 1:
                        raise ValueError("Rollout event history is not contiguous")
                    cursor += 1
                    events.append(row)
                if not page.get("has_more"):
                    if not any(
                        page.get(k)
                        for k in ("publication_pending", "cleanup_pending", "inference_pending")
                    ):
                        return events
                    await asyncio.sleep(self.args.poll_seconds)
                elif not rows:
                    raise ValueError("Rollout event pagination made no progress")
        raise ValueError("Rollout event history exceeds its bound")

    async def execute(self):
        stage = "preflight"
        t = time.perf_counter()
        agent = {"name": "codex", "model_name": MODEL, "kwargs": {"reasoning_effort": EFFORT}}
        try:
            task = checked_task(self.args)
            self.report["api_profile"] = await self.rhodes.negotiate_api(
                required_capabilities=["core-owned.harbor.execute"]
            )
            self.before_snapshots = await self.observer.snapshots()
            self.report["task"] = task.slug
            self.report["image"] = task.image
            self.report["timings_seconds"][stage] = time.perf_counter() - t
            self.save()
            stage = "upload_task"
            t = time.perf_counter()
            content = json.dumps(
                task_payload(task, task_id=task.slug, agent=agent), separators=(",", ":")
            ).encode()
            file = await upload_task(
                self.core, self.storage, content, key=self.key + "-upload", name=task.slug + ".json"
            )
            self.report["input_file"] = file
            self.report["timings_seconds"][stage] = time.perf_counter() - t
            self.save()
            stage = "launch_parent"
            t = time.perf_counter()
            request = launch_request(
                file,
                task_id=task.slug,
                seed=self.args.seed,
                budget=self.args.budget,
                wall_seconds=self.args.wall_seconds,
            )
            _write(self.output / "launch-request.json", request)
            # Durable Core idempotency key is retained before the mutation.
            self.report["launch_idempotency_key"] = self.key + "-launch"
            self.save()
            try:
                accepted = await core_request(
                    self.core,
                    "POST",
                    "/v1/runs",
                    key=self.report["launch_idempotency_key"],
                    body=request,
                )
            except httpx.TransportError:
                # Core durable-key reconciliation reattaches, never launches a second run.
                accepted = await core_request(
                    self.core,
                    "POST",
                    "/v1/runs",
                    key=self.report["launch_idempotency_key"],
                    body=request,
                )
            self.report["run_id"] = accepted["run_id"]
            self.report["timings_seconds"][stage] = time.perf_counter() - t
            self.save()
            stage = "observe"
            t = time.perf_counter()
            async with asyncio.timeout(self.args.wall_seconds + 60):
                while True:
                    run_id = self.report["run_id"]
                    snapshot = await core_request(self.core, "GET", f"/v1/runs/{run_id}")
                    report = await core_request(self.core, "GET", f"/v1/runs/{run_id}/report")
                    _write(self.output / "parent-snapshot.json", snapshot)
                    _write(self.output / "parent-report.json", report)
                    rows = (report.get("reward") or {}).get("rollouts") or []
                    if len(rows) > 1:
                        raise ValueError("Parent created more than the single authorized rollout")
                    if rows:
                        rid = rows[0]["rollout_id"]
                        self.report["rollout_id"] = rid
                        self.body = await self.rhodes.get_rollout(rid)
                        self.report["pool_id"] = self.body["pool_id"]
                        _, _, execution = _result(self.body)
                        snapshot_id = execution.get("snapshot_id")
                        if isinstance(snapshot_id, str) and not snapshot_id.startswith(
                            "image_ref:"
                        ):
                            if (
                                snapshot_id in self.before_snapshots
                                or snapshot_id in await self.observer.snapshots()
                            ):
                                self.report["checks"]["provider_snapshot_observed"] = True
                        self.save()
                        if self.body.get("status") in TERMINAL:
                            break
                    if snapshot.get("execution_closed") is True:
                        raise ValueError("Parent closed before a native Harbor result was observed")
                    await asyncio.sleep(self.args.poll_seconds)
            self.report["timings_seconds"][stage] = time.perf_counter() - t
            stage = "verify_result"
            t = time.perf_counter()
            events = await self.events(self.report["rollout_id"])
            self.report["checks"].update(result_checks(self.body, events, agent))
            self.report["checks"].setdefault("provider_snapshot_observed", False)
            _, metrics, execution = _result(self.body)
            self.report.update(reward=metrics.get("outcome_reward"), execution=execution)
            self.report["task_solved"] = self.report["reward"] == 1
            if self.args.require_solved:
                self.report["checks"]["task_solved"] = self.report["task_solved"]
            _write(self.output / "events.json", events)
            _write(self.output / "result.json", self.body)
            self.report["timings_seconds"][stage] = time.perf_counter() - t
        except (Exception, asyncio.CancelledError) as error:
            self.report["timings_seconds"][stage] = time.perf_counter() - t
            self.report["errors"].append({"stage": stage, **_error(error)})
        finally:
            t = time.perf_counter()
            try:
                async with asyncio.timeout(self.args.cleanup_seconds):
                    await self.cleanup()
            except (Exception, asyncio.CancelledError) as error:
                self.report["errors"].append({"stage": "cleanup", **_error(error)})
            self.report["timings_seconds"]["cleanup"] = time.perf_counter() - t
            self.report["timings_seconds"]["total"] = time.perf_counter() - self.started
            self.report["passed"] = (
                bool(self.report["checks"])
                and all(self.report["checks"].values())
                and not self.report["errors"]
            )
            self.save()
        return self.report

    async def cleanup(self):
        run_id = self.report.get("run_id")
        if not run_id:
            # A lost launch acknowledgement is unresolved, never a cleanup pass.
            self.report["checks"]["parent_not_created"] = (
                "launch_idempotency_key" not in self.report
            )
            return
        snapshot = await core_request(self.core, "GET", f"/v1/runs/{run_id}")
        if snapshot.get("execution_closed") is not True:
            await core_request(
                self.core, "POST", f"/v1/runs/{run_id}/cancel", key=self.key + "-cancel"
            )
        while True:
            snapshot = await core_request(self.core, "GET", f"/v1/runs/{run_id}")
            if all(
                snapshot.get(key) is True
                for key in ("execution_closed", "accounting_final", "cleanup_complete")
            ):
                break
            await asyncio.sleep(self.args.poll_seconds)
        self.report["checks"]["parent_financially_and_physically_closed"] = True
        final = await core_request(self.core, "GET", f"/v1/runs/{run_id}/report")
        while (final.get("settlement") or {}).get("status") != "settled":
            await asyncio.sleep(self.args.poll_seconds)
            final = await core_request(self.core, "GET", f"/v1/runs/{run_id}/report")
        self.report["settlement"] = final.get("settlement")
        self.report["checks"]["settlement_reported"] = True
        _write(self.output / "parent-report.json", final)
        if self.report.get("rollout_id"):
            while True:
                archive = await self.rhodes._request(
                    "GET", f"/pools/{self.report['pool_id']}/archive"
                )
                if (
                    archive.get("status") == "archived"
                    and (archive.get("archive") or {}).get("state") == "archived"
                ):
                    break
                await asyncio.sleep(self.args.poll_seconds)
            self.report["pool_archive"] = archive
            self.report["checks"]["owned_pool_archived"] = True
            self.body = await self.rhodes.get_rollout(self.report["rollout_id"])
            _, _, execution = _result(self.body)
            owners = [
                x["owner"]
                for x in execution.get("sandbox_cleanup") or []
                if OWNER.fullmatch(str(x.get("owner", "")))
            ]
            if len(owners) != 2:
                raise ValueError("Both native sandbox identities must be retained")
            while await self.observer.remaining_owners(owners):
                await asyncio.sleep(self.args.poll_seconds)
            self.report["checks"]["provider_sandboxes_absent"] = (
                self.report["checks"].get("provider_snapshot_observed") is True
            )
            snapshot_id = execution.get("snapshot_id")
            if not isinstance(snapshot_id, str) or not snapshot_id:
                raise ValueError("Native snapshot identity is missing")
            if (
                not snapshot_id.startswith("image_ref:")
                and snapshot_id not in self.before_snapshots
            ):
                while snapshot_id in await self.observer.snapshots():
                    await asyncio.sleep(self.args.poll_seconds)
            self.report["checks"]["owned_snapshot_absent_or_retained_shared"] = (
                self.report["checks"].get("provider_snapshot_observed") is True
            )
        else:
            self.report["checks"]["native_rollout_observed"] = False
