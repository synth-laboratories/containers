"""Run one DeepSWE task as a Rhodes Harbor rollout on Daytona and report what was observed.

The Rhodes backend (a local slot running ``HARBOR_EXECUTION_BACKEND=daytona``) builds a
Daytona snapshot from the task's pinned public image, runs the agent and the task's own
hidden-test verifier in a Daytona sandbox, and grades with the task's grader. The driver
then deletes the pool and checks, through the Daytona API, that every Harbor sandbox and
snapshot that appeared during the run is gone.

Agents: ``fixed_solution`` applies the task's reference patch (no model; qualifies
provisioning, grading and cleanup), ``codex`` runs a model through the backend's gateway.
DeepSWE's ``separate`` verifier maps to Rhodes' isolated verifier with patch-only handoff:
a fresh sandbox from the same snapshot receives the hidden tests and ``model.patch`` and
nothing else from the agent's sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile
import time
import tomllib
from typing import Any
from uuid import uuid4

from .pools import PoolClientError, PoolRolloutTimeout
from .rhodes_client import RhodesClient, RhodesRequestError

HARBOR_SNAPSHOT_PREFIX = "synth-pool-harbor-"
TERMINAL = {"completed", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _tar_b64(files: dict[str, bytes]) -> str:
    with io.BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(info, io.BytesIO(data))
        return _b64(buffer.getvalue())


@dataclass(frozen=True)
class DeepSweTask:
    slug: str
    image: str
    base_commit: str
    instruction: bytes
    tests: dict[str, bytes]
    solution_patch: bytes
    agent_timeout_s: int
    verifier_timeout_s: int
    cpus: int
    memory_mb: int
    storage_mb: int


def load_task(corpus: Path, pins: Path, slug: str) -> DeepSweTask:
    """Read one corpus task; its image is the digest pin the curated DeepSWE suite uses."""
    root = corpus / "tasks" / slug
    spec = tomllib.loads((root / "task.toml").read_text())
    pin = tomllib.loads(pins.read_text()).get(f"deepswe/{slug}") or {}
    image = str(pin.get("agent") or "")
    if "@sha256:" not in image:
        raise ValueError(f"no digest-pinned agent image for deepswe/{slug} in {pins}")
    tests = {
        str(path.relative_to(root / "tests")): path.read_bytes()
        for path in sorted((root / "tests").rglob("*"))
        # The verifier Dockerfile only bakes these files into the image; shared mode
        # injects them into /tests instead.
        if path.is_file() and path.name != "Dockerfile" and not path.name.startswith(".")
    }
    verifier = spec.get("verifier") or {}
    environment = verifier.get("environment") or {}
    return DeepSweTask(
        slug=slug,
        image=image,
        base_commit=str(spec["metadata"]["base_commit_hash"]),
        instruction=(root / "instruction.md").read_bytes(),
        tests=tests,
        solution_patch=(root / "solution" / "solution.patch").read_bytes(),
        agent_timeout_s=int(float((spec.get("agent") or {}).get("timeout_sec") or 5400)),
        verifier_timeout_s=int(float(verifier.get("timeout_sec") or 1800)),
        cpus=int(environment.get("cpus") or 2),
        memory_mb=int(environment.get("memory_mb") or 8192),
        storage_mb=int(environment.get("storage_mb") or 20480),
    )


def oracle_script(task: DeepSweTask) -> str:
    """The reference solution, committed and exported exactly as DeepSWE's collect step does."""
    patch = _b64(task.solution_patch)
    base = shlex.quote(task.base_commit)
    return f"""#!/bin/bash
set -euo pipefail
git config --global --add safe.directory /app
cd /app
echo {patch} | base64 -d > /tmp/solution.patch
git apply --whitespace=nowarn /tmp/solution.patch
git checkout -b feature/solution 2>/dev/null || true
git add -A
git -c user.name=oracle -c user.email=oracle@local commit -q --no-verify -m "Apply reference solution"
mkdir -p /logs/artifacts
git diff --binary {base} HEAD > /logs/artifacts/model.patch
"""


def task_payload(task: DeepSweTask, *, task_id: str, agent: dict[str, Any]) -> dict[str, Any]:
    native = {
        "instruction_b64": _b64(task.instruction),
        "tests_archive_b64": _tar_b64(task.tests),
        "setup_script_b64": None,
        "setup_timeout_s": 600,
        "agent_timeout_s": task.agent_timeout_s,
        "verifier_timeout_s": task.verifier_timeout_s,
        # DeepSWE runs the agent and verifier with no network.
        "allow_internet": False,
        "workdir": "/app",
        "agent_user": "",
        "verifier_user": "",
        "verifier_environment_mode": "isolated",
        "verifier_workspace_transfer": "model_patch",
        "collect_phase_logs": True,
    }
    return {
        "task_id": task_id,
        "context_tar_base64": _tar_b64({"Dockerfile": f"FROM {task.image}\n".encode()}),
        "entrypoint": "native_harbor_phases",
        "entrypoint_mode": "native_harbor",
        "artifact_exfil_command": "cd /app && git diff HEAD 2>/dev/null || true",
        "limits": {
            "cpu_cores": task.cpus,
            "memory_mb": task.memory_mb,
            "storage_mb": task.storage_mb,
            "gpu_count": 0,
            "timeout_s": task.agent_timeout_s + task.verifier_timeout_s + 300,
        },
        "harbor_native_phases": native,
        "harbor_agent": agent,
        "metadata": {
            "harbor_native_phases": native,
            "allow_internet": False,
            "deepswe_task": f"deepswe/{task.slug}",
        },
    }


class DaytonaObserver:
    """Read-only view of Harbor-owned Daytona resources, to prove teardown left none."""

    def __init__(self) -> None:
        from daytona_sdk import Daytona, DaytonaConfig

        self._sdk = Daytona(
            DaytonaConfig(
                api_key=os.environ["DAYTONA_API_KEY"],
                api_url=os.environ.get("DAYTONA_API_URL") or None,
            )
        )

    def sandboxes(self) -> set[str]:
        # The SDK's list() still calls the removed /sandbox/paginated route; read the
        # current /sandbox route raw, as backend Harbor recovery does.
        response = self._sdk._sandbox_api.list_sandboxes_without_preload_content(
            labels=json.dumps({"synth_subtype": "harbor"})
        )
        page = json.loads(response.data)
        if not isinstance(page, dict) or not isinstance(page.get("items"), list):
            raise ValueError("Daytona sandbox listing returned an invalid page")
        if page.get("nextCursor"):
            raise ValueError("Daytona sandbox listing is paginated; leak check would be partial")
        return {str(item["id"]) for item in page["items"] if isinstance(item, dict) and item.get("id")}

    def snapshots(self) -> set[str]:
        names: set[str] = set()
        page = 1
        while True:
            listed = self._sdk.snapshot.list(page=page, limit=100)
            items = list(getattr(listed, "items", None) or [])
            names.update(
                str(item.name) for item in items if str(getattr(item, "name", "")).startswith(HARBOR_SNAPSHOT_PREFIX)
            )
            if len(items) < 100:
                return names
            page += 1


class Run:
    def __init__(self, client: RhodesClient, observer: DaytonaObserver, args: argparse.Namespace, out: Path) -> None:
        self.client, self.observer, self.args, self.out = client, observer, args, out
        self.report: dict[str, Any] = {"run_id": out.name, "started_at": _now(), "steps": [], "errors": []}

    def log(self, message: str, **fields: Any) -> None:
        print(f"{_now()} {message} {json.dumps(fields, sort_keys=True) if fields else ''}".rstrip(), flush=True)
        self.report["steps"].append({"at": _now(), "step": message, **fields})

    def save(self) -> None:
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, sort_keys=True) + "\n")

    async def execute(self) -> dict[str, Any]:
        args = self.args
        task = load_task(args.deepswe_root, args.pins, args.task)
        if args.agent == "fixed_solution":
            agent = {"name": "fixed_solution", "script_b64": _b64(oracle_script(task).encode())}
        else:
            agent = {"name": "codex", "model_name": args.model}
        before_sandboxes = await asyncio.to_thread(self.observer.sandboxes)
        before_snapshots = await asyncio.to_thread(self.observer.snapshots)
        self.report.update(task=f"deepswe/{task.slug}", image=task.image, agent=args.agent)
        pool_id = None
        try:
            pool = await self.client.create_pool(name=f"deepswe-daytona-{self.out.name[-8:]}", backend="harbor")
            pool_id = str(pool.get("id") or pool.get("pool_id"))
            self.log("pool", pool_id=pool_id)
            started = time.monotonic()
            created = await self.client.create_task(
                pool_id, task_payload(task, task_id=task.slug, agent=agent)
            )
            self.report["snapshot_build_seconds"] = round(time.monotonic() - started, 1)
            self.log("task", task_id=task.slug, build_seconds=self.report["snapshot_build_seconds"],
                     status=created.get("status"))
            during_snapshots = await asyncio.to_thread(self.observer.snapshots)
            self.report["snapshots_created"] = sorted(during_snapshots - before_snapshots)
            submission = {
                "task_id": task.slug,
                "seed": 0,
                "idempotency_key": f"deepswe-daytona-{self.out.name}",
                "mode": "queued",
                **({"policy": {"model": args.model}} if args.agent == "codex" else {}),
            }
            rollout_id = await self.client.submit(pool_id, submission)
            self.report["rollout_id"] = rollout_id
            self.log("rollout", rollout_id=rollout_id)
            started = time.monotonic()
            outcome = await self.client.wait_for(
                rollout_id, timeout_seconds=args.rollout_seconds, poll_interval_seconds=5
            )
            self.report["rollout_seconds"] = round(time.monotonic() - started, 1)
            body = outcome.payload
            (self.out / "rollout.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
            result = body.get("result") or {}
            metrics = result.get("metrics") or body.get("metrics") or {}
            self.report.update(
                status=outcome.status,
                reward=metrics.get("outcome_reward", body.get("score")),
                reward_details=(result.get("details") or {}).get("rewards"),
            )
            self.log("result", status=outcome.status, reward=self.report["reward"],
                     seconds=self.report["rollout_seconds"])
            # Exact replay attaches to the original attempt instead of running again.
            self.report["replay_rollout_id"] = await self.client.submit(pool_id, submission)
        except (RhodesRequestError, PoolClientError, PoolRolloutTimeout, ValueError) as error:
            self.report["errors"].append({"stage": self.report["steps"][-1]["step"] if self.report["steps"] else "start",
                                          "error": type(error).__name__, "detail": str(error)[:500]})
            self.log("error", error=type(error).__name__)
        finally:
            if pool_id:
                try:
                    await self.client.delete_pool(pool_id)
                    self.report["pool_deleted"] = True
                except (RhodesRequestError, PoolClientError) as error:
                    self.report["pool_deleted"] = False
                    self.report["errors"].append({"stage": "delete_pool", "detail": str(error)[:300]})
            self.report["leaked_sandboxes"], self.report["leaked_snapshots"] = await self._leaks(
                before_sandboxes, before_snapshots
            )
        return self._finish()

    async def _leaks(self, sandboxes: set[str], snapshots: set[str]) -> tuple[list[str], list[str]]:
        """Daytona deletes asynchronously; allow the cleanup bound before calling it a leak."""
        deadline = time.monotonic() + self.args.cleanup_seconds
        while True:
            left_sandboxes = sorted(await asyncio.to_thread(self.observer.sandboxes) - sandboxes)
            left_snapshots = sorted(await asyncio.to_thread(self.observer.snapshots) - snapshots)
            if not (left_sandboxes or left_snapshots) or time.monotonic() >= deadline:
                return left_sandboxes, left_snapshots
            await asyncio.sleep(10)

    def _finish(self) -> dict[str, Any]:
        r = self.report
        expected = 1.0 if self.args.agent == "fixed_solution" else None
        r["checks"] = {
            "rollout_completed": r.get("status") == "completed",
            "graded": r.get("reward") is not None,
            # The reference solution must pass the task's own hidden tests.
            **({"oracle_scores_one": r.get("reward") == expected} if expected is not None else {}),
            "replay_attaches_to_original": r.get("replay_rollout_id") == r.get("rollout_id") is not None,
            "snapshot_built_on_daytona": bool(r.get("snapshots_created")),
            "pool_deleted": r.get("pool_deleted") is True,
            "no_daytona_sandbox_left": r.get("leaked_sandboxes") == [],
            "no_daytona_snapshot_left": r.get("leaked_snapshots") == [],
        }
        r["passed"] = all(r["checks"].values()) and not r["errors"]
        r["finished_at"] = _now()
        self.save()
        print(f"{_now()} SUMMARY run={r['run_id']} passed={r['passed']}", flush=True)
        for name, value in r["checks"].items():
            print(f"{_now()}   {name}={value}", flush=True)
        return r


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend-url", default=os.environ.get("SYNTH_BACKEND_URL"))
    parser.add_argument("--deepswe-root", type=Path, default=Path.home() / "GitHub/deep-swe")
    parser.add_argument("--pins", type=Path, required=True, help="harbor-deepswe pins.toml")
    parser.add_argument("--task", default="anko-default-function-arguments")
    parser.add_argument("--agent", choices=["fixed_solution", "codex"], default="fixed_solution")
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--rollout-seconds", type=float, default=3600)
    parser.add_argument("--cleanup-seconds", type=float, default=180)
    parser.add_argument("--out", type=Path, default=Path(".local/rhodes-harbor-daytona"))
    args = parser.parse_args()
    if not args.backend_url or not os.environ.get("SYNTH_API_KEY") or not os.environ.get("DAYTONA_API_KEY"):
        print('{"error":"rhodes.harbor_daytona.backend_api_key_and_daytona_key_required"}', file=sys.stderr)
        return 2
    out = args.out / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    out.mkdir(parents=True)

    async def go() -> dict[str, Any]:
        # Pool task creation builds the Daytona snapshot inside the request.
        async with RhodesClient(
            api_key=os.environ["SYNTH_API_KEY"], backend_url=args.backend_url, timeout_seconds=1800
        ) as client:
            return await Run(client, DaytonaObserver(), args, out).execute()

    report = asyncio.run(go())
    print(f"report: {out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
