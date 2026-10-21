"""Portable foreground evaluation over existing native/retained-evidence adapters.

No hosted scheduler, provider grant or independent admission service lives here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, UTC
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import threading
import time
from uuid import uuid4

from .eval_models import EvaluationPlanV1, EvaluationRunDeclarationV1, CandidateBindingV1
from .gold_http import GoldHttpWorld
from .native_engine import owned_gold_engine
from .operator_journal import OperatorJournal
from .package_models import content_digest
from .tracing.atomic import durable_mkdir, durable_write
from .tracing.canonical import bytes_digest, utc_now
from .package_models import content_bytes as canonical_bytes
from .checkpoints.gold import commit_gold_checkpoint, restore_gold_checkpoint
from .checkpoints.portable import load_checkpoint_bundle
from .checkpoints.portable_models import CheckpointCommitAnchorV1


def local_execution_source_digest() -> str:
    root = Path(__file__).parent
    return content_digest({str(path.relative_to(root)): bytes_digest(path.read_bytes())
                           for path in sorted(root.rglob("*.py"))})


def local_execution_dependencies_digest() -> str:
    from importlib.metadata import version
    return content_digest({"python": platform.python_version(), "adapter": "gold-http-v1",
        "packages": {name: version(name) for name in ("anyio", "certifi", "pydantic", "fastapi",
            "starlette", "idna", "httpx", "uvicorn", "websockets", "zstandard")}})


def describe_evaluation(plan: EvaluationPlanV1, candidates: tuple[CandidateBindingV1, ...]) -> dict:
    blockers = []
    if plan.adapter == "rhodes-v1":
        blockers.append("rhodes_adapter_requires_service_admission: not installed in local CLI")
    if plan.adapter == "local-gold-scripted-v1":
        if plan.source_digest != local_execution_source_digest():
            blockers.append("execution_source_identity_mismatch")
        if plan.dependencies_digest != local_execution_dependencies_digest():
            blockers.append("execution_dependency_identity_mismatch")
        if not candidates or len({c.label for c in candidates}) != len(candidates):
            blockers.append("unique_candidate_bindings_required")
    return {"schema_version": "synth.evaluation-preflight.v1", "plan_digest": plan.digest,
        "plan": plan.model_dump(mode="json"), "candidate_bindings": [
            {"label": c.label, "digest": c.content.digest, "configuration": c.content.model_dump(mode="json"),
             "model": "scripted", "provider": "none", "effort": None} for c in candidates],
        "rollouts": len(candidates) * len(plan.seeds) * plan.replicates if candidates else 1,
        "aggregate_maximum_provider_cost_usd": "0", "supported": not blockers, "blockers": blockers,
        "capabilities": {"foreground": True, "detached": False, "watch": True,
            "strict_continue": plan.adapter == "local-gold-scripted-v1", "fork": plan.adapter == "local-gold-scripted-v1",
            "provider_calls": False, "credentials_consumed": [], "admission": "local-owner-resource-pins"}}


class EvaluationEvents:
    def __init__(self, root: Path, run_id: str, sink=None):
        self.journal = OperatorJournal(root / "events.jsonl", run_id=run_id)
        self.run_id, self.sink = run_id, sink
        self.lock = threading.RLock()

    def emit(self, kind: str, payload: dict, *, lane: str | None = None):
        with self.lock:
            event = self.journal.append(lambda seq: {"schema_version": "evals.event-stream.v1",
                "run_id": self.run_id, "seq": seq, "sequence": str(seq), "event_id": f"{self.run_id}:{seq}",
                "occurred_at": utc_now(), "kind": kind, "source": "local", "lane": lane, "payload": payload})
            if self.sink:
                try:
                    self.sink(event)
                except (BrokenPipeError, OSError):
                    self.sink = None
            return event


def read_actions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if len(data) > 256 * 1024 * 1024 or (data and not data.endswith(b"\n")):
        raise ValueError("action_journal_incomplete_or_oversized")
    rows, head = [], content_digest({"action_journal": "genesis-v1"})
    for line in data.splitlines():
        row = json.loads(line)
        actual = content_digest({key: value for key, value in row.items() if key != "chain_head"})
        if row["position"] != len(rows) + 1 or row["previous_head"] != head or row["chain_head"] != actual:
            raise ValueError("action_journal_integrity")
        rows.append(row)
        head = actual
    return rows


def _append_action(path: Path, row: dict):
    durable_mkdir(path.parent)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    from .tracing.atomic import sync_directory
    sync_directory(path.parent)


def _state_fact(result) -> dict:
    return {"observation": result.observation, "reward": result.reward, "done": result.done,
            "env_steps": result.env_steps, "valid_actions": result.valid_actions,
            "frame_digest": result.frame_digest}


def _latest_checkpoint(root: Path, rows: list[dict]) -> tuple[Path, CheckpointCommitAnchorV1]:
    valid = []
    for path in (root / "commits").glob("*.json"):
        commit = json.loads(path.read_bytes())
        relative = Path(commit["bundle"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint_commit_path_invalid")
        bundle = root / relative
        manifest = load_checkpoint_bundle(bundle).manifest
        anchor = CheckpointCommitAnchorV1.model_validate({k: commit[k] for k in ("manifest", "sequence", "chain_head")})
        if manifest.sequence > len(rows):
            raise ValueError("checkpoint_position_ahead_of_durable_history")
        expected_head = rows[manifest.sequence - 1]["chain_head"] if manifest.sequence else content_digest({"action_journal": "genesis-v1"})
        if manifest.digest() != anchor.manifest or manifest.chain_head != expected_head:
            raise ValueError("checkpoint_committed_prefix_mismatch")
        if not manifest.terminal:
            valid.append((manifest.sequence, bundle, anchor))
    if not valid:
        raise ValueError("strict_resume_requires_committed_nonterminal_checkpoint")
    latest = max(row[0] for row in valid)
    matches = [row for row in valid if row[0] == latest]
    if len({row[2].manifest for row in matches}) != 1:
        raise ValueError("strict_checkpoint_position_ambiguous")
    _, bundle, anchor = matches[0]
    return bundle, anchor


def _seal_lane_trace(directory: Path, declaration: EvaluationRunDeclarationV1, lane: str, rows: list[dict], status: str, attempt_id: str):
    from .tracing.capture.supervisor import CaptureSupervisor, SupervisorConfig
    from .tracing.models.identity import TraceProvenanceV5, TraceIdentityV5
    trace_root = directory / "traces" / attempt_id
    with CaptureSupervisor(SupervisorConfig(bundle_root=trace_root,
        trace_key={"run": declaration.run_id, "lane": lane, "attempt": attempt_id},
        upstream_base_url="http://127.0.0.1:9", interception="application",
        identity=TraceIdentityV5(run_id=declaration.run_id, rollout_id=lane),
        provenance=TraceProvenanceV5("containers-local-gold", "1", producer_commit=declaration.plan.source_digest,
            container_image_digest=declaration.plan.image_digest, config_digest=declaration.plan.digest),
        workload_kind="other")) as capture:
        original = (directory / "actions.jsonl").read_bytes() if (directory / "actions.jsonl").exists() else b""
        capture.collector.artifact(role="research_log", media_type="application/octet-stream",
            content=original, logical_name="native-actions-exact.jsonl", visibility="private")
        capture.collector.event(event_type="capture.projection", payload={
            "source_digest": bytes_digest(original), "native_schema": "synth.native-action-journal.v1",
            "projection_losses": ["V5 structured payloads omit null object fields; exact original journal bytes are retained"]})
        for row in rows:
            capture.collector.event(event_type="environment.action.executed", payload=row,
                                    occurred_at=row["occurred_at"])
        capture.finalize(status="completed" if status == "completed" else "interrupted")
    return str(trace_root.relative_to(directory))


def _run_lane(run: EvaluationRunDeclarationV1, candidate: CandidateBindingV1, seed: int, replicate: int,
              root: Path, url: str, events: EvaluationEvents, cancel: threading.Event, *, resume: bool):
    lane = f"{candidate.label}-s{seed}-r{replicate}"
    directory = root / lane
    durable_mkdir(directory)
    result_path = directory / "job_result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_bytes())
        if prior.get("execution_status") == "completed":
            return prior
        if not resume:
            raise ValueError("existing_partial_lane_requires_resume")
    rows = read_actions(directory / "actions.jsonl")
    attempt_id = str(uuid4())
    durable_write(directory / "attempts" / f"{attempt_id}.json", canonical_bytes({"attempt_id": attempt_id,
        "run_digest": run.digest, "started_at": utc_now(), "mode": "strict_continue" if resume else "fresh"}))
    calls = len(list((directory / "dispatches").glob("*.json")))
    active_ns = max((row["active_ns"] for row in rows), default=0)
    if resume and run.plan.limits.clock_policy == "active_runtime":
        settled = {json.loads(path.read_bytes())["attempt_id"]: json.loads(path.read_bytes())
                   for path in (directory / "results").glob("*.json")}
        starts = [json.loads(path.read_bytes())["attempt_id"] for path in (directory / "attempts").glob("*.json")
                  if not path.name.endswith("-error.json") and path.stem != attempt_id]
        if any(ident not in settled for ident in starts):
            raise ValueError("active_runtime_liability_unknown: owner reconciliation required or use absolute-deadline profile")
        active_ns = max([active_ns, *(int(row["duration_seconds"] * 1e9) for row in settled.values())])
    world = GoldHttpWorld(run.plan.limits.steps + 100, lambda _seed, maximum: {**run.plan.task,
        "max_steps": maximum}, base_url=url, engine="craftax")
    status, reason = "completed", "script_exhausted"
    started = time.monotonic_ns()
    current = None
    events.emit("eval.phase", {"phase": "running", "attempt_id": attempt_id}, lane=lane)
    try:
        current = world.reset(seed)
        if resume:
            bundle, anchor = _latest_checkpoint(directory / "checkpoints", rows)
            source_manifest = load_checkpoint_bundle(bundle).manifest
            mode = run.restore_mode if source_manifest.run_id != run.run_id else "strict_continue"
            manifest, policy, current = restore_gold_checkpoint(bundle, run, candidate.content, world=world,
                expected_commit=anchor, mode=mode, child_budget=run.child_budget if mode != "strict_continue" else None)
            if policy["history"] != [{"action": row["action"], "reward": row["reward"]} for row in rows[:manifest.sequence]]:
                raise ValueError("checkpoint_policy_history_mismatch")
            for row in rows[manifest.sequence:]:
                replay = world.step(row["action"])
                if content_digest(_state_fact(replay)) != row["state_digest"]:
                    raise ValueError("checkpoint_durable_suffix_mismatch")
                current = replay
            events.emit("eval.phase", {"phase": "restored", "checkpoint": manifest.digest(),
                "checkpoint_position": manifest.sequence, "reconciled_suffix_actions": len(rows) - manifest.sequence,
                "consumed_calls": calls, "active_ns_lower_bound": active_ns,
                "interrupted_attempt_time": "unknown; absolute deadline still applies"}, lane=lane)
        else:
            commit_gold_checkpoint(directory / "checkpoints", run, candidate.content, world=world,
                rows=rows, chain_head=content_digest({"action_journal": "genesis-v1"}), calls=calls, active_ns=active_ns, terminal=False)
        while True:
            elapsed = active_ns + time.monotonic_ns() - started
            if cancel.is_set() or (root / "cancel.json").exists():
                status, reason = "interrupted", "cancelled"
                break
            if run.plan.limits.clock_policy == "absolute_deadline":
                deadline_value = run.plan.limits.deadline_at
                assert deadline_value is not None
                deadline = datetime.fromisoformat(deadline_value.replace("Z", "+00:00"))
                if datetime.now(UTC) >= deadline:
                    reason = "absolute_deadline"
                    break
            if elapsed / 1e9 >= run.plan.limits.active_seconds:
                reason = "active_time_limit"
                break
            position = len(rows)
            if position >= run.plan.limits.steps:
                reason = "step_limit"
                break
            if calls >= run.plan.limits.calls:
                reason = "call_limit"
                break
            if current.done:
                reason = "environment_terminal"
                break
            if not candidate.content.repeat and position >= len(candidate.content.actions):
                break
            action = candidate.content.actions[position % len(candidate.content.actions)]
            if action not in current.valid_actions:
                raise ValueError("scripted_action_unsupported_by_native_world")
            calls += 1
            durable_write(directory / "dispatches" / f"{calls:08d}.json", canonical_bytes({"dispatch": calls,
                "action_position": position + 1, "action": action, "provider": "none", "cost_usd": "0"}))
            current = world.step(action)
            native_events = world.drain_native_events()
            fact = _state_fact(current)
            row = {"position": position + 1, "action": action, "reward": current.reward,
                "occurred_at": utc_now(), "state_digest": content_digest(fact), "state": fact,
                "native_events": native_events, "calls": calls,
                "active_ns": elapsed, "previous_head": rows[-1]["chain_head"] if rows else content_digest({"action_journal": "genesis-v1"})}
            row["chain_head"] = content_digest(row)
            _append_action(directory / "actions.jsonl", row)
            rows.append(row)
            events.emit("eval.usage.sampled", {"kind": "rollout.progress", "steps": len(rows),
                "step_limit": run.plan.limits.steps, "calls": calls, "call_limit": run.plan.limits.calls,
                "active_seconds": elapsed / 1e9, "time_limit_seconds": run.plan.limits.active_seconds,
                "provider_cost_usd": "0", "reward": current.reward, "native_state_digest": row["state_digest"]}, lane=lane)
            if len(rows) % run.plan.checkpoint_every_steps == 0:
                path = commit_gold_checkpoint(directory / "checkpoints", run, candidate.content, world=world,
                    rows=rows, chain_head=row["chain_head"], calls=calls, active_ns=elapsed, terminal=current.done)
                events.emit("eval.phase", {"phase": "checkpoint_committed", "position": len(rows),
                    "manifest_digest": load_checkpoint_bundle(path).manifest_digest}, lane=lane)
    except Exception as exc:
        status, reason = "error", type(exc).__name__
        durable_write(directory / "attempts" / f"{attempt_id}-error.json", canonical_bytes({"error_type": type(exc).__name__,
            "detail": str(exc) if isinstance(exc, ValueError) else "native execution failed"}))
    finally:
        world.close()
    elapsed = active_ns + time.monotonic_ns() - started
    trace = _seal_lane_trace(directory, run, lane, rows, status, attempt_id)
    rewards = [row["reward"] for row in rows]
    score = sum(rewards) if rewards and all(value is not None for value in rewards) else None
    result = {"schema_version": "synth.evaluation-result.v1", "run_id": run.run_id, "lane": lane,
        "plan_digest": run.plan.digest, "candidate_digest": candidate.content.digest, "seed": seed, "replicate": replicate,
        "attempt_id": attempt_id, "execution_status": status, "stop_reason": reason,
        "status": "evaluated" if status == "completed" else "interrupted" if status == "interrupted" else "error",
        "benchmark_status": "graded" if status == "completed" and score is not None else "unscored", "benchmark_score": score,
        "gates": {}, "duration_seconds": elapsed / 1e9, "exit_code": 0 if status == "completed" else 130 if status == "interrupted" else 1,
        "steps": len(rows), "calls": calls, "provider_calls": 0, "provider_cost_usd": "0", "trace_bundle": trace,
        "capture": "complete" if status == "completed" else "partial", "publication": "local-durable",
        "settlement": "no-provider-usage", "cleanup": "native-rollout-deleted"}
    durable_write(directory / "results" / f"{attempt_id}.json", canonical_bytes(result))
    durable_write(result_path, canonical_bytes(result), replace=True)
    events.emit("eval.run.terminal", result, lane=lane)
    return result


@contextmanager
def _owner_lock(root: Path):
    durable_mkdir(root)
    with (root / "owner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("evaluation_run_has_live_owner") from None
        yield


def execute_evaluation(run: EvaluationRunDeclarationV1, root: Path, *, resume: bool = False, sink=None) -> dict:
    preflight = describe_evaluation(run.plan, run.candidates)
    if not preflight["supported"]:
        raise ValueError(";".join(preflight["blockers"]))
    if run.plan.limits.clock_policy == "absolute_deadline":
        deadline = run.plan.limits.deadline_at
        assert deadline is not None
        if datetime.now(UTC) >= datetime.fromisoformat(deadline.replace("Z", "+00:00")):
            raise ValueError("execution_admission_deadline_expired")
    with _owner_lock(root):
        if resume and run.plan.adapter == "local-gold-scripted-v1":
            for candidate in run.candidates:
                for seed in run.plan.seeds:
                    for replicate in range(run.plan.replicates):
                        directory = root / f"{candidate.label}-s{seed}-r{replicate}"
                        result_path = directory / "job_result.json"
                        if result_path.exists() and json.loads(result_path.read_bytes()).get("execution_status") == "completed":
                            continue
                        _latest_checkpoint(directory / "checkpoints", read_actions(directory / "actions.jsonl"))
                        if run.plan.limits.clock_policy == "active_runtime":
                            settled = {json.loads(path.read_bytes())["attempt_id"] for path in (directory / "results").glob("*.json")}
                            starts = {json.loads(path.read_bytes())["attempt_id"] for path in (directory / "attempts").glob("*.json")
                                      if not path.name.endswith("-error.json")}
                            if starts - settled:
                                raise ValueError("active_runtime_liability_unknown: owner reconciliation required")
        durable_write(root / "run.json", canonical_bytes(run.model_dump(mode="json")))
        jobs = [{"name": f"{candidate.label}-s{seed}-r{replicate}", "bench": run.plan.benchmark,
                 "task": run.plan.task.get("task_id"), "lane": candidate.label}
                for candidate in run.candidates for seed in run.plan.seeds for replicate in range(run.plan.replicates)]
        durable_write(root / "matrix_run_manifest.json", canonical_bytes({"schema_version": "synth.evaluation-matrix.v1",
            "run_id": run.run_id, "jobs": jobs or [{"name": "grading", "bench": "retained-grading"}],
            "evals_authority": {"config_path": "run.json", "head_commit": run.plan.source_digest,
                                "working_tree_tracked_clean": None}, "plan": run.plan.model_dump(mode="json")}), replace=True)
        events = EvaluationEvents(root, run.run_id, sink)
        if resume:
            (root / "cancel.json").unlink(missing_ok=True)
        events.emit("eval.run.attached" if resume else "eval.run.submitted", {"plan_digest": run.plan.digest})
        if run.plan.adapter == "retained-grading-v1":
            return _execute_grading(run, root, events)
        cancel = threading.Event()
        results = []
        engine_receipt = root / "engine-receipts" / (str(uuid4()) + ".json")
        interrupted = False
        with owned_gold_engine(run.plan, owner=run.run_id, receipt=engine_receipt) as url:
            with ThreadPoolExecutor(max_workers=run.plan.parallelism) as pool:
                futures = [pool.submit(_run_lane, run, candidate, seed, replicate, root, url, events, cancel, resume=resume)
                    for candidate in run.candidates for seed in run.plan.seeds for replicate in range(run.plan.replicates)]
                try:
                    for future in as_completed(futures):
                        results.append(future.result())
                except KeyboardInterrupt:
                    interrupted = True
                    cancel.set()
                    results = [future.result() for future in futures]
        summary = {"schema_version": "synth.evaluation-summary.v1", "run_id": run.run_id,
            "declaration_digest": run.digest, "results": results, "execution": "interrupted" if interrupted or any(r["execution_status"] == "interrupted" for r in results) else "completed" if all(r["execution_status"] == "completed" for r in results) else "error",
            "cleanup": "owned-engine-stopped", "provider_calls": 0}
        durable_write(root / "summaries" / (str(uuid4()) + ".json"), canonical_bytes(summary))
        durable_write(root / "summary.json", canonical_bytes(summary), replace=True)
        events.emit("eval.run.terminal", {"execution": summary["execution"], "cleanup": summary["cleanup"]})
        return summary


def _execute_grading(run, root, events):
    from .grading import run_grading, export_grading
    from .grading_models import GradingPlanV1
    from .tracing.store.bundle import LocalTraceBundle
    from .tracing.validation.rehydrate import trace_document_from_payload
    plan = GradingPlanV1.model_validate_json(Path(run.plan.grading_plan).read_bytes())
    bundle = LocalTraceBundle(Path(run.plan.grading_bundle))
    document = trace_document_from_payload(bundle.read_trace(plan.trace_digest))
    result = run_grading(plan, document, bundle.blobs, root / "grades",
        trusted_source_digests=frozenset(run.plan.trusted_scorer_source_digests), source_bundle=bundle)
    output = root / "grading-output"
    if not output.exists():
        export_grading(root / "grades", result, output)
    wire = result.model_dump(mode="json")
    job = {"status": "evaluated" if result.execution_status == "completed" else "error",
        "benchmark_status": "passed" if result.mean_score == 1 else "failed" if result.mean_score is not None else "unavailable",
        "benchmark_score": result.mean_score, "gates": {}, "duration_seconds": result.elapsed_ns / 1e9 if result.elapsed_ns is not None else None,
        "exit_code": 0 if result.execution_status == "completed" else 1, "grading": wire}
    durable_write(root / "grading/job_result.json", canonical_bytes(job))
    summary = {"schema_version": "synth.evaluation-summary.v1", "run_id": run.run_id,
        "results": [job], "execution": result.execution_status, "cleanup": result.cleanup, "provider_calls": 0}
    durable_write(root / "summary.json", canonical_bytes(summary), replace=True)
    events.emit("eval.run.terminal", summary, lane="grading")
    return summary


def fork_evaluation(root: Path, destination: Path, *, run_id: str,
                    candidates: tuple[CandidateBindingV1, ...] | None = None, child_budget: dict) -> EvaluationRunDeclarationV1:
    """Fork at each lane's latest committed nonterminal cut; mutable state is copied."""
    with _owner_lock(root):
        parent = EvaluationRunDeclarationV1.model_validate_json((root / "run.json").read_bytes())
        if parent.plan.adapter != "local-gold-scripted-v1":
            raise ValueError("adapter_does_not_support_execution_forks")
        if not child_budget or "authority" not in child_budget or "reference" not in child_budget:
            raise ValueError("fork_requires_explicit_child_budget_authority")
        selected = candidates or parent.candidates
        if {c.label for c in selected} != {c.label for c in parent.candidates}:
            raise ValueError("fork_requires_same_participant_set")
        mode = "same_policy_fork" if selected == parent.candidates else "edited_policy_fork"
        child = EvaluationRunDeclarationV1(run_id=run_id, lineage_id=str(uuid4()), plan=parent.plan,
            candidates=selected, parent_run_id=parent.run_id, restore_mode=mode, child_budget=child_budget)
        if child.run_id == parent.run_id or destination.exists():
            raise ValueError("fork_requires_new_child_identity_and_fresh_destination")
        cuts = []
        for candidate in selected:
            for seed in parent.plan.seeds:
                for replicate in range(parent.plan.replicates):
                    lane = f"{candidate.label}-s{seed}-r{replicate}"
                    source = root / lane
                    rows = read_actions(source / "actions.jsonl")
                    bundle, anchor = _latest_checkpoint(source / "checkpoints", rows)
                    cuts.append((lane, source, rows[:anchor.sequence], bundle, anchor))
        durable_mkdir(destination)
        for lane, source, rows, bundle, anchor in cuts:
            target = destination / lane
            relative = bundle.relative_to(source / "checkpoints")
            shutil.copytree(bundle, target / "checkpoints" / relative)
            durable_write(target / "checkpoints/commits" / (anchor.manifest[7:] + ".json"),
                canonical_bytes({**anchor.model_dump(mode="json"), "bundle": str(relative)}))
            for row in rows:
                _append_action(target / "actions.jsonl", row)
            # Preserve inherited calls/liabilities; the child receives new
            # funding reference, never a reset of cumulative consumption.
            policy_calls = next(item for item in load_checkpoint_bundle(bundle).manifest.liabilities if item.resource == "calls")
            for index in range(int(policy_calls.consumed or "0")):
                durable_write(target / "dispatches" / f"{index + 1:08d}.json", canonical_bytes({"inherited": True,
                    "parent_run": parent.run_id, "checkpoint": anchor.manifest}))
        durable_write(destination / "run.json", canonical_bytes(child.model_dump(mode="json")))
        durable_write(destination / "fork-receipt.json", canonical_bytes({"parent_run": parent.run_id,
            "child_run": child.run_id, "mode": mode, "selected_checkpoints": [anchor.manifest for *_, anchor in cuts],
            "budget": child_budget, "admission": "local-owner-only"}))
        return child


def export_evaluation(root: Path, destination: Path):
    """Exact retained directory export under an exclusive idle-owner lock."""
    with _owner_lock(root):
        if destination.exists():
            raise ValueError("evaluation_export_destination_exists")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("evaluation_export_symlink_refused")
        shutil.copytree(root, destination, ignore=shutil.ignore_patterns("owner.lock"), symlinks=False)
        inventory = {}
        for path in destination.rglob("*"):
            if path.is_file():
                inventory[str(path.relative_to(destination))] = bytes_digest(path.read_bytes())
        durable_write(destination / "export-manifest.json", canonical_bytes({"schema_version": "synth.evaluation-export.v1", "objects": inventory}))
        return {"files": len(inventory), "run_id": json.loads((root / "run.json").read_bytes())["run_id"]}
