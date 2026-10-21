"""Provider-free immutable regrading over retained V5 evidence.

This runs only pinned, owner-trusted offline rubrics. It owns no rollout scheduler,
financial grant, hosted index or arbitrary candidate execution facility.
"""

from __future__ import annotations

from decimal import Decimal
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

from .grading_models import GradingPlanV1, GradingPreflightV1, GradingResultV1, GradingAuthorityV1
from .package_models import content_digest, content_bytes
from .tracing.atomic import durable_mkdir, durable_write
from .tracing.canonical import bytes_digest, canonical_bytes, utc_now
from .tracing.models.evidence import TraceEvidenceBundleV5, TraceRefV5
from .tracing.models.artifacts import ArtifactRefV5
from .tracing.models.selectors import TraceSelectorV1, resolve_selector
from .tracing.models.standards import (
    JudgmentV1, ProducerRefV1, VerifierResultV2, JUDGMENT_SCHEMA_VERSION,
    RubricDefinitionV2, VerifierDefinitionV1, aggregate_rubric_score,
)
from .tracing.store.base import BlobStore
from .tracing.store.filesystem import FilesystemBlobStore
from .tracing.store.bundle import LocalTraceBundle
from .tracing.validation.rehydrate import build, evidence_bundle_from_payload, trace_document_from_payload
from .tracing.canonical import content_digest as trace_content_digest
from .tracing.evidence_ops import attach_many

NATIVE_FILES = frozenset({"__init__.py", "panel20.py", "rubric_v2.py", "snapshot.py", "worker.py"})


def grader_program_digest(plan: GradingPlanV1 | str, *, source_digest: str | None = None) -> str:
    scorer = plan.scorer if isinstance(plan, GradingPlanV1) else plan
    if isinstance(plan, GradingPlanV1):
        source_digest = next((item.digest for item in plan.inputs if item.name == "scorer_source"), None)
    return content_digest({"worker": bytes_digest(Path(__file__).with_name("grading_worker.py").read_bytes()),
                           "scorer": scorer, "native_source": source_digest})


def grader_dependencies_digest() -> str:
    return content_digest({"implementation": sys.implementation.name, "python": platform.python_version()})


def _inputs(plan: GradingPlanV1, blobs: BlobStore) -> tuple[dict[str, Any], list[str], list[str]]:
    values: dict[str, Any] = {}
    blockers: list[str] = []
    verified: list[str] = []
    for item in plan.inputs:
        try:
            body = blobs.get(item.digest)
            if bytes_digest(body) != item.digest or len(body) != item.size_bytes:
                raise ValueError("input_integrity")
            verified.append(item.digest)
            values[item.name] = json.loads(body) if item.media_type == "application/json" else body
        except (OSError, ValueError):
            if item.required:
                blockers.append(f"grading_input_unavailable:{item.name}")
    return values, blockers, verified


def preflight_grading(
    plan: GradingPlanV1, document, blobs: BlobStore, *, trusted_source_digests: frozenset[str] = frozenset(),
) -> GradingPreflightV1:
    plan = GradingPlanV1.model_validate(plan.model_dump(mode="python"))
    blockers: list[str] = []
    if document.content_digest != plan.trace_digest or document.trace_id != plan.trace_id or trace_content_digest(document) != plan.trace_digest:
        blockers.append("grading_trace_identity_mismatch")
    if set(plan.required_facts) - {"full_state", "trace_bound_observation"}:
        blockers.append("grading_required_fact_unsupported")
    if "full_state" in plan.required_facts:
        # A list of achievements cannot establish simulator/RNG/world closure.
        blockers.append("grading_full_state_capability_unqualified")
    if (
        "trace_bound_observation" in plan.required_facts
        and plan.scorer != "craftax-trajectory-metrics-v1"
    ):
        blockers.append("grading_trace_bound_observation_profile_unsupported")
    for selector in plan.required_selectors:
        if not resolve_selector(document, selector).resolved:
            blockers.append("grading_selector_unresolved")
    if plan.execution_profile != "trusted-offline-rubric-v1":
        blockers.append("external_grader_admission_required")
    if plan.scorer_image_digest is not None:
        blockers.append("image_execution_requires_external_grader")
    if Decimal(plan.limits.cost_ceiling_usd) != 0:
        blockers.append("offline_profile_requires_zero_provider_budget")
    if plan.scorer_program_digest != grader_program_digest(plan):
        blockers.append("grading_program_identity_mismatch")
    if plan.dependencies_digest != grader_dependencies_digest():
        blockers.append("grading_dependency_identity_mismatch")
    inputs, errors, verified = _inputs(plan, blobs)
    blockers.extend(errors)
    try:
        rubric = build(RubricDefinitionV2, inputs["rubric"])
        if rubric.content_digest != plan.rubric_digest or trace_content_digest(rubric) != plan.rubric_digest or tuple(c.criterion_id for c in rubric.criteria) != plan.case_ids:
            blockers.append("grading_rubric_identity_or_cases_mismatch")
        if plan.supersedes_result_id is not None:
            prior = evidence_bundle_from_payload(inputs["prior_evidence"])
            target = next((item for item in prior.verifier_results if item.verifier_result_id == plan.supersedes_result_id), None)
            if target is None or target.verifier_id != plan.scorer or target.rubric_digest != plan.rubric_digest or prior.trace_ref.content_digest != plan.trace_digest:
                blockers.append("grading_correction_subject_or_definition_mismatch")
            if any(item.supersedes_id == plan.supersedes_result_id for item in prior.verifier_results):
                blockers.append("grading_correction_already_superseded")
        if plan.scorer == "json-exact-v1":
            if not all(isinstance(inputs["cases"][name], dict) and {"actual", "expected"} <= inputs["cases"][name].keys() for name in plan.case_ids):
                blockers.append("grading_required_case_missing")
        elif plan.scorer == "craftax-trajectory-metrics-v1":
            from .grading_craftax import SCHEMA, observed_craftax_trajectory

            trajectory = inputs["trajectory"]
            if trajectory.get("schema_version") == SCHEMA:
                if canonical_bytes(trajectory) != canonical_bytes(
                    observed_craftax_trajectory(document)
                ):
                    blockers.append("craftax_observed_trajectory_trace_mismatch")
            elif "trace_bound_observation" in plan.required_facts:
                blockers.append("craftax_trace_bound_observation_missing")
            rows = trajectory["records"]
            if not isinstance(rows, list) or not rows or any(type(row.get("step")) is not int or row["step"] != index or "action" not in row for index, row in enumerate(rows)):
                blockers.append("craftax_action_prefix_missing")
            if "full_state" in plan.required_facts or "achievements" in plan.case_ids:
                if any(not isinstance(row.get("state"), dict) or not isinstance(row["state"].get("achievements"), list) for row in rows):
                    blockers.append("craftax_required_state_not_captured")
            if "reward" in plan.case_ids and any(type(row.get("reward")) not in (int, float) or not math.isfinite(row["reward"]) for row in rows):
                blockers.append("craftax_required_reward_not_captured")
            if set(plan.case_ids) - {"actions", "achievements", "reward"}:
                blockers.append("craftax_metric_unsupported")
        else:
            import base64
            import hashlib
            native = next(item for item in plan.inputs if item.name == "scorer_source")
            if native.digest not in trusted_source_digests:
                blockers.append("native_scorer_source_not_trusted")
            files = inputs["scorer_source"]["files"]
            if set(files) != NATIVE_FILES or any(not isinstance(source, str) for source in files.values()):
                blockers.append("native_scorer_source_closure_invalid")
            observed = inputs["observations"]
            if observed["submission_digest"] != plan.submission_digest or tuple(observed["case_ids"]) != plan.case_ids:
                blockers.append("native_observations_subject_or_case_mismatch")
            if len(observed["cases"]) != len(plan.case_ids) or len(observed["outputs"]) != len(plan.case_ids):
                blockers.append("native_observations_incomplete")
            submission = next(item for item in plan.inputs if item.name == "submission")
            if submission.digest != plan.submission_digest or not inputs.get("authority"):
                blockers.append("native_submission_or_private_authority_missing")
            frozen_files = inputs["submission"]["files"]
            from .package_models import portable_path
            if not isinstance(frozen_files, dict) or not frozen_files:
                raise ValueError("submission files missing")
            tree = []
            for name, encoded in sorted(frozen_files.items()):
                portable_path(name)
                body = base64.b64decode(encoded, validate=True)
                tree.append((name, hashlib.sha256(body).hexdigest()))
            native_tree = hashlib.sha256(json.dumps(tree, separators=(",", ":")).encode()).hexdigest()
            receipt = inputs["native_receipt"]
            if receipt["submission_sha256"] != native_tree or receipt["task"] != observed["task_id"]:
                blockers.append("native_observation_submission_binding_mismatch")
            if observed["outputs"] != inputs["native_observations"]["outputs"]:
                blockers.append("native_observation_source_mismatch")
            if content_digest(receipt["task_manifest"]) != plan.task_digest:
                blockers.append("native_task_identity_mismatch")
    except (KeyError, TypeError, StopIteration, AttributeError, ValueError):
        blockers.append("grading_required_evidence_not_captured")
    return GradingPreflightV1(plan_digest=plan.digest, sufficient=not blockers,
        blockers=tuple(dict.fromkeys(blockers)), verified_inputs=tuple(verified),
        execution_profile=plan.execution_profile, cost_ceiling_usd=plan.limits.cost_ceiling_usd)


def _worker(plan: GradingPlanV1, inputs: dict[str, Any], directory: Path):
    """Bound only an owned trusted worker; candidate compilation is unsupported here."""
    import resource
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU, (plan.limits.cpu_seconds, plan.limits.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (plan.limits.output_bytes, plan.limits.output_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (plan.limits.memory_bytes, plan.limits.memory_bytes))
    request = canonical_bytes({"scorer": plan.scorer, "case_ids": plan.case_ids,
                               "required_facts": plan.required_facts, "inputs": inputs})
    input_path = directory / "input.json"
    durable_write(input_path, request)
    stdout = bytearray()
    stderr = bytearray()
    status = "completed"
    started = time.monotonic_ns()
    with input_path.open("rb") as stdin:
        process = subprocess.Popen([sys.executable, "-P", str(Path(__file__).with_name("grading_worker.py"))],
            cwd=directory, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
            start_new_session=True, preexec_fn=limits)
    assert process.stdout is not None and process.stderr is not None
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, stdout)
            selector.register(process.stderr, selectors.EVENT_READ, stderr)
            while selector.get_map():
                if (time.monotonic_ns() - started) / 1e9 > plan.limits.wall_seconds:
                    status = "timed_out"
                    break
                if sys.platform == "darwin" and process.poll() is None:
                    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, timeout=2).stdout.strip()
                    if rss and int(rss) * 1024 > plan.limits.memory_bytes:
                        status = "memory_limit"
                        break
                for key, _ in selector.select(timeout=0.05):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    target = key.data
                    remaining = plan.limits.output_bytes - len(stdout) - len(stderr)
                    target.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        status = "output_limit"
                        break
                if status != "completed":
                    break
        if status == "completed" and process.wait(timeout=2) != 0:
            status = "failed"
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
    return status, bytes(stdout), bytes(stderr), time.monotonic_ns() - started


def run_grading(
    plan: GradingPlanV1, document, blobs: BlobStore, root: Path, *,
    trusted_source_digests: frozenset[str] = frozenset(), retry_interrupted: bool = False,
    source_bundle: LocalTraceBundle | None = None,
) -> GradingResultV1:
    plan = GradingPlanV1.model_validate(plan.model_dump(mode="python"))
    directory = root / plan.operation_id
    durable_mkdir(directory)
    durable_write(directory / "plan.json", content_bytes(plan.model_dump(mode="json")))
    with (directory / "write.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("grading_operation_running") from None
        result_path = directory / "result.json"
        if result_path.exists():
            result = GradingResultV1.model_validate_json(result_path.read_bytes())
            if result.plan_digest != plan.digest:
                raise ValueError("grading_operation_identity_conflict")
            read_grading_evidence(directory, result)
            return result
        if source_bundle is not None:
            source_bundle.read_trace(plan.trace_digest)
            if not (directory / "trace-bundle.zip").exists():
                durable_write(directory / "trace-bundle.zip", source_bundle.archive_bytes())
        retained = FilesystemBlobStore(directory / "inputs")
        completed = sorted({p.parent for pattern in ("*/completion.json", "*/worker-result.json")
                            for p in (directory / "attempts").glob(pattern)})
        if len(completed) > 1:
            raise ValueError("grading_completed_attempts_ambiguous")
        cleanup = "owned-worker-stopped"
        if completed:
            attempt = completed[0]
            attempt_id = attempt.name
            started = json.loads((attempt / "started.json").read_bytes())
            if started["plan_digest"] != plan.digest or started["attempt_id"] != attempt_id:
                raise ValueError("grading_attempt_identity_conflict")
            if (attempt / "completion.json").exists():
                completion = json.loads((attempt / "completion.json").read_bytes())
                stdout = (attempt / "stdout").read_bytes()
                stderr = (attempt / "stderr").read_bytes() if completion["stderr_digest"] is not None else None
                if bytes_digest(stdout) != completion["stdout_digest"] or (stderr is not None and bytes_digest(stderr) != completion["stderr_digest"]):
                    raise ValueError("grading_completed_output_integrity")
                status, elapsed, produced_at, cleanup = (completion[key] for key in ("status", "elapsed_ns", "produced_at", "cleanup"))
            else:
                # The worker committed before its controller died. Preserve
                # unknown elapsed/log coverage; no second grader invocation.
                stdout = (attempt / "worker-result.json").read_bytes()
                stderr = (attempt / "stderr").read_bytes() if (attempt / "stderr").exists() else None
                durable_write(attempt / "stdout", stdout)
                status, elapsed, produced_at, cleanup = "completed", None, utc_now(), "worker-exit-not-observed"
            document = trace_document_from_payload(json.loads((directory / "trace.json").read_bytes()))
            inputs, errors, _ = _inputs(plan, retained)
            if errors:
                raise ValueError("grading_retained_input_closure_corrupt")
        else:
            if list((directory / "attempts").glob("*/started.json")) and not retry_interrupted:
                raise ValueError("grading_attempt_outcome_unknown: explicit offline retry required")
            preflight = preflight_grading(plan, document, blobs, trusted_source_digests=trusted_source_digests)
            if not preflight.sufficient:
                raise ValueError(";".join(preflight.blockers))
            for item in plan.inputs:
                if item.digest in preflight.verified_inputs:
                    if retained.put(blobs.get(item.digest)) != item.digest:
                        raise ValueError("grading_input_copy_integrity")
            durable_write(directory / "trace.json", canonical_bytes(document))
            inputs, _, _ = _inputs(plan, retained)
            inputs = {name: value for name, value in inputs.items() if not isinstance(value, bytes)}
            attempt_id = str(uuid4())
            attempt = directory / "attempts" / attempt_id
            durable_mkdir(attempt)
            durable_write(attempt / "started.json", canonical_bytes({"attempt_id": attempt_id,
                "plan_digest": plan.digest, "started_at": utc_now(), "provider_calls": 0}))
            status, stdout, stderr, elapsed = _worker(plan, inputs, attempt)
            durable_write(attempt / "stdout", stdout)
            durable_write(attempt / "stderr", stderr)
            produced_at = utc_now()
        durable_write(attempt / "completion.json", content_bytes({"status": status,
            "stdout_digest": bytes_digest(stdout), "stderr_digest": bytes_digest(stderr) if stderr is not None else None,
            "elapsed_ns": elapsed, "produced_at": produced_at, "cleanup": cleanup}))
        rubric = build(RubricDefinitionV2, inputs["rubric"])
        if rubric.content_digest != plan.rubric_digest or trace_content_digest(rubric) != plan.rubric_digest:
            raise ValueError("grading_retained_rubric_integrity")
        criteria_by_id = {item.criterion_id: item for item in rubric.criteria}
        rows = {}
        if status == "completed":
            try:
                rows = json.loads(stdout)["cases"]
                if set(rows) != set(plan.case_ids):
                    raise ValueError("case coverage mismatch")
                for row in rows.values():
                    score = row.get("score")
                    if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
                        raise ValueError("invalid score")
                for name, row in rows.items():
                    criterion = rubric.criterion(name)
                    if criterion is None or (row.get("score") is not None and not criterion.min_score <= row["score"] <= criterion.max_score):
                        raise ValueError("score outside declared rubric range")
            except (ValueError, KeyError, TypeError):
                rows = {}
                status = "failed"
        result_id = "grading-" + content_digest({"plan": plan.digest, "operation": plan.operation_id})[7:]
        subject = TraceSelectorV1(plan.trace_id, plan.trace_digest, "trace")
        producer = ProducerRefV1("deterministic", plan.scorer, "1", config_digest=plan.digest)
        def judgment_for(name):
            criterion = criteria_by_id[name]
            score = rows.get(name, {}).get("score")
            decision = rows.get(name, {}).get("passed")
            if score is not None and decision is None:
                decision = score >= criterion.pass_threshold if criterion.higher_is_better else score <= criterion.pass_threshold
            return JudgmentV1(criterion_id=name, score=score,
                verdict="pass" if decision else "fail" if decision is not None else "inconclusive",
                passed=decision, evidence=plan.required_selectors or (subject,),
                grounding="grounded", judgment_id=f"{result_id}:{name}", subject=subject,
                producer=producer, produced_at=produced_at, revision=1, state="current",
                criterion_version=criterion.version, criterion_digest=criterion.content_digest,
                status="decisive" if score is not None else "inconclusive",
                schema_version=JUDGMENT_SCHEMA_VERSION).sealed()
        judgments = tuple(judgment_for(name) for name in plan.case_ids)
        scores = [item.score for item in judgments if item.score is not None]
        mean, passed, _ = aggregate_rubric_score(rubric, judgments)
        definition = VerifierDefinitionV1(plan.scorer, plan.scorer, "deterministic", rubric.rubric_id,
            rubric.version, rubric.content_digest, version="1", program_ref=plan.scorer_program_digest,
            timeout_seconds=plan.limits.wall_seconds).sealed()
        verifier = VerifierResultV2(result_id, plan.scorer, "1", rubric.rubric_id, plan.rubric_digest,
            subject, "completed" if status == "completed" else "failed", "valid" if len(scores) == len(plan.case_ids) else "inconclusive",
            "grounded", produced_at, producer, score=mean, passed=passed, criterion_results=judgments,
            supersedes_id=plan.supersedes_result_id,
            metadata={"grading_plan_digest": plan.digest, "attempt_id": attempt_id,
                      "planned_cases": len(plan.case_ids), "scored_cases": len(scores)}).sealed()
        artifacts = tuple(ArtifactRefV5(artifact_id=f"{result_id}:{name}", digest=bytes_digest(body),
            media_type="application/json" if name == "stdout" else "text/plain", size_bytes=len(body),
            role="evaluation_output", uri="blobs/sha256/" + bytes_digest(body)[7:9] + "/" + bytes_digest(body)[7:],
            visibility="private") for name, body in (("stdout", stdout), ("stderr", stderr)) if body is not None)
        evidence = TraceEvidenceBundleV5(result_id, TraceRefV5(plan.trace_id, plan.trace_digest),
            produced_at, criteria=rubric.criteria, rubrics=(rubric,), verifier_definitions=(definition,),
            verifier_results=(verifier,), artifacts=artifacts,
            metadata={"plan": plan.model_dump(mode="json")}).sealed()
        if plan.supersedes_result_id is not None:
            prior = evidence_bundle_from_payload(inputs["prior_evidence"])
            evidence = attach_many(prior, records=(
                ("verifier_result", verifier), *(("artifact", item) for item in artifacts)),
                created_at=produced_at)
        from .tracing.validation.validator import validate_evidence
        if any(str(finding.severity) == "error" for finding in validate_evidence(document, evidence)[0]):
            raise ValueError("grading_evidence_validation_failed")
        durable_write(directory / "evidence.json", canonical_bytes(evidence))
        result = GradingResultV1(operation_id=plan.operation_id, plan_digest=plan.digest,
            attempt_id=attempt_id, result_id=result_id, evidence_digest=evidence.content_digest,
            execution_status=status, planned_cases=len(plan.case_ids), completed_cases=len(rows),
            scored_cases=len(scores), mean_score=mean, elapsed_ns=elapsed,
            stdout_digest=bytes_digest(stdout), stderr_digest=bytes_digest(stderr) if stderr is not None else None,
            cleanup=cleanup, case_scores={item.criterion_id: item.score for item in judgments})
        durable_write(result_path, content_bytes(result.model_dump(mode="json")))
        return result


def read_grading_evidence(directory: Path, result: GradingResultV1) -> TraceEvidenceBundleV5:
    evidence = evidence_bundle_from_payload(json.loads((directory / "evidence.json").read_bytes()))
    if evidence.content_digest != result.evidence_digest:
        raise ValueError("grading_evidence_integrity_mismatch")
    for name, digest in (("stdout", result.stdout_digest), ("stderr", result.stderr_digest)):
        if digest is None:
            continue
        if bytes_digest((directory / "attempts" / result.attempt_id / name).read_bytes()) != digest:
            raise ValueError("grading_log_integrity_mismatch")
    return evidence


def publish_grading(bundle: LocalTraceBundle, root: Path, result: GradingResultV1) -> str:
    """Append retained judgments/logs/inputs; no grader or rollout is callable."""
    directory = root / result.operation_id
    plan = GradingPlanV1.model_validate_json((directory / "plan.json").read_bytes())
    if plan.digest != result.plan_digest:
        raise ValueError("grading_publication_plan_conflict")
    bundle.read_trace(plan.trace_digest)
    evidence = read_grading_evidence(directory, result)
    retained = FilesystemBlobStore(directory / "inputs")
    for item in plan.inputs:
        if item.required or retained.has(item.digest):
            if bundle.blobs.put(retained.get(item.digest)) != item.digest:
                raise ValueError("grading_publication_input_integrity")
    for name, digest in (("stdout", result.stdout_digest), ("stderr", result.stderr_digest)):
        if digest is not None and bundle.blobs.put((directory / "attempts" / result.attempt_id / name).read_bytes()) != digest:
            raise ValueError("grading_publication_log_integrity")
    bundle.write_evidence(evidence)
    bundle.write_receipt(f"grading-{result.result_id}", result.model_dump(mode="json"))
    bundle.write_manifest()
    return evidence.content_digest


def export_grading(root: Path, result: GradingResultV1, destination: Path) -> LocalTraceBundle:
    source = root / result.operation_id / "trace-bundle.zip"
    if not source.exists():
        raise ValueError("grading_original_bundle_not_retained")
    bundle = LocalTraceBundle.extract_archive(source, destination)
    publish_grading(bundle, root, result)
    return bundle


def aggregate_grades(results: tuple[GradingResultV1, ...], policy: GradingAuthorityV1) -> dict:
    unique = {item.result_id: item for item in results}
    if any(item.result_id in unique and item != unique[item.result_id] for item in results):
        raise ValueError("grading_result_identity_conflict")
    if any(key not in unique for key in policy.result_ids) or len(set(policy.result_ids)) != len(policy.result_ids):
        raise ValueError("grading_authority_selection_invalid")
    selected = [unique[key] for key in policy.result_ids]
    scored = sum(item.scored_cases for item in selected)
    return {"policy": policy.name, "result_ids": list(policy.result_ids),
        "planned_cases": sum(item.planned_cases for item in selected),
        "completed_cases": sum(item.completed_cases for item in selected), "scored_cases": scored,
        "mean_score": sum(item.mean_score * item.scored_cases for item in selected if item.mean_score is not None) / scored if scored else None}


def compare_grades(left_plan: GradingPlanV1, left: GradingResultV1,
                   right_plan: GradingPlanV1, right: GradingResultV1) -> dict:
    """Matched raw case deltas with coverage changes kept separate."""
    if left.plan_digest != left_plan.digest or right.plan_digest != right_plan.digest:
        raise ValueError("grading_comparison_plan_identity_mismatch")
    for name in ("trace_digest", "submission_digest", "task_digest", "world_digest", "dataset_digest", "split_digest"):
        if getattr(left_plan, name) != getattr(right_plan, name):
            raise ValueError("grading_comparison_subject_mismatch")
    common = sorted(set(left_plan.case_ids) & set(right_plan.case_ids))
    matched = [key for key in common if left.case_scores.get(key) is not None and right.case_scores.get(key) is not None]
    deltas = {}
    for key in matched:
        left_score, right_score = left.case_scores[key], right.case_scores[key]
        if left_score is not None and right_score is not None:
            deltas[key] = right_score - left_score
    return {"matching_planned_cases": common, "matching_scored_cases": matched,
        "left_only_cases": sorted(set(left_plan.case_ids) - set(right_plan.case_ids)),
        "right_only_cases": sorted(set(right_plan.case_ids) - set(left_plan.case_ids)),
        "left_scored_cases": left.scored_cases, "right_scored_cases": right.scored_cases,
        "case_deltas": deltas,
        "projection": "raw named case scores; no comparison of unmatched/missing cases"}
