"""Run one annotator as a jesterky map/reduce swarm.

The default paid grader is still Codex app-server. This runner is the third
option: ``jesterky run`` with ``--actor`` / ``--model`` / ``--effort`` taken from
the annotation request. The swarm is expand-style map over ``ledger.jobs``
(one item for a single job, many when the request seeds extra jobs).

The actor must emit ``synth.annotation-proposal.v1``. Validation, sealing, and
reservations are identical to the Codex path. Hidden chain of thought is not
requested.

Tests inject ``proposal_factory`` and never spawn the CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ..canonical import utc_now
from ..models.standards import ProducerKind
from .definitions import RunnerKind
from .execution_trace import ExecutionCapture
from .jobs import RUNNER_VERSION, AnnotationJobErrorCode, AnnotationJobErrorV1, AnnotationJobUsageV1
from .pricing import COST_STATUS_UNAVAILABLE, ModelPrice, PriceTable
from .proposal import PROPOSAL_SCHEMA_VERSION, STRICT_PROPOSAL_JSON_SCHEMA, normalize_strict_proposal
from .validation import producer_for
from .workspace import unlock_workspace

ProposalFactory = Callable[[Any], dict[str, Any]]

TRACE_ANNOTATOR_ACTOR = "trace_annotator"
DEFAULT_CONCURRENCY = 4
DEFAULT_COMMAND = ("jesterky",)


def swarm_spec(*, concurrency: int, prompt: str, schema_file: str = "proposal.schema.json") -> dict[str, Any]:
    """Map ``trace_annotator`` over ``ledger.jobs``. One job is a 1-item swarm."""

    width = max(1, int(concurrency))
    return {
        "name": "trace_v5_annotate",
        "entrypoint": ["annotate_jobs"],
        "nodes": {
            "annotate_jobs": {
                "kind": "map",
                "over": "ledger.jobs",
                "item_as": "item",
                "concurrency": width,
                "min_success": 1.0,
                "body": {
                    "kind": "actor",
                    "actor": TRACE_ANNOTATOR_ACTOR,
                    "inputs": {"job": "item"},
                    "outputs": {},
                },
                "outputs": {"job": "ledger.scans"},
            }
        },
        "runplan": {"map_concurrency": width},
        "host": {
            "roles": {TRACE_ANNOTATOR_ACTOR: {"prompt": prompt}},
            "output_schemas": {TRACE_ANNOTATOR_ACTOR: schema_file},
            "viz": {"map_node": "annotate_jobs", "item_label_field": "annotator_id"},
        },
    }


def extract_proposal(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the first annotation-proposal object out of a jesterky run manifest."""

    for record in manifest.get("recorded") or ():
        outputs = record.get("outputs")
        found = _proposal_in(outputs)
        if found is not None:
            return found
    return _proposal_in(manifest.get("args"))


def _proposal_in(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("schema_version") == PROPOSAL_SCHEMA_VERSION:
            return value
        nested = value.get("proposal")
        if isinstance(nested, dict) and nested.get("schema_version") == PROPOSAL_SCHEMA_VERSION:
            return nested
        for item in value.values():
            found = _proposal_in(item)
            if found is not None:
                return found
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        return _proposal_in(parsed)
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _proposal_in(item)
            if found is not None:
                return found
    return None


class JesterkyRunner:
    """``AnnotatorRunner`` that drives one ``jesterky run`` per job."""

    kind = RunnerKind.JESTERKY.value
    version = "jesterky@1"

    def __init__(
        self,
        *,
        command: tuple[str, ...] = DEFAULT_COMMAND,
        actor: str = "codex",
        default_model: str | None = None,
        default_effort: str | None = None,
        proposal_factory: ProposalFactory | None = None,
        usd_per_million_tokens: float | None = None,
        proxy_enforces_reservation: bool = False,
        price_table: PriceTable | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.command = tuple(command)
        self.actor = actor
        self.default_model = default_model
        self.default_effort = default_effort
        self.proposal_factory = proposal_factory
        self.usd_per_million_tokens = usd_per_million_tokens
        self.proxy_enforces_reservation = proxy_enforces_reservation
        self.price_table = price_table
        self.extra_env = dict(extra_env or {})

    def resolve_model(self, requested: str | None, definition_model: str | None) -> str | None:
        return requested or definition_model or self.default_model

    def resolve_effort(self, requested: str | None, program_default: str | None) -> str | None:
        return requested or program_default or self.default_effort

    def price_for(self, model: str | None) -> ModelPrice | None:
        return self.price_table.get(model) if self.price_table is not None else None

    def cost_enforcement(self, model: str | None = None) -> str | None:
        if self.price_for(model) is not None or self.usd_per_million_tokens:
            return "pinned_price"
        if self.proxy_enforces_reservation:
            return "provider_proxy"
        return None

    def token_ceiling(self, limits: Any, model: str | None = None) -> int | None:
        ceiling = limits.max_total_tokens
        by_cost: int | None = None
        if limits.max_cost_usd is not None:
            price = self.price_for(model)
            if price is not None:
                by_cost = price.billable_token_ceiling(limits.max_cost_usd)
            elif self.usd_per_million_tokens:
                by_cost = int(limits.max_cost_usd / self.usd_per_million_tokens * 1_000_000)
        if by_cost is not None:
            ceiling = by_cost if ceiling is None else min(ceiling, by_cost)
        return ceiling

    def run(self, context: Any) -> Any:
        from .service import RunOutcome

        job = context.job
        limits = job.request.limits
        started = utc_now()
        clock = time.monotonic()
        model = job.request.model or self.default_model
        effort = job.request.reasoning_effort or self.default_effort
        error: AnnotationJobErrorV1 | None = None
        proposal: dict[str, Any] | None = None
        events: list[dict[str, Any]] = []

        try:
            if self.proposal_factory is not None:
                events.append({"at": utc_now(), "kind": "scripted_jesterky"})
                proposal = normalize_strict_proposal(self.proposal_factory(context))
            else:
                proposal = self._run_cli(context, model=model, effort=effort, events=events)
        except ValueError as bad:
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.MALFORMED_OUTPUT, message=str(bad))
            proposal = None
        except FileNotFoundError as missing:
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.RUNNER_UNAVAILABLE, message=str(missing))
        except TimeoutError:
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.TIMEOUT, message=f"jesterky exceeded {limits.timeout_seconds}s")
        except RuntimeError as failed:
            error = AnnotationJobErrorV1(code=AnnotationJobErrorCode.INTERNAL, message=str(failed))

        usage = AnnotationJobUsageV1(
            tool_calls=len(context.tools.calls),
            tool_bytes=context.tools.total_bytes,
            cost_usd=0.0 if self.proposal_factory is not None else None,
            cost_status="free" if self.proposal_factory is not None else COST_STATUS_UNAVAILABLE,
            wall_time_seconds=time.monotonic() - clock,
        )
        capture = ExecutionCapture(
            started_at=started,
            ended_at=utc_now(),
            instructions_digest=context.instructions_digest,
            tool_calls=tuple(context.tools.calls),
            final_output=proposal if isinstance(proposal, dict) else None,
            usage=usage,
            runner_kind=self.kind,
            model=model,
            reasoning_effort=effort,
            transport_events=tuple(events),
            error=error.message if error else None,
        )
        producer = producer_for(
            context.entry.definition,
            kind=ProducerKind.AGENTIC,
            name=self.kind,
            version=RUNNER_VERSION,
            model=model,
            config_digest=context.entry.program.content_digest,
        )
        return RunOutcome(proposal=proposal, capture=capture, error=error, producer=producer)

    def _run_cli(
        self,
        context: Any,
        *,
        model: str | None,
        effort: str | None,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        workspace = Path(context.workspace_dir)
        unlock_workspace(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        trace_path = workspace / "trace.json"
        spec_path = workspace / "spec.json"
        args_path = workspace / "args.json"
        schema_path = workspace / "proposal.schema.json"
        manifest_path = workspace / "jesterky_annotate.manifest.json"
        document = context.document
        trace_path.write_text(json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        schema_path.write_text(json.dumps(STRICT_PROPOSAL_JSON_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        concurrency = int((context.entry.program.parameters or {}).get("jesterky_concurrency") or DEFAULT_CONCURRENCY)
        if context.job.request.metadata.get("jesterky_concurrency") is not None:
            concurrency = int(context.job.request.metadata["jesterky_concurrency"])
        prompt = (
            f"{context.instructions_text}\n\n"
            "You are one worker in a jesterky swarm. Read ONLY the Trace V5 JSON at `job.path`. "
            "Do not run shell commands. Return one JSON object: synth.annotation-proposal.v1 "
            f"(schema_version {PROPOSAL_SCHEMA_VERSION!r}) with findings that cite event selectors."
        )
        spec_path.write_text(json.dumps(swarm_spec(concurrency=concurrency, prompt=prompt), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        job_item = {
            "trace_id": document.trace_id,
            "trace_digest": document.content_digest,
            "path": str(trace_path),
            "annotator_id": context.job.request.annotator_id,
            "model": model,
            "reasoning_effort": effort,
        }
        extra_jobs = context.job.request.metadata.get("jesterky_jobs")
        jobs = list(extra_jobs) if isinstance(extra_jobs, list) and extra_jobs else [job_item]
        args_path.write_text(json.dumps({"jobs": jobs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        argv = [
            *self.command,
            "run",
            str(spec_path),
            "--actor",
            self.actor,
            "--args-file",
            str(args_path),
            "--out",
            str(manifest_path),
            "--cd",
            str(workspace),
            "--no-follow",
            "--run-id",
            f"ann-{context.job.job_id}",
        ]
        if model:
            argv.extend(["--model", str(model)])
        if effort:
            argv.extend(["--effort", str(effort)])
        events.append({"at": utc_now(), "kind": "jesterky_cli", "argv": argv[1:], "actor": self.actor, "model": model})
        env = {**os.environ, **self.extra_env}
        timeout = float(context.job.request.limits.timeout_seconds)
        try:
            completed = subprocess.run(argv, cwd=workspace, env=env, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as timed_out:
            raise TimeoutError(str(timed_out)) from timed_out
        if completed.returncode != 0:
            stderr = (completed.stderr or "")[-2000:]
            raise RuntimeError(f"jesterky exited {completed.returncode}: {stderr}")
        if not manifest_path.is_file():
            raise RuntimeError(f"jesterky completed without a manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extracted = extract_proposal(manifest)
        if extracted is None:
            raise ValueError("jesterky manifest contained no synth.annotation-proposal.v1 object")
        return normalize_strict_proposal(extracted)


__all__ = ["DEFAULT_COMMAND", "JesterkyRunner", "TRACE_ANNOTATOR_ACTOR", "extract_proposal", "swarm_spec"]
