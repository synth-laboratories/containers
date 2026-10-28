"""Portable eval plan/run/watch/results/cancel/resume/fork/export commands."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, UTC
import json
import re
import fcntl
from pathlib import Path
import shutil
import sys
import time
import tomllib
from uuid import uuid4

from pydantic import ValidationError

from .native_engine import NativeEngineError

from .eval_models import EvaluationPlanV1, EvaluationRunDeclarationV1, ScriptedCandidateV1, CandidateBindingV1
from .eval_run import (describe_evaluation, execute_evaluation, export_evaluation, fork_evaluation,
                       local_execution_source_digest, local_execution_dependencies_digest)
from .operator_journal import read_operator_events
from .package_models import content_digest
from .tracing.atomic import durable_write
from .package_models import content_bytes as canonical_bytes


def read_config(path: Path):
    return tomllib.loads(path.read_text()) if path.suffix == ".toml" else json.loads(path.read_bytes())


def _example_toml(value: dict, prefix: tuple[str, ...] = ()) -> str:
    lines = []
    if prefix:
        lines.append("[" + ".".join(json.dumps(key) for key in prefix) + "]")
    for key, item in value.items():
        if not isinstance(item, dict):
            lines.append(json.dumps(key) + " = " + json.dumps(item))
    for key, item in value.items():
        if isinstance(item, dict):
            lines.append(_example_toml(item, (*prefix, key)))
    return "\n".join(lines) + "\n"


def _scripted_candidate(label: str, document) -> ScriptedCandidateV1:
    """Typed candidate refusal: a model table is named, not reported as a bare ValidationError."""
    if isinstance(document, dict) and "model" in document:
        raise ValueError(f"candidate_model_binding_unsupported: candidate {_public(label) or 'label'} declares a model "
                         "table; local-gold-scripted-v1 executes scripted-actions-v1 candidates and runs no model")
    try:
        return ScriptedCandidateV1.model_validate(document)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(part) for part in first["loc"]) or "candidate"
        raise ValueError(f"candidate_invalid: candidate {_public(label) or 'label'} field "
                         f"{_public(field) or 'unnamed'} {first['type']}") from exc


def candidates_from(items: list[str], *, relative: Path) -> tuple[CandidateBindingV1, ...]:
    result = []
    for item in items:
        label, separator, name = item.partition("=")
        if not separator or not label or not name:
            raise ValueError("candidate_requires_LABEL=PATH")
        path = Path(name)
        if not path.is_absolute():
            path = relative / path
        result.append(CandidateBindingV1(label=label, content=_scripted_candidate(label, read_config(path))))
    if len({item.label for item in result}) != len(result):
        raise ValueError("candidate_labels_must_be_unique")
    return tuple(result)


class TerminalProgress:
    """Read-only view: at most ten lanes, fixed right column, actual terminal size."""
    def __init__(self, stream, *, mode="auto"):
        self.stream, self.mode = stream, mode
        self.tty = mode == "tty" or mode == "auto" and stream.isatty()
        self.lanes = {}
        self.previous_lines = 0

    def __call__(self, event):
        if self.mode == "json":
            self.stream.write(json.dumps(event, sort_keys=True) + "\n")
            self.stream.flush()
            return
        lane = event.get("lane")
        payload = event.get("payload") or {}
        if lane:
            state = self.lanes.setdefault(lane, {"status": "queued", "steps": None, "limit": None, "latest": ""})
            if event["kind"] == "eval.phase":
                state["status"] = payload.get("phase", state["status"])
            if event["kind"] == "eval.usage.sampled":
                state.update(steps=payload.get("steps"), limit=payload.get("step_limit"))
            if event["kind"] == "eval.run.terminal":
                state["status"] = payload.get("execution_status", "terminal")
            state["latest"] = payload.get("stop_reason") or payload.get("phase") or payload.get("kind") or event["kind"]
        if not self.tty:
            self.stream.write(f"{event['seq']} {lane or event['run_id']} {event['kind']} " + json.dumps(payload, sort_keys=True) + "\n")
            self.stream.flush()
            return
        width, height = shutil.get_terminal_size((100, 24))
        count = min(10, max(0, height - 3))
        sidebar = min(40, max(12, width // 3))
        left_width = max(1, width - sidebar - 3)
        lines = [f"{event['run_id']}  {len(self.lanes)} rollouts  event {event['seq']}"[:max(1, width - 1)]]
        for name, state in list(self.lanes.items())[:count]:
            progress = "?" if state["steps"] is None else f"{state['steps']}/{state['limit'] if state['limit'] is not None else '?'}"
            left = f"{name} {progress}"
            right = f"{state['status']}  {state['latest']}"
            lines.append((left[:left_width].ljust(left_width) + " | " + right[:sidebar])[:max(1, width - 1)])
        if len(self.lanes) > count:
            lines.append(f"+{len(self.lanes) - count} additional rollouts"[:max(1, width - 1)])
        if self.previous_lines:
            self.stream.write(f"\x1b[{self.previous_lines}A")
        for line in lines:
            self.stream.write("\x1b[2K" + line + "\n")
        for _ in range(max(0, self.previous_lines - len(lines))):
            self.stream.write("\x1b[2K\n")
        self.previous_lines = max(len(lines), self.previous_lines)
        self.stream.flush()


_PUBLIC_TEXT = re.compile(r"[a-zA-Z0-9_.:; /=+()-]{1,512}")


def _public(text) -> str | None:
    """Only plain diagnostic text is public; private payloads and provider text are not."""
    return text if isinstance(text, str) and _PUBLIC_TEXT.fullmatch(text) else None


def _public_limits(limits: dict) -> dict:
    """Declared and observed limits are numbers and fixed Docker identifiers only."""
    def scalars(side):
        if not isinstance(side, dict):
            return None
        return {key: value if isinstance(value, (int, float, bool)) or value is None else _public(value)
                for key, value in side.items() if isinstance(key, str)}
    return {"declared": scalars(limits.get("declared")), "observed": scalars(limits.get("observed")),
            "egress": _public(limits.get("egress"))}


def refusal_payload(exc: BaseException) -> dict:
    """Public refusal record: a typed engine failure keeps its code, stage, name, last state and cause."""
    payload = {"error": "evaluation_refused", "reason": _public(str(exc)) or type(exc).__name__}
    if isinstance(exc, NativeEngineError):
        payload.update({"reason": exc.code, "stage": exc.stage, "name": exc.name, "last_state": exc.last_state,
                        "cause": _public(exc.cause) if exc.cause is not None else None})
        if exc.limits is not None:
            payload["limits"] = _public_limits(exc.limits)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-containers eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("describe", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("plan", type=Path)
        sub.add_argument("--candidate", action="append", default=[])
        if name == "run":
            sub.add_argument("--root", type=Path, required=True)
            sub.add_argument("--run-id")
            sub.add_argument("--detach", action="store_true")
            sub.add_argument("--format", choices=("auto", "tty", "plain", "json"), default="auto")
    for name in ("watch", "results", "resume", "cancel", "export", "fork"):
        sub = commands.add_parser(name)
        sub.add_argument("root", type=Path)
        if name in ("watch", "resume"):
            sub.add_argument("--format", choices=("auto", "tty", "plain", "json"), default="auto")
        if name == "watch":
            sub.add_argument("--after", type=int, default=0)
            sub.add_argument("--follow", action="store_true")
            sub.add_argument("--timeout", type=float, default=300)
        if name in ("export", "fork"):
            sub.add_argument("--destination", type=Path, required=True)
        if name == "fork":
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--candidate", action="append", default=[])
            sub.add_argument("--child-budget", type=Path, required=True)
    commands.add_parser("identities")
    commands.add_parser("egress-capability")
    commands.add_parser("schema")
    example = commands.add_parser("example")
    example.add_argument("--image", required=True)
    example.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "identities":
            output = {"source_digest": local_execution_source_digest(), "dependencies_digest": local_execution_dependencies_digest()}
        elif args.command == "egress-capability":
            from .engine_egress import egress_capability
            output = egress_capability()
        elif args.command == "schema":
            output = EvaluationPlanV1.model_json_schema()
        elif args.command == "example":
            args.directory.mkdir(parents=True, exist_ok=False)
            plan = EvaluationPlanV1(benchmark="craftax", adapter="local-gold-scripted-v1",
                source_digest=local_execution_source_digest(), dependencies_digest=local_execution_dependencies_digest(),
                scorer_digest=content_digest({"metric": "native-environment-reward-v1"}), image_digest=args.image,
                task={"schema": "gamebench.task.craftax.v1", "task_id": "containers-craftax-scripted-v1",
                    "scenario_id": "classic", "variant": "classic", "world": {"classic": True},
                    "rules": {"base": "symbolic_survival", "upstream_do_resurrection": False},
                    "readouts": {"profile": "symbolic_compact"}},
                seeds=tuple(range(10)), parallelism=4, limits={"steps": 8, "calls": 16, "active_seconds": 120,
                    "clock_policy": "absolute_deadline", "deadline_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat()})
            # JSON is a TOML-compatible schema peer. The generated TOML makes
            # presets reproducible without any owner-specific filesystem path.
            config = json.loads(plan.model_dump_json(exclude_none=True))
            (args.directory / "evaluation.toml").write_text(_example_toml(config))
            (args.directory / "scripted.toml").write_text('kind = "scripted-actions-v1"\nactions = ["noop"]\nrepeat = true\n')
            output = {"plan": str(args.directory / "evaluation.toml"), "candidate": "scripted=scripted.toml"}
        elif args.command in {"describe", "run"}:
            plan = EvaluationPlanV1.model_validate(read_config(args.plan))
            candidates = candidates_from(args.candidate, relative=args.plan.parent)
            if args.command == "describe":
                output = describe_evaluation(plan, candidates)
                print(json.dumps(output, sort_keys=True))
                return 0 if output["supported"] else 2
            if args.detach:
                raise ValueError("detached_local_execution_unsupported: use foreground owner or admitted service")
            if args.root.exists():
                raise ValueError("fresh_run_requires_new_root: use explicit resume")
            run = EvaluationRunDeclarationV1(run_id=args.run_id or "eval-" + uuid4().hex,
                lineage_id=str(uuid4()), plan=plan, candidates=candidates)
            summary = execute_evaluation(run, args.root, sink=TerminalProgress(sys.stdout, mode=args.format))
            return 0 if summary["execution"] == "completed" else 130 if summary["execution"] == "interrupted" else 3
        elif args.command == "resume":
            run = EvaluationRunDeclarationV1.model_validate_json((args.root / "run.json").read_bytes())
            summary = execute_evaluation(run, args.root, resume=True, sink=TerminalProgress(sys.stdout, mode=args.format))
            return 0 if summary["execution"] == "completed" else 130 if summary["execution"] == "interrupted" else 3
        elif args.command == "watch":
            sink = TerminalProgress(sys.stdout, mode=args.format)
            run = json.loads((args.root / "run.json").read_bytes())
            after = args.after
            if args.timeout <= 0 or args.timeout > 3600:
                raise ValueError("watch_timeout_outside_supported_bounds")
            deadline = time.monotonic() + args.timeout
            while True:
                page = read_operator_events(args.root / "events.jsonl", run_id=run["run_id"], after_sequence=after, limit=1000)
                for event in page["events"]:
                    sink(event)
                after = page["next_sequence"]
                if page["has_more"]:
                    continue
                active = False
                with (args.root / "owner.lock").open("rb") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        active = True
                if not args.follow or ((args.root / "summary.json").exists() and not active) or time.monotonic() >= deadline:
                    return 0
                time.sleep(.1)
        elif args.command == "results":
            output = json.loads((args.root / "summary.json").read_bytes())
        elif args.command == "cancel":
            run = EvaluationRunDeclarationV1.model_validate_json((args.root / "run.json").read_bytes())
            durable_write(args.root / "cancel.json", canonical_bytes({"run_id": run.run_id, "action": "cancel"}))
            output = {"run_id": run.run_id, "cancel": "requested", "execution": "not_observed"}
        elif args.command == "export":
            output = export_evaluation(args.root, args.destination)
        else:
            bindings = candidates_from(args.candidate, relative=Path.cwd()) if args.candidate else None
            child = fork_evaluation(args.root, args.destination, run_id=args.run_id,
                candidates=bindings, child_budget=read_config(args.child_budget))
            output = child.model_dump(mode="json")
        print(json.dumps(output, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps(refusal_payload(exc), sort_keys=True), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
