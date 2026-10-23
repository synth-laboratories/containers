"""Inspect and regrade retained evidence with pinned provider-free rubrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
import tomllib

from .grading import preflight_grading, run_grading, publish_grading, export_grading, read_grading_evidence
from .grading_models import GradingPlanV1, GradingResultV1
from .tracing.store.bundle import LocalTraceBundle
from .tracing.store.filesystem import FilesystemBlobStore
from .tracing.validation.rehydrate import trace_document_from_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-containers regrade", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("describe", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("plan", type=Path)
        sub.add_argument("--bundle", type=Path, required=command == "describe")
        sub.add_argument("--state-root", type=Path, required=command == "run")
        sub.add_argument("--trust-source-digest", action="append", default=[])
        if command == "run":
            sub.add_argument("--retry-interrupted-offline", action="store_true")
    for command in ("inspect", "publish", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("operation")
        sub.add_argument("--state-root", type=Path, required=True)
        if command == "publish":
            sub.add_argument("--bundle", type=Path, required=True)
        if command == "export":
            sub.add_argument("--destination", type=Path, required=True)
    commands.add_parser("schema")
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            output = GradingPlanV1.model_json_schema()
        elif args.command in {"describe", "run"}:
            body = tomllib.loads(args.plan.read_text()) if args.plan.suffix == ".toml" else json.loads(args.plan.read_bytes())
            plan = GradingPlanV1.model_validate(body)
            bundle = LocalTraceBundle(args.bundle) if args.bundle is not None else None
            document = trace_document_from_payload(bundle.read_trace(plan.trace_digest)) if bundle else None
            trust = frozenset(args.trust_source_digest)
            if args.command == "describe":
                assert bundle is not None and document is not None
                preflight = preflight_grading(plan, document, bundle.blobs, trusted_source_digests=trust)
                output = {"plan": plan.model_dump(mode="json"), "preflight": preflight.model_dump(mode="json")}
                print(json.dumps(output, sort_keys=True))
                return 0 if preflight.sufficient else 2
            blobs = bundle.blobs if bundle else FilesystemBlobStore(args.state_root / plan.operation_id / "inputs")
            result = run_grading(plan, document, blobs, args.state_root,
                trusted_source_digests=trust, retry_interrupted=args.retry_interrupted_offline, source_bundle=bundle)
            output = result.model_dump(mode="json")
            print(json.dumps(output, sort_keys=True))
            return 0 if result.execution_status == "completed" else 130 if result.execution_status == "interrupted" else 3
        else:
            # Validate the path component before opening anything underneath it.
            from .package_models import Name
            from pydantic import TypeAdapter
            operation = TypeAdapter(Name).validate_python(args.operation)
            directory = args.state_root / operation
            result = GradingResultV1.model_validate_json((directory / "result.json").read_bytes())
            if result.operation_id != operation:
                raise ValueError("grading_operation_identity_mismatch")
            read_grading_evidence(directory, result)
            if args.command == "publish":
                output = {"evidence_digest": publish_grading(LocalTraceBundle(args.bundle), args.state_root, result)}
            elif args.command == "export":
                bundle = export_grading(args.state_root, result, args.destination)
                output = {"bundle_manifest_digest": bundle.read_manifest()["content_digest"],
                          "self_contained": bundle.verify_self_contained()[0]}
            else:
                output = result.model_dump(mode="json")
        print(json.dumps(output, sort_keys=True))
        return 0
    except (ValueError, OSError) as exc:
        # Files and private grader diagnostics stay in the private retained state.
        reason = str(exc) if isinstance(exc, ValueError) and re.fullmatch(r"[a-z0-9_:; -]{1,512}", str(exc)) else type(exc).__name__
        print(json.dumps({"error": "grading_refused", "reason": reason}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
