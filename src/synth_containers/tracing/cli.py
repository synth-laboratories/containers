"""``synth-trace`` — inspect, validate, project, rebuild, and repair local bundles.

Every subcommand operates on a bundle directory and prints JSON to stdout, so the same
commands work as acceptance evidence and as a debugging tool.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from .canonical import readable_json
from .canonical import record_id, utc_now
from .capture.spool import repair as repair_spool
from .projections.inspector import load_bundle, summarize
from .projections.v4 import project_v4
from .projections.derived import PROJECTIONS
from .adapters.atif import export_atif
from .models.projection import ProjectionLossV1, ProjectionManifestV1
from .store.bundle import LocalTraceBundle, rebuild_catalog
from .validation.schema import all_schemas
from .validation.validator import validate
from .adapters.legacy import import_legacy
from .adapters.native import import_native_to_bundle, write_imported_document
from .canonical import bytes_digest
from .native_evaluation import attach_native_evaluation
from .capture.binding import CaptureMode, Interception, WorkloadKind
from .capture.runner import run_captured_command
from .capture.supervisor import SupervisorConfig
from .models.identity import TraceKind, TraceProvenanceV5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-trace", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="summarize every trace in a bundle")
    inspect.add_argument("bundle", type=Path)

    validate_parser = subparsers.add_parser("validate", help="run invariants over a bundle")
    validate_parser.add_argument("bundle", type=Path)

    project = subparsers.add_parser("project", help="write a projection of a sealed trace")
    project.add_argument("bundle", type=Path)
    project.add_argument(
        "--format",
        default="v4",
        choices=["v4", "atif", "transcript", "memory", "training", "logprobs", "event_history"],
    )

    archive = subparsers.add_parser("archive", help="write a verified deterministic bundle ZIP")
    archive.add_argument("bundle", type=Path)
    archive.add_argument("output", type=Path)

    search = subparsers.add_parser("search", help="full-text search a rebuilt bundle catalog")
    search.add_argument("bundle", type=Path)
    search.add_argument("query")
    search.add_argument("--trace-digest")
    search.add_argument("--limit", type=int, default=100)

    import_parser = subparsers.add_parser(
        "import",
        help="import a native/legacy trace into a bundle or standalone sealed V5 JSON",
    )
    import_parser.add_argument("source", nargs="?", type=Path)
    import_parser.add_argument("--format", required=True)
    import_parser.add_argument("--input", dest="input_path", type=Path)
    import_parser.add_argument("--bundle", type=Path)
    import_parser.add_argument("--out", type=Path)

    attach_parser = subparsers.add_parser(
        "attach",
        help="append typed evidence to a sealed bundle without mutating its trace",
    )
    attach_parser.add_argument("bundle", type=Path)
    attach_parser.add_argument("--native-eval", type=Path, required=True)

    extract = subparsers.add_parser(
        "extract",
        help="safely extract and verify a portable bundle ZIP",
    )
    extract.add_argument("archive", type=Path)
    extract.add_argument("target", type=Path)

    rebuild = subparsers.add_parser("rebuild", help="rebuild the SQLite catalog projection")
    rebuild.add_argument("bundle", type=Path)

    repair_parser = subparsers.add_parser("repair", help="repair an interrupted capture spool")
    repair_parser.add_argument("capture_root", type=Path)
    repair_parser.add_argument("--capture-id", required=True)

    verify = subparsers.add_parser("verify", help="prove a bundle is self-contained")
    verify.add_argument("bundle", type=Path)

    schemas = subparsers.add_parser("schemas", help="emit generated JSON Schemas")
    schemas.add_argument("--out", type=Path)

    run = subparsers.add_parser(
        "run",
        help="run an unchanged command under a local capture supervisor",
    )
    run.add_argument("--binding", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--upstream-base-url")
    run.add_argument("--anthropic-base-url")
    run.add_argument(
        "--mode",
        choices=[str(item) for item in CaptureMode],
        default=str(CaptureMode.REQUIRED),
    )
    run.add_argument(
        "--interception",
        choices=[str(item) for item in Interception],
        default=str(Interception.PROVIDER_PROXY),
    )
    run.add_argument(
        "--workload-kind",
        choices=[str(item) for item in WorkloadKind],
        default=str(WorkloadKind.REACT),
    )
    run.add_argument(
        "--trace-kind",
        choices=[str(item) for item in TraceKind],
        default=str(TraceKind.AGENT_ROLLOUT),
    )
    run.add_argument("--root-actor-name", default="workload")
    run.add_argument("--trace-key", default="{}")
    run.add_argument("--timeout-seconds", type=float)
    run.add_argument(
        "--project",
        action="append",
        choices=["v4", "atif"],
        dest="projections",
    )
    run.add_argument("child_command", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    if args.command == "run":
        child_command = tuple(args.child_command)
        if child_command[:1] == ("--",):
            child_command = child_command[1:]
        if not child_command:
            raise SystemExit("run requires a command after --")
        trace_key = json.loads(args.trace_key)
        if not isinstance(trace_key, dict):
            raise SystemExit("--trace-key must be a JSON object")
        upstream_base_url = (
            args.upstream_base_url
            or os.environ.get("SYNTH_TRACE_UPSTREAM_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        producer_version = version("synth-containers")
        result = run_captured_command(
            SupervisorConfig(
                bundle_root=args.output,
                trace_key=trace_key or {"command": Path(child_command[0]).name},
                upstream_base_url=upstream_base_url,
                anthropic_base_url=args.anthropic_base_url,
                provenance=TraceProvenanceV5(
                    producer="synth-trace-run",
                    producer_version=producer_version,
                    producer_commit=os.environ.get("SYNTH_TRACE_PRODUCER_COMMIT"),
                    harness="synth-trace run",
                ),
                workload_kind=args.workload_kind,
                trace_kind=args.trace_kind,
                root_actor_name=args.root_actor_name,
                mode=args.mode,
                interception=args.interception,
                binding_path=args.binding,
            ),
            child_command,
            timeout_seconds=args.timeout_seconds,
            projections=tuple(args.projections or ("v4",)),
        )
        print(readable_json(result.receipt), file=sys.stderr)
        return result.exit_code

    if args.command == "inspect":
        print(readable_json([summarize(item) for item in load_bundle(args.bundle)]))
        return 0

    if args.command == "validate":
        receipts = []
        failed = False
        for inspected in load_bundle(args.bundle):
            receipt = validate(inspected.trace, inspected.evidence)
            receipts.append(receipt.to_dict())
            failed = failed or not receipt.valid
        bundle = LocalTraceBundle(args.bundle)
        bundle.write_receipt("validation", receipts)
        bundle.write_manifest()
        print(readable_json(receipts))
        return 1 if failed else 0

    if args.command == "project":
        bundle = LocalTraceBundle(args.bundle)
        written = []
        for inspected in load_bundle(args.bundle):
            if args.format == "v4":
                trace, manifest = project_v4(inspected.trace)
                payload = trace.to_dict()
                kind = "v4"
            else:
                kind = args.format
                payload = (
                    export_atif(inspected.trace)
                    if kind == "atif"
                    else PROJECTIONS[kind](inspected.trace)
                )
                losses = tuple(
                    ProjectionLossV1(field_path="*", reason=str(item), record_count=0)
                    for item in payload.get("losses")
                    or (payload.get("extra") or {}).get("projection_losses")
                    or ()
                )
                manifest = ProjectionManifestV1(
                    projection_id=record_id(
                        "proj",
                        kind=kind,
                        scope=(inspected.trace.trace_id,),
                        key=inspected.trace.content_digest,
                    ),
                    format=kind,
                    source_trace_id=inspected.trace.trace_id,
                    source_trace_digest=inspected.trace.content_digest,
                    producer="synth_containers.tracing.cli",
                    producer_version="1",
                    created_at=utc_now(),
                    losses=losses,
                )
            path, sealed = bundle.write_projection(manifest, payload, kind=kind)
            bundle.write_receipt(f"projection-{kind}", sealed)
            written.append({"path": str(path), "manifest": sealed.to_dict()})
        bundle.write_manifest()
        print(readable_json(written))
        return 0

    if args.command == "archive":
        bundle = LocalTraceBundle(args.bundle)
        digest = bundle.write_archive(args.output)
        print(readable_json({"path": str(args.output), "bytes_digest": digest}))
        return 0

    if args.command == "search":
        catalog = LocalTraceBundle(args.bundle).open_catalog()
        try:
            rows = list(
                catalog.search(
                    args.query,
                    trace_digest=args.trace_digest,
                    limit=args.limit,
                )
            )
        finally:
            catalog.close()
        print(readable_json(rows))
        return 0

    if args.command == "import":
        source = args.input_path or args.source
        if source is None:
            raise SystemExit("import requires a positional source or --input")
        if args.input_path is not None and args.source is not None:
            raise SystemExit("use either a positional source or --input, not both")
        if bool(args.bundle) == bool(args.out):
            raise SystemExit("import requires exactly one of --bundle or --out")
        if args.bundle is not None:
            bundle = LocalTraceBundle(args.bundle)
            normalized = str(args.format).lower().strip().replace("-", "_")
            if normalized in {
                "codex",
                "codex_jsonl",
                "codex_stdout_jsonl",
                "react",
                "craftax_react",
                "gamebench_react",
                "jesterky",
                "jesterky_manifest",
            }:
                result = import_native_to_bundle(
                    source,
                    source_format=args.format,
                    bundle=bundle,
                )
            else:
                source_bytes = source.read_bytes()
                payload = json.loads(source_bytes.decode("utf-8"))
                imported = import_legacy(payload, source_format=args.format)
                if imported.canonical is None:
                    raise SystemExit(
                        f"{args.format} is preserved as opaque input and has no "
                        "canonical assembler"
                    )
                stored = bundle.blobs.put(source_bytes)
                if stored != bytes_digest(source_bytes):
                    raise SystemExit("source bytes changed while importing")
                result = write_imported_document(
                    imported.canonical,
                    source_digest=imported.source_digest,
                    source_format=args.format,
                    bundle=bundle,
                )
                result["coverage"] = imported.coverage
                result["losses"] = list(imported.losses)
            result["bundle"] = str(args.bundle)
            print(readable_json(result))
            return 0
        payload = json.loads(source.read_text(encoding="utf-8"))
        imported = import_legacy(payload, source_format=args.format)
        if imported.canonical is None:
            raise SystemExit(
                f"{args.format} is preserved as opaque input and has no canonical assembler"
            )
        assert args.out is not None
        args.out.write_text(readable_json(imported.canonical) + "\n", encoding="utf-8")
        print(
            readable_json(
                {
                    "path": str(args.out),
                    "trace_id": imported.canonical.trace_id,
                    "trace_digest": imported.canonical.content_digest,
                    "coverage": imported.coverage,
                    "losses": list(imported.losses),
                }
            )
        )
        return 0

    if args.command == "attach":
        payload = json.loads(args.native_eval.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SystemExit("--native-eval must contain a JSON object")
        print(
            readable_json(
                attach_native_evaluation(
                    args.bundle,
                    payload=payload,
                    source_name=args.native_eval.name,
                )
            )
        )
        return 0

    if args.command == "extract":
        bundle = LocalTraceBundle.extract_archive(args.archive, args.target)
        print(
            readable_json(
                {
                    "archive": str(args.archive),
                    "bundle": str(bundle.root),
                    "manifest_digest": bundle.read_manifest()["content_digest"],
                }
            )
        )
        return 0

    if args.command == "rebuild":
        print(readable_json(rebuild_catalog(LocalTraceBundle(args.bundle))))
        return 0

    if args.command == "repair":
        result = repair_spool(args.capture_root, capture_id=args.capture_id)
        print(readable_json(result))
        return 0

    if args.command == "verify":
        ok, missing = LocalTraceBundle(args.bundle).verify_self_contained()
        print(readable_json({"self_contained": ok, "missing": list(missing)}))
        return 0 if ok else 1

    if args.command == "schemas":
        payload = all_schemas()
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            for name, schema in payload.items():
                (args.out / f"{name}.schema.json").write_text(
                    json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            print(readable_json({"written": sorted(payload)}))
            return 0
        print(readable_json(payload))
        return 0

    raise SystemExit(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
