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
import threading
from typing import Any

from .canonical import content_digest, readable_json
from .canonical import record_id, utc_now
from .capture.live import follow_live_pages, read_live_page
from .capture.spool import repair as repair_spool
from .capture.redaction import redact_json_source_bytes
from .projections.inspector import load_bundle, summarize
from .projections.v4 import project_v4
from .projections.derived import PROJECTIONS
from .projections.visual import visual_from_sealed
from .projections.rollout_inspector import rollout_inspector_from_sealed
from .adapters.atif import export_atif
from .models.projection import (
    ProjectionLossV1,
    ProjectionManifestV1,
    bind_projection_manifest,
)
from .models.rollout_inspector import ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION
from .store.bundle import LocalTraceBundle, rebuild_catalog
from .validation.schema import all_schemas
from .validation.validator import validate
from .adapters.legacy import import_legacy
from .adapters.native import import_native_to_bundle, write_imported_document
from .native_evaluation import attach_native_evaluation
from .inspection import inspect_trace_input
from .capture.binding import (
    CaptureMode,
    Interception,
    TraceCaptureBindingV1,
    WorkloadKind,
)
from .capture.runner import run_captured_command
from .capture.supervisor import SupervisorConfig
from .capture.control_server import DetachedCaptureConfig, DetachedCaptureSupervisor
from .models.identity import TraceKind, TraceProvenanceV5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-trace", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print the installed synth-containers version")

    inspect = subparsers.add_parser("inspect", help="summarize every trace in a bundle")
    inspect.add_argument("bundle", type=Path)

    inspect_input = subparsers.add_parser(
        "inspect-input",
        help="emit the stable storage inspection contract for a bundle, ZIP, or sealed trace",
    )
    inspect_input.add_argument("source", type=Path)
    inspect_input.add_argument(
        "--archive-output",
        type=Path,
        help="write a deterministic ZIP only when the inspected bundle is trusted",
    )

    validate_parser = subparsers.add_parser("validate", help="run invariants over a bundle")
    validate_parser.add_argument("bundle", type=Path)

    project = subparsers.add_parser("project", help="write a projection of a sealed trace")
    project.add_argument("bundle", type=Path)
    project.add_argument(
        "--format",
        default="v4",
        choices=[
            "v4",
            "atif",
            "visual",
            "rollout-inspector",
            "transcript",
            "memory",
            "training",
            "logprobs",
            "event_history",
        ],
    )

    archive = subparsers.add_parser("archive", help="write a verified deterministic bundle ZIP")
    archive.add_argument("bundle", type=Path)
    archive.add_argument("output", type=Path)

    search = subparsers.add_parser(
        "search",
        help="full-text and structured search over a rebuilt bundle catalog",
    )
    search.add_argument("bundle", type=Path)
    search.add_argument("query", nargs="?")
    search.add_argument("--trace-id")
    search.add_argument("--trace-digest")
    search.add_argument("--task-id")
    search.add_argument("--run-id")
    search.add_argument("--correlation-id")
    search.add_argument("--actor-id")
    search.add_argument("--session-id")
    search.add_argument("--provider")
    search.add_argument("--model")
    search.add_argument("--event-kind")
    search.add_argument("--span-kind")
    search.add_argument("--criterion-id")
    search.add_argument("--annotation-id")
    search.add_argument("--reward-id")
    search.add_argument("--reward-min", type=float)
    search.add_argument("--reward-max", type=float)
    search.add_argument("--workflow-address")
    search.add_argument("--started-after")
    search.add_argument("--started-before")
    search.add_argument("--completeness")
    search.add_argument("--visibility")
    search.add_argument("--digest")
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

    tail = subparsers.add_parser(
        "tail",
        help="read exact raw capture envelopes from a durable live spool",
    )
    tail.add_argument("capture_root", type=Path)
    tail.add_argument("--capture-id")
    tail.add_argument("--after", "--after-ordinal", dest="after_ordinal", type=int, default=-1)
    tail.add_argument("--limit", type=int, default=256)
    tail.add_argument("--follow", action="store_true")
    tail.add_argument("--poll-seconds", type=float, default=0.25)

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
        "--inherit-env",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "copy exactly this named variable from the parent into the child; "
            "repeat for each required variable"
        ),
    )
    run.add_argument(
        "--project",
        action="append",
        choices=["v4", "atif", "visual"],
        dest="projections",
    )
    run.add_argument("child_command", nargs=argparse.REMAINDER)

    serve = subparsers.add_parser(
        "serve",
        help="serve detached request-scoped captures without a child command",
    )
    serve.add_argument("--output", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--upstream-base-url", default="https://api.openai.com/v1")
    serve.add_argument("--capture-disk-budget-bytes", type=int)
    serve.add_argument("--capture-disk-reserve-bytes", type=int, default=0)
    serve.add_argument(
        "--budget-policy",
        choices=("refuse", "evict_oldest_sealed"),
        default="refuse",
    )
    serve.add_argument(
        "--control-token-env",
        help="read the HTTP bearer token from this environment variable",
    )

    args = parser.parse_args(argv)

    if args.command == "version":
        print(version("synth-containers"))
        return 0

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
            environ=_allowlisted_environment(args.inherit_env),
            timeout_seconds=args.timeout_seconds,
            projections=tuple(args.projections or ("v4",)),
        )
        print(readable_json(result.receipt), file=sys.stderr)
        return result.exit_code

    if args.command == "serve":
        control_token = None
        if args.control_token_env:
            control_token = os.environ.get(args.control_token_env)
            if not control_token:
                raise SystemExit(
                    f"--control-token-env requested unset variable {args.control_token_env!r}"
                )
        service = DetachedCaptureSupervisor(
            DetachedCaptureConfig(
                output_root=args.output,
                host=args.host,
                port=args.port,
                upstream_base_url=args.upstream_base_url,
                capture_disk_budget_bytes=args.capture_disk_budget_bytes,
                capture_disk_reserve_bytes=args.capture_disk_reserve_bytes,
                budget_policy=args.budget_policy,
                control_token=control_token,
            )
        ).start()
        print(
            readable_json(
                {
                    "base_url": service.base_url,
                    "output": str(args.output),
                    "budget_bytes": args.capture_disk_budget_bytes,
                    "reserve_bytes": args.capture_disk_reserve_bytes,
                    "budget_policy": args.budget_policy,
                }
            ),
            flush=True,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            service.stop()
        return 0

    if args.command == "inspect-input":
        result = inspect_trace_input(
            args.source,
            archive_output=args.archive_output,
        )
        print(readable_json(result))
        return 0 if result.compatibility in {"native", "legacy_native", "opaque"} else 1

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
            binding = _projection_binding(bundle, trace=inspected.trace)
            if args.format == "v4":
                trace, manifest = project_v4(inspected.trace)
                payload = trace.to_dict()
                kind = "v4"
            else:
                kind = args.format
                if kind == "atif":
                    payload = export_atif(inspected.trace)
                elif kind == "visual":
                    payload = visual_from_sealed(
                        inspected.trace,
                        inspected.evidence,
                    ).to_dict()
                elif kind == "rollout-inspector":
                    payload = rollout_inspector_from_sealed(
                        inspected.trace,
                        inspected.evidence,
                    ).to_dict()
                else:
                    payload = PROJECTIONS[kind](inspected.trace)
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
                    format=(
                        ROLLOUT_INSPECTOR_PROJECTION_SCHEMA_VERSION
                        if kind == "rollout-inspector"
                        else kind
                    ),
                    source_trace_id=inspected.trace.trace_id,
                    source_trace_digest=inspected.trace.content_digest,
                    producer="synth_containers.tracing.cli",
                    producer_version="1",
                    created_at=utc_now(),
                    losses=losses,
                )
            manifest = bind_projection_manifest(manifest, binding)
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
                catalog.query_traces(
                    query=args.query,
                    trace_id=args.trace_id,
                    trace_digest=args.trace_digest,
                    task_id=args.task_id,
                    run_id=args.run_id,
                    correlation_id=args.correlation_id,
                    actor_id=args.actor_id,
                    session_id=args.session_id,
                    provider=args.provider,
                    model=args.model,
                    event_kind=args.event_kind,
                    span_kind=args.span_kind,
                    criterion_id=args.criterion_id,
                    annotation_id=args.annotation_id,
                    reward_id=args.reward_id,
                    reward_min=args.reward_min,
                    reward_max=args.reward_max,
                    workflow_address=args.workflow_address,
                    started_after=args.started_after,
                    started_before=args.started_before,
                    completeness=args.completeness,
                    visibility=args.visibility,
                    digest=args.digest,
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
                safe_source, source_redaction = redact_json_source_bytes(source_bytes)
                stored = bundle.blobs.put(safe_source)
                result = write_imported_document(
                    imported.canonical,
                    source_digest=imported.source_digest,
                    source_format=args.format,
                    bundle=bundle,
                    stored_source_digest=stored,
                    source_redaction=source_redaction,
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

    if args.command == "tail":
        if args.follow:
            for page in follow_live_pages(
                args.capture_root,
                expected_capture_id=args.capture_id,
                after_ordinal=args.after_ordinal,
                limit=args.limit,
                poll_seconds=args.poll_seconds,
            ):
                for envelope in page.records:
                    print(
                        json.dumps(
                            envelope.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
            return 0
        page = read_live_page(
            args.capture_root,
            expected_capture_id=args.capture_id,
            after_ordinal=args.after_ordinal,
            limit=args.limit,
        )
        for envelope in page.records:
            print(
                json.dumps(
                    envelope.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
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


def _allowlisted_environment(names: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for name in names:
        if not name or "=" in name or "\x00" in name:
            raise SystemExit("--inherit-env requires an environment variable name")
        if name not in os.environ:
            raise SystemExit(f"--inherit-env requested unset variable {name!r}")
        selected[name] = os.environ[name]
    return selected


def _projection_binding(
    bundle: LocalTraceBundle,
    *,
    trace: Any,
) -> TraceCaptureBindingV1:
    """Load and verify the exact capture authority for a bundle projection."""

    from .validation.rehydrate import build

    path = bundle.trace_root(trace.trace_id) / "binding.json"
    if not path.is_file():
        raise ValueError(f"projection source binding is missing for trace {trace.trace_id!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    binding = build(TraceCaptureBindingV1, payload)
    if not isinstance(binding, TraceCaptureBindingV1):
        raise ValueError(f"projection source binding has the wrong schema at {path}")
    if binding.trace_id != trace.trace_id:
        raise ValueError("projection source binding trace_id does not match the trace")
    if binding.binding_id != trace.capture.binding_id:
        raise ValueError("projection source binding_id does not match the trace")
    if binding.content_digest != trace.capture.binding_digest:
        raise ValueError("projection source binding digest does not match the trace")
    if binding.content_digest != content_digest(binding):
        raise ValueError("projection source binding content digest is invalid")
    return binding


if __name__ == "__main__":
    raise SystemExit(main())
