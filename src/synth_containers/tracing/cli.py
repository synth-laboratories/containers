"""``synth-trace`` — inspect, validate, project, rebuild, and repair local bundles.

Every subcommand operates on a bundle directory and prints JSON to stdout, so the same
commands work as acceptance evidence and as a debugging tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical import readable_json
from .capture.spool import repair as repair_spool
from .projections.inspector import load_bundle, summarize
from .projections.v4 import project_v4
from .store.bundle import LocalTraceBundle, rebuild_catalog
from .validation.schema import all_schemas
from .validation.validator import validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-trace", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="summarize every trace in a bundle")
    inspect.add_argument("bundle", type=Path)

    validate_parser = subparsers.add_parser("validate", help="run invariants over a bundle")
    validate_parser.add_argument("bundle", type=Path)

    project = subparsers.add_parser("project", help="write a projection of a sealed trace")
    project.add_argument("bundle", type=Path)
    project.add_argument("--format", default="v4", choices=["v4"])

    rebuild = subparsers.add_parser("rebuild", help="rebuild the SQLite catalog projection")
    rebuild.add_argument("bundle", type=Path)

    repair_parser = subparsers.add_parser("repair", help="repair an interrupted capture spool")
    repair_parser.add_argument("capture_root", type=Path)
    repair_parser.add_argument("--capture-id", required=True)

    verify = subparsers.add_parser("verify", help="prove a bundle is self-contained")
    verify.add_argument("bundle", type=Path)

    schemas = subparsers.add_parser("schemas", help="emit generated JSON Schemas")
    schemas.add_argument("--out", type=Path)

    args = parser.parse_args(argv)

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
        print(readable_json(receipts))
        return 1 if failed else 0

    if args.command == "project":
        bundle = LocalTraceBundle(args.bundle)
        written = []
        for inspected in load_bundle(args.bundle):
            trace, manifest = project_v4(inspected.trace)
            path, sealed = bundle.write_projection(manifest, trace.to_dict(), kind="v4")
            bundle.write_receipt("projection-v4", sealed)
            written.append({"path": str(path), "manifest": sealed.to_dict()})
        print(readable_json(written))
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
