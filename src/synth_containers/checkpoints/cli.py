"""Provider-free checkpoint inspection, export and restore preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from .contracts import CheckpointFailure
from .portable import (
    CheckpointArtifactReader,
    CheckpointBundleStore,
    check_checkpoint_restore,
    export_checkpoint_bundle,
    load_checkpoint_bundle,
    read_checkpoint_descriptor,
)
from .portable_models import (
    CheckpointInspectionV1,
    CheckpointRestoreReportV1,
    CheckpointRestoreRequestV1,
    CheckpointRuntimeSupportV1,
    PortableCheckpointManifestV1,
)

SCHEMAS = {
    "manifest": PortableCheckpointManifestV1,
    "request": CheckpointRestoreRequestV1,
    "runtime-support": CheckpointRuntimeSupportV1,
    "report": CheckpointRestoreReportV1,
    "inspection": CheckpointInspectionV1,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-containers checkpoint")
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema", help="print a versioned JSON schema")
    schema.add_argument("kind", choices=tuple(SCHEMAS), default="manifest", nargs="?")
    inspect = sub.add_parser("inspect", help="verify a bundle without decoding participant state")
    inspect.add_argument("bundle", type=Path)
    export = sub.add_parser("export", help="export a manifest and its complete local blob closure")
    export.add_argument("manifest", type=Path)
    export.add_argument("--store", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check", help="check bytes and explicit restore requirements without execution")
    check.add_argument("bundle", type=Path)
    check.add_argument("--request", type=Path, required=True)
    check.add_argument("--runtime-support", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            print(json.dumps(SCHEMAS[args.kind].model_json_schema(), indent=2, sort_keys=True))
            return 0
        if args.command == "export":
            manifest = PortableCheckpointManifestV1.model_validate(read_checkpoint_descriptor(args.manifest))
            value = export_checkpoint_bundle(
                manifest,
                CheckpointArtifactReader(args.store, namespace=manifest.application.namespace),
                args.output,
            )
        else:
            inspection = load_checkpoint_bundle(args.bundle)
            if args.command == "inspect":
                value = inspection
            else:
                value = check_checkpoint_restore(
                    inspection.manifest,
                    CheckpointBundleStore(args.bundle, namespace=inspection.manifest.application.namespace),
                    CheckpointRestoreRequestV1.model_validate(read_checkpoint_descriptor(args.request)),
                    CheckpointRuntimeSupportV1.model_validate(read_checkpoint_descriptor(args.runtime_support)),
                )
                print(value.model_dump_json(indent=2))
                return 0 if value.declaration_supported else 2
        print(value.model_dump_json(indent=2))
        return 0
    except ValidationError:
        print(json.dumps({"error": "checkpoint_descriptor_invalid"}))
    except CheckpointFailure as error:
        print(json.dumps({"error": str(error)}))
    except OSError:
        print(json.dumps({"error": "checkpoint_io_unavailable"}))
    return 2
