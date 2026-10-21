"""Provider-free package planning and explicit package-owned local builds."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .package_build import build_container_package, plan_container_package
from .package_files import ContainerPackageError
from .package_models import (
    ContainerArtifactDescriptor,
    ContainerPackageManifest,
    PackageBuildOutput,
    PackageBuildPlan,
)

SCHEMAS = {
    "manifest": ContainerPackageManifest,
    "plan": PackageBuildPlan,
    "build-output": PackageBuildOutput,
    "artifact": ContainerArtifactDescriptor,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-containers package")
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema", help="print a versioned JSON schema without provider access")
    schema.add_argument("kind", choices=tuple(SCHEMAS), default="manifest", nargs="?")
    plan = sub.add_parser("plan", help="freeze/validate TOML or JSON without executing code")
    plan.add_argument("manifest", type=Path)
    build = sub.add_parser("build", help="execute the package build/resolve entrypoint")
    build.add_argument("manifest", type=Path)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--timeout-seconds", type=float, default=300)
    build.add_argument("--max-log-bytes", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            print(json.dumps(SCHEMAS[args.kind].model_json_schema(), indent=2, sort_keys=True))
            return 0
        if args.command == "plan":
            value = plan_container_package(args.manifest)
        else:
            value = asyncio.run(
                build_container_package(
                    args.manifest,
                    args.output,
                    timeout_seconds=args.timeout_seconds,
                    max_log_bytes=args.max_log_bytes,
                )
            )
        print(value.model_dump_json(indent=2))
        return 0
    except ContainerPackageError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 2
