"""Provider-free package planning and explicit package-owned local builds."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .package_build import _unique_object, build_container_package, plan_container_package
from .package_files import ContainerPackageError, read_package_file
from .package_compatibility import (
    PackageCompatibilityReport,
    PackageRuntimeSupport,
    check_package_compatibility,
)
from pydantic import ValidationError
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
    "runtime-support": PackageRuntimeSupport,
    "compatibility": PackageCompatibilityReport,
}


def _read_descriptor(path: Path) -> object:
    path = path.absolute()
    try:
        return json.loads(
            read_package_file(path.parent, path.name, maximum=8 * 1024 * 1024),
            object_pairs_hook=_unique_object,
        )
    except (OSError, ValueError, UnicodeError) as error:
        raise ContainerPackageError("container_package_descriptor_invalid") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-containers package")
    sub = parser.add_subparsers(dest="command", required=True)
    schema = sub.add_parser("schema", help="print a versioned JSON schema without provider access")
    schema.add_argument("kind", choices=tuple(SCHEMAS), default="manifest", nargs="?")
    plan = sub.add_parser("plan", help="freeze/validate TOML or JSON without executing code")
    plan.add_argument("manifest", type=Path)
    check = sub.add_parser("check", help="match artifact declarations against a runtime support policy")
    check.add_argument("artifact", type=Path)
    check.add_argument("--runtime-support", type=Path, required=True)
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
        elif args.command == "check":
            value = check_package_compatibility(
                ContainerArtifactDescriptor.model_validate(_read_descriptor(args.artifact)),
                PackageRuntimeSupport.model_validate(_read_descriptor(args.runtime_support)),
            )
            print(value.model_dump_json(indent=2))
            return 0 if value.declaration_supported else 2
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
    except ValidationError:
        print(json.dumps({"error": "container_package_descriptor_invalid"}))
        return 2
    except ContainerPackageError as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 2
