"""``synth-containers`` — the only thing that starts or stops catalog images."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .launch import (
    LaunchError,
    build_image,
    catalog_payload,
    down_image,
    logs_image,
    status_payload,
    up_image,
)
from .serve import main as serve_target


def _env_pairs(items: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            value = os.environ.get(item)
            if value is None:
                raise LaunchError(f"container_image_env_missing:{item}")
            env[item] = value
            continue
        key, value = item.split("=", 1)
        env[key.strip()] = value
    return env


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "tunnel":
        from .tunnels.cli import main as tunnel_main

        return tunnel_main(arguments[1:])
    parser = argparse.ArgumentParser(prog="synth-containers", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run create_compat_app for a public target_id (demo path)")
    serve.add_argument("--target", default=os.environ.get("SYNTH_CONTAINER_TARGET", "openenv_echo"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--storage-root", default=None)
    serve.add_argument("--allow-non-loopback", action="store_true")

    up = sub.add_parser("up", help="docker build+run a catalog image and wait for /health")
    up.add_argument("image_id")
    up.add_argument("--catalog", type=Path, default=None)
    up.add_argument("--host", default="127.0.0.1")
    up.add_argument("--port", type=int, default=None)
    up.add_argument(
        "--env", action="append", default=None, help="KEY=VALUE or KEY (forward from host)"
    )
    up.add_argument("--replace", action="store_true")
    up.add_argument("--no-build", action="store_true")
    up.add_argument("--pull", action="store_true")
    up.add_argument("--startup-timeout-seconds", type=float, default=None)

    down = sub.add_parser("down", help="reap labelled siblings, stop + remove the platform")
    down.add_argument("image_id")
    down.add_argument("--catalog", type=Path, default=None)
    down.add_argument("--port", type=int, default=None)

    build = sub.add_parser("build", help="docker build a catalog image on the local daemon")
    build.add_argument("image_id")
    build.add_argument("--catalog", type=Path, default=None)

    catalog = sub.add_parser("catalog", help="list catalog ids")
    catalog.add_argument("--catalog", type=Path, default=None)

    sub.add_parser("ps", help="list run records")

    logs = sub.add_parser("logs", help="docker logs for a running image")
    logs.add_argument("image_id")
    logs.add_argument("--port", type=int, default=None)
    logs.add_argument("--tail", type=int, default=200)

    tunnel = sub.add_parser("tunnel", help="shared connector lifecycle")
    tunnel.add_argument("arguments", nargs=argparse.REMAINDER)

    journal = sub.add_parser("journal", help="replay a local operator journal without provider access")
    journal.add_argument("path", type=Path)
    journal.add_argument("--run-id", default=None)
    journal.add_argument("--after-sequence", type=int, default=0)
    journal.add_argument("--limit", type=int, default=100)
    journal.add_argument("--follow", action="store_true")
    journal.add_argument("--timeout-seconds", type=float, default=300)

    stage = sub.add_parser("harbor-stage", help="bind a native Harbor task to a prebuilt image")
    stage.add_argument("source", type=Path)
    stage.add_argument("destination", type=Path)
    stage.add_argument("--image", required=True)
    stage.add_argument("--provider", required=True, choices=["docker", "daytona"])
    stage.add_argument("--creation-timeout-seconds", type=int, default=300)
    stage.add_argument("--resource-ttl-minutes", type=int, default=20)
    stage.add_argument("--cpus", type=int)
    stage.add_argument("--memory-mb", type=int)
    stage.add_argument("--storage-mb", type=int)
    stage.add_argument("--docker-resource-custody", action="store_true")
    stage.add_argument("--docker-egress-image")
    stage.add_argument("--environment-transfer", choices=["preserve", "image_only"], default="preserve")
    stage.add_argument("--work-timeout-seconds", type=float)
    stage.add_argument("--verifier-timeout-seconds", type=float)

    context = sub.add_parser("harbor-image-context", help="freeze a Harbor image context without provider access")
    context.add_argument("source", type=Path)
    context.add_argument("destination", type=Path)

    recovery = sub.add_parser("harbor-daytona-reconcile", help="reconcile an expired native Harbor trial")
    recovery.add_argument("trial_dir", type=Path)
    docker_recovery = sub.add_parser("harbor-docker-reconcile", help="reconcile a native Docker trial after its worker exits")
    docker_recovery.add_argument("trial_dir", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "harbor-image-context":
            from .harbor_environment import inspect_harbor_package
            from .harbor_image_context import freeze_harbor_image_context

            context = freeze_harbor_image_context(inspect_harbor_package(args.source), args.destination)
            print(json.dumps(context.as_dict(), sort_keys=True))
            return 0
        if args.command == "harbor-stage":
            from .harbor_environment import (
                HarborProviderCompatibility,
                HarborResourceRequest,
                inspect_harbor_package,
                register_harbor_environment,
            )
            from .harbor_phase_limits import NativeHarborPhaseLimits
            from .harbor_task_stage import stage_native_harbor_task

            phase_limits = None
            if args.work_timeout_seconds is not None or args.verifier_timeout_seconds is not None:
                phase_limits = NativeHarborPhaseLimits(args.work_timeout_seconds, args.verifier_timeout_seconds)

            release = register_harbor_environment(
                inspect_harbor_package(args.source), agent_image=args.image, verifier_image=args.image,
                provider=HarborProviderCompatibility(
                    provider_id=args.provider, supports_separate_verifier=False,
                ),
            )
            print(json.dumps(stage_native_harbor_task(
                release, args.destination, creation_timeout_seconds=args.creation_timeout_seconds,
                resource_ttl_minutes=args.resource_ttl_minutes,
                environment_transfer=args.environment_transfer,
                docker_resource_custody=args.docker_resource_custody,
                docker_egress_image=args.docker_egress_image,
                phase_limits=phase_limits,
                resource_request=(HarborResourceRequest(cpus=args.cpus, memory_mb=args.memory_mb,
                                                       storage_mb=args.storage_mb, gpus=0)
                                  if any(value is not None for value in (args.cpus, args.memory_mb, args.storage_mb)) else None),
            ), sort_keys=True))
            return 0
        if args.command == "harbor-daytona-reconcile":
            from daytona import AsyncDaytona

            from .harbor_daytona_recovery import reconcile_daytona_trial

            async def recover_trial() -> dict:
                async with AsyncDaytona() as client:
                    return await reconcile_daytona_trial(args.trial_dir, client)

            print(json.dumps(asyncio.run(recover_trial()), sort_keys=True))
            return 0
        if args.command == "harbor-docker-reconcile":
            from contextlib import closing

            import docker

            from .harbor_docker_recovery import reconcile_docker_trial

            with closing(docker.from_env(timeout=5)) as client:
                receipt = reconcile_docker_trial(args.trial_dir, client)
            print(json.dumps(receipt, sort_keys=True))
            return 0
        if args.command == "journal":
            from .operator_journal import follow_operator_events, read_operator_events

            if args.follow:
                for event in follow_operator_events(
                    args.path, run_id=args.run_id, after_sequence=args.after_sequence,
                    limit=args.limit, timeout_seconds=args.timeout_seconds,
                ):
                    print(json.dumps(event, sort_keys=True), flush=True)
            else:
                page = read_operator_events(
                    args.path, run_id=args.run_id,
                    after_sequence=args.after_sequence, limit=args.limit,
                )
                print(json.dumps(page, sort_keys=True))
            return 0
        if args.command == "tunnel":
            from .tunnels.cli import main as tunnel_main

            return tunnel_main(args.arguments)
        if args.command == "serve":
            serve_argv = ["--target", args.target, "--host", args.host, "--port", str(args.port)]
            if args.storage_root:
                serve_argv.extend(["--storage-root", args.storage_root])
            if args.allow_non_loopback:
                serve_argv.append("--allow-non-loopback")
            return serve_target(serve_argv)
        if args.command == "catalog":
            print(json.dumps(catalog_payload(args.catalog), indent=2, sort_keys=True))
            return 0
        if args.command == "ps":
            print(json.dumps(status_payload(), indent=2, sort_keys=True))
            return 0
        if args.command == "build":
            print(build_image(args.image_id, catalog=args.catalog))
            return 0
        if args.command == "logs":
            print(logs_image(args.image_id, port=args.port, tail=args.tail))
            return 0
        if args.command == "down":
            stopped = down_image(args.image_id, port=args.port, catalog=args.catalog)
            print(json.dumps({"id": args.image_id, "stopped": stopped}, indent=2, sort_keys=True))
            return 0
        record = up_image(
            args.image_id,
            catalog=args.catalog,
            env=_env_pairs(args.env),
            host=args.host,
            port=args.port,
            replace=args.replace,
            build=not args.no_build,
            pull=args.pull,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
        print(json.dumps(record.to_json(), indent=2, sort_keys=True), flush=True)
        return 0
    except LaunchError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
