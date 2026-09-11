"""Catalog containers and hosted pool execution, observation, and deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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
from .pools import PoolClient, PoolClientError
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

    recovery = sub.add_parser("harbor-daytona-reconcile", help="reconcile an expired native Harbor trial")
    recovery.add_argument("trial_dir", type=Path)

    watch = sub.add_parser("watch", help="observe a hosted rollout from a saved sequence")
    watch.add_argument("rollout_id")
    watch.add_argument("--after-sequence", type=int, default=0)
    watch.add_argument("--timeout-seconds", type=float, default=300.0)

    submit = sub.add_parser("submit", help="durably submit a hosted rollout request")
    submit.add_argument("pool_id")
    submit.add_argument("request", type=Path)
    submit.add_argument("--idempotency-key", required=True)
    get = sub.add_parser("get", help="read a saved hosted rollout")
    get.add_argument("rollout_id")
    result = sub.add_parser("result", help="read and verify a committed hosted result snapshot")
    result.add_argument("rollout_id")
    cancel = sub.add_parser(
        "cancel", help="request cancellation; remote stop may remain unconfirmed"
    )
    cancel.add_argument("rollout_id")

    lease_assign = sub.add_parser("lease-assign", help="acquire a project lease with verified placement")
    lease_assign.add_argument("project_id")
    lease_assign.add_argument("--image-kind", required=True)
    lease_assign.add_argument("--substrate", required=True, choices=("docker", "daytona"))
    lease_assign.add_argument("--idempotency-key", required=True)
    lease_assign.add_argument("--ttl-seconds", type=int, default=900)
    for action in ("get", "renew", "release"):
        command = sub.add_parser(f"lease-{action}", help=f"{action} a hosted container lease")
        command.add_argument("lease_id")
        if action == "renew":
            command.add_argument("--ttl-seconds", type=int, default=900)

    for action in ("get", "create", "update", "delete", "operation", "lookup"):
        command = sub.add_parser(
            f"deployment-{action}", help=f"{action} a project-owned hosted deployment"
        )
        command.add_argument("pool_id")
        command.add_argument(
            "resource_id", help="task ID, or operation ID for deployment-operation"
        )
        command.add_argument("--project-id", required=True)
        if action in {"create", "update"}:
            command.add_argument("request", type=Path)
        if action in {"create", "update", "delete", "lookup"}:
            command.add_argument("--idempotency-key", required=True)
        if action in {"update", "delete"}:
            command.add_argument("--expected-revision", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "harbor-stage":
            from .harbor_environment import (
                HarborProviderCompatibility,
                HarborResourceRequest,
                inspect_harbor_package,
                register_harbor_environment,
            )
            from .harbor_task_stage import stage_native_harbor_task

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
        if args.command.startswith("lease-"):
            async def operate_lease() -> dict:
                async with PoolClient.from_env() as client:
                    if args.command == "lease-assign":
                        return await client.assign_lease(
                            project_id=args.project_id, image_kind=args.image_kind,
                            substrate=args.substrate, idempotency_key=args.idempotency_key,
                            ttl_seconds=args.ttl_seconds,
                        )
                    if args.command == "lease-get":
                        return await client.get_lease(args.lease_id)
                    if args.command == "lease-renew":
                        return await client.renew_lease(args.lease_id, ttl_seconds=args.ttl_seconds)
                    return await client.release_lease(args.lease_id)
            print(json.dumps(asyncio.run(operate_lease()), sort_keys=True))
            return 0
        if args.command.startswith("deployment-"):
            action = args.command.removeprefix("deployment-")
            deployment_payload = {}
            if action in {"create", "update"}:
                with args.request.open("rb") as handle:
                    raw = handle.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("deployment request exceeds 1 MiB")
                deployment_payload = json.loads(raw)
                if not isinstance(deployment_payload, dict):
                    raise ValueError("deployment request must be a JSON object")
                json.dumps(deployment_payload, allow_nan=False)

            async def operate_deployment() -> dict:
                async with PoolClient.from_env() as client:
                    if action == "get":
                        return await client.get_deployment(
                            args.pool_id, args.resource_id, project_id=args.project_id
                        )
                    if action == "lookup":
                        return await client.find_deployment_operation(
                            args.pool_id,
                            args.resource_id,
                            project_id=args.project_id,
                            idempotency_key=args.idempotency_key,
                        )
                    if action == "operation":
                        return await client.get_deployment_operation(
                            args.pool_id, args.resource_id, project_id=args.project_id
                        )
                    return await client.mutate_deployment(
                        args.pool_id,
                        args.resource_id,
                        project_id=args.project_id,
                        operation=action,
                        idempotency_key=args.idempotency_key,
                        payload=deployment_payload,
                        expected_revision=getattr(args, "expected_revision", None),
                    )

            print(json.dumps(asyncio.run(operate_deployment()), sort_keys=True))
            return 0
        if args.command in {"submit", "get", "result", "cancel"}:
            payload = None
            if args.command == "submit":
                with args.request.open("rb") as handle:
                    raw = handle.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("rollout request exceeds 1 MiB")
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("rollout request must be a JSON object")
                if payload.get("idempotency_key", args.idempotency_key) != args.idempotency_key:
                    raise ValueError("request idempotency key conflicts with --idempotency-key")
                payload["idempotency_key"] = args.idempotency_key
                if (
                    not 1 <= len(args.idempotency_key) <= 128
                    or args.idempotency_key.strip() != args.idempotency_key
                ):
                    raise ValueError("idempotency key must be 1-128 non-padded characters")
                json.dumps(payload, allow_nan=False)

            async def operate() -> dict:
                async with PoolClient.from_env() as client:
                    if args.command == "submit":
                        rollout_id = await client.submit(args.pool_id, payload)
                        return {"rollout_id": rollout_id, "idempotency_key": args.idempotency_key}
                    if args.command == "get":
                        return await client.get_rollout(args.rollout_id)
                    if args.command == "result":
                        return await client.get_result_snapshot(args.rollout_id)
                    return await client.cancel(args.rollout_id)

            print(json.dumps(asyncio.run(operate()), sort_keys=True))
            return 0
        if args.command == "watch":

            async def observe() -> None:
                async with PoolClient.from_env() as client:
                    async for event in client.watch_events(
                        args.rollout_id,
                        after_sequence=args.after_sequence,
                        timeout_seconds=args.timeout_seconds,
                    ):
                        print(json.dumps(event, sort_keys=True), flush=True)

            asyncio.run(observe())
            return 0
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
    except (LaunchError, PoolClientError, ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
