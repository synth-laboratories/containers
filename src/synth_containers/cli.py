"""``synth-containers`` — the only thing that starts or stops catalog images."""

from __future__ import annotations

import argparse
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
    up.add_argument("--env", action="append", default=None, help="KEY=VALUE or KEY (forward from host)")
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

    args = parser.parse_args(argv)
    try:
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
