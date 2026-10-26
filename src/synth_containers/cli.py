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


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _env_pairs(items: list[str] | None, env_file: Path | None = None) -> dict[str, str]:
    # An env file only supplies values for bare `--env KEY` forwards; it never
    # forwards keys wholesale.
    file_values = _read_env_file(env_file) if env_file is not None else {}
    env: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            value = os.environ.get(item, file_values.get(item))
            if value is None:
                raise LaunchError(f"container_image_env_missing:{item}")
            env[item] = value
            continue
        key, value = item.split("=", 1)
        env[key.strip()] = value
    return env


def _run_rollouts(args: argparse.Namespace) -> int:
    """Prepare, complete and score seeds one at a time, saving progress.

    A rollout takes minutes; run this from your own terminal, not a tool call
    with a short wall clock. Scored rollouts count whatever `status` says.
    """
    import json
    import time
    import urllib.error
    import urllib.request

    def call(path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            args.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)

    results: list[dict] = []
    for seed in args.seeds:
        policy = {"harness": args.harness, "config": args.config}
        started = time.time()
        row: dict = {"seed": seed}
        try:
            rollout_id = call(
                "/rollouts/prepare",
                {"policy_ref": policy, "task_instance_id": args.task_template.format(seed=seed)},
            )["rollout_id"]
            row["rollout_id"] = rollout_id
            try:
                call(f"/rollouts/{rollout_id}/complete", {})
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                row["complete_error"] = str(exc)
            view = call(f"/rollouts/{rollout_id}")
            row.update(status=view.get("status"), reward=view.get("reward"),
                       reward_status=view.get("reward_status"))
        except (urllib.error.URLError, TimeoutError, OSError, KeyError) as exc:
            row["error"] = str(exc)
        row["elapsed_seconds"] = round(time.time() - started)
        results.append(row)
        print(json.dumps(row), flush=True)
        if args.out is not None:
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    scored = [row["reward"] for row in results if row.get("reward_status") == "scored"]
    print(json.dumps({"scored": len(scored), "of": len(results),
                      "mean_reward": sum(scored) / len(scored) if scored else None}))
    return 0


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
    up.add_argument(
        "--env-file", type=Path, default=None,
        help="dotenv file consulted for bare --env KEY forwards when the host lacks KEY",
    )
    up.add_argument("--replace", action="store_true")
    up.add_argument("--no-build", action="store_true")
    up.add_argument("--pull", action="store_true")
    up.add_argument("--startup-timeout-seconds", type=float, default=None)

    rollout = sub.add_parser("rollout", help="run seeds sequentially against a running container")
    rollout.add_argument("--base-url", default="http://127.0.0.1:8080")
    rollout.add_argument("--harness", default="react")
    rollout.add_argument("--config", required=True, help="registered policy config name")
    rollout.add_argument("--task-template", default="craftax:test:{seed}")
    rollout.add_argument("--seeds", type=int, nargs="+", required=True)
    rollout.add_argument("--timeout", type=float, default=900.0)
    rollout.add_argument("--out", type=Path, default=None, help="write progress JSON here")

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
        if args.command == "rollout":
            return _run_rollouts(args)
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
            env=_env_pairs(args.env, args.env_file),
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
