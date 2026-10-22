"""Explicit Rhodes registry and rollout operations; see docs/specs/rhodes-client-v1.md."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import sys
import tomllib
from typing import Any
from urllib.parse import urlsplit

import httpx

from .pools import PoolClientError, PoolRolloutTimeout
from .rhodes_client import RhodesClient, RhodesRequestError
from .rhodes_watch import RhodesTerminal, watch_rollouts


def _payload(path: Path) -> dict:
    # Bound the read itself, including streams or files changed after stat.
    with path.open("rb") as source:
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("rhodes.config_too_large")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("rhodes.config_duplicate_key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("rhodes.config_nonfinite_number")

    try:
        body = (
            tomllib.loads(raw.decode())
            if path.suffix.lower() == ".toml"
            else json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
        )
        if not isinstance(body, dict):
            raise ValueError("rhodes.config_requires_object")
        # Also rejects overflowing exponents, TOML inf/nan and non-JSON dates.
        json.dumps(body, allow_nan=False)
    except (RecursionError, TypeError) as error:
        raise ValueError("rhodes.config_requires_bounded_json_object") from error
    return body


def _seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 86400:
        raise argparse.ArgumentTypeError("duration must be positive and at most one day")
    return seconds


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, allow_nan=False), flush=True)


async def _attached_result(client: RhodesClient, rollout_id: str, seconds: float) -> dict:
    observation = asyncio.timeout(seconds)
    try:
        async with observation:
            async for event in client.follow_events(rollout_id, timeout_seconds=seconds):
                _emit(event)
            return await client.get_rollout(rollout_id)
    except TimeoutError as error:
        if not observation.expired():
            raise
        raise PoolRolloutTimeout("Rhodes observation timed out (not cancelled)") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synth-containers rhodes", description=__doc__)
    parser.add_argument("--backend-url", default=os.environ.get("SYNTH_BACKEND_URL"))
    commands = parser.add_subparsers(dest="command", required=True)
    watch = commands.add_parser("watch")
    watch.add_argument("rollout_ids", nargs="+")
    watch.add_argument("--cursors", type=Path)
    watch.add_argument("--format", choices=("auto", "tty", "json"), default="auto")
    watch.add_argument("--timeout-seconds", type=_seconds, default=1800)
    upload = commands.add_parser("upload-source")
    upload.add_argument("path", type=Path)
    upload.add_argument("--operation-id", required=True)
    for name in ("import", "exec"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "exec":
            command.add_argument("--pool-id", required=True)
            command.add_argument("--detach", action="store_true")
            command.add_argument("--timeout-seconds", type=_seconds, default=1800)
    environments = commands.add_parser("environments").add_subparsers(dest="verb", required=True)
    listing = environments.add_parser("list")
    listing.add_argument("--after", default="")
    listing.add_argument("--limit", type=int, default=50)
    for verb in ("get", "archive", "compatibility"):
        command = environments.add_parser(verb)
        command.add_argument("release_id")
        if verb == "compatibility":
            command.add_argument("--provider", required=True)
    definitions = commands.add_parser("definitions").add_subparsers(dest="verb", required=True)
    listing = definitions.add_parser("list")
    listing.add_argument("--after-name")
    listing.add_argument("--include-archived", action="store_true")
    listing.add_argument("--limit", type=int, default=100)
    operation = definitions.add_parser("operation")
    operation.add_argument("operation_id")
    for verb in ("get", "history", "bind", "archive"):
        command = definitions.add_parser(verb)
        command.add_argument("name")
        if verb == "bind":
            command.add_argument("release_id")
        if verb in {"bind", "archive"}:
            command.add_argument("--expected-revision", type=int, required=True)
            command.add_argument("--operation-id", required=True)
        if verb == "history":
            command.add_argument("--after-revision", type=int, default=0)
            command.add_argument("--limit", type=int, default=100)
    for name in ("results", "usage", "cancel", "wait", "follow"):
        command = commands.add_parser(name)
        command.add_argument("rollout_id")
        if name in {"wait", "follow"}:
            command.add_argument("--timeout-seconds", type=_seconds, default=1800)
        if name == "follow":
            command.add_argument("--after-sequence", type=int, default=0)
    return parser


async def _execute(client: RhodesClient, args: argparse.Namespace, body: dict | None) -> int:
    if args.command == "watch":
        renderer = (
            RhodesTerminal(sys.stdout, args.rollout_ids)
            if args.format == "tty" or args.format == "auto" and sys.stdout.isatty()
            else None
        )
        if renderer is not None:
            renderer.render()
        await watch_rollouts(
            client, args.rollout_ids, cursors=body or {}, timeout_seconds=args.timeout_seconds,
            emit=renderer if renderer is not None else _emit,
        )
        return 0
    if args.command == "upload-source":
        output = await client.upload_environment_source(args.path, operation_id=args.operation_id)
    elif args.command == "import":
        assert body is not None
        output = await client.import_environment(body)
    elif args.command == "environments":
        if args.verb == "list":
            output = await client.list_environments(after=args.after, limit=args.limit)
        elif args.verb == "get":
            output = await client.get_environment(args.release_id)
        elif args.verb == "archive":
            output = await client.archive_environment(args.release_id)
        else:
            output = await client.environment_compatibility(args.release_id, provider=args.provider)
    elif args.command == "definitions":
        if args.verb == "list":
            output = await client.list_definitions(
                after_name=args.after_name, include_archived=args.include_archived, limit=args.limit
            )
        elif args.verb == "operation":
            output = await client.definition_operation(args.operation_id)
        elif args.verb == "get":
            output = await client.get_definition(args.name)
        elif args.verb == "history":
            output = await client.definition_history(
                args.name, after_revision=args.after_revision, limit=args.limit
            )
        elif args.verb == "bind":
            output = await client.bind_definition(
                args.name, args.release_id, expected_revision=args.expected_revision,
                operation_id=args.operation_id,
            )
        else:
            output = await client.archive_definition(
                args.name, expected_revision=args.expected_revision, operation_id=args.operation_id
            )
    elif args.command == "exec":
        assert body is not None
        rollout_id = await client.submit(args.pool_id, body)
        # Emit accepted coordinates before any observation so interruption never
        # hides the original run identity or asks callers to submit again.
        _emit({"rollout_id": rollout_id})
        if args.detach:
            return 0
        output = await _attached_result(client, rollout_id, args.timeout_seconds)
    elif args.command == "follow":
        async for event in client.follow_events(
            args.rollout_id, after_sequence=args.after_sequence, timeout_seconds=args.timeout_seconds
        ):
            _emit(event)
        return 0
    elif args.command == "wait":
        output = (await client.wait_for(args.rollout_id, timeout_seconds=args.timeout_seconds)).payload
    elif args.command == "results":
        output = await client.get_rollout(args.rollout_id)
    elif args.command == "usage":
        output = await client.usage(args.rollout_id)
    else:
        output = await client.cancel(args.rollout_id)
    _emit(output)
    # A zero reward is a scientific result, not an execution failure.
    return 3 if args.command in {"exec", "wait"} and output.get("status") in {
        "failed", "cancelled"
    } else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        target = urlsplit(args.backend_url or "")
        if (
            target.scheme not in {"http", "https"} or not target.hostname
            or target.username is not None or target.password is not None
            or target.query or target.fragment or target.path not in {"", "/"}
        ):
            raise ValueError("rhodes.explicit_backend_origin_required")
        if args.command == "follow" and args.after_sequence < 0:
            raise ValueError("rhodes.invalid_event_cursor")
        body = _payload(args.config) if args.command in {"exec", "import"} else None
        if args.command == "watch" and args.cursors is not None:
            body = _payload(args.cursors)
        if args.command == "exec" and (
            body is None or not isinstance(body.get("idempotency_key"), str)
            or not body["idempotency_key"].strip()
        ):
            raise ValueError("rhodes.stable_idempotency_key_required")

        async def run() -> int:
            client = RhodesClient(
                api_key=os.environ.get("SYNTH_API_KEY", ""), backend_url=args.backend_url
            )
            async with client:
                return await _execute(client, args, body)

        return asyncio.run(run())
    except RhodesRequestError as error:
        # Headers/bodies/URLs may contain secrets or terminal control sequences.
        print(json.dumps({"error": error.code, "http_status": error.status_code}), file=sys.stderr)
        return 2
    except PoolRolloutTimeout:
        print('{"error":"rhodes.observation_timeout","cancelled":false}', file=sys.stderr)
        return 4
    except (ValueError, OSError, PoolClientError, httpx.HTTPError):
        print('{"error":"rhodes.command_failed"}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('{"error":"rhodes.observer_interrupted","cancelled":false}', file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
