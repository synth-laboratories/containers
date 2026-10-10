"""Thin connector CLI over the backend facade and foreground service."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import threading
from uuid import UUID
from .profiles import ConnectorProfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-containers tunnel")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["serve", "status", "doctor", "close-lease"]:
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        if name == "close-lease":
            command.add_argument("lease", type=UUID)
    args = parser.parse_args(argv)
    profile = ConnectorProfile.load(args.profile)
    if args.command == "serve":
        from .service import ConnectorService

        stop = threading.Event()

        def halt(_number, _frame):
            stop.set()

        previous = {
            number: signal.signal(number, halt) for number in [signal.SIGINT, signal.SIGTERM]
        }
        service = None
        try:
            service = ConnectorService(profile)
            service.start()
            print(json.dumps(service.status()), flush=True)
            service.wait(stop)
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                for number, handler in previous.items():
                    signal.signal(number, handler)
        return 0
    credential = os.environ.get(profile.backend_credential_env)
    if not credential:
        raise ValueError("connector_backend_credential_unavailable")
    from synth_ai import SynthClient

    with SynthClient(api_key=credential, base_url=profile.backend_url) as client:
        if args.command == "close-lease":
            from synth_ai.sdk.tunnels import BorrowedLeaseHandle

            result = (
                BorrowedLeaseHandle(profile.connector, args.lease, client.tunnels)
                .close()
                .model_dump(mode="json")
            )
        else:
            result = client.tunnels.status(profile.connector).model_dump(mode="json")
            if args.command == "doctor":
                # Resolve configured references without printing values or
                # contacting arbitrary addresses named by a remote worker.
                registry = profile.registry()
                for target in profile.targets:
                    registry.bind(target.key, service_identity=target.service_identity)
                result = {
                    "profile_valid": True,
                    "origin_secret_bindings_resolved": True,
                    "catalog": result,
                }
        print(json.dumps(result, sort_keys=True))
    return 0
