"""Serve a duck-typed Synth task container from a TargetSpec.

In Docker, pass ``--host 0.0.0.0 --allow-non-loopback``. Loopback is the
default for host processes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .platform import TARGETS, create_compat_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=os.environ.get("SYNTH_CONTAINER_TARGET", "openenv_echo"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SYNTH_CONTAINER_PORT", "8080")))
    parser.add_argument("--storage-root", default=os.environ.get("SYNTH_CONTAINER_STORAGE"))
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        default=os.environ.get("SYNTH_CONTAINER_BIND_ANY") == "1",
    )
    args = parser.parse_args(argv)
    if args.target not in TARGETS:
        raise SystemExit(f"unknown_target:{args.target}")
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_non_loopback:
        raise SystemExit("serve_refuses_non_loopback_without_allow_non_loopback")
    storage = Path(args.storage_root) if args.storage_root else None
    if storage is not None:
        storage.mkdir(parents=True, exist_ok=True)
    app = create_compat_app(args.target, storage_root=storage)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
