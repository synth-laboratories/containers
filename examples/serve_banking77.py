#!/usr/bin/env python3
"""Loopback Banking77 Containers façade. Bind 127.0.0.1 only.

Give each Workshop instance its own --storage-root.
"""

from __future__ import annotations

import argparse

import uvicorn

from synth_containers.platform import create_compat_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--target", default="banking77_classify")
    parser.add_argument("--storage-root", default=None)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("serve_banking77 binds loopback only")
    app = create_compat_app(args.target, storage_root=args.storage_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
