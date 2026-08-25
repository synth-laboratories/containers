#!/usr/bin/env python3
"""Loopback Craftax gold-react Containers façade. Bind 127.0.0.1 only.

Needs rust gold at SYNTH_CRAFTAX_URL (default http://127.0.0.1:8098) and an
OpenRouter key for the live planner. Engine/scripted CI uses craftax_engine.
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from synth_containers.platform import create_compat_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--target", default="craftax_react")
    parser.add_argument(
        "--storage-root",
        required=True,
        help="durable root for leases, seals, receipts and manifests",
    )
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("serve_craftax_react binds loopback only")
    gold_url = os.environ.get("SYNTH_CRAFTAX_URL", "").strip()
    runtime_config = {"gold_base_url": gold_url} if gold_url else None
    app = create_compat_app(
        args.target,
        storage_root=args.storage_root,
        runtime_config=runtime_config,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
