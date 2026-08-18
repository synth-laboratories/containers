#!/usr/bin/env python3
"""Loopback HealthBench Containers façade. Bind 127.0.0.1 only.

Canonical grader is gpt-4.1-2025-04-14 via OPENAI_API_KEY. Policy is selected
per request (Groq default or openai_gpt41_mini). Give each Workshop instance
its own --storage-root.
"""

from __future__ import annotations

import argparse

import uvicorn

from synth_containers.platform import create_compat_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument("--storage-root", default=None)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("serve_healthbench binds loopback only")
    app = create_compat_app("healthbench_chat", storage_root=args.storage_root)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
