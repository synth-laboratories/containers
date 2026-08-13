#!/usr/bin/env python3
"""Serve the normalized, paid HealthBench target."""

from __future__ import annotations

import argparse

import uvicorn

from synth_containers.platform import create_compat_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    args = parser.parse_args()
    uvicorn.run(create_compat_app("healthbench_chat"), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
