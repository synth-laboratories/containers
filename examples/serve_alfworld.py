#!/usr/bin/env python3
"""Serve the ALFWorld SFT/CISPO contract on loopback only.

The caller must mount or point ``ALFWORLD_DATA_ROOT`` at a downloaded official
ALFWorld artifact corpus. The service deliberately never downloads data or
loads provider credentials itself.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("serve_alfworld binds loopback only")
    data_root = Path(args.data_root).resolve()
    if not any(data_root.rglob("game.tw-pddl")):
        raise SystemExit("ALFWorld data root has no official game.tw-pddl artifacts")
    os.environ["ALFWORLD_DATA_ROOT"] = str(data_root)
    os.environ["ALFWORLD_RUN_ROOT"] = str(Path(args.run_root).resolve())
    from synth_containers.alfworld.app import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
