"""Foreground-owned Docker allocation. Parent pipe EOF stops the exact allocation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import httpx
from synth_containers.tracing.atomic import durable_write
from synth_containers.package_models import content_bytes


def engine_port(request: dict) -> int:
    port = request.get("port", 8098)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("engine port outside 1..65535")
    return port


CREATE_TIMEOUT_SECONDS = 180


def docker_run_argv(request: dict, name: str) -> list[str]:
    """Owned allocation shape; isolation limits are fixed, only command/port vary."""
    port = engine_port(request)
    command = request.get("command")
    if command is not None and (not command or any(type(a) is not str or not a for a in command)):
        raise ValueError("engine command must be non-empty strings")
    return ["docker", "run", "--pull", "never", "-d", "--name", name,
        "--label", "ai.synth.eval.owner=" + request["owner"],
        "--read-only", "--tmpfs", "/tmp:rw,size=67108864", "--memory", str(request["memory_mb"]) + "m",
        "--cpus", str(request["cpu"]), "--pids-limit", "128",
        "-p", f"127.0.0.1::{port}", "-e", f"PORT={port}", request["image"], *(command or ())]


def main():
    request = json.loads(sys.stdin.readline())
    name = "synth-eval-" + uuid4().hex[:16]
    record = Path(request["receipt"])
    record.parent.mkdir(parents=True, exist_ok=True)
    identifier = None
    started = time.monotonic()
    try:
        # A loaded daemon can take tens of seconds to create a container; the
        # wall is generous and the measured duration is retained in the receipt.
        result = subprocess.run(docker_run_argv(request, name),
            text=True, capture_output=True, check=True, timeout=CREATE_TIMEOUT_SECONDS)
        identifier = result.stdout.strip()
        port = subprocess.run(["docker", "port", identifier, f"{engine_port(request)}/tcp"], text=True,
            capture_output=True, check=True, timeout=5).stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        url = "http://127.0.0.1:" + port
        with httpx.Client(timeout=1, trust_env=False) as client:
            for _ in range(100):
                try:
                    if client.get(url + "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError("native engine readiness timeout")
        print(json.dumps({"url": url, "container": identifier, "name": name}), flush=True)
        # The controlling invocation alone holds this pipe's write end. It is
        # closed by normal finally cleanup, Ctrl-C and abrupt process death.
        sys.stdin.read()
    finally:
        result = subprocess.run(["docker", "rm", "-f", identifier or name], capture_output=True, timeout=CREATE_TIMEOUT_SECONDS)
        durable_write(record, content_bytes({"container": identifier, "name": name,
            "cleanup": "stopped" if result.returncode == 0 else "unconfirmed", "returncode": result.returncode,
            "create_seconds": round(time.monotonic() - started, 3)}))


if __name__ == "__main__":
    main()
