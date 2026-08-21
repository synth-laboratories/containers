"""Loopback-published local target. Compose binds 0.0.0.0 inside the netns."""

from __future__ import annotations

import os

import uvicorn

from synth_containers.platform import create_compat_app

_LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def main() -> None:
    target = os.environ.get("SYNTH_CONTAINER_TARGET", "").strip()
    if not target:
        raise SystemExit("SYNTH_CONTAINER_TARGET is required")
    host = os.environ.get("SYNTH_CONTAINER_BIND", "0.0.0.0").strip() or "0.0.0.0"
    if host not in _LOOPBACK:
        raise SystemExit("local targets bind loopback or the compose netns only")
    if host == "0.0.0.0":
        # Compose netns: mlx-rl on the host advertises 127.0.0.1, which is this
        # container. Rewrite to the Docker/OrbStack host alias and admit HTTP.
        os.environ.setdefault("SYNTH_CONTAINERS_ALLOW_LOOPBACK_SAMPLER", "1")
        os.environ.setdefault("SYNTH_SAMPLER_HOST_REWRITE", "host.docker.internal")
    port = int(os.environ.get("SYNTH_CONTAINER_PORT", "8080"))
    uvicorn.run(create_compat_app(target), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
