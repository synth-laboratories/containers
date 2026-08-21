"""Compose entrypoint: serve create_compat_app on the container network.

Bind 0.0.0.0 inside the netns. Compose publishes 127.0.0.1 on the host.
"""

from __future__ import annotations

import os

import uvicorn

from synth_containers.platform import create_compat_app


def main() -> None:
    target = os.environ.get("SYNTH_CONTAINER_TARGET", "").strip()
    if not target:
        raise SystemExit("SYNTH_CONTAINER_TARGET is required")
    host = os.environ.get("SYNTH_CONTAINER_BIND", "0.0.0.0")
    port = int(os.environ.get("SYNTH_CONTAINER_PORT", "8080"))
    uvicorn.run(create_compat_app(target), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
