"""``python -m <image package>`` — the shared PID 1 entrypoint for platform images.

An image module gets its whole ``main`` from here:

    from synth_containers.image_main import engine_image_main
    from .targets import TARGETS
    from .world import ENGINE_CHILD

    def main(argv=None):
        return engine_image_main(
            image_id="craftax-gamebench-rust",
            targets=TARGETS,
            default_target="craftax_react",
            children=[ENGINE_CHILD],
            argv=argv,
        )

The target is selected by ``--target`` or ``SYNTH_CONTAINER_TARGET`` and must be
one of THIS image's targets — never the public demo ``TARGETS`` registry.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .pid1 import ChildProcess, run_stack

__all__ = ["engine_image_main", "image_main"]


def _parser(description: str, default_target: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--target", default=os.environ.get("SYNTH_CONTAINER_TARGET", default_target)
    )
    parser.add_argument("--host", default=os.environ.get("SYNTH_CONTAINER_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SYNTH_CONTAINER_PORT", "8080"))
    )
    parser.add_argument(
        "--storage-root",
        default=os.environ.get("SYNTH_CONTAINER_STORAGE", ""),
        help="durable local journal/CAS root (or SYNTH_CONTAINER_STORAGE)",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="start, print the URL, then take everything down (smoke check)",
    )
    return parser


def image_main(
    *,
    image_id: str,
    targets: Mapping[str, Any],
    default_target: str,
    children: Sequence[ChildProcess] = (),
    argv: list[str] | None = None,
    description: str = "",
    extend_app: Callable[[Any], None] | None = None,
) -> int:
    """Start baked children, serve this image's target, block until SIGTERM.

    ``extend_app`` runs against the built facade before it is served. An image
    declares contracts the shared platform has no opinion about -- a training
    contract, say -- beside the task they describe, without every image having
    to reimplement argument parsing and process supervision to do it.
    """

    args = _parser(description or f"{image_id} platform image", default_target).parse_args(argv)
    spec = targets.get(args.target)
    if spec is None:
        raise SystemExit(f"unknown_target:{args.target}:{','.join(sorted(targets))}")

    from .platform import create_compat_app

    def app_factory() -> Any:
        app = create_compat_app(spec, storage_root=args.storage_root or None)
        if extend_app is not None:
            extend_app(app)
        return app

    return run_stack(
        children=list(children),
        app_factory=app_factory,
        host=args.host,
        port=args.port,
        image_id=image_id,
        startup_timeout_seconds=args.startup_timeout_seconds,
        block=not args.no_block,
    )


# Kept as the explicit name for engine images; identical contract.
engine_image_main = image_main
