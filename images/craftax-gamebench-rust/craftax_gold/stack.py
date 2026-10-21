"""Craftax image PID 1: the rust engine as a child, then the containers façade.

The qualified binary is supplied through ``SYNTH_CRAFTAX_GOLD_BIN``. Compilation
belongs to the explicit guarded mbx preparation duty, never runtime startup.
"""

from __future__ import annotations

import os
from pathlib import Path

from synth_containers.image_main import image_main
from synth_containers.pid1 import ChildProcess

from .targets import TARGETS
from .world import URL_ENV

IMAGE_ID = "craftax-gamebench-rust"
DEFAULT_TARGET = "craftax_react"


def resolve_binary() -> Path:
    """Runtime assembly must supply the qualified engine; never compile on boot."""
    pinned = os.environ.get("SYNTH_CRAFTAX_GOLD_BIN", "").strip()
    if not pinned:
        raise RuntimeError("craftax_gold_prebuilt_required")
    path = Path(pinned).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError("craftax_gold_bin_missing")
    return path


def engine_child() -> ChildProcess:
    return ChildProcess(
        name="craftax_gold",
        argv=[str(resolve_binary()), "--host", "{host}", "--port", "{port}"],
        url_env=URL_ENV,
        startup_timeout_seconds=60.0,
    )


def main(argv: list[str] | None = None) -> int:
    return image_main(
        image_id=IMAGE_ID,
        targets=TARGETS,
        default_target=DEFAULT_TARGET,
        children=[engine_child()],
        argv=argv,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    raise SystemExit(main())
