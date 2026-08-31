"""Craftax image PID 1: the rust engine as a child, then the containers façade.

The binary is baked (``SYNTH_CRAFTAX_GOLD_BIN``). Outside the image — a dev
checkout with no baked binary — it falls back to building the GameBench crate,
which is a convenience for local work, never the operator path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from synth_containers.image_main import image_main
from synth_containers.pid1 import ChildProcess

from .targets import TARGETS
from .world import URL_ENV

IMAGE_ID = "craftax-gamebench-rust"
DEFAULT_TARGET = "craftax_react"

_CRATE = Path("craftax-runtime/gold_rust")
_BIN = "craftax_gold"


def _github_root() -> Path:
    # .../containers/images/craftax-gamebench-rust/craftax_gold/stack.py
    return Path(__file__).resolve().parents[3].parent


def resolve_binary() -> Path:
    """The baked engine, or a dev-checkout cargo build. Never a host service."""

    pinned = os.environ.get("SYNTH_CRAFTAX_GOLD_BIN", "").strip()
    if pinned:
        path = Path(pinned).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise RuntimeError("craftax_gold_bin_missing")
    override = os.environ.get("SYNTH_CRAFTAX_GOLD_ROOT", "").strip()
    if override:
        crate = Path(override).expanduser().resolve()
        if (crate / "gold_rust" / "Cargo.toml").is_file():
            crate = crate / "gold_rust"
        elif not (crate / "Cargo.toml").is_file():
            raise RuntimeError(f"craftax_gold_crate_missing:{crate}")
    else:
        crate = _github_root() / _CRATE
        if not (crate / "Cargo.toml").is_file():
            raise RuntimeError(f"craftax_gold_crate_missing:{crate}")
    built = crate / "target" / "release" / _BIN
    if built.is_file() and os.access(built, os.X_OK):
        return built
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("craftax_gold_cargo_missing")
    completed = subprocess.run(  # noqa: S603
        [
            cargo,
            "build",
            "--release",
            "--quiet",
            "--manifest-path",
            str(crate / "Cargo.toml"),
            "--bin",
            _BIN,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not built.is_file():
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        detail = tail[-1][:200] if tail else f"exit {completed.returncode}"
        raise RuntimeError(f"craftax_gold_build_failed:{detail}")
    return built


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
