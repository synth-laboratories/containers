"""The durable root is named by the caller; the façade never invents one (P2-4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from synth_containers.platform import create_compat_app
from synth_containers.platform.state import CompatPlatform
from synth_containers.platform.targets import TARGETS

_PLATFORM_DIR = Path(__file__).resolve().parents[1] / "src" / "synth_containers" / "platform"


def test_platform_and_app_factory_require_an_explicit_storage_root() -> None:
    with pytest.raises(TypeError):
        CompatPlatform(TARGETS["craftax_engine"])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        create_compat_app("craftax_engine")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="storage_root_required"):
        CompatPlatform(TARGETS["craftax_engine"], storage_root="")
    with pytest.raises(ValueError, match="storage_root_required"):
        create_compat_app("craftax_engine", storage_root=None)  # type: ignore[arg-type]


def test_no_temporary_root_fallback_in_the_platform_lane() -> None:
    for name in ("state.py", "app.py", "extensions/dock.py"):
        source = (_PLATFORM_DIR / name).read_text(encoding="utf-8")
        assert "mkdtemp" not in source, f"{name} fabricates a storage root"
        assert "TemporaryDirectory" not in source, f"{name} fabricates a storage root"
