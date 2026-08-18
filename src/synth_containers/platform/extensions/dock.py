"""Private Dock package adapter for the public Containers eval lifecycle.

Dock is task content and launch configuration, not another public event fold.
This adapter validates an opaque extension manifest and constructs the existing
Harbor Docker trial runtime with a pinned bundle.  The resulting HTTP surface
is still prepare -> subscribe -> start, ``synth.trace-stream-event.v1``, and
``/reward``; no Dock event schema or route is introduced.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DOCK_EXTENSION_SCHEMA = "synth.dock-eval-extension.v1"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class DockExtensionError(RuntimeError):
    """Secret-free extension validation refusal."""


@dataclass(frozen=True)
class DockEvalExtension:
    extension_id: str
    manifest_path: Path
    bundle_root: Path
    bundle_digest: str
    agent_credentials: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "DockEvalExtension":
        manifest_path = Path(path).expanduser().resolve()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DockExtensionError("dock_extension_manifest_invalid") from exc
        if not isinstance(payload, Mapping) or payload.get("schema") != DOCK_EXTENSION_SCHEMA:
            raise DockExtensionError("dock_extension_manifest_invalid")
        if payload.get("adapter") != "harbor_pinned_bundle":
            raise DockExtensionError("dock_extension_adapter_unsupported")
        extension_id = str(payload.get("extension_id") or "").strip()
        if not extension_id or len(extension_id) > 256:
            raise DockExtensionError("dock_extension_id_invalid")
        bundle = payload.get("bundle")
        if not isinstance(bundle, Mapping):
            raise DockExtensionError("dock_extension_bundle_invalid")
        relative = str(bundle.get("path") or "").strip()
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise DockExtensionError("dock_extension_bundle_path_invalid")
        bundle_root = (manifest_path.parent / relative).resolve()
        try:
            bundle_root.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise DockExtensionError("dock_extension_bundle_path_invalid") from exc
        digest = str(bundle.get("digest") or "").strip().lower()
        if not _DIGEST.fullmatch(digest):
            raise DockExtensionError("dock_extension_bundle_pin_required")
        # Credentials for an authoring agent are launcher-owned, not bundle
        # content, so a pinned bundle stays portable and secret-free.
        credentials_raw = payload.get("agent_credentials")
        agent_credentials: str | None = None
        if credentials_raw is not None:
            credentials_path = Path(str(credentials_raw)).expanduser()
            if not credentials_path.is_absolute() or not credentials_path.is_dir():
                raise DockExtensionError("dock_extension_agent_credentials_invalid")
            agent_credentials = str(credentials_path.resolve())
        return cls(
            extension_id=extension_id,
            manifest_path=manifest_path,
            bundle_root=bundle_root,
            bundle_digest=digest,
            agent_credentials=agent_credentials,
        )

    def runtime_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "harbor_pinned_bundle": {
                "root": str(self.bundle_root),
                "digest": self.bundle_digest,
            }
        }
        if self.agent_credentials:
            config["agent_credentials"] = self.agent_credentials
        return config


def create_dock_eval_app(
    extension: DockEvalExtension | str | Path,
    *,
    storage_root: str | Path | None = None,
):
    """Create the ordinary Containers app from a private Dock extension."""
    if not isinstance(extension, DockEvalExtension):
        extension = DockEvalExtension.from_file(extension)
    from ..app import create_compat_app

    app = create_compat_app(
        "harbor_docker",
        storage_root=storage_root,
        runtime_config=extension.runtime_config(),
    )
    app.state.dock_extension_id = extension.extension_id
    return app


__all__ = [
    "DOCK_EXTENSION_SCHEMA",
    "DockEvalExtension",
    "DockExtensionError",
    "create_dock_eval_app",
]
