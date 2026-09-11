"""Read native provider resource custody without conflating it with scoring."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .operator_journal import read_operator_events


class HarborResourceCleanupPending(RuntimeError):
    """A native environment could not establish confirmed resource absence."""


def read_harbor_resource_receipt(trial_dir: Path) -> dict[str, Any]:
    """Return bounded, read-only cleanup evidence; missing data is not absence."""
    path = trial_dir / "resource-events.jsonl"
    page = read_operator_events(path, limit=1000)
    if page["has_more"]:
        raise ValueError("Native Harbor resource event count exceeds bound")
    events = page["events"]
    intents = [event for event in events if event.get("event") == "resource.create_requested"]
    if len(intents) != 1:
        raise ValueError("Native Harbor requires exactly one resource creation intent")
    intent = intents[0]
    owner = intent.get("run_id")
    provider = intent.get("provider")
    if not isinstance(owner, str) or not re.fullmatch(r"synth-harbor-[0-9a-f]{32}", owner):
        raise ValueError("Native Harbor resource owner is invalid")
    if provider not in {"docker", "daytona"}:
        raise ValueError("Native Harbor resource provider is invalid")
    last = events[-1]
    confirmed = last.get("event") == "resource.cleanup_confirmed"
    if confirmed:
        if provider == "docker":
            observations = [
                event for event in events if event.get("event") == "resource.handles_observed"
            ]
            handles = last.get("handles")
            if (
                not observations
                or not isinstance(handles, list)
                or not 1 <= len(handles) <= 48
                or handles != observations[-1].get("handles")
                or any(
                    not isinstance(handle, dict)
                    or set(handle) != {"kind", "id"}
                    or handle["kind"] not in {"container", "network", "volume"}
                    or not isinstance(handle["id"], str)
                    or not 1 <= len(handle["id"]) <= 255
                    for handle in handles
                )
                or not any(handle["kind"] == "container" for handle in handles)
            ):
                raise ValueError(
                    "Native Docker confirmed cleanup requires observed primary handles"
                )
        else:
            known = {event.get("provider_id") for event in events[:-1] if event.get("provider_id")}
            identifier = last.get("provider_id")
            if (
                not isinstance(identifier, str)
                or not 1 <= len(identifier) <= 255
                or known != {identifier}
            ):
                raise ValueError("Native Daytona confirmed cleanup requires one known sandbox")
    return {
        "schema_version": "synth.harbor-resource-receipt.v1",
        "provider": provider,
        "owner": owner,
        "cleanup_status": "confirmed" if confirmed else "pending",
        "last_event": last.get("event"),
        "last_sequence": last["seq"],
        "handles": last.get("handles", []),
        "provider_id": last.get("provider_id"),
        "journal_path": str(path),
    }
