"""Recover native Docker custody after its worker releases the lifetime lock."""

from __future__ import annotations

import fcntl
import json
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from docker.errors import NotFound

from .operator_journal import OperatorJournal, read_operator_events

LABEL = "com.docker.compose.project"


class DockerRecoveryError(ValueError):
    """Local custody cannot authorize or confirm recovery."""


def _handles(rows: list[dict]) -> list[dict[str, str]]:
    found = []
    for row in rows:
        handles = row.get("handles", [])
        if not isinstance(handles, list) or len(handles) > 48:
            raise DockerRecoveryError("Docker recovery handle inventory is invalid")
        for handle in handles:
            if (
                not isinstance(handle, dict)
                or handle.get("kind") not in {"container", "network", "volume"}
                or not isinstance(handle.get("id"), str)
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,254}", handle["id"])
            ):
                raise DockerRecoveryError("Docker recovery handle is invalid")
            value = {"kind": handle["kind"], "id": handle["id"]}
            if value not in found:
                found.append(value)
    if len(found) > 48:
        raise DockerRecoveryError("Docker recovery handle inventory exceeds bound")
    return found


def reconcile_docker_trial(trial_dir: Path, client: Any, *, now: datetime | None = None) -> dict:
    """Delete only exactly owned resources; keep ambiguous creation pending.

    The caller supplies a Docker SDK client with a finite five-second request
    timeout. Operations are synchronous: no cancellation can leave a background
    deletion thread running after this function reports its outcome. The
    90-second operation budget is checked before every provider request.
    """
    request_timeout = getattr(client.api, "timeout", None)
    if type(request_timeout) not in (int, float) or not 0 < request_timeout <= 5:
        raise DockerRecoveryError(
            "Docker recovery requires a request timeout of at most five seconds"
        )
    with (trial_dir / "resource-create-claim.json").open("rb") as handle:
        encoded = handle.read(1025)
    if len(encoded) > 1024:
        raise DockerRecoveryError("Docker recovery claim exceeds bound")
    claim = json.loads(encoded)
    if (
        not isinstance(claim, dict)
        or claim.get("provider") != "docker"
        or claim.get("recovery_protocol") != "owner_lock.v1"
        or not isinstance(claim.get("owner"), str)
        or not re.fullmatch(r"synth-harbor-[0-9a-f]{32}", claim["owner"])
        or not isinstance(claim.get("daemon_id"), str)
        or not 1 <= len(claim["daemon_id"]) <= 255
    ):
        raise DockerRecoveryError("Docker recovery requires qualified lifetime-lock custody")
    owner = claim["owner"]
    with (trial_dir / "resource-owner.lock").open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if client.info().get("ID") != claim["daemon_id"]:
                raise DockerRecoveryError("Docker recovery daemon identity mismatch")
            return _reconcile_locked(trial_dir, client, owner, now or datetime.now(UTC))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _reconcile_locked(trial_dir: Path, client: Any, owner: str, now: datetime) -> dict:
    page = read_operator_events(trial_dir / "resource-events.jsonl", run_id=owner, limit=1000)
    if page["has_more"]:
        raise DockerRecoveryError("Docker recovery history exceeds bound")
    rows = page["events"]
    intents = [row for row in rows if row.get("event") == "resource.create_requested"]
    if len(intents) != 1 or intents[0].get("provider") != "docker":
        raise DockerRecoveryError("Docker recovery requires one Docker creation intent")
    intent = intents[0]
    timeout = intent.get("creation_timeout_seconds")
    if type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise DockerRecoveryError("Docker recovery creation allowance is invalid")
    created = datetime.fromisoformat(intent["occurred_at"])
    if created.tzinfo is None or now.tzinfo is None:
        raise DockerRecoveryError("Docker recovery requires timezone-aware clocks")
    if now < created + timedelta(seconds=timeout + 60):
        raise DockerRecoveryError("Docker recovery creation allowance has not expired")
    known = _handles(rows)
    journal = OperatorJournal(
        trial_dir / "resource-events.jsonl", run_id=owner, max_record_bytes=16384
    )
    managers = {
        "container": client.containers,
        "network": client.networks,
        "volume": client.volumes,
    }
    deadline = time.monotonic() + 90

    def request(call, *args, **kwargs):
        if time.monotonic() >= deadline:
            raise DockerRecoveryError("Docker recovery operation deadline exceeded")
        return call(*args, **kwargs)

    def record(event: str, **fields):
        return journal.append(
            lambda seq: {
                "schema_version": "synth.harbor-resource-event.v1",
                "run_id": owner,
                "seq": seq,
                "event": event,
                "occurred_at": datetime.now(UTC).isoformat(),
                "provider": "docker",
                "project": owner,
                "source": "recovery",
                **fields,
            }
        )

    def check_owner(item, kind):
        labels = (
            item.attrs.get("Config", {}).get("Labels")
            if kind == "container"
            else item.attrs.get("Labels")
        )
        if (labels or {}).get(LABEL) != owner:
            raise DockerRecoveryError("Docker recovery resource ownership mismatch")

    def discover():
        found = []
        for kind, manager in managers.items():
            items = request(
                manager.list,
                filters={"label": LABEL + "=" + owner},
                **({"all": True} if kind == "container" else {}),
            )
            if len(items) > 16:
                raise DockerRecoveryError("Docker recovery discovery exceeds bound")
            for item in items:
                check_owner(item, kind)
                found.append({"kind": kind, "id": item.name if kind == "volume" else item.id})
        return _handles([{"handles": found}])

    record("resource.recovery_requested", handles=known)
    try:
        known = _handles([{"handles": known}, {"handles": discover()}])
        record("resource.handles_observed", handles=known)
        if not any(handle["kind"] == "container" for handle in known):
            raise DockerRecoveryError("Docker ambiguous creation has no observed primary handle")
        # Validate the entire saved/discovered inventory before any deletion.
        for handle in known:
            try:
                item = request(managers[handle["kind"]].get, handle["id"])
            except NotFound:
                continue
            check_owner(item, handle["kind"])
        record("resource.cleanup_requested", handles=known)
        for kind in ("container", "network", "volume"):
            for handle in (entry for entry in known if entry["kind"] == kind):
                try:
                    item = request(managers[kind].get, handle["id"])
                    check_owner(item, kind)
                    request(item.remove, **({"force": True} if kind == "container" else {}))
                except NotFound:
                    pass
        for handle in known:
            try:
                request(managers[handle["kind"]].get, handle["id"])
            except NotFound:
                continue
            raise DockerRecoveryError("Docker recovery absence is unconfirmed")
        if discover():
            raise DockerRecoveryError("Docker recovery owner still has resources")
        return record("resource.cleanup_confirmed", handles=known)
    except BaseException as error:
        record("resource.cleanup_pending", handles=known, error_type=type(error).__name__)
        raise
