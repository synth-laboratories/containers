"""Container-side authority for installed protocols and per-rollout annotation streams.

Durable layout under ``<storage_root>/live_annotation/``::

    protocols/<revision_id>.json   installed revisions (code + identity), immutable
    protocols/current.json         the default revision id
    events/<sha256(rollout_id)>.jsonl   one annotation stream journal per rollout

Both halves recover on restart. A closed annotation stream is served from its
journal with the producer gone, exactly like a rollout stream.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..event_log import CONTROL_SUBSCRIBED, RolloutEventLog, poll_payload, validate_rollout_id
from .contract import (
    PROTOCOL_SCHEMA,
    STREAM_KINDS,
    STREAM_SCHEMA,
    ProtocolRevision,
    contains_secret,
    protocol_revision_id,
)
from .model import ModelCaller, ModelSettings, OpenAICompatibleCaller
from .process import IsolatedProtocolProcess, ProtocolProcessError
from .runner import LiveAnnotationRunner, RunnerLimits

ModelCallerFactory = Callable[[ModelSettings], ModelCaller]
PROTOCOL_STATE_SCHEMA = "synth.container-annotation-protocol.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def annotation_stream_id(stream_id: str) -> str:
    return f"{stream_id}:annotations"


def annotation_channel(rollout_id: str, *, bound_transport: str) -> dict[str, Any]:
    """The descriptor fragment a rollout advertises when a protocol is bound to it."""

    validate_rollout_id(rollout_id)
    return {
        "schema": STREAM_SCHEMA,
        "id": annotation_stream_id(f"stream:{rollout_id}"),
        "events": f"/rollouts/{rollout_id}/annotations/events",
        "stream": f"/rollouts/{rollout_id}/annotations/stream"
        if bound_transport in {"sse", "websocket"}
        else None,
        "cursor": {"kind": "sequence"},
        "kinds": list(STREAM_KINDS),
        "status": "provisional",
    }


class LiveAnnotationService:
    def __init__(
        self,
        storage_root: Path,
        *,
        model_caller_factory: ModelCallerFactory | None = None,
        limits: RunnerLimits | None = None,
        spawn: Callable[[bytes, dict[str, Any]], IsolatedProtocolProcess] | None = None,
    ) -> None:
        self.root = Path(storage_root) / "live_annotation"
        self.revisions: dict[str, ProtocolRevision] = {}
        self.current_revision_id: str | None = None
        self.logs: dict[str, RolloutEventLog] = {}
        self.runners: dict[str, LiveAnnotationRunner] = {}
        self.limits = limits or RunnerLimits()
        self._model_caller_factory = model_caller_factory or (lambda settings: OpenAICompatibleCaller(settings))
        self._spawn = spawn
        self._lock = threading.RLock()
        self._recover_protocols()

    # ---------------------------------------------------------------- protocols

    def _protocol_dir(self) -> Path:
        return self.root / "protocols"

    def _recover_protocols(self) -> None:
        directory = self._protocol_dir()
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("anprev_*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            code = str(row.get("code") or "").encode("utf-8")
            revision = ProtocolRevision(
                revision_id=str(row["revision_id"]),
                protocol_id=str(row.get("protocol_id") or ""),
                code=code,
                code_sha256=str(row.get("code_sha256") or hashlib.sha256(code).hexdigest()),
                configuration=dict(row.get("configuration") or {}),
                configuration_digest=str(row.get("configuration_digest") or ""),
                source_revision=row.get("source_revision"),
                installed_at=str(row.get("installed_at") or ""),
            )
            self.revisions[revision.revision_id] = revision
        current = directory / "current.json"
        if current.is_file():
            candidate = json.loads(current.read_text(encoding="utf-8")).get("protocol_revision_id")
            if isinstance(candidate, str) and candidate in self.revisions:
                self.current_revision_id = candidate

    def _persist_protocol(self, revision: ProtocolRevision) -> None:
        directory = self._protocol_dir()
        directory.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": PROTOCOL_STATE_SCHEMA,
            "revision_id": revision.revision_id,
            "protocol_id": revision.protocol_id,
            "code": revision.code.decode("utf-8"),
            "code_sha256": revision.code_sha256,
            "configuration": revision.configuration,
            "configuration_digest": revision.configuration_digest,
            "source_revision": revision.source_revision,
            "installed_at": revision.installed_at,
        }
        self._write_json(directory / f"{revision.revision_id}.json", row)
        self._write_json(directory / "current.json", {"protocol_revision_id": revision.revision_id})

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def put_protocol(self, body: dict[str, Any]) -> dict[str, Any]:
        """Install a protocol revision. Idempotent on identical inputs; refuses credentials."""

        code = body.get("code")
        raw = code if isinstance(code, (bytes, bytearray)) else str(code or "").encode("utf-8")
        if not raw.strip():
            return {
                "error": "protocol_source_required",
                "status_code": 422,
                "detail": "PUT /annotation-protocol requires non-empty source",
            }
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"error": "protocol_source_not_utf8", "status_code": 422}
        protocol_id = str(body.get("protocol_id") or "").strip()
        if not protocol_id:
            return {
                "error": "protocol_identity_required",
                "status_code": 422,
                "detail": "PUT /annotation-protocol requires protocol_id",
            }
        configuration = body.get("configuration") or {}
        if not isinstance(configuration, dict):
            return {"error": "protocol_configuration_must_be_object", "status_code": 422}
        if contains_secret(configuration):
            return {
                "error": "protocol_credential_forbidden",
                "status_code": 422,
                "detail": "PUT /annotation-protocol accepts identity and configuration, never credentials",
            }
        source_revision = body.get("source_revision")
        source_revision = str(source_revision) if source_revision is not None else None
        revision_id, code_sha256, configuration_digest = protocol_revision_id(
            code=bytes(raw),
            protocol_id=protocol_id,
            configuration=configuration,
            source_revision=source_revision,
        )
        with self._lock:
            existing = self.revisions.get(revision_id)
            if existing is not None:
                self.current_revision_id = revision_id
                self._write_json(self._protocol_dir() / "current.json", {"protocol_revision_id": revision_id})
                return {**self.protocol_state_payload(), "idempotent": True}
            # Prove the code boots in isolation before calling it installed. A
            # protocol that cannot start is refused, not stored.
            try:
                probe = self._spawn_process(bytes(raw), configuration)
            except ProtocolProcessError as exc:
                return {
                    "error": "protocol_boot_failed",
                    "status_code": 422,
                    "detail": str(exc)[-800:],
                }
            declared_id = probe.protocol_id
            receipt = dict(probe.isolation_receipt)
            probe.close()
            if declared_id and declared_id != protocol_id:
                return {
                    "error": "protocol_identity_mismatch",
                    "status_code": 422,
                    "requested_protocol_id": protocol_id,
                    "declared_protocol_id": declared_id,
                }
            revision = ProtocolRevision(
                revision_id=revision_id,
                protocol_id=protocol_id,
                code=bytes(raw),
                code_sha256=code_sha256,
                configuration=json.loads(json.dumps(configuration, sort_keys=True)),
                configuration_digest=configuration_digest,
                source_revision=source_revision,
                installed_at=_utc_now(),
            )
            self.revisions[revision_id] = revision
            self.current_revision_id = revision_id
            self._persist_protocol(revision)
            return {**self.protocol_state_payload(), "idempotent": False, "isolation_receipt": receipt}

    def protocol_state_payload(self) -> dict[str, Any]:
        with self._lock:
            revision = self.revisions.get(str(self.current_revision_id or ""))
            model = ModelSettings.from_configuration(revision.configuration) if revision else None
            return {
                "schema_version": PROTOCOL_STATE_SCHEMA,
                "protocol_schema": PROTOCOL_SCHEMA,
                "status": "installed" if revision is not None else "not_installed",
                **(revision.public() if revision is not None else {
                    "protocol_revision_id": None,
                    "protocol_id": None,
                    "code_sha256": None,
                    "configuration_digest": None,
                    "source_revision": None,
                    "installed_at": None,
                }),
                "model": model.public() if model is not None else None,
                "installed_revisions": sorted(self.revisions),
                "credential_state": "not_exposed",
            }

    def revision_for(self, revision_id: str | None) -> ProtocolRevision | None:
        if not revision_id:
            return None
        return self.revisions.get(revision_id)

    def advertisement(self) -> dict[str, Any]:
        return {
            "schema": STREAM_SCHEMA,
            "protocol_schema": PROTOCOL_SCHEMA,
            "protocol_url": "/annotation-protocol",
            "mode": "observe_only",
            "findings_status": "provisional",
            "kinds": list(STREAM_KINDS),
            "installed": self.current_revision_id is not None,
        }

    # ---------------------------------------------------------------- streams

    def _journal_path(self, rollout_id: str) -> Path:
        validate_rollout_id(rollout_id)
        key = hashlib.sha256(rollout_id.encode("utf-8")).hexdigest()
        return self.root / "events" / f"{key}.jsonl"

    def _spawn_process(self, code: bytes, configuration: dict[str, Any]) -> IsolatedProtocolProcess:
        if self._spawn is not None:
            return self._spawn(code, configuration)
        return IsolatedProtocolProcess(code, config=configuration)

    def open_stream(self, rollout_id: str, source_stream_id: str) -> RolloutEventLog:
        """Open (or recover) the annotation stream at prepare time.

        A viewer subscribes to the declared channel *before* the rollout
        starts, exactly as it does for the rollout stream, so the durable log
        and its ``stream.subscribed`` control record must exist as soon as the
        rollout is prepared with a protocol pin. Idempotent.
        """

        with self._lock:
            existing = self.logs.get(rollout_id)
            if existing is not None:
                return existing
            output = RolloutEventLog.recover(
                rollout_id=rollout_id,
                stream_id=annotation_stream_id(source_stream_id),
                journal_path=self._journal_path(rollout_id),
            )
            if not output.closed and not any(item.control for item in output.after(0)):
                output.append_control(CONTROL_SUBSCRIBED, output.subscribed_payload())
            self.logs[rollout_id] = output
            return output

    def attach(
        self,
        *,
        rollout_id: str,
        source: RolloutEventLog,
        revision_id: str,
    ) -> LiveAnnotationRunner:
        """Start tailing ``source`` into the rollout's annotation stream."""

        revision = self.revisions.get(revision_id)
        if revision is None:
            raise KeyError(f"annotation_protocol_unknown:{revision_id}")
        with self._lock:
            existing = self.runners.get(rollout_id)
            if existing is not None and not existing.finished:
                return existing
            output = self.open_stream(rollout_id, source.stream_id)
            if output.closed:
                raise RuntimeError(f"annotation_stream_sealed:{rollout_id}")
            settings = ModelSettings.from_configuration(revision.configuration)
            model = self._model_caller_factory(settings) if settings is not None else None
            limits = self.limits
            if settings is not None:
                limits = RunnerLimits(
                    max_model_calls=min(settings.max_calls, self.limits.max_model_calls),
                    max_model_output_tokens=min(settings.max_output_tokens, self.limits.max_model_output_tokens),
                    drain_timeout_seconds=self.limits.drain_timeout_seconds,
                    idle_wait_seconds=self.limits.idle_wait_seconds,
                    model_workers=self.limits.model_workers,
                )
            runner = LiveAnnotationRunner(
                rollout_id=rollout_id,
                source=source,
                output=output,
                revision=revision,
                model=model,
                limits=limits,
                spawn=self._spawn_process,
            )
            self.runners[rollout_id] = runner
            runner.start()
            return runner

    def detach(self, rollout_id: str) -> None:
        """The rollout runtime returned. If its log never closed, stop after draining."""

        runner = self.runners.get(rollout_id)
        if runner is None:
            return
        if not runner.source.closed:
            runner.request_stop()

    def log_for(self, rollout_id: str) -> RolloutEventLog | None:
        log = self.logs.get(rollout_id)
        if log is not None:
            return log
        try:
            journal = self._journal_path(rollout_id)
        except ValueError:
            return None
        if not journal.is_file():
            return None
        recovered = RolloutEventLog.recover(
            rollout_id=rollout_id,
            stream_id=annotation_stream_id(f"stream:{rollout_id}"),
            journal_path=journal,
        )
        self.logs[rollout_id] = recovered
        return recovered

    def events_payload(self, rollout_id: str, after: int, limit: int = 1000) -> dict[str, Any]:
        log = self.log_for(rollout_id)
        if log is None:
            return {"error": "annotation_stream_not_found", "status_code": 404, "rollout_id": rollout_id}
        try:
            payload = poll_payload(log, after=after, limit=limit, subject_id=rollout_id)
        except ValueError:
            return {"error": "invalid_page_limit", "status_code": 422}
        runner = self.runners.get(rollout_id)
        payload["schema"] = STREAM_SCHEMA
        payload["summary"] = runner.summary.public() if runner is not None else None
        # A viewer folds this stream with the rollout's own by rollout_id, and
        # de-duplicates by (stream_id, sequence). Both streams start at 1, so
        # every row must carry its own stream identity or the fold drops one.
        for row in payload["events"]:
            row["stream_id"] = log.stream_id
        return payload

    def is_bound(self, rollout_id: str) -> bool:
        return rollout_id in self.logs or self._journal_path(rollout_id).is_file()

    def wait_all(self, timeout: float) -> None:
        """Test helper: wait for every runner to seal its stream."""

        for runner in list(self.runners.values()):
            runner.join(timeout)
