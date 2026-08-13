"""Detached, request-scoped Trace V5 capture control plane.

The control plane owns no trace schema.  Every request-scoped capture is an ordinary
``CaptureSupervisor`` with its own raw spool and self-contained bundle; this module
only multiplexes lifecycle and application-emission HTTP calls onto those existing
authorities.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import ipaddress
import os
from pathlib import Path
import secrets
import shutil
import threading
from typing import Any, Mapping
from urllib.parse import urlparse
import uuid

from ..canonical import bytes_digest, canonical_bytes, utc_now
from ..models.completeness import TerminationV5, TraceStatus
from ..models.identity import TraceProvenanceV5
from .binding import CaptureMode, Interception
from .spool import repair
from .supervisor import CaptureSupervisor, SupervisorConfig


CONTROL_STATE_SCHEMA_VERSION = "synth.trace-detached-capture-state.v1"


class CaptureControlError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class DetachedCaptureConfig:
    output_root: Path
    host: str = "127.0.0.1"
    port: int = 0
    upstream_base_url: str = "https://api.openai.com/v1"
    capture_disk_budget_bytes: int | None = None
    capture_disk_reserve_bytes: int = 0
    budget_policy: str = "refuse"
    control_token: str | None = None
    max_request_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        try:
            loopback = ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            loopback = self.host == "localhost"
        if not loopback and not self.control_token:
            raise ValueError("a control token is required for non-loopback hosts")
        if self.capture_disk_budget_bytes is not None and self.capture_disk_budget_bytes <= 0:
            raise ValueError("capture_disk_budget_bytes must be positive")
        if self.capture_disk_reserve_bytes < 0:
            raise ValueError("capture_disk_reserve_bytes must be non-negative")
        if (
            self.capture_disk_budget_bytes is not None
            and self.capture_disk_reserve_bytes >= self.capture_disk_budget_bytes
        ):
            raise ValueError("capture disk reserve must be smaller than the budget")
        if self.budget_policy not in {"refuse", "evict_oldest_sealed"}:
            raise ValueError("budget_policy must be refuse or evict_oldest_sealed")


class DetachedCaptureSupervisor:
    """Run capture services for a long-lived workload without spawning it."""

    def __init__(self, config: DetachedCaptureConfig) -> None:
        self.config = config
        self.root = Path(config.output_root)
        self.bundles_root = self.root / "captures"
        self.state_root = self.root / "control"
        self.bundles_root.mkdir(parents=True, exist_ok=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._captures: dict[str, CaptureSupervisor] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._load_and_recover()
        self._server = ThreadingHTTPServer(
            (config.host, config.port), _control_handler(self)
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="synth-trace-detached-control",
            daemon=True,
        )
        self._started = False
        self._stopped = False

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> "DetachedCaptureSupervisor":
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._started:
            self._server.shutdown()
        self._server.server_close()
        with self._lock:
            for capture_id, supervisor in tuple(self._captures.items()):
                try:
                    supervisor.finalize(
                        status=TraceStatus.INTERRUPTED,
                        termination=TerminationV5(reason="detached_supervisor_stopped"),
                    )
                    self._record_sealed(capture_id, supervisor, TraceStatus.INTERRUPTED)
                except Exception as exc:  # state records the failed recovery explicitly
                    state = self._states[capture_id]
                    state.update(status="interrupted", error=f"{type(exc).__name__}: {exc}")
                    self._write_state(state)
            self._captures.clear()

    def open_capture(self, body: Mapping[str, Any]) -> dict[str, Any]:
        rollout_id = str(body.get("rollout_id") or "").strip()
        if not rollout_id:
            raise CaptureControlError(400, "invalid_rollout_id", "rollout_id is required")
        labels = body.get("labels") or {}
        if not isinstance(labels, Mapping):
            raise CaptureControlError(400, "invalid_labels", "labels must be an object")
        requested_mode = str(body.get("capture_mode") or CaptureMode.BEST_EFFORT)
        mode_aliases = {"off": str(CaptureMode.DISABLED), "required": str(CaptureMode.REQUIRED)}
        mode = mode_aliases.get(requested_mode, requested_mode)
        if mode == CaptureMode.DISABLED:
            return {"capture": "off", "rollout_id": rollout_id}
        if mode not in {str(CaptureMode.BEST_EFFORT), str(CaptureMode.REQUIRED)}:
            raise CaptureControlError(400, "invalid_capture_mode", "capture_mode must be off, best_effort, or required")

        with self._lock:
            self._ensure_capacity(16 * 1024, opening=True)
            bundle_path = self.bundles_root / uuid.uuid4().hex
            supervisor = CaptureSupervisor(
                SupervisorConfig(
                    bundle_root=bundle_path,
                    trace_key={"rollout_id": rollout_id, "labels": dict(labels)},
                    upstream_base_url=self.config.upstream_base_url,
                    provenance=TraceProvenanceV5(
                        producer="synth-trace-detached",
                        producer_version="1",
                        harness="detached HTTP capture",
                    ),
                    root_actor_name="rollout",
                    rollout_id=rollout_id,
                    mode=mode,
                    interception=Interception.APPLICATION,
                    detached=True,
                )
            )
            try:
                supervisor.start_capture()
            except Exception:
                shutil.rmtree(bundle_path, ignore_errors=True)
                raise
            context = supervisor.binding.context_for_child()
            state = {
                "schema_version": CONTROL_STATE_SCHEMA_VERSION,
                "capture_id": context.capture_id,
                "trace_id": context.trace_id,
                "actor_id": context.actor_id,
                "session_id": context.actor_session_id,
                "rollout_id": rollout_id,
                "labels": dict(labels),
                "capture_mode": mode,
                "status": "open",
                "opened_at": utc_now(),
                "bundle_path": str(bundle_path),
                "event_count": 0,
                "artifact_count": 0,
                # Finalization materializes the normalized document in addition to
                # the raw authority.  Keep that future write charged while open.
                "reserved_bytes": 16 * 1024,
            }
            self._captures[context.capture_id] = supervisor
            self._states[context.capture_id] = state
            self._write_state(state)
            return self._public_state(state)

    def emit_event(self, capture_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        event_type = str(body.get("event_type") or "").strip()
        payload = body.get("payload")
        if not event_type or not isinstance(payload, Mapping):
            raise CaptureControlError(400, "invalid_event", "event_type and object payload are required")
        with self._lock:
            supervisor, state = self._open(capture_id)
            encoded_size = len(canonical_bytes(body))
            self._ensure_live_capacity(capture_id, encoded_size * 2 + 8192)
            envelope_id = supervisor.collector.event(
                event_type=event_type,
                payload=dict(payload),
                actor_id=state["actor_id"],
                session_id=state["session_id"],
                occurred_at=(str(body["occurred_at"]) if body.get("occurred_at") else None),
                caused_by=tuple(str(item) for item in (body.get("caused_by") or ())),
                structural=(dict(body["structural"]) if isinstance(body.get("structural"), Mapping) else None),
            )
            state["event_count"] = int(state.get("event_count") or 0) + 1
            state["reserved_bytes"] = int(state.get("reserved_bytes") or 0) + encoded_size + 4096
            state["updated_at"] = utc_now()
            self._write_state(state)
            return {"envelope_id": envelope_id}

    def emit_artifact(self, capture_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        encoded = body.get("content_base64")
        if not isinstance(encoded, str):
            raise CaptureControlError(400, "invalid_artifact", "content_base64 is required")
        try:
            content = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise CaptureControlError(400, "invalid_artifact", "content_base64 is invalid") from exc
        with self._lock:
            supervisor, state = self._open(capture_id)
            self._ensure_live_capacity(capture_id, len(content) + 12 * 1024)
            artifact_id = supervisor.collector.artifact(
                role=str(body.get("role") or "observation"),
                media_type=str(body.get("media_type") or "application/octet-stream"),
                content=content,
                logical_name=str(body.get("logical_name") or "artifact"),
                visibility=str(body.get("visibility") or "private"),
                actor_id=state["actor_id"],
                session_id=state["session_id"],
            )
            state["artifact_count"] = int(state.get("artifact_count") or 0) + 1
            state["reserved_bytes"] = int(state.get("reserved_bytes") or 0) + 4096
            state["updated_at"] = utc_now()
            self._write_state(state)
            return {"artifact_id": artifact_id, "digest": bytes_digest(content)}

    def seal_capture(self, capture_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        status = str(body.get("status") or TraceStatus.COMPLETED)
        if status not in {str(TraceStatus.COMPLETED), str(TraceStatus.FAILED), str(TraceStatus.INTERRUPTED)}:
            raise CaptureControlError(400, "invalid_status", "status must be completed, failed, or interrupted")
        termination_body = body.get("termination")
        termination = None
        if isinstance(termination_body, Mapping):
            termination = TerminationV5(
                reason=str(termination_body.get("reason") or status),
                detail=(str(termination_body["detail"]) if termination_body.get("detail") is not None else ""),
                exit_code=(int(termination_body["exit_code"]) if termination_body.get("exit_code") is not None else None),
                signal=(str(termination_body["signal"]) if termination_body.get("signal") is not None else None),
            )
        with self._lock:
            supervisor, _ = self._open(capture_id)
            supervisor.finalize(status=status, termination=termination)
            self._record_sealed(capture_id, supervisor, status)
            self._captures.pop(capture_id, None)
            return self._public_state(self._states[capture_id])

    def abandon_capture(self, capture_id: str) -> dict[str, Any]:
        with self._lock:
            supervisor, state = self._open(capture_id)
            supervisor.abandon()
            self._captures.pop(capture_id, None)
            bundle_path = Path(state["bundle_path"])
            shutil.rmtree(bundle_path, ignore_errors=True)
            state.update(status="abandoned", abandoned_at=utc_now(), bytes=0)
            self._write_state(state)
            return self._public_state(state)

    def status(self, capture_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            usage = self.disk_usage()
            common = {
                "disk_used_bytes": usage,
                "budget_bytes": self.config.capture_disk_budget_bytes,
                "reserve_bytes": self.config.capture_disk_reserve_bytes,
                "budget_policy": self.config.budget_policy,
            }
            if capture_id is None:
                return {**common, "captures": [self._public_state(item) for item in self._states.values()]}
            state = self._states.get(capture_id)
            if state is None:
                raise CaptureControlError(404, "capture_not_found", "capture does not exist")
            return {**self._public_state(state), **common}

    def health(self) -> dict[str, Any]:
        result = self.status()
        result["ok"] = True
        result["open_capture_count"] = len(self._captures)
        return result

    def disk_usage(self) -> int:
        total = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                pass
        return total

    def _open(self, capture_id: str) -> tuple[CaptureSupervisor, dict[str, Any]]:
        state = self._states.get(capture_id)
        supervisor = self._captures.get(capture_id)
        if state is None:
            raise CaptureControlError(404, "capture_not_found", "capture does not exist")
        if supervisor is None or state.get("status") != "open":
            raise CaptureControlError(409, "capture_not_open", f"capture is {state.get('status')}")
        return supervisor, state

    def _record_sealed(self, capture_id: str, supervisor: CaptureSupervisor, status: str) -> None:
        sealed = supervisor.sealed
        if sealed is None:
            raise RuntimeError("capture finalization did not produce a sealed trace")
        state = self._states[capture_id]
        state.update(
            status=str(status),
            sealed_at=utc_now(),
            trace_v5_digest=sealed.document.content_digest,
            event_count=len(sealed.document.events),
            bytes=_tree_size(Path(state["bundle_path"])),
            reserved_bytes=0,
        )
        self._write_state(state)

    def _limit(self) -> int | None:
        budget = self.config.capture_disk_budget_bytes
        return None if budget is None else budget - self.config.capture_disk_reserve_bytes

    def _ensure_live_capacity(self, capture_id: str, additional: int) -> None:
        try:
            self._ensure_capacity(additional, opening=False)
        except CaptureControlError:
            supervisor = self._captures.get(capture_id)
            if supervisor is not None:
                supervisor.finalize(
                    status=TraceStatus.INTERRUPTED,
                    termination=TerminationV5(reason="capture_disk_budget_exceeded"),
                )
                self._record_sealed(capture_id, supervisor, TraceStatus.INTERRUPTED)
                self._captures.pop(capture_id, None)
            raise

    def _ensure_capacity(self, additional: int, *, opening: bool) -> None:
        limit = self._limit()
        if limit is None or self._budget_usage() + additional <= limit:
            return
        if self.config.budget_policy == "evict_oldest_sealed":
            candidates = sorted(
                (
                    state for state in self._states.values()
                    if state.get("status") in {"completed", "failed", "interrupted"}
                    and Path(str(state.get("bundle_path") or "")).is_dir()
                ),
                key=lambda item: str(item.get("sealed_at") or ""),
            )
            for state in candidates:
                shutil.rmtree(Path(state["bundle_path"]), ignore_errors=True)
                state.update(status="evicted", evicted_at=utc_now(), bytes=0)
                self._write_state(state)
                if self._budget_usage() + additional <= limit:
                    return
        action = "open a capture" if opening else "append to a live capture"
        raise CaptureControlError(507, "capture_disk_budget_exceeded", f"insufficient disk budget to {action}")

    def _budget_usage(self) -> int:
        return self.disk_usage() + sum(
            int(state.get("reserved_bytes") or 0)
            for state in self._states.values()
            if state.get("status") == "open"
        )

    def _state_path(self, capture_id: str) -> Path:
        return self.state_root / f"{capture_id}.json"

    def _write_state(self, state: Mapping[str, Any]) -> None:
        path = self._state_path(str(state["capture_id"]))
        temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        payload = json.dumps(dict(state), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        with temp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def _load_and_recover(self) -> None:
        for path in sorted(self.state_root.glob("*.json")):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            capture_id = str(state.get("capture_id") or "")
            if capture_id:
                self._states[capture_id] = state
        for capture_id, state in tuple(self._states.items()):
            if state.get("status") != "open":
                continue
            try:
                bundle_path = Path(state["bundle_path"])
                trace_id = str(state["trace_id"])
                repair(bundle_path / "traces" / trace_id, capture_id=capture_id)
                supervisor = CaptureSupervisor(
                    SupervisorConfig(
                        bundle_root=bundle_path,
                        trace_key={"rollout_id": state.get("rollout_id")},
                        upstream_base_url=self.config.upstream_base_url,
                        provenance=TraceProvenanceV5(
                            producer="synth-trace-detached",
                            producer_version="1",
                            harness="detached HTTP capture recovery",
                        ),
                        root_actor_name="rollout",
                        rollout_id=str(state.get("rollout_id") or "") or None,
                        mode=state.get("capture_mode") or CaptureMode.BEST_EFFORT,
                        interception=Interception.APPLICATION,
                        trace_id=trace_id,
                        capture_id=capture_id,
                        resume=True,
                        detached=True,
                    )
                )
                supervisor.start_capture()
                supervisor.finalize(
                    status=TraceStatus.INTERRUPTED,
                    termination=TerminationV5(reason="detached_supervisor_restart"),
                )
                self._captures[capture_id] = supervisor
                self._record_sealed(capture_id, supervisor, TraceStatus.INTERRUPTED)
                self._captures.pop(capture_id, None)
            except Exception as exc:
                state.update(
                    status="interrupted",
                    interrupted_at=utc_now(),
                    error=f"recovery failed: {type(exc).__name__}: {exc}",
                )
                self._write_state(state)

    @staticmethod
    def _public_state(state: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in state.items() if key != "schema_version"}


def _tree_size(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _control_handler(control: DetachedCaptureSupervisor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "synth-trace-detached/1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_DELETE(self) -> None:
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            try:
                self._authorize()
                path = urlparse(self.path).path.rstrip("/") or "/"
                parts = [item for item in path.split("/") if item]
                if method == "GET" and path == "/healthz":
                    self._json(200, control.health())
                    return
                if method == "GET" and parts == ["captures"]:
                    self._json(200, control.status())
                    return
                if method == "POST" and parts == ["captures"]:
                    self._json(200, control.open_capture(self._body()))
                    return
                if len(parts) >= 2 and parts[0] == "captures":
                    capture_id = parts[1]
                    if method == "GET" and len(parts) == 2:
                        self._json(200, control.status(capture_id))
                        return
                    if method == "DELETE" and len(parts) == 2:
                        self._json(200, control.abandon_capture(capture_id))
                        return
                    if method == "POST" and parts[2:] == ["events"]:
                        self._json(200, control.emit_event(capture_id, self._body()))
                        return
                    if method == "POST" and parts[2:] == ["artifacts"]:
                        self._json(200, control.emit_artifact(capture_id, self._body()))
                        return
                    if method == "POST" and parts[2:] == ["seal"]:
                        self._json(200, control.seal_capture(capture_id, self._body()))
                        return
                raise CaptureControlError(404, "not_found", "route does not exist")
            except CaptureControlError as exc:
                self._json(exc.status, {"error": {"code": exc.code, "message": str(exc)}})
            except Exception as exc:
                self._json(500, {"error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}})

        def _authorize(self) -> None:
            expected = control.config.control_token
            if expected is None:
                return
            supplied = self.headers.get("authorization") or ""
            if not supplied.startswith("Bearer ") or not secrets.compare_digest(supplied[7:], expected):
                raise CaptureControlError(401, "unauthorized", "valid bearer token required")

        def _body(self) -> dict[str, Any]:
            raw_length = self.headers.get("content-length") or "0"
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise CaptureControlError(400, "invalid_content_length", "invalid content length") from exc
            if length < 0 or length > control.config.max_request_bytes:
                raise CaptureControlError(413, "request_too_large", "request exceeds max_request_bytes")
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw or b"{}")
            except ValueError as exc:
                raise CaptureControlError(400, "invalid_json", "request body must be JSON") from exc
            if not isinstance(body, dict):
                raise CaptureControlError(400, "invalid_json", "request body must be an object")
            return body

        def _json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode() + b"\n"
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


__all__ = [
    "CONTROL_STATE_SCHEMA_VERSION",
    "CaptureControlError",
    "DetachedCaptureConfig",
    "DetachedCaptureSupervisor",
]
