"""Per-rollout runner: tail the rollout log, feed the protocol, publish annotations.

One runner per rollout, one protocol process per runner. The runner is the
only writer of the annotation stream and the only caller of the protocol
process, so emissions are ordered and every published record is attributable
to a specific source sequence.

Trust boundaries the runner enforces:

* observe-only -- it reads the rollout log through the same ``after``
  cursor a remote consumer uses and never touches the policy, world or pin;
* fail-soft -- a crashing protocol closes the annotation stream cleanly with
  ``annotation.protocol.error``; the rollout it was watching is unaffected;
* bounded -- model calls are counted against a ceiling and the post-terminal
  drain is time-boxed, so a slow judge cannot keep a stream open forever.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..event_log import RolloutEventLog
from .contract import (
    CONTROL_MESSAGE,
    CONTROL_PROTOCOL_UPDATE,
    CONTROL_STOP,
    KIND_BOUND,
    KIND_CAPTURE_CLOSED,
    KIND_CLOSED,
    KIND_CONTROL_RECEIVED,
    KIND_CONTROL_REFUSED,
    KIND_PROTOCOL_REBOUND,
    KIND_FINDING,
    KIND_HIGH_WATER,
    KIND_METRIC,
    KIND_MODEL_COMPLETED,
    KIND_MODEL_FAILED,
    KIND_MODEL_REQUESTED,
    KIND_PROTOCOL_ERROR,
    KIND_RETRACTED,
    OP_FINDING,
    OP_METRIC,
    OP_MODEL_REQUEST,
    OP_RETRACT,
    Control,
    ControlError,
    EmissionError,
    ProtocolRevision,
    normalize_control,
    normalize_emission,
    text_digest,
)
from .model import ModelCaller, ModelResult, ModelUnavailable
from .process import IsolatedProtocolProcess, ProtocolProcessError

ProcessFactory = Callable[..., IsolatedProtocolProcess]
RevisionResolver = Callable[[str], ProtocolRevision | None]
ModelResolver = Callable[[ProtocolRevision], ModelCaller | None]


@dataclass(frozen=True)
class RunnerLimits:
    max_model_calls: int = 20
    max_model_output_tokens: int = 800
    drain_timeout_seconds: float = 30.0
    idle_wait_seconds: float = 0.25
    model_workers: int = 2


@dataclass
class RunnerSummary:
    consumed_high_water: int = 0
    findings: int = 0
    retractions: int = 0
    metrics: int = 0
    model_requested: int = 0
    model_completed: int = 0
    model_failed: int = 0
    protocol_errors: int = 0
    controls_received: int = 0
    controls_refused: int = 0
    rebinds: int = 0
    outcome: str = "running"
    active_findings: dict[str, dict[str, Any]] = field(default_factory=dict)
    retracted: set[str] = field(default_factory=set)

    def public(self) -> dict[str, Any]:
        return {
            "consumed_high_water": self.consumed_high_water,
            "findings": self.findings,
            "retractions": self.retractions,
            "metrics": self.metrics,
            "model_requested": self.model_requested,
            "model_completed": self.model_completed,
            "model_failed": self.model_failed,
            "protocol_errors": self.protocol_errors,
            "controls_received": self.controls_received,
            "controls_refused": self.controls_refused,
            "rebinds": self.rebinds,
            "outcome": self.outcome,
            "active_findings": len(self.active_findings),
        }


class LiveAnnotationRunner:
    def __init__(
        self,
        *,
        rollout_id: str,
        source: RolloutEventLog,
        output: RolloutEventLog,
        revision: ProtocolRevision,
        model: ModelCaller | None = None,
        limits: RunnerLimits | None = None,
        spawn: ProcessFactory | None = None,
        resolve_revision: RevisionResolver | None = None,
        model_for: ModelResolver | None = None,
    ) -> None:
        self.rollout_id = rollout_id
        self.source = source
        self.output = output
        self.revision = revision
        self.model = model
        self.limits = limits or RunnerLimits()
        self._spawn = spawn or (
            lambda code, config, state=None: IsolatedProtocolProcess(code, config=config, state=state)
        )
        self._resolve_revision = resolve_revision or (lambda _revision_id: None)
        self._model_for = model_for
        self.summary = RunnerSummary()
        self._stop = threading.Event()
        self._consumer_stop = threading.Event()
        self._controls: "queue.Queue[tuple[str, Control | dict[str, Any], str | None]]" = queue.Queue()
        self._control_counter = 0
        self._control_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name=f"live-annotation:{rollout_id}", daemon=True
        )
        self._process: IsolatedProtocolProcess | None = None
        self._results: "queue.Queue[tuple[str, ModelResult | None, str | None]]" = queue.Queue()
        self._pending: dict[str, Future[Any]] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._pending_lock = threading.Lock()
        self.error: str | None = None

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> "LiveAnnotationRunner":
        self._thread.start()
        return self

    def request_stop(self) -> None:
        """Stop after draining what is already durable. Used when the source will never close."""

        self._stop.set()
        self.source.wake_readers()

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # ---------------------------------------------------------------- consumer control

    def control(self, raw: Any) -> dict[str, Any]:
        """Accept one consumer control message for the runner thread to apply.

        Validation happens here so the sender gets an immediate answer; the
        durable acknowledgement (`annotation.control.received` / `refused`)
        is published by the runner thread, in stream order, so every consumer
        sees the same history. Returns the acknowledgement the sender gets.
        """

        with self._control_lock:
            self._control_counter += 1
            default_id = f"ctl:{self._control_counter}"
        if self.finished or self.output.closed:
            return {"accepted": False, "control_id": default_id, "reason": "annotation_stream_sealed"}
        try:
            control = normalize_control(raw, default_id=default_id)
        except ControlError as exc:
            self._controls.put(("refused", {"raw_op": raw.get("op") if isinstance(raw, dict) else None, "control_id": default_id}, str(exc)))
            self.source.wake_readers()
            return {"accepted": False, "control_id": default_id, "reason": str(exc)}
        if control.op == CONTROL_PROTOCOL_UPDATE:
            revision = self._resolve_revision(control.payload["protocol_revision_id"])
            if revision is None:
                reason = "annotation_protocol_unknown"
                self._controls.put(("refused", control.public(), reason))
                self.source.wake_readers()
                return {"accepted": False, "control_id": control.control_id, "reason": reason}
        self._controls.put(("control", control, None))
        self.source.wake_readers()
        return {"accepted": True, "control_id": control.control_id, "op": control.op, "queued": True}

    @property
    def finished(self) -> bool:
        return not self._thread.is_alive() and self.output.closed

    # ---------------------------------------------------------------- main loop

    def _run(self) -> None:
        try:
            self._process = self._spawn(self.revision.code, dict(self.revision.configuration))
        except ProtocolProcessError as exc:
            self._record_error("spawn", str(exc))
            self._close_output("spawn_failed")
            return
        self.output.append(
            KIND_BOUND,
            {
                "rollout_id": self.rollout_id,
                "source_stream_id": self.source.stream_id,
                "protocol_revision_id": self.revision.revision_id,
                "protocol_id": self._process.protocol_id or self.revision.protocol_id,
                "configuration_digest": self.revision.configuration_digest,
                "model": self.model.model if self.model is not None else None,
                "isolation_receipt": self._process.isolation_receipt,
            },
        )
        cursor = 0
        alive = True
        try:
            while alive:
                progressed = False
                alive = self._apply_controls()
                if not alive:
                    break
                for envelope in self.source.after(cursor):
                    if envelope.sequence is None:
                        continue
                    cursor = envelope.sequence
                    self.summary.consumed_high_water = cursor
                    progressed = True
                    alive = self._feed_event(envelope.to_dict(), cursor)
                    if not alive:
                        break
                    self._deliver_results()
                    alive = self._apply_controls()
                    if not alive:
                        break
                if not alive:
                    break
                progressed = self._deliver_results() or progressed
                if self._consumer_stop.is_set():
                    break
                if self.source.closed and cursor >= self.source.high_water:
                    break
                if self._stop.is_set() and cursor >= self.source.high_water:
                    break
                if not progressed:
                    self.source.wait_for_change(cursor, timeout=self.limits.idle_wait_seconds)
            if alive:
                self._drain_pending()
                self._close_process()
            outcome = "completed" if alive else "protocol_failed"
            if alive and self._consumer_stop.is_set():
                outcome = "stopped_by_consumer"
            self._close_output(outcome)
        except Exception as exc:  # noqa: BLE001 - the runner must always seal its stream
            self._record_error("runner", f"{type(exc).__name__}:{exc}")
            self._close_output("runner_failed")
        finally:
            if self._process is not None and self._process.alive:
                self._process.kill()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------- protocol I/O

    def _feed_event(self, event: dict[str, Any], cursor: int) -> bool:
        assert self._process is not None
        compact = {
            "kind": event["kind"],
            "sequence": event.get("sequence"),
            "ts": event.get("ts"),
            "payload": event.get("payload") or {},
        }
        try:
            emissions = self._process.on_event(compact)
        except ProtocolProcessError as exc:
            self._record_error("on_event", str(exc), source_sequence=cursor)
            return False
        self._publish(emissions, source_sequence=cursor)
        return True

    def _deliver_results(self) -> bool:
        """Hand completed model results to the protocol. Returns True if any were delivered."""

        delivered = False
        while True:
            try:
                request_id, result, error = self._results.get_nowait()
            except queue.Empty:
                return delivered
            delivered = True
            if self._process is None or not self._process.alive:
                continue
            source_sequence = self.summary.consumed_high_water
            if result is not None:
                self.summary.model_completed += 1
                self.output.append(
                    KIND_MODEL_COMPLETED,
                    {
                        "request_id": request_id,
                        "model": result.model,
                        "usage": {
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "total_tokens": result.total_tokens,
                        },
                        "parsed": result.parsed is not None,
                        "provider_request_id": result.provider_request_id,
                        "source_sequence": source_sequence,
                    },
                )
                payload: dict[str, Any] | None = {"text": result.text, "parsed": result.parsed}
            else:
                self.summary.model_failed += 1
                self.output.append(
                    KIND_MODEL_FAILED,
                    {"request_id": request_id, "reason": error or "unknown", "source_sequence": source_sequence},
                )
                payload = None
            try:
                emissions = self._process.on_model_result(request_id, payload, error)
            except ProtocolProcessError as exc:
                self._record_error("on_model_result", str(exc), source_sequence=source_sequence)
                continue
            self._publish(emissions, source_sequence=source_sequence)

    def _publish(self, raw_emissions: list[Any], *, source_sequence: int) -> None:
        for raw in raw_emissions:
            try:
                emission = normalize_emission(raw, source_stream_id=self.source.stream_id)
            except EmissionError as exc:
                self._record_error("emission", str(exc), source_sequence=source_sequence)
                continue
            payload = dict(emission.payload)
            payload["source_sequence"] = source_sequence
            payload["protocol_revision_id"] = self.revision.revision_id
            if emission.op == OP_FINDING:
                self._publish_finding(payload, source_sequence)
            elif emission.op == OP_RETRACT:
                self._publish_retract(payload, source_sequence)
            elif emission.op == OP_METRIC:
                self.summary.metrics += 1
                self.output.append(KIND_METRIC, payload)
            elif emission.op == OP_MODEL_REQUEST:
                self._request_model(payload, source_sequence)

    def _publish_finding(self, payload: dict[str, Any], source_sequence: int) -> None:
        finding_id = payload["finding_id"]
        if finding_id in self.summary.active_findings or finding_id in self.summary.retracted:
            self._record_error("emission", f"finding.duplicate_id:{finding_id}", source_sequence=source_sequence)
            return
        supersedes = payload.get("supersedes")
        if supersedes is not None:
            previous = self.summary.active_findings.pop(supersedes, None)
            if previous is None:
                self._record_error(
                    "emission", f"finding.supersedes_unknown:{supersedes}", source_sequence=source_sequence
                )
                return
        self.summary.findings += 1
        self.summary.active_findings[finding_id] = payload
        self.output.append(KIND_FINDING, payload)

    def _publish_retract(self, payload: dict[str, Any], source_sequence: int) -> None:
        finding_id = payload["finding_id"]
        previous = self.summary.active_findings.pop(finding_id, None)
        if previous is None:
            self._record_error("emission", f"retract.unknown_finding:{finding_id}", source_sequence=source_sequence)
            return
        self.summary.retractions += 1
        self.summary.retracted.add(finding_id)
        self.output.append(KIND_RETRACTED, payload)

    # ---------------------------------------------------------------- control application

    def _apply_controls(self) -> bool:
        """Apply queued consumer controls in order. Returns False if the protocol died."""

        while True:
            try:
                status, item, reason = self._controls.get_nowait()
            except queue.Empty:
                return True
            source_sequence = self.summary.consumed_high_water
            if status == "refused":
                self.summary.controls_refused += 1
                self.output.append(
                    KIND_CONTROL_REFUSED,
                    {**(item if isinstance(item, dict) else {}), "reason": reason, "source_sequence": source_sequence},
                )
                continue
            control = item
            assert isinstance(control, Control)
            if control.op == CONTROL_STOP:
                self.summary.controls_received += 1
                self.output.append(
                    KIND_CONTROL_RECEIVED,
                    {**control.public(), "applied": True, "source_sequence": source_sequence},
                )
                self._consumer_stop.set()
                return True
            if control.op == CONTROL_MESSAGE:
                if self._process is None or not self._process.alive:
                    return False
                try:
                    emissions = self._process.on_message(control.payload["message"])
                except ProtocolProcessError as exc:
                    self._record_error("on_message", str(exc), source_sequence=source_sequence)
                    return False
                self.summary.controls_received += 1
                self.output.append(
                    KIND_CONTROL_RECEIVED,
                    {
                        **control.public(),
                        "applied": True,
                        "handled": bool(self._process.hooks.get("on_message")),
                        "source_sequence": source_sequence,
                    },
                )
                self._publish(emissions, source_sequence=source_sequence)
                continue
            if control.op == CONTROL_PROTOCOL_UPDATE:
                if not self._rebind(control, source_sequence):
                    return False
                continue

    def _rebind(self, control: Control, source_sequence: int) -> bool:
        """Hot-swap the protocol process to another installed revision.

        The old process is asked for a snapshot first (when the control asks
        to carry state and the protocol implements it); the new revision boots
        with that state and must come up before the old one is retired. A
        revision that cannot boot leaves the current protocol running and is
        recorded as a refused control, never as a dead stream.
        """

        revision_id = control.payload["protocol_revision_id"]
        revision = self._resolve_revision(revision_id)
        if revision is None or self._process is None:
            self.summary.controls_refused += 1
            self.output.append(
                KIND_CONTROL_REFUSED,
                {**control.public(), "reason": "annotation_protocol_unknown", "source_sequence": source_sequence},
            )
            return True
        state: dict[str, Any] | None = None
        if control.payload["carry_state"] and self._process.alive:
            try:
                state = self._process.snapshot()
            except ProtocolProcessError as exc:
                self._record_error("snapshot", str(exc), source_sequence=source_sequence)
                return False
        try:
            replacement = self._spawn(revision.code, dict(revision.configuration), state)
        except ProtocolProcessError as exc:
            self.summary.controls_refused += 1
            self.output.append(
                KIND_CONTROL_REFUSED,
                {**control.public(), "reason": f"protocol_boot_failed:{str(exc)[-300:]}", "source_sequence": source_sequence},
            )
            return True
        previous = self._process
        previous_revision = self.revision.revision_id
        self._process = replacement
        self.revision = revision
        if self._model_for is not None:
            self.model = self._model_for(revision)
        previous.kill()
        self.summary.controls_received += 1
        self.summary.rebinds += 1
        self.output.append(
            KIND_CONTROL_RECEIVED,
            {**control.public(), "applied": True, "source_sequence": source_sequence},
        )
        self.output.append(
            KIND_PROTOCOL_REBOUND,
            {
                "rollout_id": self.rollout_id,
                "previous_protocol_revision_id": previous_revision,
                "protocol_revision_id": revision.revision_id,
                "protocol_id": replacement.protocol_id or revision.protocol_id,
                "configuration_digest": revision.configuration_digest,
                "state_carried": bool(state is not None and replacement.restored),
                "state_offered": state is not None,
                "model": self.model.model if self.model is not None else None,
                "isolation_receipt": replacement.isolation_receipt,
                "source_sequence": source_sequence,
                "control_id": control.control_id,
            },
        )
        return True

    # ---------------------------------------------------------------- model calls

    def _request_model(self, payload: dict[str, Any], source_sequence: int) -> None:
        request_id = payload["request_id"]
        with self._pending_lock:
            if request_id in self._pending:
                self._record_error("emission", f"model_request.duplicate_id:{request_id}", source_sequence=source_sequence)
                return
        if self.model is None:
            # Published by _deliver_results so the protocol still gets on_model_result.
            self._results.put((request_id, None, "model_not_configured"))
            return
        if self.summary.model_requested >= self.limits.max_model_calls:
            self._results.put((request_id, None, "model_call_ceiling"))
            return
        self.summary.model_requested += 1
        max_output_tokens = min(
            int(payload.get("max_output_tokens") or self.limits.max_model_output_tokens),
            self.limits.max_model_output_tokens,
        )
        self.output.append(
            KIND_MODEL_REQUESTED,
            {
                "request_id": request_id,
                "model": self.model.model,
                "instructions_digest": text_digest(payload["instructions"]),
                "context_digest": text_digest(payload["context"]),
                "context_chars": len(payload["context"]),
                "max_output_tokens": max_output_tokens,
                "structured": payload.get("schema") is not None,
                "source_sequence": source_sequence,
            },
        )
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, self.limits.model_workers), thread_name_prefix=f"annotation-model:{self.rollout_id}"
            )
        caller = self.model
        instructions = payload["instructions"]
        context = payload["context"]
        schema = payload.get("schema")

        def _call() -> None:
            try:
                result = caller(
                    instructions=instructions,
                    context=context,
                    schema=schema,
                    max_output_tokens=max_output_tokens,
                )
                self._results.put((request_id, result, None))
            except ModelUnavailable as exc:
                self._results.put((request_id, None, str(exc)))
            except Exception as exc:  # noqa: BLE001 - provider failures are evidence, not crashes
                self._results.put((request_id, None, f"{type(exc).__name__}:{str(exc)[:200]}"))
            finally:
                with self._pending_lock:
                    self._pending.pop(request_id, None)
                self.source.wake_readers()

        with self._pending_lock:
            self._pending[request_id] = self._executor.submit(_call)

    def _drain_pending(self) -> None:
        """After the source closes, wait (bounded) for in-flight judgments and deliver them."""

        deadline = time.monotonic() + self.limits.drain_timeout_seconds
        while True:
            self._deliver_results()
            with self._pending_lock:
                pending = len(self._pending)
            if pending == 0 and self._results.empty():
                return
            if time.monotonic() >= deadline:
                with self._pending_lock:
                    abandoned = list(self._pending)
                    for future in self._pending.values():
                        future.cancel()
                    self._pending.clear()
                for request_id in abandoned:
                    self.summary.model_failed += 1
                    self.output.append(
                        KIND_MODEL_FAILED,
                        {
                            "request_id": request_id,
                            "reason": "drain_timeout",
                            "source_sequence": self.summary.consumed_high_water,
                        },
                    )
                return
            time.sleep(0.05)

    # ---------------------------------------------------------------- closing

    def _close_process(self) -> None:
        if self._process is None:
            return
        try:
            emissions = self._process.close()
        except Exception as exc:  # noqa: BLE001
            self._record_error("on_close", f"{type(exc).__name__}:{exc}")
            return
        self._publish(emissions, source_sequence=self.summary.consumed_high_water)

    def _record_error(self, stage: str, detail: str, *, source_sequence: int | None = None) -> None:
        self.summary.protocol_errors += 1
        if stage in {"spawn", "on_event", "on_model_result", "on_message", "snapshot", "on_close", "runner"}:
            self.error = f"{stage}:{detail}"
        if self.output.closed:
            return
        self.output.append(
            KIND_PROTOCOL_ERROR,
            {
                "stage": stage,
                "detail": detail[-1000:],
                "source_sequence": source_sequence if source_sequence is not None else self.summary.consumed_high_water,
                "protocol_revision_id": self.revision.revision_id,
            },
        )

    def _close_output(self, outcome: str) -> None:
        if self.output.closed:
            return
        self.summary.outcome = outcome
        self.output.append(
            KIND_CLOSED,
            {
                "rollout_id": self.rollout_id,
                "outcome": outcome,
                "source_closed": self.source.closed,
                "error": self.error,
                **self.summary.public(),
            },
        )
        high_water = self.output.high_water
        self.output.append(KIND_HIGH_WATER, {"high_water": high_water})
        self.output.append(KIND_CAPTURE_CLOSED, {"high_water": high_water})
        self.output.mark_closed()
