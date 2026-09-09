"""Runner semantics: ordering, provisional findings, supersede/retract, model brokering, sealing."""

from __future__ import annotations

import threading
import time
from typing import Any

from synth_containers.event_log import RolloutEventLog
from synth_containers.live_annotation.contract import (
    KIND_BOUND,
    KIND_CAPTURE_CLOSED,
    KIND_CLOSED,
    KIND_FINDING,
    KIND_HIGH_WATER,
    KIND_METRIC,
    KIND_MODEL_COMPLETED,
    KIND_MODEL_FAILED,
    KIND_MODEL_REQUESTED,
    KIND_PROTOCOL_ERROR,
    KIND_RETRACTED,
    PROTOCOL_SCHEMA,
    ProtocolRevision,
    protocol_revision_id,
)
from synth_containers.live_annotation.model import ModelResult
from synth_containers.live_annotation.runner import LiveAnnotationRunner, RunnerLimits


def _revision(code: str, configuration: dict[str, Any] | None = None) -> ProtocolRevision:
    raw = code.encode("utf-8")
    configuration = configuration or {}
    revision_id, sha, cfg = protocol_revision_id(
        code=raw, protocol_id="test", configuration=configuration, source_revision=None
    )
    return ProtocolRevision(
        revision_id=revision_id,
        protocol_id="test",
        code=raw,
        code_sha256=sha,
        configuration=configuration,
        configuration_digest=cfg,
        source_revision=None,
        installed_at="now",
    )


def _logs(name: str) -> tuple[RolloutEventLog, RolloutEventLog]:
    source = RolloutEventLog(rollout_id=name, stream_id=f"stream:{name}")
    output = RolloutEventLog(rollout_id=name, stream_id=f"stream:{name}:annotations")
    return source, output


def _kinds(log: RolloutEventLog) -> list[str]:
    return [item.kind for item in log.after(0) if item.sequence is not None]


def _payloads(log: RolloutEventLog, kind: str) -> list[dict[str, Any]]:
    return [item.payload for item in log.after(0) if item.kind == kind]


ACHIEVEMENTS = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
PROTOCOL_ID = "test"

class Protocol:
    def __init__(self, config):
        self.seen = set()
        self.provisional = None

    def on_event(self, event):
        out = []
        kind = event["kind"]
        payload = event["payload"]
        if kind == "observation":
            for name in payload.get("achievements") or []:
                if name not in self.seen:
                    self.seen.add(name)
                    out.append({{"op": "finding", "finding_id": "ach:" + name, "kind": "achievement",
                                "label": name, "step": payload.get("step"),
                                "evidence": {{"sequences": [event["sequence"]]}}}})
        if kind == "action" and payload.get("action") == "noop":
            if self.provisional is None:
                self.provisional = "fm:dither:1"
                out.append({{"op": "finding", "finding_id": self.provisional, "kind": "failure_mode",
                            "label": "dithering", "confidence": 0.4,
                            "evidence": {{"sequences": [event["sequence"]]}}}})
            else:
                nxt = "fm:dither:2"
                out.append({{"op": "finding", "finding_id": nxt, "kind": "failure_mode",
                            "label": "dithering", "confidence": 0.8, "supersedes": self.provisional,
                            "evidence": {{"sequences": [event["sequence"]]}}}})
                self.provisional = nxt
        if kind == "action" and payload.get("action") == "do" and self.provisional:
            out.append({{"op": "retract", "finding_id": self.provisional, "reason": "progress resumed"}})
            self.provisional = None
        if kind == "action" and payload.get("action") == "bad":
            out.append({{"op": "finding", "kind": "note"}})  # malformed: no id/label
        return out

    def on_close(self):
        return [{{"op": "metric", "name": "achievements", "value": len(self.seen)}}]
'''


def test_findings_supersede_retract_and_seal_after_source_closes() -> None:
    source, output = _logs("r1")
    runner = LiveAnnotationRunner(
        rollout_id="r1", source=source, output=output, revision=_revision(ACHIEVEMENTS)
    ).start()

    source.append("trace.opened", {"rollout_id": "r1"})
    source.append("observation", {"step": 0, "achievements": []})
    source.append("action", {"action": "noop"})
    source.append("action", {"action": "noop"})
    source.append("observation", {"step": 2, "achievements": ["collect_wood"]})
    source.append("action", {"action": "do"})
    source.append("action", {"action": "bad"})
    source.append("observation", {"step": 4, "achievements": ["collect_wood", "place_table"]})
    source.append("capture.closed", {"high_water": source.high_water})
    source.mark_closed()

    assert runner.join(timeout=15), "runner did not finish"
    kinds = _kinds(output)
    assert kinds[0] == KIND_BOUND
    assert kinds[-3:] == [KIND_CLOSED, KIND_HIGH_WATER, KIND_CAPTURE_CLOSED]
    assert output.closed is True

    findings = _payloads(output, KIND_FINDING)
    ids = [row["finding_id"] for row in findings]
    assert ids == ["fm:dither:1", "fm:dither:2", "ach:collect_wood", "ach:place_table"]
    first = findings[2]
    assert first["status"] == "provisional"
    assert first["evidence"] == {"stream_id": "stream:r1", "sequences": [5]}
    assert first["source_sequence"] == 5
    assert first["step"] == 2
    assert first["protocol_revision_id"] == runner.revision.revision_id
    superseding = findings[1]
    assert superseding["supersedes"] == "fm:dither:1" and superseding["confidence"] == 0.8

    retracted = _payloads(output, KIND_RETRACTED)
    assert [row["finding_id"] for row in retracted] == ["fm:dither:2"]

    errors = _payloads(output, KIND_PROTOCOL_ERROR)
    assert len(errors) == 1 and errors[0]["stage"] == "emission"
    assert errors[0]["detail"].startswith("finding.finding_id_required")

    metrics = _payloads(output, KIND_METRIC)
    assert metrics == [
        {
            "name": "achievements",
            "value": 2.0,
            "step": None,
            "source_sequence": source.high_water,
            "protocol_revision_id": runner.revision.revision_id,
        }
    ]

    closed = _payloads(output, KIND_CLOSED)[0]
    assert closed["outcome"] == "completed"
    assert closed["consumed_high_water"] == source.high_water
    assert closed["findings"] == 4 and closed["retractions"] == 1
    assert closed["protocol_errors"] == 1 and closed["active_findings"] == 2
    assert runner.summary.retracted == {"fm:dither:2"}
    high_water = _payloads(output, KIND_HIGH_WATER)[0]["high_water"]
    assert _payloads(output, KIND_CAPTURE_CLOSED)[0]["high_water"] == high_water


def test_runner_is_ordered_by_source_sequence_under_concurrent_appends() -> None:
    source, output = _logs("r2")
    code = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
class Protocol:
    def __init__(self, config):
        self.last = 0
    def on_event(self, event):
        assert event["sequence"] == self.last + 1, (event["sequence"], self.last)
        self.last = event["sequence"]
        if event["kind"] == "action":
            return [{{"op": "metric", "name": "seq", "value": event["sequence"]}}]
        return []
'''
    runner = LiveAnnotationRunner(rollout_id="r2", source=source, output=output, revision=_revision(code)).start()

    def producer() -> None:
        for index in range(200):
            source.append("action" if index % 3 == 0 else "observation", {"i": index})
            if index % 50 == 0:
                time.sleep(0.01)
        source.mark_closed()

    thread = threading.Thread(target=producer)
    thread.start()
    thread.join()
    assert runner.join(timeout=20)
    values = [row["value"] for row in _payloads(output, KIND_METRIC)]
    assert values == [float(i + 1) for i in range(200) if i % 3 == 0]
    assert values == sorted(values)
    assert runner.summary.protocol_errors == 0


def test_protocol_crash_seals_stream_and_leaves_source_untouched() -> None:
    source, output = _logs("r3")
    code = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
class Protocol:
    def __init__(self, config):
        pass
    def on_event(self, event):
        if event["sequence"] == 2:
            raise ValueError("cannot judge this")
        return []
'''
    runner = LiveAnnotationRunner(rollout_id="r3", source=source, output=output, revision=_revision(code)).start()
    source.append("a", {})
    source.append("b", {})
    source.append("c", {})
    assert runner.join(timeout=15)
    assert output.closed is True
    assert source.closed is False and source.high_water == 3
    closed = _payloads(output, KIND_CLOSED)[0]
    assert closed["outcome"] == "protocol_failed"
    assert closed["consumed_high_water"] == 2
    assert "cannot judge this" in closed["error"]
    error = _payloads(output, KIND_PROTOCOL_ERROR)[0]
    assert error["stage"] == "on_event" and error["source_sequence"] == 2


def test_request_stop_drains_then_closes_when_source_never_seals() -> None:
    source, output = _logs("r4")
    code = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
class Protocol:
    def __init__(self, config):
        pass
    def on_event(self, event):
        return [{{"op": "metric", "name": "seen", "value": event["sequence"]}}]
'''
    runner = LiveAnnotationRunner(rollout_id="r4", source=source, output=output, revision=_revision(code)).start()
    source.append("a", {})
    source.append("b", {})
    time.sleep(0.2)
    runner.request_stop()
    assert runner.join(timeout=15)
    assert [row["value"] for row in _payloads(output, KIND_METRIC)] == [1.0, 2.0]
    closed = _payloads(output, KIND_CLOSED)[0]
    assert closed["outcome"] == "completed" and closed["source_closed"] is False


class _FakeCaller:
    model = "fake-judge"

    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, instructions: str, context: str, schema: Any, max_output_tokens: int) -> ModelResult:
        self.calls.append({"instructions": instructions, "context": context, "schema": schema, "max": max_output_tokens})
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("provider down")
        return ModelResult(
            text='{"milestone": "tools.craft_wood_pickaxe", "hit": true}',
            parsed={"milestone": "tools.craft_wood_pickaxe", "hit": True},
            model="fake-judge",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


JUDGE = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
class Protocol:
    def __init__(self, config):
        self.asked = 0
    def on_event(self, event):
        if event["kind"] == "span.policy.plan":
            self.asked += 1
            return [{{"op": "model_request", "request_id": "q" + str(self.asked),
                     "instructions": "Judge the plan. Answer with JSON.",
                     "context": "plan=" + str(event["payload"].get("actions")),
                     "schema": {{"type": "object", "properties": {{"hit": {{"type": "boolean"}}}}}},
                     "max_output_tokens": 5000}}]
        return []
    def on_model_result(self, request_id, result, error):
        if error is not None:
            return [{{"op": "metric", "name": "judge_failed", "value": 1}}]
        parsed = result.get("parsed") or {{}}
        if parsed.get("hit"):
            return [{{"op": "finding", "finding_id": "ms:" + request_id, "kind": "milestone",
                     "label": parsed.get("milestone"), "confidence": 0.7,
                     "evidence": {{"sequences": []}}}}]
        return []
'''


def test_model_requests_are_brokered_bounded_and_returned_to_the_protocol() -> None:
    source, output = _logs("r5")
    caller = _FakeCaller()
    runner = LiveAnnotationRunner(
        rollout_id="r5",
        source=source,
        output=output,
        revision=_revision(JUDGE),
        model=caller,
        limits=RunnerLimits(max_model_calls=2, max_model_output_tokens=300),
    ).start()
    for index in range(3):
        source.append("span.policy.plan", {"actions": ["up", "do"], "i": index})
    source.mark_closed()
    assert runner.join(timeout=20)

    requested = _payloads(output, KIND_MODEL_REQUESTED)
    assert [row["request_id"] for row in requested] == ["q1", "q2"]
    assert requested[0]["max_output_tokens"] == 300 and requested[0]["structured"] is True
    assert "instructions" not in requested[0] and "context" not in requested[0]
    assert requested[0]["instructions_digest"].startswith("sha256:")
    assert len(caller.calls) == 2 and caller.calls[0]["max"] == 300

    completed = _payloads(output, KIND_MODEL_COMPLETED)
    assert [row["request_id"] for row in completed] == ["q1", "q2"]
    assert completed[0]["usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    failed = _payloads(output, KIND_MODEL_FAILED)
    assert [(row["request_id"], row["reason"]) for row in failed] == [("q3", "model_call_ceiling")]

    findings = _payloads(output, KIND_FINDING)
    assert [row["finding_id"] for row in findings] == ["ms:q1", "ms:q2"]
    assert findings[0]["label"] == "tools.craft_wood_pickaxe"
    assert [row["name"] for row in _payloads(output, KIND_METRIC)] == ["judge_failed"]
    closed = _payloads(output, KIND_CLOSED)[0]
    assert closed["model_requested"] == 2 and closed["model_completed"] == 2 and closed["model_failed"] == 1


def test_model_not_configured_and_provider_failure_are_evidence_not_crashes() -> None:
    source, output = _logs("r6")
    runner = LiveAnnotationRunner(rollout_id="r6", source=source, output=output, revision=_revision(JUDGE)).start()
    source.append("span.policy.plan", {"actions": ["up"]})
    source.mark_closed()
    assert runner.join(timeout=15)
    assert [row["reason"] for row in _payloads(output, KIND_MODEL_FAILED)] == ["model_not_configured"]
    assert [row["name"] for row in _payloads(output, KIND_METRIC)] == ["judge_failed"]

    source, output = _logs("r7")
    runner = LiveAnnotationRunner(
        rollout_id="r7", source=source, output=output, revision=_revision(JUDGE), model=_FakeCaller(fail=True)
    ).start()
    source.append("span.policy.plan", {"actions": ["up"]})
    source.mark_closed()
    assert runner.join(timeout=15)
    failed = _payloads(output, KIND_MODEL_FAILED)
    assert len(failed) == 1 and failed[0]["reason"].startswith("RuntimeError:provider down")
    assert _payloads(output, KIND_CLOSED)[0]["outcome"] == "completed"


def test_in_flight_judgment_is_delivered_after_the_source_closes() -> None:
    source, output = _logs("r8")
    runner = LiveAnnotationRunner(
        rollout_id="r8",
        source=source,
        output=output,
        revision=_revision(JUDGE),
        model=_FakeCaller(delay=0.4),
        limits=RunnerLimits(drain_timeout_seconds=5.0),
    ).start()
    source.append("span.policy.plan", {"actions": ["up"]})
    source.mark_closed()
    assert runner.join(timeout=20)
    kinds = _kinds(output)
    assert kinds.index(KIND_MODEL_COMPLETED) < kinds.index(KIND_CLOSED)
    assert [row["finding_id"] for row in _payloads(output, KIND_FINDING)] == ["ms:q1"]


def test_drain_timeout_abandons_slow_judgments_and_still_seals() -> None:
    source, output = _logs("r9")
    runner = LiveAnnotationRunner(
        rollout_id="r9",
        source=source,
        output=output,
        revision=_revision(JUDGE),
        model=_FakeCaller(delay=3.0),
        limits=RunnerLimits(drain_timeout_seconds=0.3),
    ).start()
    source.append("span.policy.plan", {"actions": ["up"]})
    source.mark_closed()
    assert runner.join(timeout=20)
    assert [row["reason"] for row in _payloads(output, KIND_MODEL_FAILED)] == ["drain_timeout"]
    assert output.closed is True
