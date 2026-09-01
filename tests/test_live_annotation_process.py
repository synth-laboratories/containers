"""The protocol host: out-of-process, stdlib-only, fail-closed on a bad module."""

from __future__ import annotations

import pytest

from synth_containers.live_annotation.contract import PROTOCOL_SCHEMA
from synth_containers.live_annotation.process import IsolatedProtocolProcess, ProtocolProcessError

COUNTER = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
PROTOCOL_ID = "test.counter"

class Protocol:
    def __init__(self, config):
        self.n = 0
        self.prefix = str(config.get("prefix") or "act")

    def on_event(self, event):
        if event["kind"] == "action":
            self.n += 1
            return [{{"op": "finding", "finding_id": f"{{self.prefix}}-{{self.n}}", "kind": "note",
                     "label": "action " + str(event["payload"].get("action")),
                     "evidence": {{"sequences": [event["sequence"]]}}}}]
        return None

    def on_model_result(self, request_id, result, error):
        return {{"op": "metric", "name": "model", "value": 1.0 if error is None else 0.0}}

    def on_close(self):
        return [{{"op": "metric", "name": "actions", "value": self.n}}]
'''


def test_counter_protocol_round_trip() -> None:
    process = IsolatedProtocolProcess(COUNTER.encode("utf-8"), config={"prefix": "p"})
    try:
        assert process.protocol_id == "test.counter"
        assert process.isolation_receipt["sandbox"] == "process"
        assert process.on_event({"kind": "observation", "sequence": 1, "payload": {}}) == []
        out = process.on_event({"kind": "action", "sequence": 2, "payload": {"action": "up"}})
        assert out == [
            {
                "op": "finding",
                "finding_id": "p-1",
                "kind": "note",
                "label": "action up",
                "evidence": {"sequences": [2]},
            }
        ]
        assert process.on_model_result("r1", {"text": "x", "parsed": None}, None) == [
            {"op": "metric", "name": "model", "value": 1.0}
        ]
    finally:
        closing = process.close()
    assert closing == [{"op": "metric", "name": "actions", "value": 1}]
    assert process.alive is False
    assert process.close() == []


def test_marker_mismatch_is_refused_at_boot() -> None:
    with pytest.raises(ProtocolProcessError, match="protocol_startup_failed|protocol_not_ready"):
        IsolatedProtocolProcess(b"PROTOCOL = 'something.else'\nclass Protocol:\n    pass\n")


def test_import_error_is_reported_not_swallowed() -> None:
    code = f"PROTOCOL = {PROTOCOL_SCHEMA!r}\nimport definitely_not_a_module\nclass Protocol: pass\n"
    with pytest.raises(ProtocolProcessError, match="definitely_not_a_module"):
        IsolatedProtocolProcess(code.encode("utf-8"))


def test_child_cannot_import_the_container_package() -> None:
    """PYTHONPATH is scrubbed: protocol code is stdlib-only by construction."""

    code = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
class Protocol:
    def __init__(self, config):
        pass
    def on_event(self, event):
        try:
            import synth_containers  # noqa: F401
            return [{{"op": "metric", "name": "leak", "value": 1}}]
        except ImportError:
            return [{{"op": "metric", "name": "leak", "value": 0}}]
'''
    process = IsolatedProtocolProcess(code.encode("utf-8"))
    try:
        assert process.on_event({"kind": "x", "sequence": 1, "payload": {}}) == [
            {"op": "metric", "name": "leak", "value": 0}
        ]
    finally:
        process.close()


def test_exception_in_on_event_surfaces_as_request_failure() -> None:
    code = f'''
PROTOCOL = {PROTOCOL_SCHEMA!r}
class Protocol:
    def __init__(self, config):
        pass
    def on_event(self, event):
        raise RuntimeError("boom at " + str(event["sequence"]))
'''
    process = IsolatedProtocolProcess(code.encode("utf-8"))
    try:
        with pytest.raises(ProtocolProcessError, match="boom at 7"):
            process.on_event({"kind": "x", "sequence": 7, "payload": {}})
        # The process survives a handler exception; the runner decides what to do.
        assert process.alive is True
    finally:
        process.close()
