"""The public refusal record keeps a typed engine failure's code, stage, name, last state and cause."""

import json

from synth_containers.eval_cli import refusal_payload
from synth_containers.native_engine import NativeEngineError


def test_native_engine_error_keeps_its_typed_fields():
    exc = NativeEngineError("native_engine_create_failed", stage="startup", name="synth-eval-abc",
                            last_state="failed", cause="Error response from daemon: No such image: sha256:77a1")
    payload = refusal_payload(exc)
    assert payload == {"error": "evaluation_refused", "reason": "native_engine_create_failed", "stage": "startup",
                       "name": "synth-eval-abc", "last_state": "failed",
                       "cause": "Error response from daemon: No such image: sha256:77a1"}
    json.dumps(payload)


def test_private_text_is_withheld_but_the_code_is_not():
    exc = NativeEngineError("native_engine_startup_failed", stage="startup", name=None, last_state=None,
                            cause="provider said: {\"secret\": \"x\"}")
    payload = refusal_payload(exc)
    assert payload["reason"] == "native_engine_startup_failed"
    assert payload["cause"] is None
    plain = refusal_payload(RuntimeError("disk full; retry later"))
    assert plain == {"error": "evaluation_refused", "reason": "disk full; retry later"}
    private = refusal_payload(ValueError("{\"token\": \"abc\"}"))
    assert private == {"error": "evaluation_refused", "reason": "ValueError"}
