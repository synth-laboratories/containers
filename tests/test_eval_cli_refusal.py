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


def test_limits_mismatch_keeps_declared_and_observed():
    limits = {"declared": {"nano_cpus": 1_000_000_000, "memory_bytes": 536870912, "network_mode": "bridge"},
              "observed": {"nano_cpus": 1_000_000_000, "memory_bytes": 0, "network_mode": "bridge"},
              "egress": "unrestricted_bridge"}
    exc = NativeEngineError("native_engine_limits_mismatch", stage="limits", name="synth-eval-abc",
                            last_state="failed", cause="memory_bytes declared=536870912 observed=0", limits=limits)
    payload = refusal_payload(exc)
    assert payload["reason"] == "native_engine_limits_mismatch" and payload["stage"] == "limits"
    assert payload["limits"] == limits
    json.dumps(payload)
