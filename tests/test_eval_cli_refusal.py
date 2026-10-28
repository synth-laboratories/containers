"""The public refusal record keeps a typed engine failure's code, stage, name, last state and cause."""

import json

import pytest

from synth_containers.eval_cli import candidates_from, refusal_payload
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


def test_candidate_model_table_is_a_typed_refusal(tmp_path):
    """SYN-5210: a catalog-bound [model] table is refused by name, not as a bare ValidationError."""
    (tmp_path / "m.toml").write_text('actions = ["noop"]\nkind = "scripted-actions-v1"\n\n'
                                     '[model]\nprovider = "openai"\nroute = "gpt-6-luna"\nweights = "hosted"\n')
    with pytest.raises(ValueError) as refused:
        candidates_from(["catalog-p-r1=m.toml"], relative=tmp_path)
    payload = refusal_payload(refused.value)
    assert payload["reason"].startswith("candidate_model_binding_unsupported: candidate catalog-p-r1 declares a model")
    (tmp_path / "bad.toml").write_text('kind = "scripted-actions-v1"\n')
    with pytest.raises(ValueError) as invalid:
        candidates_from(["x=bad.toml"], relative=tmp_path)
    assert refusal_payload(invalid.value)["reason"] == "candidate_invalid: candidate x field actions missing"
    (tmp_path / "ok.toml").write_text('actions = ["noop"]\nkind = "scripted-actions-v1"\n')
    assert candidates_from(["x=ok.toml"], relative=tmp_path)[0].content.actions == ("noop",)
