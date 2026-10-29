"""SYN-4321 G6: ``workload`` -> ``trace_source`` in the trace capture binding.

v1 carries the trace source under ``workload``; v2 under ``trace_source``. Decoders
accept either key, new bindings are minted as v2, and the digest is defined per
version so a sealed v1 binding verifies identically.
"""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from synth_containers.tracing.canonical import canonical_text, content_digest
from synth_containers.tracing.capture.binding import (
    BINDING_SCHEMA_VERSION,
    BINDING_SCHEMA_VERSION_V1,
    BINDING_SCHEMA_VERSION_V2,
    BindingCaptureV1,
    BindingContextV1,
    BindingTraceSourceV1,
    BindingVersionError,
    BindingWorkloadV1,
    TraceCaptureBindingV1,
    TraceSourceKind,
    WorkloadKind,
    mint_binding,
)
from synth_containers.tracing.validation.rehydrate import RehydrationError, build


FIXTURE = Path(__file__).parent / "fixtures" / "trace_capture_binding" / "v1_mloky_wave3.json"


def _v1_payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _minted() -> TraceCaptureBindingV1:
    return mint_binding(
        trace_id="trace_g6",
        capture_id="cap_g6",
        trace_source=BindingTraceSourceV1(
            kind=TraceSourceKind.CODEX,
            root_actor_id="actor_root",
            actor_session_id="sess_root",
            run_id="run_1",
        ),
        capture=BindingCaptureV1(output_artifact_root="/bundle"),
    )


def test_current_version_is_v2() -> None:
    assert BINDING_SCHEMA_VERSION == BINDING_SCHEMA_VERSION_V2 == "synth.trace-capture-binding.v2"
    assert BINDING_SCHEMA_VERSION_V1 == "synth.trace-capture-binding.v1"
    assert BindingWorkloadV1 is BindingTraceSourceV1
    assert WorkloadKind is TraceSourceKind


def test_real_v1_binding_verifies_identically() -> None:
    raw = FIXTURE.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)
    binding = build(TraceCaptureBindingV1, payload)
    assert isinstance(binding, TraceCaptureBindingV1)
    assert binding.schema_version == BINDING_SCHEMA_VERSION_V1
    assert binding.trace_source.root_actor_id == "actor_f2886ef4778def81"
    assert binding.workload is binding.trace_source
    # The sealed digest from the mloky producer still verifies.
    assert binding.content_digest == payload["content_digest"]
    assert content_digest(binding) == payload["content_digest"]
    assert binding.verify_digest()
    # Re-serialization is the v1 wire shape, byte for byte.
    assert canonical_text(binding) == raw
    wire = binding.to_dict()
    assert "workload" in wire and "trace_source" not in wire


def test_v2_mint_emits_trace_source_and_round_trips() -> None:
    binding = _minted()
    assert binding.schema_version == BINDING_SCHEMA_VERSION_V2
    wire = binding.to_dict()
    assert wire["trace_source"]["root_actor_id"] == "actor_root"
    assert "workload" not in wire
    assert binding.verify_digest()
    rebuilt = build(TraceCaptureBindingV1, json.loads(canonical_text(binding)))
    assert rebuilt == binding
    assert rebuilt.verify_digest()


def test_digest_rule_is_per_version() -> None:
    binding = _minted()
    as_v1 = replace(binding, schema_version=BINDING_SCHEMA_VERSION_V1)
    # Same content, different version: the key differs, so the digest differs.
    assert content_digest(as_v1) != content_digest(binding)
    v1_wire = json.loads(canonical_text(as_v1))
    assert "workload" in v1_wire and "trace_source" not in v1_wire
    # Tampering is still caught under each rule.
    tampered = replace(
        binding,
        trace_source=replace(binding.trace_source, root_actor_id="actor_other"),
    )
    assert not tampered.verify_digest()


def test_either_key_decodes_under_its_version_digest() -> None:
    payload = _v1_payload()
    renamed = dict(payload)
    renamed["trace_source"] = renamed.pop("workload")
    binding = build(TraceCaptureBindingV1, renamed)
    # The key spelling on input does not change the v1 digest rule.
    assert binding.verify_digest()
    assert binding.content_digest == payload["content_digest"]


def test_both_keys_refused() -> None:
    payload = _v1_payload()
    payload["trace_source"] = payload["workload"]
    with pytest.raises(RehydrationError, match="both 'workload' and 'trace_source'"):
        build(TraceCaptureBindingV1, payload)


def test_unknown_version_refused() -> None:
    payload = _v1_payload()
    payload["schema_version"] = "synth.trace-capture-binding.v9"
    with pytest.raises(RehydrationError, match="unknown trace capture binding schema_version"):
        build(TraceCaptureBindingV1, payload)
    binding = replace(_minted(), schema_version="synth.trace-capture-binding.v9")
    with pytest.raises(BindingVersionError):
        content_digest(binding)


def test_mint_accepts_legacy_keyword_once() -> None:
    source = BindingTraceSourceV1(kind="other", root_actor_id="a", actor_session_id="s")
    capture = BindingCaptureV1()
    legacy = mint_binding(trace_id="t", capture_id="c", workload=source, capture=capture)
    current = mint_binding(trace_id="t", capture_id="c", trace_source=source, capture=capture)
    assert legacy.binding_id == current.binding_id
    assert legacy.trace_source == current.trace_source
    with pytest.raises(TypeError):
        mint_binding(trace_id="t", capture_id="c", capture=capture)
    with pytest.raises(TypeError):
        mint_binding(
            trace_id="t", capture_id="c", workload=source, trace_source=source, capture=capture
        )


def test_written_binding_file_is_v2(tmp_path: Path) -> None:
    path = _minted().write(tmp_path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == BINDING_SCHEMA_VERSION_V2
    assert "trace_source" in on_disk and "workload" not in on_disk


AGENTS_V2_FIXTURE = FIXTURE.parent / "v2_mloky_agents_service.json"


def test_rust_agents_service_v2_binding_verifies_identically() -> None:
    """Cross-language: a binding written by mloky agents-service (Rust) verifies here.

    The fixture is the byte-exact golden of mloky
    ``agents-service/tests/trace_v5.rs`` (the Rust test asserts it produces exactly
    these bytes). The Python reader rebuilds the binding from its dataclasses, so
    any wire key outside the model is dropped and the digest would stop verifying;
    the binding must re-serialize byte for byte.
    """

    raw = AGENTS_V2_FIXTURE.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)
    binding = build(TraceCaptureBindingV1, payload)
    assert isinstance(binding, TraceCaptureBindingV1)
    assert binding.schema_version == BINDING_SCHEMA_VERSION_V2
    assert binding.trace_source.root_actor_id == payload["trace_source"]["root_actor_id"]
    assert binding.content_digest == payload["content_digest"]
    assert content_digest(binding) == payload["content_digest"]
    assert binding.verify_digest()
    assert canonical_text(binding) == raw
    assert set(payload["context"]) <= {item.name for item in fields(BindingContextV1)}
