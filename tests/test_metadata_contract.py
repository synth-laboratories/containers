"""The /metadata contract: golden fixtures, typed schema, old-consumer regressions.

The two producers (the authoring SDK and the hosted compat platform) compose
their /metadata payloads through ``synth_containers.metadata``.  This module:

- renders producer-generated golden fixtures for both producers into
  ``contracts/fixtures/metadata/`` (the seed of the cross-repo golden corpus
  that optimizers and workshop consumers run in CI);
- asserts both payloads parse against the typed ``InfoResponse`` schema in
  ``openapi/container-contract-v1.yaml``;
- asserts the exact fields the OLD consumers require are still served:
  ``metadata.optimizer_contracts.gepa`` (optimizers Rust
  ``container_contract.rs``), ``capabilities.metadata.policy_ready``
  (optimizers Python preflight), ``capabilities.protocol`` and
  ``capabilities.live_frames`` (workshop capability validation).

Regenerate the fixtures after an intentional contract change with:

    SYNTH_UPDATE_METADATA_FIXTURES=1 pytest tests/test_metadata_contract.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import synth_containers.metadata as metadata_contract
from synth_containers.metadata import (
    COMPAT_OPTIMIZER_CONTRACT_DUPLICATES_THROUGH,
    CONTAINER_CONTRACT_PROTOCOL,
    EMIT_COMPAT_OPTIMIZER_CONTRACT_DUPLICATES,
    LIVE_EVAL_PROTOCOL,
    METADATA_CONTRACT_VERSION,
)
from synth_containers.platform import create_compat_app
from synth_containers.platform.targets import TARGETS
from synth_containers.sdk import Container

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "contracts" / "fixtures" / "metadata"
OPENAPI_PATH = REPO_ROOT / "openapi" / "container-contract-v1.yaml"

SDK_FIXTURE = FIXTURES_DIR / "info-response.sdk.reference-container.json"


# --- producer renderers -----------------------------------------------------


def _render_sdk_payload() -> dict[str, Any]:
    """An sdk.py-style container, rendered through the real fastapi route."""

    container = Container(
        "golden-sdk-container",
        description="Golden fixture: authoring-SDK reference container",
        metadata={"fixture": "contracts/fixtures/metadata"},
        policy_ready=True,
    )

    @container.rollout
    def _rollout(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    @container.program
    def _program() -> dict[str, Any]:  # pragma: no cover
        return {"version": "prompt_program.v1", "program_id": "golden.v1", "modules": []}

    @container.taskset
    def _taskset() -> dict[str, Any]:  # pragma: no cover
        return {"taskset_id": "golden.v1", "splits": {"train": 1}}

    @container.taskset_tasks
    def _taskset_tasks(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        return {"tasks": [], "metadata": {}}

    return TestClient(container.fastapi()).get("/metadata").json()


@pytest.fixture()
def deterministic_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin credential presence so readiness (and the fixtures) are stable."""

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-credential-present")


# --- golden fixtures --------------------------------------------------------


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _assert_matches_fixture(payload: dict[str, Any], fixture_path: Path) -> None:
    rendered = _canonical(payload)
    if os.environ.get("SYNTH_UPDATE_METADATA_FIXTURES") == "1" or not fixture_path.exists():
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        fixture_path.write_text(rendered, encoding="utf-8")
    stored = fixture_path.read_text(encoding="utf-8")
    assert rendered == stored, (
        f"{fixture_path.name} drifted from the producer's live /metadata payload. "
        "If the contract change is intentional, regenerate with "
        "SYNTH_UPDATE_METADATA_FIXTURES=1 and re-run consumer corpora."
    )


def test_sdk_metadata_matches_golden_fixture(deterministic_credentials: None) -> None:
    _assert_matches_fixture(_render_sdk_payload(), SDK_FIXTURE)


# --- typed-schema conformance ----------------------------------------------


def _info_response_validator() -> Any:
    yaml = pytest.importorskip("yaml")
    jsonschema = pytest.importorskip("jsonschema")
    doc = yaml.safe_load(OPENAPI_PATH.read_text(encoding="utf-8"))
    schemas = doc["components"]["schemas"]
    bundled = {
        **schemas["InfoResponse"],
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schemas,
    }
    # OpenAPI refs -> local $defs refs so a plain JSON Schema validator runs them.
    rewritten = json.loads(json.dumps(bundled).replace("#/components/schemas/", "#/$defs/"))
    return jsonschema.Draft202012Validator(rewritten)


def test_golden_fixtures_parse_against_typed_info_response_schema(
    deterministic_credentials: None,
) -> None:
    validator = _info_response_validator()
    payloads = {"sdk": _render_sdk_payload()}
    for fixture_path in (SDK_FIXTURE,):
        assert fixture_path.exists(), f"golden fixture missing: {fixture_path}"
        payloads[fixture_path.name] = json.loads(fixture_path.read_text(encoding="utf-8"))
    for name, payload in payloads.items():
        errors = [
            f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in validator.iter_errors(payload)
        ]
        assert not errors, f"{name} does not satisfy InfoResponse: {errors}"


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_retained_platform_info_satisfies_shared_schema(
    target: str, tmp_path: Path
) -> None:
    payload = TestClient(
        create_compat_app(target, storage_root=tmp_path / target)
    ).get("/info").json()
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in _info_response_validator().iter_errors(payload)
    ]
    assert not errors, f"{target} does not satisfy InfoResponse: {errors}"
    capabilities = payload["capabilities"]
    assert capabilities["protocol"] == LIVE_EVAL_PROTOCOL
    assert capabilities["live_frames"] == payload["live_frames"]
    assert capabilities["scale_leases"] == payload["scale_leases"]
    assert isinstance(capabilities["metadata"]["policy_ready"], bool)
    assert capabilities["metadata"]["program_ready"] is False


# --- old-consumer regressions ----------------------------------------------


def _assert_old_consumer_fields(payload: dict[str, Any]) -> None:
    # optimizers Rust container_contract.rs reads ONLY metadata.optimizer_contracts.gepa.
    gepa = payload["metadata"]["optimizer_contracts"]["gepa"]
    assert gepa["version"] == "synth_optimizers.gepa.v2"
    # optimizers Python (pre-fix) requires capabilities.metadata.policy_ready.
    capabilities = payload["capabilities"]
    assert isinstance(capabilities["metadata"]["policy_ready"], bool)
    assert isinstance(capabilities["metadata"]["program_ready"], bool)
    # workshop capability validation reads capabilities.protocol / live_frames.
    assert capabilities["protocol"]
    assert capabilities["live_frames"] in ("native", "sampled", "post_hoc", "unsupported")
    # one constant stamps the wire contract_version.
    assert capabilities["contract_version"] == METADATA_CONTRACT_VERSION
    # COMPAT transitional duplicates old consumers still read.
    assert EMIT_COMPAT_OPTIMIZER_CONTRACT_DUPLICATES is True
    assert COMPAT_OPTIMIZER_CONTRACT_DUPLICATES_THROUGH == "2026-08"
    assert payload["optimizer_contracts"]["gepa"]["version"] == gepa["version"]
    assert capabilities["optimizer_contracts"]["gepa"]["version"] == gepa["version"]


def test_sdk_gepa_advertisement_requires_every_required_callback() -> None:
    container = Container(
        "incomplete-gepa-sdk",
        metadata={"optimizer_contracts": {"gepa": {"version": "synth_optimizers.gepa.v2"}}},
    )

    @container.rollout
    def _rollout(payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    @container.program
    def _program() -> dict[str, Any]:  # pragma: no cover
        return {"version": "prompt_program.v1", "program_id": "incomplete.v1", "modules": []}

    payload = TestClient(container.fastapi()).get("/metadata").json()
    for parent in (
        payload,
        payload["capabilities"],
        payload["metadata"],
    ):
        assert "gepa" not in parent.get("optimizer_contracts", {})


def test_compat_gate_removes_only_legacy_contract_locations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metadata_contract,
        "EMIT_COMPAT_OPTIMIZER_CONTRACT_DUPLICATES",
        False,
    )
    payload = metadata_contract.compose_metadata_payload(
        base={},
        protocol=CONTAINER_CONTRACT_PROTOCOL,
        live_frames="unsupported",
        readiness=metadata_contract.RuntimeReadiness(
            policy_ready=True,
            program_ready=True,
        ),
        optimizer_contracts={"gepa": {"version": "synth_optimizers.gepa.v2"}},
    )
    assert payload["metadata"]["optimizer_contracts"]["gepa"]["version"]
    assert "optimizer_contracts" not in payload
    assert "optimizer_contracts" not in payload["capabilities"]


def test_sdk_producer_still_satisfies_old_consumers(deterministic_credentials: None) -> None:
    payload = _render_sdk_payload()
    _assert_old_consumer_fields(payload)
    assert payload["capabilities"]["protocol"] == CONTAINER_CONTRACT_PROTOCOL
    assert payload["capabilities"]["metadata"]["policy_ready"] is True
    assert payload["capabilities"]["metadata"]["program_ready"] is True
