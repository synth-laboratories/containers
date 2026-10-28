"""Frozen CISPO producer bytes for the pinned ``synth_optimizers`` consumer.

SYN-4648. Containers produces two documents the optimizer decodes before any
spend: the hashed capability document (``cispo.capabilities.v1``) and the
handshake verdict (``cispo.handshake.v1``). The optimizer consumer is closed on
schema version, on the capability hash and on clause/verdict enums, so a drift
here refuses a run in the field rather than at review.

The fixtures under ``tests/fixtures/optimizer_cispo`` are byte-identical to the
copies in ``synth-laboratories/optimizers`` at
``tests/rl/fixtures/containers_cispo``. This test proves the REAL current
producer still emits exactly the pinned key set. Regenerate (only after the
consumer has been rolled) with::

    PYTHONPATH=src:. python tests/test_strict_old_consumer_optimizers_cispo.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from synth_containers.cispo_contract import build_cispo_capability_document
from synth_containers.cispo_handshake_facts import canonical_capability_hash
from tests.test_cispo_contract import (
    minimal_declaration,
    minimal_surface,
    rich_declaration,
    rich_surface,
)
from tests.test_cispo_handshake import facts, registry, request_payload

FIXTURES = Path(__file__).parent / "fixtures" / "optimizer_cispo"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CONSUMER = MANIFEST["consumer"]
PIN = f"{CONSUMER}@{MANIFEST['source_revision']}"

# A canary the producer may declare. The consumer's own digest rule is
# ``cispo.renderer_canary.v1`` over the rendered prompt tokens.
CANARY_TOKENS = (101, 102, 103)


def compact(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _rehash(document: dict[str, Any]) -> dict[str, Any]:
    document = dict(document)
    document.pop("capability_hash", None)
    document["capability_hash"] = canonical_capability_hash(document)
    return document


def _canary_digest(tokens: tuple[int, ...]) -> str:
    # Mirrors synth_optimizers.contracts.rl_records.canary_digest at the pin.
    payload = {"canary": "cispo.renderer_canary.v1", "prompt_token_ids": list(tokens)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


def produce() -> dict[str, dict[str, Any]]:
    """Every fixture, from the real producer; refusals are derived from it."""

    rich = build_cispo_capability_document(rich_declaration(), rich_surface())
    minimal = build_cispo_capability_document(minimal_declaration(), minimal_surface())
    accepted = registry().handshake(request_payload()).to_payload()
    without = facts(
        evidence=replace(facts().evidence, tokens_in_tokens_out=False),
        reward=replace(facts().reward, settlement_window_seconds=0.0),
    )
    unsupported_optional = registry(without).handshake(request_payload()).to_payload()

    canaried = dict(rich)
    canaried["renderer_profile"] = {
        **rich["renderer_profile"],
        "canary_digest": _canary_digest(CANARY_TOKENS),
    }
    return {
        "capabilities.rich": rich,
        "capabilities.minimal": minimal,
        "capabilities.canaried": _rehash(canaried),
        "handshake_verdict.accepted": accepted,
        "handshake_verdict.unsupported_optional": unsupported_optional,
        # Refusals the pinned consumer must answer with its typed error.
        "refuse.capabilities.unsupported_version": _rehash(
            {**rich, "schema_version": "cispo.capabilities.v2"}
        ),
        "refuse.handshake_verdict.unsupported_version": {
            **accepted,
            "schema_version": "cispo.handshake.v2",
        },
        "refuse.handshake_verdict.unknown_verdict": {
            **accepted,
            "clauses": [
                {**accepted["clauses"][0], "verdict": "partially_accepted"},
                *accepted["clauses"][1:],
            ],
        },
    }


def key_paths(value: Any, prefix: str = "") -> set[str]:
    """Every object key path; list items collapse to ``[]``."""

    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.add(path)
            paths |= key_paths(item, path)
    elif isinstance(value, list):
        for item in value:
            paths |= key_paths(item, f"{prefix}[]")
    return paths


def test_fixtures_match_the_manifest() -> None:
    for name, expected in MANIFEST["fixtures"].items():
        data = (FIXTURES / f"{name}.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, f"{name} drifted from {PIN}"
        assert data == compact(json.loads(data)), f"{name} is not compact sorted JSON"


def test_current_producer_emits_exactly_the_pinned_key_set() -> None:
    current = produce()
    assert sorted(current) == sorted(MANIFEST["fixtures"]), (
        f"fixture set changed; the pinned consumer {PIN} has not seen {sorted(current)}"
    )
    for name, document in current.items():
        pinned = json.loads((FIXTURES / f"{name}.json").read_text())
        produced, frozen = key_paths(document), key_paths(pinned)
        assert produced == frozen, (
            f"{name}: producer key set differs from what {PIN} was frozen against; "
            f"added={sorted(produced - frozen)} removed={sorted(frozen - produced)}. "
            "Roll the consumer first, then regenerate both copies."
        )


def test_current_producer_versions_are_the_pinned_versions() -> None:
    current = produce()
    for name in ("capabilities.rich", "capabilities.minimal", "handshake_verdict.accepted"):
        pinned = json.loads((FIXTURES / f"{name}.json").read_text())
        assert current[name]["schema_version"] == pinned["schema_version"], f"{name} vs {PIN}"
    assert current["capabilities.rich"]["contract_version"] == "synth_optimizers.cispo.v1"


def _write() -> None:  # pragma: no cover - regeneration helper
    FIXTURES.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, document in sorted(produce().items()):
        data = compact(document)
        (FIXTURES / f"{name}.json").write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = dict(MANIFEST)
    manifest["fixtures"] = hashes
    (FIXTURES / "manifest.json").write_bytes(compact(manifest))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_write())
