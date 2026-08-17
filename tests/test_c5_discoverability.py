"""C-5: per-capability input_schema and capabilities_digest. luna_low is not a config."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from synth_containers.platform import create_compat_app
from synth_containers.platform.craftax_world import ACTIONS
from synth_containers.platform.state import TASK_INSTANCE_ID_PATTERN, CompatPlatform, _canonical_sha256
from synth_containers.platform.targets import TARGETS

_OPENAPI = (
    Path(__file__).resolve().parents[1] / "openapi" / "container-contract-v1.yaml"
)


def test_craftax_metadata_advertises_input_schema_and_digest() -> None:
    client = TestClient(create_compat_app("craftax_engine"))
    payload = client.get("/info").json()
    schema = payload["input_schema"]
    assert schema["properties"]["task_instance_id"]["pattern"] == TASK_INSTANCE_ID_PATTERN
    assert schema["properties"]["task_instance_id"]["pattern"] == r"^.*:(-?\d+)$"
    vocab = schema["properties"]["action_vocabulary"]["items"]["enum"]
    assert list(ACTIONS) == vocab
    assert payload["capabilities"]["craftax_engine"]["input_schema"] == schema
    assert payload["capabilities_digest"].startswith("sha256:")
    platform = client.app.state.platform
    assert payload["capabilities_digest"] == platform.capabilities_digest()
    assert payload["capabilities_digest"] == _canonical_sha256(platform.capability_metadata())
    advertised = {row["config"] for row in payload["policy_refs"]}
    assert advertised == {"luna_med", "sol_med"}
    assert "luna_low" not in advertised
    assert "luna_low" not in str(payload)


def test_banking77_schema_has_seed_range_and_intent_vocab() -> None:
    payload = TestClient(create_compat_app("banking77_classify")).get("/metadata").json()
    seed = payload["input_schema"]["properties"]["seed"]
    assert seed["minimum"] == 0
    assert seed["maximum"] >= 1
    vocab = payload["input_schema"]["properties"]["action_vocabulary"]["items"]["enum"]
    assert "card_arrival" in vocab
    assert payload["capabilities_digest"] == CompatPlatform(
        TARGETS["banking77_classify"]
    ).capabilities_digest()


def test_openapi_documents_task_instance_id_pattern() -> None:
    text = _OPENAPI.read_text(encoding="utf-8")
    assert r"^.*:(-?\d+)$" in text
    assert "example: seed:0" in text
    assert "capabilities_digest" in text
    assert "luna_low" not in text
