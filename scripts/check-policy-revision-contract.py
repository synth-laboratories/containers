"""Check retained policy-revision schema and canonical byte fixtures without services."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
schema = json.loads((root / "openapi/policy-revision-v1.json").read_text())
assert schema["$defs"]["PolicyManifest"]["additionalProperties"] is False
assert schema["$defs"]["PrepareRequest"]["additionalProperties"] is False
vectors = json.loads((root / "tests/fixtures/policy-revision-v1.json").read_text())
for vector in vectors["valid"]:
    encoded = json.dumps(vector["manifest"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert encoded == vector["canonical_json"]
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    assert vector["digest"] == "sha256:" + digest
    assert vector["policy_revision_id"] == "polrev_" + digest
wire = (root / "openapi/container-contract-v1.yaml").read_text()
for route in (
    "/policy-revisions:",
    "/policy-revisions/{revision_id}:",
    "/policy-revisions/{revision_id}/prepare:",
):
    assert route in wire
print(f"policy revision schema and {len(vectors['valid'])} canonical byte fixtures agree")
