# Cross-repo contract fixtures

Seed of the golden corpus shared across containers / optimizers / workshop.

## `metadata/` — producer-generated `/metadata` golden payloads

Rendered by the producers themselves over the wire (see
`tests/test_metadata_contract.py`), validated against the typed
`InfoResponse` schema in `openapi/container-contract-v1.yaml`:

- `info-response.sdk.reference-container.json` — authoring-SDK container
  (`synth_containers.sdk.Container.fastapi`).
- `info-response.platform.banking77_classify.json` — hosted compat platform
  world (`platform/state.py::metadata_payload` via `create_compat_app`).

The SDK fixture registers the full GEPA callback surface. SDK containers with
only a subset of the required callbacks intentionally omit the GEPA contract
from every metadata placement; `tests/test_gepa_contract_vendored.py` pins
both the positive and negative advertisement cases.

The top-level and `capabilities.optimizer_contracts` copies are compatibility
aliases gated by `EMIT_COMPAT_OPTIMIZER_CONTRACT_DUPLICATES` only through the
2026-08 release. Canonical consumers must read
`metadata.optimizer_contracts`; the aliases are removed in the first
subsequent release.

Regenerate after an intentional contract change with:

```
SYNTH_UPDATE_METADATA_FIXTURES=1 pytest tests/test_metadata_contract.py
```

**Downstream copies:** consumer repos (optimizers'
`tests/test_container_metadata_corpus.py`, and any workshop corpus runner)
vendor copies of these payloads. After regenerating here, re-copy the updated
fixtures into those repos — a stale downstream copy certifies a wire shape no
producer emits. Notable regeneration on 2026-08-27: the banking77 GEPA
contract gained `taskset_tasks_route` (required by the optimizers-owned
contract) — downstream copies from before that change must be refreshed.

## `gepa/` — the optimizers-owned GEPA sub-contract values

- `gepa-contract-owned.json` — committed copy of the values owned by
  `optimizers-v08-release:src/synth_optimizers/contracts/gepa_contract.py`.
  `tests/test_gepa_contract_vendored.py` asserts
  `synth_containers/vendored_gepa_contract.py` matches this copy and that both
  producers' advertised contracts satisfy the required-route set with routes
  that are actually served. When the owning module changes, update the vendor
  module and this fixture together (both edits come from the owning repo).
