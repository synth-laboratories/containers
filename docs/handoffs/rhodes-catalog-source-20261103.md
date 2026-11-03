# Draft task/resource registry and durable log custody — source handoff

Source only. No tests, imports, schema generation, builds, compilation, provider
calls or installed compatibility checks were run.

`RhodesClient.catalog` exposes typed put/archive/inspect/list/history/operation
methods at `/v1/rhodes/catalog/{task_definitions|resource_plan_templates}`.
Task/resource/metadata DTOs mirror backend `packages/rhodes/catalog.py`; schema
names explicitly end in `.draft.v1`, because accepted scope did not finalize
field/query names. `metadata_from_source` maps explicit REB/Task Factory fields
with exact source/mapping revisions and explicit absent/unsupported unknowns.
Amounts use decimal strings; units/currencies are explicit, resource/provider
coverage is checked and aggregate hard currency budgets cover phase budgets.
Registration cannot manufacture an `enforced` claim or acquire resources.

The named registry retains immutable revision receipts and exact qualified
release/provider/validation/source/resource-plan pins. Put reactivates an
archived name at the next CAS revision. Query filters apply before pagination.
The client preserves service refusal codes through `CatalogRefused`; replay-safe
writes carry the original operation ID. CLI: `rhodes catalog task|resource_plan
list|get|put|archive|history|operation`; list's `--query` accepts the complete
`CatalogQuery` JSON and put accepts the complete `CatalogPut` JSON.

`RhodesCatalogTools.definitions()` and `.call()` provide internal MCP-compatible
SDK adapters, including `rhodes_build_get` and paged original-attempt `rhodes_build_logs`. Construction requires an explicit
current role/resource authorization callback; catalog writes are distinct from
read/run privileges. No tools are automatically granted or installed into an
existing actor bundle. Existing API-key/tenant authorization remains in effect.

`BuildOperation.log_publication` is optional for legacy responses and carries
pending/published/refused state independently of terminal execution. Its typed
decoder validates attempt/digest/reference identity and refuses foreign or
contradictory custody. Release inventory remains an additive raw response and
preserves the backend facet. `list_build_log_publications` retains access to older attempt custody after later retries. No client rebuild is used to recover missing logs.

Pressure session: `7e1f3644-476d-470f-b39d-6569dfa53bf1`.
Backend draft-registry source decision: `bcdf5ed6-e33a-4d17-b88c-df297eb8e157`.

Deferred gates: complete backend migration chain, contract/SDK/CLI/MCP assertions,
current/revoked/foreign authority, passed validation pin mismatches, Artifact
source authorization/digest failures, CAS/replay/archive/pagination and immutable
history. Internal bundle registration requires its owning admitted-grant wiring.
These drafts do not claim accepted schema naming, runtime qualification, trusted
scores, verified budget enforcement, deployment or ticket completion.
