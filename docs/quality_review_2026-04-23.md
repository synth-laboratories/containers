# Quality Review: 2026-04-23

This note records the first public-package hardening pass after moving
`synth-containers` into `synth-cookbooks-public`.

## Prior Bug Patterns Reviewed

- Rollout identity drift: MiniGrid lost the optimizer-provided rollout identity,
  which made later state and checkpoint reads look broken even though rollout
  execution worked.
- Runtime-contract success vs optimizer recurrence: MiniGrid and NLE can prove
  the container contract while higher-level Go-Explore proposer logic still
  fails to produce enough non-baseline candidates.
- Artifact/read-model disagreement: several Synth bugs involved artifacts,
  summaries, and terminal state disagreeing about what completed.
- Checkpoint authority drift: checkpoint rows, recovery projections, and route
  payloads need one canonical meaning for restore eligibility and branchability.
- Local path leakage: public docs must not point at private Downloads or local
  checkout paths.

## Hardening Applied

- Added `/compatibility` to the public OpenAPI and HTTP client surface.
- Added root discovery to the OpenAPI so it matches the reference FastAPI app.
- Made `artifacts` the canonical HTTP payload key.
- Added source-layout README guidance for package ownership boundaries.
- Removed soft pydantic import fallback and enum-parse try/except wrappers.
- Rewrote historical submission handoff links so public docs do not expose local
  machine paths.
- Tightened HTTP ingress models to reject unknown request fields.
- Added `ReferenceManagedRuntime.counter_default()` as the one-call reference
  runtime constructor for docs, examples, and contract smoke checks.
- Kept recursive JSON aliases out of Pydantic ingress models after schema
  generation proved they recurse too deeply.

## Remaining Primetime Gaps

- `http_adapter.ManagedRuntime` and `protocols.py` still overlap; the adapter
  protocol should become the single runtime HTTP ownership boundary or delegate
  cleanly to the primitive protocol layer.
- Some formatter/recovery paths still operate on raw mapping payloads. Keep
  shrinking those toward typed dataclasses at ingress.
- The OpenAPI remains permissive in several response schemas. Tighten the
  public schema as downstream migrations settle.
- The package has no checked-in automated tests in the public repo yet; add
  tests only when the team explicitly chooses the coverage shape.
