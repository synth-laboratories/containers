# synth_containers Source Layout

This package owns the public Synth container contract. Keep the package rooted
in typed dataclasses, enums, and protocol surfaces; adapters should project from
those canonical objects rather than inventing parallel wire semantics.

## Ownership Boundaries

- `ontology.py`, `nouns.py`, `capabilities.py`, `tool_runtime.py`, and
  `proxying.py` own canonical vocabulary and typed model declarations.
- `profiles.py`, `compatibility.py`, and `adapters.py` own derived capability
  and consumer-fit projections. They should not reinterpret runtime state.
- `formats.py`, `wire.py`, `http_models.py`, `http_adapter.py`, and
  `http_client.py` own the public HTTP contract. The canonical artifact list key
  is `artifacts`.
- `runtime_requests.py` owns typed runtime control requests after HTTP ingress
  validation. Runtime lifecycle code should consume those dataclasses instead
  of probing loose request dictionaries.
- `contracts.py` and `recovery.py` own algorithm-facing contract dumps and
  recovery projections for long-horizon consumers.
- `reference_runtime.py` owns the executable counter runtime used to prove the
  contract surface without binding the package to a production backend.

## Adding Code

Add fields to the canonical dataclasses first, then update the formatter,
OpenAPI schema, README route inventory, and compatibility projection. Avoid
adding durable authority to process-local registries unless the type is clearly
documented as a reference or in-memory implementation.
