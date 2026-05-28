# Tasks

This folder is reserved for task organization.

Expected contents over time:

- task family registries
- split definitions
- assets
- docs
- SQLite-backed task catalogs or normalized metadata stores

The intent is to keep task data and task metadata first-class rather than
treating tasks as opaque configs attached to runtimes.

The package-level convenience API lives in
`src/synth_containers/tasks.py`, with an intentionally lightweight
`InMemoryTaskCatalog` wrapper that keeps catalog semantics explicit while
remaining evolvable toward a SQLite-backed implementation.
