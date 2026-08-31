# Target runtimes

Children of `TargetRuntime`. Platform never names these in `state.py` control flow.

**In scope:** episode / trial simulation for one `TargetRuntimeKind`, or a
`TargetSpec.runtime` supplied by an image package (``EXTERNAL``).

**Not in scope:** pins, leases, sequence logs, `/reward`, occupancy, HTTP.
Particular worlds (Craftax gold, …) do not live in this package.

**Do not bypass:** add a child here and a `TargetRuntimeKind` entry in `_BY_FAMILY`,
or set `spec.runtime`. Do not add `if target_id == ...` to `CompatPlatform`.

World selection is `environment_ref`. Do not branch on `target_id`.
