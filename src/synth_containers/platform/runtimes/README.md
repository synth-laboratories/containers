# Target runtimes

Children of `TargetRuntime`. Platform never names these in `state.py` control flow.

**In scope:** episode / trial simulation for one `TargetRuntimeKind`.

**Not in scope:** pins, leases, sequence logs, `/reward`, occupancy, HTTP.

**Do not bypass:** add a child here and a `TargetRuntimeKind` entry in `_BY_FAMILY`. Do not add `if target_id == ...` to `CompatPlatform`.

Banking77 is content (like Craftax / dig.bench), not a Harbor or OpenEnv wrap.

World selection is `environment_ref` (`env:craftax_fixture` vs `env:craftax_gold`,
`env:digbench_mock` vs `env:digbench_relay`, `env:harbor_sandbox` vs
`env:harbor_docker`, `env:echo`). Do not branch on `target_id`.
