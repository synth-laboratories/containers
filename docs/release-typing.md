# Reproducible release typing

Run from a clean checkout with uv and Python 3.13 available:

```sh
python3 scripts/check-type-debt.py
```

The gate prepares two repository-owned environments below `.local/`, installs
the dependency graph from `uv.lock`, then checks every Python source. It fails
on any error, warning, environment failure or unrecognized checker failure.
`scripts/ty-release-baseline.txt` remains unchanged as historical evidence; it
no longer permits old diagnostics to return.

## Profiles and source coverage

| Profile | Dependencies | Checked sources |
| --- | --- | --- |
| `main` | `dev` plus `release-types`: ty 0.0.86, Synth SDK source 9ab8088abf81cdb42f0897ecbe22c68e8f47485f, Harbor 0.22.0, Docker SDK 7.1.0, Daytona 0.210.0, Tinker 0.34.0, Transformers 5.20.2 and Prime renderers 0.1.12 | Every module except `tracing/capture/mitmproxy_addon.py` |
| `mitm` | Core project dependencies plus `mitm-types`: ty 0.0.86, mitmproxy 12.2.3 and pytest | `tracing/capture/mitmproxy_addon.py`, against real installed MITM interfaces |

The profiles are deliberately separate. The application development profile
requires cryptography >=50.0.0; mitmproxy 12.2.3 requires cryptography <=48.1.
Its other dependency ceilings also conflict with the native provider cohort.
The lock records these group/extra conflicts explicitly. No application crypto
requirement is lowered and the addon is checked separately, not suppressed.
The addon runs in the qualified interception process's environment. A passing
type check is not a provider/environment runtime qualification.

The SDK pin is an immutable GitHub revision because this exact 0.23.4 cohort is
not available from the package registry. GitHub SSH access is required on first
preparation. Its inspected build uses setuptools and copies the pinned contract
files; the gate installs that exact SDK source without dependencies after the
locked wheel-only sync. Other source distributions are refused, preventing
unintended local native/Rust compilation. Wheels do not start provider clients
or download model weights. These dependencies are development groups, not new
mandatory dependencies of an installed Containers core package.

The source fixes keep actual runtime contracts intact: nullable/mapping checks
use one inspected value, model factories preserve their supported call shapes,
SDK cleanup uses `aclose`, repeated HTTP response headers remain repeated, and
Daytona creation has a single-attempt retry wrapper matching Harbor's decorated
method interface. The wrapper never invokes Harbor's provider-create retries;
the durable create allowance and ambiguous-create regressions remain required.

## Reuse and verification

```sh
python3 scripts/check-type-debt.py --profile main
python3 scripts/check-type-debt.py --profile mitm
python3 scripts/check-type-debt.py --check-only
```

The default checks both profiles. A single-profile pass does not certify the
other. `--check-only` requires each prepared environment's lock digest to match
the current `uv.lock`; normal execution prepares it again. Use the default
command for release receipts. Profile environments are isolated from `.venv`
and from any installed/shared client or service.

Regression tests use the main profile interpreter with `PYTHONPATH=src` and an
owned temporary directory. Provider-backed/live infrastructure tests require
their own bounded qualification; do not infer a live pass from type checking.
