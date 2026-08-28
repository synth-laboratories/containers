"""VENDORED: the GEPA optimizer sub-contract constants.

Owning source (boundary B2 — optimizers own the GEPA sub-contract semantics
and version; containers *declare conformance*):

    optimizers-v08-release: src/synth_optimizers/contracts/gepa_contract.py

Containers cannot import optimizers at runtime, so these values are vendored
here and checked against a committed copy of the owned values by
``tests/test_gepa_contract_vendored.py`` (fixture:
``contracts/fixtures/gepa/gepa-contract-owned.json``).  When the owning module
changes, re-copy the values here AND regenerate that fixture from the owning
repo — the parity test exists so a drifted vendor copy fails loudly instead of
minting a divergent contract.

Do not add fields or routes here that the owning module does not declare.
"""

from __future__ import annotations

#: Contract id/version string a conforming container must advertise at
#: ``metadata.optimizer_contracts.gepa.version``.
GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"

#: Route fields a conforming contract must serve, each an absolute path.
GEPA_CONTRACT_REQUIRED_ROUTES: tuple[str, ...] = (
    "program_route",
    "taskset_route",
    "taskset_tasks_route",
    "rollout_route",
)

#: Route fields a contract may serve.
GEPA_CONTRACT_OPTIONAL_ROUTES: tuple[str, ...] = ("trace_route",)

#: Where the contract lives inside a container's ``/metadata`` response.
#: Top-level ``metadata`` is canonical; a duplicate under
#: ``capabilities.metadata`` is not part of this contract.
GEPA_CONTRACT_METADATA_PATH: tuple[str, ...] = (
    "metadata",
    "optimizer_contracts",
    "gepa",
)


__all__ = [
    "GEPA_CONTRACT_METADATA_PATH",
    "GEPA_CONTRACT_OPTIONAL_ROUTES",
    "GEPA_CONTRACT_REQUIRED_ROUTES",
    "GEPA_OPTIMIZER_CONTRACT_VERSION",
]
