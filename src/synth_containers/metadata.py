"""Single composition point for the container ``/metadata`` (``/info``) payload.

Both producers — the authoring SDK (:mod:`synth_containers.sdk`) and the hosted
compat platform (:mod:`synth_containers.platform.state`) — route their payloads
through :func:`compose_metadata_payload` so the wire shape cannot drift between
them again.  ``openapi/container-contract-v1.yaml`` (``InfoResponse``) types
exactly the shape composed here, and ``contracts/fixtures/metadata/`` holds
producer-generated golden fixtures for both producers.

Canonical locations (what new consumers should read):

- ``capabilities.protocol`` — protocol id string (containers own these ids).
- ``capabilities.contract_version`` — stamped from
  :data:`METADATA_CONTRACT_VERSION`, the one Python constant.
- ``capabilities.live_frames`` — one of :data:`LIVE_FRAMES_LEVELS`.
- ``capabilities.metadata.policy_ready`` / ``program_ready`` (and
  ``grader_ready`` where a runtime has a distinct scorer role) — sourced from
  the runtime's actual readiness, for every runtime family.
- ``metadata.optimizer_contracts.<name>`` — optimizer sub-contract
  advertisements (the optimizers Rust consumer reads
  ``metadata.optimizer_contracts.gepa`` and nothing else).

Transitional duplicates (kept so old consumers keep working; each is marked
with a ``COMPAT:`` comment at the emit site and removed after one release):

- top-level ``optimizer_contracts`` (the old ``platform/state.py`` location);
- ``capabilities.optimizer_contracts`` (the old healthbench-only
  ``platform/app.py`` splice location).
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .ontology import CONTRACT_VERSION

# The one Python constant the wire ``capabilities.contract_version`` is stamped
# from.  ``openapi/container-contract-v1.yaml`` documents this value; nothing
# else may mint a contract-version string.
METADATA_CONTRACT_VERSION = CONTRACT_VERSION

# Protocol id strings.  Containers own these; consumers key behavior on them
# (Workshop's admission gates on LIVE_EVAL_PROTOCOL).
CONTAINER_CONTRACT_PROTOCOL = "synth.container.contract.v1"
LIVE_EVAL_PROTOCOL = "synth.container.live-eval.v1"

# The typed vocabulary for ``capabilities.live_frames``.
LIVE_FRAMES_LEVELS = ("native", "sampled", "post_hoc", "unsupported")

# COMPAT removal gate: the canonical location has been
# ``metadata.optimizer_contracts`` since the 2026-08 contract release.  Keep
# the two legacy copies only through that release, then delete this flag and
# both emit sites in the first subsequent release.  A named, pinned gate keeps
# the migration window reviewable instead of turning the aliases permanent.
EMIT_COMPAT_OPTIMIZER_CONTRACT_DUPLICATES = True
COMPAT_OPTIMIZER_CONTRACT_DUPLICATES_THROUGH = "2026-08"

IMAGE_DIGEST_ENV = "SYNTH_CONTAINER_IMAGE_DIGEST"
PRODUCER_SOURCE_REVISION_ENV = "SYNTH_CONTAINER_PRODUCER_SOURCE_REVISION"
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PRODUCER_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class RuntimeProvenance:
    """Immutable producer identity injected by the trusted image launcher."""

    image_digest: str | None = None
    producer_source_revision: str | None = None


def runtime_provenance_from_environment(
    environ: Mapping[str, str] | None = None,
) -> RuntimeProvenance:
    source = os.environ if environ is None else environ
    image_digest = str(source.get(IMAGE_DIGEST_ENV) or "").strip().lower() or None
    producer_revision = (
        str(source.get(PRODUCER_SOURCE_REVISION_ENV) or "").strip().lower() or None
    )
    if image_digest is not None and _IMAGE_DIGEST.fullmatch(image_digest) is None:
        raise ValueError("container_image_digest_invalid")
    if (
        producer_revision is not None
        and _PRODUCER_SOURCE_REVISION.fullmatch(producer_revision) is None
    ):
        raise ValueError("container_producer_source_revision_invalid")
    return RuntimeProvenance(
        image_digest=image_digest,
        producer_source_revision=producer_revision,
    )


def attach_runtime_provenance(
    payload: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Stamp trusted launch provenance at stable top-level `/info` paths."""

    result = copy.deepcopy(payload)
    provenance = runtime_provenance_from_environment(environ)
    if provenance.image_digest is not None:
        result["imageDigest"] = provenance.image_digest
    if provenance.producer_source_revision is not None:
        result["producerSourceRevision"] = provenance.producer_source_revision
    return result


@dataclass(frozen=True)
class RuntimeReadiness:
    """A runtime family's actual readiness, as advertised on the wire.

    ``policy_ready``/``program_ready`` are emitted for every runtime family
    (the healthbench-only shim formerly in ``platform/app.py`` is lifted into
    the shared composition).  ``grader_ready`` is emitted only by runtimes
    with a distinct scorer role (healthbench).
    """

    policy_ready: bool
    program_ready: bool
    grader_ready: bool | None = None

    def capability_metadata(self) -> dict[str, bool]:
        row: dict[str, bool] = {
            "policy_ready": bool(self.policy_ready),
            "program_ready": bool(self.program_ready),
        }
        if self.grader_ready is not None:
            row["grader_ready"] = bool(self.grader_ready)
        return row


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def compose_metadata_payload(
    *,
    base: dict[str, Any],
    protocol: str,
    live_frames: str,
    readiness: RuntimeReadiness,
    optimizer_contracts: dict[str, Any] | None = None,
    scale_leases: int | None = None,
) -> dict[str, Any]:
    """Compose the full ``/metadata`` payload from a producer's base facts.

    ``base`` carries the producer-specific fields (identity, affordances,
    input schema, route hints, ...); this function owns everything the
    cross-repo contract keys on, so the two producers cannot drift.
    The input is not mutated.
    """

    if not str(protocol).strip():
        raise ValueError("metadata_protocol_required")
    if live_frames not in LIVE_FRAMES_LEVELS:
        raise ValueError(
            f"metadata_live_frames_invalid: {live_frames!r} not in {LIVE_FRAMES_LEVELS}"
        )

    payload = copy.deepcopy(dict(base))

    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    payload["capabilities"] = capabilities
    capabilities["protocol"] = str(protocol)
    capabilities["contract_version"] = METADATA_CONTRACT_VERSION
    capabilities["live_frames"] = str(live_frames)
    if scale_leases is not None:
        capabilities["scale_leases"] = int(scale_leases)

    capability_metadata = capabilities.get("metadata")
    if not isinstance(capability_metadata, dict):
        capability_metadata = {}
    # Actual readiness is authoritative: it overwrites any stale base value.
    capability_metadata.update(readiness.capability_metadata())
    capabilities["metadata"] = capability_metadata

    if optimizer_contracts:
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        payload["metadata"] = metadata
        existing = metadata.get("optimizer_contracts")
        if isinstance(existing, dict):
            # A producer-declared contract row beats the composed default
            # (preserves the SDK's user-metadata override semantics).
            contracts = _deep_merge(copy.deepcopy(dict(optimizer_contracts)), existing)
        else:
            contracts = copy.deepcopy(dict(optimizer_contracts))
        # Canonical location: the optimizers Rust consumer reads ONLY
        # metadata.optimizer_contracts.gepa.
        metadata["optimizer_contracts"] = contracts
        if EMIT_COMPAT_OPTIMIZER_CONTRACT_DUPLICATES:
            # COMPAT through 2026-08: legacy platform consumers read
            # optimizer_contracts at the payload top level (old
            # platform/state.py location). Remove in the first subsequent
            # release once all consumers read metadata.optimizer_contracts.
            payload.setdefault("optimizer_contracts", copy.deepcopy(contracts))
            # COMPAT through 2026-08: the healthbench-only platform/app.py
            # splice also published the contract under
            # capabilities.optimizer_contracts. Remove with the top-level
            # copy above.
            capabilities.setdefault("optimizer_contracts", copy.deepcopy(contracts))

    return attach_runtime_provenance(payload)
