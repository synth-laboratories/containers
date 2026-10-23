"""Adapt the shared revision protocol to the compat platform's immutable pins."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..package_models import content_digest
from ..policy_revision import (
    PolicyAdapter,
    PolicyManifest,
    PolicyRevisionError,
    PolicyRevisionRegistry,
)
from .http_requests import ISOLATED_POLICY_HARNESS
from .policy_process import IsolatedPolicyProcess
from .state_records import PolicyConfig, PolicyRevision

if TYPE_CHECKING:
    from .state import CompatPlatform


def registry_for(platform: CompatPlatform) -> PolicyRevisionRegistry:
    def prepare(revision_id: str, manifest: PolicyManifest, timeout_s: int) -> dict[str, Any]:
        components = manifest.components
        config = components.config or {}
        code: bytes | None = None
        receipt: dict[str, Any] = {"kind": "validated_configuration"}
        if components.code_bundle:
            bundle = components.code_bundle
            if (
                bundle.language != "python"
                or len(bundle.files) != 1
                or bundle.serve_contract != "process_observation_action.v1"
                or bundle.dependency_digest is not None
            ):
                raise PolicyRevisionError("policy_code_contract_unsupported", status=409)
            code = bundle.files[bundle.entrypoint].encode()
            try:
                process = IsolatedPolicyProcess(code, startup_timeout_s=timeout_s)
                receipt = {**process.isolation_receipt, "kind": "readiness_probe"}
                process.close()
            except TimeoutError as exc:
                raise PolicyRevisionError("policy_startup_timeout") from exc
            except (RuntimeError, ValueError, OSError) as exc:
                raise PolicyRevisionError("policy_startup_failed") from exc
        with platform._policy_mutation_lock:
            # Each config has the same immutable identity as its revision.
            # No default pointer or running process is changed here.
            platform.policy_configs[revision_id] = PolicyConfig(
                config_id=revision_id, harness=manifest.harness, config=deepcopy(config)
            )
            platform.policy_revisions[revision_id] = PolicyRevision(
                revision_id=revision_id,
                digest=revision_id.removeprefix("polrev_"),
                harness=manifest.harness,
                config_id=revision_id,
                code=code,
                isolation_receipt=receipt,
                namespace=manifest.namespace,
                name=manifest.name,
                configuration_digest=content_digest(config),
                model_digest=content_digest({}),
                source_revision=manifest.source.get("revision"),
                installed_at=datetime.now(timezone.utc).isoformat(),
            )
        return {
            "policy_revision_id": revision_id,
            "policy_ref": {"harness": manifest.harness, "config": revision_id},
            "readiness": receipt,
        }

    adapters: dict[str, PolicyAdapter] = {}
    if platform.spec.affordances.level("bind_policy_config") != "unsupported":
        for reference in platform._advertised_policy_refs():
            harness = reference.get("harness")
            if harness and harness != ISOLATED_POLICY_HARNESS:
                adapters[str(harness)] = PolicyAdapter(frozenset({"config"}), prepare)
    if (
        platform.spec.affordances.level("update_policy_code") != "unsupported"
        and getattr(platform.spec.runtime, "policy_revision_code_contract", None)
        == "process_observation_action.v1"
    ):
        adapters[ISOLATED_POLICY_HARNESS] = PolicyAdapter(frozenset({"code_bundle"}), prepare)
    return PolicyRevisionRegistry(adapters)
