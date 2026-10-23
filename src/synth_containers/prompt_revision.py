"""Prompt-program adapter for the shared policy revision registry."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .http_models import RolloutPolicySpecModel
from .policy_revision import PolicyAdapter, PolicyManifest, PolicyRevisionError


def prompt_adapter(runtime: Any) -> PolicyAdapter:
    def prepare(revision_id: str, manifest: PolicyManifest, timeout_s: int) -> dict[str, Any]:
        result = runtime.program()
        if inspect.isawaitable(result):

            async def bounded():
                return await asyncio.wait_for(result, timeout=timeout_s)

            try:
                result = asyncio.run(bounded())
            except TimeoutError as exc:
                raise PolicyRevisionError("policy_startup_timeout") from exc
        program = result.to_dict() if hasattr(result, "to_dict") else result
        if not isinstance(program, dict):
            raise PolicyRevisionError("policy_program_invalid")
        allowed = {
            module.get("candidate_field")
            for module in program.get("modules", [])
            if module.get("mutable") and module.get("candidate_field")
        }
        fields = manifest.components.prompt_fields or {}
        if not fields or not fields.keys() <= allowed:
            raise PolicyRevisionError("policy_prompt_field_forbidden")
        config = manifest.components.config or {}
        if set(config) - {"policy"}:
            raise PolicyRevisionError("policy_configuration_unsupported")
        if config.get("policy"):
            RolloutPolicySpecModel.model_validate(config["policy"])
        return {
            "policy_revision_id": revision_id,
            "harness": "prompt_program",
            "readiness": {"kind": "validated_configuration", "model_loaded": False},
        }

    return PolicyAdapter(frozenset({"prompt_fields", "config"}), prepare)
