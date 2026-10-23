"""Immutable policy publication and process-local next-rollout preparation.

Portable schema and adapter registry. This module does not own authorization,
artifact custody, live activation, trainer publication or checkpoint restore.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from threading import RLock
from typing import Any, Callable, Literal

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from .package_models import Digest, content_bytes, content_digest

SCHEMA = "synth.policy-revision.v1"
PROFILE = "next_rollout_update.v1"


class PolicyRevisionError(ValueError):
    def __init__(self, code: str, detail: str = "", status: int = 422) -> None:
        super().__init__(detail or code)
        self.code, self.status = code, status


class ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ArtifactRef(ClosedModel):
    digest: Digest
    uri: str = Field(min_length=1)


class CodeBundle(ClosedModel):
    language: Literal["python", "rust"]
    entrypoint: str = Field(min_length=1)
    files: dict[str, str]
    serve_contract: str = Field(min_length=1)
    dependency_digest: Digest | None = None

    @model_validator(mode="after")
    def validate_files(self) -> CodeBundle:
        if not self.files or self.entrypoint not in self.files:
            raise ValueError("entrypoint must name a supplied file")
        for name in self.files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
                raise ValueError("files require canonical relative paths")
        if sum(len(v.encode()) for v in self.files.values()) > 1_048_576:
            raise ValueError("code bundle exceeds 1 MiB")
        return self


class Components(ClosedModel):
    prompt_fields: dict[str, str] | None = None
    module_demos: dict[str, list[str]] | None = None
    code_bundle: CodeBundle | None = None
    config: dict[str, Any] | None = None
    model: dict[str, str] | None = None
    weights_ref: ArtifactRef | None = None
    renderer_ref: ArtifactRef | None = None
    sampler_ref: ArtifactRef | None = None
    protocol: ArtifactRef | None = None


class PolicyManifest(ClosedModel):
    schema_version: Literal["synth.policy-revision.v1"]
    namespace: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    harness: str = Field(min_length=1, max_length=128)
    components: Components
    parent_revision_id: str | None = None
    source: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> PolicyManifest:
        if not self.components.model_dump(exclude_none=True):
            raise ValueError("at least one component is required")

        def check(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = str(key).replace("_", "").replace("-", "").lower()
                    if normalized in {
                        "credential",
                        "credentials",
                        "authorization",
                        "authbearer",
                        "bearer",
                        "privatekey",
                    } or normalized.endswith(("apikey", "secret", "token", "password")):
                        raise ValueError("credentials belong to the existing runtime owner")
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        # Source files and prompt text are opaque text; configuration is data.
        check(self.components.config)
        check(self.components.model)
        check(self.source)
        for ref in (
            self.components.weights_ref,
            self.components.renderer_ref,
            self.components.sampler_ref,
            self.components.protocol,
        ):
            if ref and ("?" in ref.uri or "@" in ref.uri or "#" in ref.uri):
                raise ValueError("artifact URIs must not contain credentials or signed queries")
        if len(content_bytes(self.model_dump(mode="json", exclude_none=True))) > 2_097_152:
            raise ValueError("manifest exceeds 2 MiB")
        return self


class PrepareRequest(ClosedModel):
    operation_id: str = Field(min_length=1, max_length=200)
    profile: str = PROFILE
    timeout_s: int = Field(default=10, ge=1, le=30)


@dataclass(frozen=True)
class PolicyAdapter:
    components: frozenset[str]
    prepare: Callable[[str, PolicyManifest, int], dict[str, Any]]


class PolicyRevisionRegistry:
    """Register only adapters whose preparation and rollout resolver are installed."""

    def __init__(self, adapters: dict[str, PolicyAdapter]) -> None:
        self.adapters = dict(adapters)
        self._manifests: dict[str, dict[str, Any]] = {}
        self._receipts: dict[str, dict[str, Any]] = {}
        self._operations: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
        self._lock = RLock()
        self._preparation_lock = RLock()

    def capability(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA,
            "profiles": [PROFILE] if self.adapters else [],
            "durability": "process_local",
            "publish_route": "/policy-revisions",
            "prepare_route": "/policy-revisions/{policy_revision_id}/prepare",
            "inspect_route": "/policy-revisions/{policy_revision_id}",
            "adapters": {
                name: sorted(adapter.components) for name, adapter in sorted(self.adapters.items())
            },
        }

    def publish(self, manifest: PolicyManifest, expected_id: str | None = None) -> dict[str, Any]:
        document = manifest.model_dump(mode="json", exclude_none=True)
        digest = content_digest(document)
        revision_id = "polrev_" + digest[7:]
        if expected_id is not None and expected_id != revision_id:
            raise PolicyRevisionError("policy_revision_conflict", status=409)
        adapter = self.adapters.get(manifest.harness)
        if adapter is None or not document["components"].keys() <= adapter.components:
            raise PolicyRevisionError("policy_component_unsupported", status=409)
        with self._lock:
            replay = revision_id in self._manifests
            self._manifests.setdefault(revision_id, deepcopy(document))
            self._receipts.setdefault(
                revision_id,
                {
                    "schema_version": SCHEMA,
                    "policy_revision_id": revision_id,
                    "digest": digest,
                    "state": "published",
                    "actual_binding": None,
                    "durability": "process_local",
                },
            )
            return {**deepcopy(self._receipts[revision_id]), "idempotent": replay}

    def inspect(self, revision_id: str) -> dict[str, Any]:
        with self._lock:
            if revision_id not in self._receipts:
                raise PolicyRevisionError("policy_revision_unknown", status=404)
            return deepcopy(self._receipts[revision_id])

    def prepare(self, revision_id: str, request: PrepareRequest) -> dict[str, Any]:
        if request.profile != PROFILE:
            raise PolicyRevisionError("policy_profile_unsupported", status=409)
        arguments = request.model_dump()
        with self._preparation_lock:
            previous = self._operations.get(request.operation_id)
            if previous:
                if previous[:2] != (revision_id, arguments):
                    raise PolicyRevisionError("policy_operation_conflict", status=409)
                return deepcopy(previous[2])
            receipt = self.inspect(revision_id)
            manifest = PolicyManifest.model_validate(deepcopy(self._manifests[revision_id]))
            if receipt["state"] != "ready":
                with self._lock:
                    self._receipts[revision_id]["state"] = "preparing"
                try:
                    binding = self.adapters[manifest.harness].prepare(
                        revision_id, manifest, request.timeout_s
                    )
                    receipt.update(state="ready", actual_binding=deepcopy(binding))
                    receipt.pop("failure", None)
                except PolicyRevisionError as exc:
                    receipt.update(state="failed", actual_binding=None, failure=exc.code)
                except Exception:
                    receipt.update(
                        state="failed", actual_binding=None, failure="policy_preparation_failed"
                    )
                with self._lock:
                    self._receipts[revision_id] = deepcopy(receipt)
            result = {**receipt, "operation_id": request.operation_id, "profile": PROFILE}
            self._operations[request.operation_id] = (revision_id, arguments, deepcopy(result))
            return result

    def resolve(self, revision_id: str) -> tuple[PolicyManifest, dict[str, Any]]:
        receipt = self.inspect(revision_id)
        if receipt["state"] != "ready":
            raise PolicyRevisionError("policy_revision_not_ready", status=409)
        with self._lock:
            return (
                PolicyManifest.model_validate(deepcopy(self._manifests[revision_id])),
                deepcopy(receipt["actual_binding"]),
            )


def mount_policy_revisions(app: Any, registry: PolicyRevisionRegistry) -> None:
    """Routes inherit the hosting app's existing authorization boundary."""
    app.state.policy_revisions = registry

    async def invoke(request: Request, revision_id: str | None = None, *, prepare: bool = False):
        try:
            body = await request.json()
            if prepare:
                result = await run_in_threadpool(
                    registry.prepare, str(revision_id), PrepareRequest.model_validate(body)
                )
            else:
                result = registry.publish(PolicyManifest.model_validate(body), revision_id)
            return result
        except (ValidationError, ValueError) as exc:
            status = exc.status if isinstance(exc, PolicyRevisionError) else 422
            code = exc.code if isinstance(exc, PolicyRevisionError) else "policy_manifest_invalid"
            return JSONResponse(status_code=status, content={"error": code})

    async def publish(request: Request):
        return await invoke(request)

    async def put(revision_id: str, request: Request):
        return await invoke(request, revision_id)

    async def prepare(revision_id: str, request: Request):
        return await invoke(request, revision_id, prepare=True)

    app.post("/policy-revisions")(publish)
    app.put("/policy-revisions/{revision_id}")(put)
    app.post("/policy-revisions/{revision_id}/prepare")(prepare)

    @app.get("/policy-revisions/{revision_id}")
    def inspect(revision_id: str):
        try:
            return registry.inspect(revision_id)
        except PolicyRevisionError as exc:
            return JSONResponse(status_code=exc.status, content={"error": exc.code})
