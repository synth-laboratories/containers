"""Production SDK catalog -> capability-fenced gateway -> Rivet session control.

See synth-tunnel/docs/SESSION_CHECKPOINTS.md. This adapter never holds engine
credentials, sends origin work, or treats a transport close as cancellation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Protocol
from urllib.parse import urlparse
from uuid import UUID, uuid5

import httpx

from .contracts import ApplicationCheckpoint, ArtifactReference, CheckpointFailure, SessionFence
from .coordinator import CaptureRequest
from .restore import RestoreRequest, RestoreResult


class SessionCatalog(Protocol):
    async def register_session_operation(self, connector: UUID, request) -> dict: ...
    async def session_operation(self, connector: UUID, operation: UUID) -> dict: ...
    async def connector_capability(self, connector: UUID, *, ttl_seconds: int = 300): ...


class HttpSessionAuthority:
    def __init__(self, *, catalog: SessionCatalog, gateway_url: str, service_identity: str):
        parsed = urlparse(gateway_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("invalid_session_gateway")
        if (
            not service_identity
            or service_identity != service_identity.strip()
            or len(service_identity.encode()) > 128
        ):
            raise ValueError("invalid_session_service_identity")
        self.catalog = catalog
        self.gateway_url = gateway_url.rstrip("/")
        self.service_identity = service_identity
        self._captures: dict[UUID, CaptureRequest] = {}
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(5, connect=2), follow_redirects=False)

    async def close(self):
        await self._http.aclose()

    async def _gateway(self, connector: UUID, path: str, *, method="GET"):
        capability = await self.catalog.connector_capability(connector, ttl_seconds=600)
        async with self._http.stream(
            method,
            self.gateway_url + f"/v2/connectors/{connector}/" + path,
            headers={"Authorization": "Bearer " + capability.token.get_secret_value()},
        ) as response:
            response.raise_for_status()
            payload = bytearray()
            async for chunk in response.aiter_bytes():
                if len(payload) + len(chunk) > 65536:
                    raise CheckpointFailure("session_gateway_capacity")
                payload.extend(chunk)
            result = json.loads(payload)
        if not isinstance(result, dict):
            raise CheckpointFailure("session_gateway_contract")
        return result

    async def _execute(self, connector: UUID, request):
        await self.catalog.register_session_operation(connector, request)
        result = await self._gateway(
            connector, f"session-operations/{request.operation}", method="POST"
        )
        if result.get("state") != "applied" or not isinstance(result.get("result"), dict):
            raise CheckpointFailure("session_operation_" + str(result.get("state")))
        return result["result"]

    async def _lookup(self, connector: UUID, operation: UUID):
        # Unknown registration is unavailable custody, not permission to POST.
        row = await self.catalog.session_operation(connector, operation)
        if row["state"] in {"claimed", "in_doubt"}:
            row = await self._gateway(connector, f"session-operations/{operation}")
        return row.get("result") if row["state"] == "applied" else None

    async def pause(self, request: CaptureRequest):
        from synth_ai.sdk.tunnels.session_operations import Pause

        if request.operation not in self._captures and len(self._captures) >= 16:
            raise CheckpointFailure("session_capture_context_capacity")
        self._captures[request.operation] = request
        result = await self._execute(
            request.connector,
            Pause(
                action="pause",
                operation=request.operation,
                session=request.session,
                generation=request.expected_generation,
                deadline_at=request.deadline_at,
            ),
        )
        return SessionFence(
            connector=request.connector,
            session=request.session,
            source_generation=result["source_generation"],
            generation=result["generation"],
            permit_barrier_ms=result["permit_barrier_ms"],
        )

    async def seal(self, checkpoint: ApplicationCheckpoint, artifact: ArtifactReference):
        from synth_ai.sdk.tunnels.session_operations import SealCheckpoint

        request = self._captures.get(checkpoint.operation)
        if request is None:
            raise CheckpointFailure("session_capture_context_unavailable")
        first = checkpoint.participants[0].cut
        result = await self._execute(
            checkpoint.fence.connector,
            SealCheckpoint(
                action="checkpoint",
                operation=uuid5(checkpoint.operation, "seal"),
                session=checkpoint.fence.session,
                generation=checkpoint.fence.generation,
                deadline_at=request.deadline_at,
                checkpoint=checkpoint.checkpoint,
                artifact=artifact.model_dump(exclude={"size_bytes"}),
                semantics=checkpoint.semantics.value,
                source_origin_instance=first.participant.origin_instance,
                cut_receipt=first.receipt,
                pending_operations=0,
                unresolved_operations=0,
            ),
        )
        if not self._sealed(checkpoint, artifact, result):
            raise CheckpointFailure("session_checkpoint_lineage")
        self._captures.pop(checkpoint.operation, None)

    def _sealed(self, checkpoint, artifact, result):
        first = checkpoint.participants[0].cut
        return (
            result.get("id") == str(checkpoint.checkpoint)
            and result.get("connector") == str(checkpoint.fence.connector)
            and result.get("session") == str(checkpoint.fence.session)
            and result.get("source_generation") == checkpoint.fence.source_generation
            and result.get("service_identity") == self.service_identity
            and result.get("source_origin_instance") == str(first.participant.origin_instance)
            and result.get("cut_receipt") == str(first.receipt)
            and result.get("semantics") == checkpoint.semantics.value
            and result.get("artifact") == artifact.model_dump(mode="json", exclude={"size_bytes"})
        )

    async def lookup_seal(self, checkpoint, artifact):
        result = await self._lookup(checkpoint.fence.connector, uuid5(checkpoint.operation, "seal"))
        confirmed = result is not None and self._sealed(checkpoint, artifact, result)
        if confirmed:
            self._captures.pop(checkpoint.operation, None)
        return confirmed

    async def restart(self, request: RestoreRequest, manifest: ApplicationCheckpoint):
        from synth_ai.sdk.tunnels.session_operations import Restart

        result = await self._execute(
            request.connector,
            Restart(
                action="restart",
                operation=request.operation,
                session=request.session,
                generation=request.expected_generation,
                deadline_at=request.deadline_at,
                checkpoint=manifest.checkpoint,
                checkpoint_sha256=request.artifact.sha256,
            ),
        )
        reference = result.get("checkpoint", {})
        if not self._sealed(manifest, request.artifact, reference):
            raise CheckpointFailure("session_restart_lineage")
        return SessionFence(
            connector=request.connector,
            session=request.session,
            source_generation=request.expected_generation,
            generation=result["generation"],
            permit_barrier_ms=0,
        )

    async def lookup_restart(self, request: RestoreRequest, manifest: ApplicationCheckpoint):
        result = await self._lookup(request.connector, request.operation)
        if result is None or not self._sealed(
            manifest, request.artifact, result.get("checkpoint", {})
        ):
            return None
        observed = await self._gateway(request.connector, "session")
        restart = observed["lifecycle"]["restart"]
        if (
            observed["session"] != str(request.session)
            or restart is None
            or restart["operation"] != str(request.operation)
            or restart["source_generation"] != request.expected_generation
            or restart["generation"] != result["generation"]
            or restart["reference"] != result["checkpoint"]
        ):
            raise CheckpointFailure("session_restart_lineage")
        # The restart receipt is immutable. Actor wake advances the live fence
        # for completion, never the generation of the already-settled restart.
        generation = restart.get("restore_generation", restart["generation"])
        if observed["lifecycle"]["phase"] == "restoring":
            if generation != observed["generation"]:
                raise CheckpointFailure("session_restart_lineage")
        elif observed["lifecycle"]["phase"] == "open" and restart.get("completion"):
            generation = restart["completion"]["generation"]
            if generation > observed["generation"]:
                raise CheckpointFailure("session_restart_lineage")
        else:
            raise CheckpointFailure("session_restart_lineage")
        return SessionFence(
            connector=request.connector,
            session=request.session,
            source_generation=request.expected_generation,
            generation=generation,
            permit_barrier_ms=0,
        )

    async def complete_restore(self, request: RestoreRequest, result: RestoreResult):
        from synth_ai.sdk.tunnels.session_operations import CompleteRestart

        first = result.receipts[0]
        operation = uuid5(request.operation, "complete:" + str(result.fence.generation))
        # Origin execution keeps its original deadline. Reconciliation only
        # settles independently proven durable receipts under the current fence.
        # A new completion command gets a bounded control budget; retries reuse
        # the canonical deadline, including after a lost registration reply.
        from synth_ai.errors import NotFoundError

        try:
            previous = await self.catalog.session_operation(request.connector, operation)
        except NotFoundError as error:
            if error.status != 404 or error.code != "unknown_session_operation":
                raise
            deadline = datetime.now(timezone.utc) + timedelta(seconds=60)
        else:
            deadline = datetime.fromisoformat(previous["deadline_at"].replace("Z", "+00:00"))
        saved = await self._execute(
            request.connector,
            CompleteRestart(
                action="restart-complete",
                operation=operation,
                restart_operation=request.operation,
                session=request.session,
                generation=result.fence.generation,
                deadline_at=deadline,
                checkpoint=result.checkpoint,
                checkpoint_sha256=result.checkpoint_sha256,
                service_identity=self.service_identity,
                restored_origin_instance=first.restored_origin_instance,
                receipt=operation,
            ),
        )
        if saved != {"generation": result.fence.generation, "phase": "open"}:
            raise CheckpointFailure("session_restore_lineage")

    async def lookup_completion(self, request: RestoreRequest, result: RestoreResult):
        operation = uuid5(request.operation, "complete:" + str(result.fence.generation))
        saved = await self._lookup(request.connector, operation)
        return saved == {"generation": result.fence.generation, "phase": "open"}
