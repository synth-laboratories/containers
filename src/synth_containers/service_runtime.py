"""Durable asynchronous service operations using the existing origin journal.

The driver owns native execution; this facade owns identity and observation, not
a second execution scheduler. Recovery never calls submit. See services-v1.md.
"""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from .metadata import RuntimeReadiness, compose_metadata_payload
from .package_models import content_digest
from .rollout_journal import RolloutJournal, RolloutJournalError
from .service_contracts import Name, ServiceDeclaration, ServiceModel


class OperationRequest(ServiceModel):
    operation_id: Name
    input: dict[str, Any]
    expected_origin_instance: UUID | None = None


@dataclass(frozen=True)
class OperationObservation:
    native_id: str
    state: Literal[
        "queued", "running", "paused", "cancelling", "completed", "failed", "cancelled", "in_doubt"
    ]
    result: dict[str, Any] | None = None
    cleanup: Literal["pending", "confirmed", "unverified"] = "unverified"


class OperationDriver(Protocol):
    async def submit(self, operation_id: str, payload: dict[str, Any]) -> OperationObservation: ...
    async def recover(self, operation_id: str) -> OperationObservation | None: ...
    async def observe(self, native_id: str) -> OperationObservation: ...
    async def cancel(self, native_id: str) -> None: ...


class DurableServiceRuntime:
    def __init__(
        self,
        *,
        runtime_id: str,
        declaration: ServiceDeclaration,
        directory: Path,
        driver: OperationDriver,
    ) -> None:
        self.runtime_id = runtime_id
        self.declaration = ServiceDeclaration.model_validate(declaration)
        self.driver = driver
        self.journal = RolloutJournal(directory, runtime_id)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._locks = tuple(asyncio.Lock() for _ in range(64))
        self.closing = False

    @property
    def origin_instance(self) -> str:
        return self.journal.origin_instance

    def metadata(self) -> dict[str, Any]:
        return compose_metadata_payload(
            base={
                "runtime_id": self.runtime_id,
                "origin_instance": self.origin_instance,
                "services": self.declaration.model_dump(mode="json"),
                "service_declaration_digest": content_digest(
                    self.declaration.model_dump(mode="json")
                ),
                "operation_contract": "synth.service-operations.v1",
            },
            protocol="synth.service-operations.v1",
            live_frames="unsupported",
            readiness=RuntimeReadiness(
                policy_ready=not self.closing, program_ready=not self.closing
            ),
        )

    def _record(self, operation_id: str) -> dict[str, Any]:
        value = self.journal.get(operation_id)
        if value is None:
            raise KeyError(operation_id)
        return value

    def _save(self, operation_id: str, observation: OperationObservation) -> dict[str, Any]:
        record = self._record(operation_id)
        if record.get("native_id") not in (None, observation.native_id):
            raise RolloutJournalError("service_native_identity_conflict")
        terminal = observation.state in {"completed", "failed", "cancelled"}
        # Provider terminal status cannot release custody without cleanup proof.
        state = "in_doubt" if terminal and observation.cleanup != "confirmed" else observation.state
        record.update(
            status=state,
            native_id=observation.native_id,
            result=observation.result,
            native_state=observation.state,
            resource_cleanup=observation.cleanup,
        )
        self.journal.write(operation_id, record)
        return record

    async def _submit(self, request: OperationRequest) -> None:
        try:
            observation = await self.driver.submit(request.operation_id, request.input)
            self._save(request.operation_id, observation)
        except BaseException:
            # The native service may have accepted even if the response was lost.
            record = self._record(request.operation_id)
            record.update(status="in_doubt", resource_cleanup="unverified")
            self.journal.write(request.operation_id, record)
            raise

    async def submit(self, request: OperationRequest) -> dict[str, Any]:
        if self.closing:
            raise RolloutJournalError("service_closing")
        if (
            request.expected_origin_instance is not None
            and str(request.expected_origin_instance) != self.origin_instance
        ):
            raise RolloutJournalError("service_origin_instance_conflict")
        digest = content_digest(request.input).removeprefix("sha256:")
        record, created = self.journal.admit(
            request.operation_id,
            digest,
            {
                "rollout_id": request.operation_id,
                "operation_id": request.operation_id,
                "status": "running",
                "native_id": None,
                "resource_cleanup": "pending",
            },
        )
        if created:
            task = asyncio.create_task(self._submit(request))
            self.tasks[request.operation_id] = task

            def settled(task: asyncio.Task[None]) -> None:
                self.tasks.pop(request.operation_id, None)
                if not task.cancelled():
                    task.exception()  # The durable receipt retains uncertain outcome.

            task.add_done_callback(settled)
        return {"origin_instance": self.origin_instance, "input_sha256": digest, "record": record}

    async def read(self, operation_id: str, *, cancel: bool = False) -> dict[str, Any]:
        lock = self._locks[hash(operation_id) % len(self._locks)]
        async with lock:
            record = self._record(operation_id)
            if record["status"] not in {"completed", "failed", "cancelled"}:
                if operation_id not in self.tasks:
                    native_id = record.get("native_id")
                    observed = (
                        await self.driver.observe(native_id)
                        if native_id
                        else await self.driver.recover(operation_id)
                    )
                    if observed is not None:
                        record = self._save(operation_id, observed)
                if (cancel or record.get("cancel_requested")) and record["status"] not in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    record.update(cancel_requested=True, status="cancelling")
                    self.journal.write(operation_id, record)
                    if record.get("native_id"):
                        await self.driver.cancel(record["native_id"])
            receipt = self.journal.receipt(operation_id)
            if receipt is None:
                raise RolloutJournalError("service_operation_receipt_missing")
            return receipt

    async def close(self) -> None:
        self.closing = True
        tasks = tuple(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.journal.close()


def create_service_app(runtime: DurableServiceRuntime, *, api_key: str) -> FastAPI:
    """An origin bearer is mandatory; public authorization belongs to the gateway."""
    if not api_key or len(api_key) < 24:
        raise ValueError("service_origin_key_too_short")

    async def authorize(authorization: str = Header(default="")) -> None:
        if not hmac.compare_digest(authorization.encode(), f"Bearer {api_key}".encode()):
            raise HTTPException(401, "service_unauthorized")

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title="Synth service operations", lifespan=lifespan, dependencies=[Depends(authorize)]
    )
    from .service_prefix import ServicePrefixMiddleware

    app.add_middleware(ServicePrefixMiddleware, api_key=api_key)

    @app.exception_handler(RolloutJournalError)
    async def journal_error(_request, error):
        from fastapi.responses import JSONResponse

        return JSONResponse({"code": str(error)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(_request, _error):
        from fastapi.responses import JSONResponse

        return JSONResponse({"code": "service_operation_not_found"}, status_code=404)

    @app.get("/.well-known/synth-service")
    async def metadata():
        return runtime.metadata()

    @app.post("/operations", status_code=202)
    async def submit(request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 1024 * 1024:
                raise HTTPException(413, "service_input_too_large")
        from pydantic import ValidationError

        try:
            body = OperationRequest.model_validate_json(raw)
        except ValidationError:
            raise HTTPException(422, "service_input_invalid") from None
        return await runtime.submit(body)

    def verify(operation_id: str, origin_instance: str, input_sha256: str):
        receipt = runtime.journal.receipt(operation_id)
        if receipt is None:
            raise KeyError(operation_id)
        if origin_instance != runtime.origin_instance or input_sha256 != receipt["input_sha256"]:
            raise HTTPException(409, "service_operation_binding_conflict")

    @app.get("/operations/{operation_id}")
    async def read(operation_id: str, origin_instance: str, input_sha256: str):
        verify(operation_id, origin_instance, input_sha256)
        return await runtime.read(operation_id)

    @app.post("/operations/{operation_id}/cancel")
    async def cancel(operation_id: str, origin_instance: str, input_sha256: str):
        verify(operation_id, origin_instance, input_sha256)
        return await runtime.read(operation_id, cancel=True)

    return app
