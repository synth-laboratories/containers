"""Version-qualified native Harbor REST/MCP with managed execution authority.

Harbor owns jobs, retries and trial semantics. Rhodes owns admission and provider
effects. The Synth facade delegates to this same Service, never a second runner.
"""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from uuid import UUID, uuid4

from harbor.server.app import create_app
from harbor.server.database import SQLiteDatabase
from harbor.server.executor import TrialExecutor
from harbor.server.models import APIError, SubmitRequest, SubmitResponse
from harbor.server.service import Service
from harbor.server.storage import LocalStorage

from .harbor_serve_authority import HarborAuthority, TrialAuthority, trial_authority
from .package_models import content_digest
from .rollout_journal import RolloutJournal, RolloutJournalError
from .service_contracts import ServiceAttempt, ServiceDeclaration
from .service_runtime import DurableServiceRuntime, OperationObservation, create_service_app

HARBOR_VERSION = "0.26.0"
ENVIRONMENT = "synth_containers.harbor_serve_environment:RhodesEnvironment"


def harbor_services(
    *, concurrency: int, port: int = 8390, inference: dict | None = None
) -> ServiceDeclaration:
    return ServiceDeclaration.model_validate(
        {
            "interfaces": [
                {
                    "name": "harbor",
                    "protocol": "harbor.http",
                    "version": HARBOR_VERSION,
                    "port": port,
                    "schema_path": "/openapi.json",
                },
                {
                    "name": "harbor-mcp",
                    "protocol": "harbor.mcp",
                    "version": HARBOR_VERSION,
                    "port": port,
                    "base_path": "/mcp",
                    "transport": "mcp.streamable-http",
                },
                {
                    "name": "synth",
                    "protocol": "synth.service-operations",
                    "version": "1",
                    "port": port,
                    "base_path": "/synth",
                },
            ],
            "state": [{"name": "harbor-data", "mount_path": "/var/lib/harbor"}],
            "execution": {
                "protocol": "synth.service-operations",
                "version": "1",
                "interface": "synth",
                "admission_unit": "trial",
                "concurrency": concurrency,
                "inference": inference,
                "pause": "queued",
                "resume": "queued",
            },
        }
    )


class ManagedTrialExecutor(TrialExecutor):
    def __init__(self, authority: HarborAuthority) -> None:
        super().__init__()
        self.authority = authority

    async def run(self, rollout_id, config, on_result):
        binding = trial_authority.set(TrialAuthority(self.authority, str(rollout_id)))
        result = None
        try:
            admitted = await self.authority.call("start", {"native_id": str(rollout_id)})
            if config.agent.model_name:
                from harbor.server.resolution import relay_agent_config

                inference = admitted["inference"]
                config = config.model_copy(deep=True)
                config.agent = relay_agent_config(
                    config.agent, inference["base_url"], inference["token"]
                )
            result = await super().run(rollout_id, config, on_result)
            return result
        finally:
            trial_authority.reset(binding)
            # Rhodes verifies resource absence independently of Harbor's status.
            await self.authority.call(
                "finish",
                {
                    "native_id": str(rollout_id),
                    "result": result.model_dump(mode="json") if result else None,
                },
            )


class ManagedHarborService(Service):
    def __init__(
        self,
        *,
        authority: HarborAuthority,
        journal: RolloutJournal,
        task_roots: tuple[Path, ...],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.authority = authority
        self.journal = journal
        self.task_roots = tuple(path.resolve() for path in task_roots)
        self.submit_lock = asyncio.Lock()

    def _managed_request(self, request: SubmitRequest) -> SubmitRequest:
        config = request.config
        # Packaged tasks are the deployed workload. Server-side filesystem paths
        # and arbitrary Python imports are never client authority.
        if config.datasets or not config.tasks or config.user_agent is not None:
            raise APIError("ManagedProfile", "Use deployed tasks and built-in agents.", 422)
        if config.n_attempts * len(config.tasks) * len(config.agents) > 4096:
            raise APIError("ManagedProfile", "Job exceeds the bounded admission queue.", 422)
        for task in config.tasks:
            if task.path is None or not any(
                task.path.resolve().is_relative_to(root) for root in self.task_roots
            ):
                raise APIError("ManagedProfile", "Task is outside the deployed task roots.", 422)
            if task.git_url is not None:
                raise APIError(
                    "ManagedProfile", "Remote task imports require a new deployment.", 422
                )
        if any(
            agent.import_path
            or agent.env
            or agent.base_url
            or agent.model_api_key_name
            or agent.skills
            or agent.kwargs
            or agent.mcp_servers
            or agent.load_trajectory
            or agent.resume_trajectory
            or agent.extra_allowed_hosts
            for agent in config.agents
        ):
            raise APIError(
                "ManagedProfile",
                "Agent imports and credentials require deployment configuration.",
                422,
            )
        if config.verifier.import_path or config.verifier.env or config.verifier.kwargs:
            raise APIError(
                "ManagedProfile", "Verifier configuration is owned by the deployed task.", 422
            )
        environment = config.environment
        if (
            environment.import_path not in (None, ENVIRONMENT)
            or environment.force_build
            or environment.mounts
            or environment.extra_docker_compose
            or environment.kwargs
            or environment.env
            or not environment.delete
            or environment.stream
        ):
            raise APIError("ManagedProfile", "Environment configuration is managed by Rhodes.", 422)
        if config.n_concurrent_trials > self.max_concurrent or config.retry.max_retries > 3:
            raise APIError(
                "ManagedProfile", "Requested concurrency or retries exceed deployment policy.", 422
            )
        if config.extra_instruction_paths or config.skip_credential_checks:
            raise APIError(
                "ManagedProfile",
                "Server paths and credential-check overrides are not accepted.",
                422,
            )
        value = request.model_copy(deep=True)
        value.config.skip_credential_checks = (
            True  # Rhodes issues a bounded gateway session at start.
        )
        value.config.environment.type = None
        value.config.environment.import_path = ENVIRONMENT
        return value

    async def submit(self, request, *, idempotency_key=None, owner_id="local"):
        managed = self._managed_request(request)
        if request.dry_run:
            return await super().submit(managed, owner_id=owner_id)
        key = idempotency_key or str(uuid4())
        operation = content_digest([owner_id, key]).removeprefix("sha256:")
        digest = content_digest(request.model_dump(mode="json", exclude_unset=True)).removeprefix(
            "sha256:"
        )
        async with self.submit_lock:
            try:
                _record, created = self.journal.admit(
                    operation,
                    digest,
                    {
                        "rollout_id": operation,
                        "status": "running",
                    },
                )
            except RolloutJournalError as error:
                raise APIError("Conflict", str(error), 409) from error
            if not created:
                # Native idempotency lookup returns the original job ID, including
                # a crash between Harbor's commit and our receipt publication.
                self.database.check_idempotency_key("submit_job", key, owner_id=owner_id)
                raise APIError("Conflict", "Submission outcome requires reconciliation.", 409)
            try:
                response = await super().submit(managed, idempotency_key=key, owner_id=owner_id)
            except BaseException:
                self.journal.write(operation, {"rollout_id": operation, "status": "in_doubt"})
                raise
            if not isinstance(response, SubmitResponse):
                raise APIError("ManagedProfile", "Unexpected dry-run result after submission.", 500)
            self.journal.write(
                operation,
                {"rollout_id": operation, "status": "completed", "job_id": str(response.job_id)},
            )
            return response

    async def _prepare_rollout(self, job_id, config, lock, **kwargs):
        from harbor.models.task.task import Task

        task = Task(config.task.path)
        environments = [task.config.environment]
        if task.config.verifier.environment is not None:
            environments.append(task.config.verifier.environment)
        timeout = math.ceil(
            (task.config.agent.timeout_sec or 600)
            * (config.agent_timeout_multiplier or config.timeout_multiplier)
            + task.config.verifier.timeout_sec
            * (config.verifier_timeout_multiplier or config.timeout_multiplier)
            + 300
        )
        rollout = await super()._prepare_rollout(job_id, config, lock, **kwargs)
        inference = None
        if config.agent.model_name:
            from harbor.agents.agent_auth_resolution import resolve_agent_model_connection

            connection = resolve_agent_model_connection(config.agent)
            wire = {
                "openai_responses": "responses",
                "openai_chat_completions": "chat_completions",
            }.get(connection.model_api_format.value if connection.model_api_format else "")
            if wire is None or not connection.can_configure_base_url:
                raise APIError("ManagedProfile", "Agent requires a qualified gateway wire.", 422)
            inference = {"model": config.agent.model_name, "wire": wire}
        attempt = ServiceAttempt(
            native_id=str(rollout.id),
            job_id=str(job_id),
            inference=inference,
            input_digest=content_digest(lock.model_dump(mode="json")),
            retry_of=str(rollout.retry_of) if rollout.retry_of else None,
            images=tuple(
                dict.fromkeys(
                    env.docker_image for env in environments if env.docker_image is not None
                )
            ),
            timeout_seconds=timeout,
            limits={
                "cpu_cores": math.ceil(
                    config.environment.override_cpus or max(env.cpus or 1 for env in environments)
                ),
                "memory_mb": config.environment.override_memory_mb
                or max(env.memory_mb or 2048 for env in environments),
                "storage_mb": config.environment.override_storage_mb
                or max(env.storage_mb or 8192 for env in environments),
            },
        )
        await self.authority.call("prepare", attempt.model_dump(mode="json"))
        return rollout

    async def cancel_job(self, job_id, *, force, owner_id="local"):
        result = await super().cancel_job(job_id, force=force, owner_id=owner_id)
        for rollout in self.database.job_rollouts(job_id):
            if rollout.state == "cancelled" and rollout.id not in self.tasks:
                await self.authority.call("finish", {"native_id": str(rollout.id), "result": None})
        return result

    async def cancel_rollout(self, rollout_id, *, force, owner_id="local"):
        result = await super().cancel_rollout(rollout_id, force=force, owner_id=owner_id)
        if result.state == "cancelled" and rollout_id not in self.tasks:
            await self.authority.call("finish", {"native_id": str(rollout_id), "result": None})
        return result

    async def _recover(self):
        for rollout in self.database.unfinished():
            if rollout.state in {"running", "cancelling"}:
                await self.authority.call("recover", {"native_id": str(rollout.id)})
        await super()._recover()


class HarborOperationDriver:
    def __init__(self, service: ManagedHarborService) -> None:
        self.service = service

    async def submit(self, operation_id, payload):
        try:
            response = await self.service.submit(
                SubmitRequest.model_validate(payload), idempotency_key=operation_id
            )
            native_id = str(response.job_id)
        except APIError as error:
            native_id = error.details.get("job_id")
            if error.status_code != 409 or not native_id:
                raise
        return await self.observe(native_id)

    async def recover(self, operation_id):
        try:
            self.service.database.check_idempotency_key(
                "submit_job", operation_id, owner_id="local"
            )
        except APIError as error:
            native_id = error.details.get("job_id")
            if error.status_code != 409 or not native_id:
                raise
            return await self.observe(native_id)
        return None

    async def observe(self, native_id):
        job = await self.service.get_job(native_id)
        state = job.state
        cleanup = "pending"
        if state in {"finished", "cancelled"}:
            trials = self.service.database.job_rollouts(UUID(native_id))
            observations = [
                await self.service.authority.call("observe", {"native_id": str(trial.id)})
                for trial in trials
            ]
            cleanup = (
                "confirmed"
                if all(item.get("cleanup") == "confirmed" for item in observations)
                else "unverified"
            )
            state = (
                "cancelled" if state == "cancelled" else "failed" if job.n_errors else "completed"
            )
        return OperationObservation(native_id, state, job.model_dump(mode="json"), cleanup)

    async def cancel(self, native_id):
        await self.service.cancel_job(UUID(native_id), force=True)


def create_managed_harbor_app(
    *,
    data_dir: Path,
    api_key: str,
    authority: HarborAuthority,
    task_roots: tuple[Path, ...],
    concurrency: int = 4,
    port: int = 8390,
    inference: dict | None = None,
):
    if version("harbor") != HARBOR_VERSION:
        raise ValueError("harbor_serve_version_unqualified")
    if len(api_key) < 24 or not task_roots:
        raise ValueError("harbor_serve_configuration_invalid")
    data_dir.mkdir(parents=True, exist_ok=True)
    journal = RolloutJournal(data_dir / "submissions", "harbor-submit")
    try:
        service = ManagedHarborService(
            authority=authority,
            journal=journal,
            task_roots=task_roots,
            database=SQLiteDatabase(data_dir),
            storage=LocalStorage(data_dir / "files"),
            work_dir=data_dir / "work",
            max_concurrent=concurrency,
            executor=ManagedTrialExecutor(authority),
        )
        app = create_app(data_dir, service=service, api_key=api_key)
        from .service_prefix import ServicePrefixMiddleware

        app.add_middleware(ServicePrefixMiddleware, api_key=api_key)
        runtime = DurableServiceRuntime(
            runtime_id="harbor",
            declaration=harbor_services(concurrency=concurrency, port=port, inference=inference),
            directory=data_dir / "operations",
            driver=HarborOperationDriver(service),
        )
    except BaseException:
        journal.close()
        raise
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with original_lifespan(application):
                yield
        finally:
            await runtime.close()
            journal.close()

    app.router.lifespan_context = lifespan
    app.mount("/synth", create_service_app(runtime, api_key=api_key))
    app.state.managed_service = service
    app.state.synth_runtime = runtime

    @app.get("/healthz", include_in_schema=False)
    async def health():
        return {"ready": not service.closing, "service": "harbor"}

    return app
