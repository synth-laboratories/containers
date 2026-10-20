"""Execute one staged native Harbor task under durable lifecycle supervision."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import re
import tomllib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .bounded_process import run_bounded_process
from .harbor_environment import _tree_digest
from .lifecycle_limits import DurableRolloutSupervisor, LifecycleLimits, RolloutStopped
from .native_limit_admission import require_native_limit_capabilities
from .tracing.capture.redaction import assert_no_secrets, redact_payload


def validate_staged_harbor(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != "synth.harbor-native-task-stage.v1":
        raise ValueError("Native execution requires a registrar stage receipt")
    require_native_limit_capabilities(receipt.get("required_limit_capabilities", ()))
    task = Path(receipt["task_path"])
    if _tree_digest(task) != receipt.get("staged_package_digest"):
        raise ValueError("Staged native task changed before launch")
    manifest = tomllib.loads((task / "task.toml").read_text())
    phases = receipt.get("phase_limits") or {}
    resolved = phases.get("resolved_seconds") or {}
    for phase in ("agent", "verifier"):
        seconds = resolved.get(phase)
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
            or manifest.get(phase, {}).get("timeout_sec") != seconds
        ):
            raise ValueError("Native execution requires explicit staged inner phase ceilings")
    if manifest.get("environment", {}).get("docker_image") != receipt.get("image"):
        raise ValueError("Staged native image binding changed")
    flags = receipt.get("native_environment_flags")
    expected = {
        "docker": "synth_containers.harbor_docker:ObservedDockerEnvironment",
        "daytona": "synth_containers.harbor_daytona:BoundedDaytonaEnvironment",
    }
    if not isinstance(flags, list) or flags[:2] != ["--env", expected.get(receipt.get("provider"))]:
        raise ValueError("Native execution requires owned provider custody adapters")


async def execute_staged_harbor(
    output: Path,
    *,
    run_id: str,
    limits: LifecycleLimits,
    setup: Callable[[], Awaitable[Mapping[str, Any]]],
    agent: str,
    model: str | None,
    jobs_dir: Path,
    job_name: str,
    max_output_bytes: int,
    decode: Callable[[], Awaitable[Any]],
    publish: Callable[[dict], Awaitable[Any]],
    cleanup: Callable[[], Awaitable[dict]],
    env: Mapping[str, str] | None = None,
    redact: Sequence[str] = (),
    executable: Sequence[str] = ("harbor",),
    extra_cli_args: Sequence[str] = (),
    should_cancel: Callable[[], bool | Awaitable[bool]] | None = None,
    required_limit_capabilities: object = (),
) -> dict:
    """Setup produces exact staged custody; Harbor owns its inner phase timers.

    The outer work timer includes CLI setup, agent, verifier and CLI collection.
    The coordinator's verifier callback decodes the actual verifier receipt; it
    does not execute a second grader. Publication and cleanup have separate bounds.
    Execution/scientific result, publication and cleanup remain separate facts.
    """
    require_native_limit_capabilities(required_limit_capabilities)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_name):
        raise ValueError("Native Harbor job name is not a safe identifier")
    if len(extra_cli_args) > 256 or any(
        not isinstance(value, str) or len(value) > 16384 or "\x00" in value
        for value in extra_cli_args
    ):
        raise ValueError("Native Harbor extra arguments exceed admission")
    position = 0
    while position < len(extra_cli_args):
        option = extra_cli_args[position]
        position += 1
        if option == "--yes":
            continue
        if option not in {"--agent-kwarg", "--agent-env", "--verifier-env"} or position == len(
            extra_cli_args
        ):
            raise ValueError("Native Harbor extra argument is not allowlisted")
        if not extra_cli_args[position] or extra_cli_args[position].startswith("--"):
            raise ValueError("Native Harbor extra argument value is missing")
        position += 1
    supervisor = DurableRolloutSupervisor(output, run_id, limits)
    outcome: dict[str, Any] = {
        "run_id": run_id,
        "execution_returncode": None,
        "execution_error": None,
        "decoded_result": None,
        "publication": None,
        "cleanup": {"cleanup_status": "pending"},
    }
    try:
        try:
            if supervisor._phase_deadlines:
                raise RolloutStopped("Started native attempts require reconciliation, not replay")
            receipt = dict(await supervisor.run_phase("setup", setup))
            validate_staged_harbor(receipt)
            plan = {
                "schema_version": "synth.staged-harbor-execution.v1",
                "run_id": run_id,
                "source_package_digest": receipt["source_package_digest"],
                "staged_package_digest": receipt["staged_package_digest"],
                "provider": receipt["provider"],
                "inner_phase_limits": receipt["phase_limits"],
                "coordinator_phase_mapping": {
                    "work": "combined_harbor_cli",
                    "verifier": "decode_verifier_receipt",
                },
            }
            with (output / "execution-plan.json").open("x") as handle:
                json.dump(plan, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            argv = [
                *executable,
                "run",
                "--path",
                receipt["task_path"],
                *receipt["native_environment_flags"],
                "--agent",
                agent,
                *(["--model", model] if model is not None else []),
                *extra_cli_args,
                "--jobs-dir",
                str(jobs_dir),
                "--job-name",
                job_name,
                "--n-attempts",
                "1",
                "--n-concurrent",
                "1",
                "--max-retries",
                "0",
                "--timeout-multiplier",
                "1.0",
                "--environment-build-timeout-multiplier",
                "1.0",
            ]

            async def execute():
                validate_staged_harbor(receipt)

                async def cancellation_requested():
                    if should_cancel is None:
                        return False
                    value = should_cancel()
                    value = await value if inspect.isawaitable(value) else value
                    if type(value) is not bool:
                        raise ValueError("Native cancellation callback must return a boolean")
                    return value

                if await cancellation_requested():
                    raise RolloutStopped("Native execution cancelled before process creation")
                process = asyncio.create_task(
                    run_bounded_process(
                        argv,
                        output=output / "process.log",
                        max_output_bytes=max_output_bytes,
                        env=env,
                        cwd=output,
                        redact=redact,
                    )
                )
                try:
                    while not process.done():
                        if await cancellation_requested():
                            raise RolloutStopped("Native owner requested cancellation")
                        await asyncio.wait({process}, timeout=0.25)
                    return await process
                finally:
                    if not process.done():
                        process.cancel()
                    # Always observe the task outcome, including a process that
                    # exits concurrently with the owner's cancellation request.
                    await asyncio.gather(process, return_exceptions=True)

            outcome["execution_returncode"] = await supervisor.run_phase("work", execute)
            outcome["decoded_result"] = await supervisor.run_phase("verifier", decode)
            if outcome["execution_returncode"]:
                outcome["execution_error"] = "HarborProcessFailed"
        except BaseException as error:  # noqa: BLE001 - bounded cleanup follows cancellation
            outcome["execution_error"] = type(error).__name__
            supervisor.decide_stop(type(error).__name__)
        # Publish available evidence even after an interrupted execution. A sticky
        # stop forbids fresh work, not bounded preservation of already-owned data.
        try:
            publication_input, redaction = redact_payload(outcome)

            def scrub(value):
                if isinstance(value, str):
                    for secret in redact:
                        value = value.replace(secret, "[REDACTED]")
                    return value
                if isinstance(value, list):
                    return [scrub(item) for item in value]
                if isinstance(value, dict):
                    return {key: scrub(item) for key, item in value.items()}
                return value

            publication_input = scrub(publication_input)
            publication_input["redaction"] = redaction.to_dict()
            assert_no_secrets(publication_input, where="native lifecycle publication")
            outcome["publication"] = await supervisor.run_phase(
                "publication", lambda: publish(publication_input), preserve_evidence=True
            )
        except BaseException as error:  # noqa: BLE001 - publication must not suppress cleanup
            outcome["publication_error"] = type(error).__name__
        try:
            outcome["cleanup"] = await supervisor.stop(
                "execution_failed" if outcome["execution_error"] else "execution_complete", cleanup
            )
        except BaseException as error:  # noqa: BLE001 - retain an independent pending outcome
            outcome["cleanup"] = {"cleanup_status": "pending", "error_type": type(error).__name__}
        return outcome
    finally:
        supervisor.close()


async def cleanup_staged_harbor_jobs(
    jobs_root: Path,
    *,
    provider: str,
    output: Path,
    env: Mapping[str, str] | None = None,
    redact: Sequence[str] = (),
) -> dict:
    """Read exact custody, then invoke the existing age/owner-guarded reconciler.

    A cleanup CLI subprocess avoids leaving a blocking Docker SDK thread alive
    after coordinator cancellation. Provider creation/expiry guards remain in
    force; an early or ambiguous recovery remains pending.
    """
    import sys

    from .harbor_resource_receipts import read_harbor_resource_receipt

    if provider not in {"docker", "daytona"}:
        raise ValueError("Unknown native cleanup provider")
    claims = list(jobs_root.glob("*/resource-create-claim.json"))
    if len(claims) != 1:
        raise RolloutStopped("Native cleanup requires exactly one retained provider claim")
    trial = claims[0].parent
    receipt = read_harbor_resource_receipt(trial)
    if receipt["provider"] != provider:
        raise ValueError("Native cleanup provider differs from retained custody")
    if receipt["cleanup_status"] == "confirmed":
        return receipt
    returncode = await run_bounded_process(
        [sys.executable, "-m", "synth_containers", f"harbor-{provider}-reconcile", str(trial)],
        output=output,
        max_output_bytes=1024 * 1024,
        env=env,
        redact=redact,
    )
    if returncode:
        raise RolloutStopped("Native provider reconciliation remains pending")
    return read_harbor_resource_receipt(trial)
