#!/usr/bin/env python3
"""Emit redacted, standard receipts for Containers external acceptance gates.

This runner intentionally distinguishes deterministic protocol evidence from live
provider evidence. A green in-process drill never upgrades a credential-, paid-,
or Docker-gated acceptance item to PASS.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


SECRET = re.compile(r"(?:token|secret|password|authorization|cookie|api[_-]?key)", re.I)
STANDARD = (
    "requested-stream.json",
    "bound-stream.json",
    "event-kind-counts.json",
    "run-manifest.json",
    "cost-reconciliation.json",
    "trace-v5.json",
    "cua-findings.json",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def clean(value: Any, key: str = "") -> Any:
    if SECRET.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): clean(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
        return re.sub(
            r"(?i)([?&](?:token|secret|api[_-]?key)=)[^&#\s]+",
            r"\1[REDACTED]",
            value,
        )
    return value


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_tests(node_ids: list[str]) -> dict[str, Any]:
    command = [".venv/bin/pytest", "-q", *node_ids]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "exitCode": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
        "passed": completed.returncode == 0,
    }


def bundle(
    root: Path,
    acceptance_id: str,
    *,
    external_status: str,
    deterministic: dict[str, Any],
    blockers: list[dict[str, str]],
    requested: dict[str, Any],
    observations: dict[str, Any] | None = None,
) -> None:
    destination = root / acceptance_id.lower().replace("/", "-")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "screenshots").mkdir(exist_ok=True)
    write(destination / "requested-stream.json", requested)
    write(destination / "bound-stream.json", {"state": "not_bound", "reason": "external_gate_not_run"})
    write(destination / "event-kind-counts.json", {"state": "not_emitted"})
    write(
        destination / "run-manifest.json",
        {"deterministicProtocolEvidence": deterministic, "externalObservations": observations or {}},
    )
    write(destination / "cost-reconciliation.json", {"reportedCostUsd": None, "state": "not_emitted"})
    write(destination / "trace-v5.json", {"state": "not_emitted"})
    write(destination / "cua-findings.json", {"state": "not_applicable", "findings": []})
    (destination / "cursor-transcript.jsonl").write_text("", encoding="utf-8")
    write(
        destination / "receipt.json",
        {
            "schemaVersion": "synth.modern-stack-receipt.v1",
            "acceptanceId": acceptance_id,
            "status": external_status,
            "deterministicProtocolStatus": "PASS" if deterministic["passed"] else "FAIL",
            "blockers": blockers,
            "finishedAt": now(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    a2 = run_tests([
        "tests/test_harbor_docker.py::test_harbor_docker_distinct_executions_read_verifier_reward_txt",
        "tests/test_dock_eval_extension.py::test_dock_extension_uses_pinned_harbor_runtime_and_public_stream",
    ])
    bundle(
        args.root,
        "A2",
        external_status="BLOCKED",
        deterministic=a2,
        blockers=[{"code": "docker_container_start_timeout", "detail": "daemon metadata responds but a digest-pinned alpine launch exceeded 12 seconds; shared daemon was not restarted"}],
        requested={"family": "harbor", "bundle": "public GameBench", "containerRuntime": "docker"},
        observations={"dockerMetadataResponsive": True, "containerStartResponsive": False},
    )

    a8 = run_tests([
        "tests/test_digbench_live.py::test_live_relay_maps_seven_kinds_and_hides_token",
        "tests/test_digbench_live.py::test_live_agentic_mcp_spans_share_the_eval_stream",
        "tests/test_digbench_live.py::test_live_token_absent_from_trace_seal",
    ])
    token_present = bool((os.environ.get("DIGBENCH_API_TOKEN") or "").strip())
    bundle(
        args.root,
        "A8",
        external_status="BLOCKED" if not token_present else "READY",
        deterministic=a8,
        blockers=[] if token_present else [{"code": "credential_missing", "detail": "DIGBENCH_API_TOKEN is not present; public basic and agentic harnesses were not called"}],
        requested={"family": "digbench", "harnesses": ["react_legal_actions", "agentic_codex"]},
        observations={"credentialPresent": token_present},
    )

    matrices = {
        "A11": [
            "tests/test_completed_rollout_recovery.py",
            "tests/test_platform_event_journal.py::test_poll_small_pages_reconstruct_exact_evidence_without_advancing_on_control",
        ],
        "A12": [
            "tests/test_platform_leftovers.py::test_prepare_and_start_retries_replay_one_rollout_identity",
            "tests/test_platform_leftovers.py::test_start_retry_refuses_changed_identity",
            "tests/test_container_compat_floor.py::test_reference_prepare_ack_precedes_first_semantic_event",
        ],
        "O1": ["tests/test_after_bind_surface.py::test_craftax_eleventh_lease_is_typed_429"],
        "O2": [
            "tests/test_trace_v5_capture_security_regressions.py::test_proxy_and_collector_stop_before_start_return_and_are_idempotent",
        ],
        "O3": [
            "tests/test_platform_event_journal.py::test_poll_small_pages_reconstruct_exact_evidence_without_advancing_on_control",
            "tests/conformance/trace_stream/test_gate_d.py",
        ],
        "O4": [
            "tests/test_trace_v5_capture_security_regressions.py::test_non_loopback_collector_health_requires_registered_capture_auth",
            "tests/test_trace_v5_capture_security_regressions.py::test_resume_rotates_child_capability_and_concurrent_finish_is_once",
            "tests/test_trace_v5_hardening.py::test_payload_redaction_covers_compound_credential_keys",
            "tests/test_digbench_live.py::test_live_token_absent_from_trace_seal",
        ],
    }
    external_blockers = {
        "A11": "live socket/container kill requires a launch-capable isolated container runtime",
        "A12": "exactly-one paid provider invocation requires provider credentials",
        "O1": "provider-cost ceiling reconciliation requires paid provider usage",
        "O2": "mid-provider-call cancellation requires a paid live call",
        "O3": "live producer/browser backpressure is measured by the Workshop browser lane",
        "O4": "expired provider auth and cross-workspace live binding require isolated external services",
    }
    for acceptance_id, node_ids in matrices.items():
        evidence = run_tests(node_ids)
        bundle(
            args.root,
            acceptance_id,
            external_status="BLOCKED",
            deterministic=evidence,
            blockers=[{"code": "external_live_drill_remaining", "detail": external_blockers[acceptance_id]}],
            requested={"acceptanceId": acceptance_id, "mode": "destructive-live"},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
