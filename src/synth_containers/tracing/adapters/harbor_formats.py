"""Pinned current-Harbor output formats, typed refusals and field-mapping declarations.

The current Harbor runner (``harbor==0.22.0``, the version this package pins for
its native Harbor extra) writes one directory per job and one per trial:

- ``<job>/lock.json``   — job lock, ``schema_version`` 3, advertises ``harbor.version``;
- ``<job>/result.json`` — job result (stats, retries, eval aggregates);
- ``<trial>/lock.json`` — trial lock, ``schema_version`` 2;
- ``<trial>/result.json`` — Harbor ``TrialResult`` (no schema version of its own;
  its field set is pinned below and any undeclared field is refused);
- ``<trial>/agent/*`` — agent-owned output.  Agents that write ATIF write
  ``agent/trajectory.json``; the Resin shell agent writes
  ``agent/resinbench-agent.json`` (pinned by ``agent_info`` name/version).

Every observed source leaf must be covered by a declaration here.  A
declaration states what the qualified record does with the field: ``mapped``
(normalized), ``raw_only`` (kept byte-exact in the raw source, not normalized),
or ``lost`` (not representable).  A ``mapped`` field whose source value is
``null`` is reported as ``unknown`` rather than defaulted; a ``null`` container
whose children are declared (``exception_info``, ``verifier``) is ``absent``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

SUPPORTED_HARBOR_VERSIONS: frozenset[str] = frozenset({"0.22.0"})
SUPPORTED_JOB_LOCK_SCHEMA_VERSIONS: frozenset[int] = frozenset({3})
SUPPORTED_TRIAL_LOCK_SCHEMA_VERSIONS: frozenset[int] = frozenset({2})
# Only the advertised canonical spelling is accepted at the Harbor boundary:
# a trajectory without ``schema_version`` is unadvertised and refused.
SUPPORTED_ATIF_VERSIONS: frozenset[str] = frozenset({"ATIF-v1.5", "ATIF-v1.6", "ATIF-v1.7"})
RESIN_SHELL_AGENT_FORMAT = "resinbench.shell-agent.steps.v1"
# (agent_info.name, agent_info.version) -> pinned agent metadata format.
SUPPORTED_AGENT_METADATA: Mapping[tuple[str, str], str] = {
    ("resinbench-shell-agent", "0.1.0"): RESIN_SHELL_AGENT_FORMAT,
}
AGENT_METADATA_FILES: Mapping[str, str] = {
    RESIN_SHELL_AGENT_FORMAT: "agent/resinbench-agent.json",
}
ATIF_TRAJECTORY_FILE = "agent/trajectory.json"

HARBOR_TRIAL_SOURCE_FORMAT = "harbor.trial-dir@0.22"
HARBOR_JOB_SOURCE_FORMAT = "harbor.job-dir@0.22"
HARBOR_EVIDENCE_SCHEMA_VERSION = "synth.harbor-evidence.v1"


class HarborRefusalCode(StrEnum):
    MISSING_SOURCE = "harbor_missing_source"
    MALFORMED_SOURCE = "harbor_malformed_source"
    UNADVERTISED_VERSION = "harbor_unadvertised_version"
    UNSUPPORTED_VERSION = "harbor_unsupported_version"
    UNDECLARED_FIELD = "harbor_undeclared_field"
    TYPE_MISMATCH = "harbor_type_mismatch"
    INCONSISTENT_SOURCE = "harbor_inconsistent_source"
    MALFORMED_ATIF = "harbor_malformed_atif"


class HarborFormatRefusal(ValueError):
    """A Harbor output that is malformed, unadvertised or outside the pinned formats."""

    def __init__(self, code: HarborRefusalCode, *, source: str, path: str, detail: str) -> None:
        self.code = code
        self.source = source
        self.path = path
        self.detail = detail
        super().__init__(f"{code.value}: {source}:{path}: {detail}")

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "service": "synth_containers.tracing.adapters.harbor_formats",
            "stage": "harbor_qualification",
            "source": self.source,
            "path": self.path,
            "detail": self.detail,
        }


class FieldDisposition(StrEnum):
    MAPPED = "mapped"
    RAW_ONLY = "raw_only"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class FieldDeclaration:
    """How one source field (and, for subtrees, its descendants) is treated."""

    source_path: str
    disposition: FieldDisposition
    target: str | None = None
    subtree: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "disposition": self.disposition.value,
            "target": self.target,
            "subtree": self.subtree,
            "note": self.note,
        }


def _m(path: str, target: str, note: str = "") -> FieldDeclaration:
    return FieldDeclaration(path, FieldDisposition.MAPPED, target, note=note)


def _raw(path: str, note: str) -> FieldDeclaration:
    return FieldDeclaration(path, FieldDisposition.RAW_ONLY, subtree=True, note=note)


_PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")

TRIAL_RESULT_FIELDS: tuple[FieldDeclaration, ...] = (
    _m("id", "attempt.trial_id"),
    _m("task_name", "task.name"),
    _m("trial_name", "attempt.trial_name"),
    _raw("trial_uri", "producer host path; not portable"),
    _raw("task_id", "producer host path of the task definition"),
    _m("source", "task.source"),
    _m("task_checksum", "task.checksum"),
    _m("config.job_id", "attempt.job_id"),
    _m("config.agent.name", "agent.import_name"),
    _m("config.agent.model_name", "agent.requested_model"),
    _m("config.environment.type", "attempt.environment_type"),
    _raw("config", "trial configuration beyond the mapped keys is kept byte-exact"),
    _m("agent_info.name", "agent.name"),
    _m("agent_info.version", "agent.version"),
    _m("agent_info.model_info.name", "agent.model"),
    _m("agent_info.model_info.provider", "agent.provider"),
    _m("agent_result.n_input_tokens", "usage.input_tokens"),
    _m("agent_result.n_cache_tokens", "usage.cache_tokens"),
    _m("agent_result.n_output_tokens", "usage.output_tokens"),
    _m("agent_result.cost_usd", "usage.cost_usd"),
    _raw("agent_result.rollout_details", "token-level rollout details are not normalized"),
    FieldDeclaration(
        "agent_result.metadata",
        FieldDisposition.MAPPED,
        "steps",
        subtree=True,
        note="agent-owned; normalized only by a pinned agent metadata format",
    ),
    FieldDeclaration(
        "verifier_result.rewards", FieldDisposition.MAPPED, "reward.values", subtree=True
    ),
    _m("verifier_environment_mode", "verifier.environment_mode"),
    _m("exception_info.exception_type", "error.exception_type"),
    _m("exception_info.exception_message", "error.exception_message"),
    _m("exception_info.exception_traceback", "error.exception_traceback"),
    _m("exception_info.occurred_at", "error.occurred_at"),
    _m("started_at", "timing.trial.started_at"),
    _m("finished_at", "timing.trial.finished_at"),
    *(
        _m(f"{phase}.{edge}", f"timing.{phase}.{edge}")
        for phase in _PHASES
        for edge in ("started_at", "finished_at")
    ),
    _raw("step_results", "multi-step task results are not normalized"),
)

TRIAL_LOCK_FIELDS: tuple[FieldDeclaration, ...] = (
    _m("schema_version", "versions.trial_lock_schema"),
    _m("task.name", "task.lock_name"),
    _m("task.digest", "task.lock_digest"),
    _m("task.type", "task.lock_type"),
    _m("task.source", "task.lock_source"),
    _raw("task", "remaining task lock keys (host path, git pins)"),
    _raw("agent", "agent lock"),
    _raw("environment", "environment lock"),
    _raw("verifier", "verifier lock"),
    _raw("install_only", "lock flag"),
    _raw("timeout_multiplier", "lock flag"),
    _raw("bridge_inputs", "lock inputs"),
    _raw("skills", "lock inputs"),
)

RESIN_AGENT_STEP_FIELDS: tuple[FieldDeclaration, ...] = (
    _m("steps[].step", "steps[].index"),
    _m("steps[].finish_reason", "steps[].finish_reason"),
    _m("steps[].tool_calls", "steps[].tool_call_count"),
    _m("steps[].usage.prompt_tokens", "steps[].usage.input_tokens"),
    _m("steps[].usage.completion_tokens", "steps[].usage.output_tokens"),
    _m("steps[].usage.total_tokens", "steps[].usage.total_tokens"),
    _m("steps[].usage.prompt_tokens_details.cached_tokens", "steps[].usage.cached_input_tokens"),
    _m(
        "steps[].usage.completion_tokens_details.reasoning_tokens",
        "steps[].usage.reasoning_tokens",
    ),
    _m("steps[].tools[].name", "steps[].tools[].name"),
    _m("steps[].tools[].executed", "steps[].tools[].executed"),
    _m("steps[].tools[].return_code", "steps[].tools[].return_code"),
    _m("termination", "attempt.termination"),
    _m("tool_turn_limit", "attempt.tool_turn_limit"),
)

ARTIFACT_MANIFEST_FIELDS: tuple[FieldDeclaration, ...] = (
    _m("[].source", "media[].container_path"),
    _m("[].destination", "media[].relpath"),
    _m("[].type", "media[].kind"),
    _m("[].status", "media[].status"),
    _m("[].service", "media[].service"),
)

JOB_RESULT_FIELDS: tuple[FieldDeclaration, ...] = (
    _m("id", "job.job_id"),
    _m("started_at", "job.timing.started_at", note="naive local time; offset unknown"),
    _m("updated_at", "job.timing.updated_at", note="naive local time; offset unknown"),
    _m("finished_at", "job.timing.finished_at", note="naive local time; offset unknown"),
    _m("n_total_trials", "job.n_total_trials"),
    *(
        _m(f"stats.{key}", f"job.stats.{key}")
        for key in (
            "n_completed_trials",
            "n_errored_trials",
            "n_running_trials",
            "n_pending_trials",
            "n_cancelled_trials",
            "n_input_tokens",
            "n_cache_tokens",
            "n_output_tokens",
            "cost_usd",
        )
    ),
    _m("stats.n_retries", "job.retry.n_retries"),
    _raw("stats.evals", "per-eval aggregates are recomputable from trials; kept raw"),
)

JOB_LOCK_FIELDS: tuple[FieldDeclaration, ...] = (
    _m("schema_version", "versions.job_lock_schema"),
    _m("created_at", "job.timing.lock_created_at"),
    _m("harbor.version", "versions.harbor"),
    _m("harbor.is_editable", "versions.harbor_is_editable"),
    _m("n_concurrent_trials", "job.n_concurrent_trials"),
    _m("retry.max_retries", "job.retry.max_retries"),
    FieldDeclaration(
        "retry.exclude_exceptions",
        FieldDisposition.MAPPED,
        "job.retry.exclude_exceptions",
        subtree=True,
    ),
    _m("retry.wait_multiplier", "job.retry.wait_multiplier"),
    _m("retry.min_wait_sec", "job.retry.min_wait_sec"),
    _m("retry.max_wait_sec", "job.retry.max_wait_sec"),
    _raw("trials", "per-trial locks duplicated in each trial lock.json"),
)

# Whole files kept byte-exact and referenced by digest, never parsed into fields.
RAW_ONLY_FILES: tuple[str, ...] = (
    "job/config.json",
    "job/job.log",
    "trial/config.json",
    "trial/trial.log",
    "trial/verifier/test-stdout.txt",
    "trial/artifacts/** (digest-mapped as media)",
)

# Producer-level losses: facts the current Harbor + Resin shell agent output does
# not contain at all, so no consumer may claim them from this evidence.
PRODUCER_LOSSES: tuple[str, ...] = (
    "resinbench-shell-agent records no message text, tool arguments or tool output",
    "resinbench-shell-agent records no per-step timestamps; step timing is unknown",
    "trial result carries no attempt number; per-trial retry lineage is unknown",
    "Harbor phase clocks are producer UTC wall clock; no monotonic clock is recorded",
    "job result timestamps carry no UTC offset; their timezone is unknown",
    "raw provider request/response frames are not captured by Harbor",
)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    mapped: tuple[str, ...]
    raw_only: tuple[str, ...]
    lost: tuple[str, ...]
    unknown: tuple[str, ...]
    absent: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "mapped": list(self.mapped),
            "raw_only": list(self.raw_only),
            "lost": list(self.lost),
            "unknown": list(self.unknown),
            "absent": list(self.absent),
        }


def leaf_paths(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every leaf of a JSON value as ``(dotted path, value)``; arrays use ``[]``."""

    if isinstance(value, Mapping):
        if not value and prefix:
            return [(prefix, value)]
        out: list[tuple[str, Any]] = []
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            out.extend(leaf_paths(value[key], child))
        return out
    if isinstance(value, list):
        if not value:
            return [(prefix, value)]
        out = []
        for item in value:
            out.extend(leaf_paths(item, f"{prefix}[]"))
        return out
    return [(prefix, value)]


def _declaration_for(
    path: str, declarations: tuple[FieldDeclaration, ...]
) -> FieldDeclaration | None:
    exact = next((item for item in declarations if item.source_path == path), None)
    if exact is not None:
        return exact
    best: FieldDeclaration | None = None
    for item in declarations:
        if not item.subtree:
            continue
        stem = item.source_path
        is_descendant = path.startswith((stem + ".", stem + "[]"))
        if is_descendant and (best is None or len(stem) > len(best.source_path)):
            best = item
    return best


def check_coverage(
    payload: Any,
    declarations: tuple[FieldDeclaration, ...],
    *,
    source: str,
) -> CoverageReport:
    """Refuse undeclared fields; classify every observed leaf."""

    mapped: set[str] = set()
    raw_only: set[str] = set()
    lost: set[str] = set()
    unknown: set[str] = set()
    absent: set[str] = set()
    for path, value in leaf_paths(payload):
        declaration = _declaration_for(path, declarations)
        if declaration is None:
            # A null/empty container whose children are declared is a declared absence.
            if value in (None, {}, []) and any(
                item.source_path.startswith(path + ".") or item.source_path.startswith(path + "[]")
                for item in declarations
            ):
                absent.add(path)
                continue
            raise HarborFormatRefusal(
                HarborRefusalCode.UNDECLARED_FIELD,
                source=source,
                path=path,
                detail="field is not in the pinned mapping declaration",
            )
        bucket = {
            FieldDisposition.MAPPED: mapped,
            FieldDisposition.RAW_ONLY: raw_only,
            FieldDisposition.LOST: lost,
        }[declaration.disposition]
        if declaration.disposition is FieldDisposition.MAPPED and value is None:
            unknown.add(path)
        else:
            bucket.add(path)
    return CoverageReport(
        mapped=tuple(sorted(mapped)),
        raw_only=tuple(sorted(raw_only)),
        lost=tuple(sorted(lost)),
        unknown=tuple(sorted(unknown)),
        absent=tuple(sorted(absent)),
    )


def mapping_declaration() -> dict[str, Any]:
    """The full, versioned per-field mapping declaration (for receipts and docs)."""

    return {
        "schema_version": HARBOR_EVIDENCE_SCHEMA_VERSION,
        "supported": {
            "harbor_versions": sorted(SUPPORTED_HARBOR_VERSIONS),
            "job_lock_schema_versions": sorted(SUPPORTED_JOB_LOCK_SCHEMA_VERSIONS),
            "trial_lock_schema_versions": sorted(SUPPORTED_TRIAL_LOCK_SCHEMA_VERSIONS),
            "atif_versions": sorted(SUPPORTED_ATIF_VERSIONS),
            "agent_metadata": {
                f"{name}@{version}": fmt
                for (name, version), fmt in sorted(SUPPORTED_AGENT_METADATA.items())
            },
        },
        "files": {
            "trial/result.json": [item.to_dict() for item in TRIAL_RESULT_FIELDS],
            "trial/lock.json": [item.to_dict() for item in TRIAL_LOCK_FIELDS],
            "trial/" + AGENT_METADATA_FILES[RESIN_SHELL_AGENT_FORMAT]: [
                item.to_dict() for item in RESIN_AGENT_STEP_FIELDS
            ],
            "trial/artifacts/manifest.json": [item.to_dict() for item in ARTIFACT_MANIFEST_FIELDS],
            "job/result.json": [item.to_dict() for item in JOB_RESULT_FIELDS],
            "job/lock.json": [item.to_dict() for item in JOB_LOCK_FIELDS],
        },
        "raw_only_files": list(RAW_ONLY_FILES),
        "producer_losses": list(PRODUCER_LOSSES),
    }
