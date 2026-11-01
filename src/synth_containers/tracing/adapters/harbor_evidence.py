"""Qualify current Harbor job/trial output into raw-preserving normalized evidence.

``qualify_harbor_trial`` reads one Harbor trial directory, keeps every source
file's exact bytes (``RawSourceV1``) beside a normalized record whose
``raw_sources`` name those bytes by digest, and returns a per-field mapping
coverage report (mapped / raw_only / lost / unknown).  Anything outside the
pinned formats in :mod:`.harbor_formats` raises :class:`HarborFormatRefusal`.

Normalization is a pure function of the raw bytes: ``requalify_from_raw``
rebuilds the record from the preserved bytes alone and must reproduce the same
``normalized_digest``.
"""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ..canonical import bytes_digest, canonical_bytes
from .atif import inspect_atif_import
from .harbor_formats import (
    AGENT_METADATA_FILES,
    ARTIFACT_MANIFEST_FIELDS,
    ATIF_TRAJECTORY_FILE,
    HARBOR_EVIDENCE_SCHEMA_VERSION,
    HARBOR_JOB_SOURCE_FORMAT,
    HARBOR_TRIAL_SOURCE_FORMAT,
    JOB_LOCK_FIELDS,
    JOB_RESULT_FIELDS,
    PRODUCER_LOSSES,
    RESIN_AGENT_STEP_FIELDS,
    SUPPORTED_AGENT_METADATA,
    SUPPORTED_ATIF_VERSIONS,
    SUPPORTED_HARBOR_VERSIONS,
    SUPPORTED_JOB_LOCK_SCHEMA_VERSIONS,
    SUPPORTED_TRIAL_LOCK_SCHEMA_VERSIONS,
    TRIAL_LOCK_FIELDS,
    TRIAL_RESULT_FIELDS,
    CoverageReport,
    HarborFormatRefusal,
    HarborRefusalCode,
    check_coverage,
)

_PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")


@dataclass(frozen=True, slots=True)
class RawSourceV1:
    """One source file, byte-exact, addressed by digest."""

    relpath: str
    content: bytes

    @property
    def digest(self) -> str:
        return bytes_digest(self.content)

    def ref(self) -> dict[str, Any]:
        media_type = mimetypes.guess_type(self.relpath)[0] or "application/octet-stream"
        return {
            "relpath": self.relpath,
            "digest": self.digest,
            "size_bytes": len(self.content),
            "media_type": media_type,
        }


@dataclass(frozen=True, slots=True)
class HarborEvidenceV1:
    source_format: str
    raw: tuple[RawSourceV1, ...]
    normalized: dict[str, Any]
    coverage: dict[str, CoverageReport]

    @property
    def raw_digest(self) -> str:
        return bytes_digest(canonical_bytes([item.ref() for item in self.raw]))

    @property
    def normalized_digest(self) -> str:
        return bytes_digest(canonical_bytes(self.normalized))

    def coverage_dict(self) -> dict[str, Any]:
        return {name: report.to_dict() for name, report in sorted(self.coverage.items())}

    def write_raw(self, root: str | Path) -> Path:
        """Restore the preserved raw bytes under ``root`` (byte-exact)."""

        base = Path(root)
        for item in self.raw:
            target = base / item.relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.content)
        return base


# ---------------------------------------------------------------- loading


def _collect_raw(root: Path) -> tuple[RawSourceV1, ...]:
    if not root.is_dir():
        raise HarborFormatRefusal(
            HarborRefusalCode.MISSING_SOURCE, source=str(root), path="$", detail="not a directory"
        )
    return tuple(
        RawSourceV1(path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _json(raw: Mapping[str, RawSourceV1], relpath: str, *, required: bool) -> Any:
    item = raw.get(relpath)
    if item is None:
        if required:
            raise HarborFormatRefusal(
                HarborRefusalCode.MISSING_SOURCE,
                source=relpath,
                path="$",
                detail="required file absent",
            )
        return None
    try:
        return json.loads(item.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_SOURCE, source=relpath, path="$", detail=str(error)
        ) from None


def _expect(value: Any, kind: type | tuple[type, ...], *, source: str, path: str) -> Any:
    if value is not None and (
        not isinstance(value, kind) or isinstance(value, bool) and kind is int
    ):
        raise HarborFormatRefusal(
            HarborRefusalCode.TYPE_MISMATCH,
            source=source,
            path=path,
            detail=f"expected {kind}, got {type(value).__name__}",
        )
    return value


def _schema_version(lock: Any, supported: frozenset[int], *, source: str) -> int:
    if not isinstance(lock, Mapping):
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_SOURCE,
            source=source,
            path="$",
            detail="lock must be an object",
        )
    version = lock.get("schema_version")
    if version is None:
        raise HarborFormatRefusal(
            HarborRefusalCode.UNADVERTISED_VERSION,
            source=source,
            path="schema_version",
            detail="lock does not advertise a schema_version",
        )
    if isinstance(version, bool) or not isinstance(version, int) or version not in supported:
        raise HarborFormatRefusal(
            HarborRefusalCode.UNSUPPORTED_VERSION,
            source=source,
            path="schema_version",
            detail=f"{version!r} not in {sorted(supported)}",
        )
    return version


def _utc(value: Any, *, source: str, path: str, require_offset: bool = True) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_SOURCE, source=source, path=path, detail="not ISO 8601"
        ) from None
    if require_offset and parsed.utcoffset() is None:
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_SOURCE,
            source=source,
            path=path,
            detail="missing UTC offset",
        )
    return str(value)


def _duration_s(started: str | None, finished: str | None) -> str | None:
    if started is None or finished is None:
        return None
    start = datetime.fromisoformat(started)
    end = datetime.fromisoformat(finished)
    delta = end - start
    micros = (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    return str(Decimal(micros) / Decimal(1_000_000))


# ---------------------------------------------------------------- agent output


def _resin_steps(metadata: Any, *, source: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("steps"), list):
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_SOURCE,
            source=source,
            path="steps",
            detail="steps array required",
        )
    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(metadata["steps"]):
        path = f"steps[{index}]"
        if not isinstance(raw, Mapping) or raw.get("step") != index:
            raise HarborFormatRefusal(
                HarborRefusalCode.MALFORMED_SOURCE,
                source=source,
                path=f"{path}.step",
                detail=f"expected sequential step index {index}",
            )
        usage = raw.get("usage") or {}
        tools = raw.get("tools") or []
        call_count = _expect(raw.get("tool_calls"), int, source=source, path=f"{path}.tool_calls")
        if tools and call_count is not None and len(tools) != call_count:
            raise HarborFormatRefusal(
                HarborRefusalCode.INCONSISTENT_SOURCE,
                source=source,
                path=f"{path}.tools",
                detail=f"{len(tools)} tool records for tool_calls={call_count}",
            )
        steps.append(
            {
                "index": index,
                "finish_reason": raw.get("finish_reason"),
                "tool_call_count": call_count,
                "usage": {
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    ),
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens"
                    ),
                },
                "tools": [
                    {
                        "name": tool.get("name"),
                        "executed": tool.get("executed"),
                        "return_code": tool.get("return_code"),
                        "return_code_recorded": "return_code" in tool,
                    }
                    for tool in tools
                ],
                "timestamp": None,
            }
        )
    attempt = {
        "termination": metadata.get("termination"),
        "tool_turn_limit": metadata.get("tool_turn_limit"),
    }
    return steps, attempt


def _atif_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_ATIF,
            source=ATIF_TRAJECTORY_FILE,
            path="$",
            detail="not an object",
        )
    version = payload.get("schema_version")
    if version is None:
        raise HarborFormatRefusal(
            HarborRefusalCode.UNADVERTISED_VERSION,
            source=ATIF_TRAJECTORY_FILE,
            path="schema_version",
            detail="ATIF trajectory does not advertise schema_version",
        )
    if version not in SUPPORTED_ATIF_VERSIONS:
        raise HarborFormatRefusal(
            HarborRefusalCode.UNSUPPORTED_VERSION,
            source=ATIF_TRAJECTORY_FILE,
            path="schema_version",
            detail=f"{version!r} not in {sorted(SUPPORTED_ATIF_VERSIONS)}",
        )
    try:
        assessed = inspect_atif_import(payload)
    except ValueError as error:
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_ATIF,
            source=ATIF_TRAJECTORY_FILE,
            path="$",
            detail=str(error),
        ) from None
    return {
        "source_format": assessed["source_format"],
        "source_digest": assessed["source_digest"],
        "session_id": assessed["session_id"],
        "step_count": len(assessed["steps"]),
        "subagent_count": assessed["subagent_count"],
        "losses": list(assessed["losses"]),
    }


# ---------------------------------------------------------------- trial


def _normalize_trial(raw_items: tuple[RawSourceV1, ...]) -> HarborEvidenceV1:
    raw = {item.relpath: item for item in raw_items}
    coverage: dict[str, CoverageReport] = {}
    lock = _json(raw, "lock.json", required=True)
    trial_lock_schema = _schema_version(
        lock, SUPPORTED_TRIAL_LOCK_SCHEMA_VERSIONS, source="lock.json"
    )
    coverage["lock.json"] = check_coverage(lock, TRIAL_LOCK_FIELDS, source="lock.json")
    result = _json(raw, "result.json", required=True)
    if not isinstance(result, Mapping):
        raise HarborFormatRefusal(
            HarborRefusalCode.MALFORMED_SOURCE,
            source="result.json",
            path="$",
            detail="not an object",
        )
    coverage["result.json"] = check_coverage(result, TRIAL_RESULT_FIELDS, source="result.json")
    for key in ("id", "task_name", "trial_name"):
        if not isinstance(result.get(key), str) or not result[key]:
            raise HarborFormatRefusal(
                HarborRefusalCode.MALFORMED_SOURCE,
                source="result.json",
                path=key,
                detail="required string",
            )
    task_lock = lock.get("task") or {}
    if task_lock.get("name") not in (None, result["task_name"]):
        raise HarborFormatRefusal(
            HarborRefusalCode.INCONSISTENT_SOURCE,
            source="lock.json",
            path="task.name",
            detail="lock task name differs from result task_name",
        )

    agent_info = result.get("agent_info") or {}
    agent_key = (str(agent_info.get("name") or ""), str(agent_info.get("version") or ""))
    agent_format = SUPPORTED_AGENT_METADATA.get(agent_key)
    agent_result = result.get("agent_result")
    steps: list[dict[str, Any]] = []
    attempt_extra: dict[str, Any] = {"termination": None, "tool_turn_limit": None}
    step_source = "absent"
    if agent_result is not None:
        metadata = agent_result.get("metadata")
        if metadata is not None and agent_format is None:
            raise HarborFormatRefusal(
                HarborRefusalCode.UNSUPPORTED_VERSION,
                source="result.json",
                path="agent_info",
                detail=f"agent metadata format for {agent_key!r} is not pinned",
            )
        if metadata is not None and agent_format is not None:
            agent_file = AGENT_METADATA_FILES[agent_format]
            on_disk = _json(raw, agent_file, required=False)
            if on_disk is not None and on_disk != metadata:
                raise HarborFormatRefusal(
                    HarborRefusalCode.INCONSISTENT_SOURCE,
                    source=agent_file,
                    path="$",
                    detail="agent file differs from result.json agent_result.metadata",
                )
            coverage[agent_file] = check_coverage(
                metadata, RESIN_AGENT_STEP_FIELDS, source=agent_file
            )
            steps, attempt_extra = _resin_steps(metadata, source=agent_file)
            step_source = agent_format
    atif_payload = _json(raw, ATIF_TRAJECTORY_FILE, required=False)
    atif = _atif_summary(atif_payload) if atif_payload is not None else None

    rewards = (result.get("verifier_result") or {}).get("rewards")
    reward_txt = raw.get("verifier/reward.txt")
    reward_file_value: str | None = None
    if reward_txt is not None:
        reward_file_value = reward_txt.content.decode("utf-8").strip()
        declared = (rewards or {}).get("reward")
        try:
            parsed_reward = Decimal(reward_file_value)
        except ArithmeticError:
            raise HarborFormatRefusal(
                HarborRefusalCode.MALFORMED_SOURCE,
                source="verifier/reward.txt",
                path="$",
                detail="reward file is not a number",
            ) from None
        if declared is None or Decimal(str(declared)) != parsed_reward:
            raise HarborFormatRefusal(
                HarborRefusalCode.INCONSISTENT_SOURCE,
                source="verifier/reward.txt",
                path="$",
                detail="reward file differs from verifier_result.rewards.reward",
            )

    manifest = _json(raw, "artifacts/manifest.json", required=False)
    media: list[dict[str, Any]] = []
    if manifest is not None:
        if not isinstance(manifest, list):
            raise HarborFormatRefusal(
                HarborRefusalCode.MALFORMED_SOURCE,
                source="artifacts/manifest.json",
                path="$",
                detail="manifest must be an array",
            )
        coverage["artifacts/manifest.json"] = check_coverage(
            manifest, ARTIFACT_MANIFEST_FIELDS, source="artifacts/manifest.json"
        )
        for entry in manifest:
            relpath = str(entry.get("destination") or "")
            file_item = raw.get(relpath)
            under = sorted(key for key in raw if key.startswith(relpath + "/"))
            media.append(
                {
                    "container_path": entry.get("source"),
                    "relpath": relpath,
                    "kind": entry.get("type"),
                    "status": entry.get("status"),
                    "service": entry.get("service"),
                    "digest": file_item.digest if file_item is not None else None,
                    "contained_files": [raw[key].ref() for key in under],
                    "retained": file_item is not None or bool(under),
                }
            )

    exception = result.get("exception_info")
    error = None
    if exception is not None:
        error = {
            "exception_type": exception.get("exception_type"),
            "exception_message": exception.get("exception_message"),
            "exception_traceback": exception.get("exception_traceback"),
            "occurred_at": _utc(
                exception.get("occurred_at"),
                source="result.json",
                path="exception_info.occurred_at",
            ),
        }
    timing: dict[str, Any] = {
        "clock": "producer_utc_wall",
        "monotonic": None,
        "trial": {
            "started_at": _utc(result.get("started_at"), source="result.json", path="started_at"),
            "finished_at": _utc(
                result.get("finished_at"), source="result.json", path="finished_at"
            ),
        },
    }
    timing["trial"]["duration_s"] = _duration_s(
        timing["trial"]["started_at"], timing["trial"]["finished_at"]
    )
    for phase in _PHASES:
        window = result.get(phase)
        if window is None:
            timing[phase] = None
            continue
        started = _utc(window.get("started_at"), source="result.json", path=f"{phase}.started_at")
        finished = _utc(
            window.get("finished_at"), source="result.json", path=f"{phase}.finished_at"
        )
        timing[phase] = {
            "started_at": started,
            "finished_at": finished,
            "duration_s": _duration_s(started, finished),
        }

    config = result.get("config") or {}
    usage = agent_result or {}
    normalized = {
        "schema_version": HARBOR_EVIDENCE_SCHEMA_VERSION,
        "source_format": HARBOR_TRIAL_SOURCE_FORMAT,
        "versions": {"trial_lock_schema": trial_lock_schema, "agent_metadata": agent_format},
        "task": {
            "name": result["task_name"],
            "source": result.get("source"),
            "checksum": result.get("task_checksum"),
            "lock_digest": task_lock.get("digest"),
            "lock_type": task_lock.get("type"),
        },
        "attempt": {
            "trial_id": result["id"],
            "trial_name": result["trial_name"],
            "job_id": config.get("job_id"),
            "environment_type": (config.get("environment") or {}).get("type"),
            "status": "errored" if error is not None else "completed",
            "attempt_number": None,
            **attempt_extra,
        },
        "agent": {
            "name": agent_info.get("name"),
            "version": agent_info.get("version"),
            "import_name": (config.get("agent") or {}).get("name"),
            "requested_model": (config.get("agent") or {}).get("model_name"),
            "model": (agent_info.get("model_info") or {}).get("name"),
            "provider": (agent_info.get("model_info") or {}).get("provider"),
        },
        "usage": {
            "input_tokens": usage.get("n_input_tokens"),
            "cache_tokens": usage.get("n_cache_tokens"),
            "output_tokens": usage.get("n_output_tokens"),
            "cost_usd": usage.get("cost_usd"),
        },
        "steps": steps,
        "step_source": step_source,
        "atif": atif,
        "reward": {
            "values": dict(rewards) if isinstance(rewards, Mapping) else None,
            "reward_file": reward_file_value,
            "verifier_environment_mode": result.get("verifier_environment_mode"),
        },
        "error": error,
        "retry": {"attempt_number": None, "lineage": "unknown"},
        "timing": timing,
        "media": media,
        "raw_sources": [item.ref() for item in raw_items],
        "producer_losses": list(PRODUCER_LOSSES),
    }
    if steps and usage.get("n_input_tokens") is not None:
        step_sum = sum(int(step["usage"]["input_tokens"] or 0) for step in steps)
        if step_sum != usage["n_input_tokens"]:
            raise HarborFormatRefusal(
                HarborRefusalCode.INCONSISTENT_SOURCE,
                source="result.json",
                path="agent_result.n_input_tokens",
                detail=f"step prompt tokens sum {step_sum} != {usage['n_input_tokens']}",
            )
    return HarborEvidenceV1(HARBOR_TRIAL_SOURCE_FORMAT, raw_items, normalized, coverage)


def qualify_harbor_trial(trial_dir: str | Path) -> HarborEvidenceV1:
    """Qualify one current-Harbor trial directory (raw bytes + normalized record)."""

    return _normalize_trial(_collect_raw(Path(trial_dir)))


def requalify_from_raw(evidence: HarborEvidenceV1) -> HarborEvidenceV1:
    """Rebuild evidence from preserved raw bytes only (round-trip proof)."""

    with TemporaryDirectory(prefix="synth-harbor-requalify-") as scratch:
        root = evidence.write_raw(Path(scratch) / "src")
        if evidence.source_format == HARBOR_JOB_SOURCE_FORMAT:
            return qualify_harbor_job(root)
        return qualify_harbor_trial(root)


# ---------------------------------------------------------------- job


def qualify_harbor_job(job_dir: str | Path) -> HarborEvidenceV1:
    """Qualify a Harbor job directory and every trial directory under it."""

    root = Path(job_dir)
    raw_items = _collect_raw(root)
    raw = {item.relpath: item for item in raw_items}
    lock = _json(raw, "lock.json", required=True)
    job_lock_schema = _schema_version(lock, SUPPORTED_JOB_LOCK_SCHEMA_VERSIONS, source="lock.json")
    harbor_version = (lock.get("harbor") or {}).get("version")
    if harbor_version is None:
        raise HarborFormatRefusal(
            HarborRefusalCode.UNADVERTISED_VERSION,
            source="lock.json",
            path="harbor.version",
            detail="job lock does not advertise the Harbor version",
        )
    if harbor_version not in SUPPORTED_HARBOR_VERSIONS:
        raise HarborFormatRefusal(
            HarborRefusalCode.UNSUPPORTED_VERSION,
            source="lock.json",
            path="harbor.version",
            detail=f"{harbor_version!r} not in {sorted(SUPPORTED_HARBOR_VERSIONS)}",
        )
    coverage = {"lock.json": check_coverage(lock, JOB_LOCK_FIELDS, source="lock.json")}
    result = _json(raw, "result.json", required=True)
    coverage["result.json"] = check_coverage(result, JOB_RESULT_FIELDS, source="result.json")
    trial_names = sorted(
        {
            key.split("/", 1)[0]
            for key in raw
            if "/" in key and key.split("/", 1)[1] == "result.json"
        }
    )
    trials = []
    for name in trial_names:
        prefix = name + "/"
        trial_raw = tuple(
            RawSourceV1(item.relpath[len(prefix) :], item.content)
            for item in raw_items
            if item.relpath.startswith(prefix)
        )
        try:
            evidence = _normalize_trial(trial_raw)
        except HarborFormatRefusal as refusal:
            raise HarborFormatRefusal(
                refusal.code,
                source=f"{name}/{refusal.source}",
                path=refusal.path,
                detail=refusal.detail,
            ) from None
        if evidence.normalized["attempt"]["job_id"] != result.get("id"):
            raise HarborFormatRefusal(
                HarborRefusalCode.INCONSISTENT_SOURCE,
                source=f"{name}/result.json",
                path="config.job_id",
                detail="trial job_id differs from job result id",
            )
        for file_name, report in evidence.coverage.items():
            coverage[f"{name}/{file_name}"] = report
        trials.append(
            {
                "trial_dir": name,
                "normalized_digest": evidence.normalized_digest,
                **evidence.normalized,
            }
        )
    stats = result.get("stats") or {}
    if result.get("n_total_trials") is not None and result["n_total_trials"] != len(trials):
        raise HarborFormatRefusal(
            HarborRefusalCode.INCONSISTENT_SOURCE,
            source="result.json",
            path="n_total_trials",
            detail=f"{result['n_total_trials']} declared, {len(trials)} trial dirs present",
        )
    retry_policy = lock.get("retry") or {}
    normalized = {
        "schema_version": HARBOR_EVIDENCE_SCHEMA_VERSION,
        "source_format": HARBOR_JOB_SOURCE_FORMAT,
        "versions": {"harbor": harbor_version, "job_lock_schema": job_lock_schema},
        "job": {
            "job_id": result.get("id"),
            "n_total_trials": result.get("n_total_trials"),
            "n_concurrent_trials": lock.get("n_concurrent_trials"),
            "stats": {
                key: value for key, value in stats.items() if key not in {"evals", "n_retries"}
            },
            "timing": {
                "lock_created_at": _utc(
                    lock.get("created_at"), source="lock.json", path="created_at"
                ),
                # Harbor writes job result timestamps without an offset.
                "started_at": _utc(
                    result.get("started_at"),
                    source="result.json",
                    path="started_at",
                    require_offset=False,
                ),
                "finished_at": _utc(
                    result.get("finished_at"),
                    source="result.json",
                    path="finished_at",
                    require_offset=False,
                ),
                "updated_at": _utc(
                    result.get("updated_at"),
                    source="result.json",
                    path="updated_at",
                    require_offset=False,
                ),
                "timezone": "unknown",
            },
            "retry": {
                "n_retries": stats.get("n_retries"),
                "max_retries": retry_policy.get("max_retries"),
                "exclude_exceptions": retry_policy.get("exclude_exceptions"),
                "wait_multiplier": retry_policy.get("wait_multiplier"),
                "min_wait_sec": retry_policy.get("min_wait_sec"),
                "max_wait_sec": retry_policy.get("max_wait_sec"),
            },
        },
        "trials": trials,
        "raw_sources": [item.ref() for item in raw_items],
        "producer_losses": list(PRODUCER_LOSSES),
    }
    return HarborEvidenceV1(HARBOR_JOB_SOURCE_FORMAT, raw_items, normalized, coverage)
