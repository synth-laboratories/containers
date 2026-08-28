"""Harbor job-dir importer: a real, trusted, self-contained Trace V5 bundle.

A harbor job directory (as produced by a RuneBench/Harbor run) contains:

- ``trajectory.json`` — agent turns (roles, content, tool calls, per-call usage);
- ``skill_tracking.json`` — periodic xp/level score samples;
- ``reward.json`` — the verifier output for the rollout;
- ``frames/<step>.png`` — pre-extracted observation frames (optional);
- ``recording.mp4`` — the raw screen recording (optional, referenced by digest
  only; the video is never embedded in the bundle).

This adapter maps that job directory, together with the rollout's journaled
event stream, into a sealed :class:`TraceDocumentV5` inside a
:class:`LocalTraceBundle`:

- trajectory turns   → messages (typed tool-call/result parts), model-call
  spans with observed-harness usage, and lossless ``harbor.turn`` events;
- skill samples      → ``harbor.score_sample`` events plus completeness
  metadata;
- journal envelopes  → ``harbor.journal.<kind>`` events;
- frames             → content-addressed screenshot artifacts embedded in the
  bundle blob store;
- ``reward.json``    → typed evidence (reward definition/record and an
  episode-level aggregation) via the shared native-evaluation attachment.

Determinism: every wall-clock read during materialization is pinned (via
:func:`synth_containers.tracing.canonical.pinned_utc_now`) to the last
timestamp observed in the inputs, so re-importing the same job directory
yields byte-identical bundles and archive digests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Sequence

from ...event_log import chain_extend, chain_genesis, envelope_digest
from ..canonical import (
    bytes_digest,
    canonical_bytes,
    pinned_utc_now,
    record_id,
)
from ..capture.binding import WorkloadKind
from ..capture.redaction import redact_json_source_bytes
from ..evidence_ops import attach
from ..models.actors import (
    ActorKind,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
)
from ..models.artifacts import ArtifactCompleteness, ArtifactRefV5, ArtifactRole
from ..models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..models.document import TraceDocumentV5
from ..models.identity import AliasV1, TraceIdentityV5, TraceKind, mint_trace_id
from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType
from ..models.spans import SpanKind, SpanStatus, SpanV5, UsageProvenance, UsageV5
from ..models.standards import RewardAggregationV1
from ..native_evaluation import attach_native_evaluation
from ..projections.inspector import load_bundle
from ..store.bundle import LocalTraceBundle
from .native import IMPORTED_AT, _assemble_events, _latest_timestamp


HARBOR_SOURCE_FORMAT = "harbor.job-dir.v1"
HARBOR_EXTENSION_SCHEMA_VERSION = "synth.trace-extension.harbor.v1"
HARBOR_REWARD_AUTHORITY = "harbor.verifier"

_FRAME_STEP_PATTERN = re.compile(r"^(\d+)\.png$")

_ROLE_MAP = {
    "system": MessageRole.SYSTEM,
    "user": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "agent": MessageRole.ASSISTANT,
    "tool": MessageRole.TOOL,
    "environment": MessageRole.ENVIRONMENT,
}


@dataclass(frozen=True, slots=True)
class HarborProvenancePins:
    """Producer identity pinned into ``TraceProvenanceV5`` by the caller."""

    producer_commit: str | None = None
    container_image_digest: str | None = None
    runtime_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HarborJob:
    """A loaded harbor job directory, exactly as found on disk."""

    job_dir: Path
    trajectory_raw: Any
    turns: tuple[dict[str, Any], ...]
    skill_samples: tuple[dict[str, Any], ...]
    reward: Any
    frames: tuple[tuple[int, str, bytes], ...]  # (step, filename, content)
    recording_path: Path | None


def load_harbor_job(job_dir: str | Path) -> HarborJob:
    """Read a harbor job directory without interpreting it."""

    root = Path(job_dir)
    if not root.is_dir():
        raise ValueError(f"harbor job dir does not exist: {root}")
    trajectory_raw = _read_json(root / "trajectory.json", required=True)
    skill_raw = _read_json(root / "skill_tracking.json", required=False)
    reward = _read_json(root / "reward.json", required=False)
    frames: list[tuple[int, str, bytes]] = []
    frames_dir = root / "frames"
    if frames_dir.is_dir():
        for path in sorted(frames_dir.iterdir()):
            match = _FRAME_STEP_PATTERN.match(path.name)
            if match is None or not path.is_file():
                continue
            frames.append((int(match.group(1)), path.name, path.read_bytes()))
    frames.sort(key=lambda item: item[0])
    recording = root / "recording.mp4"
    return HarborJob(
        job_dir=root,
        trajectory_raw=trajectory_raw,
        turns=_rows(trajectory_raw, "turns"),
        skill_samples=_rows(skill_raw, "samples"),
        reward=reward,
        frames=tuple(frames),
        recording_path=recording if recording.is_file() else None,
    )


def load_journal_events(
    journal_events: str | Path | Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    """Journal envelopes from a JSONL/JSON file or an in-memory sequence."""

    if journal_events is None:
        return ()
    if isinstance(journal_events, (str, Path)):
        text = Path(journal_events).read_text(encoding="utf-8").strip()
        if not text:
            return ()
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = list(journal_events)
    envelopes: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("harbor journal event must be a JSON object")
        if row.get("control"):
            continue
        envelopes.append(dict(row))
    return tuple(envelopes)


def verify_journal_chain(
    journal: tuple[dict[str, Any], ...],
    *,
    rollout_id: str,
) -> str | None:
    """Verify a journaled event stream's hash chain; return its head.

    Chain semantics are ``synth.rollout.event-chain.v1`` (see
    :func:`synth_containers.event_log.chain_genesis`).  Verification needs the
    complete sequenced history (sequences contiguous from 1); a partial or
    unsequenced journal returns ``None`` — nothing is recorded rather than a
    head that cannot be proven.  A journal whose per-event digests or declared
    ``capture.closed`` chain head disagree with the recomputation raises.
    """

    sequenced = sorted(
        (
            row
            for row in journal
            if isinstance(row.get("sequence"), int) and not isinstance(row.get("sequence"), bool)
        ),
        key=lambda row: row["sequence"],
    )
    if not sequenced:
        return None
    if [row["sequence"] for row in sequenced] != list(range(1, len(sequenced) + 1)):
        return None
    head = chain_genesis(rollout_id)
    heads_by_sequence: dict[int, str] = {0: head}
    for row in sequenced:
        kind = str(row.get("kind") or row.get("event_type") or "")
        payload = row.get("payload")
        if not kind or not isinstance(payload, Mapping):
            return None
        expected = envelope_digest(kind, int(row["sequence"]), dict(payload))
        declared = row.get("digest")
        if declared is not None and declared != expected:
            raise ValueError(f"harbor_journal_digest_mismatch:sequence={row['sequence']}")
        head = chain_extend(head, expected)
        heads_by_sequence[int(row["sequence"])] = head
        if kind == "capture.closed":
            declared_head = payload.get("chain_head")
            declared_water = payload.get("high_water")
            if isinstance(declared_head, str) and isinstance(declared_water, int):
                if heads_by_sequence.get(declared_water) != declared_head:
                    raise ValueError(
                        f"harbor_journal_chain_head_mismatch:sequence={row['sequence']}"
                    )
    return head


def import_harbor_job(
    job_dir: str | Path,
    *,
    bundle: LocalTraceBundle,
    rollout_id: str,
    journal_events: str | Path | Sequence[Mapping[str, Any]] | None = None,
    pins: HarborProvenancePins | None = None,
) -> dict[str, Any]:
    """Import one harbor job directory into ``bundle`` as a sealed trace.

    Writes the sealed trace, the content-addressed frame artifacts, the typed
    reward evidence, the import receipts, and the bundle manifest.  Returns a
    summary with the trace/evidence identities and digests.
    """

    pins = pins or HarborProvenancePins()
    job = load_harbor_job(job_dir)
    journal = load_journal_events(journal_events)
    if not str(rollout_id).strip():
        raise ValueError("harbor import requires a rollout_id")
    # Verify the supplied journal's hash chain before anything is imported;
    # the head (when provable) is recorded in provenance, the harbor
    # extension, and the bundle manifest metadata.
    journal_chain_head = verify_journal_chain(journal, rollout_id=rollout_id)

    frame_digests = [
        {"step": step, "name": name, "digest": bytes_digest(content), "size_bytes": len(content)}
        for step, name, content in job.frames
    ]
    recording_meta = None
    if job.recording_path is not None:
        recording_bytes = job.recording_path.read_bytes()
        recording_meta = {
            "name": job.recording_path.name,
            "digest": bytes_digest(recording_bytes),
            "size_bytes": len(recording_bytes),
        }
    source_payload = {
        "format": HARBOR_SOURCE_FORMAT,
        "rollout_id": rollout_id,
        "trajectory": job.trajectory_raw,
        "skill_tracking": [dict(item) for item in job.skill_samples],
        "reward": job.reward,
        "journal": [dict(item) for item in journal],
        "frames": frame_digests,
        "recording": recording_meta,
    }
    source_bytes = canonical_bytes(source_payload)
    source_digest = bytes_digest(source_bytes)
    events = _harbor_events(job, journal)
    ended_at = _latest_timestamp((IMPORTED_AT, *(item["occurred_at"] for item in events)))

    with pinned_utc_now(ended_at):
        safe_source, redaction = redact_json_source_bytes(source_bytes)
        stored_source_digest = bundle.blobs.put(safe_source)
        trace_id = mint_trace_id(kind="imported_harbor", key=source_digest)
        frame_artifacts = _frame_artifacts(
            job,
            bundle=bundle,
            trace_id=trace_id,
            ingested_at=ended_at,
        )
        recording_artifact = _recording_artifact(
            job,
            trace_id=trace_id,
            recording_meta=recording_meta,
            ingested_at=ended_at,
        )
        result = _assemble_events(
            events,
            trace_id=trace_id,
            source_digest=source_digest,
            source_format=HARBOR_SOURCE_FORMAT,
            bundle=bundle,
            workload_kind=WorkloadKind.OTHER,
            identity=TraceIdentityV5(
                rollout_id=rollout_id,
                run_id=rollout_id,
                correlation_id=rollout_id,
            ),
            trace_kind=TraceKind.AGENT_ROLLOUT,
            document_adapter=lambda document: promote_harbor_document(
                document,
                job=job,
                journal=journal,
                rollout_id=rollout_id,
                source_digest=source_digest,
                pins=pins,
                artifacts=(
                    *frame_artifacts,
                    *((recording_artifact,) if recording_artifact else ()),
                ),
                journal_chain_head=journal_chain_head,
            ),
        )
        bundle.write_receipt(
            "harbor-import-source",
            {
                "source_digest": source_digest,
                "stored_source_digest": stored_source_digest,
                "source_format": HARBOR_SOURCE_FORMAT,
                "rollout_id": rollout_id,
                "frame_count": len(job.frames),
                "score_sample_count": len(job.skill_samples),
                "journal_event_count": len(journal),
                "journal_chain_head": journal_chain_head,
                "redaction": redaction.to_dict(),
            },
        )
        manifest_metadata = {
            "imported_source_digest": source_digest,
            "imported_stored_source_digest": stored_source_digest,
            "imported_source_format": HARBOR_SOURCE_FORMAT,
        }
        if journal_chain_head is not None:
            # The verified journal chain head travels on the bundle manifest so
            # the lite seal and the Trace V5 bundle carry the same head.
            manifest_metadata["journal_chain_head"] = journal_chain_head
        bundle.write_manifest(metadata=manifest_metadata)
        evidence_summary: dict[str, Any] = {}
        if job.reward is not None:
            evidence_summary = _attach_reward_evidence(
                bundle,
                job=job,
                rollout_id=rollout_id,
                source_digest=source_digest,
                produced_at=ended_at,
            )
            # The evidence attachment rewrote the manifest through its own
            # handles (dropping metadata); restore it on a fresh handle so the
            # final manifest carries both the evidence entries and the
            # import/chain metadata.
            bundle = LocalTraceBundle(bundle.root, bundle_id=bundle.bundle_id)
            bundle.write_manifest(metadata=manifest_metadata)
    return {
        **result,
        "stored_source_digest": stored_source_digest,
        "rollout_id": rollout_id,
        "frame_count": len(job.frames),
        "score_sample_count": len(job.skill_samples),
        "journal_event_count": len(journal),
        "journal_chain_head": journal_chain_head,
        **evidence_summary,
    }


def materialize_harbor_trace_bundle(
    job_dir: str | Path,
    *,
    archive_path: str | Path,
    rollout_id: str,
    journal_events: str | Path | Sequence[Mapping[str, Any]] | None = None,
    producer_commit: str | None = None,
    container_image_digest: str | None = None,
    runtime_version: str | None = None,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize a harbor job dir into a portable, deterministic bundle ZIP.

    This is the public entry point for the RuneBench facade (and Workshop):
    it builds the sealed bundle (in ``bundle_root``, or a private scratch
    directory), then writes the verified deterministic archive to
    ``archive_path`` and returns the import summary including
    ``archive_digest``.
    """

    pins = HarborProvenancePins(
        producer_commit=producer_commit,
        container_image_digest=container_image_digest,
        runtime_version=runtime_version,
    )

    def _run(root: Path) -> dict[str, Any]:
        bundle = LocalTraceBundle(root, bundle_id=f"bundle-{rollout_id}")
        result = import_harbor_job(
            job_dir,
            bundle=bundle,
            rollout_id=rollout_id,
            journal_events=journal_events,
            pins=pins,
        )
        archive_digest = bundle.write_archive(Path(archive_path))
        return {
            **result,
            "archive_path": str(archive_path),
            "archive_digest": archive_digest,
            "bundle_id": bundle.bundle_id,
        }

    if bundle_root is not None:
        return _run(Path(bundle_root))
    with TemporaryDirectory(prefix="synth-harbor-bundle-") as scratch:
        return _run(Path(scratch) / "bundle")


def promote_harbor_document(
    document: TraceDocumentV5,
    *,
    job: HarborJob,
    journal: tuple[dict[str, Any], ...],
    rollout_id: str,
    source_digest: str,
    pins: HarborProvenancePins,
    artifacts: tuple[ArtifactRefV5, ...],
    journal_chain_head: str | None = None,
) -> TraceDocumentV5:
    """Return the sealed V5 document with typed harbor planes.

    Native records remain losslessly available in event payloads; this adds
    the common V5 entities (messages, model-call spans, usage, artifacts,
    lifecycle, provenance) so consumers do not have to understand the harbor
    job-dir layout.
    """

    root_actor = document.actors[0]
    root_session = document.sessions[0]
    models = sorted(
        {str(turn.get("model") or "").strip() for turn in job.turns} - {""}
    )
    providers = sorted(
        {str(turn.get("provider") or "").strip() for turn in job.turns} - {""}
    )
    model = models[0] if len(models) == 1 else None
    provider = providers[0] if len(providers) == 1 else None
    task_id = _task_id(job)

    messages, spans = _turn_entities(
        job,
        document=document,
        actor_id=root_actor.actor_id,
        session_id=root_session.session_id,
    )
    usage = _aggregate_usage(spans, source_digest=source_digest)

    started_at = min((event.occurred_at for event in document.events), default=IMPORTED_AT)
    ended_at = max((event.occurred_at for event in document.events), default=IMPORTED_AT)
    reward_summary = _reward_summary(job.reward)
    has_tool_calls = any(_tool_calls(turn) for turn in job.turns)
    has_usage = any(span.usage is not None for span in spans)

    actor = replace(
        root_actor,
        kind=ActorKind.AGENT,
        display_name=model or "Harbor rollout",
        role="policy",
        subtype="harbor_job",
        harness="harbor",
        model=model,
        provider=provider,
        task_id=task_id,
        aliases=(
            AliasV1(
                namespace="harbor.rollout",
                value=rollout_id,
                target_id=root_actor.actor_id,
                target_kind="actor",
                provenance="observed",
            ),
        ),
        metadata={
            "rollout_id": rollout_id,
            "turn_count": len(job.turns),
        },
        content_digest="",
    ).sealed()
    session = replace(
        root_session,
        status=SessionStatus.COMPLETED,
        started_at=started_at,
        ended_at=ended_at,
        attempt_id=rollout_id,
        harness="harbor",
        provider=provider,
        coverage=SessionCoverageV5(
            model_calls=CoverageState.COMPLETE if has_usage else CoverageState.NOT_CAPTURED,
            agent_events=CoverageState.COMPLETE,
            environment_events=(
                CoverageState.COMPLETE if journal else CoverageState.NOT_CAPTURED
            ),
            tool_events=(
                CoverageState.COMPLETE if has_tool_calls else CoverageState.NOT_CAPTURED
            ),
            usage=CoverageState.COMPLETE if has_usage else CoverageState.NOT_CAPTURED,
            raw_provider=CoverageState.NOT_CAPTURED,
            reasons=("harbor job-dir native records",),
        ),
        aliases=(
            AliasV1(
                namespace="harbor.rollout",
                value=rollout_id,
                target_id=root_session.session_id,
                target_kind="session",
                provenance="observed",
            ),
        ),
        content_digest="",
    ).sealed()

    completeness = TraceCompletenessV5(
        capture_status=CaptureStatus.COMPLETE,
        terminal_event_observed=bool(job.reward is not None or journal),
        model_calls=CoverageState.COMPLETE if has_usage else CoverageState.NOT_CAPTURED,
        raw_provider=CoverageState.NOT_CAPTURED,
        agent_events=CoverageState.COMPLETE,
        environment_events=CoverageState.COMPLETE if journal else CoverageState.NOT_CAPTURED,
        tool_events=CoverageState.COMPLETE if has_tool_calls else CoverageState.NOT_CAPTURED,
        usage=CoverageState.COMPLETE if has_usage else CoverageState.NOT_CAPTURED,
        expected_record_count=len(document.events),
        captured_record_count=len(document.events),
        reasons=(
            "turns, tool calls, and usage were observed by the harbor harness",
            "raw provider transport was not captured",
        ),
        metadata={
            "adapter": "harbor_job@1",
            "frame_count": len(job.frames),
            "score_sample_count": len(job.skill_samples),
            "journal_event_count": len(journal),
        },
    )
    lifecycle = TraceLifecycleV5(
        status=TraceStatus.COMPLETED,
        started_at=started_at,
        ended_at=ended_at,
        termination=TerminationV5(
            reason=str(reward_summary.get("status") or "completed"),
            detail=str(reward_summary.get("detail") or ""),
        ),
    )
    provenance = replace(
        document.provenance,
        producer="synth_containers.tracing.adapters.harbor",
        producer_version="1",
        source_format=HARBOR_SOURCE_FORMAT,
        producer_commit=pins.producer_commit,
        container_image_digest=pins.container_image_digest,
        runtime_version=pins.runtime_version,
        model=model,
        provider=provider,
        harness="harbor",
        captured_at=ended_at,
        transformation_chain=(
            *document.provenance.transformation_chain,
            "harbor_job_import@1",
        ),
        extra={
            **document.provenance.extra,
            **dict(pins.extra),
            "source_digest": source_digest,
            "frame_count": len(job.frames),
            "score_sample_count": len(job.skill_samples),
            **(
                {"journal_chain_head": journal_chain_head}
                if journal_chain_head is not None
                else {}
            ),
        },
    )
    identity = replace(
        document.identity,
        rollout_id=rollout_id,
        run_id=rollout_id,
        correlation_id=rollout_id,
        task_id=task_id,
        benchmark="harbor",
        seed=_seed(job),
    )
    extensions = dict(document.extensions)
    extensions["harbor"] = {
        "schema_version": HARBOR_EXTENSION_SCHEMA_VERSION,
        "rollout_id": rollout_id,
        "task_id": task_id,
        "model": model,
        "provider": provider,
        "turn_count": len(job.turns),
        "tool_call_count": sum(len(_tool_calls(turn)) for turn in job.turns),
        "score_samples": [dict(item) for item in job.skill_samples],
        "frames": [
            {"step": step, "name": name, "digest": bytes_digest(content)}
            for step, name, content in job.frames
        ],
        "reward": reward_summary,
        "source_digest": source_digest,
        "journal_chain_head": journal_chain_head,
    }
    return replace(
        document,
        identity=identity,
        lifecycle=lifecycle,
        provenance=provenance,
        completeness=completeness,
        actors=(actor,),
        sessions=(session,),
        messages=tuple(messages),
        spans=tuple(spans),
        artifacts=(*document.artifacts, *artifacts),
        usage=usage,
        extensions=extensions,
        content_digest="",
    ).sealed()


# -- event assembly ----------------------------------------------------------


def _harbor_events(
    job: HarborJob,
    journal: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in journal:
        kind = str(row.get("kind") or row.get("event_type") or "event")
        sequence = row.get("sequence")
        payload = row.get("payload")
        events.append(
            {
                "event_type": f"harbor.journal.{kind}",
                "payload": {
                    **(dict(payload) if isinstance(payload, Mapping) else {"value": payload}),
                    "journal_sequence": sequence,
                    "journal_digest": row.get("digest"),
                },
                "source_id": f"journal:{sequence if sequence is not None else len(events)}",
                "occurred_at": _timestamp(row.get("ts") or row.get("occurred_at")),
            }
        )
    for index, turn in enumerate(job.turns):
        events.append(
            {
                "event_type": "harbor.turn",
                "payload": {**dict(turn), "turn_index": index},
                "source_id": f"turn:{index}",
                "occurred_at": _timestamp(turn.get("ts") or turn.get("timestamp")),
            }
        )
    for index, sample in enumerate(job.skill_samples):
        events.append(
            {
                "event_type": "harbor.score_sample",
                "payload": {**dict(sample), "sample_index": index},
                "source_id": f"score_sample:{index}",
                "occurred_at": _timestamp(sample.get("ts") or sample.get("timestamp")),
            }
        )
    events.sort(key=lambda item: _sort_moment(item["occurred_at"]))
    return events


def _turn_entities(
    job: HarborJob,
    *,
    document: TraceDocumentV5,
    actor_id: str,
    session_id: str,
) -> tuple[list[MessageNodeV5], list[SpanV5]]:
    turn_events = {
        int(event.payload.get("turn_index")): event
        for event in document.events
        if str(event.event_type) == "harbor.turn"
        and isinstance(event.payload.get("turn_index"), int)
    }
    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    predecessor: str | None = None
    known_call_ids: set[str] = set()
    for index, turn in enumerate(job.turns):
        for call in _tool_calls(turn):
            call_id = str(call.get("id") or call.get("tool_call_id") or "").strip()
            if call_id:
                known_call_ids.add(call_id)
    for index, turn in enumerate(job.turns):
        event = turn_events.get(index)
        occurred_at = event.occurred_at if event is not None else IMPORTED_AT
        role = _ROLE_MAP.get(str(turn.get("role") or "assistant").strip().lower())
        if role is None:
            role = MessageRole.ASSISTANT
        message_id = record_id(
            "msg",
            kind="harbor_turn",
            scope=(session_id,),
            key=index,
        )
        usage_row = turn.get("usage") if isinstance(turn.get("usage"), Mapping) else None
        span_id: str | None = None
        if role == MessageRole.ASSISTANT and usage_row is not None:
            span_id = record_id(
                "span",
                kind="harbor_model_call",
                scope=(document.trace_id, session_id),
                key=index,
            )
        parts = _turn_parts(turn, message_id=message_id, known_call_ids=known_call_ids)
        message = MessageNodeV5(
            message_id=message_id,
            role=role,
            parts=parts,
            sender_actor_id=actor_id,
            session_id=session_id,
            predecessor_message_ids=(predecessor,) if predecessor else (),
            produced_by_span_id=span_id,
            occurred_at=occurred_at,
            metadata={
                "turn_index": index,
                "native_event_id": event.event_id if event is not None else None,
            },
        ).sealed()
        messages.append(message)
        if span_id is not None:
            prompt_tokens = _int(usage_row.get("prompt_tokens") or usage_row.get("input_tokens"))
            completion_tokens = _int(
                usage_row.get("completion_tokens") or usage_row.get("output_tokens")
            )
            spans.append(
                SpanV5(
                    span_id=span_id,
                    span_kind=SpanKind.MODEL_CALL,
                    actor_id=actor_id,
                    session_id=session_id,
                    started_at=occurred_at,
                    ended_at=occurred_at,
                    status=SpanStatus.OK,
                    input_message_ids=(predecessor,) if predecessor else (),
                    output_message_ids=(message_id,),
                    usage=UsageV5(
                        provenance=UsageProvenance.OBSERVED_HARNESS,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=(
                            _int(usage_row.get("total_tokens"))
                            or (prompt_tokens or 0) + (completion_tokens or 0)
                        ),
                        requests=1,
                        source_refs=(
                            (event.event_id,) if event is not None else ()
                        ),
                    ),
                    detail={
                        "turn_index": index,
                        "model": turn.get("model"),
                        "finish_reason": turn.get("finish_reason"),
                        "native_event_id": event.event_id if event is not None else None,
                    },
                ).sealed()
            )
        predecessor = message_id
    return messages, spans


def _turn_parts(
    turn: Mapping[str, Any],
    *,
    message_id: str,
    known_call_ids: set[str],
) -> tuple[MessagePartV5, ...]:
    parts: list[MessagePartV5] = []

    def part_id() -> str:
        return f"{message_id}:{len(parts)}"

    text = turn.get("content") if isinstance(turn.get("content"), str) else turn.get("text")
    if isinstance(text, str) and text:
        parts.append(MessagePartV5(part_id=part_id(), type=PartType.TEXT, text=text))
    for call in _tool_calls(turn):
        call_id = str(call.get("id") or call.get("tool_call_id") or "").strip()
        arguments = call.get("arguments")
        parts.append(
            MessagePartV5(
                part_id=part_id(),
                type=PartType.TOOL_CALL,
                tool_call_id=call_id or None,
                tool_name=str(call.get("name") or call.get("tool") or "") or None,
                arguments_json=(
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, sort_keys=True)
                    if arguments is not None
                    else None
                ),
            )
        )
    result = turn.get("tool_result")
    if isinstance(result, Mapping):
        result_call_id = str(result.get("tool_call_id") or result.get("id") or "").strip()
        result_text = result.get("content") if isinstance(result.get("content"), str) else None
        if result_call_id and result_call_id in known_call_ids:
            parts.append(
                MessagePartV5(
                    part_id=part_id(),
                    type=PartType.TOOL_RESULT,
                    tool_call_id=result_call_id,
                    text=result_text,
                    is_error=bool(result.get("is_error")) or None,
                )
            )
        else:
            # A result that cannot cite its call degrades to structured text so
            # the trace never carries an orphan tool_result part.
            parts.append(
                MessagePartV5(
                    part_id=part_id(),
                    type=PartType.STRUCTURED,
                    structured=dict(result),
                    raw_kind="harbor.tool_result",
                )
            )
    if not parts:
        parts.append(
            MessagePartV5(
                part_id=part_id(),
                type=PartType.STRUCTURED,
                structured={
                    key: value
                    for key, value in dict(turn).items()
                    if isinstance(key, str)
                },
                raw_kind="harbor.turn",
            )
        )
    return tuple(parts)


# -- artifacts ---------------------------------------------------------------


def _frame_artifacts(
    job: HarborJob,
    *,
    bundle: LocalTraceBundle,
    trace_id: str,
    ingested_at: str,
) -> tuple[ArtifactRefV5, ...]:
    artifacts: list[ArtifactRefV5] = []
    for step, name, content in job.frames:
        digest = bundle.blobs.put(content)
        artifacts.append(
            ArtifactRefV5(
                artifact_id=record_id(
                    "art",
                    kind="harbor_frame",
                    scope=(trace_id,),
                    key={"step": step, "digest": digest},
                ),
                digest=digest,
                media_type="image/png",
                size_bytes=len(content),
                role=ArtifactRole.SCREENSHOT,
                uri=bundle.blobs.uri(digest),
                producer="synth_containers.tracing.adapters.harbor",
                source_authority="harbor",
                ingested_at=ingested_at,
                logical_name=f"frames/{name}",
                metadata={"step": step},
            )
        )
    return tuple(artifacts)


def _recording_artifact(
    job: HarborJob,
    *,
    trace_id: str,
    recording_meta: Mapping[str, Any] | None,
    ingested_at: str,
) -> ArtifactRefV5 | None:
    if job.recording_path is None or recording_meta is None:
        return None
    return ArtifactRefV5(
        artifact_id=record_id(
            "art",
            kind="harbor_recording",
            scope=(trace_id,),
            key=recording_meta["digest"],
        ),
        digest=str(recording_meta["digest"]),
        media_type="video/mp4",
        size_bytes=int(recording_meta["size_bytes"]),
        role=ArtifactRole.SCREENSHOT,
        completeness=ArtifactCompleteness.REFERENCE_ONLY,
        producer="synth_containers.tracing.adapters.harbor",
        source_authority="harbor",
        ingested_at=ingested_at,
        logical_name=str(recording_meta["name"]),
        metadata={"note": "video referenced by digest; bytes not embedded"},
    )


# -- evidence ----------------------------------------------------------------


def _attach_reward_evidence(
    bundle: LocalTraceBundle,
    *,
    job: HarborJob,
    rollout_id: str,
    source_digest: str,
    produced_at: str,
) -> dict[str, Any]:
    payload = _reward_evidence_payload(job, rollout_id=rollout_id)
    attach_result = attach_native_evaluation(
        bundle.root,
        payload=payload,
        source_name="reward.json",
        produced_at=produced_at,
    )
    summary = {
        "evidence_bundle_id": attach_result["evidence_bundle_id"],
        "evidence_bundle_digest": attach_result["evidence_bundle_digest"],
        "evaluation_id": attach_result["evaluation_id"],
        "reward_record_id": attach_result.get("reward_record_id"),
        "reward_validation_valid": attach_result["validation_valid"],
    }
    record_id_value = attach_result.get("reward_record_id")
    if not record_id_value:
        return summary
    # attach_native_evaluation wrote through its own LocalTraceBundle handle;
    # reload so this handle's manifest state includes those evidence revisions
    # (a manifest write from a stale handle would drop them).
    bundle = LocalTraceBundle(bundle.root, bundle_id=bundle.bundle_id)
    inspected = next(
        item
        for item in load_bundle(bundle.root)
        if item.trace.identity.correlation_id == rollout_id
    )
    evidence = inspected.evidence
    assert evidence is not None
    record = next(
        item for item in evidence.reward_records if item.reward_record_id == record_id_value
    )
    definition = next(
        item for item in evidence.reward_definitions if item.reward_id == record.reward_id
    )
    aggregation = RewardAggregationV1(
        aggregation_id=record_id(
            "ragg",
            kind="harbor_reward",
            scope=(inspected.trace.trace_id, record.reward_id),
            key=source_digest,
        ),
        reward_id=record.reward_id,
        input_reward_record_ids=(record.reward_record_id,),
        input_digests=(record.content_digest,),
        definition_digest=definition.content_digest,
        value=record.value,
        produced_at=produced_at,
        components=dict(record.components),
        grouping="episode",
        calculation="identity",
        metadata={"native": True, "source": "harbor.reward"},
    ).sealed()
    evidence = attach(
        evidence,
        kind="reward_aggregation",
        record=aggregation,
        created_at=produced_at,
    )
    bundle.write_evidence(evidence)
    bundle.write_receipt(
        "harbor-reward-aggregation",
        {
            "aggregation_id": aggregation.aggregation_id,
            "reward_record_id": record.reward_record_id,
            "evidence_bundle_digest": evidence.content_digest,
        },
    )
    bundle.write_manifest()
    summary["evidence_bundle_id"] = evidence.bundle_id
    summary["evidence_bundle_digest"] = evidence.content_digest
    summary["reward_aggregation_id"] = aggregation.aggregation_id
    return summary


def _reward_evidence_payload(job: HarborJob, *, rollout_id: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "authority": HARBOR_REWARD_AUTHORITY,
        "schema_version": "harbor.reward.v1",
        "trace_correlation_id": rollout_id,
        "benchmark": "harbor",
    }
    task_id = _task_id(job)
    if task_id:
        base["task_id"] = task_id
    reward = job.reward
    if isinstance(reward, Mapping):
        row = dict(reward)
        if "reward" in row or "verifier" in row:
            return {**base, **row}
        return {**base, "reward": row}
    return {**base, "reward": reward}


def _reward_summary(reward: Any) -> dict[str, Any]:
    if isinstance(reward, Mapping):
        return {key: value for key, value in dict(reward).items() if isinstance(key, str)}
    if reward is None:
        return {}
    return {"value": reward}


# -- small readers -----------------------------------------------------------


def _read_json(path: Path, *, required: bool) -> Any:
    if not path.is_file():
        if required:
            raise ValueError(f"harbor job dir is missing {path.name}")
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    # Prove the payload is canonicalizable before it enters identity/digest logic.
    canonical_bytes(loaded)
    return loaded


def _rows(payload: Any, key: str) -> tuple[dict[str, Any], ...]:
    if payload is None:
        return ()
    rows = payload
    if isinstance(payload, Mapping):
        rows = payload.get(key) or ()
    if not isinstance(rows, list):
        raise ValueError(f"harbor {key} must be a list of objects")
    return tuple(dict(item) for item in rows if isinstance(item, Mapping))


def _tool_calls(turn: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    calls = turn.get("tool_calls")
    if not isinstance(calls, list):
        return ()
    return tuple(item for item in calls if isinstance(item, Mapping))


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return IMPORTED_AT
    try:
        _sort_moment(value)
    except ValueError:
        return IMPORTED_AT
    return value


def _sort_moment(value: str) -> Any:
    from datetime import UTC, datetime

    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.utcoffset() is None:
        raise ValueError(f"harbor timestamp must include a timezone: {value!r}")
    return moment.astimezone(UTC)


def _aggregate_usage(spans: list[SpanV5], *, source_digest: str) -> UsageV5:
    observed = [span.usage for span in spans if span.usage is not None]
    if not observed:
        return UsageV5()
    prompt = sum(int(item.prompt_tokens or 0) for item in observed)
    completion = sum(int(item.completion_tokens or 0) for item in observed)
    return UsageV5(
        provenance=UsageProvenance.OBSERVED_HARNESS,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        requests=len(observed),
        source_refs=(source_digest,),
    )


def _task_id(job: HarborJob) -> str | None:
    payload = job.trajectory_raw if isinstance(job.trajectory_raw, Mapping) else {}
    for key in ("task_id", "task", "benchmark_task"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(job.reward, Mapping):
        value = job.reward.get("task_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _seed(job: HarborJob) -> int | None:
    payload = job.trajectory_raw if isinstance(job.trajectory_raw, Mapping) else {}
    value = payload.get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


__all__ = [
    "HARBOR_EXTENSION_SCHEMA_VERSION",
    "HARBOR_SOURCE_FORMAT",
    "HarborJob",
    "HarborProvenancePins",
    "import_harbor_job",
    "load_harbor_job",
    "load_journal_events",
    "materialize_harbor_trace_bundle",
    "promote_harbor_document",
    "verify_journal_chain",
]
