"""The harbor job-dir adapter produces a real, trusted, deterministic Trace V5 bundle.

A synthetic job dir (small trajectory, skill samples, reward, three tiny PNG
frames — frames arrive pre-extracted; no docker, no mp4 decoding) is imported
through the public materialization entry point, then graded through the real
inspection path (``synth_containers.tracing.inspection``).
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest

from synth_containers.event_log import chain_head_for, envelope_digest
from synth_containers.tracing.adapters.harbor import (
    HARBOR_SOURCE_FORMAT,
    materialize_harbor_trace_bundle,
    verify_journal_chain,
)
from synth_containers.tracing.cli import main as synth_trace_main
from synth_containers.tracing.inspection import inspect_trace_input
from synth_containers.tracing.projections.inspector import load_bundle
from synth_containers.tracing.store.bundle import LocalTraceBundle

ROLLOUT_ID = "rollout_harbor_fixture"
PRODUCER_COMMIT = "0123abcd0123abcd0123abcd0123abcd0123abcd"
IMAGE_DIGEST = "sha256:" + "ab" * 32

FRAME_COLORS = ((10, 20, 30), (40, 50, 60), (70, 80, 90))


def _tiny_png(color: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    body = zlib.compress(b"\x00" + bytes(color))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", body)
        + chunk(b"IEND", b"")
    )


@pytest.fixture()
def job_dir(tmp_path: Path) -> Path:
    root = tmp_path / "harbor-job"
    (root / "frames").mkdir(parents=True)
    (root / "trajectory.json").write_text(
        json.dumps(
            {
                "task_id": "runebench.editor_task",
                "seed": 7,
                "turns": [
                    {
                        "role": "user",
                        "content": "Open the editor and fix the bug.",
                        "ts": "2026-08-27T00:00:01Z",
                    },
                    {
                        "role": "assistant",
                        "content": "Opening the editor.",
                        "model": "test-model",
                        "provider": "test-provider",
                        "tool_calls": [
                            {"id": "call_1", "name": "click", "arguments": {"x": 4, "y": 9}}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                        "ts": "2026-08-27T00:00:02Z",
                    },
                    {
                        "role": "tool",
                        "tool_result": {"tool_call_id": "call_1", "content": "clicked"},
                        "ts": "2026-08-27T00:00:03Z",
                    },
                    {
                        "role": "assistant",
                        "content": "Done.",
                        "model": "test-model",
                        "provider": "test-provider",
                        "usage": {"prompt_tokens": 12, "completion_tokens": 2},
                        "ts": "2026-08-27T00:00:04Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "skill_tracking.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"ts": "2026-08-27T00:00:02Z", "xp": 10, "level": 1},
                    {"ts": "2026-08-27T00:00:04Z", "xp": 25, "level": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "reward.json").write_text(
        json.dumps(
            {
                "reward": {
                    "value": 0.75,
                    "components": {"xp": 0.5, "completion": 1.0},
                },
                "passed": True,
                "task_id": "runebench.editor_task",
            }
        ),
        encoding="utf-8",
    )
    for step, color in enumerate(FRAME_COLORS):
        (root / "frames" / f"{step}.png").write_bytes(_tiny_png(color))
    return root


def _journal_rows() -> list[dict]:
    rows = [
        {
            "kind": "rollout.started",
            "sequence": 1,
            "ts": "2026-08-27T00:00:00Z",
            "payload": {"rollout_id": ROLLOUT_ID},
        },
        {
            "kind": "step.applied",
            "sequence": 2,
            "ts": "2026-08-27T00:00:03Z",
            "payload": {"step": 0},
        },
        {
            "kind": "rollout.terminal",
            "sequence": 3,
            "ts": "2026-08-27T00:00:05Z",
            "payload": {"status": "succeeded"},
        },
    ]
    for row in rows:
        row["digest"] = envelope_digest(row["kind"], row["sequence"], row["payload"])
    evidence_head = chain_head_for(ROLLOUT_ID, [row["digest"] for row in rows])
    closed_payload = {"high_water": 3, "chain_head": evidence_head}
    rows.append(
        {
            "kind": "capture.closed",
            "sequence": 4,
            "ts": "2026-08-27T00:00:05Z",
            "payload": closed_payload,
            "digest": envelope_digest("capture.closed", 4, closed_payload),
        }
    )
    return rows


def _journal_chain_head() -> str:
    return chain_head_for(ROLLOUT_ID, [row["digest"] for row in _journal_rows()])


@pytest.fixture()
def journal_path(tmp_path: Path) -> Path:
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _journal_rows()), encoding="utf-8")
    return path


def _materialize(job_dir: Path, journal_path: Path, archive_path: Path) -> dict:
    return materialize_harbor_trace_bundle(
        job_dir,
        archive_path=archive_path,
        rollout_id=ROLLOUT_ID,
        journal_events=journal_path,
        producer_commit=PRODUCER_COMMIT,
        container_image_digest=IMAGE_DIGEST,
    )


def test_harbor_bundle_grades_native_trusted_self_contained(
    job_dir: Path, journal_path: Path, tmp_path: Path
) -> None:
    archive = tmp_path / "bundle.zip"
    result = _materialize(job_dir, journal_path, archive)
    inspection = inspect_trace_input(archive)

    assert inspection.compatibility == "native"
    assert inspection.trusted is True
    assert inspection.self_contained is True
    assert inspection.validation.valid is True
    assert inspection.validation.issues == ()

    (trace,) = inspection.traces
    assert trace.verified is True
    assert trace.projectable is True
    assert trace.source_format == HARBOR_SOURCE_FORMAT
    assert trace.producer == "synth_containers.tracing.adapters.harbor"
    assert trace.model == "test-model"
    assert trace.benchmark == "harbor"
    assert trace.task_id == "runebench.editor_task"
    # 4 journal events + 4 turns + 2 score samples.
    assert trace.event_count == 10
    # One model-call span per assistant turn with usage.
    assert trace.span_count == 2
    assert trace.prompt_tokens == 22
    assert trace.completion_tokens == 6
    # Three embedded frames (the synthetic job has no recording.mp4).
    assert trace.artifact_count == 3
    assert result["frame_count"] == 3
    assert result["score_sample_count"] == 2
    assert result["journal_event_count"] == 4
    assert result["journal_chain_head"] == _journal_chain_head()


def test_harbor_archive_digest_is_deterministic_across_runs(
    job_dir: Path, journal_path: Path, tmp_path: Path
) -> None:
    first = _materialize(job_dir, journal_path, tmp_path / "first.zip")
    second = _materialize(job_dir, journal_path, tmp_path / "second.zip")
    assert first["archive_digest"] == second["archive_digest"]
    assert (tmp_path / "first.zip").read_bytes() == (tmp_path / "second.zip").read_bytes()
    assert first["trace_digest"] == second["trace_digest"]
    assert first["evidence_bundle_digest"] == second["evidence_bundle_digest"]


def test_harbor_document_planes_and_evidence(
    job_dir: Path, journal_path: Path, tmp_path: Path
) -> None:
    bundle_root = tmp_path / "bundle-dir"
    result = materialize_harbor_trace_bundle(
        job_dir,
        archive_path=tmp_path / "bundle.zip",
        rollout_id=ROLLOUT_ID,
        journal_events=journal_path,
        producer_commit=PRODUCER_COMMIT,
        container_image_digest=IMAGE_DIGEST,
        bundle_root=bundle_root,
    )
    (inspected,) = load_bundle(bundle_root)
    document = inspected.trace
    evidence = inspected.evidence

    # Provenance pins.
    assert document.provenance.producer_commit == PRODUCER_COMMIT
    assert document.provenance.container_image_digest == IMAGE_DIGEST
    assert document.identity.rollout_id == ROLLOUT_ID
    assert document.identity.correlation_id == ROLLOUT_ID

    # Trajectory -> messages with typed tool call/result parts and usage spans.
    assert len(document.messages) == 4
    tool_call_parts = [
        part
        for message in document.messages
        for part in message.parts
        if str(part.type) == "tool_call"
    ]
    tool_result_parts = [
        part
        for message in document.messages
        for part in message.parts
        if str(part.type) == "tool_result"
    ]
    assert [part.tool_call_id for part in tool_call_parts] == ["call_1"]
    assert [part.tool_call_id for part in tool_result_parts] == ["call_1"]
    assert len(document.spans) == 2
    assert int(document.usage.prompt_tokens or 0) == 22
    assert int(document.usage.total_tokens or 0) == 28

    # Skill tracking -> score_sample events and completeness metadata.
    score_events = document.events_of_type("harbor.score_sample")
    assert len(score_events) == 2
    assert score_events[-1].payload["xp"] == 25
    assert document.completeness.metadata["score_sample_count"] == 2
    assert document.completeness.metadata["frame_count"] == 3

    # Journal stream -> harbor.journal.* events in chronological order.
    journal_kinds = [
        str(event.event_type)
        for event in document.events
        if str(event.event_type).startswith("harbor.journal.")
    ]
    assert journal_kinds == [
        "harbor.journal.rollout.started",
        "harbor.journal.step.applied",
        "harbor.journal.rollout.terminal",
        "harbor.journal.capture.closed",
    ]

    # Journal chain head: verified from the supplied journal and recorded in
    # provenance, the harbor extension, and the bundle manifest metadata.
    expected_head = _journal_chain_head()
    assert document.provenance.extra["journal_chain_head"] == expected_head
    assert document.extensions["harbor"]["journal_chain_head"] == expected_head
    manifest = LocalTraceBundle(bundle_root).read_manifest()
    assert manifest["metadata"]["journal_chain_head"] == expected_head
    assert manifest["metadata"]["imported_source_format"] == HARBOR_SOURCE_FORMAT

    # Frames -> content-addressed screenshot artifacts embedded in the blob store.
    bundle = LocalTraceBundle(bundle_root)
    assert len(document.artifacts) == 3
    for artifact, color in zip(document.artifacts, FRAME_COLORS):
        assert str(artifact.role) == "screenshot"
        assert bundle.blobs.path_for(artifact.digest).read_bytes() == _tiny_png(color)

    # reward.json -> typed evidence with a reward record and an aggregation.
    assert evidence is not None
    (record,) = evidence.reward_records
    assert record.value == 0.75
    assert record.components == {"xp": 0.5, "completion": 1.0}
    (aggregation,) = evidence.reward_aggregations
    assert aggregation.input_reward_record_ids == (record.reward_record_id,)
    assert aggregation.value == 0.75
    assert aggregation.grouping == "episode"
    assert result["reward_aggregation_id"] == aggregation.aggregation_id
    assert result["reward_validation_valid"] is True


def test_harbor_rejects_a_tampered_journal(job_dir: Path, tmp_path: Path) -> None:
    rows = _journal_rows()
    rows[1]["payload"] = {"step": 999}  # digest no longer matches the payload
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="harbor_journal_digest_mismatch"):
        materialize_harbor_trace_bundle(
            job_dir,
            archive_path=tmp_path / "never.zip",
            rollout_id=ROLLOUT_ID,
            journal_events=tampered,
        )

    # A consistent rewrite (payload + digest recomputed) is betrayed by the
    # chain head declared in the capture.closed record.
    rows = _journal_rows()
    rows[1]["payload"] = {"step": 999}
    rows[1]["digest"] = envelope_digest("step.applied", 2, {"step": 999})
    rewritten = tmp_path / "rewritten.jsonl"
    rewritten.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="harbor_journal_chain_head_mismatch"):
        materialize_harbor_trace_bundle(
            job_dir,
            archive_path=tmp_path / "never2.zip",
            rollout_id=ROLLOUT_ID,
            journal_events=rewritten,
        )


def test_harbor_records_no_head_for_an_unverifiable_journal(
    job_dir: Path, tmp_path: Path
) -> None:
    # A journal slice that does not start at sequence 1 cannot be proven from
    # genesis: import proceeds, but no head is recorded.
    partial = tuple(row for row in _journal_rows() if row["sequence"] > 1)
    assert verify_journal_chain(partial, rollout_id=ROLLOUT_ID) is None
    result = materialize_harbor_trace_bundle(
        job_dir,
        archive_path=tmp_path / "partial.zip",
        rollout_id=ROLLOUT_ID,
        journal_events=list(partial),
    )
    assert result["journal_chain_head"] is None


def test_synth_trace_cli_import_harbor_writes_trusted_archive(
    job_dir: Path, journal_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    archive = tmp_path / "cli-bundle.zip"
    code = synth_trace_main(
        [
            "import-harbor",
            str(job_dir),
            "--rollout-id",
            ROLLOUT_ID,
            "--journal",
            str(journal_path),
            "--archive",
            str(archive),
            "--producer-commit",
            PRODUCER_COMMIT,
            "--image-digest",
            IMAGE_DIGEST,
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["rollout_id"] == ROLLOUT_ID
    assert printed["archive_digest"].startswith("sha256:")
    inspection = inspect_trace_input(archive)
    assert inspection.compatibility == "native"
    assert inspection.trusted is True
