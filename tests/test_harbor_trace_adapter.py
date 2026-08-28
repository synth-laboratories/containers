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

from synth_containers.tracing.adapters.harbor import (
    HARBOR_SOURCE_FORMAT,
    materialize_harbor_trace_bundle,
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


@pytest.fixture()
def journal_path(tmp_path: Path) -> Path:
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
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
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
    # 3 journal events + 4 turns + 2 score samples.
    assert trace.event_count == 9
    # One model-call span per assistant turn with usage.
    assert trace.span_count == 2
    assert trace.prompt_tokens == 22
    assert trace.completion_tokens == 6
    # Three embedded frames (the synthetic job has no recording.mp4).
    assert trace.artifact_count == 3
    assert result["frame_count"] == 3
    assert result["score_sample_count"] == 2
    assert result["journal_event_count"] == 3


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
    ]

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
