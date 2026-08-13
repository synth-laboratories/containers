from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from synth_containers.tracing.adapters.atif import import_atif
from synth_containers.tracing.adapters.native import (
    import_native_to_bundle,
    write_imported_document,
)
from synth_containers.tracing.canonical import bytes_digest, canonical_text, content_digest
from synth_containers.tracing.cli import main as trace_cli_main
from synth_containers.tracing.inspection import (
    ROLLOUT_INSPECTOR_PROJECTION_FORMAT,
    TRACE_INSPECTION_SCHEMA_VERSION,
    inspect_trace_input,
)
from synth_containers.tracing.projections.rollout_inspector import (
    rollout_inspector_from_sealed,
)
from synth_containers.tracing.store.bundle import LocalTraceBundle


def _trace():
    return import_atif(
        {
            "schema_version": "ATIF-v1.7",
            "trajectory_id": "inspection-demo",
            "agent": {"name": "demo-agent", "version": "1"},
            "steps": [
                {
                    "step_id": 1,
                    "timestamp": "2026-01-01T00:00:00Z",
                    "source": "user",
                    "message": "inspect me",
                },
                {
                    "step_id": 2,
                    "timestamp": "2026-01-01T00:00:01Z",
                    "source": "agent",
                    "message": "done",
                },
            ],
        }
    )


def _bundle(root: Path) -> LocalTraceBundle:
    bundle = LocalTraceBundle(root)
    write_imported_document(
        _trace(),
        source_digest=bytes_digest(b"inspection-source"),
        source_format="ATIF-v1.7",
        bundle=bundle,
    )
    return bundle


def test_inspect_standalone_sealed_v5_is_native_and_projectable(tmp_path: Path) -> None:
    trace = _trace()
    source = tmp_path / "trace.json"
    source.write_text(canonical_text(trace) + "\n", encoding="utf-8")

    result = inspect_trace_input(source)

    assert result.schema_version == TRACE_INSPECTION_SCHEMA_VERSION
    assert result.input_kind == "standalone_trace"
    assert result.compatibility == "native"
    assert result.trusted is True
    assert result.validation.valid is True
    assert result.source_bytes_digest == bytes_digest(source.read_bytes())
    assert result.traces[0].trace_digest == trace.content_digest
    assert result.traces[0].verified is True
    assert result.traces[0].projectable is True
    assert result.assets[0].semantic_digest == trace.content_digest


def test_inspect_current_bundle_and_archive_share_semantic_identities(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "bundle")
    archive = tmp_path / "portable.zip"

    directory_result = inspect_trace_input(bundle.root, archive_output=archive)
    archive_result = inspect_trace_input(archive)

    assert directory_result.compatibility == "native"
    assert directory_result.input_kind == "bundle_directory"
    assert directory_result.bundle_digest == bundle.read_manifest()["content_digest"]
    assert directory_result.archive_digest == bytes_digest(archive.read_bytes())
    assert directory_result.self_contained is True
    assert directory_result.trusted is True
    assert directory_result.assets
    assert all(item.available and item.verified for item in directory_result.assets)
    assert archive_result.input_kind == "bundle_archive"
    assert archive_result.compatibility == "native"
    assert archive_result.source_bytes_digest == directory_result.archive_digest
    assert archive_result.archive_digest == directory_result.archive_digest
    assert archive_result.bundle_digest == directory_result.bundle_digest
    assert archive_result.traces[0].trace_digest == directory_result.traces[0].trace_digest


def test_push1_inline_manifest_remains_legacy_native(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "push1")
    manifest = bundle.read_manifest()
    manifest["objects"] = []
    manifest["content_digest"] = content_digest(manifest)
    bundle.manifest_path.write_text(canonical_text(manifest) + "\n", encoding="utf-8")

    result = inspect_trace_input(bundle.root)

    assert result.compatibility == "legacy_native"
    assert result.validation.valid is True
    assert result.self_contained is True
    assert result.trusted is True
    assert result.traces[0].projectable is True
    assert {item.kind for item in result.assets} >= {"trace", "binding", "receipt"}


def test_missing_object_is_partial_and_corruption_is_invalid(tmp_path: Path) -> None:
    partial_bundle = _bundle(tmp_path / "partial")
    partial_manifest = partial_bundle.read_manifest()
    sealed_path = partial_bundle.root / partial_manifest["traces"][0]["sealed_path"]
    sealed_path.unlink()

    partial = inspect_trace_input(partial_bundle.root)

    assert partial.compatibility == "partial"
    assert partial.self_contained is False
    assert partial.trusted is False
    assert partial.traces[0].projectable is False
    assert {item.code for item in partial.validation.issues} >= {
        "missing_object",
        "trace_unreadable",
    }
    partial_archive = tmp_path / "partial.zip"
    with zipfile.ZipFile(partial_archive, "w") as archive:
        for path in partial_bundle.root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(partial_bundle.root))
    archived_partial = inspect_trace_input(partial_archive)
    assert archived_partial.input_kind == "bundle_archive"
    assert archived_partial.compatibility == "partial"
    assert archived_partial.archive_digest is None

    corrupt_bundle = _bundle(tmp_path / "corrupt")
    corrupt_manifest = corrupt_bundle.read_manifest()
    corrupt_path = corrupt_bundle.root / corrupt_manifest["traces"][0]["sealed_path"]
    corrupt_path.chmod(0o644)
    payload = json.loads(corrupt_path.read_text(encoding="utf-8"))
    payload["trace_id"] = "tampered"
    corrupt_path.write_text(json.dumps(payload), encoding="utf-8")

    corrupt = inspect_trace_input(corrupt_bundle.root)

    assert corrupt.compatibility == "invalid"
    assert corrupt.trusted is False
    assert "corrupt_object" in {item.code for item in corrupt.validation.issues}


def test_cli_inspect_input_emits_contract_and_verified_archive(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = _bundle(tmp_path / "cli-bundle")
    archive = tmp_path / "cli-portable.zip"

    assert (
        trace_cli_main(["inspect-input", str(bundle.root), "--archive-output", str(archive)]) == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == TRACE_INSPECTION_SCHEMA_VERSION
    assert payload["compatibility"] == "native"
    assert payload["archive_digest"] == bytes_digest(archive.read_bytes())
    assert payload["traces"][0]["trace_digest"].startswith("sha256:")
    assert payload["assets"][0]["bytes_digest"].startswith("sha256:")


def test_rollout_inspector_projection_is_versioned_and_discoverable(
    tmp_path: Path,
) -> None:
    trace = _trace()
    packet = rollout_inspector_from_sealed(trace)
    assert packet.schema_version == ROLLOUT_INSPECTOR_PROJECTION_FORMAT
    assert packet.trace_digest == trace.content_digest
    assert packet.visual.trace_digest == trace.content_digest
    assert packet.content_digest.startswith("sha256:")

    bundle = _bundle(tmp_path / "projected")
    assert trace_cli_main(["project", str(bundle.root), "--format", "rollout-inspector"]) == 0
    result = inspect_trace_input(bundle.root)

    assert len(result.projections) == 1
    projection = result.projections[0]
    assert projection.format == ROLLOUT_INSPECTOR_PROJECTION_FORMAT
    assert projection.source_trace_digest == result.traces[0].trace_digest
    assert projection.verified is True


def test_craftax_native_import_promotes_policy_lanes_into_canonical_v5(
    tmp_path: Path,
) -> None:
    source = tmp_path / "craftax.json"
    events: list[dict[str, object]] = []
    for lane_index, (effort, reward, tokens) in enumerate((("low", 3.0, 120), ("high", 4.0, 180))):
        lane = f"craftax:gpt-5.6-luna:{effort}:s7"
        rollout_id = f"rollout-{effort}"
        base = lane_index * 10
        events.extend(
            [
                {
                    "event_id": f"{lane}:open",
                    "event_type": "craftax.eval.phase",
                    "occurred_at": f"2026-01-01T00:00:{base:02d}Z",
                    "payload": {
                        "lane": lane,
                        "rollout_id": rollout_id,
                        "task_id": "gamebench/craftax-singleplayer",
                        "phase": "rollout.opened",
                        "policy": {
                            "id": "committed-plan",
                            "kind": "llm_committed_plan",
                            "model": "gpt-5.6-luna",
                            "provider": "openai",
                            "reasoning_effort": effort,
                            "seed": 7,
                        },
                    },
                },
                {
                    "event_id": f"{lane}:call",
                    "event_type": "craftax.transcript",
                    "occurred_at": f"2026-01-01T00:00:{base + 1:02d}Z",
                    "payload": {
                        "lane": lane,
                        "rollout_id": rollout_id,
                        "kind": "policy.call",
                        "call_index": 1,
                        "model": "gpt-5.6-luna",
                        "reasoning_effort": effort,
                        "prompt": "Choose an action",
                        "reply": "MOVE_NORTH",
                        "prompt_tokens": tokens - 20,
                        "completion_tokens": 20,
                    },
                },
                {
                    "event_id": f"{lane}:action",
                    "event_type": "craftax.transcript",
                    "occurred_at": f"2026-01-01T00:00:{base + 2:02d}Z",
                    "payload": {
                        "lane": lane,
                        "rollout_id": rollout_id,
                        "kind": "action_applied",
                        "step_index": 1,
                        "action": "MOVE_NORTH",
                        "transition": "applied",
                        "payload": {"reason": "policy_plan"},
                    },
                },
                {
                    "event_id": f"{lane}:snapshot",
                    "event_type": "craftax.snapshot",
                    "occurred_at": f"2026-01-01T00:00:{base + 3:02d}Z",
                    "payload": {
                        "lane": lane,
                        "rollout_id": rollout_id,
                        "kind": "game.frame",
                        "step_index": 1,
                        "usage": {
                            "prompt_tokens": tokens - 20,
                            "completion_tokens": 20,
                            "total_tokens": tokens,
                            "cached_prompt_tokens": 10,
                            "calls": 1,
                            "estimated_usd": 0.001 * (lane_index + 1),
                        },
                    },
                },
                {
                    "event_id": f"{lane}:terminal",
                    "event_type": "craftax.eval.run.terminal",
                    "occurred_at": f"2026-01-01T00:00:{base + 4:02d}Z",
                    "payload": {
                        "lane": lane,
                        "rollout_id": rollout_id,
                        "task_id": "gamebench/craftax-singleplayer",
                        "stopped_on": "max_steps",
                        "reward": reward,
                        "env_steps": 1,
                    },
                },
            ]
        )
    source.write_text(
        json.dumps({"run_id": "paired-craftax", "events": events}),
        encoding="utf-8",
    )
    bundle = LocalTraceBundle(tmp_path / "craftax-bundle")

    imported = import_native_to_bundle(
        source,
        source_format="craftax_react",
        bundle=bundle,
    )
    trace = bundle.read_trace(imported["trace_digest"])

    assert trace["schema_version"] == "synth.trace.v5"
    assert trace["trace_kind"] == "evaluation_attempt"
    assert len(trace["actors"]) == 3
    assert len(trace["sessions"]) == 3
    assert [span["span_kind"] for span in trace["spans"]].count("model_call") == 2
    assert [span["span_kind"] for span in trace["spans"]].count("environment_step") == 2
    assert trace["usage"]["total_tokens"] == 300
    assert trace["usage"]["requests"] == 2
    assert trace["completeness"]["model_calls"] == "complete"
    craftax = trace["extensions"]["craftax"]
    assert craftax["paired"] is True
    assert [rollout["reward"] for rollout in craftax["rollouts"]] == [4.0, 3.0]

    assert trace_cli_main(["project", str(bundle.root), "--format", "rollout-inspector"]) == 0
    projection_path = next((bundle.root / "projections" / "rollout-inspector").glob("*.json"))
    projection = json.loads(projection_path.read_text(encoding="utf-8"))["payload"]
    assert projection["visual"]["summary"]["craftax"] == craftax
    assert len(projection["visual"]["lanes"]) == 2


def test_gamebench_is_not_accepted_as_a_trace_format(tmp_path: Path) -> None:
    source = tmp_path / "craftax.json"
    source.write_text(json.dumps({"events": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported native trace format"):
        import_native_to_bundle(
            source,
            source_format="gamebench_react",
            bundle=LocalTraceBundle(tmp_path / "not-a-format"),
        )
