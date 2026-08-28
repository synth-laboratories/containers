"""Journal contract v2: hash chain, acked subscription, retention, long-poll.

All additions are strictly additive on the events page — consumers that ignore
`cursor.chain_head`, `cursor.acked`, `retention`, `ack`, and `wait_ms` see the
same behavior as before.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_containers.event_log import (
    RolloutEventLog,
    chain_head_for,
    envelope_digest,
)
from synth_containers.platform import create_compat_app
from synth_containers.platform.seal import (
    _digest as seal_digest,
    seal_rollout_log,
    validate_rollout_seal,
)

ROLLOUT_ID = "roll_journal_v2"


def _make_app(tmp_path: Path, **runtime_config):
    app = create_compat_app(
        "craftax_engine",
        storage_root=tmp_path / "storage",
        runtime_config=runtime_config or None,
    )
    return app, app.state.platform


def _open_log(platform, rollout_id: str = ROLLOUT_ID) -> RolloutEventLog:
    platform.prepare(rollout_id, "sse", "run")
    return platform.logs[rollout_id]


# -- hash chain --------------------------------------------------------------


def test_chain_head_verifies_across_a_multi_page_read(tmp_path: Path) -> None:
    app, platform = _make_app(tmp_path)
    log = _open_log(platform)
    for step in range(7):
        log.append("env.step", {"step": step})
    client = TestClient(app)

    after = 0
    digests: list[str] = []
    final_cursor = None
    while True:
        page = client.get(
            f"/rollouts/{ROLLOUT_ID}/events", params={"after": after, "limit": 3}
        ).json()
        final_cursor = page["cursor"]
        for row in page["events"]:
            if row["control"]:
                assert after == 0, "control records only appear on the first page"
                continue
            digests.append(row["digest"])
        if not final_cursor["has_more"]:
            break
        after = final_cursor["next"]

    assert len(digests) == 7
    assert final_cursor["chain_head"] == chain_head_for(ROLLOUT_ID, digests)
    assert final_cursor["chain_head"] == log.chain_head


def test_capture_closed_carries_the_evidence_chain_head(tmp_path: Path) -> None:
    _, platform = _make_app(tmp_path)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    log.append("env.step", {"step": 1})
    evidence_head = log.chain_head
    log.seal_capture()
    closed = [item for item in log.after(0) if item.kind == "capture.closed"]
    assert len(closed) == 1
    assert closed[0].payload["chain_head"] == evidence_head
    assert closed[0].payload["high_water"] == 2
    # The final head (including the capture.* records) differs and matches
    # a full recomputation from the sequenced digests.
    all_digests = [item.digest for item in log.after(0) if item.sequence is not None]
    assert log.chain_head == chain_head_for(ROLLOUT_ID, all_digests)


def test_tampered_journal_fails_the_chain_check_on_recovery(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    log = RolloutEventLog(rollout_id=ROLLOUT_ID, stream_id="s", journal_path=journal)
    for step in range(3):
        log.append("env.step", {"step": step})
    log.seal_capture()

    lines = journal.read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        row = json.loads(line)
        envelope = row.get("envelope") or {}
        if envelope.get("sequence") == 2:
            # A consistent mutation: change the payload AND recompute the
            # per-event digest so only the chain betrays the rewrite.
            envelope["payload"] = {"step": 999}
            envelope["digest"] = envelope_digest("env.step", 2, {"step": 999})
            row["envelope"] = envelope
        rewritten.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    journal.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="event_log_chain_head_mismatch"):
        RolloutEventLog.recover(
            rollout_id=ROLLOUT_ID, stream_id="s", journal_path=journal
        )


def test_lite_seal_carries_and_validates_the_chain_head(tmp_path: Path) -> None:
    log = RolloutEventLog(
        rollout_id=ROLLOUT_ID, stream_id="s", journal_path=tmp_path / "j.jsonl"
    )
    log.append("env.step", {"step": 0})
    log.seal_capture()
    seal = seal_rollout_log(log)
    assert seal["chain_head"] == log.chain_head
    validate_rollout_seal(seal)

    corrupted = {key: value for key, value in seal.items() if key != "content_digest"}
    corrupted["chain_head"] = "0" * 64
    corrupted["content_digest"] = seal_digest(corrupted)
    with pytest.raises(ValueError, match="trace_seal_chain_head_mismatch"):
        validate_rollout_seal(corrupted)


# -- acked subscription + retention -----------------------------------------


def test_ack_bookkeeping_is_monotonic_clamped_and_durable(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    log = RolloutEventLog(rollout_id=ROLLOUT_ID, stream_id="s", journal_path=journal)
    log.append("env.step", {"step": 0})
    log.append("env.step", {"step": 1})
    assert log.record_ack(5) == 2, "acking the future clamps to high_water"
    assert log.record_ack(1) == 2, "acks are monotonic"
    recovered = RolloutEventLog.recover(
        rollout_id=ROLLOUT_ID, stream_id="s", journal_path=journal
    )
    assert recovered.last_acked == 2, "the ack head survives recovery"


def test_events_ack_param_records_and_echoes_the_ack_head(tmp_path: Path) -> None:
    app, platform = _make_app(tmp_path)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    log.append("env.step", {"step": 1})
    client = TestClient(app)
    page = client.get(f"/rollouts/{ROLLOUT_ID}/events", params={"ack": 1}).json()
    assert page["cursor"]["acked"] == 1
    page = client.get(f"/rollouts/{ROLLOUT_ID}/events").json()
    assert page["cursor"]["acked"] == 1, "ack head persists between requests"


def test_retention_transitions_acked_and_ttl(tmp_path: Path) -> None:
    app, platform = _make_app(tmp_path, journal_retention_ttl_seconds=60)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    client = TestClient(app)

    retention = client.get(f"/rollouts/{ROLLOUT_ID}/events").json()["retention"]
    assert retention["policy"] == "until-acked-or-ttl"
    assert retention["ttl_seconds"] == 60
    assert retention["released"] is False, "open journals are always retained"

    log.seal_capture()
    retention = client.get(f"/rollouts/{ROLLOUT_ID}/events").json()["retention"]
    assert retention["released"] is False, "closed but unacked stays retained"
    assert retention["expires_at"] is not None

    high_water = log.high_water
    retention = client.get(
        f"/rollouts/{ROLLOUT_ID}/events", params={"ack": high_water}
    ).json()["retention"]
    assert retention["released"] is True
    assert retention["released_reason"] == "acked"

    removed: list[tuple[str, list[str]]] = []
    rows = platform.release_retained_journals(
        remove=lambda rollout_id, paths: removed.append((rollout_id, paths))
    )
    (row,) = rows
    assert row["rollout_id"] == ROLLOUT_ID
    assert row["reason"] == "acked"
    assert removed == [(ROLLOUT_ID, row["paths"])]
    assert any("frame_assets" in path for path in row["paths"])
    # The default hook deletes nothing: the journal file is still there.
    assert log.journal_path is not None and log.journal_path.exists()


def test_retention_ttl_expiry_releases_unacked_closed_journals(tmp_path: Path) -> None:
    _, platform = _make_app(tmp_path, journal_retention_ttl_seconds=60)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    log.seal_capture()
    assert platform.release_retained_journals() == []
    closed_epoch = datetime.fromisoformat(log.closed_at.replace("Z", "+00:00")).timestamp()
    rows = platform.release_retained_journals(now=closed_epoch + 61)
    (row,) = rows
    assert row["reason"] == "ttl_expired"


# -- resume + long-poll ------------------------------------------------------


def test_after_remains_the_resume_token(tmp_path: Path) -> None:
    app, platform = _make_app(tmp_path)
    log = _open_log(platform)
    for step in range(4):
        log.append("env.step", {"step": step})
    client = TestClient(app)
    first = client.get(f"/rollouts/{ROLLOUT_ID}/events", params={"after": 0}).json()
    assert any(row["control"] for row in first["events"])
    resumed = client.get(f"/rollouts/{ROLLOUT_ID}/events", params={"after": 2}).json()
    assert [row["sequence"] for row in resumed["events"]] == [3, 4]
    assert not any(row["control"] for row in resumed["events"])
    assert resumed["cursor"]["next"] == 4


def test_wait_ms_returns_immediately_when_events_or_closure_exist(tmp_path: Path) -> None:
    app, platform = _make_app(tmp_path)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    client = TestClient(app)
    started = time.monotonic()
    page = client.get(
        f"/rollouts/{ROLLOUT_ID}/events", params={"after": 0, "wait_ms": 5000}
    ).json()
    assert time.monotonic() - started < 2.0
    assert page["cursor"]["high_water"] == 1

    log.seal_capture()
    started = time.monotonic()
    page = client.get(
        f"/rollouts/{ROLLOUT_ID}/events",
        params={"after": page["cursor"]["high_water"] + 10, "wait_ms": 5000},
    ).json()
    assert time.monotonic() - started < 2.0, "closed journals never block"


def test_wait_ms_bounds_the_empty_poll(tmp_path: Path) -> None:
    app, platform = _make_app(tmp_path)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    client = TestClient(app)
    started = time.monotonic()
    page = client.get(
        f"/rollouts/{ROLLOUT_ID}/events", params={"after": 1, "wait_ms": 300}
    ).json()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.1, "an empty page with wait_ms should actually wait"
    assert elapsed < 5.0
    assert page["events"] == []
    assert client.get(
        f"/rollouts/{ROLLOUT_ID}/events", params={"wait_ms": 20_000}
    ).status_code == 422, "wait_ms is bounded"


# -- OpenAPI conformance -----------------------------------------------------


def test_events_page_parses_against_the_openapi_schema(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    jsonschema = pytest.importorskip("jsonschema")
    openapi_path = Path(__file__).resolve().parents[1] / "openapi" / "container-contract-v1.yaml"
    doc = yaml.safe_load(openapi_path.read_text(encoding="utf-8"))
    schemas = doc["components"]["schemas"]
    bundled = {
        **schemas["RolloutEventsPage"],
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schemas,
    }
    schema = json.loads(json.dumps(bundled).replace("#/components/schemas/", "#/$defs/"))
    validator = jsonschema.Draft202012Validator(schema)

    events_route = doc["paths"]["/rollouts/{rollout_id}/events"]["get"]
    declared_params = {row["name"] for row in events_route["parameters"] if "name" in row}
    assert {"after", "limit", "ack", "wait_ms"} <= declared_params

    app, platform = _make_app(tmp_path)
    log = _open_log(platform)
    log.append("env.step", {"step": 0})
    log.seal_capture()
    page = TestClient(app).get(
        f"/rollouts/{ROLLOUT_ID}/events", params={"ack": log.high_water}
    ).json()
    errors = [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in validator.iter_errors(page)
    ]
    assert not errors, f"events page does not satisfy RolloutEventsPage: {errors}"
