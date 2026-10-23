"""Crash/retry and semantic conformance without any provider or rollout calls."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from synth_containers.tracing.atomic import durable_write
from synth_containers.tracing.canonical import canonical_text, bytes_digest
from synth_containers.tracing.capture.envelope import make_envelope
from synth_containers.tracing.capture.spool import RawSpool, repair, load_live_manifest
from synth_containers.tracing.adapters.native import import_native_to_bundle
from synth_containers.tracing.adapters.openai_responses import usage_observation_from_responses
from synth_containers.tracing.profiles import (
    BodyAvailabilityV1, CaptureClassV1, DurabilityProgressV1, DurationIntervalV1,
    EventTimingV1, TimeFactV1, TraceSemanticsV1, UsageFieldsV1,
    UsageObservationV1, duration_totals, reduce_usage, time_filtered_events,
)
from synth_containers.tracing.store.bundle import LocalTraceBundle
from synth_containers.tracing.store.filesystem import FilesystemBlobStore
from synth_containers.tracing.store.publication import prepare_publication, publish, read_publication, load_publication
from synth_containers.tracing.store.semantics import write_trace_semantics

D = "sha256:" + "a" * 64


def envelope(ordinal: int):
    return make_envelope(capture_id="capture", ordinal=ordinal, record_type="application.event",
        actor_id="actor", session_id="session", occurred_at="2026-10-21T12:00:00Z")


def test_immutable_racing_writers_publish_one_fsynced_identity(tmp_path):
    path = tmp_path / "blob"
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: durable_write(path, b"frozen"), range(8)))
    assert results.count(True) == 1
    assert path.read_bytes() == b"frozen"
    with pytest.raises(FileExistsError):
        durable_write(path, b"replacement")


def test_repair_retains_exact_tail_and_receipt_and_survives_commit_reply_loss(tmp_path, monkeypatch):
    import synth_containers.tracing.capture.spool as module
    segments = tmp_path / "segments"
    segments.mkdir()
    original = (canonical_text(envelope(0)) + "\n").encode() + b'{"interrupted":\xff'
    partial = segments / "000001.partial"
    partial.write_bytes(original)
    original_write = module.durable_write
    def lost_reply(path, payload, **kwargs):
        result = original_write(path, payload, **kwargs)
        if path.name == "latest.json":
            raise OSError("simulated process death after manifest commit")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, "durable_write", lost_reply)
        with pytest.raises(OSError):
            repair(tmp_path, capture_id="capture")
    assert partial.read_bytes() == original
    first = load_live_manifest(tmp_path)
    result = repair(tmp_path, capture_id="capture")
    assert result.recovered_records == 1 and result.truncated_tail_bytes == len(b'{"interrupted":\xff')
    assert not partial.exists()
    final = load_live_manifest(tmp_path)
    assert first == final and len(final.segments) == 1
    receipt = json.loads((tmp_path / final.metadata["repair_receipts"][0]).read_bytes())
    assert (tmp_path / receipt["original_path"]).read_bytes() == original
    assert receipt["original_digest"] == bytes_digest(original)
    assert list(RawSpool(tmp_path, capture_id="capture").records())[0]["ordinal"] == 0


@pytest.mark.parametrize("corruption", [b'broken\n', b'\xff\n', b'{}\n', b'\n'])
def test_repair_refuses_corrupt_middle_without_mutation(tmp_path, corruption):
    segments = tmp_path / "segments"
    segments.mkdir()
    body = (canonical_text(envelope(0)) + "\n").encode() + corruption + (canonical_text(envelope(2)) + "\n").encode()
    partial = segments / "000001.partial"
    partial.write_bytes(body)
    with pytest.raises(ValueError, match="corrupt complete record"):
        repair(tmp_path, capture_id="capture")
    assert partial.read_bytes() == body
    assert not list(tmp_path.glob("repairs/*"))


def test_uncertain_append_fences_owner_and_does_not_reuse_ordinal(tmp_path, monkeypatch):
    import synth_containers.tracing.capture.spool as module
    spool = RawSpool(tmp_path, capture_id="capture")
    original = module.os.fsync
    def fail(descriptor):
        raise OSError("uncertain fsync")
    with monkeypatch.context() as patch:
        patch.setattr(module.os, "fsync", fail)
        with pytest.raises(OSError):
            spool.append(envelope(0))
    monkeypatch.setattr(module.os, "fsync", original)
    with pytest.raises(RuntimeError, match="fenced"):
        spool.append(envelope(0))


def test_real_worker_death_preserves_fsynced_prefix_and_explicit_repair(tmp_path):
    child = subprocess.run([sys.executable, "-c", """
import os,sys
from pathlib import Path
from synth_containers.tracing.capture.spool import RawSpool
from synth_containers.tracing.capture.envelope import make_envelope
spool=RawSpool(Path(sys.argv[1]),capture_id='capture')
for i in range(3):
    spool.append(make_envelope(capture_id='capture',ordinal=i,record_type='application.event',actor_id='a',session_id='s'))
os._exit(73)
""", str(tmp_path)], capture_output=True, timeout=10)
    assert child.returncode == 73, child.stderr
    with pytest.raises(RuntimeError, match="partial"):
        RawSpool(tmp_path, capture_id="capture")
    assert repair(tmp_path, capture_id="capture").recovered_records == 3
    reopened = RawSpool(tmp_path, capture_id="capture")
    assert [row["ordinal"] for row in reopened.records()] == [0, 1, 2]
    reopened.append(envelope(3))
    reopened.close()
    assert [row["ordinal"] for row in reopened.records()] == [0, 1, 2, 3]


def test_gap_cannot_be_contiguous_committed_high_water():
    base = dict(producer_id="p", stream_id="s", restart_id="r", observed=8,
                local_durable=1, uploaded=1, manifest_committed=1, indexed=-1,
                retained_ranges=[{"first": 0, "last": 1}, {"first": 3, "last": 8}])
    assert DurabilityProgressV1.model_validate(base).local_durable == 1
    with pytest.raises(ValueError, match="gap"):
        DurabilityProgressV1.model_validate({**base, "local_durable": 8})
    with pytest.raises(ValueError, match="prerequisites"):
        DurabilityProgressV1.model_validate({**base, "indexed": 2})


def test_failed_execution_can_have_complete_capture_and_sealed_partial_is_valid():
    coverage = CaptureClassV1(name="actions", required=True, expected_count=1,
        retained_count=1, state="complete", policy_digest=D)
    base = TraceSemanticsV1(trace_digest=D, execution_outcome="failed", lifecycle="sealed",
        capture="complete", terminal_observed=True, capture_classes=(coverage,))
    assert base.capture == "complete"
    partial = base.model_dump(mode="python")
    partial.update(capture="partial", terminal_observed=False, execution_outcome="interrupted")
    partial["capture_classes"] = [dict(coverage.model_dump(), state="partial", reason="worker death", retained_count=None)]
    assert TraceSemanticsV1.model_validate(partial).lifecycle == "sealed"
    partial["capture"] = "complete"
    with pytest.raises(ValueError, match="required capture"):
        TraceSemanticsV1.model_validate(partial)


@pytest.mark.parametrize("disposition", ["reference_only", "redacted", "encrypted", "digest_only", "omitted", "missing", "truncated"])
def test_body_dispositions_remain_distinct_and_unknown_bytes_stay_unknown(disposition):
    body = BodyAvailabilityV1(artifact_id="a", disposition=disposition, required=True,
        retrievable=False, readable=False, reason="policy-qualified", policy_digest=D)
    assert body.retained_bytes is None and body.disposition == disposition
    with pytest.raises(ValueError, match="required body"):
        TraceSemanticsV1(trace_digest=D, execution_outcome="succeeded", lifecycle="sealed",
            capture="complete", terminal_observed=True, capture_classes=(), bodies=(body,))


def test_clock_skew_equal_times_unknown_time_and_parallel_duration():
    unknown = TimeFactV1(basis="unavailable")
    def event(sequence, value):
        occurred = TimeFactV1(value=value, basis="producer", clock_domain="producer:a", precision="second") if value else unknown
        return EventTimingV1(event_id=str(sequence), producer_id="p", stream_id="s", restart_id="r",
            source_sequence=sequence, occurred=occurred, observed=unknown, ingested=unknown, committed=unknown)
    events = (event(2, "2026-10-21T12:00:00Z"), event(0, "2026-10-21T12:00:00Z"), event(1, "2026-10-21T11:59:00Z"), event(3, None))
    result = time_filtered_events(events, basis="occurred", start=datetime(2026,10,21,tzinfo=UTC))
    assert [item.source_sequence for item in result] == [0, 1, 2]
    assert events[-1].occurred.value is None
    assert duration_totals((DurationIntervalV1(clock_domain="m", start_ns=0, end_ns=10),
        DurationIntervalV1(clock_domain="m", start_ns=5, end_ns=15))) == {
        "work_ns": 20, "elapsed_union_ns": 15, "critical_path_ns": None}
    assert duration_totals((DurationIntervalV1(clock_domain="m", start_ns=0),))["work_ns"] is None


def usage(ident, *, call="c", revision=0, mode="snapshot", fields=None, **kwargs):
    return UsageObservationV1(observation_id=ident, call_id=call, authority="provider", revision=revision,
        mode=mode, provenance="observed_provider", fields=UsageFieldsV1.model_validate(fields or {}),
        source_refs=("retained:raw",), **kwargs)


def test_usage_snapshot_delta_replay_unknowns_and_explicit_corrections():
    a = usage("a", fields={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    b = usage("b", revision=1, fields={"input_tokens": 10, "output_tokens": 7, "total_tokens": 17})
    result = reduce_usage((b, a, b))
    assert result.fields.total_tokens == 17 and result.selected_observation_ids == ("b",)
    unknown = usage("u", call="d", fields={"input_tokens": 3})
    partial = reduce_usage((a, unknown))
    assert partial.fields.input_tokens == 13 and partial.fields.output_tokens is None
    assert partial.known_subtotals.output_tokens == 5 and partial.cost_usd is None
    one = usage("d1", mode="delta", fields={"requests": 1})
    corrected = usage("d2", mode="delta", revision=1, supersedes="d1", fields={"requests": 2})
    assert reduce_usage((one, corrected, corrected)).fields.requests == 2
    with pytest.raises(ValueError, match="snapshot/delta"):
        reduce_usage((a, one))
    with pytest.raises(ValueError, match="identity conflicts"):
        reduce_usage((a, a.model_copy(update={"revision": 99})))


@pytest.mark.parametrize("value", [-1, True, 1.5, "3"])
def test_strict_provider_usage_does_not_coerce_invalid_token_values(value):
    with pytest.raises(ValueError):
        usage_observation_from_responses({"input_tokens": value}, call_id="c", observation_id="o", revision=0, source_ref="raw")


def make_bundle(root: Path):
    root.mkdir()
    source = root / "agent.jsonl"
    source.write_text('{"type":"item.completed","item":{"id":"i","type":"agent_message","text":"retained"}}\n')
    bundle = LocalTraceBundle(root / "bundle")
    result = import_native_to_bundle(source, source_format="codex_jsonl", bundle=bundle)
    return bundle, result


@pytest.mark.parametrize("failure", ["upload", "commit", "index", "commit_lost_reply"])
def test_publication_retries_from_retained_outbox_after_source_cleanup(tmp_path, failure):
    bundle, result = make_bundle(tmp_path / "source")
    before = bundle.read_trace(result["trace_digest"])
    profile = TraceSemanticsV1(trace_digest=result["trace_digest"], execution_outcome="unknown",
        lifecycle="sealed", capture="partial", terminal_observed=False, capture_classes=())
    write_trace_semantics(bundle, profile)
    bundle.write_manifest()
    outbox = tmp_path / "outbox"
    plan = prepare_publication(bundle, outbox, operation_id="pub", attribution={"run": "r", "org": "o"})
    with pytest.raises(FileExistsError):
        prepare_publication(bundle, outbox, operation_id="pub", attribution={"run": "different"})
    shutil.rmtree(tmp_path / "source")
    class Interrupted(FilesystemBlobStore):
        active = True
        def put_if_absent(self, content, **kwargs):
            commit = kwargs.get("media_type") != "application/zip"
            if self.active and ((failure == "upload" and not commit) or (failure in {"commit", "commit_lost_reply"} and commit)):
                self.active = False
                if failure == "commit_lost_reply":
                    super().put_if_absent(content, **kwargs)
                raise OSError("interrupted")
            return super().put_if_absent(content, **kwargs)
    remote = Interrupted(tmp_path / "remote")
    indexed = {}
    index_attempts = []
    def index(plan, remote):
        index_attempts.append(plan.commit_digest)
        if failure == "index" and len(index_attempts) == 1:
            raise OSError("index unavailable")
        indexed[plan.commit_digest] = plan.trace_digests
    first = publish(plan, outbox, remote, index=index)
    assert not first.indexed
    assert first.manifest_committed == (failure == "index")
    assert publish(load_publication(outbox, "pub"), outbox, remote, index=index).indexed
    assert publish(plan, outbox, remote, index=index).indexed
    assert len(indexed) == 1
    restored = read_publication(remote, plan.commit_digest, tmp_path / "fresh")
    assert restored.verify_self_contained() == (True, ())
    assert restored.read_trace(result["trace_digest"]) == before
    assert any("trace-semantics" in path for path in restored.read_manifest()["receipt_paths"])
    # The only retained callable was indexing; no original-agent/grader exists here.
    path = remote.path_for(plan.archive_digest)
    path.chmod(0o600)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        read_publication(remote, plan.commit_digest, tmp_path / "corrupt")
