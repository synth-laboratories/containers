from dataclasses import replace

import pytest

from synth_containers.tracing.adapters.atif import import_atif
from synth_containers.tracing.annotation_store import AnnotationStore
from synth_containers.tracing.evidence_ops import new_evidence_bundle
from synth_containers.tracing.models.selectors import selector_for


def setup(tmp_path):
    trace = import_atif({'schema_version': 'ATIF-v1.7', 'trajectory_id': 'annotation-store', 'agent': {'name': 'test', 'version': '1'}, 'steps': [{'step_id': 1, 'source': 'user', 'message': 'inspect'}]})
    base = new_evidence_bundle(trace)
    target = selector_for(trace, kind='message', entity_id=trace.messages[0].message_id).to_dict()
    request = {'target': target, 'body': 'Retained instruction', 'author': 'Test reviewer', 'author_kind': 'human', 'expected_digest': base.content_digest}
    return trace, base, request, AnnotationStore(tmp_path)


def test_roundtrip_review_retains_old_evidence_and_trace(tmp_path):
    trace, base, request, store = setup(tmp_path)
    original = trace.content_digest
    first, note = store.append(trace, base, request)
    second, reviewed = store.append(trace, base, {**request, 'expected_digest': first.content_digest, 'supersedes_id': note.annotation_id, 'review_state': 'accepted'})
    assert reviewed.supersedes_id == note.annotation_id
    assert reviewed.revision == 2
    assert len(AnnotationStore(tmp_path).load(trace, base).annotations) == 2
    assert len(second.annotations) == 2 and len(first.annotations) == 1 and not base.annotations
    assert trace.content_digest == original
    assert len(list(tmp_path.rglob('sha256-*.json'))) == 3


def test_conflict_and_stale_selector_do_not_advance_head(tmp_path):
    trace, base, request, store = setup(tmp_path)
    updated, note = store.append(trace, base, request)
    with pytest.raises(ValueError, match='Evidence changed'):
        store.append(trace, base, request)
    with pytest.raises(ValueError, match='Unresolved'):
        store.append(trace, base, {**request, 'expected_digest': updated.content_digest, 'target': {**request['target'], 'trace_digest': 'sha256:stale'}})
    assert store.load(trace, base).content_digest == updated.content_digest
