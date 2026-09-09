"""Digest-keyed trace index and incremental evidence validation.

The cache is exact only because it is keyed by sealed digest: two traces never
alias, an index built for one digest refuses selectors bound to another, and
the incremental validator finds exactly what the full validator finds.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from synth_containers.tracing.annotation import (
    AnnotationService,
    AnnotationStore,
    DefinitionRegistry,
    IndexedTraceDocument,
    SealedTraceCache,
    SealedTraceIndex,
    build_craftax_compaction_trace,
    build_craftax_smoke_trace,
    register_builtin_annotators,
    validate_appended_evidence,
)
from synth_containers.tracing.annotation.builtin import ENVIRONMENT_STEP_STATUS_ID, TOOL_CALL_INTEGRITY_ID
from synth_containers.tracing.canonical import content_digest
from synth_containers.tracing.evidence_ops import attach_many
from synth_containers.tracing.models.selectors import SelectorKind, TraceSelectorV1, resolve_selector, selector_for
from synth_containers.tracing.models.standards import AnnotationStatus
from synth_containers.tracing.validation.validator import Severity, validate_evidence


def _errors(findings) -> list[str]:
    return sorted(item.code for item in findings if str(item.severity) == Severity.ERROR)


def _scaled(document, spans: int):
    """Replicate environment_step spans under fresh ids so lookups are worth indexing."""

    steps = [item for item in document.spans if str(item.span_kind) == "environment_step"]
    extra = []
    while len(document.spans) + len(extra) < spans:
        for item in steps:
            extra.append(replace(item, span_id=f"{item.span_id}.x{len(extra)}"))
            if len(document.spans) + len(extra) >= spans:
                break
    return replace(document, spans=document.spans + tuple(extra), content_digest="").sealed()


def test_sealed_trace_cache_is_keyed_by_digest_and_never_aliases() -> None:
    smoke = build_craftax_smoke_trace()
    compaction = build_craftax_compaction_trace()
    same_id_other_digest = replace(smoke, extensions={"perturbed": True}, content_digest="").sealed()
    assert same_id_other_digest.trace_id == smoke.trace_id and same_id_other_digest.content_digest != smoke.content_digest

    cache = SealedTraceCache(max_traces=3)
    smoke_index = cache.get(smoke)
    compaction_index = cache.get(compaction)
    perturbed_index = cache.get(same_id_other_digest)
    assert cache.get(smoke) is smoke_index and len(cache) == 3
    assert {smoke_index.trace_digest, compaction_index.trace_digest, perturbed_index.trace_digest} == {smoke.content_digest, compaction.content_digest, same_id_other_digest.content_digest}

    span_id = smoke.spans[0].span_id
    assert compaction.span(span_id) is None  # exists only in the smoke trace
    smoke_selector = selector_for(smoke, kind=SelectorKind.SPAN, entity_id=span_id)
    assert smoke_index.resolve(smoke_selector).resolved
    assert smoke_index.resolve(smoke_selector).resolved and smoke_index.stats()["hits"] == 1  # memoized on repeat

    # The same entity id bound to another digest is a different question with a different answer.
    foreign = selector_for(compaction, kind=SelectorKind.SPAN, entity_id=span_id)
    assert compaction_index.resolve(foreign).reason == "entity_not_found"
    # A selector bound to another trace is refused by this index and never memoized here.
    assert smoke_index.resolve(foreign).reason == "trace_id_mismatch"
    assert compaction_index.resolve(smoke_selector).reason == "trace_id_mismatch"
    assert smoke_index.stats()["memoized_selectors"] == 1 and compaction_index.stats()["memoized_selectors"] == 1
    # Same trace id, different sealed digest: separate entry, the smoke citation does not carry over.
    assert perturbed_index.resolve(smoke_selector).reason == "trace_digest_mismatch"
    assert perturbed_index.resolve(selector_for(same_id_other_digest, kind=SelectorKind.SPAN, entity_id=span_id)).resolved

    # Bounded: the least recently used digest is evicted, and comes back as a fresh index.
    cache.get(compaction)
    cache.get(same_id_other_digest)  # smoke is now the least recently used
    cache.get(build_craftax_smoke_trace(lane="other"))
    assert smoke.content_digest not in cache and compaction.content_digest in cache and len(cache) == 3
    assert cache.get(smoke) is not smoke_index

    with pytest.raises(ValueError):
        cache.get(replace(smoke, content_digest=""))


def test_indexed_document_is_the_same_document_with_constant_time_lookups() -> None:
    document = _scaled(build_craftax_smoke_trace(), 400)
    view = IndexedTraceDocument.of(document)
    assert IndexedTraceDocument.of(view) is view
    assert view.content_digest == document.content_digest and content_digest(view) == document.content_digest
    assert view.to_dict() == document.to_dict()
    for span in (document.spans[0], document.spans[-1]):
        assert view.span(span.span_id) is document.span(span.span_id) is span
    assert view.span("nope") is None and view.message("nope") is None and view.event("nope") is None
    assert view.message(document.messages[0].message_id) is document.messages[0]
    assert view.actor(document.actors[0].actor_id) is document.actors[0]
    assert view.session(document.sessions[0].session_id) is document.sessions[0]
    # Duplicate ids resolve to the first occurrence, exactly like the linear scan.
    duplicated = replace(document, spans=document.spans + (replace(document.spans[0], detail={**document.spans[0].detail, "shadow": True}),), content_digest="").sealed()
    assert IndexedTraceDocument.of(duplicated).span(document.spans[0].span_id) is duplicated.spans[0]
    # A replace() on the view is still a view, with a lazily rebuilt index.
    derived = replace(view, extensions={"derived": True}, content_digest="").sealed()
    assert isinstance(derived, IndexedTraceDocument) and derived.span(document.spans[0].span_id) is document.spans[0]
    # The real resolver runs on the view with identical results.
    selector = selector_for(document, kind=SelectorKind.MESSAGE, entity_id=document.messages[0].message_id, quote=document.messages[0].text()[:12])
    assert resolve_selector(view, selector) == resolve_selector(document, selector)


def _service(tmp_path: Path, document):
    registry = DefinitionRegistry()
    register_builtin_annotators(registry)
    service = AnnotationService(store=AnnotationStore(tmp_path / "store"), registry=registry)
    service.register_trace(document)
    return service


def test_incremental_validation_finds_exactly_what_full_validation_finds(tmp_path: Path) -> None:
    document = _scaled(build_craftax_smoke_trace(), 300)
    service = _service(tmp_path, document)
    first = service.submit_and_run(service.request_for(document, ENVIRONMENT_STEP_STATUS_ID))
    assert str(first.state) == "sealed" and first.applied_count == 295
    index = service.index_for(document)
    head = service.evidence_head(document.trace_id)
    assert index.bundle_verified(head.content_digest)  # the seal path validated it in full
    # Every finding cites its target span as evidence: the second resolve of each is a memo hit.
    assert index.stats()["hits"] >= 295 and index.stats()["misses"] <= 295 + 5

    second = service.submit_and_run(service.request_for(document, TOOL_CALL_INTEGRITY_ID))
    assert second.terminal and second.error is None
    head2 = service.evidence_head(document.trace_id)
    assert head2.content_digest != head.content_digest and index.bundle_verified(head2.content_digest)

    # A valid appended annotation: incremental mode, no findings, identical to the authority.
    good = replace(head2.annotations[0], annotation_id="ann_manual_good", content_digest="").sealed()
    candidate = attach_many(head2, records=(("annotation", good),))
    findings, report = validate_appended_evidence(index, candidate, prior=head2)
    assert report["mode"] == "incremental" and report["new_annotations"] == 1 and report["prior_annotations"] == len(head2.annotations)
    assert _errors(findings) == _errors(validate_evidence(document, candidate)[0]) == []
    assert index.bundle_verified(candidate.content_digest)

    # Evidence naming a span that does not exist: attach_many cannot see it (trace id/digest match),
    # the incremental check must, and it must say what the authority says.
    ghost = TraceSelectorV1(trace_id=document.trace_id, trace_digest=document.content_digest, kind=SelectorKind.SPAN, entity_id="span_does_not_exist")
    bad_evidence = replace(head2.annotations[1], annotation_id="ann_manual_ghost", evidence=(ghost,), content_digest="").sealed()
    candidate = attach_many(head2, records=(("annotation", bad_evidence),))
    findings, report = validate_appended_evidence(index, candidate, prior=head2)
    assert report["mode"] == "incremental" and "selector_unresolved" in _errors(findings)
    assert _errors(findings) == _errors(validate_evidence(document, candidate)[0])
    assert not index.bundle_verified(candidate.content_digest)

    # A quote that does not match the cited entity is caught the same way.
    span = document.spans[0]
    misquote = selector_for(document, kind=SelectorKind.SPAN, entity_id=span.span_id, quote="this text is not in the span")
    bad_quote = replace(head2.annotations[2], annotation_id="ann_manual_quote", evidence=(misquote,), content_digest="").sealed()
    candidate = attach_many(head2, records=(("annotation", bad_quote),))
    findings, _ = validate_appended_evidence(index, candidate, prior=head2)
    assert "selector_unresolved" in _errors(findings) and _errors(findings) == _errors(validate_evidence(document, candidate)[0])

    # A record digest that no longer matches its content is caught without re-digesting the prior head.
    tampered = replace(good, rationale="edited after sealing")  # keeps the stale content_digest
    candidate = replace(attach_many(head2, records=(("annotation", good),)), annotations=head2.annotations + (tampered,), content_digest="").sealed()
    findings, report = validate_appended_evidence(index, candidate, prior=head2)
    assert report["mode"] == "incremental" and "evidence_record_digest_mismatch" in _errors(findings)
    assert _errors(findings) == _errors(validate_evidence(document, candidate)[0])

    # A candidate that is not an extension of a verified head falls back to the authority in full.
    reordered = replace(candidate, annotations=tuple(reversed(candidate.annotations)), content_digest="").sealed()
    _, report = validate_appended_evidence(index, reordered, prior=head2)
    assert report == {"mode": "full", "reason": "not_an_extension"}
    other_index = SealedTraceIndex(document)  # fresh process: nothing verified yet
    _, report = validate_appended_evidence(other_index, attach_many(head2, records=(("annotation", good),)), prior=head2)
    assert report == {"mode": "full", "reason": "no_verified_prior"}


def test_store_caches_are_keyed_by_digest_and_pinned_to_the_file(tmp_path: Path) -> None:
    smoke = build_craftax_smoke_trace()
    compaction = build_craftax_compaction_trace()
    store = AnnotationStore(tmp_path / "store", source_cache_size=1)
    store.put_source(smoke)
    assert store.get_source(smoke.trace_id, smoke.content_digest) is not None
    assert store.get_source(smoke.trace_id, compaction.content_digest) is None  # wrong digest never aliases
    store.put_source(compaction)
    assert store.cache_stats()["sources"] == 1  # bounded: the smoke entry was evicted
    reloaded = store.get_source(smoke.trace_id, smoke.content_digest)
    assert reloaded is not None and reloaded.content_digest == smoke.content_digest and reloaded is not smoke
    # Out-of-band replacement of the sealed file drops the entry and the read fails closed.
    path = store.source_path(compaction.trace_id, compaction.content_digest)
    path.chmod(0o644)
    path.write_text(path.read_text().replace(compaction.trace_id, "trace_tampered", 1))
    from synth_containers.tracing.annotation import StoreCorruption

    with pytest.raises(StoreCorruption):
        store.get_source(compaction.trace_id, compaction.content_digest)


def test_review_and_consensus_use_the_incremental_path(tmp_path: Path) -> None:
    document = build_craftax_smoke_trace()
    service = _service(tmp_path, document)
    job = service.submit_and_run(service.request_for(document, ENVIRONMENT_STEP_STATUS_ID))
    annotation_id = job.annotation_ids[0]
    revised = service.review(annotation_id, decision="accepted", reviewer="qa", evidence=[{"kind": "trace"}])
    assert revised.supersedes_id == annotation_id and str(revised.status) == AnnotationStatus.APPLIED
    index = service.index_for(document)
    assert index.bundle_verified(service.evidence_head(document.trace_id).content_digest)
    with pytest.raises(Exception):
        service.review(annotation_id, decision="accepted", reviewer="qa", evidence=[{"kind": "span", "entity_id": "nope"}])
