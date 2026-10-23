"""Opt-in semantics bound to sealed V5 identities, without rewriting V5 bytes.

This profile is evidence, not an execution, retention, financial or access grant.
See docs/specs/trace-semantics-v1.md for required interpretation and projections.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field, model_validator

from synth_containers.package_models import Digest, Name, PackageModel

Count = Annotated[int, Field(strict=True, ge=0)]
Position = Annotated[int, Field(strict=True, ge=-1)]


class TimeFactV1(PackageModel):
    value: str | None = None
    basis: Literal["producer", "collector", "ingestion", "commit", "derived", "unavailable"]
    clock_domain: str | None = None
    precision: Literal["second", "millisecond", "microsecond", "nanosecond"] | None = None
    unit: Literal["rfc3339"] = "rfc3339"
    source_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_time(self) -> TimeFactV1:
        if self.value is None:
            if self.basis != "unavailable" or self.clock_domain is not None or self.precision is not None:
                raise ValueError("unknown time requires unavailable basis and no invented clock/precision")
        else:
            moment = datetime.fromisoformat(self.value.replace("Z", "+00:00"))
            if moment.utcoffset() is None or not self.clock_domain or self.precision is None:
                raise ValueError("known time requires timezone, clock domain and precision")
            if self.basis == "unavailable" or (self.basis == "derived" and not self.source_refs):
                raise ValueError("known/derived time requires an explicit basis and derivation sources")
        return self


class EventTimingV1(PackageModel):
    event_id: str
    producer_id: str
    stream_id: str
    restart_id: str
    source_sequence: Count
    occurred: TimeFactV1
    observed: TimeFactV1
    ingested: TimeFactV1
    committed: TimeFactV1
    caused_by_event_ids: tuple[str, ...] = ()


def time_filtered_events(
    events: tuple[EventTimingV1, ...], *, basis: Literal["occurred", "observed", "ingested", "committed"],
    start: datetime | None = None, end: datetime | None = None,
) -> tuple[EventTimingV1, ...]:
    """Half-open time filter, unknowns excluded; stable identity order, never clock order.

    Callers paginate this frozen result with the existing scoped stream protocol.
    Equal/regressing times never replace sequence/causality as replay authority.
    """
    if any(value is not None and value.utcoffset() is None for value in (start, end)):
        raise ValueError("time bounds require timezone")
    if start is not None and end is not None and start > end:
        raise ValueError("reversed time bounds")
    found = []
    for item in events:
        fact: TimeFactV1 = getattr(item, basis)
        if fact.value is None:
            continue
        moment = datetime.fromisoformat(fact.value.replace("Z", "+00:00"))
        if (start is None or moment >= start) and (end is None or moment < end):
            found.append(item)
    return tuple(sorted(found, key=lambda e: (e.producer_id, e.stream_id, e.restart_id,
                                              e.source_sequence, e.event_id)))


class DurationIntervalV1(PackageModel):
    clock_domain: str
    start_ns: Count
    end_ns: Count | None = None

    @model_validator(mode="after")
    def ordered(self) -> DurationIntervalV1:
        if self.end_ns is not None and self.end_ns < self.start_ns:
            raise ValueError("duration clock regressed")
        return self


def duration_totals(intervals: tuple[DurationIntervalV1, ...]) -> dict[str, int | None]:
    """Explicit sum-of-work versus elapsed union; no inferred critical path."""
    if len({item.clock_domain for item in intervals}) > 1:
        raise ValueError("durations from different clock domains are incomparable")
    if not intervals or any(item.end_ns is None for item in intervals):
        return {"work_ns": None, "elapsed_union_ns": None, "critical_path_ns": None}
    closed = sorted((item.start_ns, item.end_ns) for item in intervals if item.end_ns is not None)
    work = sum(end - start for start, end in closed)
    start, end = closed[0]
    elapsed = 0
    for left, right in closed[1:]:
        if left > end:
            elapsed += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return {"work_ns": work, "elapsed_union_ns": elapsed + end - start, "critical_path_ns": None}


class RetainedRangeV1(PackageModel):
    first: Count
    last: Count

    @model_validator(mode="after")
    def ordered(self) -> RetainedRangeV1:
        if self.last < self.first:
            raise ValueError("reversed retained range")
        return self


class CaptureClassV1(PackageModel):
    name: Name
    required: bool
    expected_count: Count | None = None
    retained_count: Count | None = None
    retained_bytes: Count | None = None
    retained_ranges: tuple[RetainedRangeV1, ...] = ()
    state: Literal["complete", "partial", "missing", "unsupported", "omitted"]
    reason: str | None = None
    policy_digest: Digest

    @model_validator(mode="after")
    def coverage(self) -> CaptureClassV1:
        last = -1
        for item in self.retained_ranges:
            if item.first <= last:
                raise ValueError("retained ranges must be ordered and disjoint")
            last = item.last
        if self.state != "complete" and not self.reason:
            raise ValueError("incomplete coverage requires reason/policy")
        if self.state == "complete" and self.expected_count is not None and self.retained_count != self.expected_count:
            raise ValueError("complete coverage contradicts expected count")
        return self


class BodyAvailabilityV1(PackageModel):
    artifact_id: str
    disposition: Literal["complete", "reference_only", "redacted", "encrypted", "digest_only",
                         "omitted", "missing", "truncated"]
    required: bool
    stored_digest: Digest | None = None
    original_digest: Digest | None = None
    retained_bytes: Count | None = None
    expected_bytes: Count | None = None
    retrievable: bool
    readable: bool
    reason: str | None = None
    policy_digest: Digest
    retention_root: str | None = None
    access_authority: str | None = None

    @model_validator(mode="after")
    def availability(self) -> BodyAvailabilityV1:
        if self.readable and (not self.retrievable or self.stored_digest is None):
            raise ValueError("readable body requires retrievable digest-addressed bytes")
        if self.retrievable and (not self.retention_root or self.stored_digest is None):
            raise ValueError("retrievable body requires retention root and stored digest")
        if self.disposition != "complete" and not self.reason:
            raise ValueError("body disposition requires reason")
        if self.disposition in {"missing", "omitted", "digest_only"} and self.retrievable:
            raise ValueError("unavailable body cannot claim retrievability")
        return self


class TraceSemanticsV1(PackageModel):
    schema_version: Literal["synth.trace-semantics.v1"] = "synth.trace-semantics.v1"
    trace_digest: Digest
    execution_outcome: Literal["running", "succeeded", "failed", "cancelled", "interrupted", "unknown"]
    lifecycle: Literal["live", "sealed"]
    capture: Literal["complete", "partial", "truncated", "unknown"]
    terminal_observed: bool
    capture_classes: tuple[CaptureClassV1, ...]
    bodies: tuple[BodyAvailabilityV1, ...] = ()
    timings: tuple[EventTimingV1, ...] = ()
    occurred_at_extension: Literal["nullable-sidecar-v1"] = "nullable-sidecar-v1"
    legacy_projection_losses: tuple[str, ...] = ()

    @model_validator(mode="after")
    def consistent(self) -> TraceSemanticsV1:
        for items, key in ((self.capture_classes, "name"), (self.bodies, "artifact_id"), (self.timings, "event_id")):
            ids = [getattr(item, key) for item in items]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate semantic identity")
        if self.capture == "complete":
            if any(item.required and item.state != "complete" for item in self.capture_classes):
                raise ValueError("required capture classes are incomplete")
            if any(item.required and (not item.readable or item.disposition != "complete") for item in self.bodies):
                raise ValueError("required body is unavailable or transformed")
        return self


class DurabilityProgressV1(PackageModel):
    schema_version: Literal["synth.trace-durability.v1"] = "synth.trace-durability.v1"
    producer_id: str
    stream_id: str
    restart_id: str
    observed: Position
    local_durable: Position
    uploaded: Position
    manifest_committed: Position
    indexed: Position
    retained_ranges: tuple[RetainedRangeV1, ...]

    @model_validator(mode="after")
    def contiguous(self) -> DurabilityProgressV1:
        values = (self.observed, self.local_durable, self.uploaded, self.manifest_committed, self.indexed)
        if list(values) != sorted(values, reverse=True):
            raise ValueError("durability stages cannot advance beyond their prerequisites")
        high = -1
        prior = -1
        for item in self.retained_ranges:
            if item.first <= prior:
                raise ValueError("retained ranges must be sorted and disjoint")
            if item.first == high + 1:
                high = item.last
            prior = item.last
        if self.local_durable > high:
            raise ValueError("contiguous durable high-water crosses a gap")
        return self


class UsageSubtotalsV1(PackageModel):
    input_tokens: Count | None = None
    output_tokens: Count | None = None
    total_tokens: Count | None = None
    cached_input_tokens: Count | None = None
    cache_write_input_tokens: Count | None = None
    reasoning_output_tokens: Count | None = None
    requests: Count | None = None


class UsageFieldsV1(UsageSubtotalsV1):
    @model_validator(mode="after")
    def subsets(self) -> UsageFieldsV1:
        if self.input_tokens is not None:
            for value in (self.cached_input_tokens, self.cache_write_input_tokens):
                if value is not None and value > self.input_tokens:
                    raise ValueError("cache tokens are an inclusive input subset")
        if self.output_tokens is not None and self.reasoning_output_tokens is not None and self.reasoning_output_tokens > self.output_tokens:
            raise ValueError("reasoning tokens are an inclusive output subset")
        if self.total_tokens is not None and self.input_tokens is not None and self.output_tokens is not None:
            if self.total_tokens != self.input_tokens + self.output_tokens:
                raise ValueError("total tokens must equal inclusive input plus output")
        return self


class CostFactV1(PackageModel):
    amount: Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")]
    currency: Literal["USD"] = "USD"
    basis: Literal["provider_supplied", "estimated"]
    authority: str
    source_ref: str
    model: str
    rate_card_digest: Digest | None = None

    @model_validator(mode="after")
    def estimated_identity(self) -> CostFactV1:
        if self.basis == "estimated" and self.rate_card_digest is None:
            raise ValueError("estimated cost requires the exact rate card identity")
        return self


class UsageObservationV1(PackageModel):
    schema_version: Literal["synth.usage-observation.v1"] = "synth.usage-observation.v1"
    observation_id: str
    call_id: str
    authority: str
    revision: Count
    mode: Literal["snapshot", "delta"]
    semantics: Literal["inclusive-input-output-v1"] = "inclusive-input-output-v1"
    provenance: Literal["observed_provider", "observed_harness", "partial", "unavailable"]
    fields: UsageFieldsV1
    cost: CostFactV1 | None = None
    supersedes: str | None = None
    source_refs: tuple[str, ...]


class UsageSummaryV1(PackageModel):
    fields: UsageFieldsV1
    known_subtotals: UsageSubtotalsV1
    missing_fields: tuple[str, ...]
    cost_usd: str | None
    known_cost_usd: str | None
    provenance: Literal["derived", "partial", "unavailable"]
    selected_observation_ids: tuple[str, ...]
    source_refs: tuple[str, ...]


def reduce_usage(observations: tuple[UsageObservationV1, ...]) -> UsageSummaryV1:
    """Deduplicate replay, select explicit revisions, and sum only disjoint facts."""
    unique: dict[str, UsageObservationV1] = {}
    for item in observations:
        item = UsageObservationV1.model_validate(item.model_dump(mode="python"))
        if item.observation_id in unique and unique[item.observation_id] != item:
            raise ValueError("usage observation identity conflicts")
        unique[item.observation_id] = item
    superseded: set[str] = set()
    corrections: dict[str, str] = {}
    for item in unique.values():
        if item.supersedes:
            if item.supersedes in corrections and corrections[item.supersedes] != item.observation_id:
                raise ValueError("usage correction branches require an explicit authority selection")
            corrections[item.supersedes] = item.observation_id
            prior = unique.get(item.supersedes)
            if prior is None or (prior.call_id, prior.authority, prior.mode) != (item.call_id, item.authority, item.mode) or prior.revision >= item.revision:
                raise ValueError("usage correction has no compatible prior authority/revision")
            superseded.add(item.supersedes)
    groups: dict[str, list[UsageObservationV1]] = {}
    for item in unique.values():
        if item.observation_id not in superseded:
            groups.setdefault(item.call_id, []).append(item)
    selected: list[UsageObservationV1] = []
    for items in groups.values():
        if len({item.authority for item in items}) != 1 or len({item.mode for item in items}) != 1:
            raise ValueError("usage authority or snapshot/delta selection is ambiguous")
        if items[0].mode == "snapshot":
            revisions = [item.revision for item in items]
            if len(set(revisions)) != len(revisions):
                raise ValueError("conflicting usage snapshots at the same revision")
            selected.append(max(items, key=lambda item: item.revision))
        else:
            selected.extend(items)
    selected.sort(key=lambda item: item.observation_id)
    totals: dict[str, int | None] = {}
    known: dict[str, int | None] = {}
    missing: list[str] = []
    for name in UsageFieldsV1.model_fields:
        values = [getattr(item.fields, name) for item in selected]
        observed = [value for value in values if value is not None]
        known[name] = sum(observed) if observed else None
        totals[name] = known[name] if selected and len(observed) == len(values) else None
        if totals[name] is None:
            missing.append(name)
    costs = [Decimal(item.cost.amount) for item in selected if item.cost is not None]
    known_cost = str(sum(costs, Decimal(0))) if costs else None
    complete = bool(selected) and not missing and len(costs) == len(selected) and all(
        item.provenance not in {"partial", "unavailable"} for item in selected)
    # Partial subtotals need not satisfy aggregate subset relations when fields
    # have different coverage. They are retained as field facts, never totals.
    return UsageSummaryV1(fields=UsageFieldsV1(**totals),
        known_subtotals=UsageSubtotalsV1(**known), missing_fields=tuple(missing),
        cost_usd=known_cost if selected and len(costs) == len(selected) else None,
        known_cost_usd=known_cost, provenance="derived" if complete else "partial" if selected else "unavailable",
        selected_observation_ids=tuple(item.observation_id for item in selected),
        source_refs=tuple(sorted({ref for item in selected for ref in item.source_refs})))


class EvidenceSubjectV1(PackageModel):
    kind: Literal["trace", "session", "span", "artifact", "rollout", "experiment", "external"]
    namespace: str
    identity: str
    digest: Digest


class MetadataFactV1(PackageModel):
    schema_version: Literal["synth.metadata-fact.v1"] = "synth.metadata-fact.v1"
    fact_id: str
    subject: EvidenceSubjectV1
    field: str
    value_json: str
    definition_digest: Digest
    authority: str
    revision: Count
    unit: str | None = None
    merge: Literal["replace-by-explicit-revision", "coexist"]
    supersedes: str | None = None
    inspected_evidence: tuple[Digest, ...] = ()

    @model_validator(mode="after")
    def valid_json(self) -> MetadataFactV1:
        import json
        json.loads(self.value_json, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        return self


def select_metadata(facts: tuple[MetadataFactV1, ...], *, authority: str) -> tuple[MetadataFactV1, ...]:
    """No latest-time authority. Alternatives coexist; corrections name predecessors."""
    unique: dict[str, MetadataFactV1] = {}
    for item in facts:
        if item.fact_id in unique and unique[item.fact_id] != item:
            raise ValueError("metadata fact identity conflict")
        unique[item.fact_id] = item
    superseded = set()
    for item in unique.values():
        if item.supersedes:
            prior = unique.get(item.supersedes)
            if prior is None or (prior.subject, prior.field, prior.authority, prior.definition_digest) != (
                item.subject, item.field, item.authority, item.definition_digest) or prior.revision >= item.revision:
                raise ValueError("metadata correction authority/definition/subject mismatch")
            superseded.add(item.supersedes)
    selected = [item for item in unique.values() if item.authority == authority and item.fact_id not in superseded]
    keys = [(item.subject, item.field, item.definition_digest) for item in selected if item.merge != "coexist"]
    # Pydantic subjects are frozen but may evolve to contain mappings: compare
    # canonical strings instead of relying on hashability of nested models.
    if len({str(key) for key in keys}) != len(keys):
        raise ValueError("metadata interpretations are ambiguous; declare supersession or coexistence")
    return tuple(sorted(selected, key=lambda item: item.fact_id))
