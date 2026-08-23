"""Attach a consumer-native evaluator payload to a sealed trace as typed evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .canonical import bytes_digest, canonical_bytes, content_digest, record_id, utc_now
from .capture.redaction import assert_no_secrets, redact_payload
from .evidence_ops import attach_many, new_evidence_bundle
from .models.artifacts import ArtifactRefV5, ArtifactRole
from .models.evidence import TraceEvidenceBundleV5
from .models.selectors import GroundingStatus, SelectorKind, selector_for
from .models.standards import (
    BenchmarkVerdictV1,
    CriterionDefinitionV1,
    CriterionRole,
    EvaluationResultV1,
    ExecutionStatus,
    JUDGMENT_SCHEMA_VERSION,
    JudgmentStatus,
    JudgmentV1,
    ProducerRefV1,
    RecordState,
    RewardDefinitionV1,
    RewardEmission,
    RewardRecordV1,
    RewardSourceKind,
    RubricAggregationV1,
    RubricDefinitionV2,
    VerificationStatus,
    VerifierDefinitionV1,
    VerifierKind,
    VerifierResultV2,
    aggregate_rubric_score,
)
from .projections.inspector import InspectedBundle, load_bundle
from .store.bundle import LocalTraceBundle
from .validation.validator import validate


def attach_native_evaluation(
    bundle_root: Path,
    *,
    payload: Mapping[str, Any],
    source_name: str,
) -> dict[str, Any]:
    """Redact, retain, and type one native evaluation without mutating the trace."""

    redacted, report = redact_payload(payload)
    if not isinstance(redacted, dict):
        raise ValueError("native evaluation payload must be a JSON object")
    assert_no_secrets(redacted, where=f"native evaluation {source_name}")
    inspected = _select_trace(load_bundle(bundle_root), redacted)
    document = inspected.trace
    bundle = LocalTraceBundle(bundle_root)
    body = canonical_bytes(redacted)
    source_digest = bytes_digest(body)
    blob_digest = bundle.blobs.put(body)
    existing = _existing_attachment(inspected.evidence, source_digest)
    if existing is not None:
        return {
            "trace_id": document.trace_id,
            "trace_digest": document.content_digest,
            "evidence_bundle_id": inspected.evidence.bundle_id,
            "evidence_bundle_digest": inspected.evidence.content_digest,
            "evaluation_id": existing.evaluation_id,
            "native_source_digest": source_digest,
            "artifact_digest": (
                existing.artifacts[0].digest if existing.artifacts else blob_digest
            ),
            "aggregate_score": existing.aggregate_score,
            "validation_valid": validate(document, inspected.evidence).valid,
            "idempotent": True,
        }

    now = utc_now()
    authority = str(redacted.get("authority") or "native-evaluator")
    source_version = str(redacted.get("schema_version") or "unknown")
    producer = ProducerRefV1(
        kind="evaluator",
        name=authority,
        version=source_version,
    )
    artifact = ArtifactRefV5(
        artifact_id=record_id(
            "art",
            kind="native_evaluation",
            scope=(document.trace_id,),
            key=source_digest,
        ),
        digest=blob_digest,
        media_type="application/json",
        size_bytes=len(body),
        role=ArtifactRole.EVALUATION_OUTPUT,
        uri=bundle.blobs.uri(blob_digest),
        producer="synth_containers.tracing.native_evaluation",
        source_authority=authority,
        ingested_at=now,
        logical_name=source_name,
        metadata={
            "native_schema_version": source_version,
            "redaction_profile": report.profile,
            "source_digest": source_digest,
        },
    )

    typed = _typed_native_records(
        document=document,
        payload=redacted,
        artifact=artifact,
        authority=authority,
        source_version=source_version,
        source_digest=source_digest,
        producer=producer,
        produced_at=now,
    )
    evidence = inspected.evidence
    if evidence is None:
        evidence = new_evidence_bundle(document)
        bundle.write_evidence(evidence)
    additions = _new_records(evidence, typed["records"])
    if additions:
        evidence = attach_many(evidence, records=additions)
        bundle.write_evidence(evidence)

    validation = validate(document, evidence)
    bundle.write_receipt("native-evaluation-validation", validation)
    bundle.write_receipt(
        "native-evaluation-attachment",
        {
            "trace_id": document.trace_id,
            "trace_digest": document.content_digest,
            "evidence_bundle_digest": evidence.content_digest,
            "evaluation_id": typed["evaluation"].evaluation_id,
            "native_source_digest": source_digest,
            "validation_valid": validation.valid,
        },
    )
    catalog = bundle.open_catalog()
    try:
        catalog.index_evidence(evidence)
    finally:
        catalog.close()
    bundle.write_manifest()
    return {
        "trace_id": document.trace_id,
        "trace_digest": document.content_digest,
        "evidence_bundle_id": evidence.bundle_id,
        "evidence_bundle_digest": evidence.content_digest,
        "evaluation_id": typed["evaluation"].evaluation_id,
        "verifier_result_id": (
            typed["verifier_result"].verifier_result_id
            if typed["verifier_result"] is not None
            else None
        ),
        "reward_record_id": (
            typed["reward_record"].reward_record_id
            if typed["reward_record"] is not None
            else None
        ),
        "verdict_id": (
            typed["verdict"].verdict_id if typed["verdict"] is not None else None
        ),
        "native_source_digest": source_digest,
        "artifact_digest": blob_digest,
        "aggregate_score": typed["evaluation"].aggregate_score,
        "validation_valid": validation.valid,
        "idempotent": False,
    }


def _typed_native_records(
    *,
    document: Any,
    payload: Mapping[str, Any],
    artifact: ArtifactRefV5,
    authority: str,
    source_version: str,
    source_digest: str,
    producer: ProducerRefV1,
    produced_at: str,
) -> dict[str, Any]:
    subject = selector_for(document, kind=SelectorKind.TRACE)
    execution_status = _execution_status(payload)
    execution_completed = execution_status == ExecutionStatus.COMPLETED
    verifier_payload = _mapping(payload.get("verifier"))
    rubric_payload = _mapping(payload.get("rubric"))
    judgment_producer = ProducerRefV1(
        kind=str(_verifier_kind(verifier_payload)),
        name=producer.name,
        version=producer.version,
        model=_optional_text(verifier_payload.get("model")) or producer.model,
        config_digest=(
            _optional_text(verifier_payload.get("config_digest"))
            or producer.config_digest
        ),
        credential_profile=producer.credential_profile,
    )
    source_definition_rows = _criterion_rows(
        rubric_payload.get("criteria") or payload.get("criteria")
    )
    native_result_payload = None
    if verifier_payload:
        native_result_payload = (
            verifier_payload.get("criterion_results")
            or verifier_payload.get("criteria")
        )
    declared_result_rows = _criterion_rows(native_result_payload)
    source_result_rows = (
        declared_result_rows
        if execution_completed
        else ()
    )
    has_verifier = bool(
        verifier_payload
        or rubric_payload
        or source_definition_rows
        or declared_result_rows
    )
    verifier_score = (
        _number(verifier_payload.get("score"))
        if verifier_payload and execution_completed
        else None
    )
    threshold = _first_number(
        verifier_payload.get("pass_threshold") if verifier_payload else None,
        rubric_payload.get("pass_threshold") if rubric_payload else None,
        payload.get("pass_threshold"),
        payload.get("threshold"),
    )
    if threshold is None:
        explicit_passed = _first_bool(
            verifier_payload.get("passed") if verifier_payload else None,
            payload.get("passed"),
            payload.get("accepted"),
        )
        # A declared pass with no bar is "finite score counts", not a hidden
        # 0.5 quality gate. Keep 0.5 only when the native payload is silent.
        threshold = 0.0 if explicit_passed is True else 0.5

    criteria: list[CriterionDefinitionV1] = []
    criterion_results: list[JudgmentV1] = []
    rubric: RubricDefinitionV2 | None = None
    verifier_definition: VerifierDefinitionV1 | None = None
    verifier_result: VerifierResultV2 | None = None
    aggregate_gate_id: str | None = None
    verifier_threshold = threshold
    rubric_passed: bool | None = None

    if has_verifier:
        definition_rows = list(source_definition_rows)
        result_rows_by_id = {
            _native_criterion_id(row, index): row
            for index, row in enumerate(source_result_rows)
        }
        if not definition_rows and declared_result_rows:
            definition_rows = list(declared_result_rows)
        for index, row in enumerate(definition_rows):
            native_id = _native_criterion_id(row, index)
            result_row = result_rows_by_id.get(native_id)
            native_role = str(row.get("role") or CriterionRole.REQUIRED)
            role = native_role
            if result_row is None:
                role = CriterionRole.INFORMATIONAL
            criterion = _criterion_definition(
                authority=authority,
                payload=payload,
                row=row,
                native_id=native_id,
                role=role,
                default_threshold=threshold,
            )
            criteria.append(criterion)
            if result_row is not None:
                criterion_results.append(
                    _judgment(
                        document=document,
                        criterion=criterion,
                        row=result_row,
                        subject=subject,
                        producer=judgment_producer,
                        produced_at=produced_at,
                        source_digest=source_digest,
                    )
                )

        if not criterion_results and execution_completed:
            aggregate_gate_id = "native_aggregate_score"
            aggregate_range_values = tuple(
                value
                for value in (0.0, 1.0, threshold, verifier_score)
                if value is not None
            )
            aggregate = _criterion_definition(
                authority=authority,
                payload=payload,
                row={
                    "id": aggregate_gate_id,
                    "name": "Native aggregate verifier score",
                    "description": (
                        "The aggregate score reported by the native evaluator."
                    ),
                    "min_score": min(aggregate_range_values),
                    "max_score": max(aggregate_range_values),
                },
                native_id=aggregate_gate_id,
                role=CriterionRole.GATING,
                default_threshold=threshold,
            )
            criteria.append(aggregate)
            if verifier_score is not None:
                criterion_results.append(
                    _judgment(
                        document=document,
                        criterion=aggregate,
                        row={
                            "criterion_id": aggregate_gate_id,
                            "score": verifier_score,
                            "passed": _passes(verifier_score, aggregate),
                            "verdict": (
                                "pass"
                                if _passes(verifier_score, aggregate)
                                else "fail"
                            ),
                            "rationale": (
                                "Imported from the native verifier aggregate score."
                            ),
                        },
                        subject=subject,
                        producer=judgment_producer,
                        produced_at=produced_at,
                        source_digest=source_digest,
                    )
                )

        if aggregate_gate_id and criteria:
            verifier_threshold = _normalized_criterion_score(
                threshold,
                criteria[-1],
            )
        rubric_key = {
            "task": _task_key(payload),
            "native_rubric": {
                "id": rubric_payload.get("id") if rubric_payload else None,
                "name": rubric_payload.get("name") if rubric_payload else None,
                "criteria": [
                    {
                        "id": item.metadata.get("native_criterion_id"),
                        "digest": item.content_digest,
                    }
                    for item in criteria
                ],
                "threshold": threshold,
            },
        }
        rubric = RubricDefinitionV2(
            rubric_id=record_id(
                "rubric",
                kind="native_evaluation",
                scope=(authority,),
                key=rubric_key,
            ),
            name=str(
                (rubric_payload.get("name") if rubric_payload else None)
                or f"{authority} native rubric"
            ),
            task_family=str(
                payload.get("benchmark_family")
                or payload.get("benchmark")
                or payload.get("suite")
                or "native"
            ),
            criteria=tuple(criteria),
            version=str(
                (rubric_payload.get("version") if rubric_payload else None)
                or source_version
            ),
            benchmark=_optional_text(
                payload.get("benchmark") or payload.get("benchmark_family")
            ),
            aggregation=RubricAggregationV1(
                strategy="weighted_mean",
                pass_threshold=verifier_threshold,
                tie_break="pass_closed",
            ),
            scoring_instructions=str(
                (rubric_payload.get("scoring_instructions") if rubric_payload else None)
                or "Imported native evaluator scoring."
            ),
            metadata={
                "native": True,
                "native_rubric_id": str(
                    (rubric_payload.get("id") if rubric_payload else None) or ""
                ),
            },
        ).sealed()
        if criterion_results:
            verifier_score, rubric_passed, _ = aggregate_rubric_score(
                rubric,
                tuple(criterion_results),
            )
        verifier_definition = VerifierDefinitionV1(
            verifier_id=record_id(
                "verifier",
                kind="native_evaluation",
                scope=(authority,),
                key={
                    "task": _task_key(payload),
                    "rubric_digest": rubric.content_digest,
                },
            ),
            name=str(
                verifier_payload.get("name")
                or verifier_payload.get("id")
                or f"{authority} native verifier"
            ),
            kind=_verifier_kind(verifier_payload),
            rubric_id=rubric.rubric_id,
            rubric_version=rubric.version,
            rubric_digest=rubric.content_digest,
            version=str(verifier_payload.get("version") or source_version),
            program_ref=_optional_text(
                verifier_payload.get("program_ref")
                or verifier_payload.get("program")
            ),
            model=_optional_text(verifier_payload.get("model")),
            requires_citation=False,
            deterministic=bool(verifier_payload.get("deterministic", True)),
            metadata={"native": True},
        ).sealed()
        verification_status = (
            VerificationStatus.VALID
            if execution_status == ExecutionStatus.COMPLETED
            and verifier_score is not None
            else (
                VerificationStatus.INVALID
                if execution_status == ExecutionStatus.FAILED
                else VerificationStatus.INCONCLUSIVE
            )
        )
        explicit_passed = _first_bool(
            verifier_payload.get("passed"),
            verifier_payload.get("accepted"),
        )
        passed = explicit_passed
        if passed is None:
            passed = rubric_passed
        if passed is None and verifier_score is not None:
            passed = verifier_score >= verifier_threshold
        if verification_status != VerificationStatus.VALID:
            passed = None
        verifier_result = VerifierResultV2(
            verifier_result_id=record_id(
                "vresult",
                kind="native_evaluation",
                scope=(document.trace_id, verifier_definition.verifier_id),
                key=source_digest,
            ),
            verifier_id=verifier_definition.verifier_id,
            verifier_version=verifier_definition.version,
            rubric_id=rubric.rubric_id,
            rubric_digest=rubric.content_digest,
            subject=subject,
            execution_status=execution_status,
            verification_status=verification_status,
            grounding=GroundingStatus.SUMMARY_ONLY,
            produced_at=produced_at,
            producer=producer,
            score=verifier_score,
            pass_threshold=verifier_threshold,
            passed=passed,
            verdict=(
                (
                    str(verifier_payload.get("verdict") or "")
                    or ("pass" if passed else "fail" if passed is False else "")
                )
                if verification_status == VerificationStatus.VALID
                else ""
            ),
            criterion_results=tuple(criterion_results),
            artifacts=(artifact,),
            metadata={
                "native": True,
                "native_source_digest": source_digest,
            },
        ).sealed()

    reward_definition: RewardDefinitionV1 | None = None
    reward_record: RewardRecordV1 | None = None
    reward_payload = _mapping(payload.get("reward"))
    reward_scalar = _number(payload.get("reward"))
    reward_present = payload.get("reward") is not None
    reward_value = (
        _first_number(
            reward_payload.get("value"),
            reward_payload.get("score"),
        )
        if reward_payload
        else reward_scalar
    )
    if reward_present:
        reward_name = str(
            reward_payload.get("primary_metric")
            or reward_payload.get("name")
            or reward_payload.get("id")
            or "native_reward"
        )
        components = _numeric_mapping(reward_payload.get("components"))
        reward_kind = _reward_source_kind(reward_payload)
        reward_definition = RewardDefinitionV1(
            reward_id=record_id(
                "reward",
                kind="native_evaluation",
                scope=(authority,),
                key={
                    "task": _task_key(payload),
                    "name": reward_name,
                    "version": reward_payload.get("version") or source_version,
                },
            ),
            name=reward_name,
            intent=str(
                reward_payload.get("intent")
                or f"Native evaluator reward metric {reward_name}."
            ),
            source_kind=reward_kind,
            emission=RewardEmission.POST_HOC,
            subject_scope="trace",
            version=str(reward_payload.get("version") or source_version),
            producer_authority=authority,
            vector_components=tuple(sorted(components)),
            units=str(reward_payload.get("units") or "scalar"),
            lower_bound=_number(
                reward_payload.get("lower_bound")
                if reward_payload.get("lower_bound") is not None
                else reward_payload.get("min")
            ),
            upper_bound=_number(
                reward_payload.get("upper_bound")
                if reward_payload.get("upper_bound") is not None
                else reward_payload.get("max")
            ),
            higher_is_better=bool(reward_payload.get("higher_is_better", True)),
            trainable=bool(reward_payload.get("trainable", False)),
            metadata={
                "native": True,
                "native_metric": reward_name,
            },
        ).sealed()
        if reward_value is not None or components:
            explicit_sources = tuple(
                str(item)
                for item in reward_payload.get("source_result_ids") or ()
            )
            if (
                not explicit_sources
                and verifier_result is not None
                and reward_kind
                in {RewardSourceKind.VERIFIER, RewardSourceKind.COMPOSITE}
            ):
                explicit_sources = (verifier_result.verifier_result_id,)
            reward_record = RewardRecordV1(
                reward_record_id=record_id(
                    "rrecord",
                    kind="native_evaluation",
                    scope=(document.trace_id, reward_definition.reward_id),
                    key=source_digest,
                ),
                reward_id=reward_definition.reward_id,
                reward_version=reward_definition.version,
                reward_digest=reward_definition.content_digest,
                subject=subject,
                value=reward_value,
                provenance=str(reward_payload.get("provenance") or "imported"),
                produced_at=produced_at,
                components=components,
                raw_value=_number(reward_payload.get("raw_value")),
                normalized_value=_number(reward_payload.get("normalized_value")),
                source_result_ids=explicit_sources,
                grounding=GroundingStatus.SUMMARY_ONLY,
                metadata={
                    "native": True,
                    "native_source_digest": source_digest,
                },
            ).sealed()

    score = (
        verifier_score if verifier_score is not None else _score(payload)
    ) if execution_completed else None
    aggregate_source = ""
    if verifier_result is not None and verifier_result.score is not None:
        aggregate_source = verifier_result.verifier_result_id
    elif reward_record is not None and reward_record.value is not None:
        aggregate_source = reward_record.reward_record_id
    evaluation = EvaluationResultV1(
        evaluation_id=record_id(
            "eval",
            kind="native_evaluation",
            scope=(document.trace_id,),
            key={"source_digest": source_digest, "authority": authority},
        ),
        subject=subject,
        evaluator_kind="native",
        execution_status=execution_status,
        produced_at=produced_at,
        producer=producer,
        suite=_optional_text(payload.get("suite")),
        benchmark=_optional_text(
            payload.get("benchmark") or payload.get("benchmark_family")
        ),
        task_id=_optional_text(payload.get("task_id")),
        instance_id=_optional_text(payload.get("instance_id")),
        split=_optional_text(payload.get("split")),
        seed=_integer(payload.get("seed")),
        objective_metrics=_numeric_metrics(
            payload,
            include_score=execution_completed,
        ),
        environment_reward_record_ids=(
            (reward_record.reward_record_id,) if reward_record is not None else ()
        ),
        verifier_result_ids=(
            (verifier_result.verifier_result_id,)
            if verifier_result is not None
            else ()
        ),
        rubric_ids=((rubric.rubric_id,) if rubric is not None else ()),
        aggregate_score=score,
        threshold=(
            verifier_threshold
            if execution_completed and verifier_result is not None
            else None
        ),
        artifacts=(artifact,),
        trace_completeness=str(document.completeness.capture_status),
        error=_optional_text(payload.get("error")),
        metadata={
            "native_schema_version": source_version,
            "native_source_digest": source_digest,
            "trace_correlation_id": str(
                payload.get("trace_correlation_id") or ""
            ),
            "aggregate_score_source": aggregate_source,
        },
    ).sealed()

    decision = (
        _decision(payload, verifier_payload, score, verifier_threshold)
        if execution_completed
        else None
    )
    verdict: BenchmarkVerdictV1 | None = None
    if decision is not None:
        required_gates: tuple[str, ...] = ()
        if aggregate_gate_id and criteria:
            gate = next(
                (
                    criterion.criterion_id
                    for criterion in criteria
                    if criterion.metadata.get("native_criterion_id")
                    == aggregate_gate_id
                ),
                "",
            )
            required_gates = (gate,) if gate else ()
        verdict = BenchmarkVerdictV1(
            verdict_id=record_id(
                "verdict",
                kind="native_evaluation",
                scope=(document.trace_id,),
                key=source_digest,
            ),
            benchmark_authority=authority,
            decision=decision,
            produced_at=produced_at,
            score_source=aggregate_source or evaluation.evaluation_id,
            required_evaluation_ids=(evaluation.evaluation_id,),
            required_gates=required_gates,
            threshold=verifier_threshold if score is not None else None,
            failure_reasons=(
                ()
                if decision == "pass"
                else tuple(
                    str(item)
                    for item in (
                        verifier_payload.get("failure_reasons")
                        or payload.get("failure_reasons")
                        or ()
                    )
                )
            ),
            artifacts=(artifact,),
            metadata={
                "native": True,
                "native_source_digest": source_digest,
            },
        ).sealed()

    records: list[tuple[str, Any]] = []
    records.extend(("criterion", item) for item in criteria)
    if rubric is not None:
        records.append(("rubric", rubric))
    if verifier_definition is not None:
        records.append(("verifier_definition", verifier_definition))
    if reward_definition is not None:
        records.append(("reward_definition", reward_definition))
    records.append(("artifact", artifact))
    if verifier_result is not None:
        records.append(("verifier_result", verifier_result))
    if reward_record is not None:
        records.append(("reward_record", reward_record))
    records.append(("evaluation_result", evaluation))
    if verdict is not None:
        records.append(("benchmark_verdict", verdict))
    return {
        "records": tuple(records),
        "evaluation": evaluation,
        "verifier_result": verifier_result,
        "reward_record": reward_record,
        "verdict": verdict,
    }


def _select_trace(
    candidates: list[InspectedBundle],
    payload: Mapping[str, Any],
) -> InspectedBundle:
    if not candidates:
        raise ValueError("bundle contains no sealed trace")
    correlation = str(payload.get("trace_correlation_id") or "")
    if correlation:
        matches = [
            item
            for item in candidates
            if item.trace.identity.correlation_id == correlation
            or correlation in {alias.value for alias in item.trace.aliases}
        ]
        if len(matches) != 1:
            raise ValueError(
                f"trace_correlation_id {correlation!r} selected {len(matches)} traces"
            )
        return matches[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "native evaluation must declare trace_correlation_id for a multi-trace bundle"
    )


def _existing_attachment(
    evidence: TraceEvidenceBundleV5 | None,
    source_digest: str,
) -> EvaluationResultV1 | None:
    if evidence is None:
        return None
    return next(
        (
            item
            for item in evidence.evaluation_results
            if item.metadata.get("native_source_digest") == source_digest
        ),
        None,
    )


def _new_records(
    evidence: TraceEvidenceBundleV5,
    records: tuple[tuple[str, Any], ...],
) -> tuple[tuple[str, Any], ...]:
    collections = {
        "criterion": (evidence.criteria, "criterion_id"),
        "rubric": (evidence.rubrics, "rubric_id"),
        "verifier_definition": (
            evidence.verifier_definitions,
            "verifier_id",
        ),
        "reward_definition": (evidence.reward_definitions, "reward_id"),
        "artifact": (evidence.artifacts, "artifact_id"),
        "verifier_result": (
            evidence.verifier_results,
            "verifier_result_id",
        ),
        "reward_record": (evidence.reward_records, "reward_record_id"),
        "evaluation_result": (
            evidence.evaluation_results,
            "evaluation_id",
        ),
        "benchmark_verdict": (
            evidence.benchmark_verdicts,
            "verdict_id",
        ),
    }
    additions: list[tuple[str, Any]] = []
    for kind, record in records:
        collection, id_field = collections[kind]
        record_id_value = str(getattr(record, id_field))
        existing = next(
            (
                item
                for item in collection
                if str(getattr(item, id_field)) == record_id_value
            ),
            None,
        )
        if existing is None:
            additions.append((kind, record))
            continue
        if content_digest(existing) != content_digest(record):
            raise ValueError(
                f"native evaluation {kind} id {record_id_value!r} conflicts "
                "with an existing sealed record"
            )
    return tuple(additions)


def _criterion_definition(
    *,
    authority: str,
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    native_id: str,
    role: CriterionRole | str,
    default_threshold: float,
) -> CriterionDefinitionV1:
    minimum = _first_number(row.get("min_score"), row.get("min"))
    maximum = _first_number(row.get("max_score"), row.get("max"))
    minimum = 0.0 if minimum is None else minimum
    maximum = 1.0 if maximum is None else maximum
    threshold = _first_number(row.get("pass_threshold"), row.get("threshold"))
    threshold = default_threshold if threshold is None else threshold
    weight = _first_number(row.get("weight"))
    return CriterionDefinitionV1(
        criterion_id=record_id(
            "criterion",
            kind="native_evaluation",
            scope=(authority, _task_key(payload)),
            key=native_id,
        ),
        name=str(row.get("name") or row.get("title") or native_id),
        requirement=str(
            row.get("requirement")
            or row.get("description")
            or row.get("instructions")
            or row.get("name")
            or native_id
        ),
        version=str(row.get("version") or payload.get("schema_version") or "v1"),
        role=role,
        weight=1.0 if weight is None else weight,
        min_score=minimum,
        max_score=maximum,
        pass_threshold=threshold,
        higher_is_better=bool(row.get("higher_is_better", True)),
        allows_abstention=bool(row.get("allows_abstention", False)),
        allows_not_applicable=bool(row.get("allows_not_applicable", False)),
        requires_citation=False,
        metadata={
            "native": True,
            "native_criterion_id": native_id,
            "native_role": str(row.get("role") or role),
        },
    ).sealed()


def _judgment(
    *,
    document: Any,
    criterion: CriterionDefinitionV1,
    row: Mapping[str, Any],
    subject: Any,
    producer: ProducerRefV1,
    produced_at: str,
    source_digest: str,
) -> JudgmentV1:
    score = _number(row.get("score"))
    passed = _first_bool(row.get("passed"), row.get("accepted"))
    if score is None and passed is not None:
        score = criterion.max_score if passed else criterion.min_score
    if passed is None and score is not None:
        passed = _passes(score, criterion)
    verdict = str(
        row.get("verdict")
        or ("pass" if passed else "fail" if passed is False else "inconclusive")
    )
    return JudgmentV1(
        criterion_id=criterion.criterion_id,
        score=score,
        verdict=verdict,
        passed=passed,
        rationale=str(row.get("rationale") or row.get("reason") or ""),
        failure_modes=tuple(
            str(item) for item in row.get("failure_modes") or ()
        ),
        grounding=GroundingStatus.SUMMARY_ONLY,
        confidence=_number(row.get("confidence")),
        metadata={
            "native_criterion_id": str(
                row.get("criterion_id") or row.get("id") or row.get("name") or ""
            )
        },
        judgment_id=record_id(
            "judgment",
            kind="native_evaluation",
            scope=(document.trace_id, criterion.criterion_id),
            key=source_digest,
        ),
        criterion_version=criterion.version,
        criterion_digest=criterion.content_digest,
        subject=subject,
        status=_judgment_status(row, verdict=verdict, score=score, passed=passed),
        producer=producer,
        produced_at=produced_at,
        revision=1,
        state=RecordState.CURRENT,
        schema_version=JUDGMENT_SCHEMA_VERSION,
    ).sealed()


def _judgment_status(
    row: Mapping[str, Any],
    *,
    verdict: str,
    score: float | None,
    passed: bool | None,
) -> JudgmentStatus | str:
    declared = row.get("status")
    if declared is not None:
        return str(declared).strip().lower()
    normalized = verdict.strip().lower()
    if normalized in {"abstain", "abstained"}:
        return JudgmentStatus.ABSTAINED
    if normalized == "not_applicable":
        return JudgmentStatus.NOT_APPLICABLE
    if normalized == "invalid":
        return JudgmentStatus.INVALID
    if normalized == "inconclusive":
        return JudgmentStatus.INCONCLUSIVE
    if passed is not None or score is not None:
        return JudgmentStatus.DECISIVE
    return JudgmentStatus.INCONCLUSIVE


def _criterion_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, Mapping))
    if not isinstance(value, Mapping):
        return ()
    rows: list[Mapping[str, Any]] = []
    for key, raw in value.items():
        if isinstance(raw, Mapping):
            rows.append({"id": str(key), **dict(raw)})
        elif isinstance(raw, bool):
            rows.append({"id": str(key), "passed": raw})
        elif _number(raw) is not None:
            rows.append({"id": str(key), "score": raw})
    return tuple(rows)


def _native_criterion_id(row: Mapping[str, Any], index: int) -> str:
    return str(
        row.get("criterion_id")
        or row.get("id")
        or row.get("name")
        or f"criterion_{index + 1}"
    )


def _execution_status(payload: Mapping[str, Any]) -> ExecutionStatus:
    verifier = _mapping(payload.get("verifier"))
    if verifier.get("returncode") is not None:
        return (
            ExecutionStatus.COMPLETED
            if int(verifier["returncode"]) == 0
            else ExecutionStatus.FAILED
        )
    status = str(payload.get("status") or "completed").lower()
    if status in {"failed", "error", "invalid"}:
        return ExecutionStatus.FAILED
    if status in {"timed_out", "timeout"}:
        return ExecutionStatus.TIMED_OUT
    if status == "skipped":
        return ExecutionStatus.SKIPPED
    if payload.get("error"):
        return ExecutionStatus.FAILED
    return ExecutionStatus.COMPLETED


def _score(payload: Mapping[str, Any]) -> float | None:
    verifier = _mapping(payload.get("verifier"))
    score = _number(verifier.get("score"))
    if score is not None:
        return score
    reward = _mapping(payload.get("reward"))
    score = _first_number(reward.get("value"), reward.get("score"))
    if score is not None:
        return score
    reward_scalar = _number(payload.get("reward"))
    if reward_scalar is not None:
        return reward_scalar
    return _first_number(payload.get("aggregate_score"), payload.get("score"))


def _numeric_metrics(
    payload: Mapping[str, Any],
    *,
    include_score: bool = True,
) -> dict[str, float]:
    metrics = _numeric_mapping(payload.get("metrics"))
    score = _score(payload) if include_score else None
    if score is not None:
        metrics.setdefault("native_score", score)
    return metrics


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if _number(item) is not None
    }


def _decision(
    payload: Mapping[str, Any],
    verifier: Mapping[str, Any],
    score: float | None,
    threshold: float,
) -> str | None:
    raw = (
        verifier.get("verdict")
        or payload.get("verdict")
        or payload.get("decision")
    )
    if raw is not None:
        normalized = str(raw).strip().lower()
        if normalized in {"pass", "passed", "accept", "accepted", "success"}:
            return "pass"
        if normalized in {"fail", "failed", "reject", "rejected"}:
            return "fail"
        return None
    accepted = _first_bool(
        verifier.get("passed"),
        verifier.get("accepted"),
        payload.get("passed"),
        payload.get("accepted"),
        _mapping(payload.get("heldout_scorecard")).get("accepted"),
    )
    if accepted is not None:
        return "pass" if accepted else "fail"
    if score is not None:
        return "pass" if score >= threshold else "fail"
    return None


def _verifier_kind(payload: Mapping[str, Any]) -> VerifierKind | str:
    kind = str(payload.get("kind") or "").strip().lower()
    if kind in {item.value for item in VerifierKind}:
        return kind
    if payload.get("model"):
        return VerifierKind.MODEL
    return VerifierKind.DETERMINISTIC


def _reward_source_kind(payload: Mapping[str, Any]) -> RewardSourceKind | str:
    kind = str(payload.get("source_kind") or "").strip().lower()
    if kind in {item.value for item in RewardSourceKind}:
        return kind
    return RewardSourceKind.DETERMINISTIC_METRIC


def _task_key(payload: Mapping[str, Any]) -> str:
    return str(
        payload.get("task_id")
        or payload.get("instance_id")
        or payload.get("benchmark_family")
        or payload.get("benchmark")
        or "native"
    )


def _passes(score: float, criterion: CriterionDefinitionV1) -> bool:
    return (
        score >= criterion.pass_threshold
        if criterion.higher_is_better
        else score <= criterion.pass_threshold
    )


def _normalized_criterion_score(
    score: float,
    criterion: CriterionDefinitionV1,
) -> float:
    bounded = min(criterion.max_score, max(criterion.min_score, score))
    normalized = (bounded - criterion.min_score) / (
        criterion.max_score - criterion.min_score
    )
    return normalized if criterion.higher_is_better else 1.0 - normalized


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first_number(*values: Any) -> float | None:
    return next(
        (number for value in values if (number := _number(value)) is not None),
        None,
    )


def _first_bool(*values: Any) -> bool | None:
    return next((value for value in values if isinstance(value, bool)), None)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_text(value: Any) -> str | None:
    text = str(value or "")
    return text or None


__all__ = ["attach_native_evaluation"]
