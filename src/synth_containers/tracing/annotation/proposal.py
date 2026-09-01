"""The structured output an annotator returns: ``synth.annotation-proposal.v1``.

A proposal is *not* evidence. It becomes ``AnnotationV1`` / ``VerifierResultV2``
records only after every selector it cites has been resolved against the sealed
source trace. Hidden reasoning is never part of this contract; ``rationale`` is a
concise retained explanation the annotator is allowed to show.
"""

from __future__ import annotations

import json
from typing import Any

PROPOSAL_SCHEMA_VERSION = "synth.annotation-proposal.v1"

_SELECTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {
            "type": "string",
            "enum": [
                "trace",
                "actor",
                "session",
                "branch",
                "span",
                "event",
                "message",
                "part",
                "artifact",
                "actor_group",
                "interaction",
                "context_epoch",
                "joint_turn",
            ],
        },
        "entity_id": {"type": ["string", "null"]},
        "part_id": {"type": ["string", "null"]},
        "json_pointer": {"type": ["string", "null"]},
        "quote": {"type": ["string", "null"]},
        "range": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "start": {"type": "integer", "minimum": 0},
                "end": {"type": "integer", "minimum": 0},
                "unit": {"type": "string", "enum": ["character", "token"]},
            },
            "required": ["start", "end"],
        },
        "token_sequence": {"type": ["string", "null"], "enum": ["prompt", "completion", None]},
    },
    "required": ["kind"],
}

_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target": _SELECTOR_SCHEMA,
        "annotation_type": {"type": "string", "minLength": 1},
        "labels": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "payload": {"type": "object"},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 2000},
        "evidence": {"type": "array", "items": _SELECTOR_SCHEMA},
        "source_annotation_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target", "annotation_type", "labels", "evidence"],
}

_ABSTENTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target": {"anyOf": [_SELECTOR_SCHEMA, {"type": "null"}]},
        "annotation_type": {"type": "string", "minLength": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
        "requirement": {"type": "string", "maxLength": 500},
        "attempted_selector": {"anyOf": [_SELECTOR_SCHEMA, {"type": "null"}]},
    },
    "required": ["annotation_type", "reason"],
}

_JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "criterion_id": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": ["decisive", "abstained", "not_applicable", "inconclusive"],
        },
        "score": {"type": ["number", "null"]},
        "verdict": {"type": "string", "maxLength": 200},
        "rationale": {"type": "string", "maxLength": 2000},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "array", "items": _SELECTOR_SCHEMA},
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    },
    "required": ["criterion_id", "status", "score", "verdict", "evidence"],
}

PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": PROPOSAL_SCHEMA_VERSION,
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "const": PROPOSAL_SCHEMA_VERSION},
        "source_trace_id": {"type": "string"},
        "source_trace_digest": {"type": "string"},
        "findings": {"type": "array", "items": _FINDING_SCHEMA},
        "abstentions": {"type": "array", "items": _ABSTENTION_SCHEMA},
        "judgments": {"type": "array", "items": _JUDGMENT_SCHEMA},
        "summary": {"type": "string", "maxLength": 4000},
    },
    "required": ["schema_version", "source_trace_id", "source_trace_digest", "findings", "abstentions"],
}


_STRICT_SELECTOR: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": _SELECTOR_SCHEMA["properties"]["kind"],
        "entity_id": {"type": ["string", "null"]},
        "part_id": {"type": ["string", "null"]},
        "json_pointer": {"type": ["string", "null"]},
        "quote": {"type": ["string", "null"]},
        "range": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "unit": {"type": "string", "enum": ["character", "token"]},
                    },
                    "required": ["start", "end", "unit"],
                },
                {"type": "null"},
            ]
        },
        "token_sequence": {"type": ["string", "null"], "enum": ["prompt", "completion", None]},
    },
    "required": ["kind", "entity_id", "part_id", "json_pointer", "quote", "range", "token_sequence"],
}

STRICT_PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "enum": [PROPOSAL_SCHEMA_VERSION]},
        "source_trace_id": {"type": "string"},
        "source_trace_digest": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target": _STRICT_SELECTOR,
                    "annotation_type": {"type": "string"},
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "payload_json": {"type": "string", "description": "JSON object encoded as a string; the payload fields the taxonomy declares"},
                    "confidence": {"type": ["number", "null"]},
                    "rationale": {"type": "string"},
                    "evidence": {"type": "array", "items": _STRICT_SELECTOR},
                    "source_annotation_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["target", "annotation_type", "labels", "payload_json", "confidence", "rationale", "evidence", "source_annotation_ids"],
            },
        },
        "abstentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target": {"anyOf": [_STRICT_SELECTOR, {"type": "null"}]},
                    "annotation_type": {"type": "string"},
                    "reason": {"type": "string"},
                    "requirement": {"type": "string"},
                    "attempted_selector": {"anyOf": [_STRICT_SELECTOR, {"type": "null"}]},
                },
                "required": ["target", "annotation_type", "reason", "requirement", "attempted_selector"],
            },
        },
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "criterion_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["decisive", "abstained", "not_applicable", "inconclusive"]},
                    "score": {"type": ["number", "null"]},
                    "verdict": {"type": "string"},
                    "rationale": {"type": "string"},
                    "failure_modes": {"type": "array", "items": {"type": "string"}},
                    "evidence": {"type": "array", "items": _STRICT_SELECTOR},
                    "confidence": {"type": ["number", "null"]},
                },
                "required": ["criterion_id", "status", "score", "verdict", "rationale", "failure_modes", "evidence", "confidence"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["schema_version", "source_trace_id", "source_trace_digest", "findings", "abstentions", "judgments", "summary"],
}


def normalize_strict_proposal(payload: Any) -> Any:
    """Turn a strict-schema reply into the canonical proposal shape.

    Strict structured output cannot express a free-form object, so the payload
    travels as ``payload_json``; nullable optionals are dropped so the canonical
    shape check sees the same thing a deterministic program would emit.
    """

    if not isinstance(payload, dict):
        return payload

    def clean_selector(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return {key: item for key, item in value.items() if item is not None}

    out = dict(payload)
    findings = []
    for finding in out.get("findings") or []:
        if not isinstance(finding, dict):
            findings.append(finding)
            continue
        item = dict(finding)
        if "payload_json" in item:
            raw = item.pop("payload_json")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
            except ValueError:
                parsed = {"_invalid_payload_json": raw}
            item["payload"] = (
                {key: value for key, value in parsed.items() if value is not None}
                if isinstance(parsed, dict)
                else {"_invalid_payload_json": raw}
            )
        item["target"] = clean_selector(item.get("target"))
        item["evidence"] = [clean_selector(sel) for sel in item.get("evidence") or []]
        if item.get("confidence") is None:
            item.pop("confidence", None)
        if not item.get("source_annotation_ids"):
            item.pop("source_annotation_ids", None)
        findings.append(item)
    out["findings"] = findings
    abstentions = []
    for abstention in out.get("abstentions") or []:
        if not isinstance(abstention, dict):
            abstentions.append(abstention)
            continue
        item = {key: value for key, value in abstention.items() if value is not None}
        if "target" in item:
            item["target"] = clean_selector(item["target"])
        if "attempted_selector" in item:
            item["attempted_selector"] = clean_selector(item["attempted_selector"])
        abstentions.append(item)
    out["abstentions"] = abstentions
    judgments = []
    for judgment in out.get("judgments") or []:
        if not isinstance(judgment, dict):
            judgments.append(judgment)
            continue
        item = dict(judgment)
        item["evidence"] = [clean_selector(sel) for sel in item.get("evidence") or []]
        if item.get("confidence") is None:
            item.pop("confidence", None)
        judgments.append(item)
    out["judgments"] = judgments
    return out


class ProposalShapeError(ValueError):
    """The proposal is not a ``synth.annotation-proposal.v1`` object."""


def _is_selector(value: Any, path: str, problems: list[str]) -> None:
    if not isinstance(value, dict):
        problems.append(f"{path}: selector must be an object")
        return
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind:
        problems.append(f"{path}.kind: required string")
    for key in ("entity_id", "part_id", "json_pointer", "quote", "token_sequence"):
        item = value.get(key)
        if item is not None and not isinstance(item, str):
            problems.append(f"{path}.{key}: must be a string")
    rng = value.get("range")
    if rng is not None:
        if not isinstance(rng, dict):
            problems.append(f"{path}.range: must be an object")
        else:
            for key in ("start", "end"):
                if not isinstance(rng.get(key), int) or isinstance(rng.get(key), bool):
                    problems.append(f"{path}.range.{key}: must be an integer")
    unknown = set(value) - set(_SELECTOR_SCHEMA["properties"])
    if unknown:
        problems.append(f"{path}: unknown selector fields {sorted(unknown)}")


def check_proposal_shape(payload: Any) -> list[str]:
    """Structural check without a JSON Schema dependency; empty list means OK."""

    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["proposal must be a JSON object"]
    if payload.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        problems.append(f"schema_version must be {PROPOSAL_SCHEMA_VERSION!r}")
    for key in ("source_trace_id", "source_trace_digest"):
        if not isinstance(payload.get(key), str) or not payload.get(key):
            problems.append(f"{key}: required string")
    unknown = set(payload) - set(PROPOSAL_JSON_SCHEMA["properties"])
    if unknown:
        problems.append(f"unknown top-level fields {sorted(unknown)}")
    findings = payload.get("findings")
    if not isinstance(findings, list):
        problems.append("findings: required array")
        findings = []
    for index, finding in enumerate(findings):
        path = f"findings[{index}]"
        if not isinstance(finding, dict):
            problems.append(f"{path}: must be an object")
            continue
        unknown = set(finding) - set(_FINDING_SCHEMA["properties"])
        if unknown:
            problems.append(f"{path}: unknown fields {sorted(unknown)}")
        _is_selector(finding.get("target"), f"{path}.target", problems)
        if not isinstance(finding.get("annotation_type"), str) or not finding.get("annotation_type"):
            problems.append(f"{path}.annotation_type: required string")
        labels = finding.get("labels")
        if not isinstance(labels, list) or any(not isinstance(item, str) or not item for item in labels):
            problems.append(f"{path}.labels: required array of non-empty strings")
        evidence = finding.get("evidence")
        if not isinstance(evidence, list):
            problems.append(f"{path}.evidence: required array")
        else:
            for eindex, selector in enumerate(evidence):
                _is_selector(selector, f"{path}.evidence[{eindex}]", problems)
        payload_value = finding.get("payload", {})
        if payload_value is not None and not isinstance(payload_value, dict):
            problems.append(f"{path}.payload: must be an object")
        confidence = finding.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        ):
            problems.append(f"{path}.confidence: must be a number or null")
        rationale = finding.get("rationale", "")
        if not isinstance(rationale, str):
            problems.append(f"{path}.rationale: must be a string")
        elif len(rationale) > 2000:
            problems.append(f"{path}.rationale: exceeds 2000 characters")
        sources = finding.get("source_annotation_ids", [])
        if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
            problems.append(f"{path}.source_annotation_ids: must be an array of strings")
    abstentions = payload.get("abstentions")
    if not isinstance(abstentions, list):
        problems.append("abstentions: required array")
        abstentions = []
    for index, abstention in enumerate(abstentions):
        path = f"abstentions[{index}]"
        if not isinstance(abstention, dict):
            problems.append(f"{path}: must be an object")
            continue
        unknown = set(abstention) - set(_ABSTENTION_SCHEMA["properties"])
        if unknown:
            problems.append(f"{path}: unknown fields {sorted(unknown)}")
        if not isinstance(abstention.get("annotation_type"), str) or not abstention.get("annotation_type"):
            problems.append(f"{path}.annotation_type: required string")
        if not isinstance(abstention.get("reason"), str) or not abstention.get("reason"):
            problems.append(f"{path}.reason: required string")
        if abstention.get("target") is not None:
            _is_selector(abstention.get("target"), f"{path}.target", problems)
        if abstention.get("attempted_selector") is not None:
            _is_selector(abstention.get("attempted_selector"), f"{path}.attempted_selector", problems)
    judgments = payload.get("judgments", [])
    if judgments is None:
        judgments = []
    if not isinstance(judgments, list):
        problems.append("judgments: must be an array")
        judgments = []
    for index, judgment in enumerate(judgments):
        path = f"judgments[{index}]"
        if not isinstance(judgment, dict):
            problems.append(f"{path}: must be an object")
            continue
        unknown = set(judgment) - set(_JUDGMENT_SCHEMA["properties"])
        if unknown:
            problems.append(f"{path}: unknown fields {sorted(unknown)}")
        if not isinstance(judgment.get("criterion_id"), str) or not judgment.get("criterion_id"):
            problems.append(f"{path}.criterion_id: required string")
        if judgment.get("status") not in {"decisive", "abstained", "not_applicable", "inconclusive"}:
            problems.append(f"{path}.status: unsupported")
        score = judgment.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            problems.append(f"{path}.score: must be a number or null")
        evidence = judgment.get("evidence")
        if not isinstance(evidence, list):
            problems.append(f"{path}.evidence: required array")
        else:
            for eindex, selector in enumerate(evidence):
                _is_selector(selector, f"{path}.evidence[{eindex}]", problems)
    summary = payload.get("summary", "")
    if not isinstance(summary, str):
        problems.append("summary: must be a string")
    return problems


def empty_proposal(*, trace_id: str, trace_digest: str) -> dict[str, Any]:
    return {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "source_trace_id": trace_id,
        "source_trace_digest": trace_digest,
        "findings": [],
        "abstentions": [],
        "judgments": [],
        "summary": "",
    }


__all__ = [
    "PROPOSAL_JSON_SCHEMA",
    "STRICT_PROPOSAL_JSON_SCHEMA",
    "normalize_strict_proposal",
    "PROPOSAL_SCHEMA_VERSION",
    "ProposalShapeError",
    "check_proposal_shape",
    "empty_proposal",
]
