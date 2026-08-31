"""Register-then-run admission for v0.8 portable candidates.

Registration is intentionally local and in-memory here. Runtime owners may
persist returned receipts, but may not derive identity from a folder, name,
port, or other ambient state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

DOMAIN = b"synth.canonical-json.v1\0"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class AdmissionError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}{': ' + detail if detail else ''}")
        self.code = code


def _validate(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if not (-(2**53) + 1 <= value <= 2**53 - 1):
            raise AdmissionError("canonical_integer_range")
        return
    if isinstance(value, float):
        raise AdmissionError(
            "canonical_non_finite" if not math.isfinite(value) else "canonical_float_forbidden"
        )
    if isinstance(value, list):
        for item in value:
            _validate(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate(item)
        return
    raise AdmissionError("canonical_type_forbidden")


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    keys = sorted(value, key=lambda key: key.encode("utf-8"))
    return "{" + ",".join(f"{_encode(key)}:{_encode(value[key])}" for key in keys) + "}"


def canonical_digest(value: Mapping[str, Any], *, omit_self_digest: bool = False) -> str:
    if omit_self_digest:
        value = {key: item for key, item in value.items() if key != "digest"}
    _validate(value)
    return "sha256:" + hashlib.sha256(DOMAIN + _encode(value).encode()).hexdigest()


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        raise AdmissionError("digest_malformed", field)
    return value


def _exact(value: Mapping[str, Any], required: set[str], optional: set[str] = set()) -> None:
    if missing := required - value.keys():
        raise AdmissionError("field_missing", sorted(missing)[0])
    if unknown := value.keys() - required - optional:
        raise AdmissionError("field_unknown", sorted(unknown)[0])


def _verify(document: Mapping[str, Any], kind: str) -> str:
    if kind == "candidate":
        _exact(
            document,
            {"schema_version", "candidate_id", "kind", "content", "task_contract_digest", "digest"},
            {"producing_run_id", "metadata"},
        )
        if document["schema_version"] != "synth.candidate.v1":
            raise AdmissionError("schema_unsupported")
        _exact(document["content"], {"digest", "media_type", "size_bytes"})
        _digest(document["content"]["digest"], "content.digest")
        if (
            not isinstance(document["content"]["size_bytes"], int)
            or document["content"]["size_bytes"] < 0
        ):
            raise AdmissionError("field_type", "content.size_bytes")
        _digest(document["task_contract_digest"], "task_contract_digest")
    else:
        _exact(
            document,
            {
                "schema_version",
                "task_id",
                "family",
                "revision",
                "runtime",
                "datasets",
                "evaluator",
                "seed_policy",
                "capability_requirements",
                "digest",
            },
            {"metadata"},
        )
        if document["schema_version"] != "synth.task-contract.v1":
            raise AdmissionError("schema_unsupported")
        _exact(document["runtime"], {"image_digest", "entrypoint"})
        _exact(document["evaluator"], {"evaluator_id", "digest"})
        _exact(document["seed_policy"], {"kind", "seeds"})
        _digest(document["runtime"]["image_digest"], "runtime.image_digest")
        _digest(document["evaluator"]["digest"], "evaluator.digest")
        if document["seed_policy"]["kind"] != "explicit":
            raise AdmissionError("seed_policy_invalid")
        seeds = document["seed_policy"]["seeds"]
        if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
            raise AdmissionError("seed_policy_invalid")
        if len(seeds) != len(set(seeds)):
            raise AdmissionError("seed_duplicate")
        for dataset in document["datasets"]:
            _exact(dataset, {"dataset_id", "digest"})
            _digest(dataset["digest"], "datasets.digest")
    offered = _digest(document.get("digest"), "digest")
    if offered != canonical_digest(document, omit_self_digest=True):
        raise AdmissionError("digest_mismatch")
    return offered


def _raw_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _evidence_ref(kind: str, uri: str, admission_digest: str) -> dict[str, str]:
    identity = {"kind": kind, "uri": uri, "admission_digest": admission_digest}
    return {**identity, "digest": canonical_digest(identity)}


@dataclass(frozen=True, slots=True)
class RegistrationReceipt:
    registration_id: str
    candidate_digest: str
    task_contract_digest: str
    candidate_bytes_digest: str
    task_bytes_digest: str
    candidate_content_digest: str
    recorded_at: str
    digest: str


@dataclass(frozen=True, slots=True)
class RunAdmission:
    registration_id: str
    run_id: str
    candidate_digest: str
    task_contract_digest: str
    evaluator_id: str
    seed: int
    execution_ref: dict[str, str]
    evaluation_ref: dict[str, str]
    trajectory_ref: dict[str, str]
    idempotency_key: str
    digest: str


class PortableCandidateRegistry:
    def __init__(self) -> None:
        self._registrations: dict[
            str, tuple[RegistrationReceipt, Mapping[str, Any], Mapping[str, Any]]
        ] = {}
        self._runs: dict[str, RunAdmission] = {}

    def register(
        self,
        *,
        candidate_bytes: bytes,
        candidate_content_bytes: bytes,
        task_bytes: bytes,
        recorded_at: str,
    ) -> RegistrationReceipt:
        try:
            candidate = json.loads(candidate_bytes)
            task = json.loads(task_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdmissionError("document_invalid_json") from exc
        candidate_digest = _verify(candidate, "candidate")
        task_digest = _verify(task, "task")
        if candidate["task_contract_digest"] != task_digest:
            raise AdmissionError("candidate_task_substitution")
        content_digest = _raw_digest(candidate_content_bytes)
        if content_digest != candidate["content"]["digest"]:
            raise AdmissionError("candidate_content_mismatch")
        if len(candidate_content_bytes) != candidate["content"]["size_bytes"]:
            raise AdmissionError("candidate_content_size_mismatch")
        identity = {
            "schema_version": "synth.registration-receipt.v1",
            "candidate_digest": candidate_digest,
            "task_contract_digest": task_digest,
            "candidate_bytes_digest": _raw_digest(candidate_bytes),
            "task_bytes_digest": _raw_digest(task_bytes),
            "candidate_content_digest": content_digest,
            "recorded_at": recorded_at,
        }
        digest = canonical_digest(identity)
        registration_id = "reg_" + digest[7:31]
        receipt = RegistrationReceipt(
            registration_id,
            candidate_digest,
            task_digest,
            identity["candidate_bytes_digest"],
            identity["task_bytes_digest"],
            content_digest,
            recorded_at,
            digest,
        )
        self._registrations[registration_id] = (receipt, candidate, task)
        return receipt

    def admit_run(
        self,
        *,
        registration_id: str,
        candidate_bytes: bytes,
        candidate_content_bytes: bytes,
        task_bytes: bytes,
        run_id: str,
        evaluator_id: str,
        seed: int,
        idempotency_key: str,
    ) -> RunAdmission:
        registration = self._registrations.get(registration_id)
        if registration is None:
            raise AdmissionError("registration_not_found")
        receipt, _candidate, task = registration
        if _raw_digest(candidate_bytes) != receipt.candidate_bytes_digest:
            raise AdmissionError("candidate_bytes_changed")
        if _raw_digest(task_bytes) != receipt.task_bytes_digest:
            raise AdmissionError("task_bytes_changed")
        if _raw_digest(candidate_content_bytes) != receipt.candidate_content_digest:
            raise AdmissionError("candidate_content_changed")
        # Revalidate at the execution boundary; registration is not a TOCTOU waiver.
        if _verify(json.loads(candidate_bytes), "candidate") != receipt.candidate_digest:
            raise AdmissionError("candidate_substitution")
        if _verify(json.loads(task_bytes), "task") != receipt.task_contract_digest:
            raise AdmissionError("task_substitution")
        if evaluator_id != task["evaluator"]["evaluator_id"]:
            raise AdmissionError("evaluator_mismatch")
        if seed not in task["seed_policy"]["seeds"]:
            raise AdmissionError("seed_not_admitted")
        identity = {
            "schema_version": "synth.run-admission.v1",
            "registration_id": registration_id,
            "run_id": run_id,
            "candidate_digest": receipt.candidate_digest,
            "task_contract_digest": receipt.task_contract_digest,
            "evaluator_id": evaluator_id,
            "seed": seed,
            "idempotency_key": idempotency_key,
        }
        digest = canonical_digest(identity)
        admission = RunAdmission(
            registration_id,
            run_id,
            receipt.candidate_digest,
            receipt.task_contract_digest,
            evaluator_id,
            seed,
            _evidence_ref("execution", f"run://{run_id}", digest),
            _evidence_ref("evaluation", f"run://{run_id}/evaluation", digest),
            _evidence_ref("trajectory", f"run://{run_id}/trajectory", digest),
            idempotency_key,
            digest,
        )
        prior = self._runs.get(idempotency_key)
        if prior is not None:
            if prior.digest != admission.digest:
                raise AdmissionError("idempotency_conflict")
            return prior
        self._runs[idempotency_key] = admission
        return admission
