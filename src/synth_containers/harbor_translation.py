"""Static Harbor bundle -> container.toml translation (Reading B authoring adapter).

See docs/specs/harbor-translation-v1.md. Translation never builds an image,
executes a Dockerfile, imports bundle Python, runs shell commands or reads
credentials. It copies only the declared visible build context, emits a
`synth.container-package.v1` manifest, an execution-requirements sidecar, a
generated build script and a translation receipt. Verifier (`tests/`) and
reference (`solution/`) content never enters the package; the receipt records
their digests so a later trusted verifier can bind the exact hidden inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import ValidationError

from .execution_package import (
    EXECUTION_REQUIREMENTS_FILENAME,
    ExecutionPackageError,
    ExecutionPackagePlan,
    plan_execution_package,
)
from .execution_requirements import (
    ExecutionNetwork,
    ExecutionPlatform,
    ExecutionRequirements,
    ExecutionResources,
    GPUKind,
    NetworkMode,
    VerifierIsolation,
)
from .package_files import ContainerPackageError, read_package_file, snapshot_inputs
from .package_models import Digest, Name, PackageFileIdentity, PackageModel, content_digest

__all__ = [
    "HarborDialect",
    "HarborTranslationCause",
    "HarborTranslationCode",
    "HarborTranslationError",
    "HarborTranslationPolicy",
    "HarborTranslationReceipt",
    "HarborTranslationRefusal",
    "load_translation_policy",
    "translate_harbor_bundle",
]

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

POLICY_SCHEMA = "synth.harbor-translation-policy.v1"
RECEIPT_SCHEMA = "synth.harbor-translation-receipt.v1"
RECEIPT_FILENAME = "translation-receipt.json"
CONTEXT_DIR = "context"
CARGO_HOME_DIR = ".synth-cargo-home"
_IMAGE_RECEIPT_FILENAME = "docker_build_receipt.py"
_MAX_TASK_TOML_BYTES = 256 * 1024
_MAX_DOCKERFILE_BYTES = 256 * 1024
_ALLOWED_INSTRUCTIONS = frozenset(
    {"FROM", "RUN", "COPY", "WORKDIR", "ENV", "CMD", "ARG", "LABEL", "USER", "EXPOSE"}
)
_UNSAFE_SHELL = re.compile(r"\|\||[;|`()]|\$\(|>|<")
# Any cargo mention in a RUN routes to the rewrite, which then fails closed.
_CARGO_ANY = re.compile(r"\bcargo\b")


class HarborDialect(StrEnum):
    HARBOR_TASK_1_0 = "harbor-task-1.0"
    REB_TASK_METADATA = "reb-task-metadata"


class HarborTranslationCode(StrEnum):
    BUNDLE_INVALID = "harbor_translation_bundle_invalid"
    DIALECT_UNKNOWN = "harbor_translation_dialect_unknown"
    INPUT_INVALID = "harbor_translation_input_invalid"
    HIDDEN_INPUT_VISIBLE = "harbor_translation_hidden_input_visible"
    VERIFIER_UNDECLARED = "harbor_translation_verifier_undeclared"
    VERIFIER_ISOLATION_REQUIRED = "harbor_translation_verifier_isolation_required"
    DOCKERFILE_UNSUPPORTED = "harbor_translation_dockerfile_unsupported"
    RAW_CARGO_UNSUPPORTED = "harbor_translation_raw_cargo_unsupported"
    POLICY_INVALID = "harbor_translation_policy_invalid"
    POLICY_CONFLICT = "harbor_translation_policy_conflicts_source"
    REQUIREMENT_MISSING = "harbor_translation_requirement_missing"
    OUTPUT_EXISTS = "harbor_translation_output_exists"
    PACKAGE_REFUSED = "harbor_translation_package_refused"


class HarborTranslationStage(StrEnum):
    INSPECT = "harbor_translation_inspect"
    EMIT = "harbor_translation_emit"


class HarborTranslationCause(StrEnum):
    TASK_TOML_MISSING = "task_toml_missing"
    TASK_TOML_INVALID = "task_toml_invalid"
    DIALECT_UNRECOGNIZED = "dialect_unrecognized"
    ENVIRONMENT_MISSING = "environment_build_context_missing"
    INPUT_SYMLINK_OR_SPECIAL = "input_symlink_special_or_escape"
    INPUT_BOUND_EXCEEDED = "input_bound_exceeded"
    HIDDEN_CONTENT_DUPLICATED = "hidden_content_in_visible_context"
    HIDDEN_SPLIT_NAMED = "hidden_split_named_in_visible_context"
    VERIFIER_ENTRYPOINT_MISSING = "verifier_entrypoint_missing"
    VERIFIER_MODE_MISSING = "verifier_mode_missing"
    VERIFIER_MODE_UNKNOWN = "verifier_mode_unknown"
    SHARED_WITH_HIDDEN_SPLITS = "shared_verifier_with_hidden_splits"
    INSTRUCTION_UNSUPPORTED = "dockerfile_instruction_unsupported"
    COPY_SOURCE_INVALID = "dockerfile_copy_source_invalid"
    CMD_SHELL_FORM = "dockerfile_cmd_shell_form"
    CARGO_SHELL_UNPARSEABLE = "cargo_step_shell_unparseable"
    CARGO_SUBCOMMAND_UNSUPPORTED = "cargo_subcommand_unsupported"
    CARGO_INVOCATION_UNRECOGNIZED = "cargo_invocation_unrecognized"
    CARGO_SURVIVED_REWRITE = "cargo_survived_rewrite"
    CARGO_DIRECTORY_UNMAPPED = "cargo_directory_not_a_copied_context_directory"
    POLICY_UNREADABLE = "policy_unreadable"
    POLICY_CONTRACT = "policy_contract_invalid"
    SOURCE_DECLARATION_DIFFERS = "source_declaration_differs_from_policy"
    RESOURCES_UNDECLARED = "resources_undeclared"
    VERIFIER_REQUIREMENT_UNDECLARED = "verifier_requirement_undeclared"
    WALL_TIME_UNDECLARED = "agent_wall_time_undeclared"
    RUNTIME_COMMAND_UNDECLARED = "runtime_command_undeclared"
    OUTPUT_NOT_EMPTY = "output_directory_not_empty"
    PACKAGE_PLAN_REFUSED = "emitted_package_plan_refused"


@dataclass(frozen=True, slots=True)
class HarborTranslationRefusal:
    code: HarborTranslationCode
    stage: HarborTranslationStage
    cause: HarborTranslationCause
    bundle_id: str
    detail: str


class HarborTranslationError(ValueError):
    """Exception transport for a secret-free typed refusal."""

    def __init__(self, refusal: HarborTranslationRefusal) -> None:
        self.refusal = refusal
        super().__init__(refusal.code.value)


def _refuse(
    bundle_id: str,
    code: HarborTranslationCode,
    cause: HarborTranslationCause,
    detail: str = "",
    stage: HarborTranslationStage = HarborTranslationStage.INSPECT,
) -> HarborTranslationError:
    refusal = HarborTranslationRefusal(code, stage, cause, bundle_id, detail)
    # One structured refusal log; the detail names paths/fields, never file contents.
    LOGGER.warning(
        "harbor_translation_refused",
        extra={"harbor_translation_refusal": {k: str(v) for k, v in asdict(refusal).items()}},
    )
    return HarborTranslationError(refusal)


# --------------------------------------------------------------------------- policy


class VerifierPhasePolicy(PackageModel):
    """Verifier-phase facts a bundle does not declare; source declarations win."""

    network: ExecutionNetwork | None = None
    max_wall_seconds: int | None = None
    resources: ExecutionResources | None = None


class HarborTranslationPolicy(PackageModel):
    """Operator translation inputs. They fill only what the source leaves undeclared.

    A policy value that disagrees with a source declaration refuses; a policy
    cannot widen network, verifier isolation, resources or wall time the bundle
    states. Platform and the training-container network are always policy
    facts, because Harbor bundles do not declare them.
    """

    schema_version: Literal["synth.harbor-translation-policy.v1"]
    platform: ExecutionPlatform
    network: ExecutionNetwork
    resources: ExecutionResources | None = None
    runtime_command: tuple[str, ...] | None = None
    package_id: Name | None = None
    verifier: VerifierPhasePolicy = VerifierPhasePolicy()

    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- receipt


class ExcludedInput(PackageModel):
    path: str
    digest: Digest
    size_bytes: int
    reason: Literal[
        "verifier_only",
        "reference_solution",
        "alternate_recipe",
        "translated_recipe",
        "outside_build_context",
    ]


class DockerfileRewrite(PackageModel):
    line: int
    action: Literal["kept", "removed", "replaced"]
    original: str
    replacement: tuple[str, ...]
    reason: str


class PhaseDeclaration(PackageModel):
    name: Literal["build", "agent", "verifier"]
    entrypoint: str
    packaged: bool
    isolation: VerifierIsolation | None = None
    source_isolation_spelling: str | None = None
    network_mode: str | None = None
    timeout_seconds: int | None = None


class HarborTranslationReceipt(PackageModel):
    schema_version: Literal["synth.harbor-translation-receipt.v1"]
    bundle_id: str
    source_dialect: HarborDialect
    source_task_toml_digest: Digest
    source_declarations: dict[str, Any]
    policy_digest: Digest
    phases: tuple[PhaseDeclaration, ...]
    visible_inputs: tuple[PackageFileIdentity, ...]
    excluded_inputs: tuple[ExcludedInput, ...]
    dockerfile_rewrites: tuple[DockerfileRewrite, ...]
    requirement_provenance: dict[str, str]
    package_id: str
    manifest_digest: Digest
    build_input_digest: Digest
    execution_requirements: ExecutionRequirements
    execution_requirements_digest: Digest
    # The verifier phase is preflighted separately; it is never part of the package.
    verifier_requirements: ExecutionRequirements
    verifier_requirements_digest: Digest
    verifier_requirement_provenance: dict[str, str]

    def digest(self) -> str:
        return content_digest(self.model_dump(mode="json"))


# --------------------------------------------------------------------------- source


@dataclass(frozen=True, slots=True)
class _Source:
    root: Path
    bundle_id: str
    dialect: HarborDialect
    task: dict[str, Any]
    task_digest: str


def _load_source(bundle: Path) -> _Source:
    bundle_id = bundle.name
    try:
        raw = read_package_file(bundle, "task.toml", maximum=_MAX_TASK_TOML_BYTES)
    except ContainerPackageError as error:
        raise _refuse(
            bundle_id,
            HarborTranslationCode.BUNDLE_INVALID,
            HarborTranslationCause.TASK_TOML_MISSING,
            str(error),
        ) from None
    try:
        task = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise _refuse(
            bundle_id,
            HarborTranslationCode.BUNDLE_INVALID,
            HarborTranslationCause.TASK_TOML_INVALID,
        ) from None
    if str(task.get("version")) == "1.0" and isinstance(task.get("verifier"), dict):
        dialect = HarborDialect.HARBOR_TASK_1_0
    elif isinstance(task.get("task"), dict) and isinstance(task.get("splits"), dict):
        dialect = HarborDialect.REB_TASK_METADATA
        bundle_id = str(task["task"].get("id") or bundle_id)
    else:
        raise _refuse(
            bundle_id,
            HarborTranslationCode.DIALECT_UNKNOWN,
            HarborTranslationCause.DIALECT_UNRECOGNIZED,
            ",".join(sorted(task)),
        )
    return _Source(bundle, bundle_id, dialect, task, "sha256:" + hashlib.sha256(raw).hexdigest())


def _snapshot(source: _Source, paths: tuple[str, ...]) -> tuple[PackageFileIdentity, ...]:
    present = tuple(
        p for p in paths if (source.root / p).exists() or (source.root / p).is_symlink()
    )
    try:
        files, _ = snapshot_inputs(source.root, present)
    except ContainerPackageError as error:
        cause = (
            HarborTranslationCause.INPUT_BOUND_EXCEEDED
            if "limit" in str(error)
            else HarborTranslationCause.INPUT_SYMLINK_OR_SPECIAL
        )
        raise _refuse(
            source.bundle_id, HarborTranslationCode.INPUT_INVALID, cause, str(error)
        ) from None
    return files


def _hidden_split_names(source: _Source) -> tuple[str, ...]:
    if source.dialect is not HarborDialect.REB_TASK_METADATA:
        return ()
    hidden = source.task["splits"].get("hidden") or ()
    return tuple(sorted(str(name).lower() for name in hidden))


def _hidden_data_files(
    source: _Source, verifier_files: tuple[PackageFileIdentity, ...]
) -> tuple[PackageFileIdentity, ...]:
    """Files whose bytes must never appear in the visible build context."""
    if source.dialect is HarborDialect.HARBOR_TASK_1_0:
        return verifier_files  # Harbor uploads all of tests/ only at verification.
    paths = source.task.get("paths") or {}
    evaluator = str(paths.get("evaluator") or "evaluator")
    heldout_eval = str(paths.get("heldout_eval") or f"{evaluator}/heldout_eval.json")
    roots = (f"tests/{evaluator}/heldout/", f"tests/{heldout_eval}")
    return tuple(
        f for f in verifier_files if any(f.path == r or f.path.startswith(r) for r in roots)
    )


def _check_hidden(
    source: _Source,
    visible: tuple[PackageFileIdentity, ...],
    verifier_files: tuple[PackageFileIdentity, ...],
) -> None:
    hidden_digests = {
        f.digest for f in _hidden_data_files(source, verifier_files) if f.size_bytes > 0
    }
    for entry in visible:
        if entry.size_bytes > 0 and entry.digest in hidden_digests:
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.HIDDEN_INPUT_VISIBLE,
                HarborTranslationCause.HIDDEN_CONTENT_DUPLICATED,
                entry.path,
            )
    names = _hidden_split_names(source)
    for entry in visible:
        parts = [p.lower() for p in entry.path.split("/")]
        stems = {p.rsplit(".", 1)[0] for p in parts} | set(parts)
        if "heldout" in stems or any(name in stems for name in names):
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.HIDDEN_INPUT_VISIBLE,
                HarborTranslationCause.HIDDEN_SPLIT_NAMED,
                entry.path,
            )


# --------------------------------------------------------------------------- dockerfile


@dataclass(frozen=True, slots=True)
class _Instruction:
    line: int
    keyword: str
    body: str
    original: str


def _parse_dockerfile(bundle_id: str, text: str) -> tuple[_Instruction, ...]:
    instructions: list[_Instruction] = []
    buffer: list[str] = []
    start = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if not buffer:
            start = number
        if buffer and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buffer.append(stripped[:-1].strip())
            continue
        buffer.append(stripped)
        joined = " ".join(part for part in buffer if part)
        buffer = []
        keyword, _, body = joined.partition(" ")
        keyword = keyword.upper()
        if keyword not in _ALLOWED_INSTRUCTIONS:
            raise _refuse(
                bundle_id,
                HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
                HarborTranslationCause.INSTRUCTION_UNSUPPORTED,
                f"line {start}: {keyword}",
            )
        instructions.append(_Instruction(start, keyword, body.strip(), joined))
    if buffer:
        raise _refuse(
            bundle_id,
            HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
            HarborTranslationCause.INSTRUCTION_UNSUPPORTED,
            "dangling continuation",
        )
    return tuple(instructions)


def _copy_mapping(bundle_id: str, instruction: _Instruction) -> tuple[tuple[str, str], ...]:
    """Return (context source, image destination) pairs for a context COPY."""
    try:
        words = shlex.split(instruction.body)
    except ValueError:
        words = []
    if any(word.startswith("--") for word in words) or len(words) < 2:
        raise _refuse(
            bundle_id,
            HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
            HarborTranslationCause.COPY_SOURCE_INVALID,
            f"line {instruction.line}",
        )
    *sources, destination = words
    pairs = []
    for raw in sources:
        normalized = raw.rstrip("/") or "."
        if normalized.startswith("/") or ".." in normalized.split("/") or "*" in normalized:
            raise _refuse(
                bundle_id,
                HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
                HarborTranslationCause.COPY_SOURCE_INVALID,
                f"line {instruction.line}: {raw}",
            )
        pairs.append((normalized, destination.rstrip("/") or "/"))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class _CargoFetch:
    context_directory: str


@dataclass(frozen=True, slots=True)
class _Dockerfile:
    text: str
    rewrites: tuple[DockerfileRewrite, ...]
    cargo_fetches: tuple[_CargoFetch, ...]
    command: tuple[str, ...] | None
    working_directory: str


def _rewrite_cargo_run(
    bundle_id: str,
    instruction: _Instruction,
    copies: dict[str, str],
) -> tuple[tuple[str, ...], tuple[_CargoFetch, ...], str]:
    """Map a raw cargo RUN to mbx host fetch plus the remaining non-cargo shell steps.

    Supported: an `&&` chain of `cd <dir>`, plain commands, `cargo fetch` and
    `cargo build` (a compile-cache warm-up). Other cargo subcommands or shell
    control flow refuse rather than being guessed at.
    """
    if instruction.body.startswith("[") or _UNSAFE_SHELL.search(instruction.body):
        raise _refuse(
            bundle_id,
            HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
            HarborTranslationCause.CARGO_SHELL_UNPARSEABLE,
            f"line {instruction.line}",
        )
    kept: list[str] = []
    fetches: list[_CargoFetch] = []
    removed: list[str] = []
    directory: str | None = None
    for segment in (part.strip() for part in instruction.body.split("&&")):
        try:
            words = shlex.split(segment)
        except ValueError:
            raise _refuse(
                bundle_id,
                HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
                HarborTranslationCause.CARGO_SHELL_UNPARSEABLE,
                f"line {instruction.line}",
            ) from None
        if not words:
            continue
        if words[0] == "cd" and len(words) == 2:
            directory = words[1].rstrip("/")
            kept.append(segment)
        elif words[0] == "cargo":
            subcommand = words[1] if len(words) > 1 else ""
            manifest_dir = directory
            if "--manifest-path" in words[:-1]:
                manifest_dir = words[words.index("--manifest-path") + 1].rsplit("/", 1)[0]
            if subcommand == "fetch":
                context_dir = copies.get(manifest_dir or "")
                if context_dir is None:
                    raise _refuse(
                        bundle_id,
                        HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
                        HarborTranslationCause.CARGO_DIRECTORY_UNMAPPED,
                        f"line {instruction.line}: {manifest_dir}",
                    )
                fetches.append(_CargoFetch(context_dir))
            elif subcommand != "build":
                raise _refuse(
                    bundle_id,
                    HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
                    HarborTranslationCause.CARGO_SUBCOMMAND_UNSUPPORTED,
                    f"line {instruction.line}: cargo {subcommand}",
                )
            removed.append(segment)
        elif _CARGO_ANY.search(segment):
            # Prefixed env, absolute paths, wrappers: never passed through verbatim.
            raise _refuse(
                bundle_id,
                HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
                HarborTranslationCause.CARGO_INVOCATION_UNRECOGNIZED,
                f"line {instruction.line}",
            )
        else:
            kept.append(segment)
    lines: list[str] = []
    if any(not s.startswith("cd ") for s in kept):
        lines.append("RUN " + " && ".join(kept))
    if fetches:
        lines.append(f"COPY {CARGO_HOME_DIR}/registry/ /usr/local/cargo/registry/")
    reason = (
        "raw cargo removed from the image build: `cargo fetch` runs on the build host as "
        "`mbx fetch` into the frozen context and is copied into CARGO_HOME; `cargo build` "
        f"was a compile warm-up only. Removed: {removed}"
    )
    return tuple(lines), tuple(fetches), reason


def _translate_dockerfile(source: _Source, text: str) -> _Dockerfile:
    instructions = _parse_dockerfile(source.bundle_id, text)
    copies: dict[str, str] = {}
    rewrites: list[DockerfileRewrite] = []
    lines: list[str] = [
        "# Generated by synth_containers.harbor_translation from the Harbor environment/",
        "# Dockerfile. Build context is ./context; no tests/ or solution/ content.",
    ]
    fetches: list[_CargoFetch] = []
    command: tuple[str, ...] | None = None
    workdir = "/"
    for instruction in instructions:
        if instruction.keyword == "COPY":
            for src, dst in _copy_mapping(source.bundle_id, instruction):
                copies[dst] = src
        elif instruction.keyword == "WORKDIR":
            workdir = instruction.body
        elif instruction.keyword == "CMD":
            try:
                parsed = json.loads(instruction.body)
            except json.JSONDecodeError:
                parsed = None
            if not (
                isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed)
            ):
                raise _refuse(
                    source.bundle_id,
                    HarborTranslationCode.DOCKERFILE_UNSUPPORTED,
                    HarborTranslationCause.CMD_SHELL_FORM,
                    f"line {instruction.line}",
                )
            command = tuple(parsed)
        if instruction.keyword == "RUN" and _CARGO_ANY.search(instruction.body):
            replacement, new_fetches, reason = _rewrite_cargo_run(
                source.bundle_id, instruction, copies
            )
            fetches.extend(new_fetches)
            lines.extend(replacement)
            rewrites.append(
                DockerfileRewrite(
                    line=instruction.line,
                    action="replaced" if replacement else "removed",
                    original=instruction.original,
                    replacement=replacement,
                    reason=reason,
                )
            )
            continue
        lines.append(instruction.original)
    if any(line.startswith("RUN ") and _CARGO_ANY.search(line) for line in lines):
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.RAW_CARGO_UNSUPPORTED,
            HarborTranslationCause.CARGO_SURVIVED_REWRITE,
            "emitted Dockerfile",
        )
    return _Dockerfile(
        text="\n".join(lines) + "\n",
        rewrites=tuple(rewrites),
        cargo_fetches=tuple(fetches),
        command=command,
        working_directory=workdir,
    )


# --------------------------------------------------------------------------- requirements


def _int_seconds(bundle_id: str, value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise _refuse(
            bundle_id,
            HarborTranslationCode.BUNDLE_INVALID,
            HarborTranslationCause.TASK_TOML_INVALID,
            field,
        )
    return int(value)


def _phases_and_isolation(
    source: _Source, verifier_files: tuple[PackageFileIdentity, ...]
) -> tuple[tuple[PhaseDeclaration, ...], VerifierIsolation, str]:
    if not any(f.path == "tests/test.sh" for f in verifier_files):
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.VERIFIER_UNDECLARED,
            HarborTranslationCause.VERIFIER_ENTRYPOINT_MISSING,
            "tests/test.sh",
        )
    if source.dialect is HarborDialect.HARBOR_TASK_1_0:
        verifier = source.task["verifier"]
        spelling = verifier.get("environment_mode")
        mapping = {"separate": VerifierIsolation.ISOLATED, "shared": VerifierIsolation.SHARED}
        if spelling is None:
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.VERIFIER_UNDECLARED,
                HarborTranslationCause.VERIFIER_MODE_MISSING,
                "verifier.environment_mode",
            )
        if spelling not in mapping:
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.VERIFIER_UNDECLARED,
                HarborTranslationCause.VERIFIER_MODE_UNKNOWN,
                str(spelling),
            )
        isolation = mapping[spelling]
        provenance = f"source:verifier.environment_mode={spelling}"
        network = verifier.get("network_mode")
        timeout = verifier.get("timeout_sec")
        agent_timeout = (source.task.get("agent") or {}).get("timeout_sec")
    else:
        # The REB dialect has no verifier mode. tests/test.sh is the trusted held-out
        # scorer; with hidden splits declared it can only run isolated, never shared.
        spelling = None
        isolation = VerifierIsolation.ISOLATED
        provenance = "derived:splits.hidden declared; tests/test.sh is a trusted held-out scorer"
        if not _hidden_split_names(source):
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.VERIFIER_UNDECLARED,
                HarborTranslationCause.VERIFIER_MODE_MISSING,
                "splits.hidden",
            )
        network, timeout = None, None
        agent_timeout = (source.task.get("budgets") or {}).get("agent_wall_time_seconds")
    phases = (
        PhaseDeclaration(name="build", entrypoint="build.sh", packaged=True),
        PhaseDeclaration(
            name="agent",
            entrypoint="instruction.md",
            packaged=True,
            timeout_seconds=(
                None
                if agent_timeout is None
                else _int_seconds(source.bundle_id, agent_timeout, "agent timeout")
            ),
        ),
        PhaseDeclaration(
            name="verifier",
            entrypoint="tests/test.sh",
            packaged=False,
            isolation=isolation,
            source_isolation_spelling=spelling,
            network_mode=None if network is None else str(network),
            timeout_seconds=None
            if timeout is None
            else _int_seconds(source.bundle_id, timeout, "verifier.timeout_sec"),
        ),
    )
    return phases, isolation, provenance


def _requirements(
    source: _Source,
    policy: HarborTranslationPolicy,
    phases: tuple[PhaseDeclaration, ...],
    isolation: VerifierIsolation,
    isolation_provenance: str,
) -> tuple[ExecutionRequirements, dict[str, str]]:
    provenance = {
        "platform": "policy",
        "network": "policy",
        "privileged": "fixed:false (translation never grants privilege)",
        "verifier_isolation": (
            "derived:training-container scores are self-reported/unattested; the held-out "
            f"verifier is a separate phase ({isolation_provenance})"
        ),
    }
    env = source.task.get("environment") or {}
    resources: ExecutionResources | None = None
    if source.dialect is HarborDialect.HARBOR_TASK_1_0 and {"cpus", "memory_mb"} <= set(env):
        gpus = env.get("gpus", 0)
        gpu_kind = GPUKind.NONE
        if gpus:
            # Only an explicit environment GPU type counts; [metadata] never grants one.
            gpu_kind = GPUKind(str(env.get("gpu_type", "")).lower())
        try:
            resources = ExecutionResources(
                cpu_millis=int(env["cpus"] * 1000),
                memory_bytes=int(env["memory_mb"]) * 1024 * 1024,
                disk_bytes=int(env.get("storage_mb", 0)) * 1024 * 1024,
                gpu_kind=gpu_kind,
                gpu_count=int(gpus),
            )
        except (ValueError, ValidationError, TypeError):
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.BUNDLE_INVALID,
                HarborTranslationCause.TASK_TOML_INVALID,
                "environment",
            ) from None
        provenance["resources"] = "source:environment.{cpus,memory_mb,storage_mb,gpus}"
        if policy.resources is not None and policy.resources != resources:
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.POLICY_CONFLICT,
                HarborTranslationCause.SOURCE_DECLARATION_DIFFERS,
                "resources",
            )
    elif policy.resources is not None:
        resources = policy.resources
        provenance["resources"] = "policy"
    if resources is None:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.RESOURCES_UNDECLARED,
            "resources",
        )
    agent = next(p for p in phases if p.name == "agent")
    if agent.timeout_seconds is None:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.WALL_TIME_UNDECLARED,
            "agent wall time",
        )
    provenance["max_wall_seconds"] = (
        "source:agent.timeout_sec"
        if source.dialect is HarborDialect.HARBOR_TASK_1_0
        else "source:budgets.agent_wall_time_seconds"
    )
    try:
        requirements = ExecutionRequirements(
            schema_version="synth.container-execution-requirements.v1",
            platform=policy.platform,
            resources=resources,
            max_wall_seconds=agent.timeout_seconds,
            network=policy.network,
            privileged=False,
            verifier_isolation=VerifierIsolation.SELF_REPORTED,
        )
    except ValidationError:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.WALL_TIME_UNDECLARED,
            "execution requirements contract",
        ) from None
    return requirements, provenance


def _source_or_policy(
    source: _Source, field: str, declared: T | None, policed: T | None, origin: str
) -> tuple[T, str]:
    if declared is not None and policed is not None and declared != policed:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.POLICY_CONFLICT,
            HarborTranslationCause.SOURCE_DECLARATION_DIFFERS,
            f"verifier.{field}",
        )
    if declared is not None:
        return declared, origin
    if policed is None:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.VERIFIER_REQUIREMENT_UNDECLARED,
            f"verifier.{field}",
        )
    return policed, "policy:verifier"


def _verifier_requirements(
    source: _Source,
    policy: HarborTranslationPolicy,
    phases: tuple[PhaseDeclaration, ...],
    training: ExecutionRequirements,
    training_provenance: dict[str, str],
    isolation_provenance: str,
) -> tuple[ExecutionRequirements, dict[str, str]]:
    """Requirements of the isolated/shared verifier phase, checked on their own."""
    verifier = next(p for p in phases if p.name == "verifier")
    assert verifier.isolation is not None
    harbor = source.dialect is HarborDialect.HARBOR_TASK_1_0
    declared_network: ExecutionNetwork | None = None
    if verifier.network_mode == "no-network":
        declared_network = ExecutionNetwork(mode=NetworkMode.NONE, hosts=())
    elif verifier.network_mode is not None:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.VERIFIER_REQUIREMENT_UNDECLARED,
            f"verifier.network_mode={verifier.network_mode} has no exact allow-list",
        )
    network, network_origin = _source_or_policy(
        source, "network", declared_network, policy.verifier.network, "source:verifier.network_mode"
    )
    wall, wall_origin = _source_or_policy(
        source,
        "max_wall_seconds",
        verifier.timeout_seconds,
        policy.verifier.max_wall_seconds,
        "source:verifier.timeout_sec",
    )
    # A Harbor verifier environment inherits the bundle's [environment] declaration.
    declared_resources = (
        training.resources if harbor and training_provenance["resources"] != "policy" else None
    )
    resources, resources_origin = _source_or_policy(
        source,
        "resources",
        declared_resources,
        policy.verifier.resources,
        "source:environment (Harbor verifier inherits it)",
    )
    try:
        requirements = ExecutionRequirements(
            schema_version="synth.container-execution-requirements.v1",
            platform=policy.platform,
            resources=resources,
            max_wall_seconds=wall,
            network=network,
            privileged=False,
            verifier_isolation=verifier.isolation,
        )
    except ValidationError:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.VERIFIER_REQUIREMENT_UNDECLARED,
            "verifier requirements contract",
        ) from None
    return requirements, {
        "platform": "policy",
        "network": network_origin,
        "max_wall_seconds": wall_origin,
        "resources": resources_origin,
        "privileged": "fixed:false (translation never grants privilege)",
        "verifier_isolation": isolation_provenance,
    }


# --------------------------------------------------------------------------- emit


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_array(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_toml_string(v) for v in values) + "]"


def _requirements_toml(requirements: ExecutionRequirements) -> str:
    r = requirements
    return "\n".join(
        [
            f"schema_version = {_toml_string(r.schema_version)}",
            f"platform = {_toml_string(r.platform.value)}",
            f"max_wall_seconds = {r.max_wall_seconds}",
            f"privileged = {'true' if r.privileged else 'false'}",
            f"verifier_isolation = {_toml_string(r.verifier_isolation.value)}",
            "",
            "[resources]",
            f"cpu_millis = {r.resources.cpu_millis}",
            f"memory_bytes = {r.resources.memory_bytes}",
            f"disk_bytes = {r.resources.disk_bytes}",
            f"gpu_kind = {_toml_string(r.resources.gpu_kind.value)}",
            f"gpu_count = {r.resources.gpu_count}",
            "",
            "[network]",
            f"mode = {_toml_string(r.network.mode.value)}",
            f"hosts = {_toml_array(r.network.hosts)}",
            "",
        ]
    )


def _build_script(platform: str, fetches: tuple[_CargoFetch, ...]) -> str:
    fetch_lines = [
        f'CARGO_HOME="$root/{CONTEXT_DIR}/{CARGO_HOME_DIR}" mbx fetch '
        f'--manifest-path "$root/{CONTEXT_DIR}/{fetch.context_directory}/Cargo.toml"'
        for fetch in fetches
    ]
    return "\n".join(
        [
            "#!/bin/sh",
            "# Generated by synth_containers.harbor_translation. Runs on an admitted build",
            "# host only; Rust dependency resolution goes through mbx, never raw cargo.",
            "set -eu",
            'root="$SYNTH_CONTAINER_PACKAGE_ROOT"',
            f"platform={shlex.quote(platform)}",
            *fetch_lines,
            (
                'docker build --platform "$platform" --file "$root/Dockerfile" '
                f'--iidfile "$root/.image-id" "$root/{CONTEXT_DIR}"'
            ),
            (
                f'python3 -I "$root/{_IMAGE_RECEIPT_FILENAME}" "$SYNTH_CONTAINER_BUILD_PLAN" '
                '"$root/.image-id" "$platform" "$SYNTH_CONTAINER_BUILD_OUTPUT"'
            ),
            "",
        ]
    )


def _manifest_toml(
    package_id: str, platform: str, command: tuple[str, ...], workdir: str, inputs: tuple[str, ...]
) -> str:
    return "\n".join(
        [
            'schema_version = "synth.container-package.v1"',
            f"package_id = {_toml_string(package_id)}",
            "",
            "[build]",
            'entrypoint = "build.sh"',
            f"inputs = {_toml_array(inputs)}",
            f"platform = {_toml_string(platform)}",
            "",
            "[runtime]",
            'profile = "command"',
            f"command = {_toml_array(command)}",
            f"working_directory = {_toml_string(workdir)}",
            "",
            "[runtime.readiness]",
            'kind = "process"',
            "",
        ]
    )


def _source_declarations(source: _Source) -> dict[str, Any]:
    keep = (
        ("version", "metadata", "agent", "verifier", "environment", "artifacts")
        if source.dialect is HarborDialect.HARBOR_TASK_1_0
        else ("task", "family", "splits", "paths", "policy", "budgets")
    )
    return {key: source.task[key] for key in keep if key in source.task}


def translate_harbor_bundle(
    bundle: str | Path,
    output: str | Path,
    policy: HarborTranslationPolicy,
) -> tuple[HarborTranslationReceipt, ExecutionPackagePlan]:
    """Statically translate a Harbor-shaped bundle and freeze the emitted package.

    Returns the translation receipt and the execution-package plan of the
    emitted `container.toml`. The output directory must be absent or empty.
    """
    root = Path(bundle).absolute()
    out = Path(output).absolute()
    source = _load_source(root)
    if not (root / "environment").is_dir() or (root / "environment").is_symlink():
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.BUNDLE_INVALID,
            HarborTranslationCause.ENVIRONMENT_MISSING,
            "environment/",
        )
    context_files = _snapshot(source, ("environment",))
    instruction_files = _snapshot(source, ("instruction.md",))
    verifier_files = _snapshot(source, ("tests",))
    solution_files = _snapshot(source, ("solution",))
    # Everything else at the bundle root (e.g. REB's top-level fixtures/) is not
    # Harbor build context; it is accounted for in the receipt, never packaged.
    known = {"task.toml", "instruction.md", "environment", "tests", "solution"}
    outside_files = _snapshot(
        source, tuple(sorted(entry.name for entry in root.iterdir() if entry.name not in known))
    )
    dockerfile_entry = next((f for f in context_files if f.path == "environment/Dockerfile"), None)
    if dockerfile_entry is None:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.BUNDLE_INVALID,
            HarborTranslationCause.ENVIRONMENT_MISSING,
            "environment/Dockerfile",
        )
    alternates = tuple(
        f
        for f in context_files
        if f.path.count("/") == 1
        and f.path.split("/")[1].startswith("Dockerfile")
        and f is not dockerfile_entry
    )
    visible = (
        tuple(f for f in context_files if f not in alternates and f is not dockerfile_entry)
        + instruction_files
    )
    _check_hidden(source, visible, verifier_files)
    phases, isolation, isolation_provenance = _phases_and_isolation(source, verifier_files)
    requirements, provenance = _requirements(
        source, policy, phases, isolation, isolation_provenance
    )
    verifier_requirements, verifier_provenance = _verifier_requirements(
        source, policy, phases, requirements, provenance, isolation_provenance
    )
    dockerfile_bytes = read_package_file(
        root, "environment/Dockerfile", maximum=_MAX_DOCKERFILE_BYTES
    )
    dockerfile = _translate_dockerfile(source, dockerfile_bytes.decode("utf-8"))
    command = dockerfile.command or policy.runtime_command
    if not command:
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.REQUIREMENT_MISSING,
            HarborTranslationCause.RUNTIME_COMMAND_UNDECLARED,
            "CMD",
        )
    if (
        dockerfile.command
        and policy.runtime_command
        and policy.runtime_command != dockerfile.command
    ):
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.POLICY_CONFLICT,
            HarborTranslationCause.SOURCE_DECLARATION_DIFFERS,
            "runtime_command",
        )
    package_id = policy.package_id or source.bundle_id
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.OUTPUT_EXISTS,
            HarborTranslationCause.OUTPUT_NOT_EMPTY,
            str(out),
            HarborTranslationStage.EMIT,
        )
    out.mkdir(parents=True, exist_ok=True)
    # Copy exactly the snapshotted visible bytes, re-verified against their digests.
    for entry in visible:
        data = read_package_file(root, entry.path)
        if "sha256:" + hashlib.sha256(data).hexdigest() != entry.digest:
            raise _refuse(
                source.bundle_id,
                HarborTranslationCode.INPUT_INVALID,
                HarborTranslationCause.INPUT_SYMLINK_OR_SPECIAL,
                f"{entry.path} changed during translation",
                HarborTranslationStage.EMIT,
            )
        relative = (
            CONTEXT_DIR + entry.path[len("environment") :]
            if entry.path.startswith("environment/")
            else entry.path
        )
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(0o755 if entry.executable else 0o644)
    # The translated Dockerfile replaces the source one at the package root.
    (out / "Dockerfile").write_text(dockerfile.text, encoding="utf-8")
    build = out / "build.sh"
    build.write_text(
        _build_script(policy.platform.value, dockerfile.cargo_fetches), encoding="utf-8"
    )
    build.chmod(0o755)
    helper = Path(__file__).with_name(_IMAGE_RECEIPT_FILENAME)
    (out / _IMAGE_RECEIPT_FILENAME).write_bytes(helper.read_bytes())
    (out / EXECUTION_REQUIREMENTS_FILENAME).write_text(
        _requirements_toml(requirements), encoding="utf-8"
    )
    inputs = tuple(
        sorted(
            {"Dockerfile", CONTEXT_DIR, EXECUTION_REQUIREMENTS_FILENAME, _IMAGE_RECEIPT_FILENAME}
            | ({"instruction.md"} if instruction_files else set())
        )
    )
    manifest_text = _manifest_toml(
        package_id, policy.platform.value, command, dockerfile.working_directory, inputs
    )
    (out / "container.toml").write_text(manifest_text, encoding="utf-8")
    try:
        plan = plan_execution_package(out / "container.toml")
    except (ExecutionPackageError, ContainerPackageError) as error:
        detail = (
            error.refusal.code.value if isinstance(error, ExecutionPackageError) else str(error)
        )
        raise _refuse(
            source.bundle_id,
            HarborTranslationCode.PACKAGE_REFUSED,
            HarborTranslationCause.PACKAGE_PLAN_REFUSED,
            detail,
            HarborTranslationStage.EMIT,
        ) from None
    excluded = tuple(
        ExcludedInput(path=f.path, digest=f.digest, size_bytes=f.size_bytes, reason=reason)
        for files, reason in (
            (verifier_files, "verifier_only"),
            (solution_files, "reference_solution"),
            (alternates, "alternate_recipe"),
            (outside_files, "outside_build_context"),
            ((dockerfile_entry,), "translated_recipe"),
        )
        for f in files
    )
    receipt = HarborTranslationReceipt(
        schema_version=RECEIPT_SCHEMA,
        bundle_id=source.bundle_id,
        source_dialect=source.dialect,
        source_task_toml_digest=source.task_digest,
        source_declarations=_source_declarations(source),
        policy_digest=policy.digest(),
        phases=phases,
        visible_inputs=visible,
        excluded_inputs=excluded,
        dockerfile_rewrites=dockerfile.rewrites,
        requirement_provenance=provenance,
        package_id=package_id,
        manifest_digest="sha256:" + hashlib.sha256(manifest_text.encode()).hexdigest(),
        build_input_digest=plan.build_plan.input_digest,
        execution_requirements=requirements,
        execution_requirements_digest=plan.execution_requirements_digest,
        verifier_requirements=verifier_requirements,
        verifier_requirements_digest=verifier_requirements.digest(),
        verifier_requirement_provenance=verifier_provenance,
    )
    # The receipt sits beside the package and is deliberately not a build input.
    (out / RECEIPT_FILENAME).write_text(receipt.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return receipt, plan


def load_translation_policy(path: str | Path) -> HarborTranslationPolicy:
    file = Path(path).absolute()
    try:
        payload = tomllib.loads(
            read_package_file(file.parent, file.name, maximum=64 * 1024).decode()
        )
    except (ContainerPackageError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        raise _refuse(
            file.name,
            HarborTranslationCode.POLICY_INVALID,
            HarborTranslationCause.POLICY_UNREADABLE,
        ) from None
    try:
        return HarborTranslationPolicy.model_validate(payload)
    except ValidationError:
        raise _refuse(
            file.name, HarborTranslationCode.POLICY_INVALID, HarborTranslationCause.POLICY_CONTRACT
        ) from None
