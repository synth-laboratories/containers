"""First-class Harbor source-package and immutable-release contracts.

Harbor is an environment package executed by a provider; it is not a provider
itself. This module deliberately performs no Docker, shell, or package-code
execution. It turns a Harbor task directory into a stable draft, validates the
draft against provider limits, and binds a scored attempt to digest-pinned
agent and verifier images.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "HARBOR_ENVIRONMENT_RELEASE_SCHEMA",
    "HARBOR_PACKAGE_DRAFT_SCHEMA",
    "HarborEnvironmentDraft",
    "HarborEnvironmentError",
    "HarborEnvironmentRelease",
    "HarborFreshnessReceipt",
    "HarborProviderCompatibility",
    "HarborResourceRequest",
    "HarborValidation",
    "inspect_harbor_package",
    "register_harbor_environment",
]

HARBOR_PACKAGE_DRAFT_SCHEMA = "synth.harbor-package-draft.v1"
HARBOR_ENVIRONMENT_RELEASE_SCHEMA = "synth.harbor-environment-release.v1"
_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"[^a-z0-9._-]+")
_ISOLATED_NETWORKS = frozenset({"no-network", "none", "isolated", ""})


class HarborEnvironmentError(ValueError):
    """A stable, secret-free Harbor package or release refusal."""


@dataclass(frozen=True, slots=True)
class HarborResourceRequest:
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    gpus: int

    def as_dict(self) -> dict[str, int | None]:
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "gpus": self.gpus,
        }


@dataclass(frozen=True, slots=True)
class HarborProviderCompatibility:
    provider_id: str
    supports_network_none: bool = True
    supports_separate_verifier: bool = True
    supports_gpus: bool = False
    supports_docker_socket_in_verifier: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "supports_network_none": self.supports_network_none,
            "supports_separate_verifier": self.supports_separate_verifier,
            "supports_gpus": self.supports_gpus,
            "supports_docker_socket_in_verifier": self.supports_docker_socket_in_verifier,
        }


@dataclass(frozen=True, slots=True)
class HarborValidation:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": list(self.errors), "warnings": list(self.warnings)}


@dataclass(frozen=True, slots=True)
class HarborEnvironmentDraft:
    """Static, credential-free projection of a Harbor task directory."""

    root: Path
    package_id: str
    title: str
    description: str
    language: str | None
    source_package_digest: str
    task_tree_digest: str
    instruction_digest: str
    environment_digest: str
    verifier_digest: str
    task_contract_digest: str
    agent_timeout_seconds: float | None
    verifier_timeout_seconds: float | None
    agent_network: str
    verifier_network: str
    verifier_environment_mode: str
    candidate_artifacts: tuple[str, ...]
    collect_commands: tuple[str, ...]
    resource_request: HarborResourceRequest

    def validate(self, provider: HarborProviderCompatibility) -> HarborValidation:
        errors: list[str] = []
        warnings: list[str] = []
        if self.agent_network in _ISOLATED_NETWORKS and not provider.supports_network_none:
            errors.append("harbor_provider_agent_network_unsupported")
        if self.verifier_network in _ISOLATED_NETWORKS and not provider.supports_network_none:
            errors.append("harbor_provider_verifier_network_unsupported")
        if self.verifier_environment_mode == "separate" and not provider.supports_separate_verifier:
            errors.append("harbor_provider_separate_verifier_unsupported")
        if self.resource_request.gpus and not provider.supports_gpus:
            errors.append("harbor_provider_gpu_unsupported")
        if self.verifier_environment_mode == "separate" and not self.candidate_artifacts:
            errors.append("harbor_candidate_artifacts_required_for_separate_verifier")
        if self.verifier_environment_mode not in {"separate", "shared", "same"}:
            errors.append("harbor_verifier_environment_mode_invalid")
        if self.verifier_environment_mode == "separate" and not self.collect_commands:
            warnings.append("harbor_separate_verifier_has_no_collect_commands")
        return HarborValidation(valid=not errors, errors=tuple(errors), warnings=tuple(warnings))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": HARBOR_PACKAGE_DRAFT_SCHEMA,
            "package_id": self.package_id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "source_package_digest": self.source_package_digest,
            "task_tree_digest": self.task_tree_digest,
            "instruction_digest": self.instruction_digest,
            "environment_digest": self.environment_digest,
            "verifier_digest": self.verifier_digest,
            "task_contract_digest": self.task_contract_digest,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "verifier_timeout_seconds": self.verifier_timeout_seconds,
            "agent_network": self.agent_network,
            "verifier_network": self.verifier_network,
            "verifier_environment_mode": self.verifier_environment_mode,
            "candidate_artifacts": list(self.candidate_artifacts),
            "collect_commands": list(self.collect_commands),
            "resource_request": self.resource_request.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class HarborFreshnessReceipt:
    release_id: str
    release_digest: str
    expected_source_package_digest: str
    observed_source_package_digest: str
    fresh: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "synth.harbor-source-freshness.v1",
            "release_id": self.release_id,
            "release_digest": self.release_digest,
            "expected_source_package_digest": self.expected_source_package_digest,
            "observed_source_package_digest": self.observed_source_package_digest,
            "fresh": self.fresh,
        }


@dataclass(frozen=True, slots=True)
class HarborEnvironmentRelease:
    """A scored-run-ready Harbor environment bound to immutable image inputs."""

    draft: HarborEnvironmentDraft
    release_id: str
    release_digest: str
    agent_image: str
    verifier_image: str
    provider: HarborProviderCompatibility
    validation: HarborValidation

    def freshness(self) -> HarborFreshnessReceipt:
        observed = inspect_harbor_package(self.draft.root).source_package_digest
        return HarborFreshnessReceipt(
            release_id=self.release_id,
            release_digest=self.release_digest,
            expected_source_package_digest=self.draft.source_package_digest,
            observed_source_package_digest=observed,
            fresh=observed == self.draft.source_package_digest,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": HARBOR_ENVIRONMENT_RELEASE_SCHEMA,
            "environment_release_id": self.release_id,
            "environment_release_digest": self.release_digest,
            "package": self.draft.as_dict(),
            "agent_image": self.agent_image,
            "verifier_image": self.verifier_image,
            "provider": self.provider.as_dict(),
            "validation": self.validation.as_dict(),
            "freshness": self.freshness().as_dict(),
        }


def inspect_harbor_package(root: str | Path) -> HarborEnvironmentDraft:
    """Parse a Harbor package without executing any of its content."""

    package_root = Path(root).expanduser().resolve()
    task_path = package_root / "task.toml"
    instruction_path = package_root / "instruction.md"
    environment_path = package_root / "environment" / "Dockerfile"
    verifier_path = package_root / "tests" / "test.sh"
    for path, code in (
        (task_path, "harbor_package_task_toml_missing"),
        (instruction_path, "harbor_package_instruction_missing"),
        (environment_path, "harbor_package_environment_missing"),
        (verifier_path, "harbor_package_verifier_missing"),
    ):
        if not path.is_file() or path.is_symlink():
            raise HarborEnvironmentError(code)
    try:
        with task_path.open("rb") as handle:
            manifest = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise HarborEnvironmentError("harbor_package_task_toml_invalid") from exc
    if not isinstance(manifest, dict):
        raise HarborEnvironmentError("harbor_package_task_toml_invalid")

    task = _mapping(manifest.get("task"), "harbor_package_task_missing")
    metadata = _optional_mapping(manifest.get("metadata"))
    agent = _optional_mapping(manifest.get("agent"))
    verifier = _mapping(manifest.get("verifier"), "harbor_package_verifier_contract_missing")
    environment = _mapping(manifest.get("environment"), "harbor_package_environment_contract_missing")
    package_id = _text(metadata.get("task_id") or task.get("name"), "harbor_package_id_missing")
    title = str(metadata.get("display_title") or task.get("name") or package_id).strip()
    description = str(metadata.get("display_description") or task.get("description") or "").strip()
    resource = HarborResourceRequest(
        cpus=_optional_int(environment.get("cpus"), "harbor_package_cpus_invalid"),
        memory_mb=_optional_int(environment.get("memory_mb"), "harbor_package_memory_invalid"),
        storage_mb=_optional_int(environment.get("storage_mb"), "harbor_package_storage_invalid"),
        gpus=_optional_int(environment.get("gpus"), "harbor_package_gpus_invalid") or 0,
    )
    source_digest = _tree_digest(package_root)
    task_tree_digest = _tree_digest(package_root / "tests")
    instruction_digest = _file_digest(instruction_path)
    environment_digest = _tree_digest(package_root / "environment")
    verifier_digest = _tree_digest(package_root / "tests")
    contract = {
        "package_id": package_id,
        "agent_timeout_seconds": _optional_float(agent.get("timeout_sec"), "harbor_package_agent_timeout_invalid"),
        "verifier_timeout_seconds": _optional_float(verifier.get("timeout_sec"), "harbor_package_verifier_timeout_invalid"),
        "agent_network": _network_mode(agent.get("network_mode")),
        "verifier_network": _network_mode(verifier.get("network_mode")),
        "verifier_environment_mode": _text(verifier.get("environment_mode") or "shared", "harbor_package_verifier_environment_mode_invalid").lower(),
        "candidate_artifacts": list(_string_array(manifest.get("artifacts"), "harbor_package_artifacts_invalid")),
        "collect_commands": list(_collect_commands(verifier)),
        "resource_request": resource.as_dict(),
        "source_package_digest": source_digest,
        "task_tree_digest": task_tree_digest,
        "instruction_digest": instruction_digest,
        "environment_digest": environment_digest,
        "verifier_digest": verifier_digest,
    }
    return HarborEnvironmentDraft(
        root=package_root,
        package_id=package_id,
        title=title,
        description=description,
        language=_optional_text(metadata.get("language")),
        source_package_digest=source_digest,
        task_tree_digest=task_tree_digest,
        instruction_digest=instruction_digest,
        environment_digest=environment_digest,
        verifier_digest=verifier_digest,
        task_contract_digest=_canonical_digest(contract),
        agent_timeout_seconds=contract["agent_timeout_seconds"],
        verifier_timeout_seconds=contract["verifier_timeout_seconds"],
        agent_network=contract["agent_network"],
        verifier_network=contract["verifier_network"],
        verifier_environment_mode=contract["verifier_environment_mode"],
        candidate_artifacts=tuple(contract["candidate_artifacts"]),
        collect_commands=tuple(contract["collect_commands"]),
        resource_request=resource,
    )


def register_harbor_environment(
    draft: HarborEnvironmentDraft,
    *,
    agent_image: str,
    verifier_image: str,
    provider: HarborProviderCompatibility,
) -> HarborEnvironmentRelease:
    """Register prebuilt pinned images as a scored Harbor environment release."""

    if not _PINNED_IMAGE.fullmatch(agent_image):
        raise HarborEnvironmentError("harbor_release_agent_image_unpinned")
    if not _PINNED_IMAGE.fullmatch(verifier_image):
        raise HarborEnvironmentError("harbor_release_verifier_image_unpinned")
    validation = draft.validate(provider)
    if not validation.valid:
        raise HarborEnvironmentError(validation.errors[0])
    identity = {
        "schema": HARBOR_ENVIRONMENT_RELEASE_SCHEMA,
        "package_id": draft.package_id,
        "source_package_digest": draft.source_package_digest,
        "task_contract_digest": draft.task_contract_digest,
        "agent_image": agent_image,
        "verifier_image": verifier_image,
        "provider": provider.as_dict(),
    }
    digest = _canonical_digest(identity)
    slug = _IDENTIFIER.sub("-", draft.package_id.lower()).strip("-.") or "environment"
    return HarborEnvironmentRelease(
        draft=draft,
        release_id=f"harbor:{slug}:{digest.removeprefix('sha256:')[:16]}",
        release_digest=digest,
        agent_image=agent_image,
        verifier_image=verifier_image,
        provider=provider,
        validation=validation,
    )


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarborEnvironmentError(code)
    return value


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any, code: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 4_096 or "\x00" in text:
        raise HarborEnvironmentError(code)
    return text


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > 4_096 or "\x00" in text:
        raise HarborEnvironmentError("harbor_package_text_invalid")
    return text


def _optional_float(value: Any, code: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HarborEnvironmentError(code) from exc
    if result <= 0 or result != result:
        raise HarborEnvironmentError(code)
    return result


def _optional_int(value: Any, code: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise HarborEnvironmentError(code)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise HarborEnvironmentError(code) from exc
    if result < 0:
        raise HarborEnvironmentError(code)
    return result


def _string_array(value: Any, code: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise HarborEnvironmentError(code)
    return tuple(_text(item, code) for item in value)


def _collect_commands(verifier: Mapping[str, Any]) -> tuple[str, ...]:
    rows = verifier.get("collect")
    if rows is None:
        return ()
    if not isinstance(rows, list):
        raise HarborEnvironmentError("harbor_package_collect_invalid")
    commands: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise HarborEnvironmentError("harbor_package_collect_invalid")
        commands.append(_text(row.get("command"), "harbor_package_collect_invalid"))
    return tuple(commands)


def _network_mode(value: Any) -> str:
    return str(value or "no-network").strip().lower()


def _file_digest(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise HarborEnvironmentError("harbor_package_unreadable") from exc
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _tree_digest(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise HarborEnvironmentError("harbor_package_tree_missing")
    digest = hashlib.sha256()
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    except OSError as exc:
        raise HarborEnvironmentError("harbor_package_unreadable") from exc
    for path in paths:
        if path.is_symlink():
            raise HarborEnvironmentError("harbor_package_symlink_refused")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        try:
            contents = path.read_bytes()
        except OSError as exc:
            raise HarborEnvironmentError("harbor_package_unreadable") from exc
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return f"sha256:{digest.hexdigest()}"


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
