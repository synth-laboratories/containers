"""Service declarations participate in package identity and support negotiation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from synth_containers import (
    check_package_compatibility,
    plan_container_package,
    validate_container_build_output,
)
from synth_containers.package_models import ContainerPackageManifest
from synth_containers.service_contracts import ServiceDeclaration, ServiceEndpoint
from tests.test_container_package import output, package, runtime_support


def service_package(root: Path):
    path = package(root)
    text = path.read_text().replace("synth.container-package.v1", "synth.container-package.v2")
    text = text.replace('profile = "command"', 'profile = "http"').replace(
        'kind = "process"', 'kind = "http"\nport = 8390\npath = "/healthz"'
    )
    text += """
[[services.interfaces]]
name = "synth"
protocol = "synth.service-operations"
version = "1"
port = 8390
[services.execution]
protocol = "synth.service-operations"
version = "1"
interface = "synth"
concurrency = 2
[[services.state]]
name = "data"
mount_path = "/var/lib/service"
"""
    path.write_text(text)
    return path


def test_version_two_is_explicit_and_changes_are_digest_bound(tmp_path):
    path = service_package(tmp_path / "source")
    plan = plan_container_package(path)
    artifact = validate_container_build_output(plan, output(plan))
    assert plan.schema_version == "synth.container-build-plan.v2"
    assert artifact.schema_version == "synth.container-artifact.v2"
    with pytest.raises(ValidationError):
        ContainerPackageManifest.model_validate(plan.manifest.model_dump(mode="json"))
    support = runtime_support(runtime_profiles=["http"], readiness_kinds=["http"])
    report = check_package_compatibility(artifact, support)
    assert {issue.code for issue in report.issues} == {
        "service_protocol_unsupported",
        "persistent_state_unsupported",
    }
    support = support.model_copy(
        update={
            "service_protocols": ({"name": "synth.service-operations", "version": "1"},),
            "persistent_single_writer": True,
        }
    )
    assert check_package_compatibility(artifact, support).declaration_supported
    path.write_text(path.read_text().replace("concurrency = 2", "concurrency = 3"))
    assert plan_container_package(path).input_digest != plan.input_digest


def test_endpoint_urls_allow_routed_roots_and_refuse_traversal():
    endpoint = ServiceEndpoint(
        name="mcp", protocol="harbor.mcp", version="0.26.0", url="https://rhodes.test/services/mcp/"
    )
    assert endpoint.url.endswith("/")
    for path in ("/%2e%2e/private", "/a/../private", "//private", "/a?token=x"):
        with pytest.raises(ValidationError):
            ServiceDeclaration.model_validate(
                {
                    "interfaces": [
                        {
                            "name": "native",
                            "protocol": "http",
                            "version": "1",
                            "port": 8000,
                            "base_path": path,
                        }
                    ]
                }
            )
