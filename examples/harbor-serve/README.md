# Harbor Serve on Rhodes

Build a Python 3.12 image containing `synth-containers[harbor-serve]`, your Harbor task packages under `/tasks`, and writable `/var/lib/harbor`. Resolve both controller and task images to immutable OCI references through the normal package builder. Do not include Docker credentials, an organization API key, or a Docker socket in the controller image.

Place the built wheel under `dist/`. The included `Dockerfile` consumes it through `SYNTH_WHEEL=dist/<wheel-name>.whl` and a pinned Python 3.12 base through `PYTHON_IMAGE`. `prepare_task.py --image <digest-reference>` writes an oracle smoke task using the supplied immutable task image. Build through the workspace’s admitted image builder, then register the resulting digest. The Dockerfile-specific ignore file admits only this wheel and the packaged task files. Its `CMD` can be replaced by Rhodes’ `runtime_command` without duplicating an image entrypoint.

Use this deployment configuration with the existing Core-owned deployment command, replacing the controller reference with the resolved digest:

```python
from synth_containers.harbor_serve import harbor_services

configuration = {
    "runtime_kind": "image_ref",
    "image_ref": resolved_controller_reference,
    "interface_mode": "service_http",
    "service_auth_mode": "managed",
    "port": 8390,
    "health_path": "/healthz",
    "runtime_command": ["synth-harbor-serve", "--task-root", "/tasks", "--max-concurrent", "4"],
    "services": harbor_services(concurrency=4).model_dump(mode="json"),
}
```

Discover the `harbor`, `harbor-mcp`, and `synth` endpoint URLs through `RhodesClient.deployment_services(pool_id=…, task_id=…)`. Authenticate to those URLs with the caller's Synth credential. Submit native Harbor requests to the `harbor` URL plus `jobs/submit`, or use the generic facade:

```python
from synth_containers import ServiceOperationClient

operations = ServiceOperationClient(authenticated_http_client, synth_endpoint_url)
metadata = await operations.metadata()
receipt = await operations.submit(
    "evaluation-001",
    {"config": {"tasks": [{"path": "/tasks/example"}], "agents": [{"name": "oracle"}]}},
    origin_instance=metadata["origin_instance"],
)
result = await operations.read(
    "evaluation-001",
    origin_instance=receipt["origin_instance"],
    input_sha256=receipt["input_sha256"],
)
```

Retain the receipt to reconnect. Polling timeouts do not cancel work. Use `read(..., cancel=True)` for explicit cancellation and continue observing until cleanup is confirmed. See [the service specification](../../docs/specs/services-v1.md) for supported providers, inference policy and state retention.
