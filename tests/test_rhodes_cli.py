"""CLI uses the ordinary SDK wire and never invents replay or cancellation."""

import json
import asyncio

import httpx
import pytest

from synth_containers import rhodes_cli
from synth_containers.cli import main
from synth_containers.rhodes_client import RhodesClient


def install_client(monkeypatch, respond):
    requests = []

    def handler(request):
        requests.append(request)
        return respond(request)

    def factory(**kwargs):
        assert kwargs == {"api_key": "secret-fixture", "backend_url": "http://rhodes.invalid"}
        # Owned client retains the normal bearer header and closure semantics.
        transport = httpx.AsyncClient(
            base_url=kwargs["backend_url"], transport=httpx.MockTransport(handler),
            headers={"Authorization": "Bearer " + kwargs["api_key"]},
        )
        client = RhodesClient(**kwargs, client=transport)
        client._owns_client = True
        return client

    monkeypatch.setenv("SYNTH_API_KEY", "secret-fixture")
    monkeypatch.setenv("SYNTH_BACKEND_URL", "http://rhodes.invalid")
    monkeypatch.setattr(rhodes_cli, "RhodesClient", factory)
    return requests


@pytest.mark.parametrize("suffix,content", [
    (".toml", 'idempotency_key="original"\nseed=17\n'),
    (".json", '{"idempotency_key":"original","seed":17}'),
])
def test_exec_detach_emits_original_id_and_body_once(monkeypatch, capsys, tmp_path, suffix, content):
    path = tmp_path / ("request" + suffix)
    path.write_text(content)
    requests = install_client(monkeypatch, lambda _: httpx.Response(200, json={"rollout_id": "run-17"}))
    assert main(["rhodes", "exec", str(path), "--pool-id", "pool", "--detach"]) == 0
    assert json.loads(capsys.readouterr().out) == {"rollout_id": "run-17"}
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/v1/pools/pool/rollouts"
    assert json.loads(requests[0].content) == {"idempotency_key": "original", "seed": 17}
    assert requests[0].headers["Authorization"] == "Bearer secret-fixture"


def test_follow_keeps_committed_event_and_cursor_without_submit(monkeypatch, capsys):
    event = {"sequence": 8, "rollout_id": "run", "event_type": "completed", "payload": {"reward": 0}}

    def respond(request):
        if request.url.params.get("format") == "sse":
            assert request.headers["Last-Event-ID"] == "7"
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 content="id: 8\ndata: " + json.dumps(event) + "\n\n")
        return httpx.Response(200, json={"status": "completed", "high_water": 8})

    requests = install_client(monkeypatch, respond)
    assert main(["rhodes", "follow", "run", "--after-sequence", "7"]) == 0
    assert json.loads(capsys.readouterr().out) == event
    assert len(requests) == 2 and all(request.method == "GET" for request in requests)


def test_watch_json_preserves_per_rollout_cursors_and_events(monkeypatch, capsys, tmp_path):
    cursors = tmp_path / "cursors.json"
    cursors.write_text('{"a":7,"b":12}')
    expected = []

    def respond(request):
        identity = request.url.path.split("/")[-2]
        sequence = {"a": 8, "b": 13}[identity]
        if request.url.params.get("format") == "sse":
            assert request.headers["Last-Event-ID"] == str(sequence - 1)
            event = {"sequence": sequence, "rollout_id": identity, "score": 0}
            expected.append(event)
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 content=f"id: {sequence}\ndata: " + json.dumps(event) + "\n\n")
        return httpx.Response(200, json={"status": "completed", "high_water": sequence})

    requests = install_client(monkeypatch, respond)
    assert main(["rhodes", "watch", "a", "b", "--cursors", str(cursors), "--format", "json"]) == 0
    emitted = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sorted(emitted, key=lambda event: event["rollout_id"]) == expected
    assert len(requests) == 4 and all(request.method == "GET" for request in requests)


@pytest.mark.parametrize("status,exit_code", [("completed", 0), ("failed", 3), ("cancelled", 3)])
def test_wait_exit_tracks_execution_status_not_zero_reward(monkeypatch, capsys, status, exit_code):
    payload = {"status": status, "reward": 0, "cleanup_pending": True}
    requests = install_client(monkeypatch, lambda _: httpx.Response(200, json=payload))
    assert main(["rhodes", "wait", "run"]) == exit_code
    assert json.loads(capsys.readouterr().out) == payload
    assert len(requests) == 1 and requests[0].method == "GET"


@pytest.mark.parametrize("arguments,method,path,body", [
    (["environments", "get", "release"], "GET", "/v1/environment_releases/release", None),
    (["environments", "archive", "release"], "DELETE", "/v1/environment_releases/release", None),
    (["definitions", "bind", "craftax", "release", "--expected-revision", "2", "--operation-id", "same"],
     "PUT", "/v1/environment_definitions/craftax", {
         "schema_version": "synth.rhodes-definition-bind.v1", "release_id": "release",
         "expected_revision": 2, "operation_id": "same"}),
    (["definitions", "archive", "craftax", "--expected-revision", "3", "--operation-id", "archive"],
     "POST", "/v1/environment_definitions/craftax/archive", {
         "schema_version": "synth.rhodes-definition-archive.v1", "expected_revision": 3,
         "operation_id": "archive"}),
    (["cancel", "run"], "POST", "/v1/rollouts/run/cancel", None),
])
def test_registry_and_explicit_cancel_wire(monkeypatch, capsys, arguments, method, path, body):
    requests = install_client(monkeypatch, lambda _: httpx.Response(200, json={"original": True}))
    assert main(["rhodes", *arguments]) == 0
    assert json.loads(capsys.readouterr().out) == {"original": True}
    assert len(requests) == 1
    assert (requests[0].method, requests[0].url.path) == (method, path)
    assert (json.loads(requests[0].content) if requests[0].content else None) == body


PROJECT = "11111111-2222-4333-8444-555555555555"


@pytest.mark.parametrize("arguments,method,path,params,body", [
    (["pools", "get", "pool"], "GET", "/v1/pools/pool", {}, None),
    (["pools", "delete", "pool"], "DELETE", "/v1/pools/pool", {}, None),
    (["deployments", "get", "pool", "seed-1", "--project-id", PROJECT],
     "GET", "/v1/pools/pool/deployments/seed-1", {"project_id": PROJECT}, None),
    (["deployments", "operation", "pool", "op", "--project-id", PROJECT],
     "GET", "/v1/pools/pool/deployment-operations/op", {"project_id": PROJECT}, None),
    (["deployments", "delete", "pool", "seed-1", "--project-id", PROJECT,
      "--expected-revision", "r1", "--operation-id", "undeploy"],
     "POST", "/v1/pools/pool/deployments/seed-1/operations", {}, {
         "project_id": PROJECT, "operation": "delete", "idempotency_key": "undeploy",
         "expected_revision": "r1", "payload": {}}),
    (["readiness", "pool", "seed-1"], "GET", "/v1/pools/pool/tasks/seed-1/readiness", {}, None),
])
def test_lifecycle_verbs_use_original_wire(monkeypatch, capsys, arguments, method, path, params, body):
    requests = install_client(monkeypatch, lambda _: httpx.Response(200, json={"original": True}))
    assert main(["rhodes", *arguments]) == 0
    assert json.loads(capsys.readouterr().out) == {"original": True}
    assert len(requests) == 1
    assert (requests[0].method, requests[0].url.path) == (method, path)
    assert dict(requests[0].url.params) == params
    assert (json.loads(requests[0].content) if requests[0].content else None) == body


def test_deployment_create_and_pool_create_send_config_unchanged(monkeypatch, capsys, tmp_path):
    requests = install_client(monkeypatch, lambda _: httpx.Response(200, json={"pool_id": "p"}))
    config = tmp_path / "task.json"
    config.write_text('{"provider":"docker","image_ref":"sha256:' + "a" * 64 + '","port":8080}')
    assert main(["rhodes", "deployments", "create", "pool", "seed-1", str(config),
                 "--project-id", PROJECT, "--operation-id", "deploy"]) == 0
    assert json.loads(requests[0].content) == {
        "project_id": PROJECT, "operation": "create", "idempotency_key": "deploy",
        "expected_revision": None,
        "payload": {"provider": "docker", "image_ref": "sha256:" + "a" * 64, "port": 8080}}
    pool = tmp_path / "pool.toml"
    pool.write_text(f'name="lane0"\nbackend="arbitrary"\nproject_id="{PROJECT}"\n')
    assert main(["rhodes", "pools", "create", str(pool)]) == 0
    assert (requests[1].method, requests[1].url.path) == ("POST", "/v1/pools")
    assert json.loads(requests[1].content) == {
        "name": "lane0", "backend": "arbitrary", "project_id": PROJECT}


def test_pool_create_is_never_retried_without_replay_identity(monkeypatch, capsys, tmp_path):
    requests = install_client(monkeypatch, lambda _: httpx.Response(503, json={}))
    pool = tmp_path / "pool.json"
    pool.write_text('{"name":"lane0","backend":"arbitrary"}')
    assert main(["rhodes", "pools", "create", str(pool)]) == 2
    assert len(requests) == 1


def test_observed_lease_acquire_and_release_wire(monkeypatch, capsys):
    def respond(request):
        if request.url.path.endswith("/capacity_leases"):
            return httpx.Response(200, json={"lease": {"lease_id": "L"}, "reused": False})
        return httpx.Response(200, json={"lease": {"lease_id": "L", "status": "released"}})

    requests = install_client(monkeypatch, respond)
    assert main(["rhodes", "leases", "acquire", "pool", "seed-1", "--image-kind", "craftax_eval",
                 "--operation-id", "lease", "--project-id", PROJECT]) == 0
    assert json.loads(capsys.readouterr().out) == {"lease_id": "L", "reused": False}
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/pools/pool/tasks/seed-1/capacity_leases"
    assert body["idempotency_key"] == "lease" and body["project_id"] == PROJECT
    assert body["reset"] == "fresh_instance" and body["slots"] == 1
    assert main(["rhodes", "leases", "release", "L"]) == 0
    assert (requests[1].method, requests[1].url.path) == ("POST", "/v1/container_leases/L/release")


@pytest.mark.parametrize("content", [
    "{}", '{"idempotency_key":" "}', '[]',
    '{"idempotency_key":"a","idempotency_key":"b"}',
    '{"idempotency_key":"a","limit":NaN}',
    '{"idempotency_key":"a","limit":1e999}', " " * (1024 * 1024 + 1),
])
def test_invalid_run_configuration_refuses_before_network(monkeypatch, capsys, tmp_path, content):
    path = tmp_path / "request.json"
    path.write_text(content)
    requests = install_client(monkeypatch, lambda _: pytest.fail("unexpected request"))
    assert main(["rhodes", "exec", str(path), "--pool-id", "pool", "--detach"]) == 2
    assert not requests
    assert capsys.readouterr().out == ""


def test_no_implicit_backend_and_no_credential_in_refusal(monkeypatch, capsys):
    monkeypatch.delenv("SYNTH_BACKEND_URL", raising=False)
    monkeypatch.setattr(rhodes_cli, "RhodesClient", lambda **_: pytest.fail("unexpected client"))
    assert main(["rhodes", "results", "run"]) == 2
    assert "secret" not in capsys.readouterr().err
    requests = install_client(monkeypatch, lambda _: httpx.Response(
        409, json={"detail": {"code": "rhodes.revision_conflict", "secret": "secret-fixture"}}
    ))
    assert main(["rhodes", "definitions", "get", "craftax"]) == 2
    error = capsys.readouterr()
    assert error.out == ""
    assert json.loads(error.err) == {"error": "rhodes.revision_conflict", "http_status": 409}
    assert "secret-fixture" not in error.err
    assert len(requests) == 1


def test_timeout_detaches_without_hidden_cancel(monkeypatch, capsys):
    requests = install_client(monkeypatch, lambda _: httpx.Response(200, json={"status": "running"}))
    assert main(["rhodes", "wait", "run", "--timeout-seconds", "0.01"]) == 4
    assert json.loads(capsys.readouterr().err)["cancelled"] is False
    assert requests and all(request.method == "GET" for request in requests)


def test_attached_exec_retains_id_when_final_read_exceeds_deadline(monkeypatch, capsys, tmp_path):
    config = tmp_path / "run.toml"
    config.write_text('idempotency_key="original"\n')

    async def respond(request):
        if request.method == "POST":
            return httpx.Response(200, json={"rollout_id": "original-run"})
        if request.url.params.get("format") == "sse":
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 content='id: 1\ndata: {"sequence":1}\n\n')
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"status": "completed", "high_water": 1})
        await asyncio.sleep(10)
        pytest.fail("final read must be interrupted by observation deadline")

    requests = install_client(monkeypatch, respond)
    assert main(["rhodes", "exec", str(config), "--pool-id", "pool",
                 "--timeout-seconds", "0.05"]) == 4
    output = capsys.readouterr()
    assert [json.loads(line) for line in output.out.splitlines()] == [
        {"rollout_id": "original-run"}, {"sequence": 1},
    ]
    assert json.loads(output.err)["cancelled"] is False
    assert sum(request.method == "POST" for request in requests) == 1
    assert all(not request.url.path.endswith("/cancel") for request in requests)
