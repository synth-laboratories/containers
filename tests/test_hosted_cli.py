import json
from unittest.mock import AsyncMock

import pytest

from synth_containers import cli


def client(monkeypatch):
    instance = AsyncMock()
    instance.__aenter__.return_value = instance
    instance.submit.return_value = "r1"
    instance.cancel.return_value = {"status": "cancelled", "stop_confirmed": False}
    monkeypatch.setattr(cli.PoolClient, "from_env", lambda: instance)
    return instance


def test_submit_reuses_explicit_identity_without_printing_request(monkeypatch, tmp_path, capsys):
    target = client(monkeypatch)
    request = tmp_path / "request.json"
    request.write_text('{"seed": 1, "env": {"private": "not-output"}}')
    assert cli.main(["submit", "p1", str(request), "--idempotency-key", "stable"]) == 0
    target.submit.assert_awaited_once_with(
        "p1", {"seed": 1, "env": {"private": "not-output"}, "idempotency_key": "stable"}
    )
    output = capsys.readouterr().out
    assert json.loads(output) == {"rollout_id": "r1", "idempotency_key": "stable"}
    assert "not-output" not in output


@pytest.mark.parametrize("body", ["[]", '{"idempotency_key":"other"}', '{"seed":NaN}'])
def test_invalid_request_never_submits(monkeypatch, tmp_path, body):
    target = client(monkeypatch)
    request = tmp_path / "request.json"
    request.write_text(body)
    with pytest.raises(SystemExit):
        cli.main(["submit", "p1", str(request), "--idempotency-key", "stable"])
    target.submit.assert_not_called()


def test_cancel_preserves_unconfirmed_stop(monkeypatch, capsys):
    client(monkeypatch)
    assert cli.main(["cancel", "r1"]) == 0
    assert json.loads(capsys.readouterr().out)["stop_confirmed"] is False


def test_result_uses_verified_custody_read(monkeypatch, capsys):
    target = client(monkeypatch)
    target.get_result_snapshot.return_value = {"status": "failed", "score": None}
    assert cli.main(["result", "r1"]) == 0
    target.get_result_snapshot.assert_awaited_once_with("r1")
    assert json.loads(capsys.readouterr().out)["score"] is None



def test_deployment_lookup_does_not_repeat_mutation(monkeypatch, capsys):
    target = client(monkeypatch)
    target.find_deployment_operation.return_value = {"state": "recovery_required", "operation_id": "operation"}
    assert cli.main(["deployment-lookup", "pool", "task", "--project-id", "project", "--idempotency-key", "stable"]) == 0
    target.find_deployment_operation.assert_awaited_once_with("pool", "task", project_id="project", idempotency_key="stable")
    target.mutate_deployment.assert_not_called()
    assert json.loads(capsys.readouterr().out)["state"] == "recovery_required"


def test_lease_assign_uses_shared_client_and_prints_receipt(monkeypatch, capsys):
    target = client(monkeypatch)
    target.assign_lease.return_value = {'lease':{'lease_id':'lease-1','execution_substrate':'docker'}}
    assert cli.main(['lease-assign','project','--image-kind','synth_sdk','--substrate','docker',
                     '--idempotency-key','stable','--ttl-seconds','300']) == 0
    target.assign_lease.assert_awaited_once_with(project_id='project',image_kind='synth_sdk',
                                                substrate='docker',idempotency_key='stable',ttl_seconds=300)
    assert json.loads(capsys.readouterr().out)['lease']['lease_id']=='lease-1'


def test_lease_release_does_not_delete_deployment(monkeypatch, capsys):
    target = client(monkeypatch)
    target.release_lease.return_value = {'lease':{'status':'released'}}
    assert cli.main(['lease-release','lease-1']) == 0
    target.release_lease.assert_awaited_once_with('lease-1')
    target.mutate_deployment.assert_not_called()
