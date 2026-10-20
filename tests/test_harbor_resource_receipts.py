"""A native cleanup receipt is independent from a scientific reward."""

import json

import pytest

from synth_containers.harbor_resource_receipts import read_harbor_resource_receipt

OWNER = "synth-harbor-" + "a" * 32
HANDLES = [{"kind": "container", "id": "a" * 64}]


def journal(tmp_path, rows):
    path = tmp_path / "resource-events.jsonl"
    path.write_text(
        "".join(json.dumps({"seq": i, "run_id": OWNER, **r}) + "\n" for i, r in enumerate(rows, 1))
    )
    return path


@pytest.mark.parametrize("provider", ["docker", "daytona"])
def test_confirmed_cleanup_requires_saved_identity_and_is_read_only(tmp_path, provider):
    data = {"handles": HANDLES} if provider == "docker" else {"provider_id": "owned"}
    path = journal(
        tmp_path,
        [
            {"event": "resource.create_requested", "provider": provider},
            {
                "event": "resource.handles_observed"
                if provider == "docker"
                else "resource.created",
                **data,
            },
            {"event": "resource.cleanup_confirmed", **data},
        ],
    )
    before = path.read_bytes()
    receipt = read_harbor_resource_receipt(tmp_path)
    assert receipt["cleanup_status"] == "confirmed"
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "last", ["resource.created", "resource.cleanup_pending", "resource.cleanup_requested"]
)
def test_nonterminal_cleanup_stays_pending(tmp_path, last):
    journal(
        tmp_path, [{"event": "resource.create_requested", "provider": "docker"}, {"event": last}]
    )
    assert read_harbor_resource_receipt(tmp_path)["cleanup_status"] == "pending"


def test_empty_confirmation_is_not_absence_proof(tmp_path):
    journal(
        tmp_path,
        [
            {"event": "resource.create_requested", "provider": "docker"},
            {"event": "resource.cleanup_confirmed", "handles": []},
        ],
    )
    with pytest.raises(ValueError, match="primary handles"):
        read_harbor_resource_receipt(tmp_path)


def test_missing_journal_is_not_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_harbor_resource_receipt(tmp_path)
    assert not list(tmp_path.iterdir())
