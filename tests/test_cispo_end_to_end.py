"""The whole declared CISPO surface, over HTTP, against a real installed target.

Every other CISPO test in this repo drives one module. This one drives the
container: a FastAPI app built by ``create_reference_app`` over a runtime with
:class:`~synth_containers.cispo_target.CispoReferenceTarget` installed, called
through the seventeen declared routes and nothing else. Metadata, capabilities,
handshake, probe, bind, submit, poll, finalize, trace, reward, terminate --
in that order, once, as an executor would.

The point of the ordering is that each step is *only* reachable because the one
before it succeeded: an attempt with no admitted agreement is refused, a reward
before the horizon attestation is refused, and a trace on an attempt that
sealed nothing is refused. A surface that answers all eleven in sequence is a
surface something actually ran behind.

The last three tests hand what came off the wire to the optimizer's own
validators -- ``synth_optimizers.contracts.rl_records`` and
``synth_optimizers.rl.probe`` -- rather than restating their rules here.
``optimizer_module`` is the single place that decides whether those are
reachable, and ``test_the_optimizer_validators_were_reachable`` says which path
this run took.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from types import SimpleNamespace

from synth_containers.cispo_contract import (
    CISPO_OPTIMIZER_CONTRACT_VERSION,
    cispo_declared_routes,
)
from synth_containers.cispo_evidence import EvidenceError
from synth_containers.cispo_target import CispoTargetError
from synth_containers.cispo_probe import (
    PROBE_ROLLOUT_PREFIX,
    PROBE_TOKEN_CAPTURE_PROVENANCE,
    REQUIRED_PROBE_OPERATIONS,
    run_probe,
)
from synth_containers.http_adapter import create_reference_app

from tests.test_cispo_target import (
    admitted,
    bind_request,
    handshake_request,
    installed,
    submit_request,
)

ROUTES = cispo_declared_routes()
ROLLOUT_ID = "rollout_e2e"


# --------------------------------------------------------------------------- #
# The optimizer's own contract, when it is reachable from this worktree
# --------------------------------------------------------------------------- #


def _optimizer_src() -> Path | None:
    """``SYNTH_OPTIMIZERS_SRC`` first, then a sibling ``optimizers`` checkout."""

    configured = os.environ.get("SYNTH_OPTIMIZERS_SRC")
    if configured and (Path(configured) / "synth_optimizers").is_dir():
        return Path(configured)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "optimizers" / "src" / "synth_optimizers"
        if candidate.is_dir():
            return candidate.parent
    return None


def optimizer_module(name: str) -> Any | None:
    """Import one optimizer module, or ``None`` when it is not reachable."""

    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    source = _optimizer_src()
    if source is None:
        return None
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - a broken sibling checkout
        return None


OPTIMIZER_RECORDS = optimizer_module("synth_optimizers.contracts.rl_records")
OPTIMIZER_PROBE = optimizer_module("synth_optimizers.rl.probe")


# --------------------------------------------------------------------------- #
# One run of the whole surface, shared by everything below
# --------------------------------------------------------------------------- #


class Run:
    """One container, one agreement, one attempt, and every response it gave."""

    def __init__(self) -> None:
        self.runtime, self.target = installed(target_count=2)
        self.client = TestClient(create_reference_app(self.runtime))
        self.responses: dict[str, Any] = {}

    def json(self, name: str) -> Any:
        return self.responses[name]


@pytest.fixture(scope="module")
def run() -> Run:
    """Drive the eleven declared steps once, in order, and keep every answer.

    Module-scoped on purpose: the ordering *is* the assertion, and re-running
    the sequence per test would let a later step pass against a fresher
    container than the one the earlier step admitted.
    """

    state = Run()
    client, target = state.client, state.target

    # 1. metadata -- the advertisement an executor discovers the container by.
    state.responses["metadata"] = client.get("/metadata").json()

    # 2. capabilities -- the hashed document every later step is bound to.
    capabilities = client.get(ROUTES["capabilities_route"])
    assert capabilities.status_code == 200
    state.responses["capabilities"] = capabilities.json()

    # 3. health, on the CISPO port rather than the legacy one.
    state.responses["health"] = client.get(ROUTES["health_route"]).json()

    # 4. discovery -- the taskset and the topology the container declares.
    state.responses["taskset"] = client.get(ROUTES["taskset_route"]).json()
    task_id = target.facts.discovery.rows[0].task_id
    state.responses["rows"] = client.post(
        ROUTES["taskset_tasks_route"], json={"ids": [task_id]}
    ).json()
    state.responses["topology"] = client.get(
        ROUTES["topology_route"].format(topology_id=target.facts.topology.topology_id)
    ).json()

    # 5. handshake -- clause by clause, before any spend.
    handshake = client.post(
        ROUTES["handshake_route"], json=handshake_request(target)
    ).json()
    assert handshake["accepted"], handshake["rejected_mandatory_clauses"]
    state.responses["handshake"] = handshake

    # 6. probe -- the unpaid binding kind, bound over the declared route.
    state.responses["probe_binding"] = client.post(
        ROUTES["policy_bind_route"],
        json={
            "handshake_id": handshake["handshake_id"],
            "agreement_digest": handshake["agreement_digest"],
            "kind": "probe",
            "model_family": "family",
            "model_id": "family/model",
        },
    ).json()

    # 7. bind -- one session-scoped sampler for the trainable instance.
    binding = client.post(
        ROUTES["policy_bind_route"], json=bind_request(handshake)
    ).json()
    state.responses["binding"] = binding

    # 8. submit -- asynchronous, idempotent, and answered 202.
    submitted = client.post(
        ROUTES["rollout_route"],
        json=submit_request(handshake, binding["config_id"], rollout_id=ROLLOUT_ID),
    )
    assert submitted.status_code == 202, submitted.text
    state.responses["submit"] = submitted.json()
    state.responses["resubmit"] = client.post(
        ROUTES["rollout_route"],
        json=submit_request(
            handshake, binding["config_id"], rollout_id="rollout_e2e_retry"
        ),
    ).json()

    # 9. poll -- state, events, and a heartbeat on the declared renew route.
    state.responses["state"] = client.get(
        ROUTES["rollout_state_route"].format(rollout_id=ROLLOUT_ID)
    ).json()
    state.responses["events"] = client.get(
        ROUTES["rollout_events_route"].format(rollout_id=ROLLOUT_ID)
    ).json()
    state.responses["renew"] = client.post(
        ROUTES["rollout_renew_route"].format(rollout_id=ROLLOUT_ID),
        json={"handshake_id": handshake["handshake_id"]},
    ).json()

    # 10. finalize -- clip at the horizon and attest quiescence.
    state.responses["finalize"] = client.post(
        ROUTES["rollout_finalize_route"].format(rollout_id=ROLLOUT_ID), json={}
    ).json()

    # 11. trace, reward, artifacts, terminate.
    trace = client.get(ROUTES["trace_route"].format(rollout_id=ROLLOUT_ID)).json()
    state.responses["trace"] = trace
    state.responses["reward"] = client.get(
        ROUTES["reward_route"],
        params={"rollout_id": ROLLOUT_ID, "trace_digest": trace["trace_digest"]},
    ).json()
    state.responses["artifacts"] = client.get(
        ROUTES["artifacts_route"].format(rollout_id=ROLLOUT_ID)
    ).json()
    state.responses["terminate"] = client.post(
        ROUTES["rollout_terminate_route"].format(rollout_id=ROLLOUT_ID),
        json={"reason": "executor_done"},
    ).json()
    return state


# --------------------------------------------------------------------------- #
# Discovery and agreement
# --------------------------------------------------------------------------- #


def test_the_container_advertises_the_routes_it_actually_mounted(run: Run) -> None:
    """Discovery, not decoration: the advertisement is the mounted route table.

    ``gepa`` is absent because this runtime declares no program, which is the
    same rule read the other way: a container advertises the contracts it
    declares and no others.
    """

    metadata = run.json("metadata")["metadata"]
    contracts = metadata["optimizer_contracts"]
    assert "gepa" not in contracts
    block = contracts["cispo"]
    assert block["version"] == CISPO_OPTIMIZER_CONTRACT_VERSION
    for name, route in ROUTES.items():
        assert block[name] == route, name
    assert run.json("metadata")["runtime"]["runtime_id"] == "counter.reference"
    # Installing the target published the surface it supplies, and only that.
    published = run.json("metadata")["capabilities"]
    assert published["proxied_inference"] is True
    assert published["token_emission"]["logprobs"] is True


def test_the_capability_document_is_the_one_the_target_hashed(run: Run) -> None:
    document = run.json("capabilities")
    assert document == run.target.capability_document
    assert document["capability_hash"] == run.target.capability_hash
    assert document["routes"] == dict(sorted(ROUTES.items()))
    assert document["evidence"]["trace_v5"] is True
    assert document["evidence"]["behavior_logprobs"] is True
    assert document["reward"]["authority"] == "container"


def test_health_names_the_build_a_receipt_has_to_name(run: Run) -> None:
    health = run.json("health")
    assert health["status"] == "ok"
    assert health["container_image_digest"] == "sha256:reference-counter"
    assert health["capability_hash"] == run.json("capabilities")["capability_hash"]


def test_discovery_resolves_the_task_the_agreement_then_names(run: Run) -> None:
    taskset = run.json("taskset")
    assert taskset["deterministic_lookup"] is True
    assert taskset["duplicate_free"] is True
    (row,) = run.json("rows")["rows"]
    assert row["content_digest"].startswith("sha256:")
    assert run.json("topology")["topology_id"] == run.json("capabilities")["topology_ref"]
    resolution = run.json("handshake")["taskset_resolution"]
    assert [item["task_id"] for item in resolution] == [row["task_id"]]
    assert resolution[0]["content_digest"] == row["content_digest"]


def test_the_handshake_answers_every_clause_with_a_reason(run: Run) -> None:
    handshake = run.json("handshake")
    assert handshake["accepted"] is True
    assert handshake["capability_hash"] == run.json("capabilities")["capability_hash"]
    assert handshake["agreement_digest"].startswith("sha256:")
    for answer in handshake["clauses"]:
        assert answer["verdict"] in {"accepted", "degraded", "rejected", "unsupported"}
        assert answer["reason"].strip(), answer["clause_id"]


def test_both_binding_kinds_are_served_and_only_one_of_them_reaches_a_provider(
    run: Run,
) -> None:
    probe = run.json("probe_binding")
    assert probe["policy_kind"] == "probe"
    assert probe["provider_requests_expected"] == 0
    assert probe["sampler_origin"] is None

    binding = run.json("binding")
    assert binding["policy_kind"] == "trainable"
    assert binding["sampler_ready"] is True
    assert binding["immutable"] is True
    # The credential arrived inside the origin and does not come back out.
    assert binding["sampler_origin"]["credential"] is None
    assert binding["sampler_origin"]["credential_supplied"] is True
    assert "session-prid-1" not in str(binding)


# --------------------------------------------------------------------------- #
# The attempt
# --------------------------------------------------------------------------- #


def test_submission_is_accepted_asynchronously_with_a_lease(run: Run) -> None:
    submitted = run.json("submit")
    assert submitted["rollout_id"] == ROLLOUT_ID
    assert submitted["state"] == "running"
    assert submitted["lease_expires_at"] > 0
    assert submitted["correlation"]["group_id"] == "group-1"
    assert submitted["correlation"]["sample_index"] == 0


def test_a_lost_response_resubmitted_resolves_to_the_same_attempt(run: Run) -> None:
    again = run.json("resubmit")
    assert again["duplicate"] is True
    assert again["rollout_id"] == ROLLOUT_ID


def test_polling_is_cheap_and_the_cursor_resumes(run: Run) -> None:
    # The episode ran inside the submission, so polling reports that there is
    # nothing left to wait for. It is not a terminal result: finalize decides
    # that, and it has not been called yet at this point in the sequence.
    assert run.json("state")["state"] == "awaiting_score"
    assert run.json("state")["terminal"] is None
    assert run.json("state")["overdue"] is False
    events = run.json("events")
    kinds = [row["kind"] for row in events["events"]]
    assert "rollout.attempt.accepted" in kinds
    assert "rollout.lease.granted" in kinds
    assert int(events["next_cursor"]) == int(events["high_water_cursor"])
    assert run.json("renew")["renewals"] == 1


def test_finalize_attests_the_horizon_before_anything_is_scored(run: Run) -> None:
    finalize = run.json("finalize")
    assert finalize["state"] == "completed"
    assert finalize["terminal"]["kind"] == "episode"
    assert finalize["quiescence"]["quiesced"] is True
    assert finalize["horizon"]["horizon_kind"] == "steps"


def test_the_trace_is_sealed_evidence_and_not_a_summary(run: Run) -> None:
    trace = run.json("trace")
    assert trace["trace_digest"].startswith("sha256:")
    assert trace["document"]["content_digest"] == trace["trace_digest"]
    assert len(trace["calls"]) == 2
    assert len(trace["episodes"]) == 1
    assert trace["episodes"][0]["trace_digest"] == trace["trace_digest"]


def test_the_reward_is_bound_to_the_trace_the_attempt_sealed(run: Run) -> None:
    reward = run.json("reward")
    assert reward["rollout_id"] == ROLLOUT_ID
    assert reward["trace_digest"] == run.json("trace")["trace_digest"]
    assert reward["terminal_status"] == "completed"
    assert reward["evaluation_plan_id"] == run.json("capabilities")["reward"][
        "evaluation_plan_id"
    ]
    assert reward["channels"] and reward["channels"][0]["measure"] == 1.0
    assert reward["horizon"]["quiescence_attested"] is True


def test_the_artifacts_route_inventories_the_trace_it_serves(run: Run) -> None:
    (row,) = run.json("artifacts")["artifacts"]
    assert row["content_digest"] == run.json("trace")["trace_digest"]
    assert row["route"] == ROUTES["trace_route"]


def test_terminating_a_finished_attempt_reports_its_one_terminal_result(
    run: Run,
) -> None:
    terminate = run.json("terminate")
    assert terminate["already_terminal"] is True
    assert terminate["terminal"]["terminal_status"] == "completed"


# --------------------------------------------------------------------------- #
# The optimizer's own validators, on what came off the wire
# --------------------------------------------------------------------------- #


def test_the_optimizer_validators_were_reachable() -> None:
    """Say which path this run took rather than passing quietly on the weaker one."""

    if OPTIMIZER_RECORDS is None or OPTIMIZER_PROBE is None:
        pytest.skip(
            "synth-optimizers is not reachable from this worktree; the evidence and "
            "reward tests below fall back to asserting the same properties directly"
        )
    assert OPTIMIZER_RECORDS.INFERENCE_CALL_SCHEMA_VERSION == "cispo.inference_call.v2"
    assert OPTIMIZER_RECORDS.REWARD_RECORD_SCHEMA_VERSION == "cispo.reward_record.v1"
    assert callable(OPTIMIZER_PROBE.validate_probe)


def test_the_served_evidence_passes_the_optimizers_record_validators(run: Run) -> None:
    trace = run.json("trace")
    records = OPTIMIZER_RECORDS

    if records is None:
        for row in trace["calls"]:
            assert row["token_capture_provenance"] == "engine_meta"
            assert len(row["generation_logprobs"]) == len(row["generation_token_ids"])
            assert any(value != 0.0 for value in row["generation_logprobs"])
        for episode in trace["episodes"]:
            assert episode["probe"] is False
            for segment in episode["segments"]:
                assert len(segment["loss_mask"]) == len(segment["token_ids"])
                assert any(segment["loss_mask"])
        return

    from tests.test_cispo_evidence import _optimizer_call, _optimizer_segment

    calls = [_optimizer_call(records, row) for row in trace["calls"]]
    for call in calls:
        call.validate_for_training()
    # Two turns, so the strict-prefix rule is provable rather than asserted.
    records.assert_strict_prefix(calls[0], calls[1])
    for episode_payload in trace["episodes"]:
        episode = records.TrainableEpisode(
            rollout_id=episode_payload["rollout_id"],
            task_id=episode_payload["task_id"],
            seed=episode_payload["seed"],
            policy_revision=episode_payload["policy_revision"],
            behavior_fingerprint=episode_payload["behavior_fingerprint"],
            segments=tuple(
                _optimizer_segment(records, item) for item in episode_payload["segments"]
            ),
            terminal_status=episode_payload["terminal_status"],
            trace_digest=episode_payload["trace_digest"],
        )
        episode.validate()


def test_the_served_reward_passes_the_optimizers_reward_validator(run: Run) -> None:
    reward = run.json("reward")
    trace_digest = run.json("trace")["trace_digest"]
    records = OPTIMIZER_RECORDS

    if records is None:
        assert reward["optimized_channel"] in {
            row["channel_id"] for row in reward["channels"]
        }
        assert reward["trace_digest"] == trace_digest
        assert reward["horizon"]["quiescence_attested"] or reward["horizon"]["clipped"]
        return

    theirs = records.RewardRecord(
        reward_id=reward["reward_id"],
        rollout_id=reward["rollout_id"],
        trace_digest=reward["trace_digest"],
        channels=tuple(
            records.RewardChannel(**channel) for channel in reward["channels"]
        ),
        optimized_channel=reward["optimized_channel"],
        terminal_status=reward["terminal_status"],
        evaluation_plan_id=reward["evaluation_plan_id"],
        horizon=records.HorizonEvidence(**reward["horizon"]),
        metadata=reward["metadata"],
    )
    # Bound to the episode the container actually sealed, by the optimizer's rule.
    theirs.validate(episode_trace_digest=trace_digest)
    assert theirs.value_for_team("team-0") == 1.0


def test_the_declared_probe_passes_the_optimizers_validate_probe(run: Run) -> None:
    """The unpaid path, walked against the same admitted agreement."""

    handshake = run.json("handshake")
    agreement = run.target.admission.assert_admissible(
        handshake["handshake_id"], handshake["agreement_digest"]
    )
    attempt, report = run_probe(
        run.target.facts,
        agreement,
        task_id=run.target.facts.discovery.rows[0].task_id,
        seed=3,
        model_family="family",
        model_id="family/model",
    )
    assert set(report.operations) == set(REQUIRED_PROBE_OPERATIONS)

    if OPTIMIZER_PROBE is None or OPTIMIZER_RECORDS is None:
        payload = attempt.to_payload()
        assert payload["rollout_id"].startswith("probe_")
        assert payload["episode"]["probe"] is True
        for call in payload["calls"]:
            assert call["token_capture_provenance"] == "probe_synthetic"
        return

    from tests.test_cispo_probe import _optimizer_probe_attempt

    opt_attempt, profile = _optimizer_probe_attempt(attempt)
    validated = OPTIMIZER_PROBE.validate_probe(
        opt_attempt,
        expected_profile=profile,
        quiescence_accepted=agreement.quiescence_accepted,
    )
    assert validated.trainable is False
    assert validated.trace_digest == attempt.trace_digest
    # Probe evidence and real evidence are never mistakable for each other.
    assert attempt.trace_digest != run.json("trace")["trace_digest"]


def test_a_second_container_reaches_the_same_agreement_digest() -> None:
    """The document, the renderer, the tasks and the obligations are all pinned.

    Two identically built containers issuing against the same requirement
    document must agree, or the digest is not an agreement -- it is a nonce.
    """

    _, left = installed(target_count=2)
    _, right = installed(target_count=2)
    assert left.capability_hash == right.capability_hash
    first = admitted(left)
    second = admitted(right)
    assert first["capability_hash"] == second["capability_hash"]
    assert first["taskset_resolution"] == second["taskset_resolution"]
    assert first["obligations"] == second["obligations"]


# --------------------------------------------------------------------------- #
# A probe through the target's own attempt machine
# --------------------------------------------------------------------------- #
#
# ``cispo_probe`` builds a probe attempt out of canned generations and its own
# ``ProbeSession``. That proves the probe *shape*, and nothing about the path
# the executor actually takes at startup, which is to bind a probe and submit an
# attempt against it over the declared routes -- through ``admit_attempt``,
# ``CounterAttemptRuntime._run_episode``, the evidence builder, the lifecycle
# and the reward authority, exactly as a paid attempt goes. That path had never
# been walked, and the whole point of running a probe is that it walks the paid
# one.
#
# So this section drives both kinds against one container and compares them.


class ProbeRun:
    """One container, one agreement, and two attempts: one unpaid, one paid."""

    def __init__(self) -> None:
        self.runtime, self.target = installed(target_count=2)
        self.client = TestClient(create_reference_app(self.runtime))
        self.responses: dict[str, Any] = {}
        self.probe_rollout_id = ""
        self.paid_rollout_id = ""

    def json(self, name: str) -> Any:
        return self.responses[name]


def _drive(client: TestClient, rollout_id: str, handshake: dict[str, Any]) -> dict[str, Any]:
    """State, finalize, trace, reward, artifacts, terminate -- once, in order."""

    answers: dict[str, Any] = {}
    answers["state"] = client.get(
        ROUTES["rollout_state_route"].format(rollout_id=rollout_id)
    ).json()
    answers["events"] = client.get(
        ROUTES["rollout_events_route"].format(rollout_id=rollout_id)
    ).json()
    answers["finalize"] = client.post(
        ROUTES["rollout_finalize_route"].format(rollout_id=rollout_id), json={}
    ).json()
    trace = client.get(ROUTES["trace_route"].format(rollout_id=rollout_id)).json()
    answers["trace"] = trace
    answers["reward"] = client.get(
        ROUTES["reward_route"],
        params={"rollout_id": rollout_id, "trace_digest": trace["trace_digest"]},
    ).json()
    answers["artifacts"] = client.get(
        ROUTES["artifacts_route"].format(rollout_id=rollout_id)
    ).json()
    answers["terminate"] = client.post(
        ROUTES["rollout_terminate_route"].format(rollout_id=rollout_id),
        json={"reason": "executor_done", "handshake_id": handshake["handshake_id"]},
    ).json()
    return answers


@pytest.fixture(scope="module")
def probe_run() -> ProbeRun:
    """Bind a probe and a paid policy, submit one attempt against each, drive both.

    Module-scoped for the same reason ``run`` is: the two attempts are compared
    against each other, and re-running the sequence per test would compare an
    attempt against a container that had not run the other one.
    """

    state = ProbeRun()
    client, target = state.client, state.target
    handshake = client.post(
        ROUTES["handshake_route"], json=handshake_request(target)
    ).json()
    assert handshake["accepted"], handshake["rejected_mandatory_clauses"]
    state.responses["handshake"] = handshake
    task_id = handshake["taskset_resolution"][0]["task_id"]

    # The probe binding. ``behavior_fingerprint`` is the executor's, pinned here
    # the way a sampler binding pins the one on its origin: the behavior a probe
    # proves the evidence path for is the model the run will go on to train, and
    # the container cannot derive it.
    probe_binding = client.post(
        ROUTES["policy_bind_route"],
        json={
            "handshake_id": handshake["handshake_id"],
            "agreement_digest": handshake["agreement_digest"],
            "kind": "probe",
            "model_family": "family",
            "model_id": "family/model",
            "behavior_fingerprint": "bf-probe-7",
        },
    ).json()
    state.responses["probe_binding"] = probe_binding

    paid_binding = client.post(
        ROUTES["policy_bind_route"], json=bind_request(handshake)
    ).json()
    state.responses["paid_binding"] = paid_binding

    # No rollout id: the container mints one, and which namespace it mints into
    # is part of what makes probe evidence unmistakable.
    probe_submission = submit_request(
        handshake,
        probe_binding["config_id"],
        idempotency_key="probe-k-1",
        task_id=task_id,
    )
    probe_submission.pop("rollout_id")
    accepted = client.post(ROUTES["rollout_route"], json=probe_submission)
    assert accepted.status_code == 202, accepted.text
    state.responses["probe_submit"] = accepted.json()
    state.probe_rollout_id = accepted.json()["rollout_id"]

    paid = client.post(
        ROUTES["rollout_route"],
        json=submit_request(
            handshake,
            paid_binding["config_id"],
            rollout_id="rollout_probe_peer",
            idempotency_key="paid-k-1",
            task_id=task_id,
        ),
    )
    assert paid.status_code == 202, paid.text
    state.paid_rollout_id = paid.json()["rollout_id"]

    state.responses["probe"] = _drive(client, state.probe_rollout_id, handshake)
    state.responses["paid"] = _drive(client, state.paid_rollout_id, handshake)
    return state


def test_a_probe_binding_is_submittable_and_reaches_no_provider(probe_run: ProbeRun) -> None:
    """The binding an executor submits against, and what it promises not to do."""

    binding = probe_run.json("probe_binding")
    assert binding["config_id"] == binding["binding_id"]
    assert binding["probe"] is True
    assert binding["trainable"] is False
    # A probe has no origin and no credential, and that absence is the thing
    # that makes it unable to spend. It is not an omission to be filled in.
    assert binding["sampler_origin"] is None
    assert binding["credential"] is None
    assert binding["provider_requests_expected"] == 0
    # The fingerprint the caller pinned is the one the binding carries.
    assert binding["behavior_fingerprint"] == "bf-probe-7"
    assert binding["config_id"].startswith("probe_")


def test_a_probe_attempt_walks_the_same_episode_a_paid_attempt_walks(
    probe_run: ProbeRun,
) -> None:
    """Same environment, same renderer, same turns, same tokens.

    This is the whole reason for running a probe: if the unpaid attempt took a
    shorter or different path, it would prove nothing about the paid one.
    """

    probe = probe_run.json("probe")["trace"]
    paid = probe_run.json("paid")["trace"]
    assert len(probe["calls"]) == len(paid["calls"]) > 1
    for unpaid_call, paid_call in zip(probe["calls"], paid["calls"], strict=True):
        assert unpaid_call["prompt_token_ids"] == paid_call["prompt_token_ids"]
        assert unpaid_call["generation_token_ids"] == paid_call["generation_token_ids"]
        assert unpaid_call["generation_logprobs"] == paid_call["generation_logprobs"]
        assert unpaid_call["stop_token_ids"] == paid_call["stop_token_ids"]
        assert (
            unpaid_call["renderer_profile_fingerprint"]
            == paid_call["renderer_profile_fingerprint"]
        )
        assert unpaid_call["author_kind"] == paid_call["author_kind"] == "policy"
    # And the same lifecycle, down to the terminal result and the horizon.
    assert probe_run.json("probe")["finalize"]["terminal"]["kind"] == "episode"
    assert (
        probe_run.json("probe")["finalize"]["quiescence"]["quiesced"]
        is probe_run.json("paid")["finalize"]["quiescence"]["quiesced"]
    )
    assert probe_run.json("probe")["artifacts"]["artifacts"][0]["role"] == "trace"


def test_a_probe_attempt_carries_a_per_turn_identity_of_its_own(
    probe_run: ProbeRun,
) -> None:
    """A probe has no origin to read a proxy request id off, and still has one.

    The paid session takes the per-attempt id from the sampler origin. A probe
    reaches no provider, so the identity is minted by the session that drove it
    -- per attempt and per turn -- rather than by fabricating an origin whose
    URL and credential would both be claims about a provider that was not there.
    """

    calls = probe_run.json("probe")["trace"]["calls"]
    ids = [call["proxy_request_id"] for call in calls]
    assert len(set(ids)) == len(ids)
    for index, value in enumerate(ids, start=1):
        assert value == f"{probe_run.probe_rollout_id}:proxy-{index}"
    # The paid attempt's calls share the one id its origin named.
    paid_ids = {call["proxy_request_id"] for call in probe_run.json("paid")["trace"]["calls"]}
    assert paid_ids == {"prid-1"}


def test_probe_evidence_is_refused_for_training_by_its_own_fields(
    probe_run: ProbeRun,
) -> None:
    """Every structural mark ``cispo_probe.PROBE_MARKERS`` names, on this path too."""

    rollout_id = probe_run.probe_rollout_id
    assert rollout_id.startswith(PROBE_ROLLOUT_PREFIX)
    trace = probe_run.json("probe")["trace"]
    for call in trace["calls"]:
        assert call["token_capture_provenance"] == PROBE_TOKEN_CAPTURE_PROVENANCE
        assert call["trainable"] is False
        assert call["wire_request"]["synthetic"] is True
        assert call["wire_request"]["provider"] is None
        assert call["wire_response"]["synthetic"] is True
        assert call["wire_response"]["provider"] is None
        assert call["usage"]["provider_requests"] == 0
        assert call["usage"]["billed"] is False
    assert [episode["probe"] for episode in trace["episodes"]] == [True]
    assert probe_run.json("probe")["reward"]["metadata"]["probe"] is True
    # The paid attempt is the control: the same fields say the opposite.
    paid = probe_run.json("paid")["trace"]
    for call in paid["calls"]:
        assert call["token_capture_provenance"] == "engine_meta"
        assert call["trainable"] is True
    assert [episode["probe"] for episode in paid["episodes"]] == [False]
    assert probe_run.json("paid")["reward"]["metadata"]["probe"] is False


def test_the_container_refuses_to_train_on_the_probe_it_just_sealed(
    probe_run: ProbeRun,
) -> None:
    """The training gate, run against the evidence the attempt machine produced.

    A probe that *passed* this gate would be evidence a trainer could not tell
    from the real thing, so the refusal is the assertion.
    """

    evidence = probe_run.target.attempts.evidence(probe_run.probe_rollout_id)
    with pytest.raises(EvidenceError):
        evidence.validate()
    for call in evidence.calls:
        with pytest.raises(EvidenceError):
            call.validate_for_training()
    # The paid attempt's evidence passes the same gate untouched.
    probe_run.target.attempts.evidence(probe_run.paid_rollout_id).validate()


def test_the_terminate_receipt_says_which_kind_of_attempt_it_settled(
    probe_run: ProbeRun,
) -> None:
    """Identity plus digests, and whether anything was bought."""

    probe_receipt = probe_run.json("probe")["terminate"]["receipt"]
    paid_receipt = probe_run.json("paid")["terminate"]["receipt"]
    assert probe_receipt["probe"] is True
    assert paid_receipt["probe"] is False
    for receipt, rollout_id in (
        (probe_receipt, probe_run.probe_rollout_id),
        (paid_receipt, probe_run.paid_rollout_id),
    ):
        assert receipt["rollout_id"] == rollout_id
        assert receipt["terminal_status"] == "completed"
        assert receipt["proxy_request_id"]
        assert receipt["behavior_fingerprint"]
        assert receipt["trace_digest"].startswith("sha256:")
    assert probe_receipt["behavior_fingerprint"] == "bf-probe-7"


def test_a_probe_may_not_be_submitted_outside_its_own_id_namespace(
    probe_run: ProbeRun,
) -> None:
    """A probe id that could collide with a real rollout id is refused, not fixed.

    Asserted at the port rather than over HTTP: the adapter types only its 501,
    so every other refusal reaches a client as a bare 500 and the reason is only
    readable here.
    """

    handshake = probe_run.json("handshake")
    binding = probe_run.json("probe_binding")
    with pytest.raises(CispoTargetError, match=PROBE_ROLLOUT_PREFIX):
        probe_run.target.rollouts.cispo_submit_rollout(
            submit_request(
                handshake,
                binding["config_id"],
                rollout_id="rollout_pretending_to_be_paid",
                idempotency_key="probe-k-2",
            )
        )
    # And the converse: a paid binding may not claim the probe namespace.
    with pytest.raises(CispoTargetError, match=PROBE_ROLLOUT_PREFIX):
        probe_run.target.rollouts.cispo_submit_rollout(
            submit_request(
                handshake,
                probe_run.json("paid_binding")["config_id"],
                rollout_id="probe_pretending_to_be_unpaid",
                idempotency_key="paid-k-2",
            )
        )


def test_the_probe_this_container_ran_passes_the_optimizers_probe_validators(
    probe_run: ProbeRun,
) -> None:
    """The unpaid attempt, checked by the party that requires it to be unpaid."""

    trace = probe_run.json("probe")["trace"]
    if OPTIMIZER_PROBE is None or OPTIMIZER_RECORDS is None:
        # Mirrored: the same refusal, stated against the emitted payload.
        for call in trace["calls"]:
            assert call["token_capture_provenance"] not in {"engine_meta"}
            assert call["trainable"] is False
        return
    calls = [
        OPTIMIZER_RECORDS.InferenceCall(
            call_id=row["call_id"],
            proxy_request_id=row["proxy_request_id"],
            rollout_id=row["rollout_id"],
            group_id=row["group_id"],
            sample_index=row["sample_index"],
            behavior_fingerprint=row["behavior_fingerprint"],
            policy_revision=row["policy_revision"],
            wire_api=row["wire_api"],
            sampling_transport=row["sampling_transport"],
            token_capture_provenance=row["token_capture_provenance"],
            prompt_token_ids=tuple(row["prompt_token_ids"]),
            generation_token_ids=tuple(row["generation_token_ids"]),
            generation_logprobs=tuple(row["generation_logprobs"]),
            sampled_mask=tuple(row["sampled_mask"]),
            finish_reason=row["finish_reason"],
            stop_token_ids=tuple(row["stop_token_ids"]),
            renderer_profile_fingerprint=row["renderer_profile_fingerprint"],
            trainable=row["trainable"],
            author_kind=row["author_kind"],
            branch_id=row["branch_id"],
            agent_instance_id=row["agent_instance_id"],
            team_id=row["team_id"],
            wire_request=dict(row["wire_request"]),
            wire_response=dict(row["wire_response"]),
            usage=dict(row["usage"]),
            created_at=row["created_at"],
        )
        for row in trace["calls"]
    ]
    # The optimizer's own conformance check: probe evidence that could be
    # mistaken for real evidence fails here rather than one training step later.
    OPTIMIZER_PROBE.assert_probe_not_trainable(
        SimpleNamespace(calls=tuple(calls), rollout_id=probe_run.probe_rollout_id)
    )
    for previous, following in zip(calls, calls[1:], strict=False):
        OPTIMIZER_RECORDS.assert_strict_prefix(previous, following)
