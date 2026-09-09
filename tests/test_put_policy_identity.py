"""`PUT /policy` identity must survive the parse layer.

`state.put_policy` rejects an empty ``namespace`` or ``name`` with
``policy_identity_required``. That check is only safe while
``parse_put_policy``/``to_put_policy_dict`` actually carry those two fields
across -- an earlier revision of the pair forwarded ``{code, harness}`` alone,
and pairing that parser with this validator 422s every well-formed request no
matter what the caller sent. The two halves have to move together, so pin the
contract here rather than in a comment.
"""

from __future__ import annotations

from synth_containers.platform.http_requests import (
    parse_put_policy,
    to_put_policy_dict,
)

# What src/core/client.py:put_policy sends for the nanohorizon harness.
_BODY = {
    "code": "def act(): ...",
    "harness": "nanohorizon",
    "namespace": "nanohorizon",
    "name": "qwen08b_mlx",
}


def _round_trip(body: dict) -> dict:
    return to_put_policy_dict(parse_put_policy(body))


def test_parse_forwards_policy_identity() -> None:
    parsed = _round_trip(_BODY)
    assert parsed["namespace"] == "nanohorizon"
    assert parsed["name"] == "qwen08b_mlx"


def test_round_trip_survives_the_identity_validator() -> None:
    """The exact rejection state.put_policy applies, against the parsed body."""

    parsed = _round_trip(_BODY)
    harness = str(parsed.get("harness") or "unset")
    namespace = str(parsed.get("namespace") or harness).strip()
    name = str(parsed.get("name") or "").strip()
    assert namespace and name, (
        "parse_put_policy dropped the policy identity; state.put_policy would "
        f"answer 422 policy_identity_required (namespace={namespace!r} name={name!r})"
    )


def test_identity_is_not_recoverable_from_harness_alone() -> None:
    """Guards the failure mode: harness present, identity gone -> still a 422."""

    parsed = _round_trip({"code": "x", "harness": "nanohorizon"})
    assert not str(parsed.get("name") or "").strip()


def test_revision_identity_is_independent_of_per_episode_config_id() -> None:
    """A policy is installed once; sampler configs are bound per episode.

    ``PUT /policy`` fixes ``revision.name`` (nanohorizon sends ``qwen08b_mlx``)
    while the episode binder binds ``nh-<session_id>`` so the sampler base_url can
    carry the session. Requiring ``revision.name == config_id`` in the rollout
    admission path 409s every episode for any caller that binds config per
    episode -- and the revision itself declines to own a config id, storing
    ``config_id=None``. Keep the two namespaces apart.
    """

    import inspect

    from synth_containers.platform import state as state_mod

    src = inspect.getsource(state_mod)
    marker = 'return {\n                    "error": "policy_configuration_mismatch"'
    head, _, _ = src.partition(marker)
    guard = head.rsplit("if ", 1)[1].split(":\n")[0]
    assert "config_id" not in guard, (
        "policy_configuration_mismatch must not compare the revision identity "
        f"against the per-episode config id; guard is: if {guard}:"
    )
