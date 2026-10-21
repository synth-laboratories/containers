"""Observed Craftax grading inputs derived from immutable V5 occurrences.

See docs/specs/immutable-grading-v1.md. This is a symbolic readout profile, not a
full simulator snapshot or a claim of full-attempt capture. Missing or ambiguous
step evidence refuses instead of inventing actions, rewards or achievements.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .tracing.canonical import content_digest
from .tracing.models.document import TraceDocumentV5
from .tracing.models.events import EventV5

SCHEMA = "synth.craftax-observed-trajectory.v1"
MAX_STEPS = 10_000


def observed_craftax_trajectory(document: TraceDocumentV5) -> dict[str, Any]:
    if not document.content_digest or content_digest(document) != document.content_digest:
        raise ValueError("craftax_observed_trace_identity_invalid")
    indexed: dict[str, dict[int, EventV5]] = {
        name: {} for name in ("action", "reward_signal", "observation")
    }
    terminal = []
    event_ids: set[str] = set()
    for event in document.events:
        kind = str(event.event_type)
        if kind == "env.episode.closed":
            terminal.append(event.payload)
        if kind not in indexed:
            continue
        if not event.event_id or event.event_id in event_ids:
            raise ValueError("craftax_observed_event_identity_invalid")
        event_ids.add(event.event_id)
        step = event.payload.get("step")
        if type(step) is not int or not 0 <= step <= MAX_STEPS or step in indexed[kind]:
            raise ValueError("craftax_observed_step_identity_invalid")
        indexed[kind][step] = event
    actions, rewards, observations = (
        indexed["action"],
        indexed["reward_signal"],
        indexed["observation"],
    )
    count = len(actions)
    if (
        not count
        or len(terminal) != 1
        or type(terminal[0].get("steps")) is not int
        or terminal[0]["steps"] != count
    ):
        raise ValueError("craftax_observed_terminal_prefix_missing")
    if (
        set(actions) != set(range(1, count + 1))
        or set(rewards) != set(actions)
        or set(observations) != set(range(count + 1))
    ):
        raise ValueError("craftax_observed_complete_step_evidence_missing")
    records = []
    for step in range(1, count + 1):
        action, reward, observation = actions[step], rewards[step], observations[step]
        value = reward.payload.get("value")
        readout = observation.payload.get("readout")
        if not isinstance(action.payload.get("action"), str) or not action.payload["action"]:
            raise ValueError("craftax_observed_action_missing")
        if (
            reward.payload.get("authority") != "environment"
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError("craftax_observed_reward_missing")
        if (
            not isinstance(readout, dict)
            or type(readout.get("env_steps")) is not int
            or readout["env_steps"] != step
        ):
            raise ValueError("craftax_observed_readout_missing")
        achievements = readout.get("achievements")
        if (
            not isinstance(achievements, list)
            or any(not isinstance(item, str) for item in achievements)
            or len(set(achievements)) != len(achievements)
        ):
            raise ValueError("craftax_observed_achievement_readout_missing")
        records.append(
            {
                "step": step - 1,
                "source_step": step,
                "action": action.payload["action"],
                "reward": value,
                "state": deepcopy(readout),
                "source_events": {
                    "action": action.event_id,
                    "reward": reward.event_id,
                    "observation": observation.event_id,
                },
            }
        )
    return {
        "schema_version": SCHEMA,
        "trace_id": document.trace_id,
        "trace_digest": document.content_digest,
        "state_semantics": "observed_readout",
        "records": records,
    }
