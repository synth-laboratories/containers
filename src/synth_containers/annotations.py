"""Container-facing rollout annotations (orthogonal to evidence AnnotationV1).

Finished rollouts expose extractable metadata via
``GET /rollouts/{rollout_id}/annotations`` as a typed list. These records are
lightweight HTTP-surface facts (achievements, token/action stats, inventory /
survival / progress, action histograms, parse/continuity/compaction,
termination, teacher labels) — not the full trace-evidence taxonomy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .formats import utc_now_iso
from .nouns import ExecutionRecord
from .serde import JsonDataclassMixin

ROLLOUT_ANNOTATIONS_SCHEMA = "synth.rollout_annotations.v1"


@dataclass(slots=True)
class RolloutAnnotation(JsonDataclassMixin):
    annotation_id: str
    kind: str
    rollout_id: str
    created_at: str = ""
    labels: list[str] = field(default_factory=list)
    confidence: float | None = None
    source: str = "runtime"
    target: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    ok: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutAnnotationList(JsonDataclassMixin):
    schema: str = ROLLOUT_ANNOTATIONS_SCHEMA
    rollout_id: str = ""
    status: str = "ready"
    annotations: list[RolloutAnnotation] = field(default_factory=list)
    count: int = 0
    trace_correlation_id: str | None = None

    def sealed(self) -> "RolloutAnnotationList":
        self.count = len(self.annotations)
        return self


def _new_id(kind: str) -> str:
    return f"{kind}_{uuid4().hex[:12]}"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _action_stats_payload(
    *,
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build action-quality stats (actions/LLM, invalid/noop fractions)."""

    meta = dict(metadata or {})
    use = dict(usage or {})
    llm = _first_number(
        summary.get("llm_calls"),
        summary.get("llm_call_count"),
        meta.get("llm_call_count"),
        meta.get("llm_calls"),
    )
    actions = _first_number(
        summary.get("actions"),
        summary.get("environment_action_count"),
        summary.get("action_count"),
        meta.get("environment_action_count"),
        meta.get("action_count"),
        use.get("actions"),
    )
    invalid = _first_number(
        summary.get("invalid_actions"),
        summary.get("invalid_action_count"),
        meta.get("invalid_action_count"),
        meta.get("invalid_actions"),
    )
    noops = _first_number(
        summary.get("effective_noops"),
        summary.get("effective_noop_count"),
        meta.get("effective_noop_count"),
        meta.get("effective_noops"),
    )
    proposed_noop_frac = _first_number(
        summary.get("proposed_noop_frac"),
        meta.get("proposed_noop_frac"),
    )
    action_list = summary.get("action_list")
    if proposed_noop_frac is None and isinstance(action_list, Sequence) and not isinstance(
        action_list, (str, bytes)
    ):
        flat = [str(item) for item in action_list]
        if flat:
            proposed_noop_frac = sum(1 for item in flat if item == "noop") / float(len(flat))

    if llm is None and actions is None and invalid is None and noops is None:
        return {}

    payload: dict[str, Any] = {}
    if llm is not None:
        payload["llm_calls"] = int(llm)
    if actions is not None:
        payload["actions"] = int(actions)
    if invalid is not None:
        payload["invalid_actions"] = int(invalid)
    if noops is not None:
        payload["effective_noops"] = int(noops)
    if llm is not None and llm > 0 and actions is not None:
        payload["actions_per_llm"] = float(actions) / float(llm)
    elif summary.get("actions_per_llm") is not None:
        payload["actions_per_llm"] = float(summary["actions_per_llm"])
    if actions is not None and actions > 0 and invalid is not None:
        payload["invalid_frac"] = float(invalid) / float(actions)
    if actions is not None and actions > 0 and noops is not None:
        payload["noop_frac"] = float(noops) / float(actions)
    if proposed_noop_frac is not None:
        payload["proposed_noop_frac"] = float(proposed_noop_frac)
    return payload


_INVENTORY_SNAPSHOT_KEYS = (
    "health",
    "energy",
    "drink",
    "food",
    "mana",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "torches",
    "arrows",
    "pickaxe",
    "sword",
    "bow",
    "boss_progress",
    "xp",
)


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("event_type") or event.get("type") or "")


def _event_data(event: Mapping[str, Any]) -> dict[str, Any]:
    data = event.get("data")
    if isinstance(data, Mapping):
        return dict(data)
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        # Flatten common shapes: payload may already be the transition body.
        if any(key in payload for key in ("observation_after", "action", "step", "achievements")):
            return dict(payload)
        nested = payload.get("data")
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def _observation_from_transition(data: Mapping[str, Any]) -> dict[str, Any]:
    after = data.get("observation_after")
    if isinstance(after, Mapping):
        obs = after.get("observation")
        if isinstance(obs, Mapping):
            return dict(obs)
        if any(key in after for key in ("inventory", "player", "floor_state")):
            return dict(after)
    obs = data.get("observation")
    if isinstance(obs, Mapping):
        return dict(obs)
    return {}


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _inventory_snapshot(inv: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in _INVENTORY_SNAPSHOT_KEYS:
        if key not in inv:
            continue
        value = inv.get(key)
        if isinstance(value, (int, float, str, bool)) or value is None:
            snapshot[key] = value
    tools = {
        key: inv.get(key)
        for key in ("pickaxe", "sword", "bow", "sword_enchantment", "bow_enchantment")
        if key in inv
    }
    if tools:
        snapshot["tools"] = tools
    return snapshot


def extract_env_progress(
    events: Sequence[Mapping[str, Any]] | None = None,
    *,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract terminal inventory, survival, unlocks, and floor/progress proxies.

    Accepts Craftax / GameBench ``environment.transition`` events and optional
    precomputed summary/metadata fields.
    """

    summary_map = _mapping(summary)
    meta_map = _mapping(metadata)
    precomputed = {
        key: summary_map[key]
        for key in (
            "terminal_inventory",
            "survival",
            "achievement_unlocks",
            "progress",
        )
        if isinstance(summary_map.get(key), Mapping)
    }
    for key in ("terminal_inventory", "survival", "achievement_unlocks", "progress"):
        if key not in precomputed and isinstance(meta_map.get(key), Mapping):
            precomputed[key] = dict(meta_map[key])

    transitions: list[dict[str, Any]] = []
    reward_events: list[dict[str, Any]] = []
    for event in events or ():
        if not isinstance(event, Mapping):
            continue
        kind = _event_type(event)
        data = _event_data(event)
        if kind in {"environment.transition", "transition"} or (
            not kind and ("observation_after" in data or "action" in data)
        ):
            transitions.append(data)
        elif kind in {"environment.reward", "reward"}:
            reward_events.append(data)

    health_points: list[tuple[int, float]] = []
    floor_points: list[tuple[int, int]] = []
    boss_points: list[tuple[int, float]] = []
    unlock_steps: dict[str, int] = {}
    last_obs: dict[str, Any] = {}
    last_inv: dict[str, Any] = {}
    last_native: list[Mapping[str, Any]] = []

    for data in transitions:
        step = _as_int(data.get("step"))
        if step is None:
            step = len(health_points) + 1
        obs = _observation_from_transition(data)
        if obs:
            last_obs = obs
        inv = obs.get("inventory") if isinstance(obs.get("inventory"), Mapping) else {}
        if isinstance(inv, Mapping) and inv:
            last_inv = dict(inv)
            health = _as_float(inv.get("health"))
            if health is not None:
                health_points.append((step, health))
            boss = _as_float(inv.get("boss_progress"))
            if boss is not None:
                boss_points.append((step, boss))
        player = obs.get("player") if isinstance(obs.get("player"), Mapping) else {}
        level = _as_int(player.get("level")) if isinstance(player, Mapping) else None
        if level is not None:
            floor_points.append((step, level))

        new_achievements = data.get("new_achievements")
        if isinstance(new_achievements, Sequence) and not isinstance(new_achievements, (str, bytes)):
            for name in new_achievements:
                key = str(name)
                unlock_steps.setdefault(key, step)
        elif isinstance(new_achievements, Mapping):
            for name, enabled in new_achievements.items():
                if enabled:
                    unlock_steps.setdefault(str(name), step)

        native = data.get("native_events")
        if isinstance(native, Sequence):
            last_native = [item for item in native if isinstance(item, Mapping)]

    for data in reward_events:
        name = data.get("achievement")
        step = _as_int(data.get("step"))
        if name is not None and step is not None:
            unlock_steps.setdefault(str(name), step)

    # Prefer explicit unlock maps from summary/metadata when present.
    for source in (summary_map, meta_map):
        raw_unlocks = source.get("achievement_unlock_steps") or source.get("first_unlock_steps")
        if isinstance(raw_unlocks, Mapping):
            for name, step in raw_unlocks.items():
                step_i = _as_int(step)
                if step_i is not None:
                    unlock_steps.setdefault(str(name), step_i)

    terminal_inventory = dict(precomputed.get("terminal_inventory") or {})
    if not terminal_inventory and last_inv:
        terminal_inventory = _inventory_snapshot(last_inv)
        player = last_obs.get("player") if isinstance(last_obs.get("player"), Mapping) else {}
        if isinstance(player, Mapping):
            if player.get("level") is not None:
                terminal_inventory["player_level"] = _as_int(player.get("level"))
            if player.get("pos") is not None:
                terminal_inventory["player_pos"] = player.get("pos")

    survival = dict(precomputed.get("survival") or {})
    if health_points and not survival:
        first_step, first_health = health_points[0]
        last_step, last_health = health_points[-1]
        min_step, min_health = min(health_points, key=lambda item: item[1])
        died = last_health <= 0
        death_reason = None
        death_source = None
        if died:
            death_reason = "health_depleted"
            for item in reversed(last_native):
                kind = str(item.get("kind") or "")
                transition = str(item.get("transition") or "")
                payload = item.get("payload") if isinstance(item.get("payload"), Mapping) else {}
                if kind in {"death", "terminal"} or transition == "death":
                    death_reason = str(payload.get("reason") or transition or kind or death_reason)
                if kind == "combat" or transition == "mob_attack":
                    entity = payload.get("entity") if isinstance(payload.get("entity"), Mapping) else {}
                    if entity.get("kind") is not None:
                        death_source = str(entity.get("kind"))
        survival = {
            "health_start": first_health,
            "health_end": last_health,
            "health_min": min_health,
            "health_start_step": first_step,
            "health_end_step": last_step,
            "health_min_step": min_step,
            "died": died,
            "death_reason": death_reason,
            "death_source": death_source,
        }
    elif summary_map.get("health_end") is not None and not survival:
        health_end = _as_float(summary_map.get("health_end"))
        survival = {
            "health_start": _as_float(summary_map.get("health_start")),
            "health_end": health_end,
            "health_min": _as_float(summary_map.get("health_min")),
            "died": bool(health_end is not None and health_end <= 0),
            "death_reason": summary_map.get("death_reason"),
            "death_source": summary_map.get("death_source"),
        }

    achievement_unlocks = dict(precomputed.get("achievement_unlocks") or {})
    if not achievement_unlocks:
        final_achievements = summary_map.get("achievements")
        if not _is_achievement_payload(final_achievements):
            final_achievements = meta_map.get("achievements")
        # Accept both shapes. `new_achievements` above already handles a mapping OR a
        # sequence; this branch used to require a Mapping, so the Craftax Rust compact
        # lane — which reports a list of unlocked names — fell through silently and
        # left `unlocked` empty on every rollout.
        final_names = _achievement_names(final_achievements)
        if unlock_steps or final_names:
            unique = sorted(unlock_steps) or sorted(final_names)
            achievement_unlocks = {
                "unique_achievements": float(len(unique)),
                "unlocked": unique,
                "first_unlock_step": dict(sorted(unlock_steps.items(), key=lambda item: item[1])),
            }

    progress = dict(precomputed.get("progress") or {})
    if not progress:
        deepest_floor = max((level for _, level in floor_points), default=None)
        final_floor = floor_points[-1][1] if floor_points else None
        boss_max = max((value for _, value in boss_points), default=None)
        boss_final = boss_points[-1][1] if boss_points else None
        if boss_final is None and last_inv.get("boss_progress") is not None:
            boss_final = _as_float(last_inv.get("boss_progress"))
            boss_max = boss_final if boss_max is None else boss_max
        floor_state = (
            last_obs.get("floor_state") if isinstance(last_obs.get("floor_state"), Mapping) else {}
        )
        floor_proxies = {}
        if isinstance(floor_state, Mapping):
            for key in ("monsters_killed", "chests_opened", "down_ladders", "up_ladders"):
                if key in floor_state:
                    floor_proxies[key] = floor_state.get(key)
        if any(
            value is not None
            for value in (deepest_floor, final_floor, boss_max, boss_final)
        ) or floor_proxies:
            progress = {
                "deepest_floor": deepest_floor,
                "final_floor": final_floor if final_floor is not None else deepest_floor,
                "boss_progress": boss_final,
                "boss_progress_max": boss_max,
                "floor_state": floor_proxies,
            }

    out: dict[str, Any] = {}
    if terminal_inventory:
        out["terminal_inventory"] = terminal_inventory
    if survival:
        out["survival"] = survival
    if achievement_unlocks:
        out["achievement_unlocks"] = achievement_unlocks
    if progress:
        out["progress"] = progress
    return out


def _env_progress_annotations(
    *,
    rollout_id: str,
    created: str,
    env_progress: Mapping[str, Any],
) -> list[RolloutAnnotation]:
    rows: list[RolloutAnnotation] = []
    inventory = env_progress.get("terminal_inventory")
    if isinstance(inventory, Mapping) and inventory:
        labels = []
        for key in ("health", "energy", "drink", "wood", "stone", "iron"):
            if inventory.get(key) is not None:
                labels.append(f"{key}={inventory.get(key)}")
        rows.append(
            make_annotation(
                kind="terminal_inventory",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels[:6],
                payload=dict(inventory),
                ok=True,
            )
        )

    survival = env_progress.get("survival")
    if isinstance(survival, Mapping) and survival:
        labels = []
        if survival.get("died"):
            labels.append("died")
        if survival.get("death_reason"):
            labels.append(str(survival["death_reason"]))
        if survival.get("death_source"):
            labels.append(str(survival["death_source"]))
        if survival.get("health_end") is not None:
            labels.append(f"hp_end={survival['health_end']}")
        rows.append(
            make_annotation(
                kind="survival",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(survival),
                ok=True,
            )
        )

    unlocks = env_progress.get("achievement_unlocks")
    if isinstance(unlocks, Mapping) and unlocks:
        unlocked = [str(x) for x in (unlocks.get("unlocked") or [])]
        rows.append(
            make_annotation(
                kind="achievement_unlocks",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=unlocked,
                payload=dict(unlocks),
                ok=True,
            )
        )

    progress = env_progress.get("progress")
    if isinstance(progress, Mapping) and progress:
        labels = []
        if progress.get("deepest_floor") is not None:
            labels.append(f"floor={progress['deepest_floor']}")
        if progress.get("boss_progress") is not None:
            labels.append(f"boss={progress['boss_progress']}")
        rows.append(
            make_annotation(
                kind="progress",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(progress),
                ok=True,
            )
        )
    return rows


def _is_achievement_payload(value: Any) -> bool:
    """A usable achievements payload is a mapping name→flag or a sequence of names."""
    if isinstance(value, Mapping):
        return True
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _achievement_names(value: Any) -> set[str]:
    """Unlocked achievement names from either shape; empty for anything else."""
    if isinstance(value, Mapping):
        return {str(name) for name, enabled in value.items() if enabled}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {str(name) for name in value if name}
    return set()


def _counter_shares(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {key: float(value) / total for key, value in counter.items()}


def _turns_from_sources(
    turns: Sequence[Mapping[str, Any]] | None,
    events: Sequence[Mapping[str, Any]] | None,
    summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes)):
        mapped = [item for item in turns if isinstance(item, Mapping)]
        if mapped:
            return mapped
    for source in (summary, metadata):
        raw = source.get("turns")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            mapped = [item for item in raw if isinstance(item, Mapping)]
            if mapped:
                return mapped
    # Some payloads nest turns under inference on a synthetic event.
    for event in events or ():
        if not isinstance(event, Mapping):
            continue
        data = _event_data(event)
        nested = data.get("turns")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            mapped = [item for item in nested if isinstance(item, Mapping)]
            if mapped:
                return mapped
    return []


def extract_behavior_diagnostics(
    events: Sequence[Mapping[str, Any]] | None = None,
    turns: Sequence[Mapping[str, Any]] | None = None,
    *,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Action hist, batch sizes, noop/invalid codes, parse, continuity, compaction."""

    summary_map = _mapping(summary)
    meta_map = _mapping(metadata)
    turn_rows = _turns_from_sources(turns, events, summary_map, meta_map)

    action_counts: Counter[str] = Counter()
    batch_counts: Counter[str] = Counter()
    continuity_counts: Counter[str] = Counter()
    noop_reasons: Counter[str] = Counter()
    invalid_codes: Counter[str] = Counter()
    invalid_parse_turns = 0
    prompt_deltas: list[dict[str, float]] = []
    prev_rendered: float | None = None

    for turn in turn_rows:
        actions = turn.get("actions") or turn.get("executed_actions") or []
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
            action_list = [str(action) for action in actions]
            action_counts.update(action_list)
            batch_counts[str(len(action_list))] += 1
        if turn.get("invalid_parse"):
            invalid_parse_turns += 1
        continuity = turn.get("prompt_continuity")
        if isinstance(continuity, Mapping):
            kind = str(continuity.get("kind") or "unknown")
            continuity_counts[kind] += 1
            rendered = _as_float(
                continuity.get("rendered_tokens")
                if continuity.get("rendered_tokens") is not None
                else len(turn.get("prompt_token_ids") or [])
            )
            if (
                prev_rendered is not None
                and rendered is not None
                and kind in {"initial", "fork"}
            ):
                prompt_deltas.append(
                    {
                        "before": float(prev_rendered),
                        "after": float(rendered),
                        "delta": float(rendered) - float(prev_rendered),
                    }
                )
            if rendered is not None:
                prev_rendered = float(rendered)
        else:
            prompt_len = len(turn.get("prompt_token_ids") or [])
            if prompt_len:
                prev_rendered = float(prompt_len)

    dropped_items = 0
    retained_items = 0
    compaction_count = 0
    for event in events or ():
        if not isinstance(event, Mapping):
            continue
        kind = _event_type(event)
        data = _event_data(event)
        if kind in {"environment.transition", "transition"}:
            for reason in data.get("noop_reasons") or []:
                noop_reasons[str(reason)] += 1
            for code in data.get("invalid_codes") or []:
                invalid_codes[str(code)] += 1
            if data.get("invalid") and not (data.get("invalid_codes") or []):
                invalid_codes["invalid"] += 1
        elif kind in {"agent.context_compacted", "context_compacted", "compaction"}:
            compaction_count += 1
            dropped_items += int(data.get("dropped_item_count") or 0)
            retained_items += int(data.get("retained_item_count") or 0)

    # Prefer metadata totals when events were stripped.
    if not compaction_count:
        compaction_count = int(
            _first_number(
                summary_map.get("compactions"),
                summary_map.get("compaction_count"),
                meta_map.get("compaction_count"),
                meta_map.get("compactions"),
            )
            or 0
        )
    meta_invalid_parse = int(
        _first_number(
            summary_map.get("invalid_parse_turn_count"),
            meta_map.get("invalid_parse_turn_count"),
            invalid_parse_turns,
        )
        or 0
    )
    if meta_invalid_parse > invalid_parse_turns:
        invalid_parse_turns = meta_invalid_parse
    llm_calls = int(
        _first_number(
            summary_map.get("llm_calls"),
            summary_map.get("llm_call_count"),
            meta_map.get("llm_call_count"),
            len(turn_rows),
        )
        or 0
    )
    terminal_parse_error = (
        summary_map.get("terminal_parse_error")
        if summary_map.get("terminal_parse_error") is not None
        else meta_map.get("terminal_parse_error")
    )

    out: dict[str, Any] = {}
    if action_counts:
        total = int(sum(action_counts.values()))
        top = action_counts.most_common(16)
        out["action_histogram"] = {
            "total_actions": total,
            "counts": dict(action_counts),
            "shares": _counter_shares(action_counts),
            "top": [{"action": name, "count": count, "share": count / total} for name, count in top],
            "unique_actions": len(action_counts),
        }
    if batch_counts:
        sizes = []
        for size_key, count in batch_counts.items():
            try:
                size = int(size_key)
            except ValueError:
                continue
            sizes.extend([size] * int(count))
        mean_batch = (sum(sizes) / len(sizes)) if sizes else None
        out["batch_size_hist"] = {
            "counts": {key: int(val) for key, val in sorted(batch_counts.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else item[0])},
            "shares": _counter_shares(batch_counts),
            "mean": mean_batch,
            "n_llm_calls": int(sum(batch_counts.values())),
        }
    if noop_reasons or invalid_codes or summary_map.get("noop_reasons") or summary_map.get("invalid_codes"):
        if not noop_reasons and isinstance(summary_map.get("noop_reasons"), Mapping):
            noop_reasons.update({str(k): int(v) for k, v in summary_map["noop_reasons"].items()})
        if not invalid_codes and isinstance(summary_map.get("invalid_codes"), Mapping):
            invalid_codes.update({str(k): int(v) for k, v in summary_map["invalid_codes"].items()})
        out["transition_diagnostics"] = {
            "noop_reasons": dict(noop_reasons),
            "invalid_codes": dict(invalid_codes),
            "noop_reason_shares": _counter_shares(noop_reasons),
            "invalid_code_shares": _counter_shares(invalid_codes),
        }
    if llm_calls or invalid_parse_turns or terminal_parse_error not in (None, False, ""):
        out["parse_stats"] = {
            "llm_calls": llm_calls,
            "invalid_parse_turn_count": invalid_parse_turns,
            "invalid_parse_rate": (
                float(invalid_parse_turns) / float(llm_calls) if llm_calls > 0 else None
            ),
            "terminal_parse_error": terminal_parse_error,
            "terminal_parse_error_bool": bool(terminal_parse_error),
        }
    if continuity_counts or isinstance(summary_map.get("prompt_continuity_counts"), Mapping):
        if not continuity_counts and isinstance(summary_map.get("prompt_continuity_counts"), Mapping):
            continuity_counts.update(
                {str(k): int(v) for k, v in summary_map["prompt_continuity_counts"].items()}
            )
        total_c = float(sum(continuity_counts.values())) or 1.0
        out["prompt_continuity"] = {
            "counts": dict(continuity_counts),
            "rates": {key: float(val) / total_c for key, val in continuity_counts.items()},
            "n_turns": int(sum(continuity_counts.values())),
        }
    if compaction_count or prompt_deltas or dropped_items:
        delta_mean = (
            sum(item["delta"] for item in prompt_deltas) / len(prompt_deltas)
            if prompt_deltas
            else None
        )
        before_mean = (
            sum(item["before"] for item in prompt_deltas) / len(prompt_deltas)
            if prompt_deltas
            else None
        )
        after_mean = (
            sum(item["after"] for item in prompt_deltas) / len(prompt_deltas)
            if prompt_deltas
            else None
        )
        out["compaction"] = {
            "compaction_count": compaction_count,
            "dropped_item_count_total": dropped_items,
            "retained_item_count_total": retained_items,
            "dropped_items_per_compaction": (
                float(dropped_items) / float(compaction_count) if compaction_count else None
            ),
            "prompt_tokens_before_mean": before_mean,
            "prompt_tokens_after_mean": after_mean,
            "prompt_tokens_delta_mean": delta_mean,
            "prompt_reset_events": len(prompt_deltas),
            "prompt_resets": prompt_deltas[:16],
        }
    # Allow fully precomputed blobs.
    for key in (
        "action_histogram",
        "batch_size_hist",
        "transition_diagnostics",
        "parse_stats",
        "prompt_continuity",
        "compaction",
    ):
        if key not in out and isinstance(summary_map.get(key), Mapping):
            out[key] = dict(summary_map[key])
        elif key not in out and isinstance(meta_map.get(key), Mapping):
            out[key] = dict(meta_map[key])
    return out


def _behavior_annotations(
    *,
    rollout_id: str,
    created: str,
    behavior: Mapping[str, Any],
) -> list[RolloutAnnotation]:
    rows: list[RolloutAnnotation] = []

    histogram = behavior.get("action_histogram")
    if isinstance(histogram, Mapping) and histogram:
        top = histogram.get("top") or []
        labels = []
        if isinstance(top, Sequence):
            for item in list(top)[:5]:
                if isinstance(item, Mapping):
                    labels.append(f"{item.get('action')}={item.get('share', 0):.2f}")
        rows.append(
            make_annotation(
                kind="action_histogram",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(histogram),
                ok=True,
            )
        )

    batches = behavior.get("batch_size_hist")
    if isinstance(batches, Mapping) and batches:
        labels = []
        if batches.get("mean") is not None:
            labels.append(f"mean={float(batches['mean']):.2f}")
        counts = batches.get("counts") if isinstance(batches.get("counts"), Mapping) else {}
        for size, count in list(counts.items())[:4]:
            labels.append(f"b{size}={count}")
        rows.append(
            make_annotation(
                kind="batch_size_hist",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(batches),
                ok=True,
            )
        )

    transition = behavior.get("transition_diagnostics")
    if isinstance(transition, Mapping) and transition:
        labels = []
        for reason, count in list((transition.get("noop_reasons") or {}).items())[:4]:
            labels.append(f"noop:{reason}={count}")
        for code, count in list((transition.get("invalid_codes") or {}).items())[:4]:
            labels.append(f"inv:{code}={count}")
        rows.append(
            make_annotation(
                kind="transition_diagnostics",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(transition),
                ok=True,
            )
        )

    parse_stats = behavior.get("parse_stats")
    if isinstance(parse_stats, Mapping) and parse_stats:
        labels = []
        if parse_stats.get("invalid_parse_turn_count") is not None:
            labels.append(f"bad_parse={parse_stats['invalid_parse_turn_count']}")
        if parse_stats.get("invalid_parse_rate") is not None:
            labels.append(f"rate={float(parse_stats['invalid_parse_rate']):.2f}")
        if parse_stats.get("terminal_parse_error_bool"):
            labels.append("terminal_parse_error")
        rows.append(
            make_annotation(
                kind="parse_stats",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(parse_stats),
                ok=True,
            )
        )

    continuity = behavior.get("prompt_continuity")
    if isinstance(continuity, Mapping) and continuity:
        rates = continuity.get("rates") if isinstance(continuity.get("rates"), Mapping) else {}
        labels = [f"{key}={float(val):.2f}" for key, val in list(rates.items())[:4]]
        rows.append(
            make_annotation(
                kind="prompt_continuity",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(continuity),
                ok=True,
            )
        )

    compaction = behavior.get("compaction")
    if isinstance(compaction, Mapping) and compaction:
        labels = []
        if compaction.get("compaction_count") is not None:
            labels.append(f"n={compaction['compaction_count']}")
        if compaction.get("prompt_tokens_delta_mean") is not None:
            labels.append(f"Δtok={float(compaction['prompt_tokens_delta_mean']):.0f}")
        if compaction.get("dropped_item_count_total") is not None:
            labels.append(f"dropped={compaction['dropped_item_count_total']}")
        rows.append(
            make_annotation(
                kind="compaction",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=dict(compaction),
                ok=True,
            )
        )
    return rows


def make_annotation(
    *,
    kind: str,
    rollout_id: str,
    payload: Mapping[str, Any] | None = None,
    labels: Sequence[str] | None = None,
    source: str = "runtime",
    target: Mapping[str, Any] | None = None,
    confidence: float | None = None,
    ok: bool | None = None,
    metadata: Mapping[str, Any] | None = None,
    annotation_id: str | None = None,
    created_at: str | None = None,
) -> RolloutAnnotation:
    return RolloutAnnotation(
        annotation_id=annotation_id or _new_id(kind),
        kind=str(kind),
        rollout_id=str(rollout_id),
        created_at=created_at or utc_now_iso(),
        labels=[str(item) for item in (labels or ())],
        confidence=confidence,
        source=str(source),
        target=dict(target or {}),
        payload=dict(payload or {}),
        ok=ok,
        metadata=dict(metadata or {}),
    )


def annotation_list(
    rollout_id: str,
    annotations: Sequence[RolloutAnnotation],
    *,
    status: str = "ready",
    trace_correlation_id: str | None = None,
) -> RolloutAnnotationList:
    rows = list(annotations)
    return RolloutAnnotationList(
        schema=ROLLOUT_ANNOTATIONS_SCHEMA,
        rollout_id=str(rollout_id),
        status=str(status),
        annotations=rows,
        count=len(rows),
        trace_correlation_id=trace_correlation_id,
    )


def coerce_annotation_list(
    value: RolloutAnnotationList | Sequence[RolloutAnnotation] | Mapping[str, Any] | None,
    *,
    rollout_id: str,
    trace_correlation_id: str | None = None,
) -> RolloutAnnotationList | None:
    if value is None:
        return None
    if isinstance(value, RolloutAnnotationList):
        if not value.rollout_id:
            value.rollout_id = rollout_id
        if value.trace_correlation_id is None and trace_correlation_id is not None:
            value.trace_correlation_id = trace_correlation_id
        return value.sealed()
    if isinstance(value, Mapping):
        rows_raw = value.get("annotations") or []
        rows: list[RolloutAnnotation] = []
        if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes)):
            for item in rows_raw:
                if isinstance(item, RolloutAnnotation):
                    rows.append(item)
                elif isinstance(item, Mapping):
                    rows.append(
                        RolloutAnnotation(
                            annotation_id=str(item.get("annotation_id") or _new_id(str(item.get("kind") or "note"))),
                            kind=str(item.get("kind") or "note"),
                            rollout_id=str(item.get("rollout_id") or rollout_id),
                            created_at=str(item.get("created_at") or ""),
                            labels=[str(x) for x in (item.get("labels") or [])],
                            confidence=(
                                float(item["confidence"])
                                if item.get("confidence") is not None
                                else None
                            ),
                            source=str(item.get("source") or "runtime"),
                            target=_mapping(item.get("target")),
                            payload=_mapping(item.get("payload")),
                            ok=item.get("ok") if isinstance(item.get("ok"), bool) else None,
                            metadata=_mapping(item.get("metadata")),
                        )
                    )
        return annotation_list(
            str(value.get("rollout_id") or rollout_id),
            rows,
            status=str(value.get("status") or "ready"),
            trace_correlation_id=(
                str(value["trace_correlation_id"])
                if value.get("trace_correlation_id") is not None
                else trace_correlation_id
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = [item for item in value if isinstance(item, RolloutAnnotation)]
        return annotation_list(
            rollout_id,
            rows,
            status="ready",
            trace_correlation_id=trace_correlation_id,
        )
    raise TypeError(
        "get_rollout_annotations() must return RolloutAnnotationList, "
        "a sequence of RolloutAnnotation, a mapping, or None"
    )


def derive_annotations_from_execution(execution: ExecutionRecord) -> RolloutAnnotationList:
    """Default annotations derived from a finished ExecutionRecord."""

    rollout_id = execution.rollout_id
    created = execution.updated_at or execution.created_at or utc_now_iso()
    summary = _mapping(execution.summary)
    usage = _mapping(execution.usage)
    metadata = _mapping(execution.metadata)
    rows: list[RolloutAnnotation] = []

    achievements = summary.get("achievements")
    if not isinstance(achievements, Mapping):
        achievements = metadata.get("achievements")
    if isinstance(achievements, Mapping) and achievements:
        unlocked = {str(k): bool(v) for k, v in achievements.items() if v}
        rows.append(
            make_annotation(
                kind="achievement",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=sorted(unlocked),
                payload={
                    "achievements": {str(k): bool(v) for k, v in achievements.items()},
                    "unique_achievements": float(len(unlocked)),
                },
                ok=True,
            )
        )

    termination = metadata.get("termination")
    if isinstance(termination, Mapping) and termination:
        reason = str(termination.get("reason") or termination.get("stop_action") or "other")
        rows.append(
            make_annotation(
                kind="termination",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=[reason],
                payload=dict(termination),
                ok=True,
            )
        )
    elif summary.get("termination_reason") is not None:
        reason = str(summary.get("termination_reason"))
        rows.append(
            make_annotation(
                kind="termination",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=[reason],
                payload={"reason": reason},
                ok=True,
            )
        )

    token_payload: dict[str, Any] = {}
    for key in (
        "generated_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "prompt_tokens",
        "llm_calls",
        "actions",
        "compactions",
    ):
        if summary.get(key) is not None:
            token_payload[key] = summary.get(key)
        elif usage.get(key) is not None:
            token_payload[key] = usage.get(key)
    if usage.get("prompt_tokens") is not None:
        token_payload.setdefault("prompt_tokens", usage.get("prompt_tokens"))
    if usage.get("completion_tokens") is not None:
        token_payload.setdefault("completion_tokens", usage.get("completion_tokens"))
    if token_payload:
        rows.append(
            make_annotation(
                kind="token_stats",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                payload=token_payload,
                ok=True,
            )
        )

    action_payload = _action_stats_payload(summary=summary, metadata=metadata, usage=usage)
    if action_payload:
        labels: list[str] = []
        if action_payload.get("noop_frac") is not None:
            labels.append(f"noop={float(action_payload['noop_frac']):.2f}")
        if action_payload.get("invalid_frac") is not None:
            labels.append(f"invalid={float(action_payload['invalid_frac']):.2f}")
        if action_payload.get("actions_per_llm") is not None:
            labels.append(f"apl={float(action_payload['actions_per_llm']):.2f}")
        rows.append(
            make_annotation(
                kind="action_stats",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                labels=labels,
                payload=action_payload,
                ok=True,
            )
        )

    events: list[Mapping[str, Any]] = []
    turns: list[Mapping[str, Any]] = []
    try:
        trace = execution.trace_payload()
        raw_events = trace.get("events") if isinstance(trace, Mapping) else None
        if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
            events = [item for item in raw_events if isinstance(item, Mapping)]
        inference = trace.get("inference") if isinstance(trace, Mapping) else None
        raw_turns = inference.get("turns") if isinstance(inference, Mapping) else None
        if isinstance(raw_turns, Sequence) and not isinstance(raw_turns, (str, bytes)):
            turns = [item for item in raw_turns if isinstance(item, Mapping)]
        elif isinstance(trace.get("turns"), Sequence):
            turns = [item for item in trace["turns"] if isinstance(item, Mapping)]
    except Exception:
        events = []
        turns = []
    env_progress = extract_env_progress(events, summary=summary, metadata=metadata)
    if env_progress:
        # Enrich the basic achievement row with unlock steps when we have them.
        unlocks = env_progress.get("achievement_unlocks")
        if isinstance(unlocks, Mapping) and unlocks.get("first_unlock_step"):
            for row in rows:
                if row.kind == "achievement":
                    row.payload["first_unlock_step"] = dict(unlocks["first_unlock_step"])
                    if unlocks.get("unique_achievements") is not None:
                        row.payload["unique_achievements"] = unlocks["unique_achievements"]
                    break
        rows.extend(
            _env_progress_annotations(
                rollout_id=rollout_id,
                created=created,
                env_progress=env_progress,
            )
        )

    behavior = extract_behavior_diagnostics(
        events,
        turns,
        summary=summary,
        metadata=metadata,
    )
    if behavior:
        rows.extend(
            _behavior_annotations(
                rollout_id=rollout_id,
                created=created,
                behavior=behavior,
            )
        )

    reward = None
    try:
        reward = float(execution.outcome_reward())
    except (TypeError, ValueError):
        reward = None
    if reward is not None or summary.get("outcome_reward") is not None:
        rows.append(
            make_annotation(
                kind="outcome",
                rollout_id=rollout_id,
                created_at=created,
                source="derived",
                payload={
                    "outcome_reward": (
                        float(summary["outcome_reward"])
                        if summary.get("outcome_reward") is not None
                        else reward
                    ),
                    "status": execution.status,
                    "success_status": execution.success_status,
                },
                ok=True,
            )
        )

    existing = metadata.get("annotations")
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        for item in existing:
            if isinstance(item, Mapping):
                rows.append(
                    make_annotation(
                        kind=str(item.get("kind") or "note"),
                        rollout_id=rollout_id,
                        created_at=str(item.get("created_at") or created),
                        source=str(item.get("source") or "runtime"),
                        labels=[str(x) for x in (item.get("labels") or [])],
                        confidence=(
                            float(item["confidence"])
                            if item.get("confidence") is not None
                            else None
                        ),
                        target=_mapping(item.get("target")),
                        payload=_mapping(item.get("payload")),
                        ok=item.get("ok") if isinstance(item.get("ok"), bool) else None,
                        metadata=_mapping(item.get("metadata")),
                        annotation_id=(
                            str(item["annotation_id"]) if item.get("annotation_id") else None
                        ),
                    )
                )

    status = "ready"
    if str(execution.status).lower() in {"queued", "running", "paused"}:
        status = "pending"
    elif not rows:
        status = "unavailable"

    return annotation_list(
        rollout_id,
        rows,
        status=status,
        trace_correlation_id=execution.trace_correlation_id,
    )
