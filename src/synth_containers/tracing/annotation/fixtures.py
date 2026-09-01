"""A small sealed Craftax-shaped Trace V5 for smoke tests and skill fixtures.

The shape mirrors ``tracing/adapters/craftax.py``: one orchestrator actor plus one
lane actor/session; ``model_call`` spans whose input is a text observation and
whose output is a ``THOUGHT:/ACTIONS:`` reply; ``environment_step`` spans with
``action``/``transition``/``reason``; native ``craftax.transcript`` events for
engine facts (achievements, resource deltas); and ``extensions.craftax``.

No provider is called. Everything is deterministic, so digests are stable across
runs and the fixture can anchor idempotency tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..canonical import record_id
from ..models.actors import (
    ActorKind,
    ActorV5,
    CoverageState,
    SessionCoverageV5,
    SessionStatus,
    SessionV5,
    Visibility,
)
from ..models.completeness import (
    CaptureStatus,
    TerminationV5,
    TraceCompletenessV5,
    TraceLifecycleV5,
    TraceStatus,
)
from ..models.document import TraceCaptureSummaryV5, TraceDocumentV5
from ..models.events import EventOrderV1, EventStatus, EventV5
from ..models.identity import AliasV1, TraceIdentityV5, TraceKind, TraceProvenanceV5
from ..models.messages import MessageNodeV5, MessagePartV5, MessageRole, PartType
from ..models.spans import SpanKind, SpanStatus, SpanV5, UsageProvenance, UsageV5


SMOKE_LANE = "smoke/craftax#glm-5.3-flash:low#s0"
SMOKE_MODEL = "glm-5.3-flash"
SMOKE_TASK = "gamebench/craftax-singleplayer"


def _ts(seconds: int) -> str:
    minute, second = divmod(seconds, 60)
    return f"2026-08-31T00:{minute:02d}:{second:02d}.000000Z"


def _observation(
    *,
    position: tuple[int, int],
    direction: tuple[int, int],
    front_tile: str,
    wood: int,
    achievements: tuple[str, ...],
    local_map: tuple[str, ...],
    sapling: int = 0,
) -> str:
    return "\n".join(
        [
            "level: 0",
            f"position: [{position[0]},{position[1]}] direction=[{direction[0]},{direction[1]}]",
            f"front_tile: {front_tile}",
            "local_map:",
            *local_map,
            "VITALS: health=9  food=9  drink=9  energy=9",
            (
                "inventory: coal=0, diamond=0, iron=0, pickaxe=0, "
                f"sapling={sapling}, stone=0, sword=0, wood={wood}"
            ),
            f"achievements: {', '.join(achievements)}",
            "nearby_entities: ",
            "valid_actions: noop, left, right, up, down, do, sleep, place_stone, place_table, "
            "place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_wood_sword, rest",
        ]
    )


MAP_TREE_UP_RIGHT = (
    ".........",
    "....T....",
    ".........",
    "....P....",
    ".........",
)
MAP_TREE_FRONT = (
    ".........",
    "....T....",
    "....P....",
    ".........",
    ".........",
)


@dataclass(frozen=True, slots=True)
class _Call:
    index: int
    at: int
    observation: str
    reply: str


@dataclass(frozen=True, slots=True)
class _Step:
    index: int
    at: int
    action: str
    transition: str
    reason: str | None = None
    achievement: str | None = None
    resource_delta: tuple[str, int] | None = None


def smoke_script() -> tuple[tuple[_Call, ...], tuple[_Step, ...]]:
    obs0 = _observation(
        position=(24, 24), direction=(0, 1), front_tile="grass", wood=0,
        achievements=(), local_map=MAP_TREE_UP_RIGHT,
    )
    obs1 = obs0
    obs2 = _observation(
        position=(24, 22), direction=(0, -1), front_tile="tree", wood=1,
        achievements=("collect_wood",), local_map=MAP_TREE_FRONT,
    )
    obs3 = obs2
    obs4 = _observation(
        position=(24, 22), direction=(0, -1), front_tile="tree", wood=4,
        achievements=("collect_wood",), local_map=MAP_TREE_FRONT,
    )
    calls = (
        _Call(
            0,
            0,
            obs0,
            "THOUGHT: There is a tree directly in front of me, so I will collect wood now.\n"
            "ACTIONS: do, do",
        ),
        _Call(
            1,
            10,
            obs1,
            "THOUGHT: The tree is two tiles up. I will move up until I face it, then harvest.\n"
            "ACTIONS: up, up, do",
        ),
        _Call(
            2,
            20,
            obs2,
            "THOUGHT: I have enough wood to place a crafting table and craft a pickaxe.\n"
            "ACTIONS: place_table, make_wood_pickaxe",
        ),
        _Call(3, 30, obs3, "ACTIONS: do, do, do"),
        _Call(
            4,
            40,
            obs4,
            "THOUGHT: Wood is 4 now, so the table placement will succeed.\n"
            "ACTIONS: place_table",
        ),
    )
    steps = (
        _Step(0, 2, "do", "noop", "nothing_to_do:grass"),
        _Step(1, 3, "do", "noop", "nothing_to_do:grass"),
        _Step(2, 12, "up", "move"),
        _Step(3, 13, "up", "blocked", "blocked:tree"),
        _Step(4, 14, "do", "harvest", None, "collect_wood", ("wood", 1)),
        _Step(5, 22, "place_table", "noop", "missing_resources"),
        _Step(6, 23, "make_wood_pickaxe", "noop", "needs_crafting_table"),
        _Step(7, 32, "do", "harvest", None, None, ("wood", 1)),
        _Step(8, 33, "do", "harvest", None, None, ("wood", 1)),
        _Step(9, 34, "do", "harvest", None, None, ("wood", 1)),
        _Step(10, 42, "place_table", "place", None, "place_table", ("wood", -1)),
    )
    return calls, steps


def build_craftax_smoke_trace(*, lane: str = SMOKE_LANE, model: str = SMOKE_MODEL) -> TraceDocumentV5:
    """A deterministic, sealed Craftax-like rollout with known belief/plan/recovery facts."""

    calls, steps = smoke_script()
    trace_id = record_id("trace", kind="craftax_smoke", key={"lane": lane, "model": model})
    root_actor_id = record_id("actor", kind="craftax_smoke_root", scope=(trace_id,))
    root_session_id = record_id("sess", kind="craftax_smoke_root", scope=(trace_id,))
    actor_id = record_id("actor", kind="craftax_smoke_lane", scope=(trace_id,), key=lane)
    session_id = record_id("sess", kind="craftax_smoke_lane", scope=(trace_id, actor_id), key=lane)
    started = _ts(0)
    ended = _ts(50)

    actors = (
        ActorV5(
            actor_id=root_actor_id,
            kind=ActorKind.ORCHESTRATOR,
            display_name="Craftax evaluation",
            role="orchestrator",
            harness="suites.nonproduct.craftax",
        ).sealed(),
        ActorV5(
            actor_id=actor_id,
            kind=ActorKind.AGENT,
            display_name=f"{lane}",
            role="policy",
            parent_actor_id=root_actor_id,
            harness="suites.nonproduct.craftax",
            model=model,
            provider="smoke",
            task_id=SMOKE_TASK,
            aliases=(AliasV1(namespace="craftax.lane", value=lane, target_id=actor_id, target_kind="actor"),),
        ).sealed(),
    )
    sessions = (
        SessionV5(
            session_id=root_session_id,
            actor_id=root_actor_id,
            started_at=started,
            ended_at=ended,
            status=SessionStatus.COMPLETED,
            coverage=SessionCoverageV5(agent_events=CoverageState.COMPLETE),
        ).sealed(),
        SessionV5(
            session_id=session_id,
            actor_id=actor_id,
            started_at=started,
            ended_at=ended,
            parent_session_id=root_session_id,
            status=SessionStatus.COMPLETED,
            harness="suites.nonproduct.craftax",
            provider="smoke",
            coverage=SessionCoverageV5(
                model_calls=CoverageState.COMPLETE,
                agent_events=CoverageState.COMPLETE,
                environment_events=CoverageState.COMPLETE,
                usage=CoverageState.AGGREGATE_ONLY,
            ),
        ).sealed(),
    )

    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    events: list[EventV5] = []
    predecessor: str | None = None
    sequence = 0

    def message(message_id: str, role: MessageRole, text: str, at: str, *, span_id: str | None, metadata: dict[str, Any]) -> MessageNodeV5:
        nonlocal predecessor
        node = MessageNodeV5(
            message_id=message_id,
            role=role,
            parts=(MessagePartV5(part_id=f"{message_id}:0", type=PartType.TEXT, text=text),),
            sender_actor_id=actor_id,
            session_id=session_id,
            predecessor_message_ids=(predecessor,) if predecessor else (),
            produced_by_span_id=span_id,
            occurred_at=at,
            metadata=metadata,
        ).sealed()
        predecessor = message_id
        return node

    prefix_id = record_id("msg", kind="craftax_system", scope=(session_id,), key="smoke-prefix")
    messages.append(
        message(
            prefix_id,
            MessageRole.SYSTEM,
            "You control a Craftax agent. Reply with THOUGHT: <one line> then ACTIONS: <comma list>.",
            started,
            span_id=None,
            metadata={"prefix_digest": "smoke"},
        )
    )

    def event(kind: str, at: str, payload: dict[str, Any], *, span_id: str | None, event_type: str = "craftax.transcript") -> None:
        nonlocal sequence
        sequence += 1
        events.append(
            EventV5(
                event_id=record_id("evt", kind="craftax_smoke_event", scope=(trace_id,), key=sequence),
                event_type=event_type,
                actor_id=actor_id,
                session_id=session_id,
                occurred_at=at,
                span_id=span_id,
                order=EventOrderV1(chronological_sequence=sequence, actor_sequence=sequence),
                payload={"lane": lane, "rollout_id": "smoke-rollout", "task_id": SMOKE_TASK, **payload},
                status=EventStatus.OK,
            ).sealed()
        )

    event(
        "phase",
        started,
        {
            "phase": "rollout.opened",
            "policy": {"id": "react_committed_plan", "model": model, "env_seed": 0, "reasoning_effort": "low"},
        },
        span_id=None,
        event_type="craftax.eval.phase",
    )
    call_by_time = {call.at: call for call in calls}
    step_by_time = {step.at: step for step in steps}
    wood = 0
    achievements: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for second in range(0, 50):
        at = _ts(second)
        call = call_by_time.get(second)
        if call is not None:
            span_id = record_id("span", kind="craftax_model_call", scope=(trace_id, session_id), key=call.index)
            prompt_id = record_id("msg", kind="craftax_prompt", scope=(session_id,), key=call.index)
            reply_id = record_id("msg", kind="craftax_reply", scope=(session_id,), key=call.index)
            input_ids = [prefix_id, prompt_id] if call.index == 0 else [prompt_id]
            messages.append(
                message(prompt_id, MessageRole.USER, call.observation, at, span_id=None, metadata={"call_index": call.index, "observation": "text"})
            )
            messages.append(
                message(reply_id, MessageRole.ASSISTANT, call.reply, at, span_id=span_id, metadata={"call_index": call.index, "finish_reason": "stop"})
            )
            spans.append(
                SpanV5(
                    span_id=span_id,
                    span_kind=SpanKind.MODEL_CALL,
                    actor_id=actor_id,
                    session_id=session_id,
                    started_at=at,
                    ended_at=at,
                    status=SpanStatus.OK,
                    input_message_ids=tuple(input_ids),
                    output_message_ids=(reply_id,),
                    usage=UsageV5(
                        provenance=UsageProvenance.OBSERVED_HARNESS,
                        prompt_tokens=400,
                        completion_tokens=40,
                        total_tokens=440,
                        requests=1,
                    ),
                    detail={"call_index": call.index, "model": model, "reasoning_effort": "low", "observation": "text", "finish_reason": "stop"},
                    aliases=(AliasV1(namespace="craftax.policy_call", value=f"{lane}:{call.index}", target_id=span_id, target_kind="span"),),
                ).sealed()
            )
            event("policy.call", at, {"kind": "policy.call", "call_index": call.index, "model": model, "reasoning_effort": "low", "prompt_tokens": 400, "completion_tokens": 40, "finish_reason": "stop"}, span_id=span_id)
        step = step_by_time.get(second)
        if step is not None:
            span_id = record_id("span", kind="craftax_environment_step", scope=(trace_id, session_id), key=step.index)
            spans.append(
                SpanV5(
                    span_id=span_id,
                    span_kind=SpanKind.ENVIRONMENT_STEP,
                    actor_id=actor_id,
                    session_id=session_id,
                    started_at=at,
                    ended_at=at,
                    status=SpanStatus.OK,
                    turn_id=f"{session_id}:step:{step.index}",
                    detail={"step_index": step.index, "action": step.action, "transition": step.transition, "reason": step.reason},
                ).sealed()
            )
            actions.append({"step": step.index, "action": step.action, "transition": step.transition, "reason": step.reason})
            event(
                "action_applied",
                at,
                {"kind": "action_applied", "step_index": step.index, "tick": step.index, "action": step.action, "transition": step.transition, "payload": {"reason": step.reason} if step.reason else {}, "message": f"ActionApplied({step.action},step={step.index})", "severity": "info"},
                span_id=span_id,
            )
            if step.resource_delta is not None:
                resource, delta = step.resource_delta
                wood += delta if resource == "wood" else 0
                event(
                    "resource_delta",
                    at,
                    {"kind": "resource_delta", "step_index": step.index, "action": step.action, "payload": {"resource": resource, "delta": delta, "after": wood}, "message": f"ResourceDelta({resource},{delta:+d})", "severity": "info"},
                    span_id=span_id,
                )
            if step.achievement is not None:
                achievements.append({"step": step.index, "name": step.achievement})
                event(
                    "achievement_unlocked",
                    at,
                    {"kind": "achievement_unlocked", "step_index": step.index, "action": step.action, "payload": {"achievement": step.achievement}, "message": f"AchievementUnlocked({step.achievement})", "severity": "info"},
                    span_id=span_id,
                )
    reward = float(len(achievements))
    event(
        "terminal",
        ended,
        {
            "reward": reward,
            "env_steps": len(steps),
            "stopped_on": "max_steps",
            "terminated": False,
            "truncated": True,
            "usage": {"calls": len(calls), "prompt_tokens": 400 * len(calls), "completion_tokens": 40 * len(calls), "actions_planned": 11, "actions_taken": 11, "actions_wasted": 4},
        },
        span_id=None,
        event_type="craftax.eval.run.terminal",
    )
    usage = UsageV5(
        provenance=UsageProvenance.OBSERVED_HARNESS,
        prompt_tokens=400 * len(calls),
        completion_tokens=40 * len(calls),
        total_tokens=440 * len(calls),
        requests=len(calls),
    )
    document = TraceDocumentV5(
        trace_id=trace_id,
        trace_kind=TraceKind.AGENT_ROLLOUT,
        identity=TraceIdentityV5(rollout_id="smoke-rollout", run_id="smoke-run", correlation_id="smoke", task_id=SMOKE_TASK, benchmark="craftax", seed=0),
        lifecycle=TraceLifecycleV5(status=TraceStatus.COMPLETED, started_at=started, ended_at=ended, termination=TerminationV5(reason="max_steps")),
        capture=TraceCaptureSummaryV5(
            capture_id=record_id("cap", kind="craftax_smoke", scope=(trace_id,)),
            binding_id="smoke",
            binding_digest="sha256:" + "0" * 64,
            capture_profile="craftax_smoke_fixture",
            interception="none",
            mode="fixture",
        ),
        provenance=TraceProvenanceV5(
            producer="synth_containers.tracing.annotation.fixtures",
            producer_version="1",
            source_format="craftax.react-native-events.v1",
            model=model,
            provider="smoke",
            harness="suites.nonproduct.craftax",
            captured_at=ended,
        ),
        completeness=TraceCompletenessV5(
            capture_status=CaptureStatus.COMPLETE,
            terminal_event_observed=True,
            model_calls=CoverageState.COMPLETE,
            raw_provider=CoverageState.UNAVAILABLE,
            agent_events=CoverageState.COMPLETE,
            environment_events=CoverageState.COMPLETE,
            tool_events=CoverageState.NOT_CAPTURED,
            usage=CoverageState.AGGREGATE_ONLY,
        ),
        actors=actors,
        sessions=sessions,
        messages=tuple(messages),
        spans=tuple(spans),
        events=tuple(events),
        usage=usage,
        visibility=Visibility.PRIVATE,
        extensions={
            "craftax": {
                "rollouts": [
                    {
                        "lane": lane,
                        "rollout_id": "smoke-rollout",
                        "actor_id": actor_id,
                        "session_id": session_id,
                        "model": model,
                        "provider": "smoke",
                        "task_id": SMOKE_TASK,
                        "reward": reward,
                        "env_steps": len(steps),
                        "stopped_on": "max_steps",
                        "achievements": achievements,
                        "actions": actions,
                        "calls": [{"call_index": call.index, "prompt_tokens": 400, "completion_tokens": 40} for call in calls],
                    }
                ]
            }
        },
    )
    return document.sealed()




def build_craftax_compaction_trace(*, lane: str = "smoke/craftax#glm-5.3-flash:low#s1", model: str = SMOKE_MODEL) -> TraceDocumentV5:
    """A second sealed fixture for the cases the smoke trace lacks.

    - call 0: a hedged, wrong facing claim (``might`` / ``check``) -> uncertainty acknowledged;
    - a ``context.compaction`` event plus a summary system message between calls 1 and 2;
    - call 2: an inventory claim that was true *before* compaction -> ``belief.stale``;
    - call 2's reply carries the plan in a structured ``tool_call`` part, not text;
    - a step whose ``caused_by_span_ids`` names its model call (explicit linkage).
    """

    from ..models.messages import PartType as _PartType

    trace_id = record_id("trace", kind="craftax_compaction", key={"lane": lane, "model": model})
    actor_id = record_id("actor", kind="craftax_compaction_lane", scope=(trace_id,), key=lane)
    session_id = record_id("sess", kind="craftax_compaction_lane", scope=(trace_id, actor_id), key=lane)
    started, ended = _ts(0), _ts(40)
    actor = ActorV5(actor_id=actor_id, kind=ActorKind.AGENT, display_name=lane, role="policy", harness="suites.nonproduct.craftax", model=model, provider="smoke", task_id=SMOKE_TASK,
                    aliases=(AliasV1(namespace="craftax.lane", value=lane, target_id=actor_id, target_kind="actor"),)).sealed()
    session = SessionV5(session_id=session_id, actor_id=actor_id, started_at=started, ended_at=ended, status=SessionStatus.COMPLETED, harness="suites.nonproduct.craftax", provider="smoke",
                        coverage=SessionCoverageV5(model_calls=CoverageState.COMPLETE, agent_events=CoverageState.COMPLETE, environment_events=CoverageState.COMPLETE, usage=CoverageState.AGGREGATE_ONLY)).sealed()
    obs_a = _observation(position=(24, 24), direction=(0, -1), front_tile="grass", wood=4, achievements=("collect_wood",), local_map=MAP_TREE_UP_RIGHT)
    obs_b = _observation(position=(24, 24), direction=(0, -1), front_tile="table", wood=3, achievements=("collect_wood", "place_table"), local_map=MAP_TREE_FRONT)
    obs_c = obs_b
    messages: list[MessageNodeV5] = []
    spans: list[SpanV5] = []
    events: list[EventV5] = []
    predecessor: str | None = None
    sequence = 0

    def add_message(message_id: str, role: MessageRole, parts: tuple[MessagePartV5, ...], at: str, *, span_id: str | None = None, metadata: dict[str, Any] | None = None) -> MessageNodeV5:
        nonlocal predecessor
        node = MessageNodeV5(message_id=message_id, role=role, parts=parts, sender_actor_id=actor_id, session_id=session_id, predecessor_message_ids=(predecessor,) if predecessor else (), produced_by_span_id=span_id, occurred_at=at, metadata=dict(metadata or {})).sealed()
        messages.append(node)
        predecessor = message_id
        return node

    def text_part(message_id: str, text: str) -> MessagePartV5:
        return MessagePartV5(part_id=f"{message_id}:0", type=PartType.TEXT, text=text)

    def add_event(kind: str, at: str, payload: dict[str, Any], *, span_id: str | None, event_type: str = "craftax.transcript") -> None:
        nonlocal sequence
        sequence += 1
        events.append(EventV5(event_id=record_id("evt", kind="craftax_compaction_event", scope=(trace_id,), key=sequence), event_type=event_type, actor_id=actor_id, session_id=session_id, occurred_at=at, span_id=span_id,
                              order=EventOrderV1(chronological_sequence=sequence, actor_sequence=sequence), payload={"lane": lane, "rollout_id": "compaction-rollout", "task_id": SMOKE_TASK, **payload}, status=EventStatus.OK).sealed())

    def add_call(index: int, at: int, observation: str, reply_parts: Callable[[str], tuple[MessagePartV5, ...]]) -> str:
        span_id = record_id("span", kind="craftax_model_call", scope=(trace_id, session_id), key=index)
        prompt_id = record_id("msg", kind="craftax_prompt", scope=(session_id,), key=index)
        reply_id = record_id("msg", kind="craftax_reply", scope=(session_id,), key=index)
        add_message(prompt_id, MessageRole.USER, (text_part(prompt_id, observation),), _ts(at), metadata={"call_index": index, "observation": "text"})
        add_message(reply_id, MessageRole.ASSISTANT, reply_parts(reply_id), _ts(at), span_id=span_id, metadata={"call_index": index, "finish_reason": "stop"})
        spans.append(SpanV5(span_id=span_id, span_kind=SpanKind.MODEL_CALL, actor_id=actor_id, session_id=session_id, started_at=_ts(at), ended_at=_ts(at), status=SpanStatus.OK, input_message_ids=(prompt_id,), output_message_ids=(reply_id,),
                            usage=UsageV5(provenance=UsageProvenance.OBSERVED_HARNESS, prompt_tokens=400, completion_tokens=40, total_tokens=440, requests=1), detail={"call_index": index, "model": model, "reasoning_effort": "low", "observation": "text"}).sealed())
        return span_id

    def add_step(index: int, at: int, action: str, transition: str, reason: str | None, *, caused_by: str | None = None, achievement: str | None = None) -> str:
        span_id = record_id("span", kind="craftax_environment_step", scope=(trace_id, session_id), key=index)
        spans.append(SpanV5(span_id=span_id, span_kind=SpanKind.ENVIRONMENT_STEP, actor_id=actor_id, session_id=session_id, started_at=_ts(at), ended_at=_ts(at), status=SpanStatus.OK, turn_id=f"{session_id}:step:{index}",
                            caused_by_span_ids=(caused_by,) if caused_by else (), detail={"step_index": index, "action": action, "transition": transition, "reason": reason}).sealed())
        add_event("action_applied", _ts(at), {"kind": "action_applied", "step_index": index, "action": action, "transition": transition, "payload": {"reason": reason} if reason else {}}, span_id=span_id)
        if achievement:
            add_event("achievement_unlocked", _ts(at), {"kind": "achievement_unlocked", "step_index": index, "action": action, "payload": {"achievement": achievement}}, span_id=span_id)
        return span_id

    # call 0: hedged wrong facing claim; plan checks first
    call0 = add_call(0, 0, obs_a, lambda rid: (text_part(rid, "THOUGHT: I might be facing the tree, but I should check the tile first before harvesting.\nACTIONS: noop"),))
    add_step(0, 2, "noop", "noop", None, caused_by=call0)
    # call 1: place the table (wood 4 -> 3)
    call1 = add_call(1, 10, obs_a, lambda rid: (text_part(rid, "THOUGHT: Wood is 4, so I will place the table now.\nACTIONS: place_table"),))
    add_step(1, 12, "place_table", "place", None, caused_by=call1, achievement="place_table")
    # compaction between calls 1 and 2, with a summary that carries the pre-placement count
    add_event("compaction", _ts(15), {"kind": "context.compaction", "summary": "You have 4 wood and stand on grass near a tree."}, span_id=None, event_type="context.compaction")
    summary_id = record_id("msg", kind="craftax_summary", scope=(session_id,), key="compaction-1")
    add_message(summary_id, MessageRole.SYSTEM, (MessagePartV5(part_id=f"{summary_id}:0", type=_PartType.TEXT, text="Summary of earlier context: you have 4 wood and stand on grass near a tree."),), _ts(15), metadata={"compaction": True})
    # call 2: structured reply (reasoning part + tool_call part), stale inventory claim
    def structured(rid: str) -> tuple[MessagePartV5, ...]:
        return (
            MessagePartV5(part_id=f"{rid}:0", type=_PartType.REASONING, text="I have 4 wood, so I can craft a pickaxe right away.", reasoning_availability="captured"),
            MessagePartV5(part_id=f"{rid}:1", type=_PartType.TOOL_CALL, tool_call_id="call-2", tool_name="craftax_actions", arguments_json='{"actions": ["make_wood_pickaxe"]}'),
        )
    call2 = add_call(2, 20, obs_b, structured)
    add_step(2, 22, "make_wood_pickaxe", "craft", None, caused_by=call2, achievement="make_wood_pickaxe")
    # call 3: plain text, aligned
    call3 = add_call(3, 30, obs_c, lambda rid: (text_part(rid, "THOUGHT: The pickaxe is done; go find stone.\nACTIONS: left, left"),))
    add_step(3, 32, "left", "move", None, caused_by=call3)
    add_step(4, 33, "left", "move", None, caused_by=call3)
    achievements = [{"step": 1, "name": "place_table"}, {"step": 2, "name": "make_wood_pickaxe"}]
    add_event("terminal", ended, {"reward": 2.0, "env_steps": 5, "stopped_on": "max_steps", "terminated": False, "truncated": True}, span_id=None, event_type="craftax.eval.run.terminal")
    document = TraceDocumentV5(
        trace_id=trace_id, trace_kind=TraceKind.AGENT_ROLLOUT,
        identity=TraceIdentityV5(rollout_id="compaction-rollout", run_id="smoke-run", correlation_id="smoke-compaction", task_id=SMOKE_TASK, benchmark="craftax", seed=1),
        lifecycle=TraceLifecycleV5(status=TraceStatus.COMPLETED, started_at=started, ended_at=ended, termination=TerminationV5(reason="max_steps")),
        capture=TraceCaptureSummaryV5(capture_id=record_id("cap", kind="craftax_compaction", scope=(trace_id,)), binding_id="smoke", binding_digest="sha256:" + "0" * 64, capture_profile="craftax_smoke_fixture", interception="none", mode="fixture"),
        provenance=TraceProvenanceV5(producer="synth_containers.tracing.annotation.fixtures", producer_version="1", source_format="craftax.react-native-events.v1", model=model, provider="smoke", harness="suites.nonproduct.craftax", captured_at=ended),
        completeness=TraceCompletenessV5(capture_status=CaptureStatus.COMPLETE, terminal_event_observed=True, model_calls=CoverageState.COMPLETE, raw_provider=CoverageState.UNAVAILABLE, agent_events=CoverageState.COMPLETE, environment_events=CoverageState.COMPLETE, tool_events=CoverageState.NOT_CAPTURED, usage=CoverageState.AGGREGATE_ONLY),
        actors=(actor,), sessions=(session,), messages=tuple(messages), spans=tuple(spans), events=tuple(events),
        usage=UsageV5(provenance=UsageProvenance.OBSERVED_HARNESS, prompt_tokens=1600, completion_tokens=160, total_tokens=1760, requests=4),
        visibility=Visibility.PRIVATE,
        extensions={"craftax": {"rollouts": [{"lane": lane, "rollout_id": "compaction-rollout", "actor_id": actor_id, "session_id": session_id, "model": model, "provider": "smoke", "task_id": SMOKE_TASK, "reward": 2.0, "env_steps": 5, "stopped_on": "max_steps", "achievements": achievements, "actions": [], "calls": []}]}},
    )
    return document.sealed()


__all__ = ["SMOKE_LANE", "SMOKE_MODEL", "SMOKE_TASK", "build_craftax_compaction_trace", "build_craftax_smoke_trace", "smoke_script"]
