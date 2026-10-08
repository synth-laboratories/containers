"""Source-preserving RuneBench event streams -> sealed Trace V5 and evidence.

No provider calls. Native decision IDs join concurrent actors; clock proximity
never pairs calls. Original payloads and file/line digests remain inspectable.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from ..canonical import bytes_digest, canonical_bytes, record_id
from ..models.actors import ActorV5, SessionV5, SessionCoverageV5
from ..models.document import TraceDocumentV5, TraceCaptureSummaryV5
from ..models.identity import TraceIdentityV5, TraceProvenanceV5
from ..models.completeness import TraceLifecycleV5, TraceCompletenessV5
from ..models.events import EventV5, EventOrderV1
from ..models.messages import MessageNodeV5, MessagePartV5
from ..models.spans import SpanV5
from ..models.evidence import TraceEvidenceBundleV5, TraceRefV5
from ..models.standards import RewardDefinitionV1, RewardRecordV1
from ..models.selectors import selector_for
from ..projections.rollout_inspector import rollout_inspector_from_sealed

VERSION = '1.0.0'

def import_runebench(run: Path):
    run = Path(run)
    manifest = json.loads((run / 'matrix_run_manifest.json').read_text())
    result = json.loads((run / 'episode/job_result.json').read_text())
    config = manifest['config']
    source_digests = {}
    rows = []
    for name in ['events.jsonl', 'episode/events.jsonl']:
        path = run / name
        if not path.exists():
            continue
        source_digests[name] = bytes_digest(path.read_bytes())
        for line, raw in enumerate(path.read_text().splitlines(), 1):
            if raw.strip():
                rows.append((name, line, json.loads(raw)))
    source_digests['episode/job_result.json'] = bytes_digest((run / 'episode/job_result.json').read_bytes())
    source_digests['matrix_run_manifest.json'] = bytes_digest((run / 'matrix_run_manifest.json').read_bytes())
    origin = result.get('cutoff', {}).get('baseline', {}).get('at')
    origin = origin or next((r.get('occurred_at') for _, _, r in rows if r.get('occurred_at')), '1970-01-01T00:00:00Z')
    def timestamp(ms):
        return (datetime.fromisoformat(origin.replace('Z', '+00:00')) + timedelta(milliseconds=ms)).isoformat()
    def ident(kind, value):
        return record_id(kind, kind=kind, scope=(run.name, VERSION), key=value)
    actors = list(config['scenario']['actors'])
    actors.append({'id': 'environment', 'role': 'environment', 'team': ''})
    sessions = {a['id']: ident('session', a['id']) for a in actors}
    ids = {a['id']: ident('actor', a['id']) for a in actors}
    events = []; messages = []; spans = []; calls = defaultdict(list)
    previous_message = {}
    for name, line, row in sorted(rows, key=lambda entry: (entry[2].get('payload', {}).get('elapsedMs', 0), entry[0], entry[1])):
        p = dict(row.get('payload', {})); kind = p.get('kind', row.get('kind', 'event'))
        native_actor = row.get('actor_id') or 'environment'
        if native_actor not in ids:
            raise ValueError(f'Unknown actor {native_actor}')
        actor = ids[native_actor]; session = sessions[native_actor]
        decision = p.get('decisionId') or p.get('action', {}).get('decisionId')
        ms = p.get('elapsedMs', 0); occurred = timestamp(ms)
        event_id = ident('event', f'{name}:{line}')
        mapped = {'model.requested': 'model_call.started', 'model.completed': 'model_call.completed',
                  'policy.action': 'tool.called', 'policy.result': 'tool.result',
                  'message.observed': 'coordination.message.observed'}.get(kind, kind)
        detail = {**p, 'native_kind': kind, 'native_actor_id': native_actor, 'decision_id': decision,
                  'elapsed_ms': ms, 'source': {'file': name, 'line': line, 'digest': source_digests[name]}}
        if decision:
            calls[(native_actor, decision)].append((event_id, kind, occurred, detail))
        message_id = None
        part = None; role = 'assistant'
        if kind == 'model.requested':
            role = 'user'; part = MessagePartV5(ident('part', event_id), 'observation', structured=p.get('observation', {}))
        elif kind == 'model.completed':
            part = MessagePartV5(ident('part', event_id), 'text', text=p.get('content', ''))
        elif kind == 'policy.action':
            action = p.get('action', {})
            part = MessagePartV5(ident('part', event_id), 'tool_call', tool_call_id=decision, tool_name=action.get('type'), arguments_json=json.dumps(action))
        elif kind == 'policy.result':
            role = 'tool'; part = MessagePartV5(ident('part', event_id), 'tool_result', tool_call_id=decision, structured={'result': p.get('result')}, is_error=isinstance(p.get('result'), dict) and p['result'].get('success') is False)
        if part:
            message_id = ident('message', event_id)
            parts = (part,)
            if kind == 'model.completed' and p.get('reasoning'):
                parts = (MessagePartV5(ident('part', event_id + ':reasoning'), 'reasoning', text=p['reasoning']), part)
            messages.append(MessageNodeV5(message_id, role, parts, actor, session,
                predecessor_message_ids=(previous_message[actor],) if actor in previous_message else (),
                occurred_at=occurred, produced_by_event_id=event_id, metadata={'decision_id': decision}).sealed())
            previous_message[actor] = message_id
        events.append(EventV5(event_id, mapped, actor, session, occurred, message_id=message_id,
            order=EventOrderV1(chronological_sequence=len(events)), payload=detail,
            raw_source_ref=f'{name}:{line}').sealed())
    # Reward deltas are derived from consecutive authoritative ticks, never
    # inferred from action completion or assigned causally to a nearby call.
    engine = run / 'episode/engine-states.jsonl'
    cutoff = result.get('cutoff', {})
    previous = {a: v['xp'] for a, v in cutoff.get('baseline', {}).get('actors', {}).items() if v}
    final_ms = cutoff.get('final', {}).get('elapsedMs')
    if engine.exists() and final_ms is not None:
        source_digests['episode/engine-states.jsonl'] = bytes_digest(engine.read_bytes())
        for line, raw in enumerate(engine.read_text().splitlines(), 1):
            state = json.loads(raw); ms = state['elapsedMs']
            if ms > final_ms: continue
            for a, v in state['actors'].items():
                if v is None or a not in previous: continue
                delta = v['xp'] - previous[a]; previous[a] = v['xp']
                if not delta: continue
                events.append(EventV5(ident('event', f'reward:{line}:{a}'), 'environment.reward', ids[a], sessions[a], timestamp(ms),
                    order=EventOrderV1(chronological_sequence=len(events)), payload={
                        'elapsed_ms': ms, 'native_actor_id': a, 'value': delta, 'units': 'XP',
                        'cumulative': v['xp'] - cutoff['baseline']['actors'][a]['xp'],
                        'provenance': 'derived from consecutive complete engine ticks',
                        'source': {'file': 'episode/engine-states.jsonl', 'line': line, 'digest': source_digests['episode/engine-states.jsonl']}
                    }).sealed())
    for (a, decision), records in calls.items():
        requested = next((x for x in records if x[1] == 'model.requested'), None)
        completed = next((x for x in records if x[1] == 'model.completed'), None)
        if not requested: continue
        input_ids = tuple(m.message_id for m in messages if m.produced_by_event_id == requested[0])
        output_ids = tuple(m.message_id for m in messages if completed and m.produced_by_event_id == completed[0])
        spans.append(SpanV5(ident('span', f'{a}:{decision}'), 'model_call', ids[a], sessions[a], requested[2],
            ended_at=completed[2] if completed else None, status='ok' if completed else 'truncated',
            input_message_ids=input_ids, output_message_ids=output_ids,
            detail={'decision_id': decision, 'native_actor_id': a, 'elapsed_ms': requested[3]['elapsed_ms'],
                    'input': requested[3].get('observation'), 'input_messages_availability': 'recorded_in_source_event' if requested[3].get('messages') else 'not_recorded',
                    'reasoning_availability': 'recorded' if completed and completed[3].get('reasoning') else 'unavailable',
                    **({'reasoning_effort':config['reasoning_effort']} if config.get('reasoning_effort') else {}),
                    'output': completed[3].get('content') if completed else None,
                    'usage': completed[3].get('usage') if completed else None,
                    'event_ids': [x[0] for x in records]}).sealed())
    # One global order across streams, preserving source ordinals independently.
    from dataclasses import replace
    events = [replace(e, order=EventOrderV1(chronological_sequence=i)).sealed()
              for i, e in enumerate(sorted(events, key=lambda e: (e.payload['elapsed_ms'], e.event_id)))]
    digest = bytes_digest(canonical_bytes(source_digests)); capture_id = ident('capture', digest)
    ended = max((e.occurred_at for e in events), default=origin)
    document = TraceDocumentV5(ident('trace', digest), 'evaluation_attempt',
        TraceIdentityV5(run_id=run.name, episode_id=run.name, benchmark='runebench'),
        TraceLifecycleV5('completed' if result.get('status') == 'evaluated' else 'failed', origin, ended),
        TraceCaptureSummaryV5(capture_id, ident('binding', digest), digest, 'native-import', 'none', 'import', raw_record_count=len(rows)),
        TraceProvenanceV5('runebench-adapter', VERSION, source_format='evals.event-stream.v1', transformation_chain=(f'runebench:{VERSION}',), extra={'sources': source_digests}),
        TraceCompletenessV5('partial', True, model_calls='partial', agent_events='complete', environment_events='partial', tool_events='complete', usage='partial', reasons=(('Provider request messages and returned reasoning fields are retained in source events; provider-internal reasoning is not assumed.' if config.get('reasoning_effort') else 'Provider request messages and reasoning text were not recorded; observations and responses are retained.'),)),
        actors=tuple(ActorV5(ids[a['id']], 'environment' if a['id']=='environment' else 'agent', a['id'], role=a['role'], metadata={'team': a.get('team')}).sealed() for a in actors),
        sessions=tuple(SessionV5(sessions[a['id']], ids[a['id']], origin, ended_at=ended, status='completed', coverage=SessionCoverageV5(model_calls='partial', agent_events='complete', environment_events='partial', tool_events='complete')).sealed() for a in actors),
        messages=tuple(messages), events=tuple(events), spans=tuple(spans)).sealed()
    definition = RewardDefinitionV1('runebench.woodcutting_xp', 'Woodcutting XP', 'XP gained at the last complete tick before cutoff', 'environment', 'terminal', 'actor', units='XP').sealed()
    rewards = tuple(RewardRecordV1(ident('reward', a), definition.reward_id, definition.version, definition.content_digest,
        selector_for(document, kind='actor', entity_id=ids[a]), value, 'authoritative engine cutoff receipt', ended,
        actor_id=ids[a], session_id=sessions[a], components={'woodcutting_xp': value}).sealed() for a, value in result.get('actor_scores', {}).items())
    evidence = TraceEvidenceBundleV5(ident('evidence', digest), TraceRefV5(document.trace_id, document.content_digest), ended,
        reward_definitions=(definition,), reward_records=rewards).sealed()
    return document, evidence

def runebench_projection(run: Path):
    document, evidence = import_runebench(run)
    return document, evidence, rollout_inspector_from_sealed(document, evidence)
