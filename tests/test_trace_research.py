from dataclasses import replace

import pytest

from synth_containers.tracing.annotation.fixtures import build_craftax_smoke_trace
from synth_containers.tracing.models.selectors import resolve_selector, TraceSelectorV1
from synth_containers.tracing.projections.inspector import InspectedBundle
from synth_containers.tracing.research import SCHEMA, ResearchIndex, related_entities, validate_query, read_source
from synth_containers.tracing.store.bundle import LocalTraceBundle


def query(**kw):
    return {'schemaVersion': SCHEMA, 'evalJobIds': ['job'], **kw}


def test_core_queries_keep_zero_missing_failed_and_450_rows(tmp_path):
    index = ResearchIndex(tmp_path/'cache.sqlite')
    episodes = [{'jobId':'job','trialId':str(i),'reward':0 if i%2 else None,'status':'failed' if i%2 else 'completed'} for i in range(450)]
    first = index.execute(query(limit=2), episodes)
    assert first['resultCount'] == 450
    assert not first['truncated']
    assert index.execute(query(limit=200),episodes)['resultDigest'] == first['resultDigest']
    mean = index.execute(query(aggregate='reward'), episodes)['facets']['rows'][0]
    assert (mean['measuredCount'],mean['missingCount'],mean['rewardMean']) == (225,225,0)
    index.close()
    index=ResearchIndex(tmp_path/'cache.sqlite')
    assert index.execute(query(),episodes)['resultIds']==first['resultIds']
    assert first['facets']['rows'][0]['traceAvailability']=='not_produced'
    index.close()


def test_annotation_exists_does_not_multiply_rewards_and_revision_changes_snapshot(tmp_path):
    index=ResearchIndex(tmp_path/'cache.sqlite')
    source=[{'jobId':'job','trialId':'t','reward':1,'traceDigest':'sha256:t'}]
    annotations=[{'annotationId':str(i),'itemId':str(i),'traceDigest':'sha256:t','label':['loop'],'score':i,'reviewState':'unreviewed','evidenceDigest':'head1'} for i in range(3)]
    q=query(annotationWhere=[{'field':'label','op':'contains','value':'loop'}],aggregate='reward')
    first=index.execute(q,source,annotations)
    assert first['facets']['rows'][0]['measuredCount']==1
    annotations[0]={**annotations[0],'reviewState':'accepted','evidenceDigest':'head2'}
    second=index.execute(q,source,annotations)
    assert first['snapshotId']!=second['snapshotId']
    assert first['facets']['provenance']['annotations'][0]['reviewState']=='unreviewed'
    index.close()


def test_entity_status_is_not_prefiltered_by_episode_status(tmp_path):
    trace=build_craftax_smoke_trace()
    index=ResearchIndex(tmp_path/'cache.sqlite')
    index.index_records('fixture',[InspectedBundle(trace,None)])
    source=[{'jobId':'job','trialId':'t','traceDigest':trace.content_digest,'cacheKey':'fixture','status':'completed'}]
    result=index.execute(query(grain='entities',where=[{'field':'kind','value':'event'}]),source)
    assert result['resultCount']==len(trace.events)
    for row in result['facets']['rows']:
        assert resolve_selector(trace,TraceSelectorV1(**row['selector'])).resolved
    status=str(trace.events[0].status)
    filtered=index.execute(query(grain='entities',where=[{'field':'kind','value':'event'},{'field':'status','value':status}]),source)
    assert filtered['resultCount']>=1
    index.close()


def test_sequence_relation_never_crosses_actors():
    def row(i, actor, order, status='error'):
        return dict(itemId=i,kind='event',actorId=actor,sessionId='s',sourceOrder=order,recordedOrder=order,status=status,action='move',selector={'entity_id':i})
    a,b,c=row('a','one',1),row('b','two',2),row('c','one',3,'ok')
    result=related_entities([a,b,c],[a,b,c],'repeated_failed_action')
    assert len(result)==1 and result[0]['itemId']=='a' and result[0]['relatedSelector']['entity_id']=='c'


def test_pairs_report_missing_and_ambiguous_keys(tmp_path):
    index=ResearchIndex(tmp_path/'cache.sqlite')
    def arm(job,t,reward,**kw): return dict(jobId=job,trialId=t,taskId='task',seed=1,reward=reward,environment='craftax',environmentVersion='engine-v1',definitionDigest='sha256:reward',**kw)
    q=query(evalJobIds=['a','b'],aggregate='paired_reward')
    out=index.execute(q,[arm('a','a',0),arm('b','b',1)])['facets']['rows']
    assert out[0]['rewardDelta']==1
    unknown=[arm('a','a',0),arm('b','b',1)]
    for row in unknown:row['environmentVersion']=None
    assert all(row['matchStatus']=='unknown_environment_version' and row.get('rewardDelta') is None for row in index.execute(q,unknown)['facets']['rows'])
    out=index.execute(q,[arm('a','a',0),arm('a','a2',1),arm('b','b',1)])['facets']['rows']
    assert out[0]['matchStatus']=='ambiguous' and out[0]['rewardDelta'] is None
    repeated=[arm('a','a',0),arm('b','b',1)]
    for row in repeated:row.update(seed=None,repeat=0)
    out=index.execute(q,repeated)['facets']['rows']
    assert out[0]['matchStatus']=='matched' and out[0]['rewardDelta']==1
    for row in repeated:row['repeat']=None
    assert all(row['matchStatus']=='missing_match_key' for row in index.execute(q,repeated)['facets']['rows'])
    index.close()


def test_episode_definition_requires_unique_valid_trace_wide_agreement():
    from synth_containers.tracing.research import bind_episode_reward_definition
    reward = {'reward': 0, 'current': True, 'valid': True, 'selector': {'kind': 'trace'},
              'definitionDigest': 'sha256:definition', 'rewardId': 'metric', 'rewardVersion': '1', 'units': 'score'}
    row = {'reward': 0}
    bind_episode_reward_definition(row, [reward])
    assert row['definitionDigest'] == reward['definitionDigest'] and row['reward'] == 0
    for records in ([{**reward, 'valid': False}], [{**reward, 'current': False}],
                    [{**reward, 'selector': {'kind': 'event'}}], [{**reward, 'reward': 1}],
                    [reward, {**reward, 'definitionDigest': 'sha256:different'}]):
        row = {'reward': 0}
        bind_episode_reward_definition(row, records)
        assert row == {'reward': 0}
    pinned = {'reward': 0, 'definitionDigest': 'sha256:explicit'}
    bind_episode_reward_definition(pinned, [reward])
    assert pinned['definitionDigest'] == 'sha256:explicit'


def test_archive_source_resolves_exact_digest_with_paging(tmp_path):
    trace=build_craftax_smoke_trace()
    bundle=LocalTraceBundle(tmp_path/'bundle')
    from synth_containers.tracing.capture.binding import BindingCaptureV1, BindingWorkloadV1, WorkloadKind, mint_binding
    binding=mint_binding(trace_id=trace.trace_id,capture_id=trace.capture.capture_id,workload=BindingWorkloadV1(kind=WorkloadKind.OTHER,root_actor_id=trace.actors[0].actor_id,actor_session_id=trace.sessions[0].session_id),capture=BindingCaptureV1(output_artifact_root=str(bundle.root)),trace_kind=trace.trace_kind)
    bundle.write_binding(binding)
    bundle.write_trace(trace, binding=binding, segments=())
    bundle.write_manifest()
    archive=tmp_path/'trace.zip'
    bundle.write_archive(archive)
    selector=TraceSelectorV1(trace_id=trace.trace_id,trace_digest=trace.content_digest,kind='event',entity_id=trace.events[0].event_id).to_dict()
    first=read_source(dict(operation='source',archivePath=str(archive),selector=selector,limit=12))
    assert first['resolved'] and len(first['resolved_text'])==12 and first['nextOffset']==12
    second=read_source(dict(operation='source',archivePath=str(archive),selector=selector,offset=12,limit=12))
    assert first['textDigest']==second['textDigest']
    selector['trace_digest']='sha256:wrong'
    assert not read_source(dict(operation='source',archivePath=str(archive),selector=selector))['resolved']


def test_invalid_queries_do_not_become_broad_queries():
    for q in [query(sql='select *'),query(grain='surprise'),query(limit=201),query(where=[{'field':'reward','op':'gte','value':float('nan')}]),query(aggregate='reward',grain='entities'),query(evalJobIds=['left','right'],aggregate='paired_reward',groupBy=['model'])]:
        with pytest.raises(ValueError):validate_query(q)


def test_sequence_relation_requires_actor_session_and_recorded_order():
    def row(i,actor,session,order):
        return dict(itemId=i,kind='event',actorId=actor,sessionId=session,recordedOrder=order,status='error',action='move',selector={'entity_id':i})
    for rows in ([row('a',None,'s',1),row('b',None,'s',2)],
                 [row('a','actor',None,1),row('b','actor',None,2)],
                 [row('a','actor','first',1),row('b','actor','second',2)],
                 [row('a','actor','s',None),row('b','actor','s',2)]):
        assert related_entities(rows,rows,'repeated_failed_action')==[]


def test_recorded_acknowledgement_query_resolves_typed_coordination_and_absence_is_unknown(tmp_path):
    from synth_containers.tracing.models.coordination import CoordinationGraphV1, InteractionEdgeV1, TraceAnchorV1
    trace=build_craftax_smoke_trace()
    anchor=TraceAnchorV1(basis='canonical',entity_kind='actor',entity_id=trace.actors[0].actor_id)
    edges=tuple(InteractionEdgeV1(interaction_id=f'communication-{i}',kind='send_message',source=anchor,target=anchor,started_sequence=i,started_at='2026-09-08T00:00:00Z',status=status,correlation_id='reused-native-id').sealed() for i,status in enumerate(['acknowledged','delivered']))
    trace=replace(trace,coordination=CoordinationGraphV1(interaction_edges=edges).sealed()).sealed()
    index=ResearchIndex(tmp_path/'coordination.sqlite');index.index_records('coordination',[InspectedBundle(trace,None)])
    episodes=[{'jobId':'job','trialId':'trial','traceDigest':trace.content_digest,'cacheKey':'coordination','status':'completed'}]
    q=query(grain='entities',relation='recorded_link',where=[{'field':'kind','value':'interaction'},{'field':'status','value':'acknowledged'}])
    result=index.execute(q,episodes)
    assert result['resultCount']==1
    row=result['facets']['rows'][0]
    assert row['itemId']=='communication-0'
    assert resolve_selector(trace,TraceSelectorV1(**row['selector'])).resolved
    assert result['facets']['relationshipEvidence']['absenceMeaning']=='unknown'
    q['where'][1]['value']='failed'
    missing=index.execute(q,episodes)
    assert missing['resultCount']==0 and missing['facets']['relationshipEvidence']['absenceMeaning']=='unknown'
    index.close()
