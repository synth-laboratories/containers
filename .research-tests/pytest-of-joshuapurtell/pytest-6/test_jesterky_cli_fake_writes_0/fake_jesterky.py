import json, sys
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index('--out') + 1])
args = json.loads(Path(argv[argv.index('--args-file') + 1]).read_text())
job = args['jobs'][0]
trace = json.loads(Path(job['path']).read_text())
events = trace['events']
first, second = events[0], events[-1]
proposal = {
    'schema_version': 'synth.annotation-proposal.v1',
    'source_trace_id': job['trace_id'],
    'source_trace_digest': job['trace_digest'],
    'findings': [{
        'target': {'kind': 'trace', 'entity_id': job['trace_id']},
        'annotation_type': 'problem_model',
        'labels': ['problem_model.matches_issue'],
        'payload': {},
        'confidence': 0.7,
        'rationale': 'cli fake swarm worker',
        'evidence': [
            {'kind': 'event', 'entity_id': first['event_id']},
            {'kind': 'event', 'entity_id': second['event_id']},
        ],
    }],
    'abstentions': [],
    'judgments': [],
    'summary': 'cli',
}
out.write_text(json.dumps({'recorded': [{'outputs': {'proposal': proposal}}]}))