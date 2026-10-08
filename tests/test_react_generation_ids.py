"""Offline response identity preservation; no network or credentials required."""
import json
import pytest
from synth_containers.policies.react import OpenRouterReAct


@pytest.mark.parametrize('streaming', [False, True])
def test_generation_ids_survive_failed_parse_retry_and_final_response(monkeypatch, streaming):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-only')
    bodies = iter([
        {'id': 'gen-failed', 'choices': [{'message': {'content': 'invalid'}}]},
        {'id': 'gen-success', 'choices': [{'message': {'content': '{"actions":["do"]}'}}]},
    ])
    class Response:
        def __init__(self, body):
            self.headers = {'Content-Type': 'text/event-stream' if streaming else 'application/json'}
            self.raw = ((f'data: {json.dumps(body)}\n\ndata: [DONE]\n\n') if streaming else json.dumps(body)).encode()
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self, n=None):
            chunk, self.raw = (self.raw, b'') if n is None else (self.raw[:n], self.raw[n:])
            return chunk
    monkeypatch.setattr('urllib.request.urlopen', lambda *_a, **_k: Response(next(bodies)))
    policy = OpenRouterReAct(config_id='offline', config={'parse_retries': 1})
    assert policy.plan({'valid_actions': ['do'], 'observation_text': 'test'}) == ['do']
    trace = policy.trace_data()
    assert trace['generation_id'] == 'gen-success'
    assert trace['prior_attempts'][0]['generation_id'] == 'gen-failed'
    assert not trace['fallback']


def test_stream_usage_chunk_without_id_does_not_erase_generation_id():
    policy = OpenRouterReAct(config_id='offline', config={})
    body = policy._consume_sse_text('data: {"id":"gen-first","choices":[]}\n\ndata: {"usage":{"cost":0.01}}\n\n', None)
    assert body['id'] == 'gen-first' and body['usage']['cost'] == .01


def test_failed_final_parse_retains_identity(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-only')
    policy = OpenRouterReAct(config_id='offline', config={})
    monkeypatch.setattr(policy, '_complete', lambda *_: {'id': 'gen-final-failure', 'choices': []})
    assert policy.plan({'valid_actions': ['do']}) == ['do']
    trace = policy.trace_data()
    assert trace['fallback'] and trace['generation_id'] == 'gen-final-failure'
    assert trace['prior_attempts'][0]['generation_id'] == 'gen-final-failure'
