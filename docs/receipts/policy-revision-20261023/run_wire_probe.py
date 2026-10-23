import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
import importlib.metadata

import uvicorn

root = Path('/Users/joshuapurtell/GitHub')
cb = root / 'workspaces/cookbooks-policy-revisions-20261023/cookbooks/optimizers/gepa/banking77_container'
out = root / 'workspaces/containers-policy-revisions-20261023/artifacts/policy-revision-20261023'
os.environ['BANKING77_STREAM_ROOT'] = str(out / ('wire-store-' + uuid.uuid4().hex))
sys.path.insert(0, str(cb))
spec = importlib.util.spec_from_file_location('fixture_loader', cb / 'test_contract.py')
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
service = loader._load_service()
seen, applied = [], []
async def predict(text, *, policy, system_prompt):
    applied.append(system_prompt)
    return 'card_arrival', {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost_usd': 0}
service._predict_label = predict
@service.app.middleware('http')
async def capture(request, call_next):
    body = await request.json() if request.method in ('POST', 'PUT') else None
    seen.append({'method': request.method, 'path': request.url.path, 'body': body})
    return await call_next(request)
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
port = listener.getsockname()[1]
server = uvicorn.Server(uvicorn.Config(service.app, log_level='error', access_log=False))
thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
thread.start()
try:
    deadline = time.monotonic() + 8
    while not server.started:
        if time.monotonic() > deadline:
            raise TimeoutError('fixture startup')
        time.sleep(.02)
    probe = root / 'workspaces/optimizers-policy-revisions-20261023/target/debug/examples/policy_revision_probe'
    result = subprocess.run([str(probe), f'http://127.0.0.1:{port}'], text=True, capture_output=True, timeout=30)
    (out / 'wire-probe.stdout').write_text(result.stdout)
    (out / 'wire-probe.stderr').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    calls = [row for row in seen if row['path'] == '/rollout' and row['method'] == 'POST']
    assert len(calls) == 3
    assert applied == ['Policy A', 'Policy B', 'Policy A'], applied
    for row in calls:
        assert 'policy_revision_id' in row['body']
        assert not {'candidate', 'candidate_overlay', 'policy'} & row['body'].keys()
    assert len([r for r in seen if r['path'] == '/policy-revisions']) == 2
    receipt = {'schema': 'policy-revision.cross-language-http.v1', 'provider_calls': 0,
        'paid_cost_usd': 0, 'applied_prompts': applied, 'requests': seen,
        'rust_result': json.loads(result.stdout),
        'containers_version': importlib.metadata.version('synth-containers'),
        'producer_source_sha256': hashlib.sha256((cb / 'synth_service_app.py').read_bytes()).hexdigest()}
    (out / 'wire-probe-installed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'rollouts': len(calls), 'provider_calls': 0}))
finally:
    server.should_exit = True
    thread.join(8)
    listener.close()
    assert not thread.is_alive(), 'owned fixture did not stop'

receipt['cleanup_complete'] = True
(out / 'wire-probe-installed.json').write_text(json.dumps(receipt, indent=2) + '\n')
