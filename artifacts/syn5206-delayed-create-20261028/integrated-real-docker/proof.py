#!/usr/bin/env python3
"""EOF during an injected Docker create delay; real daemon, owned allocation."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

root = Path(__file__).resolve().parent
marker = root / 'create-entered.txt'
receipt = root / 'engine-receipt.json'
if marker.exists() or receipt.exists():
    raise SystemExit('Refusing to overwrite a previous proof')
owner = 'slowcreate-' + uuid.uuid4().hex
py = '/Users/joshuapurtell/GitHub/worktrees/codex-eval-containers-20261028/.venv/bin/python'
request = {'owner': owner, 'receipt': str(receipt),
           'image': 'sha256:d82b2771566e9b3bcddb21b2e25e66b1e76aa0e5bc47d17b82af66d3fce61cfc',
           'cpu': 1, 'memory_mb': 512, 'port': 8098,
           'command': ['/opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_gold', '--host', '0.0.0.0', '--port', '8098'],
           'docker_cli': str(root / 'docker-delayed'), 'heartbeat_seconds': 1}
(root / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
with (root / 'guard.stderr').open('w') as stderr, (root / 'events.jsonl').open('w') as events:
    proc = subprocess.Popen([py, '-c', 'import time; import synth_containers.engine_guard as g; w=g.durable_write; g.durable_write=lambda *a,**k:(time.sleep(6),w(*a,**k))[1]; g.main()'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                            cwd='/Users/joshuapurtell/GitHub/worktrees/codex-eval-containers-20261028')
    updates = queue.Queue()
    def read_events():
        for line in proc.stdout:
            events.write(line)
            events.flush()
            updates.put(json.loads(line))
    threading.Thread(target=read_events, daemon=True).start()
    hello = updates.get(timeout=30)
    if hello.get('event') != 'hello':
        raise RuntimeError('guard did not send its initial hello')
    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()
    print('guard hello; waiting for injected create delay', flush=True)
    deadline = time.monotonic() + 120
    while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(.1)
    entered = marker.exists()
    closed_at = time.time()
    proc.stdin.close()
    print(f'parent pipe closed; create_entered={entered}', flush=True)
    while proc.poll() is None:
        try:
            event = updates.get(timeout=2)
            print('guard event: ' + json.dumps({k: event.get(k) for k in ('event', 'state', 'phase', 'at')}), flush=True)
        except queue.Empty:
            print('guard still running; awaiting owned cleanup', flush=True)
    rc = proc.wait()
    time.sleep(.2)
    guard = json.loads(receipt.read_text()) if receipt.exists() else None
    def inspect():
        return subprocess.run(['/opt/homebrew/bin/docker', 'ps', '-a', '--filter', f'label=ai.synth.eval.owner={owner}', '--format', '{{.ID}} {{.State}}'], capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    immediately = inspect()
    time.sleep(3)
    after = inspect()
    transitions = [e.get('state') for e in (guard or {}).get('transitions', [])]
    result = {'owner': owner, 'guard_pid': proc.pid, 'parent_pipe_closed_at': closed_at,
              'injected_delay_seconds': 8, 'create_entered_before_close': entered, 'guard_exit': rc,
              'cleanup': (guard or {}).get('cleanup'), 'error': (guard or {}).get('error'),
              'create_seconds': (guard or {}).get('create_seconds'), 'transitions': transitions,
              'containers_immediately_after': immediately, 'containers_after_3s': after}
    receipt_heartbeats = sum(1 for line in (root / 'events.jsonl').read_text().splitlines() if json.loads(line).get('phase') == 'receipt')
    result['receipt_heartbeats'] = receipt_heartbeats
    result['proved'] = (receipt_heartbeats >= 1 and entered and rc == 0 and guard is not None and guard.get('cleanup') == 'removed'
                        and guard.get('error') == 'native_engine_startup_cancelled'
                        and (guard.get('create_seconds') or 0) >= 8
                        and 'created' in transitions and not immediately and not after)
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    sys.exit(0 if result['proved'] else 1)
