"""Bidirectional proof on the live engine: consumers steer the annotator mid-rollout (never the policy).

Three rollouts run concurrently on a live Craftax gold engine; while they run,
consumers send control messages through the declared `annotation.control` route:
a note (becomes a finding), a hot-swap to a second installed revision (state
carried), a live threshold change, an early stop, and two refused controls.

    SYNTH_CRAFTAX_URL=http://127.0.0.1:18098 \
    PYTHONPATH=src:<containers-main>/images/craftax-gamebench-rust:<evals> \
    python scripts/live_annotation_craftax_control_e2e.py <out_dir>

Proven 2026-09-01. Writes <out_dir>/receipt.json.
"""
import json, os, sys, threading, time, socket
from pathlib import Path
import urllib.request

os.environ.setdefault("SYNTH_CRAFTAX_URL", "http://127.0.0.1:18098")
os.environ.setdefault("SYNTH_CRAFTAX_MAX_STEPS", "60")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
import uvicorn
from craftax_gold.targets import TARGETS
from synth_containers.platform import create_compat_app
from domains.craftax.annotations import live

app = create_compat_app(TARGETS["craftax_code_policy"], storage_root=OUT / "storage")
sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
BASE = f"http://127.0.0.1:{port}"
for _ in range(100):
    try: urllib.request.urlopen(BASE + "/health", timeout=1); break
    except Exception: time.sleep(0.1)

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")

REV_A = call("PUT", "/annotation-protocol", live.install_body())[1]["protocol_revision_id"]
REV_B = call("PUT", "/annotation-protocol", live.install_body(configuration={"blocked_streak_threshold": 6}))[1]["protocol_revision_id"]
print("revisions", REV_A, REV_B)
T0 = time.monotonic()
log = []
def note(*a): log.append((round(time.monotonic() - T0, 3),) + a); print(f"t={log[-1][0]:7.3f}", *a)

def tail_sse(url, sink, done, on_event=None):
    req = urllib.request.Request(BASE + url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=600) as r:
        while True:
            chunk = r.readline()
            if not chunk: break
            if chunk.startswith(b"data:"):
                row = json.loads(chunk[5:].strip()); row["t"] = round(time.monotonic() - T0, 3); sink.append(row)
                if on_event: on_event(row)
                if row["kind"] == "capture.closed": break
    done.set()

results = {}
def run_seed(seed, plan):
    rid = f"ctl_seed{seed}"
    st, prepared = call("POST", "/rollouts/prepare", {"rollout_id": rid, "task_instance_id": f"seed:{seed}", "telemetry": {"enabled": True, "transport": "websocket", "retention": "run"}, "annotation_protocol_revision_id": REV_A})
    assert st == 200, prepared
    chan = prepared["stream"]["annotation"]; assert chan["control"] and chan["websocket"], chan
    ann, done = [], threading.Event(); fired = set()
    def on_ann(row):
        seq = (row.get("payload") or {}).get("source_sequence") or 0
        for at, name, body in plan:
            if name not in fired and seq >= at:
                fired.add(name)
                st, ack = call("POST", chan["control"], body)
                note(rid, "control", name, "->", st, ack.get("accepted"), ack.get("reason") or ack.get("control_id"))
    threading.Thread(target=tail_sse, args=(chan["stream"], ann, done, on_ann), daemon=True).start()
    time.sleep(0.3)
    st, started = call("POST", "/rollouts", {"rollout_id": rid, "slot": "stream", "task_instance_id": f"seed:{seed}", "submission_mode": "sync", "policy_ref": {"harness": "isolated_policy_process", "config": None}, "telemetry": {"enabled": True, "transport": "websocket", "retention": "run"}, "annotation_protocol_revision_id": REV_A})
    assert st == 200, started
    done.wait(120)
    results[rid] = {"annotation_events": ann, "start": started, "plan": [p[1] for p in plan]}

PLANS = {
    0: [(60, "note", {"op": "message", "control_id": "human-note", "message": {"type": "note", "text": "operator: it keeps hitting do on grass", "author": "josh"}}),
        (120, "swap", {"op": "protocol.update", "control_id": "swap-to-b", "protocol_revision_id": REV_B}),
        (200, "set", {"op": "message", "control_id": "loosen", "message": {"type": "set", "name": "blocked_streak_threshold", "value": 12}})],
    1: [(150, "stop", {"op": "stop", "control_id": "stop-early", "reason": "operator saw enough"})],
    2: [(90, "bad", {"op": "message", "message": {"type": "note", "api_key": "sk-nope"}}),
        (95, "unknown-rev", {"op": "protocol.update", "protocol_revision_id": "anprev_0000000000000000"})],
}
threads = [threading.Thread(target=run_seed, args=(s, PLANS[s])) for s in (0, 1, 2)]
[t.start() for t in threads]; [t.join() for t in threads]

for rid, r in sorted(results.items()):
    ann = r["annotation_events"]
    kinds = [e["kind"] for e in ann]
    closed = next((e["payload"] for e in ann if e["kind"] == "annotation.closed"), {})
    print(f"\n{rid}: rollout {r['start']['status']}, annotation outcome={closed.get('outcome')} findings={closed.get('findings')} controls received={closed.get('controls_received')} refused={closed.get('controls_refused')} rebinds={closed.get('rebinds')} consumed={closed.get('consumed_high_water')}")
    for e in ann:
        if e["kind"] in ("annotation.control.received", "annotation.control.refused", "annotation.protocol.rebound") or (e["kind"] == "annotation.finding" and e["payload"].get("kind") == "note") or (e["kind"] == "annotation.metric" and e["payload"]["name"].startswith("setting")):
            p = e["payload"]
            print(f"   t={e['t']:7.3f} seq={e['sequence']:>4} src={p.get('source_sequence'):>4} {e['kind']:32} {json.dumps({k: v for k, v in p.items() if k in ('op','control_id','reason','applied','handled','label','name','value','previous_protocol_revision_id','protocol_revision_id','state_carried')})}")
    revs = [e["payload"].get("protocol_revision_id") for e in ann if e["kind"] == "annotation.finding"]
    print(f"   finding revisions in order: {[r[-6:] if r else None for r in revs]}")
(OUT / "receipt.json").write_text(json.dumps(results, indent=1, default=str))
server.should_exit = True
