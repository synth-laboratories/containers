"""Real Craftax rollouts with craftax.live.v1 subscribed: prove the annotation stream is incremental.

Runs the compat facade in-process against a live Craftax gold engine and tails
both the rollout stream and the declared annotation stream over real SSE while
each rollout runs, then checks the poll authority against what SSE delivered.

    SYNTH_CRAFTAX_URL=http://127.0.0.1:18098 \
    PYTHONPATH=src:<containers-main>/images/craftax-gamebench-rust:<evals> \
    python scripts/live_annotation_craftax_e2e.py <out_dir> 0,1,2

Writes <out_dir>/receipt.json. Proven 2026-09-01 (see docs/specs/live-annotation-protocol-v1.md).
"""
import json, os, sys, threading, time, socket
from pathlib import Path
import urllib.request

os.environ.setdefault("SYNTH_CRAFTAX_URL", "http://127.0.0.1:18098")
os.environ.setdefault("SYNTH_CRAFTAX_MAX_STEPS", "60")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]

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
    try:
        urllib.request.urlopen(BASE + "/health", timeout=1); break
    except Exception: time.sleep(0.1)

def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")

info = call("GET", "/info")[1]
assert info["capabilities"]["operations"]["annotation.live"] is True
status, installed = call("PUT", "/annotation-protocol", live.install_body())
assert status == 200, installed
REV = installed["protocol_revision_id"]
print("protocol", REV, "idempotent", installed["idempotent"], "receipt", installed.get("isolation_receipt", {}).get("sandbox"))

T0 = time.monotonic()
def tail_sse(url, sink, done):
    """Consume SSE as a real remote viewer would; record arrival time per event."""
    req = urllib.request.Request(BASE + url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=600) as r:
        buf = b""
        while True:
            chunk = r.readline()
            if not chunk: break
            if chunk.startswith(b"data:"):
                row = json.loads(chunk[5:].strip())
                sink.append({"t": round(time.monotonic() - T0, 3), "kind": row["kind"], "sequence": row.get("sequence"), "digest": row.get("digest"), "payload": row.get("payload")})
                if row["kind"] == "capture.closed": break
    done.set()

results = {}
def run_seed(seed):
    rid = f"live_e2e_seed{seed}"
    st, prepared = call("POST", "/rollouts/prepare", {"rollout_id": rid, "task_instance_id": f"seed:{seed}", "telemetry": {"enabled": True, "transport": "sse", "retention": "run"}, "annotation_protocol_revision_id": REV})
    assert st == 200, prepared
    stream = prepared["stream"]; chan = stream["annotation"]; assert chan and chan["stream"], stream
    roll_ev, ann_ev = [], []; roll_done, ann_done = threading.Event(), threading.Event()
    threading.Thread(target=tail_sse, args=(stream["transports"]["sse"]["url"], roll_ev, roll_done), daemon=True).start()
    threading.Thread(target=tail_sse, args=(chan["stream"], ann_ev, ann_done), daemon=True).start()
    time.sleep(0.3)
    t_start = round(time.monotonic() - T0, 3)
    st, started = call("POST", "/rollouts", {"rollout_id": rid, "slot": "stream", "task_instance_id": f"seed:{seed}", "submission_mode": "sync", "policy_ref": {"harness": "isolated_policy_process", "config": None}, "telemetry": {"enabled": True, "transport": "sse", "retention": "run"}, "annotation_protocol_revision_id": REV})
    t_end = round(time.monotonic() - T0, 3)
    assert st == 200, started
    roll_done.wait(120); ann_done.wait(120)
    # poll authority must agree with SSE
    page = call("GET", chan["events"] + "?after=0&limit=10000")[1]
    polled = [(e["sequence"], e["digest"]) for e in page["events"] if e.get("sequence") is not None]
    sse = [(e["sequence"], e["digest"]) for e in ann_ev if e.get("sequence") is not None]
    results[rid] = {"seed": seed, "start": started, "t_start": t_start, "t_end": t_end, "rollout_events": roll_ev, "annotation_events": ann_ev, "poll_matches_sse": polled == sse, "poll_summary": page.get("summary"), "descriptor": stream}
threads = [threading.Thread(target=run_seed, args=(s,)) for s in SEEDS]
[t.start() for t in threads]; [t.join() for t in threads]
(OUT / "receipt.json").write_text(json.dumps({"base": BASE, "engine": os.environ["SYNTH_CRAFTAX_URL"], "protocol": installed, "rollouts": results}, indent=1, default=str))

print(f"\n{'rollout':22} {'status':10} {'steps':>5} {'reward':>7} {'roll ev':>7} {'ann ev':>6} {'findings':>8} {'retract':>7} {'1st ann':>8} {'last roll':>9} {'ann closed':>10} {'poll==sse'}")
for rid, r in sorted(results.items()):
    ann = r["annotation_events"]; roll = r["rollout_events"]
    findings = [e for e in ann if e["kind"] == "annotation.finding"]
    first_ann = next((e["t"] for e in ann if e["kind"].startswith("annotation.") and e["kind"] != "annotation.protocol.bound"), None)
    closed = next((e for e in ann if e["kind"] == "annotation.closed"), {}).get("payload", {})
    print(f"{rid:22} {r['start']['status']:10} {r['start'].get('usage') and '' or ''}{closed.get('consumed_high_water', 0):>5} {sum(v or 0 for v in (r['start'].get('reward') or {}).values()) if isinstance(r['start'].get('reward'), dict) else r['start'].get('reward'):>7} {len(roll):>7} {len(ann):>6} {len(findings):>8} {sum(1 for e in ann if e['kind']=='annotation.finding.retracted'):>7} {str(first_ann):>8} {roll[-1]['t'] if roll else None:>9} {str(next((e['t'] for e in ann if e['kind']=='capture.closed'), None)):>10} {r['poll_matches_sse']}")
    for e in findings[:12]:
        p = e["payload"]; print(f"   t={e['t']:7.3f} src_seq={p['source_sequence']:>4} step={str(p.get('step')):>3} {p['kind']:12} {p['label']:50} conf={p.get('confidence')} {('supersedes '+p['supersedes']) if p.get('supersedes') else ''}")
    for e in ann:
        if e["kind"] in ("annotation.finding.retracted", "annotation.protocol.error"): print(f"   t={e['t']:7.3f} {e['kind']} {json.dumps(e['payload'])[:160]}")
    # interleaving proof: annotation events that arrived before the rollout's last event
    early = sum(1 for e in ann if roll and e["t"] < roll[-1]["t"] and e["kind"] != "annotation.protocol.bound")
    print(f"   annotation events that arrived before the rollout's last event: {early}/{len(ann)} ; outcome={closed.get('outcome')} errors={closed.get('protocol_errors')}")
server.should_exit = True
