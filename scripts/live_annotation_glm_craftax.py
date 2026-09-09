"""Run GLM 5.3 flash on Craftax with craftax.live.v1 (code + LLM judge) streaming into a live viewer.

Everything runs on the host against a live Craftax gold engine: the containers
compat façade in-process (target craftax_react), a ReAct policy on
`z-ai/glm-5.3-flash` over OpenRouter, the craftax.live.v1 protocol installed
with the same model as its bounded judge, N rollouts prepared -> subscribed ->
started concurrently, and the standalone viewer polling both streams.

    set -a; source <evals>/.env; set +a            # OPENROUTER_API_KEY, never printed
    SYNTH_CRAFTAX_URL=http://127.0.0.1:18098 \\
    PYTHONPATH=src:<containers-main>/images/craftax-gamebench-rust:<evals> \\
    .venv/bin/python scripts/live_annotation_glm_craftax.py --out /tmp/glm-live --seeds 0,1,2 --max-steps 60

Open the printed viewer URL before the rollouts start (they start ~2 s after
the URL prints). Ctrl-C stops the viewer when you are done judging.
"""
import argparse, json, os, sys, threading, time, socket
from pathlib import Path
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--max-steps", type=int, default=60)
ap.add_argument("--policy-model", default="z-ai/glm-5.3-flash")
ap.add_argument("--judge-model", default="z-ai/glm-5.3-flash")
ap.add_argument("--judge-every-calls", type=int, default=2)
ap.add_argument("--judge-max-calls", type=int, default=8)
ap.add_argument("--effort", default="low")
ap.add_argument("--plan-max", type=int, default=5)
ap.add_argument("--viewer-port", type=int, default=8765)
ap.add_argument("--no-hold", action="store_true", help="exit when rollouts finish instead of keeping the viewer up")
ap.add_argument("--policy", default="glm", choices=["glm", "heuristic"], help="heuristic = no model, free")
args = ap.parse_args()

os.environ.setdefault("SYNTH_CRAFTAX_URL", "http://127.0.0.1:18098")
os.environ["SYNTH_CRAFTAX_MAX_STEPS"] = str(args.max_steps)
if args.policy == "glm" and not os.environ.get("OPENROUTER_API_KEY"):
    sys.exit("OPENROUTER_API_KEY is not set (source the evals .env); or pass --policy heuristic")
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

import uvicorn
from craftax_gold.targets import TARGETS
from synth_containers.platform import create_compat_app
from domains.craftax.annotations import live

target = "craftax_react" if args.policy == "glm" else "craftax_code_policy"
app = create_compat_app(TARGETS[target], storage_root=OUT / "storage")
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
        with urllib.request.urlopen(req, timeout=3600) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")

info = call("GET", "/info")[1]
assert info["capabilities"]["operations"]["annotation.live"] is True, "façade lacks the live annotation lane"

# Policy: ReAct on GLM via OpenRouter. The key is read by the façade from the env at call time.
if args.policy == "glm":
    st, cfg = call("POST", "/policy-configs", {"config_id": "glm53_flash", "harness": "react", "config": {
        "model": args.policy_model, "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
        "effort": args.effort, "max_tokens": 768, "plan_min": 1, "plan_max": args.plan_max}})
    assert st == 200, cfg
    policy_ref = {"harness": "react", "config": "glm53_flash"}
else:
    policy_ref = {"harness": "isolated_policy_process", "config": None}

# Protocol: deterministic signals always; the judge only when a model block is configured.
judge = {"model": args.judge_model, "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
         "max_calls": args.judge_max_calls, "max_output_tokens": 600, "drain_timeout_seconds": 120} if args.policy == "glm" else None
st, installed = call("PUT", "/annotation-protocol", live.install_body(configuration={"judge_every_calls": args.judge_every_calls}, model=judge))
assert st == 200, installed
REV = installed["protocol_revision_id"]
print(f"façade {BASE}  target {target}  protocol {REV}  judge {'on: ' + args.judge_model if judge else 'off'}", flush=True)

seeds = [int(s) for s in args.seeds.split(",")]
rollout_ids = [f"glm_live_seed{s}" for s in seeds]
viewer = Path(__file__).with_name("live_annotation_viewer.py")
viewer_port = args.viewer_port
probe = socket.socket()
try:
    probe.bind(("127.0.0.1", viewer_port))
except OSError:
    probe.close(); probe = socket.socket(); probe.bind(("127.0.0.1", 0)); viewer_port = probe.getsockname()[1]
probe.close()
viewer_proc = __import__("subprocess").Popen([sys.executable, str(viewer), "--facade", BASE, "--port", str(viewer_port), "--rollouts", ",".join(rollout_ids)])
print(f"\n>>> VIEWER: http://127.0.0.1:{viewer_port}/   (open it now; rollouts start in 2 s)\n", flush=True)
time.sleep(2)

results = {}
def run_seed(seed):
    rid = f"glm_live_seed{seed}"
    st, prepared = call("POST", "/rollouts/prepare", {"rollout_id": rid, "task_instance_id": f"seed:{seed}", "telemetry": {"enabled": True, "transport": "sse", "retention": "run"}, "annotation_protocol_revision_id": REV})
    assert st == 200, prepared
    t0 = time.monotonic()
    st, started = call("POST", "/rollouts", {"rollout_id": rid, "slot": "stream", "task_instance_id": f"seed:{seed}", "submission_mode": "sync", "policy_ref": policy_ref, "telemetry": {"enabled": True, "transport": "sse", "retention": "run"}, "annotation_protocol_revision_id": REV})
    results[rid] = {"status": st, "start": started, "seconds": round(time.monotonic() - t0, 1), "channel": prepared["stream"]["annotation"]}
    print(f"{rid}: HTTP {st} status={started.get('status')} reward={started.get('reward')} usage={started.get('usage')} in {results[rid]['seconds']}s", flush=True)
threads = [threading.Thread(target=run_seed, args=(s,)) for s in seeds]
[t.start() for t in threads]; [t.join() for t in threads]

# Wait for every annotation stream to seal, then summarize.
for rid, r in results.items():
    for _ in range(600):
        page = call("GET", r["channel"]["events"] + "?after=0&limit=10000")[1]
        if page.get("cursor", {}).get("closed"): break
        time.sleep(0.2)
    events = [e for e in page.get("events", []) if e.get("sequence") is not None]
    findings = [e["payload"] for e in events if e["kind"] == "annotation.finding"]
    closed = next((e["payload"] for e in events if e["kind"] == "annotation.closed"), {})
    r["annotation"] = {"events": len(events), "findings": len(findings), "outcome": closed.get("outcome"),
                       "model_requested": closed.get("model_requested"), "model_completed": closed.get("model_completed"), "model_failed": closed.get("model_failed"),
                       "labels": [(f["kind"], f["label"], f.get("confidence"), f.get("detail", {}).get("basis")) for f in findings]}
    print(f"\n{rid}: annotations {len(events)} events, {len(findings)} findings, judge {closed.get('model_completed')}/{closed.get('model_requested')} (failed {closed.get('model_failed')}), outcome {closed.get('outcome')}")
    for f in findings: print(f"   {f['kind']:12} {f['label']:55} conf={f.get('confidence')} step={f.get('step')} basis={f.get('detail', {}).get('basis')}{(' · ' + f['detail']['rationale'][:120]) if f.get('detail', {}).get('rationale') else ''}")
    for e in events:
        if e["kind"] in ("annotation.model.failed", "annotation.protocol.error"): print(f"   ! {e['kind']} {json.dumps(e['payload'])[:200]}")
(OUT / "receipt.json").write_text(json.dumps({"facade": BASE, "protocol": installed, "rollouts": results}, indent=1, default=str))
print(f"\nreceipt: {OUT / 'receipt.json'}")
if args.no_hold:
    viewer_proc.terminate(); server.should_exit = True
else:
    print("viewer still serving; Ctrl-C to exit", flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        viewer_proc.terminate(); server.should_exit = True
