"""Standalone live viewer for a façade's rollouts and their annotation streams.

Serves one HTML page that polls each rollout's declared rollout stream and
annotation stream (poll is the durable authority), folds them by rollout, and
renders the underlying step/reward/vitals/achievements beside the provisional
annotation layer: findings with confidence and supersede/retract history,
judge metrics, protocol rebinds, and an activity feed. A control box sends
`note`, `judge_now`, `set` and `stop` controls to a rollout's annotator.

    python scripts/live_annotation_viewer.py --facade http://127.0.0.1:PORT --port 8765 [--rollouts id1,id2]

Without --rollouts it discovers rollouts from GET /rollouts on the façade if
that route exists, else from the ids you add in the page. No build step, no
Workshop: this is the fallback surface so a human can judge the lane before
the Workshop pane (live.annotated_rollouts.v1) is driven end to end.
"""
import argparse, json, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Live annotated rollouts</title>
<style>
body{font:13px/1.4 -apple-system,system-ui,sans-serif;margin:0;background:#f7f5f2;color:#1f1f1f}
header{padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.metric{display:grid;font-size:11px;color:#666}.metric b{font-size:15px;color:#111}
main{padding:12px 16px;display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(380px,1fr))}
article{background:#fff;border:1px solid #ddd;border-radius:10px;padding:12px;display:grid;gap:8px}
.bar{height:7px;background:#e6e2dc;border-radius:8px;overflow:hidden}.bar>div{height:100%;background:#ff5c00;transition:width .2s}
.chip{display:inline-flex;gap:4px;align-items:center;padding:2px 7px;border-radius:999px;font-size:10px;border:1px solid;margin:2px;font-family:ui-monospace,monospace}
.chip.superseded{opacity:.55}.chip.retracted{opacity:.55;text-decoration:line-through}
.markers{position:relative;height:14px;background:#e6e2dc;border-radius:7px}.markers span{position:absolute;top:2px;width:10px;height:10px;margin-left:-5px;border-radius:999px;border:2px solid}
.markers span.failure_mode{border-radius:2px;transform:rotate(45deg)}
.mono{font-family:ui-monospace,monospace;font-size:10px;color:#777}
.feed{max-height:180px;overflow:auto;border-top:1px solid #eee;padding-top:6px}.feed div{display:grid;grid-template-columns:52px 1fr;gap:8px;font-size:11px;padding:2px 0}
.ann{color:#2f6fdd}.ctl{display:flex;gap:6px;flex-wrap:wrap}.ctl input,.ctl select{font-size:11px}.ctl button{font-size:11px;cursor:pointer}
.vitals{display:flex;gap:10px}.vital{display:grid;gap:2px;font-size:9px;color:#888}.vital i{display:block;width:42px;height:4px;background:#e6e2dc;border-radius:9px}.vital i b{display:block;height:100%;border-radius:9px}
.tally{font-size:11px;display:grid;gap:2px}
</style></head><body>
<header>
 <div class="metric">Rollouts<b id="m-roll">–</b></div><div class="metric">Achievements<b id="m-ach">0</b></div><div class="metric">Milestones<b id="m-ms">0</b></div>
 <div class="metric">Failure modes<b id="m-fm">0</b></div><div class="metric">Retracted<b id="m-ret">0</b></div><div class="metric">Judge calls<b id="m-judge">0</b></div>
 <div class="metric">Stream<b id="m-state">connecting</b></div>
 <div style="margin-left:auto" class="ctl"><input id="add" placeholder="rollout id" size="24"><button onclick="addRollout()">watch</button><label><input type="checkbox" id="hist"> history</label></div>
</header>
<main id="lanes"></main>
<script>
const COLORS={achievement:"#39a46b",milestone:"#2f6fdd",failure_mode:"#d84b3f",intent:"#8a5bd6",note:"#8c8c8c"};
const state={rollouts:{}};
function lane(id){return state.rollouts[id]??={id,after:0,annAfter:0,events:[],ann:[],closed:false,annClosed:false};}
async function poll(){
  const ids=Object.keys(state.rollouts);
  for(const id of ids){const l=lane(id);
    try{ if(!l.closed){const p=await (await fetch(`/proxy/rollouts/${id}/events?after=${l.after}&limit=2000`)).json(); for(const e of p.events){ if(e.sequence!=null){l.events.push(e);l.after=Math.max(l.after,e.sequence);} } l.closed=!!p.cursor.closed; }
        if(!l.annClosed){const p=await (await fetch(`/proxy/rollouts/${id}/annotations/events?after=${l.annAfter}&limit=2000`)); if(p.status===200){const j=await p.json(); for(const e of j.events){ if(e.sequence!=null){l.ann.push(e);l.annAfter=Math.max(l.annAfter,e.sequence);} } l.annClosed=!!j.cursor.closed; l.summary=j.summary;} }
        l.error=null;}catch(err){l.error=String(err);} }
  render(); setTimeout(poll,500);
}
function project(l){
  const o={done:0,total:null,reward:0,ach:[],inv:{},calls:0,status:"starting",findings:[],markers:[],metrics:{},model:{req:0,done:0,fail:0},rebinds:0,controls:[],proto:null,outcome:null,last:"",lastAnn:""};
  for(const e of l.events){const p=e.payload||{};
    if(e.kind==="env.episode.opened"){o.total=p.max_steps??o.total;o.status="running"}
    if(e.kind==="observation"){o.done=p.step??o.done;const r=p.readout||{};if(Array.isArray(r.achievements))o.ach=r.achievements;o.inv=r.inventory||o.inv;o.status="running"}
    if(e.kind==="reward_signal"&&typeof p.value==="number")o.reward+=p.value;
    if(e.kind==="span.policy.plan")o.calls++;
    if((e.kind==="env.episode.closed"||e.kind==="status")&&p.status)o.status=p.status==="completed"||p.status==="truncated"?"finished":p.status;
    o.last=e.kind+(p.action?" · "+p.action:"");}
  for(const e of l.ann){const p=e.payload||{};
    if(e.kind==="annotation.protocol.bound")o.proto=p.protocol_revision_id;
    if(e.kind==="annotation.protocol.rebound"){o.proto=p.protocol_revision_id;o.rebinds++}
    if(e.kind==="annotation.finding"){ if(p.supersedes){const f=o.findings.find(x=>x.id===p.supersedes);if(f&&f.status==="provisional")f.status="superseded";}
      o.findings.push({id:p.finding_id,kind:p.kind,label:p.label,status:"provisional",step:p.step,conf:p.confidence,basis:p.detail?.basis,rationale:p.detail?.rationale}); }
    if(e.kind==="annotation.finding.retracted"){const f=o.findings.find(x=>x.id===p.finding_id);if(f){f.status="retracted";f.reason=p.reason}}
    if(e.kind==="annotation.metric")o.metrics[p.name]=p.value;
    if(e.kind==="annotation.model.requested")o.model.req++; if(e.kind==="annotation.model.completed")o.model.done++; if(e.kind==="annotation.model.failed")o.model.fail++;
    if(e.kind.startsWith("annotation.control."))o.controls.push({ok:e.kind.endsWith("received"),op:p.op,reason:p.reason,id:p.control_id});
    if(e.kind==="annotation.closed")o.outcome=p.outcome;
    o.lastAnn=e.kind+(p.label?" · "+p.label:p.name?" · "+p.name+"="+p.value:"");}
  return o;}
function vital(n,v){const pct=v==null?0:Math.min(100,v/9*100);const c=pct<34?"#d84b3f":pct<67?"#e5a226":"#39a46b";return `<div class="vital">${n}<i><b style="width:${pct}%;background:${c}"></b></i></div>`}
function render(){
  const hist=document.getElementById("hist").checked;const lanes=Object.values(state.rollouts).map(l=>({l,o:project(l)}));
  let ach=new Set(),ms=0,fm=0,ret=0,judge=0;
  const html=lanes.map(({l,o})=>{o.ach.forEach(a=>ach.add(a));const act=o.findings.filter(f=>f.status==="provisional");ms+=act.filter(f=>f.kind==="milestone").length;fm+=act.filter(f=>f.kind==="failure_mode").length;ret+=o.findings.filter(f=>f.status==="retracted").length;judge+=o.model.req;
    const span=Math.max(o.total||0,o.done,1);
    const chips=o.findings.filter(f=>hist||f.status==="provisional").map(f=>`<span class="chip ${f.status}" style="border-color:${COLORS[f.kind]||"#888"};color:${COLORS[f.kind]||"#888"}" title="${f.kind}: ${f.label} · step ${f.step??"?"} · ${f.status}${f.reason?" · "+f.reason:""}${f.rationale?" · "+f.rationale:""}">${f.label}${f.conf!=null&&f.kind!=="achievement"?" "+Math.round(f.conf*100)+"%":""}${f.basis==="model"?" judge":""}</span>`).join("");
    const markers=o.findings.map(f=>`<span class="${f.kind}" style="left:${Math.min(99,((f.step??o.done)/span)*100)}%;border-color:${COLORS[f.kind]||"#888"};background:${f.status==="provisional"?(COLORS[f.kind]||"#888"):"transparent"}" title="${f.kind}: ${f.label} @ ${f.step}"></span>`).join("");
    const feed=[...l.events.map(e=>({t:e.ts,a:false,k:e.kind,d:e.payload?.action||e.payload?.status||""})),...l.ann.map(e=>({t:e.ts,a:true,k:e.kind,d:e.payload?.label||e.payload?.name||e.payload?.reason||e.payload?.outcome||""}))].sort((x,y)=>(x.t||"").localeCompare(y.t||"")).slice(-40).reverse().map(r=>`<div><span class="mono">${(r.t||"").slice(11,19)}</span><span class="${r.a?"ann":""}">${r.a?"◌ ":"· "}${r.k}${r.d?" · "+r.d:""}</span></div>`).join("");
    const judgeP=o.metrics.judge_progress;const ctl=o.controls.length?`${o.controls.filter(c=>c.ok).length} controls${o.controls.some(c=>!c.ok)?` (${o.controls.filter(c=>!c.ok).length} refused)`:""}${o.rebinds?` · ${o.rebinds} rebinds`:""}`:"";
    return `<article><div style="display:flex;justify-content:space-between"><b>${l.id}</b><span class="mono" style="color:#ff5c00">${o.status}${o.outcome?" · annotations "+o.outcome:o.proto?" · annotating":""}${l.error?" · "+l.error:""}</span></div>
      <div class="bar"><div style="width:${o.total?Math.min(100,o.done/o.total*100):0}%"></div></div>
      <div style="display:flex;gap:10px;font-size:11px;color:#555;flex-wrap:wrap"><span class="mono">${o.done}${o.total?" / "+o.total:""} steps</span><span>reward <b>${(o.metrics.cumulative_reward??o.reward).toFixed(2)}</b></span><span>${o.ach.length} achievements</span><span>${o.calls} calls</span>${judgeP!=null?`<span>judge ${judgeP>0?"advancing":judgeP<0?"regressing":"stalled"}</span>`:""}</div>
      <div class="vitals">${vital("HLTH",o.inv.health)}${vital("FOOD",o.inv.food)}${vital("DRNK",o.inv.drink)}${vital("NRGY",o.inv.energy)}<span class="mono" style="margin-left:auto">› ${o.last}</span></div>
      <div class="markers"><div style="position:absolute;inset:0;width:${Math.min(100,o.done/span*100)}%;background:#ff5c00;opacity:.25;border-radius:7px"></div>${markers}</div>
      <div class="mono">${act.filter(f=>f.kind==="achievement").length} achievement · ${act.filter(f=>f.kind==="milestone").length} milestone · ${act.filter(f=>f.kind==="failure_mode").length} failure mode · ${act.filter(f=>f.kind==="intent").length} intent · ${o.findings.filter(f=>f.status==="retracted").length} retracted${o.model.req?` · judge ${o.model.done}/${o.model.req}${o.model.fail?" ("+o.model.fail+" failed)":""}`:""}${ctl?" · "+ctl:""}</div>
      <div>${chips||'<span class="mono">'+(o.proto?"no findings yet":"no protocol bound")+"</span>"}</div>
      <div class="mono">◌ ${o.lastAnn}${o.proto?" · "+o.proto:""}</div>
      <div class="ctl"><select id="op-${l.id}"><option value="note">note</option><option value="judge_now">judge_now</option><option value="set">set</option><option value="stop">stop</option></select><input id="txt-${l.id}" placeholder="note text / name=value" size="26"><button onclick="control('${l.id}')">send</button><span class="mono" id="ack-${l.id}"></span></div>
      <div class="feed">${feed}</div></article>`;}).join("");
  document.getElementById("lanes").innerHTML=html||'<p class="mono">add a rollout id above, or start the runner with --viewer</p>';
  document.getElementById("m-roll").textContent=`${lanes.filter(x=>x.o.status==="finished").length}/${lanes.length||"–"} done`;
  document.getElementById("m-ach").textContent=ach.size;document.getElementById("m-ms").textContent=ms;document.getElementById("m-fm").textContent=fm;document.getElementById("m-ret").textContent=ret;document.getElementById("m-judge").textContent=judge;
  document.getElementById("m-state").textContent=lanes.length?(lanes.every(x=>x.l.annClosed)?"complete":"receiving"):"idle";
}
async function control(id){const op=document.getElementById("op-"+id).value,txt=document.getElementById("txt-"+id).value.trim();let body={op};
  if(op==="note")body={op:"message",message:{type:"note",text:txt||"operator note",author:"viewer"}};
  if(op==="judge_now")body={op:"message",message:{type:"judge_now"}};
  if(op==="set"){const [n,v]=txt.split("=");body={op:"message",message:{type:"set",name:n,value:Number(v)}}}
  if(op==="stop")body={op:"stop",reason:txt||"viewer"};
  const r=await fetch(`/proxy/rollouts/${id}/annotations/control`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});const j=await r.json();
  document.getElementById("ack-"+id).textContent=(j.accepted?"accepted ":"refused ")+(j.control_id||j.reason||"");}
function addRollout(){const v=document.getElementById("add").value.trim();if(v){lane(v);document.getElementById("add").value="";render();}}
(async()=>{try{const r=await fetch("/rollouts");const ids=await r.json();ids.forEach(lane);}catch(e){} poll();})();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facade", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--rollouts", default="")
    args = ap.parse_args()
    facade = args.facade.rstrip("/")
    rollouts = [r for r in args.rollouts.split(",") if r]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, status, body, ctype="application/json"):
            self.send_response(status); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/rollouts":
                return self._send(200, json.dumps(rollouts).encode())
            if self.path.startswith("/proxy/"):
                return self._proxy("GET")
            self._send(404, b"{}")

        def do_POST(self):
            if self.path.startswith("/proxy/"):
                return self._proxy("POST")
            self._send(404, b"{}")

        def _proxy(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(length) if length else None
            req = urllib.request.Request(facade + self.path[len("/proxy"):], data=data, method=method, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    self._send(r.status, r.read())
            except urllib.error.HTTPError as e:
                self._send(e.code, e.read() or b"{}")
            except Exception as e:  # noqa: BLE001
                self._send(502, json.dumps({"error": str(e)}).encode())

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"viewer: http://127.0.0.1:{args.port}/  (façade {facade}; rollouts {rollouts or 'add in page'})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
