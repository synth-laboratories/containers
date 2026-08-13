#!/usr/bin/env python3
"""Ten Craftax seeds through synth-containers HTTP (C1-08).

Not evals/suites/nonproduct. Not a guessed /events URL.
Default is `craftax_react` + gold HTTP (`SYNTH_CRAFTAX_URL`) + OpenRouter
`gpt-5.6-luna` medium. `--scripted` uses in-process `craftax_engine`.

Serves a real uvicorn port so poll/SSE can observe partial traces while
POST /rollouts is still in flight. TestClient cannot do that.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import httpx
import uvicorn

from synth_containers.platform import create_compat_app

WORLD_REF = "world:craftax_default@symbolic_survival"
SLOT = "stream"
SCHEMA_EVENT = "synth.trace-stream-event.v1"
SCHEMA_STREAM = "synth.rollout.stream.v1"
TELEMETRY = {
    "enabled": True,
    "transport": "sse",
    "detail": "standard",
    "frame": {"enabled": False},
}
POLICY_REF = {"harness": "react", "config": "luna_med"}
LUNA_CONFIG = {
    "provider": "openrouter",
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "api_key_env": "OPENROUTER_API_KEY",
}


def log(message: str, *, file: TextIO = sys.stderr) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}Z] {message}", file=file, flush=True)


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("kind")) for row in events]


def _subscribed(events: list[dict[str, Any]]) -> bool:
    for row in events:
        if row.get("kind") != "stream.subscribed":
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        ready = row.get("ready") if row.get("ready") is not None else payload.get("ready")
        if ready is True:
            return True
    return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(base: str, *, timeout_s: float = 8.0) -> None:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base}/health", timeout=1.0)
            if response.status_code == 200:
                return
            last = f"{response.status_code} {response.text[:200]}"
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(0.05)
    raise RuntimeError(f"façade never became healthy: {last}")


def _drain_poll(
    client: httpx.Client,
    poll_url: str,
    *,
    after: int = 0,
    timeout: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read every evidence page. A single GET after=0 is capped at limit=1000."""
    envelopes: list[dict[str, Any]] = []
    cursor: dict[str, Any] = {}
    seen: set[tuple[Any, Any]] = set()
    while True:
        response = client.get(
            poll_url,
            params={"after": after, "limit": 1000},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"poll failed {response.status_code}: {response.text[:300]}")
        body = response.json()
        cursor = body.get("cursor") if isinstance(body.get("cursor"), dict) else {}
        page = body.get("events") or []
        for row in page:
            identity = (row.get("event_id"), row.get("sequence"), row.get("kind"))
            if identity in seen:
                continue
            seen.add(identity)
            envelopes.append(row)
        next_cursor = cursor.get("next")
        has_more = bool(cursor.get("has_more"))
        if not has_more:
            return envelopes, cursor
        if not isinstance(next_cursor, int) or next_cursor <= after:
            raise RuntimeError(
                f"poll claimed has_more without advancing cursor.next "
                f"(after={after} next={next_cursor!r})"
            )
        after = next_cursor


def _assert_stream_descriptor(stream: dict[str, Any]) -> tuple[str, str]:
    if stream.get("schema") != SCHEMA_STREAM:
        raise RuntimeError(f"prepare omitted {SCHEMA_STREAM}: {stream.get('schema')!r}")
    cursor = stream.get("cursor") if isinstance(stream.get("cursor"), dict) else {}
    if cursor.get("kind") != "sequence":
        raise RuntimeError(f"cursor.kind must be sequence, got {cursor.get('kind')!r}")
    transports = stream.get("transports") if isinstance(stream.get("transports"), dict) else {}
    poll = transports.get("poll") if isinstance(transports.get("poll"), dict) else {}
    sse = transports.get("sse") if isinstance(transports.get("sse"), dict) else {}
    poll_url = stream.get("poll_url") or poll.get("url")
    sse_url = stream.get("sse_url") or sse.get("url")
    if not poll_url:
        raise RuntimeError("prepare omitted declared transports.poll.url; refusing to guess /events")
    if not sse_url:
        raise RuntimeError("sse was requested; prepare omitted transports.sse.url (no silent degrade)")
    if not stream.get("retention"):
        raise RuntimeError("stream descriptor omitted advertised retention")
    reward = stream.get("reward")
    if not (isinstance(reward, dict) and reward.get("url")):
        raise RuntimeError("stream descriptor omitted reward.url")
    return str(poll_url), str(sse_url)


def _ontology_defects(
    *,
    envelopes: list[dict[str, Any]],
    paid: bool,
    saw_live_partial: bool,
    cursor: dict[str, Any],
    secret: str,
) -> list[str]:
    defects: list[str] = []
    blob = json.dumps(envelopes)
    if secret and secret in blob:
        defects.append("api_key_leaked_into_event_log")
    if "Bearer " in blob:
        defects.append("authorization_header_leaked_into_event_log")
    if not saw_live_partial:
        defects.append("no_partial_trace_before_terminal")
    if cursor.get("kind") != "sequence":
        defects.append(f"cursor.kind={cursor.get('kind')!r}")
    if cursor.get("closed") is not True:
        defects.append("log_not_closed")
    semantic = [row for row in envelopes if not row.get("control")]
    control = [row for row in envelopes if row.get("control")]
    if not any(row.get("kind") == "stream.subscribed" for row in control):
        defects.append("missing_stream.subscribed")
    for row in control:
        if row.get("sequence") is not None:
            defects.append("control_record_advanced_sequence")
            break
    if not semantic:
        defects.append("no_semantic_events")
        return defects
    if semantic[0].get("kind") != "trace.opened":
        defects.append(f"first_semantic={semantic[0].get('kind')!r} (want trace.opened)")
    sequences = [row.get("sequence") for row in semantic]
    if sequences != list(range(1, len(sequences) + 1)):
        defects.append(f"sequence_gap:{sequences[:12]}")
    for row in envelopes:
        if row.get("schema") != SCHEMA_EVENT:
            defects.append(f"event_schema={row.get('schema')!r}")
            break
    kinds = _kinds(semantic)
    for required in ("observation", "frame", "action", "reward_signal", "status"):
        if required not in kinds:
            defects.append(f"missing_kind:{required}")
    open_count = kinds.count("span.policy.opened")
    close_count = kinds.count("span.policy.closed")
    if open_count == 0:
        defects.append("missing_span.policy.opened")
    if open_count != close_count:
        defects.append(f"span_open_close_mismatch:{open_count}/{close_count}")
    depth = 0
    for row in semantic:
        kind = row.get("kind")
        if kind == "span.policy.opened":
            depth += 1
        elif kind == "span.policy.closed":
            if depth == 0:
                defects.append("span.policy.closed_before_open")
            else:
                depth -= 1
        elif kind in {"span.policy.data", "span.policy.plan"} and depth == 0:
            defects.append(f"{kind}_outside_span")
    if "status" in kinds and kinds.index("status") < max(
        (i for i, kind in enumerate(kinds) if kind == "span.policy.closed"),
        default=0,
    ):
        pass
    if depth != 0:
        defects.append("span_still_open_at_end")
    for required in ("env.episode.opened", "env.episode.closed", "capture.closed"):
        if required not in kinds:
            defects.append(f"missing_kind:{required}")
    if paid:
        data_rows = [row for row in semantic if row.get("kind") == "span.policy.data"]
        if not data_rows:
            defects.append("paid_missing_span.policy.data")
        else:
            payload = data_rows[0].get("payload") if isinstance(data_rows[0].get("payload"), dict) else {}
            if payload.get("model") != LUNA_CONFIG["model"]:
                defects.append(f"policy_model={payload.get('model')!r}")
            if payload.get("provider") != "openrouter":
                defects.append(f"policy_provider={payload.get('provider')!r}")
        opened = next((row for row in semantic if row.get("kind") == "span.policy.opened"), None)
        call = ((opened or {}).get("payload") or {}).get("call") if opened else None
        if isinstance(call, dict) and call.get("kind") != "openrouter_react":
            defects.append(f"planner_kind={call.get('kind')!r}")
        for required in ("action_applied", "task_resolved"):
            if required not in kinds:
                defects.append(f"missing_nev:{required}")
    return defects


def _poll_during_start(
    client: httpx.Client,
    *,
    poll_url: str,
    stop: threading.Event,
    snapshots: list[dict[str, Any]],
    seed: int,
) -> None:
    after = 0
    while not stop.is_set():
        try:
            response = client.get(poll_url, params={"after": after}, timeout=5.0)
        except httpx.HTTPError:
            time.sleep(0.05)
            continue
        if response.status_code != 200:
            time.sleep(0.05)
            continue
        body = response.json()
        cursor = body.get("cursor") if isinstance(body.get("cursor"), dict) else {}
        events = body.get("events") or []
        high_water = cursor.get("high_water") or 0
        closed = bool(cursor.get("closed"))
        new_kinds = _kinds(events)
        if events or closed:
            snapshots.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "source": "poll",
                    "after": after,
                    "high_water": high_water,
                    "closed": closed,
                    "new_kinds": new_kinds,
                }
            )
            if new_kinds:
                log(
                    f"seed {seed}: live poll after={after} high_water={high_water} "
                    f"closed={closed} +{new_kinds}"
                )
        if isinstance(high_water, int) and high_water > after:
            after = high_water
        if closed:
            return
        time.sleep(0.05)


def _sse_during_start(
    client: httpx.Client,
    *,
    sse_url: str,
    stop: threading.Event,
    snapshots: list[dict[str, Any]],
    seed: int,
) -> None:
    kinds: list[str] = []
    try:
        with client.stream(
            "GET",
            sse_url,
            headers={"Accept": "text/event-stream", "Last-Event-ID": "0"},
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0),
        ) as response:
            if response.status_code != 200:
                log(f"seed {seed}: SSE {response.status_code} (not a silent poll fallback)")
                return
            event_name = ""
            for line in response.iter_lines():
                if stop.is_set():
                    return
                if line is None:
                    continue
                text = line.decode("utf-8") if isinstance(line, bytes) else str(line)
                if text.startswith("event:"):
                    event_name = text.split(":", 1)[1].strip()
                elif text.startswith("data:") and event_name:
                    kinds.append(event_name)
                    snapshots.append(
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "source": "sse",
                            "kind": event_name,
                        }
                    )
                    log(f"seed {seed}: live sse {event_name}")
                    event_name = ""
                elif text.startswith(": heartbeat"):
                    continue
    except httpx.HTTPError as exc:
        if not stop.is_set():
            log(f"seed {seed}: SSE error {exc}")


def run_one(
    client: httpx.Client,
    *,
    seed: int,
    paid: bool,
    events_dir: Path,
    secret: str,
) -> dict[str, Any]:
    rollout_id = f"craftax_seed_{seed}"
    log(f"seed {seed}: POST /rollouts/prepare rollout_id={rollout_id}")
    prepared = client.post(
        "/rollouts/prepare",
        json={"rollout_id": rollout_id, "telemetry": TELEMETRY},
        timeout=30.0,
    )
    if prepared.status_code != 200:
        raise RuntimeError(f"prepare failed {prepared.status_code}: {prepared.text}")
    prepared_body = prepared.json()
    stream = prepared_body["stream"]
    poll_url, sse_url = _assert_stream_descriptor(stream)
    log(
        f"seed {seed}: declared stream.id={stream.get('id') or stream.get('stream.id')} "
        f"poll={poll_url} sse={sse_url} slot={SLOT} cursor.kind=sequence"
    )
    log(f"seed {seed}: GET {poll_url}?after=0 (wait stream.subscribed ready=true)")
    before = client.get(poll_url, params={"after": "0"}, timeout=10.0)
    if before.status_code != 200:
        raise RuntimeError(f"poll failed {before.status_code}: {before.text}")
    before_body = before.json()
    before_events = before_body.get("events") or []
    kinds = _kinds(before_events)
    if not _subscribed(before_events):
        raise RuntimeError(f"C1-08: refusing POST /rollouts; poll kinds={kinds}")
    if any(not row.get("control") for row in before_events):
        semantic = [row.get("kind") for row in before_events if not row.get("control")]
        raise RuntimeError(f"C1-08: semantic event before start: {semantic}")
    subscribed = next(row for row in before_events if row.get("kind") == "stream.subscribed")
    if subscribed.get("sequence") is not None or subscribed.get("control") is not True:
        raise RuntimeError("C1-08: stream.subscribed must be non-advancing control")
    log(f"seed {seed}: stream.subscribed ready=true (heartbeats ignored); starting + live poll/SSE")

    start_body = {
        "rollout_id": rollout_id,
        "telemetry": TELEMETRY,
        "slot": SLOT,
        "world_ref": WORLD_REF,
        "task_instance_id": f"seed:{seed}",
        "policy_ref": POLICY_REF,
    }
    snapshots: list[dict[str, Any]] = []
    stop = threading.Event()
    start_holder: dict[str, Any] = {}
    watcher_base = str(client.base_url)

    def _start() -> None:
        start_holder["response"] = client.post(
            "/rollouts",
            json=start_body,
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
        )

    starter = threading.Thread(target=_start, name=f"start-{seed}", daemon=True)
    poll_client = httpx.Client(base_url=watcher_base)
    sse_client = httpx.Client(base_url=watcher_base)
    poller = threading.Thread(
        target=_poll_during_start,
        kwargs={
            "client": poll_client,
            "poll_url": poll_url,
            "stop": stop,
            "snapshots": snapshots,
            "seed": seed,
        },
        name=f"poll-{seed}",
        daemon=True,
    )
    sse_thread = threading.Thread(
        target=_sse_during_start,
        kwargs={
            "client": sse_client,
            "sse_url": sse_url,
            "stop": stop,
            "snapshots": snapshots,
            "seed": seed,
        },
        name=f"sse-{seed}",
        daemon=True,
    )
    poller.start()
    sse_thread.start()
    starter.start()
    starter.join()
    stop.set()
    poller.join(timeout=2.0)
    sse_thread.join(timeout=15.0)
    poll_client.close()
    sse_client.close()

    started = start_holder.get("response")
    if started is None:
        raise RuntimeError("POST /rollouts thread produced no response")
    if started.status_code != 200:
        raise RuntimeError(f"start failed {started.status_code}: {started.text}")
    started_body = started.json()

    envelopes, cursor = _drain_poll(client, poll_url)
    events_path = events_dir / f"{rollout_id}.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for envelope in envelopes:
            handle.write(json.dumps(envelope) + "\n")
    (events_dir / f"{rollout_id}.partials.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in snapshots),
        encoding="utf-8",
    )

    live_before_close = [
        row
        for row in snapshots
        if row.get("source") == "poll"
        and int(row.get("high_water") or 0) > 0
        and row.get("closed") is False
    ]
    sse_kinds = [row.get("kind") for row in snapshots if row.get("source") == "sse"]
    saw_live_partial = bool(live_before_close) or (
        "trace.opened" in sse_kinds and "status" in sse_kinds and sse_kinds.index("trace.opened") < sse_kinds.index("status")
    )
    # Stronger: a policy span opened on the wire before the start POST returned.
    opened_live = any(
        "span.policy.opened" in (row.get("new_kinds") or []) and row.get("closed") is False
        for row in snapshots
        if row.get("source") == "poll"
    ) or (
        "span.policy.opened" in sse_kinds
        and ("span.policy.closed" not in sse_kinds or sse_kinds.index("span.policy.opened") < sse_kinds.index("span.policy.closed"))
    )
    saw_live_partial = saw_live_partial or opened_live

    defects = _ontology_defects(
        envelopes=envelopes,
        paid=paid,
        saw_live_partial=saw_live_partial,
        cursor=cursor,
        secret=secret,
    )
    if defects:
        raise RuntimeError(f"seed {seed} ontology defects: {defects}")

    scored = client.post("/reward", json={"rollout_id": rollout_id, "mode": "terminal"}, timeout=30.0)
    reward_body = scored.json() if scored.status_code == 200 else {"status": "error", "reward": None}
    reward = reward_body.get("reward")
    status = reward_body.get("status")
    if status == "absent":
        reward = None
    if status in {"absent", "gated", "refused"} and reward == 0:
        raise RuntimeError("missing reward coerced to 0")
    log(
        f"seed {seed}: started rollout_id={started_body.get('rollout_id')} "
        f"status={status} reward={reward!r} events={events_path.name} "
        f"planner={'openrouter_react' if paid else 'scripted_react'} "
        f"live_partials={len(live_before_close)} sse_events={len(sse_kinds)}"
    )
    return {
        "seed": seed,
        "rollout_id": started_body.get("rollout_id"),
        "task_instance_id": started_body.get("task_instance_id"),
        "world_ref": started_body.get("world_ref"),
        "policy_ref": {
            "harness": (started_body.get("policy_ref") or {}).get("harness"),
            "config": (started_body.get("policy_ref") or {}).get("config"),
        },
        "stream.id": (started_body.get("stream") or {}).get("id")
        or (started_body.get("stream") or {}).get("stream.id"),
        "poll_url": poll_url,
        "sse_url": sse_url,
        "slot": SLOT,
        "cursor.kind": "sequence",
        "subscribed_before_start": True,
        "saw_live_partial": saw_live_partial,
        "live_poll_snapshots": len(live_before_close),
        "sse_event_count": len(sse_kinds),
        "status": status,
        "reward": reward,
        "events_path": str(events_path),
        "envelope_count": len(envelopes),
        "high_water": cursor.get("high_water"),
        "closed": cursor.get("closed"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--paid",
        action="store_true",
        help="default: craftax_react + gold HTTP + OpenRouter Luna medium",
    )
    mode.add_argument(
        "--scripted",
        action="store_true",
        help="in-process craftax_engine + ScriptedReAct (no LLM)",
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    paid = not args.scripted
    target = "craftax_react" if paid else "craftax_engine"
    gold_url = os.environ.get("SYNTH_CRAFTAX_URL", "http://127.0.0.1:18100")
    max_steps = os.environ.get("SYNTH_CRAFTAX_MAX_STEPS", "8")
    run_dir = args.run_dir
    events_dir = run_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "seals").mkdir(exist_ok=True)
    secret = os.environ.get("OPENROUTER_API_KEY", "").strip()

    log("setup: synth-containers platform HTTP (NOT evals/suites/nonproduct.craftax)")
    log(f"target: {target}  runtime_family=craftax  world_ref={WORLD_REF}")
    log(f"policy_ref: {json.dumps(POLICY_REF)}  slot={SLOT}  telemetry.transport=sse (auto forbidden)")
    log("contract: POST /rollouts/prepare → GET declared poll until stream.subscribed → POST /rollouts")
    log("streaming: live poll + SSE while start is in flight (not a post-terminal dump)")
    if paid:
        log(f"world: GoldCraftaxWorld via SYNTH_CRAFTAX_URL={gold_url}")
        log(f"policy: OpenRouterReAct model={LUNA_CONFIG['model']} effort={LUNA_CONFIG['effort']}")
        if not secret:
            log("error: Luna medium requires OPENROUTER_API_KEY (key never written to the log)")
            return 2
        os.environ.setdefault("SYNTH_CRAFTAX_URL", gold_url)
    else:
        log("world: in-process CraftaxWorld (fixture). Not gold rust. Not a Luna eval.")
        log("policy: ScriptedReAct (no model). Do not report this as A1 paid.")
    log(f"max_steps: {max_steps}  seeds: 0..{args.seeds - 1}  sequential (occupancy scale_leases=10)")
    log(f"run_dir: {run_dir}")

    app = create_compat_app(target, storage_root=run_dir)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="compat-http", daemon=True)
    server_thread.start()
    try:
        _wait_health(base)
        log(f"façade: {base}  target={target}")
        with httpx.Client(base_url=base) as client:
            info = client.get("/info", timeout=10.0)
            if info.status_code == 200:
                payload = info.json()
                log(
                    f"/info runtime_family={payload.get('runtime_family')} "
                    f"target_id={payload.get('target_id')} live_frames={payload.get('live_frames')} "
                    f"partial_trace={(payload.get('affordances') or {}).get('partial_trace') or payload.get('partial_trace')}"
                )
            pin = client.post(
                "/policy-configs",
                json={"config_id": "luna_med", "harness": "react", "config": LUNA_CONFIG},
                timeout=10.0,
            )
            log(f"POST /policy-configs luna_med → {pin.status_code}")

            rows: list[dict[str, Any]] = []
            for seed in range(args.seeds):
                rows.append(
                    run_one(
                        client,
                        seed=seed,
                        paid=paid,
                        events_dir=events_dir,
                        secret=secret,
                    )
                )

        summary = {
            "schema": "synth.containers.temp-run.v1",
            "setup": "synth-containers-platform",
            "not": ["evals.nonproduct.craftax", "guessed /events", "telemetry.transport=auto"],
            "target": target,
            "paid": paid,
            "slot": SLOT,
            "world_ref": WORLD_REF,
            "policy_ref": POLICY_REF,
            "gold_url": gold_url if paid else None,
            "max_steps": int(max_steps),
            "streaming": "poll+sse during POST /rollouts",
            "note": (
                "paid Luna against gold HTTP; live partial traces on declared stream"
                if paid
                else "headless Containers HTTP; scripted ReAct; not evals gold CLI; not Luna"
            ),
            "leaderboard": rows,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        rewards = [row["reward"] for row in rows]
        present = [value for value in rewards if value is not None]
        log(f"done: {len(rows)} rollouts  rewards_present={present}  missing={rewards.count(None)}")
        log(f"summary: {summary_path}")
        print(json.dumps(summary, indent=2), flush=True)
        return 0
    finally:
        server.should_exit = True


if __name__ == "__main__":
    raise SystemExit(main())
