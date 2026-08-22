"""Workshop container surface for the clean-room ALFWorld text environment.

The service deliberately owns episode execution and reward calculation.  SFT
clients consume stable teacher rows; CISPO clients submit sampled actions or an
OpenAI-compatible policy proxy and receive an auditable trajectory/reward.
No ALFWorld or TextWorld package is imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .artifact_game import ArtifactAlfWorldGame

app = FastAPI(title="ALFWorld clean-room container", version="1.0")
DATA_ROOT = Path(os.environ.get("ALFWORLD_DATA_ROOT", "/data/alfworld"))
# Docker sets /runs explicitly.  A relative default keeps `run_container.sh`
# usable on a host without root filesystem write access.
RUN_ROOT = Path(os.environ.get("ALFWORLD_RUN_ROOT", ".alfworld-runs"))
MAX_TURNS = int(os.environ.get("ALFWORLD_MAX_TURNS", "50"))
SYSTEM_PROMPT = """You are playing a text-only ALFWorld episode. Reply with exactly one command from VALID ACTIONS, and no explanation."""
TRAINING_REQUEST_VERSION = "training.rollout.request.v1"
TRAINING_ACTION_VERSION = "training.rollout.action.v1"
TRAINING_SUMMARY_VERSION = "training.rollout.summary.v1"
INVALID_EMPTY_ACTION = "<invalid:empty-model-output>"


def _files() -> list[Path]:
    if not DATA_ROOT.exists():
        return []
    return sorted(DATA_ROOT.rglob("game.tw-pddl"))


def _dataset_digest() -> str:
    """Bind the advertised workload to exact artifact paths and bytes."""
    digest = hashlib.sha256()
    for source in _files():
        relative = source.relative_to(DATA_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = source.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def _split(path: Path) -> str:
    # Stable, artifact-identity split; no train/test leakage by path ordering.
    return "test" if int(hashlib.sha256(path.read_bytes()).hexdigest()[:8], 16) % 10 == 0 else "train"


def _id(path: Path) -> str:
    return hashlib.sha256(path.relative_to(DATA_ROOT).as_posix().encode()).hexdigest()[:16]


def _rows(split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for source in _files():
        row_split = _split(source)
        if split and row_split != split:
            continue
        artifact = json.loads(source.read_text())
        rel = source.relative_to(DATA_ROOT).as_posix()
        rows.append({
            "id": f"alfworld:{row_split}:{_id(source)}", "example_id": f"alfworld:{row_split}:{_id(source)}",
            "task_id": "alfworld.text.v1", "split": row_split, "gamefile": rel,
            "family": source.parent.parent.name.split("-", 1)[0],
            "walkthrough": artifact.get("walkthrough", []),
        })
    return rows


def _find(task_id: str) -> dict[str, Any]:
    for row in _rows():
        if row["id"] == task_id or row["example_id"] == task_id:
            return row
    raise HTTPException(404, f"unknown task id: {task_id}")


def _game(row: dict[str, Any]) -> ArtifactAlfWorldGame:
    return ArtifactAlfWorldGame(DATA_ROOT / row["gamefile"])


def _prompt(observation: str, actions: list[str], history: list[str]) -> str:
    transcript = "\n".join(f"Action: {action}" for action in history[-8:]) or "(none)"
    return f"{SYSTEM_PROMPT}\n\nOBSERVATION:\n{observation}\n\nHISTORY:\n{transcript}\n\nVALID ACTIONS:\n" + "\n".join(actions) + "\n\nAction:"


def _sft_examples(row: dict[str, Any]) -> list[dict[str, Any]]:
    game = _game(row); state = game.reset(); history: list[str] = []; examples = []
    for step, action in enumerate(row["walkthrough"]):
        examples.append({
            "id": f"{row['id']}:step:{step}", "episode_id": row["id"], "step": step,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _prompt(state.observation, game.available_actions(), history)}, {"role": "assistant", "content": action}],
            "completion": action,
        })
        state = game.update(action); history.append(action)
        if state.terminal: break
    return examples


def _record(rollout_id: str, value: dict[str, Any]) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    target = RUN_ROOT / f"{rollout_id}.json"; temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def _load(rollout_id: str) -> dict[str, Any]:
    try: return json.loads((RUN_ROOT / f"{rollout_id}.json").read_text())
    except FileNotFoundError: raise HTTPException(404, f"unknown rollout id: {rollout_id}")


def _parse_action(text: str, allowed: list[str]) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return INVALID_EMPTY_ACTION
    line = re.sub(r"^\s*action\s*:\s*", "", lines[0], flags=re.I).strip().lower()
    if not line:
        return INVALID_EMPTY_ACTION
    return next((action for action in allowed if action.lower() == line), line)


async def _policy_action(policy: dict[str, Any], prompt: str, allowed: list[str]) -> str:
    endpoint = str(policy.get("inference_url") or policy.get("base_url") or "").rstrip("/")
    model = str(policy.get("model") or "")
    if not endpoint or not model:
        raise HTTPException(422, "policy requires inference_url/base_url and model")
    url = endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"
    headers = {"content-type": "application/json"}
    if os.environ.get("ALFWORLD_POLICY_BEARER_TOKEN"):
        headers["authorization"] = f"Bearer {os.environ['ALFWORLD_POLICY_BEARER_TOKEN']}"
    async with httpx.AsyncClient(timeout=float(policy.get("timeout_seconds", 60))) as client:
        response = await client.post(url, headers=headers, json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": float(policy.get("temperature", 0.7)), "max_tokens": int(policy.get("max_tokens", 32))})
    if response.is_error: raise HTTPException(502, f"policy request failed: {response.status_code} {response.text[:300]}")
    try: text = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc: raise HTTPException(502, "policy response has no chat completion") from exc
    return _parse_action(str(text), allowed)


@app.get("/health")
async def health() -> dict[str, str]: return {"status": "ok"}


@app.get("/training/capabilities")
@app.get("/rollout/training/capabilities")
async def training_capabilities() -> dict[str, Any]:
    """Advertise the same rollout protocol consumed by local MLX CISPO."""
    contract = {
        "task_id": "alfworld.text.v1",
        "dataset_digest": _dataset_digest(),
        "protocol_versions": [TRAINING_REQUEST_VERSION, TRAINING_ACTION_VERSION, TRAINING_SUMMARY_VERSION],
        "operations": ["rollout", "reward", "heartbeat"],
        "max_concurrency": 1,
        "supports_idempotency": True,
        "supports_sampler_https": True,
        "connection_modes": ["close", "keep_alive"],
    }
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": "training.rollout.capabilities.v1",
        "container_id": "alfworld_text_cleanroom",
        "container_digest": f"sha256:{digest}",
        **contract,
        "capability_hash": f"sha256:{digest}",
    }


@app.post("/training/rollouts")
@app.post("/rollout/training/rollouts")
async def training_rollout(request: Request) -> dict[str, Any]:
    """Run ALFWorld using the MLX sampler and return token-aligned actions."""
    payload = await request.json()
    if payload.get("schema_version") != TRAINING_REQUEST_VERSION:
        raise HTTPException(422, "unsupported training rollout schema")
    task = payload.get("task") or {}
    if task.get("task_id") != "alfworld.text.v1":
        raise HTTPException(422, f"task mismatch: {task.get('task_id')!r}")
    sampler = payload.get("sampler") or {}
    sampler_url = str(sampler.get("url") or "")
    if not sampler_url:
        raise HTTPException(422, "sampler.url is required")
    split = "test" if str(task.get("world_ref") or "").endswith("@test") else "train"
    candidates = _rows(split)
    if not candidates:
        raise HTTPException(422, f"no ALFWorld rows for split {split}")
    seed_text = str(task.get("task_instance_id") or payload.get("rollout_id") or "0")
    seed_match = re.search(r"(\d+)$", seed_text)
    row = candidates[(int(seed_match.group(1)) if seed_match else int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)) % len(candidates)]
    game = _game(row)
    state = game.reset()
    initial_progress = game.runtime.goal_progress()
    history: list[str] = []
    actions: list[dict[str, Any]] = []
    invalid_completion = False
    headers = {"content-type": "application/json"}
    if sampler.get("bearer_token"):
        headers["authorization"] = f"Bearer {sampler['bearer_token']}"
    max_turns = min(int(task.get("max_turns") or MAX_TURNS), MAX_TURNS)
    max_tokens = min(int(task.get("max_tokens") or 32), 32)
    policy_version = str(payload.get("policy_version") or "")
    sampler_snapshot = policy_version if policy_version.startswith("snap_") else None
    async with httpx.AsyncClient(timeout=120.0) as client:
        for _turn in range(max_turns):
            allowed = game.available_actions()
            if not allowed or state.terminal:
                break
            sample_response = await client.post(
                sampler_url,
                headers=headers,
                json={
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _prompt(state.observation, allowed, history)},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": float(task.get("temperature") or 0.7),
                    "policy_snapshot_id": sampler_snapshot,
                },
            )
            if sample_response.is_error:
                raise HTTPException(502, f"sampler failed: {sample_response.status_code} {sample_response.text[:300]}")
            sampled = sample_response.json()
            action = _parse_action(str(sampled.get("text") or ""), allowed)
            history.append(action)
            actions.append({
                "schema_version": TRAINING_ACTION_VERSION,
                "prompt_token_ids": sampled.get("prompt_token_ids") or [],
                "token_ids": sampled.get("token_ids") or [],
                "log_probs": sampled.get("log_probs") or [],
            })
            if action == INVALID_EMPTY_ACTION:
                # Empty model output is an explicit failed generation.  Do not
                # invent a valid environment action or award exploration shaping.
                invalid_completion = True
                break
            state = game.update(action)
            if state.terminal:
                break
    rollout_id = str(payload.get("rollout_id") or f"alfworld_{uuid.uuid4().hex}")
    final_progress = game.runtime.goal_progress()
    progress_delta = max(0.0, final_progress - initial_progress)
    exploration_coverage = (
        0.0 if invalid_completion else len(set(history)) / max(1, max_turns)
    )
    shaped_reward = (
        0.0
        if invalid_completion
        else 1.0 if state.won else 0.25 * progress_delta + 0.01 * exploration_coverage
    )
    result = {
        "schema_version": TRAINING_SUMMARY_VERSION,
        "rollout_id": rollout_id,
        "reward": {"reward": shaped_reward, "won": state.won},
        "actions": actions,
        "trajectory": history,
        "reward_info": {
            "outcome_reward": float(state.won),
            "goal_progress_before": initial_progress,
            "goal_progress_after": final_progress,
            "goal_progress_delta": progress_delta,
            "shaping_weight": 0.25,
            "exploration_coverage": exploration_coverage,
            "exploration_weight": 0.01,
            "invalid_completion": invalid_completion,
        },
        "container_digest": (await training_capabilities())["container_digest"],
        "capability_hash": (await training_capabilities())["capability_hash"],
    }
    _record(rollout_id, result)
    return result


@app.get("/metadata")
@app.get("/info")
async def metadata() -> dict[str, Any]:
    return {"runtime": {"runtime_id": "alfworld_text_cleanroom.v1", "name": "ALFWorld text clean-room", "description": "Artifact-backed, dependency-free ALFWorld runtime."}, "capabilities": {"contract_version": "container_contract.v1", "rollout_modes": ["blocking"], "policy_ready": True, "sft_export": True, "cispo": {"version": "cispo.text-trajectory.v1", "group_rollouts": True, "token_trace_support": "policy-provider-dependent"}}, "metadata": {"optimizer_contracts": {"sft": {"version": "synth_optimizers.sft.v1", "dataset_route": "/sft/dataset", "train_jsonl_route": "/sft/train.jsonl", "eval_jsonl_route": "/sft/eval.jsonl"}, "cispo": {"version": "synth_optimizers.cispo.v1", "rollout_route": "/rollout", "reward_route": "/reward"}}, "task_catalog_route": "/task_catalog", "workshop_manifest_route": "/workshop/manifest"}}


@app.get("/task_catalog")
async def task_catalog() -> dict[str, Any]:
    rows = _rows(); return {"catalog_id": "alfworld.public-artifacts.v1", "tasks": [{"task_id": "alfworld.text.v1", "task_name": "ALFWorld text-only", "task_family": "alfworld", "description": "Six-family interactive household planning benchmark."}], "instances": rows, "metadata": {"instance_count": len(rows), "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "test")}}}


@app.get("/task_info")
async def task_info() -> dict[str, Any]:
    return {"task": {"task_id": "alfworld.text.v1", "name": "ALFWorld text-only clean-room"}, "output_space": {"kind": "single_text_command", "contract": "Return exactly one command from the episode's VALID ACTIONS."}, "dataset": {"dataset_id": "alfworld.public-artifacts.v1", "visible_splits": ["train", "test"]}, "reward": {"outcome_reward": "1.0 iff the artifact PDDL goal is satisfied; otherwise 0.0", "max_turns": MAX_TURNS}}


@app.get("/dataset")
async def dataset() -> dict[str, Any]:
    rows = _rows(); return {"dataset_id": "alfworld.public-artifacts.v1", "splits": {s: sum(r["split"] == s for r in rows) for s in ("train", "test")}, "format": "episode rows; use /sft/dataset for teacher action rows"}


@app.post("/dataset/rows")
async def dataset_rows(request: Request) -> dict[str, Any]:
    payload = await request.json(); split = str(payload.get("split") or "train"); rows = _rows(split)
    seeds = payload.get("seeds") or list(range(len(rows)))
    return {"rows": [rows[int(seed) % len(rows)] for seed in seeds] if rows else []}


@app.get("/sft/dataset")
async def sft_dataset(split: str = "train", limit_episodes: int = 0) -> dict[str, Any]:
    rows = _rows(split); rows = rows[:limit_episodes] if limit_episodes else rows
    return {"format": "openai.chat.completions.jsonl.v1", "split": split, "examples": [example for row in rows for example in _sft_examples(row)]}


def _jsonl(split: str, limit_episodes: int = 0):
    """Yield the exact rows returned by /sft/dataset as portable JSONL.

    Workshop's local MLX recipe consumes JSONL paths.  A bind-mounted copy of
    either endpoint is therefore byte-for-byte the teacher dataset advertised
    by the container, rather than a separately generated training corpus.
    """
    rows = _rows(split)
    if limit_episodes:
        rows = rows[:limit_episodes]
    for row in rows:
        for example in _sft_examples(row):
            yield json.dumps(example, separators=(",", ":"), ensure_ascii=False) + "\n"


@app.get("/sft/train.jsonl")
async def sft_train_jsonl(limit_episodes: int = 0) -> StreamingResponse:
    return StreamingResponse(_jsonl("train", limit_episodes), media_type="application/jsonl")


@app.get("/sft/eval.jsonl")
async def sft_eval_jsonl(limit_episodes: int = 0) -> StreamingResponse:
    return StreamingResponse(_jsonl("test", limit_episodes), media_type="application/jsonl")


@app.get("/workshop/manifest")
async def workshop_manifest() -> dict[str, Any]:
    """Stable hand-off contract for a first-class Workshop ALFWorld recipe."""
    return {
        "schema_version": "workshop.alfworld.training.v1",
        "task": "alfworld.text.v1",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "sft": {
            "format": "openai.chat.completions.jsonl.v1",
            "train": {"route": "/sft/train.jsonl", "split": "train"},
            "evaluation": {"route": "/sft/eval.jsonl", "split": "test"},
        },
        "cispo": {
            "contract": "cispo.text-trajectory.v1",
            "harness": "text_trajectory",
            "plan_ref": "alfworld_text_eval.v1",
            "rollout_route": "/rollout",
            "reward_route": "/reward",
            "train_world_ref": "world:alfworld@train",
            "heldout_world_ref": "world:alfworld@test",
        },
    }


@app.post("/rollout")
@app.post("/rollouts")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json(); row = _find(str(payload.get("task_id") or payload.get("episode_id") or ""))
    actions = payload.get("actions"); policy = payload.get("policy")
    if actions is not None and not isinstance(actions, list): raise HTTPException(422, "actions must be a list of commands")
    if actions is None and not isinstance(policy, dict): raise HTTPException(422, "provide actions or an OpenAI-compatible policy")
    game = _game(row); state = game.reset(); history: list[str] = []; trace: list[dict[str, Any]] = []
    for turn in range(int(payload.get("max_turns") or MAX_TURNS)):
        allowed = game.available_actions()
        if not allowed or state.terminal: break
        action = str(actions[turn]) if actions is not None and turn < len(actions) else (await _policy_action(policy, _prompt(state.observation, allowed, history), allowed) if actions is None else "")
        if not action: break
        state = game.update(action); history.append(action); trace.append({"turn": turn, "action": action, "observation": state.observation, "reward": state.reward, "terminal": state.terminal})
        if state.terminal: break
    rollout_id = str(payload.get("rollout_id") or f"alfworld_{uuid.uuid4().hex}")
    result = {"rollout_id": rollout_id, "task_id": row["id"], "status": "completed", "reward": float(state.won), "reward_info": {"outcome_reward": float(state.won), "won": state.won}, "summary": {"outcome_reward": float(state.won), "turns": len(trace), "family": row["family"]}, "trace": trace, "metadata": {"environment": "alfworld_text_cleanroom.v1", "gamefile": row["gamefile"]}}
    _record(rollout_id, result); return result


@app.get("/rollouts/{rollout_id}")
async def get_rollout(rollout_id: str) -> dict[str, Any]: return _load(rollout_id)


@app.get("/reward")
async def reward(rollout_id: str) -> dict[str, Any]:
    result = _load(rollout_id); return {"rollout_id": rollout_id, "reward": result["reward"], "reward_info": result["reward_info"]}
