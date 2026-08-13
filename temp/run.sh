#!/usr/bin/env bash
# Drive Craftax seeds 0–9 through synth-containers HTTP (C1-08).
# Default: OpenRouter Luna medium ReAct against gold HTTP.
# Not evals/suites/nonproduct. Not a guessed /events URL.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTAINERS="$(cd "$ROOT/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$ROOT/_runs/luna_med_10x}"
LOG="$RUN_DIR/run.log"
PAID=1
SEEDS="${SEEDS:-10}"

usage() {
  cat <<'EOF'
temp/run.sh — synth-containers platform 10-seed Craftax

  bash temp/run.sh              craftax_react + gold HTTP + OpenRouter Luna medium
  bash temp/run.sh --paid       same as default
  bash temp/run.sh --scripted   in-process craftax_engine (ScriptedReAct; no LLM)

Env:
  SYNTH_CRAFTAX_URL          gold rust base (default http://127.0.0.1:18100)
  SYNTH_CRAFTAX_MAX_STEPS    episode cap (default 120 Luna / 8 scripted)
  OPENROUTER_API_KEY         required unless --scripted; never written to the event log
  SYNTH_AI_ENV_PATH          optional dotenv loaded for Luna
  RUN_DIR SEEDS
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paid) PAID=1; shift ;;
    --scripted) PAID=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *.toml)
      echo "error: evals TOML is no longer a driver. Use bash temp/run.sh (Luna medium) or --scripted." >&2
      echo "       (old luna_med_10x.toml was suites.nonproduct.craftax; that path is retired here.)" >&2
      exit 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RUN_DIR/events" "$RUN_DIR/seals"
export PYTHONUNBUFFERED=1
if [[ "$PAID" -eq 1 ]]; then
  export SYNTH_CRAFTAX_URL="${SYNTH_CRAFTAX_URL:-http://127.0.0.1:18100}"
  export SYNTH_CRAFTAX_MAX_STEPS="${SYNTH_CRAFTAX_MAX_STEPS:-120}"
  ENV_FILE="${SYNTH_AI_ENV_PATH:-$ROOT/../../synth-ai/.env}"
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
else
  export SYNTH_CRAFTAX_MAX_STEPS="${SYNTH_CRAFTAX_MAX_STEPS:-8}"
fi

{
  echo "============================================================"
  echo "synth-containers platform  (NOT evals/suites/nonproduct.craftax)"
  echo "stamp:     $STAMP"
  echo "containers:$CONTAINERS"
  echo "run dir:   $RUN_DIR"
  echo "log:       $LOG"
  if [[ "$PAID" -eq 1 ]]; then
    echo "target:    craftax_react"
    echo "world:     GoldCraftaxWorld  SYNTH_CRAFTAX_URL=$SYNTH_CRAFTAX_URL"
    echo "policy:    OpenRouterReAct gpt-5.6-luna effort=medium (key from env, not logged)"
  else
    echo "target:    craftax_engine"
    echo "world:     in-process CraftaxWorld fixture"
    echo "policy:    ScriptedReAct  — not a Luna eval, not A1 paid"
  fi
  echo "contract:  POST /rollouts/prepare"
  echo "           GET  declared transports.poll.url until stream.subscribed ready=true"
  echo "           POST /rollouts  slot=stream  telemetry.transport=sse"
  echo "           live poll + SSE during start (partial traces, not a terminal dump)"
  echo "           POST /reward    mode=terminal  (missing stays null)"
  echo "max_steps: $SYNTH_CRAFTAX_MAX_STEPS"
  echo "seeds:     0..$((SEEDS - 1)) sequential"
  echo "============================================================"
} | tee "$LOG"

cd "$CONTAINERS"
ARGS=(--run-dir "$RUN_DIR" --seeds "$SEEDS")
if [[ "$PAID" -eq 1 ]]; then
  ARGS+=(--paid)
else
  ARGS+=(--scripted)
fi
exec > >(tee -a "$LOG") 2>&1
exec uv run python "$ROOT/run_platform.py" "${ARGS[@]}"
