#!/usr/bin/env bash
# Full Rhodes lifecycle for Craftax over the public API, one command:
#   per seed: deploy (server-provisioned) -> observed readiness -> capacity lease -> rollout
#   -> committed events -> results/usage -> lease release -> undeploy -> absence readback;
#   per lane: one pool (the observed-capacity profile holds a whole pool per lease).
#
#   SYNTH_API_KEY=... ./scripts/stream-rhodes.sh --backend-url http://127.0.0.1:8000 --project-id <uuid>
#   ./scripts/stream-rhodes.sh --backend-url URL --project-id UUID --seeds 0-9 --parallel 4
#   ./scripts/stream-rhodes.sh ... --image sha256:<digest> --max-steps 10 --harness scripted_react
#   Flags: --seeds N|N-M|a,b  --parallel 1-32  --image  --image-kind  --port  --policy-config
#          --ready-seconds  --rollout-seconds  --lease-ttl-seconds  --progress-seconds  --out DIR
#
# Output: <out>/<stamp>-<id>/{stream.log,events.jsonl,result-seed*.json,report.json,summary.txt}
# Exit: 0 every check passed; 1 a check failed; 2 refused before running; 130 interrupted.
# Ctrl-C: cancels only this run's rollouts, then still releases leases, undeploys and deletes pools.
# Default policy is the scripted harness: no model or paid provider calls. Secrets are never printed.
set -euo pipefail
A="$(cd "$(dirname "$0")/.." && pwd -P)"
PY="${RHODES_STREAM_PYTHON:-$A/.venv/bin/python}"
for a in "$@"; do [[ "$a" == -h || "$a" == --help ]] && { sed -n 2,16p "$0"; exit 0; }; done
[[ -x "$PY" ]] || { echo "python not found: $PY (set RHODES_STREAM_PYTHON)" >&2; exit 2; }
[[ -n "${SYNTH_API_KEY:-}" ]] || { echo "SYNTH_API_KEY must be set (org-scoped key)" >&2; exit 2; }
# Idle/system sleep would freeze the Docker VM mid-run.
command -v caffeinate >/dev/null && { caffeinate -ims -w $$ >/dev/null 2>&1 & }
exec "$PY" -u -m synth_containers.rhodes_lifecycle --out "${RHODES_STREAM_OUT:-$A/.local/rhodes-runs}" "$@"
