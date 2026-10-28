#!/bin/sh
set -u
n=0
while :; do
  load=$(sysctl -n vm.loadavg | awk '{print $2}')
  printf '%s load1=%s\n' "$(date -u +%FT%TZ)" "$load"
  awk -v l="$load" 'BEGIN {exit !(l<20)}' && break
  n=$((n+1)); [ "$n" -lt 60 ] || { echo 'calm-host gate expired without running' >&2; exit 1; }
  sleep 15
done
TMPDIR=/Users/joshuapurtell/GitHub/.tmp .venv/bin/python -m pytest -p no:cacheprovider -q tests/test_engine_guard_progress.py tests/test_eval_engine_override.py > artifacts/syn5206-delayed-create-20261028/tests-qualified.log 2>&1
status=$?
cat artifacts/syn5206-delayed-create-20261028/tests-qualified.log
exit "$status"
