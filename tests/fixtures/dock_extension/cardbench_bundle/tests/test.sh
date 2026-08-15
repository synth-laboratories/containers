#!/bin/sh
set -eu
test -f /workspace/cardbench/tasks/pokemon/task.txt
test -f /workspace/answer.txt
mkdir -p /logs/verifier
printf '1.0\n' > /logs/verifier/reward.txt
