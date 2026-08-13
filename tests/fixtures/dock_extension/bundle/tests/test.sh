#!/bin/sh
set -eu
test -f /workspace/gamebench/tasks/example/task.txt
test -f /workspace/answer.txt
mkdir -p /logs/verifier
printf '1.0\n' > /logs/verifier/reward.txt
