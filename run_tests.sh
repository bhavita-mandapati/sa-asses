#!/usr/bin/env bash
# Runs every task's suite in its own process (each task is a self-contained directory).
set -u
cd "$(dirname "$0")"
status=0
for d in task1_mcp_server task2_mcp_gateway task3_stream_guardrail task4_router; do
  printf '%-24s ' "$d"
  (cd "$d" && python3 -W error::SyntaxWarning -m pytest -q 2>&1 | tail -1) || status=1
done
exit $status
