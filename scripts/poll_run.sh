#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${AGENT_ID:-ijustworkhere}"
RUN_PREFIX="${1:-results/${AGENT_ID}/vllm-baseline-run1}"

if ! hf buckets --help >/dev/null 2>&1; then
  echo "hf buckets is required. Upgrade with: pip install -U huggingface_hub" >&2
  exit 1
fi

BASE="hf://buckets/gemma-challenge/gemma-${AGENT_ID}/${RUN_PREFIX}"
echo "job_status.json"
hf buckets cp "$BASE/job_status.json" -
echo
echo "summary.json"
hf buckets cp "$BASE/summary.json" - || true
echo
echo "job_logs.txt"
hf buckets cp "$BASE/job_logs.txt" - || true
