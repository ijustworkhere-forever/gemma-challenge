#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_NAME="${1:-vllm-baseline}"
RUN_NAME="${2:-${SUBMISSION_NAME}-$(date -u +%Y%m%dT%H%M%SZ)}"
API="${API:-https://gemma-challenge-gemma-bucket-sync.hf.space}"
AGENT_ID="${AGENT_ID:-ijustworkhere}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for /v1/jobs:run." >&2
  exit 1
fi

curl -X POST "$API/v1/jobs:run" \
  -H "authorization: Bearer $HF_TOKEN" \
  -H 'content-type: application/json' -d "{
    \"agent_id\": \"$AGENT_ID\",
    \"submission_prefix\": \"submissions/$AGENT_ID/$SUBMISSION_NAME\",
    \"run_prefix\": \"results/$AGENT_ID/$RUN_NAME\"
  }"
