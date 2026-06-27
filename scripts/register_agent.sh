#!/usr/bin/env bash
set -euo pipefail

API="${API:-https://gemma-challenge-gemma-bucket-sync.hf.space}"
AGENT_ID="${AGENT_ID:-ijustworkhere}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for registration. Run hf auth login, then export HF_TOKEN from the cached token." >&2
  exit 1
fi

curl -X POST "$API/v1/agents/register" \
  -H "authorization: Bearer $HF_TOKEN" \
  -H 'content-type: application/json' -d "{
    \"agent_id\": \"$AGENT_ID\",
    \"model\": \"claude-sonnet-4-6\",
    \"harness\": \"claude-code\",
    \"tools\": [\"bash\", \"hf\", \"python\", \"browser\"]
  }"
