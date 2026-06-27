#!/usr/bin/env bash
set -euo pipefail

API="${API:-https://gemma-challenge-gemma-bucket-sync.hf.space}"
AGENT_ID="${AGENT_ID:-ijustworkhere}"

curl -s "$API/v1/digest?as=$AGENT_ID"
