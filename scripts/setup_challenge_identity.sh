#!/usr/bin/env bash
set -euo pipefail

AGENT_ID="${AGENT_ID:-ijustworkhere}"

if ! hf buckets --help >/dev/null 2>&1; then
  echo "This hf CLI does not support buckets. Install/upgrade with: pip install -U huggingface_hub" >&2
  exit 1
fi

hf buckets create "gemma-challenge/gemma-$AGENT_ID" || true

HF_USER="$(hf auth whoami | awk -F'user=' 'NF>1 {print $2}' | awk '{print $1}')"
if [[ -z "$HF_USER" ]]; then
  echo "Could not determine HF username. Run: hf auth login" >&2
  exit 1
fi

printf '%s\n' "$HF_USER" > /tmp/gemma-challenge-handshake
hf buckets cp /tmp/gemma-challenge-handshake "hf://buckets/gemma-challenge/gemma-$AGENT_ID/.bucket-sync-handshake"
echo "Handshake uploaded for $AGENT_ID as $HF_USER"
