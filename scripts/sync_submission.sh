#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR="${1:-submissions/vllm_baseline}"
SUBMISSION_NAME="${2:-vllm-baseline}"
AGENT_ID="${AGENT_ID:-ijustworkhere}"

if ! hf buckets --help >/dev/null 2>&1; then
  echo "hf buckets is required. Upgrade with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ ! -d "$SUBMISSION_DIR" ]]; then
  echo "Submission directory not found: $SUBMISSION_DIR" >&2
  exit 1
fi

REMOTE_PATH="hf://buckets/gemma-challenge/gemma-${AGENT_ID}/submissions/${AGENT_ID}/${SUBMISSION_NAME}"

echo "Uploading $SUBMISSION_DIR -> $REMOTE_PATH"
hf buckets sync "$SUBMISSION_DIR" "$REMOTE_PATH"
echo "Uploaded submission to $REMOTE_PATH"
