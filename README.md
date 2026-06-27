# gemma-challenge

Local-first workspace for the official Hugging Face Gemma Challenge:

https://huggingface.co/gemma-challenge

The current project goal is to build, track, and iterate on contest submissions
for `google/gemma-4-E4B-it`.

## Current Direction

This repo is now organized around the official contest execution path:

1. Develop submission variants locally.
2. Upload a submission directory to the Hugging Face challenge bucket.
3. Run the official harness on HF Jobs `a10g-small`.
4. Save `summary.json` and log TPS/PPL locally in SQLite.
5. Promote only measured, reproducible improvements.

Local hardware and free tiers are for smoke tests only. The contest-comparable
number is the official A10G run.

## Runtime Targets

| Target | Purpose | Notes |
|---|---|---|
| `local_cpu` | Packaging, scripts, SQLite, mock flow | No performance claims |
| `local_3050ti` | Tiny CUDA smoke tests | 4 GB VRAM is not contest-equivalent |
| `zerogpu` | Free exploratory HF Spaces smoke tests | Use Gradio SDK; not equivalent to A10G Jobs |
| `hf_a10g` | Official benchmark target | Required for real TPS/PPL comparisons |

## Layout

```text
submissions/
  vllm_baseline/
    manifest.json
    serve.py

scripts/
  check_hf_env.py
  sync_submission.sh
  run_hf_job.sh
  poll_run.sh

experiment_log.py
data/experiments.sqlite3  # generated locally
```

Older swarm, Redis, and provider API prototypes are still present, but they are
not the primary contest path right now.

## ZeroGPU Smoke Space

If creating a Hugging Face Space for free smoke testing, choose **Gradio** as
the Space SDK. The scaffold lives at:

```text
spaces/zerogpu_smoke/
```

This Space only checks whether a GPU-decorated function can run. It is not an
official benchmark and should not drive optimization decisions.

## Local Setup

```bash
python scripts/check_hf_env.py
python experiment_log.py list
```

Set these before using the HF helper scripts:

```bash
export AGENT_ID=ijustworkhere
export HF_TOKEN=your-token
export API=https://gemma-challenge-gemma-bucket-sync.hf.space
```

Do not commit a real token. Use [.env.example](/gemma-challenge/.env.example)
as the template for local configuration, or run `huggingface-cli login` and
paste the token there.

The challenge requires `hf buckets`, which is available in newer
`huggingface_hub` releases:

```bash
pip install -U huggingface_hub
hf auth login
```

## Submission Flow

```bash
export AGENT_ID=ijustworkhere
scripts/setup_challenge_identity.sh
scripts/register_agent.sh
scripts/post_message.sh
scripts/sync_submission.sh submissions/vllm_baseline vllm-baseline
scripts/run_hf_job.sh vllm-baseline
scripts/poll_run.sh results/ijustworkhere/vllm-baseline-run1
```

See [docs/CHALLENGE_WORKFLOW.md](/gemma-challenge/docs/CHALLENGE_WORKFLOW.md)
for the complete official workflow and [docs/PROJECT_STATE.md](/gemma-challenge/docs/PROJECT_STATE.md)
for the current public leaderboard read.

After a run produces `summary.json`:

```bash
python experiment_log.py add \
  --name vllm-baseline-001 \
  --target hf_a10g \
  --submission submissions/vllm_baseline \
  --summary path/to/summary.json \
  --status measured
```
