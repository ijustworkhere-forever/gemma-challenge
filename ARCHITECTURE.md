# Gemma Challenge Architecture

This repository is a local-first submission factory for the official Hugging
Face Gemma Challenge.

The system is intentionally small until the baseline is measured.

```text
Submission Variant
  -> HF bucket upload
  -> HF Jobs official runner on a10g-small
  -> summary.json
  -> local SQLite experiment ledger
  -> next variant decision
```

## Primary Components

## 1. Submission Directories

Each submission lives under `submissions/`.

Current baseline:

```text
submissions/vllm_baseline/
  manifest.json
  serve.py
```

Copy a submission directory before tuning a new variable. Do not mutate a
measured baseline in place.

## 2. Hugging Face Helper Scripts

Scripts under `scripts/` handle local preflight checks, upload, job launch, and
job inspection.

Required environment:

- `HF_AGENT_ID`
- authenticated Hugging Face CLI session

Optional environment:

- `HF_TOKEN`
- `HF_REPO_PREFIX`
- `HF_RUN_PREFIX`

## 3. Local SQLite Ledger

`experiment_log.py` writes to `data/experiments.sqlite3`.

Tracked fields:

- submission path
- runtime target
- HF run URI
- HF job id
- TPS
- PPL
- latency
- status
- notes
- raw `summary.json`

SQLite is the default because this is single-user, portable, and easy to back
up. Postgres can be revisited only after there is real multi-user or hosted
leaderboard pressure.

## 4. Runtime Targets

| Target | Role | Performance Meaning |
|---|---|---|
| `local_cpu` | Packaging and ledger tests | None |
| `local_3050ti` | Tiny local CUDA smoke tests | Exploratory only |
| `zerogpu` | Free HF Spaces smoke tests via Gradio | Exploratory only |
| `hf_a10g` | Official contest benchmark | Contest-comparable |

ZeroGPU can be useful for free smoke testing if available, but official
optimization decisions should be based on `hf_a10g` runs.

For ZeroGPU, create a Gradio Space and use `spaces/zerogpu_smoke/`. Docker and
Static Spaces are not the intended path for ZeroGPU smoke testing.

## Deferred Components

The previous Redis/Postgres/swarm files remain in the repo as prototypes.
They are deferred because they introduce operational complexity before the
official contest loop is working.

Deferred until needed:

- Redis queue workers
- hosted Postgres leaderboard
- multi-provider API swarm
- autonomous long-running optimizer

## Guiding Rule

If a change does not help produce, run, compare, or explain an official
Gemma Challenge submission, it is secondary for now.
