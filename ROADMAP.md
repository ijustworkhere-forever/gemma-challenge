# ROADMAP.md

> Living execution plan for the Gemma Challenge.
>
> Current strategy: local-first development, official A10G measurement on
> Hugging Face Jobs, SQLite experiment tracking.

---

# North Star Goal

Maximize:

- Tokens Per Second (TPS)

while maintaining:

- Valid Perplexity (PPL)
- Stable outputs
- Reproducibility
- Official-run comparability

---

# Runtime Reality

| Environment | Available | Use |
|---|---:|---|
| Windows 11 laptop, i9, 64 GB RAM | yes | development, scripts, SQLite |
| RTX 3050 Ti, 4 GB VRAM | yes | tiny CUDA smoke tests only |
| Hugging Face ZeroGPU | possible/free | Gradio Space smoke tests |
| HF Jobs `a10g-small` | paid or org-credit | official benchmark target |
| AWS Ubuntu host | available | deferred; avoid securing DB/Redis for now |

Only `hf_a10g` results should drive contest optimization decisions.

---

# Status Overview

| Phase | Status | Notes |
|---|---|---|
| 0. Contest alignment | in progress | official HF flow now drives repo |
| 1. Submission scaffold | in progress | baseline directory added and aligned to official vLLM example |
| 2. Local experiment ledger | in progress | SQLite utility added |
| 3. Official baseline | pending | needs HF registration/run command |
| 4. First optimization pass | pending | wait for measured baseline |
| 5. Profiling | pending | only after baseline and runner are stable |
| 6. Final tuning | pending | after repeated official runs |

---

# PHASE 0 - Contest Alignment

## Objectives

- [x] Identify official target hardware: HF Jobs `a10g-small`
- [x] Decide local-first architecture
- [x] Defer Redis/Postgres/swarm infrastructure
- [x] Treat ZeroGPU as exploratory, not contest-equivalent
- [x] Add Gradio ZeroGPU smoke-test scaffold
- [x] Confirm official org-credit job API
- [x] Confirm agent id: `ijustworkhere`
- [ ] Register agent and post intro message
- [ ] Create scratch bucket and handshake

## Exit Criteria

- Repo docs describe one primary execution path
- No required hosted services for local iteration

---

# PHASE 1 - Baseline Submission

## Objectives

Create the simplest valid submission and measure it officially.

## Tasks

- [x] Add `submissions/vllm_baseline/manifest.json`
- [x] Add `submissions/vllm_baseline/serve.py`
- [x] Add HF helper scripts
- [x] Replace placeholder HF job command with official `/v1/jobs:run` call
- [ ] Upload baseline submission to challenge bucket
- [ ] Run official A10G benchmark
- [ ] Save `summary.json`
- [ ] Log run with `experiment_log.py`

## Outputs

- `summary.json`
- SQLite experiment row
- baseline notes in `EXPERIMENTS.md`

## Exit Criteria

- One measured `hf_a10g` baseline exists
- TPS and PPL are recorded
- Run can be repeated from documented commands

---

# PHASE 2 - Local Tooling

## Objectives

Make experimentation cheap and organized before spending GPU credits.

## Tasks

- [x] Add SQLite experiment ledger
- [x] Add `.gitignore` entries for generated data/logs
- [ ] Add dependency manifest
- [ ] Add validation command for submission directories
- [ ] Add summary import command for official results
- [ ] Add README examples for Windows/Git Bash/WSL if needed

## Exit Criteria

- Local commands work without GPU
- Generated files do not pollute git status

---

# PHASE 3 - Baseline Analysis

## Objectives

Understand the initial official result before tuning.

## Tasks

- [ ] Compare TPS against reference/public examples
- [ ] Confirm PPL is below allowed threshold
- [ ] Review server startup and steady-state behavior
- [ ] Identify whether bottleneck is load time, prefill, decode, or overhead
- [ ] Decide first single-variable experiment

## Exit Criteria

- One bottleneck hypothesis is chosen
- One controlled experiment is queued

---

# PHASE 4 - Controlled Optimization

## Objectives

Improve one variable at a time.

Candidate areas:

- vLLM engine config
- dtype / quantization options
- max model length and cache settings
- CUDA graph compatibility
- sampling overhead
- startup/warmup behavior
- memory utilization

Avoid early focus on:

- distributed serving
- continuous batching
- hosted leaderboards
- multi-provider API research

The official benchmark is single-runner and contest-scored, so batching and
swarm infrastructure are lower priority until proven relevant.

## Exit Criteria

- Each accepted change has official TPS/PPL evidence
- Regressions are reverted or archived

---

# PHASE 5 - Profiling

## Objectives

Profile only after the official baseline is reproducible.

## Tasks

- [ ] Run Nsight Systems if available in the HF job path
- [ ] Capture Python/runtime timing if Nsight is unavailable
- [ ] Separate prefill and decode timing
- [ ] Measure memory pressure and OOM margin
- [ ] Record profiling evidence in `EXPERIMENTS.md`

---

# PHASE 6 - Final Submission Tuning

## Objectives

Stabilize the best measured submission.

## Tasks

- [ ] Repeat best run 3x
- [ ] Confirm PPL margin
- [ ] Remove debug overhead
- [ ] Freeze submission directory
- [ ] Record final config and result provenance

---

# Current Risks

- HF auth/token is not configured in this shell yet
- Installed `hf` CLI is too old for `hf buckets`; upgrade `huggingface_hub`
- Spending credits before the runner command is correct
- Drawing conclusions from non-A10G environments
- Optimizing before baseline evidence exists
- Breaking PPL while chasing TPS
- Letting infrastructure work distract from submission performance

---

# Next Action

1. Upgrade Hugging Face CLI: `pip install -U huggingface_hub`.
2. Run `hf auth login` with a token that has `gemma-challenge` write access.
3. Export `AGENT_ID=ijustworkhere` and `HF_TOKEN`.
4. Run `scripts/setup_challenge_identity.sh`.
5. Run `scripts/register_agent.sh`.
6. Run `scripts/post_message.sh`.
7. Upload and run `submissions/vllm_baseline`.
