# ROADMAP.md

> Living execution plan for the Gemma Challenge.
>
> This is not theory — this is the actual order of operations.
>
> Update continuously as results come in.

---

# North Star Goal

Maximize:

- Tokens Per Second (TPS)

while maintaining:

- Valid Perplexity (PPL)
- Stable outputs
- Reproducibility

---

# Status Overview

| Phase | Status | Notes |
|------|--------|------|
| 0. Setup | ⏳ | Repo initialized |
| 1. Baseline | ⏳ | Not yet measured |
| 2. Profiling | ⏳ | Not started |
| 3. Bottleneck Analysis | ⏳ | Pending profiling |
| 4. Core Optimizations | ⏳ | Pending analysis |
| 5. Kernel Work | ⏳ | Advanced stage |
| 6. Submission Tuning | ⏳ | Final stage |

---

# PHASE 0 — Setup

## Objectives

- [x] Create repo structure
- [x] Add documentation system
- [ ] Clone baseline inference stack
- [ ] Confirm GPU environment (A10G)
- [ ] Validate Gemma model runs

---

# PHASE 1 — Baseline Establishment

## Objectives

Establish ground truth metrics.

### Tasks

- [ ] Run official benchmark (no modifications)
- [ ] Record TPS / TTFT / PPL
- [ ] Capture VRAM usage
- [ ] Capture GPU utilization
- [ ] Save Nsight Systems trace
- [ ] Save Torch profiler trace
- [ ] Repeat run (3x average)

### Outputs

- baseline.json
- baseline_trace.qdrep
- baseline_report.md

### Exit Criteria

- Stable reproducible benchmark exists
- Variance < 5%

---

# PHASE 2 — Profiling

## Objectives

Understand where time is actually spent.

### Tasks

- [ ] Break down pipeline:
  - tokenizer
  - prefill
  - decode loop
  - attention
  - KV cache
  - sampling
- [ ] Run Nsight Systems
- [ ] Run Nsight Compute
- [ ] Generate flame graphs
- [ ] Identify top 3 bottlenecks

### Exit Criteria

- At least 80% of runtime is attributed
- Bottleneck ranking established

---

# PHASE 3 — Bottleneck Validation

## Objectives

Confirm root causes before optimizing.

### Tasks

- [ ] Memory-bound vs compute-bound analysis
- [ ] Kernel launch overhead measurement
- [ ] CPU vs GPU split
- [ ] KV cache bandwidth test
- [ ] Scheduler idle time measurement

### Exit Criteria

- One primary bottleneck identified
- Secondary bottlenecks ranked

---

# PHASE 4 — High Impact Optimizations

## Objectives

Fix biggest inefficiencies first.

---

## Scheduler Optimization

- [ ] Continuous batching
- [ ] Request grouping
- [ ] Decode prioritization tuning
- [ ] Queue efficiency measurement

Expected gain: 5–15%

---

## KV Cache Optimization

- [ ] Layout analysis
- [ ] Paging vs contiguous comparison
- [ ] L2 cache hit profiling
- [ ] Memory bandwidth optimization

Expected gain: 5–20%

---

## CUDA Graphs

- [ ] Capture decode graph
- [ ] Replay execution
- [ ] Measure launch overhead reduction

Expected gain: 5–10%

---

# PHASE 5 — Kernel Optimization

## Objectives

Push GPU utilization to the limit.

### Tasks

- [ ] FlashAttention tuning
- [ ] Triton kernel experiments
- [ ] CUTLASS GEMM review
- [ ] RMSNorm fusion
- [ ] Rotary embedding optimization
- [ ] Sampling kernel optimization

### Exit Criteria

- SM occupancy > 90%
- Tensor core utilization maximized

---

# PHASE 6 — Memory & System Optimization

## Objectives

Remove non-kernel overhead.

### Tasks

- [ ] CPU profiling (tokenizer, runtime)
- [ ] Async pipeline improvements
- [ ] Memory allocator tuning
- [ ] Reduce Python overhead
- [ ] Eliminate sync points

---

# PHASE 7 — Quantization (Optional / Risk-Based)

## Objectives

Trade precision carefully for speed.

### Tasks

- [ ] FP16 baseline confirmation
- [ ] FP8 testing
- [ ] INT8 evaluation
- [ ] Weight-only quantization experiments

### Exit Criteria

- PPL remains within acceptable threshold
- TPS improves measurably

---

# PHASE 8 — Engine Evaluation

## Objectives

Compare full inference stacks.

### Tasks

- [ ] vLLM baseline tuning
- [ ] TensorRT-LLM benchmark
- [ ] SGLang comparison
- [ ] llama.cpp CUDA test (optional)

---

# PHASE 9 — Submission Optimization

## Objectives

Polish for leaderboard performance.

### Tasks

- [ ] Final scheduler tuning
- [ ] Final batching configuration
- [ ] Warmup optimization
- [ ] Remove debug overhead
- [ ] Stabilize variance
- [ ] Final 3-run averaging

---

# Performance Targets

| Metric | Target |
|------|--------|
| TPS | Maximize |
| TTFT | Minimize |
| GPU Utilization | >95% |
| Variance | <3–5% |
| PPL | Within threshold |

---

# Optimization Philosophy

1. Measure before optimizing
2. Fix largest bottleneck first
3. Avoid premature kernel work
4. Keep experiments isolated
5. Always preserve correctness
6. Never trust a single benchmark

---

# Current Risks

- Optimizing wrong layer
- Noise in benchmark results
- Hidden CPU bottlenecks
- KV cache misdiagnosis
- Overfitting to benchmark dataset

---

# Open Questions

- What is the true bottleneck at scale?
- Does decode saturate memory bandwidth or compute?
- How much is scheduler overhead costing?
- Can CUDA Graphs be applied safely at scale?
- Is batching currently optimal?

---

# Next Action

👉 Run baseline benchmark  
👉 Capture full profiler trace  
👉 Identify first bottleneck  

---
