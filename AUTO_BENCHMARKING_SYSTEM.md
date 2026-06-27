# AUTO_BENCHMARKING_SYSTEM.md

> Automated benchmarking, regression detection, and performance validation system.
>
> Purpose:
> Ensure every optimization is real, reproducible, and not noise.

---

# 1. Core Philosophy

Performance work fails when:

- results are noisy
- benchmarks are inconsistent
- changes are not isolated
- improvements are not reproducible

This system enforces:

> "No merge without proof."

---

# 2. System Overview

Every commit triggers:

```
Commit
  ↓
Build environment
  ↓
Run baseline benchmark
  ↓
Run optimized benchmark
  ↓
Compare results
  ↓
Generate report
  ↓
Pass / Fail decision
```

---

# 3. Required Outputs (Every Run)

Each benchmark must generate:

```
/auto_bench/
  run_metadata.json
  baseline.json
  candidate.json
  comparison_report.md
  profiler_trace.qdrep
  torch_profile.json
  charts/
```

---

# 4. Benchmark Pipeline

## Step 1: Environment Lock

Lock all variables:

- CUDA version
- Driver version
- PyTorch version
- Model version
- Engine version
- Batch size
- Context length
- Precision mode

---

## Step 2: Warmup Phase

Before measurement:

- Run 5–10 warmup iterations
- Stabilize GPU clocks
- Initialize KV cache
- Prevent cold-start bias

---

## Step 3: Baseline Run

Execute:

- 3–5 runs minimum
- Record all metrics
- Compute mean + variance

---

## Step 4: Candidate Run

Run identical benchmark:

- same inputs
- same configuration
- same hardware state

---

## Step 5: Comparison Engine

Compare:

| Metric | Baseline | Candidate | Delta |
|------|----------|----------|------|
| TPS | | | |
| TTFT | | | |
| Latency | | | |
| GPU Utilization | | | |
| PPL | | | |

---

# 5. Acceptance Rules

## A change is VALID only if:

### Condition 1
TPS improves by > threshold (default: +2%)

AND

### Condition 2
PPL remains within acceptable drift (configurable)

AND

### Condition 3
Variance is stable (<5%)

---

# 6. Regression Detection

If ANY of the following occur:

- TPS drops > 1%
- PPL increases beyond threshold
- GPU utilization drops
- latency variance increases

→ Mark as REGRESSION ❌

---

# 7. Noise Filtering System

To avoid false conclusions:

## Rule 1: Multi-run averaging

- Minimum 3 runs
- Prefer 5 runs for final decision

---

## Rule 2: Outlier removal

Remove:

- cold-start runs
- first-run bias
- anomalous spikes

---

## Rule 3: Stability threshold

Only accept results if:

```
standard deviation < 5%
```

---

# 8. Metrics Collected

## Core Performance

- Tokens per second (TPS)
- Time to First Token (TTFT)
- End-to-end latency
- P95 / P99 latency

---

## GPU Metrics

- SM occupancy
- Tensor core utilization
- DRAM bandwidth
- L2 cache hit rate
- Warp efficiency

---

## System Metrics

- CPU utilization
- memory usage
- thread contention
- sync overhead

---

# 9. Benchmark Modes

## Mode A: Throughput Test

- Long generation runs
- Maximize TPS
- Ignore TTFT

---

## Mode B: Latency Test

- Short prompts
- Measure responsiveness
- Focus TTFT

---

## Mode C: Balanced Mode

- Realistic workloads
- Weighted scoring

---

# 10. Scoring Function (Optional)

You can define a local leaderboard score:

```
score =
  (TPS * 1.0)
  - (TTFT * 0.2)
  - (PPL_penalty * 5)
  - (variance_penalty * 2)
```

---

# 11. CI Integration (Recommended)

Every pull request should:

1. Run benchmark suite
2. Compare against main branch
3. Generate diff report
4. Block merge if regression detected

---

# 12. GitHub Actions Flow

```
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: gpu-enabled-runner

    steps:
      - checkout
      - setup environment
      - run baseline benchmark
      - run candidate benchmark
      - compare results
      - upload artifacts
```

---

# 13. Output Report Format

## comparison_report.md

```
# Benchmark Comparison

## Summary

TPS: +3.2% ✅
PPL: stable ✅
Latency: -1.1% ✅

---

## Decision

✔ ACCEPTED

---

## Notes

Improvement primarily due to KV cache layout change.
```

---

# 14. Failure Cases

## Case 1: Fake Improvement

TPS improves but:

- PPL worsens
- variance increases

→ REJECT

---

## Case 2: Noise Gain

TPS improves in one run only

→ REJECT

---

## Case 3: Hidden Regression

GPU utilization drops

→ INVESTIGATE

---

# 15. Integration Points

Connect with:

- `EXPERIMENTS.md` → logs each run
- `ROADMAP.md` → determines stage
- `PROFILING_PLAYBOOK.md` → validates bottlenecks
- `DECISIONS.md` → records final outcome

---

# 16. Key Principle

> If it is not reproducible, it is not real.

---

# 17. Final Rule

Never accept a performance claim without:

- baseline comparison
- multi-run validation
- profiler confirmation

---

# 18. Next Step

👉 Implement benchmark runner  
👉 Wire Nsight capture into CI  
👉 Define initial baseline run  
👉 Lock environment configuration  

---
