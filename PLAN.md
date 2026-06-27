# Gemma Challenge Performance Optimization Roadmap

> Goal:
>
> Maximize Tokens Per Second (TPS) while maintaining acceptable Perplexity (PPL) on the official benchmark using the provided hardware.

---

# Overall Strategy

This project will follow an engineering-driven optimization process rather than random experimentation.

Every optimization must be:

- Measured
- Repeatable
- Versioned
- Reversible

Performance work without metrics is guessing.

---

# Repository Structure

```
gemma-optimizer/

│
├── benchmark/
│   ├── benchmark.py
│   ├── datasets/
│   ├── configs/
│   └── results/
│
├── profiler/
│   ├── nsight/
│   ├── torch/
│   └── scripts/
│
├── experiments/
│   ├── exp001_baseline.md
│   ├── exp002_cuda_graphs.md
│   ├── exp003_scheduler.md
│   └── ...
│
├── kernels/
│
├── configs/
│
├── docs/
│
└── PLAN.md
```

---

# Phase 1 — Establish Baseline

## Objective

Never optimize until the baseline is known.

Collect:

- TPS
- Time To First Token
- Decode latency
- Prefill latency
- GPU utilization
- VRAM usage
- SM occupancy
- Memory bandwidth
- CPU utilization
- Power draw

---

## Benchmark Harness

Create a script that automatically records:

```
Git Commit

↓

Run Benchmark

↓

Collect Metrics

↓

Save JSON

↓

Generate Charts
```

Output example:

```json
{
  "commit": "1a3ef2",
  "engine": "vLLM",
  "TPS": 132.4,
  "TTFT": 224,
  "PPL": 8.13,
  "GPU": 96.2,
  "VRAM": 18.4
}
```

---

# Phase 2 — Build Profiling Pipeline

## Tools

### Nsight Systems

Determine:

- kernel launch overhead
- synchronization
- CPU idle time
- CUDA streams

---

### Nsight Compute

Determine:

- occupancy
- register pressure
- shared memory
- memory throughput
- tensor core utilization

---

### Torch Profiler

Measure:

- Python overhead
- operator timing
- graph execution

---

# Phase 3 — Identify Bottlenecks

Create flame graphs.

Break inference into:

```
Tokenizer

↓

Prefill

↓

Attention

↓

KV Cache

↓

Decode

↓

Sampling

↓

Output
```

Measure each independently.

---

# Phase 4 — Optimization Backlog

---

## Stage A

### Scheduler

Experiment with

- Continuous batching
- Dynamic batching
- Request grouping
- Sequence ordering

Record TPS differences.

---

## Stage B

### CUDA Graphs

Investigate:

- static decode graph
- graph replay
- launch overhead

Expected gain:

5–15%

---

## Stage C

### KV Cache

Investigate:

- layout
- paging
- fragmentation
- locality
- FP8 cache
- cache reuse

Expected gain:

5–20%

---

## Stage D

### Flash Attention

Test:

- FlashAttention-2
- FlashAttention-3 (if supported)

Tune

- tile size
- block size
- occupancy

---

## Stage E

### Kernel Fusion

Locate

```
Load

↓

Matmul

↓

Activation

↓

Normalization

↓

Store
```

Fuse where possible.

---

## Stage F

### Quantization

Evaluate

- FP16
- BF16
- FP8
- INT8
- Weight-only

Measure

TPS

vs

PPL

---

## Stage G

### Memory

Investigate

- pinned memory
- async copies
- memory fragmentation
- allocator behavior

---

## Stage H

### CPU

Profile

- tokenizer
- networking
- serialization
- Python runtime

CPU bottlenecks often cost several TPS.

---

# Phase 5 — Engine Comparison

Benchmark

- vLLM
- TensorRT-LLM
- SGLang
- llama.cpp CUDA

Record

TPS

Latency

Memory

PPL

for every engine.

---

# Phase 6 — Custom Kernel Work

Only after profiling.

Potential targets:

- Attention
- RMSNorm
- Rotary embeddings
- Sampling
- KV cache writes

Possible technologies:

- Triton
- CUDA
- CUTLASS

---

# Phase 7 — Automated Experiment Tracking

Every run should generate

```
results/

baseline.json

exp001.json

exp002.json

exp003.json
```

Never overwrite.

Always compare.

---

# Phase 8 — Continuous Benchmarking

Use GitHub Actions or local automation.

Every commit should

Run benchmark

↓

Compare against previous

↓

Generate report

↓

Flag regressions

---

# Metrics Dashboard

Track

| Metric | Goal |
|----------|------|
| TPS | Maximize |
| TTFT | Minimize |
| Decode Latency | Minimize |
| PPL | Maintain |
| GPU Utilization | >95% |
| Tensor Core Usage | Maximize |
| VRAM | Stable |
| CPU | Low |

---

# Experiment Log Template

## Experiment #

Date:

Commit:

Description:

Hypothesis:

Changes Made:

Results:

TPS:

TTFT:

PPL:

GPU Utilization:

Observations:

Decision:

Keep

or

Revert

---

# Optimization Priority

1. Measurement
2. Profiling
3. Scheduler
4. CUDA Graphs
5. KV Cache
6. Flash Attention
7. Quantization
8. Kernel Fusion
9. Memory Optimization
10. CPU Optimization
11. Custom CUDA

---

# Rules

Never optimize blind.

Never trust one benchmark.

Run every benchmark at least three times.

Average all measurements.

Never sacrifice correctness for speed.

Document every experiment.

Commit every meaningful change.

Keep optimization incremental.

---

# Final Goal

Achieve the highest sustainable TPS while preserving output quality and reproducibility.

Success is measured by:

- Higher throughput
- Stable perplexity
- Reproducible benchmarks
- Clean engineering practices
- Well-documented optimizations
